"""Fine-tune the multi-task AfroXLMR model.

One seed (42). Checkpoints are written every epoch and can be resumed.
Early stopping watches the mean of validation issue macro-F1, sentiment
macro-F1, and urgency QWK. Hyperparameters come from the YAML config.
``--tiny`` trains a random 4-layer encoder for a few steps and does not
download Davlan/afro-xlmr-base.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from capstone.labels import ISSUE_CLASSES, SENTIMENT_CLASSES, URGENCY_CLASSES
from capstone.models.data import (
    LABEL_LISTS,
    MultiExample,
    class_weight_vector,
    head_splits,
    merged_split,
)
from capstone.models.metrics import per_class_scores, quadratic_weighted_kappa
from capstone.models.multitask import (
    MODEL_ID,
    CharTokenizer,
    MultiTaskEncoder,
    MultiTaskPredictor,
    build_encoder,
    freeze_bottom,
    masked_loss,
    save_inference_checkpoint,
)
from capstone.pii import mask_pii

SEED = 42


class ExampleDataset(Dataset):
    def __init__(self, rows: list[MultiExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MultiExample:
        return self.rows[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the multi-task AfroXLMR model.")
    parser.add_argument("--config", type=Path, default=Path("configs/afroxlmr.yaml"))
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/afroxlmr"))
    parser.add_argument("--tiny", action="store_true", help="Random tiny encoder, no download.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--kind", choices=("harness", "data"), default="data")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.tiny:
        config["tiny"] = True
        config["max_steps"] = args.max_steps if args.max_steps is not None else 1
    elif args.max_steps is not None:
        config["max_steps"] = args.max_steps
    train(config, args.labels, args.manifests, args.out, resume=args.resume, kind=args.kind)
    return 0


def train(
    config: dict,
    label_dir: Path,
    manifest_dir: Path,
    out_dir: Path,
    *,
    resume: bool = False,
    kind: str = "data",
) -> dict:
    seed = int(config.get("seed", SEED))
    _set_seed(seed)
    tiny = bool(config.get("tiny", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(config.get("fp16", True)) and device.type == "cuda"
    splits = head_splits(label_dir, manifest_dir)
    train_rows = merged_split(splits, "train")
    val_rows = {head: splits[head]["val"] for head in ("issue", "sentiment", "urgency")}
    if not train_rows:
        raise RuntimeError("training split is empty")

    encoder, encoder_config = build_encoder(tiny, config.get("model_name", MODEL_ID))
    hidden = int(encoder.config.hidden_size)
    model = MultiTaskEncoder(encoder, hidden)
    frozen = freeze_bottom(
        model.encoder,
        bottom_layers=int(config.get("freeze_bottom_layers", 8)),
        freeze_embeddings=bool(config.get("freeze_embeddings", True)),
    )
    model.to(device)
    tokenizer = _tokenizer(config, tiny)
    weights = _weight_tensors(splits, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_fp16)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_score = float("-inf")
    best_state = None
    history: list[dict] = []
    if resume:
        loaded = _load_last(checkpoint_dir / "last.pt", model, optimizer, scaler)
        if loaded is not None:
            start_epoch = loaded["epoch"]
            global_step = loaded["step"]
            best_score = loaded["best_score"]
            best_state = loaded["best_state"]
            history = loaded["history"]

    batch_size = int(config.get("batch_size", 16))
    max_length = int(config.get("max_length", 128))
    epochs = int(config.get("epochs", 6))
    patience = int(config.get("patience", 2))
    max_steps = config.get("max_steps")
    steps_per_epoch = max(1, (len(train_rows) + batch_size - 1) // batch_size)
    total_steps = int(max_steps) if max_steps else epochs * steps_per_epoch
    _print_banner(
        kind=kind,
        tiny=tiny,
        device=device,
        fp16=use_fp16,
        model=model,
        frozen_layers=frozen,
        total_steps=total_steps,
        config=config,
    )

    stall = 0
    started = time.perf_counter()
    stop = False
    epoch_limit = epochs
    if max_steps is not None and global_step < total_steps:
        epoch_limit = max(epochs, start_epoch + 1)
    for epoch in range(start_epoch, epoch_limit):
        if global_step >= total_steps:
            break
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        loader = DataLoader(
            ExampleDataset(train_rows),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda batch: _collate(batch, tokenizer, max_length),
        )
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in loader:
            if global_step >= total_steps:
                stop = True
                break
            global_step += 1
            loss_value = _step(
                model,
                batch,
                optimizer,
                scaler,
                weights,
                device,
                use_fp16,
                config,
                global_step,
                total_steps,
            )
            epoch_loss += loss_value
            epoch_batches += 1
            _print_eta(global_step, total_steps, started)
        val_scores = _validate(model, tokenizer, val_rows, max_length, device)
        mean_score = float(
            np.mean(
                [
                    val_scores["issue_macro_f1"],
                    val_scores["sentiment_macro_f1"],
                    val_scores["urgency_qwk"],
                ]
            )
        )
        history.append(
            {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": epoch_loss / max(1, epoch_batches),
                "val": val_scores,
                "val_mean": mean_score,
            }
        )
        print(
            f"epoch={epoch + 1} step={global_step} "
            f"val_issue_f1={val_scores['issue_macro_f1']:.4f} "
            f"val_sentiment_f1={val_scores['sentiment_macro_f1']:.4f} "
            f"val_urgency_qwk={val_scores['urgency_qwk']:.4f}"
        )
        improved = mean_score > best_score + 1e-4
        if improved:
            best_score = mean_score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stall = 0
        else:
            stall += 1
        _write_checkpoint(
            checkpoint_dir / f"epoch-{epoch + 1:02d}.pt",
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch=epoch + 1,
            step=global_step,
            best_score=best_score,
            best_state=best_state,
            history=history,
        )
        if stop or (max_steps is None and stall >= patience):
            if stall >= patience and max_steps is None:
                print(f"early_stop patience={patience} best_val_mean={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    saved = _export(
        model, tokenizer, config, encoder_config, out_dir, tiny, max_length, hidden, frozen
    )
    curves = {"history": history, "best_val_mean": best_score, "kind": kind}
    (out_dir / "curves.json").write_text(
        json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"sha256={saved}")
    print(f"out={out_dir}")
    return {"sha256": saved, "best_val_mean": best_score, "steps": global_step}


def _step(
    model, batch, optimizer, scaler, weights, device, use_fp16, config, step, total_steps
) -> float:
    lr = _learning_rate(
        step,
        total_steps,
        float(config.get("warmup_ratio", 0.1)),
        float(config.get("learning_rate", 2e-5)),
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
        issue_logits, sentiment_logits, urgency_logits = model(
            batch["input_ids"], batch["attention_mask"]
        )
        loss = masked_loss(
            issue_logits,
            sentiment_logits,
            urgency_logits,
            batch["issue_y"],
            batch["sentiment_y"],
            batch["urgency_y"],
            batch["urgency_weight"],
            weights,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(config.get("grad_clip", 1.0)),
    )
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu())


def _validate(model, tokenizer, val_rows, max_length, device) -> dict:
    model.eval()
    predictor = MultiTaskPredictor(model, tokenizer, max_length=max_length, device=device)
    scores = {}
    specs = (
        ("issue", ISSUE_CLASSES, "issue_macro_f1"),
        ("sentiment", SENTIMENT_CLASSES, "sentiment_macro_f1"),
    )
    for head, classes, key in specs:
        examples = val_rows[head]
        if not examples:
            scores[key] = 0.0
            continue
        predictions = predictor.predict([row.text for row in examples])
        gold = [row.label for row in examples]
        pred = [item[head]["label"] for item in predictions]
        _per_class, macro, _matrix = per_class_scores(gold, pred, list(classes))
        scores[key] = macro
    urgency_examples = val_rows["urgency"]
    if urgency_examples:
        predictions = predictor.predict([row.text for row in urgency_examples])
        gold = [row.label for row in urgency_examples]
        pred = [item["urgency"]["label"] for item in predictions]
        scores["urgency_qwk"] = quadratic_weighted_kappa(gold, pred, list(URGENCY_CLASSES))
    else:
        scores["urgency_qwk"] = 0.0
    model.train()
    return scores


def _collate(batch: list[MultiExample], tokenizer, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [mask_pii(row.text) for row in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "issue_y": _labels([row.issue for row in batch], ISSUE_CLASSES),
        "sentiment_y": _labels([row.sentiment for row in batch], SENTIMENT_CLASSES),
        "urgency_y": _labels([row.urgency for row in batch], URGENCY_CLASSES),
        "urgency_weight": torch.tensor(
            [row.weight if row.urgency is not None else 0.0 for row in batch],
            dtype=torch.float,
        ),
    }


def _labels(values: list[str | None], classes: tuple[str, ...]) -> torch.Tensor:
    encoded = [-100 if value is None else classes.index(value) for value in values]
    return torch.tensor(encoded, dtype=torch.long)


def _weight_tensors(splits, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = {}
    for head, classes in LABEL_LISTS.items():
        labels = [row.label for row in splits[head]["train"]]
        tensors[head] = torch.tensor(
            class_weight_vector(labels, classes), dtype=torch.float, device=device
        )
    return tensors


def _tokenizer(config: dict, tiny: bool):
    if tiny:
        return CharTokenizer(max_length=int(config.get("max_length", 64)))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.get("model_name", MODEL_ID))


def _export(
    model, tokenizer, config, encoder_config, out_dir, tiny, max_length, hidden, frozen
) -> str:
    payload = {
        "model_name": MODEL_ID if not tiny else "tiny-random-xlm-roberta",
        "tiny": tiny,
        "seed": int(config.get("seed", SEED)),
        "hidden_size": hidden,
        "num_hidden_layers": int(encoder_config.get("num_hidden_layers", 0)),
        "freeze_bottom_layers": frozen,
        "freeze_embeddings": True,
        "max_length": max_length,
        "tokenizer_type": "char" if tiny else "huggingface",
        "issue_labels": list(ISSUE_CLASSES),
        "sentiment_labels": list(SENTIMENT_CLASSES),
        "urgency_labels": list(URGENCY_CLASSES),
        "encoder": encoder_config,
    }
    return save_inference_checkpoint(model, tokenizer, payload, out_dir)


def _write_checkpoint(epoch_path, last_path, model, optimizer, scaler, **payload) -> None:
    blob = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        **payload,
    }
    torch.save(blob, epoch_path)
    torch.save(blob, last_path)


def _load_last(path: Path, model, optimizer, scaler):
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    scaler.load_state_dict(blob["scaler"])
    return blob


def _learning_rate(step: int, total: int, warmup_ratio: float, base: float) -> float:
    warm = max(1, int(total * warmup_ratio))
    if step <= warm:
        return base * step / warm
    remain = max(1, total - warm)
    progress = (step - warm) / remain
    return base * max(0.0, 1.0 - progress)


def _print_eta(step: int, total: int, started: float) -> None:
    if step != 1 and step % 10 != 0 and step != total:
        return
    elapsed = time.perf_counter() - started
    remaining = (total - step) * (elapsed / step)
    print(f"step={step}/{total} eta_min={remaining / 60:.1f}")


def _print_banner(*, kind, tiny, device, fp16, model, frozen_layers, total_steps, config) -> None:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if kind == "harness" or tiny:
        print("HARNESS CHECK")
        print("tiny random encoder; this is not the Colab T4 run and not a result")
    else:
        print(
            "Budget: Davlan/afro-xlmr-base, embeddings and the bottom "
            f"{config.get('freeze_bottom_layers', 8)} of 12 layers frozen, "
            "fp16 on one Colab T4, batch "
            f"{config.get('batch_size', 16)}, max length {config.get('max_length', 128)}, "
            f"up to {config.get('epochs', 6)} epochs. Sized for about 60-90 minutes "
            "on the real label split. eta_min is an upper bound before early stopping."
        )
    print(
        f"device={device.type} fp16={fp16} frozen_layers={frozen_layers} "
        f"trainable_params={trainable} total_params={total} planned_steps={total_steps}"
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
