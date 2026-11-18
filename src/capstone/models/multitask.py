"""AfroXLMR-base with three heads, and the inference loader the API calls.

The real encoder is ``Davlan/afro-xlmr-base`` (XLM-RoBERTa, 12 layers).
Embeddings and the bottom 8 layers stay frozen. The top 4 layers and the
heads train. Urgency is a CORN head with 2 conditional logits (3 ordinal
levels). ``--tiny`` builds a random 4-layer encoder so smoke does not download
the checkpoint.

``load_for_inference(directory).predict(texts)`` runs on CPU. Each record has
the issue label and probabilities, the sentiment label and probability, and
the urgency level, probabilities, and emergency flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import XLMRobertaConfig, XLMRobertaModel

from capstone.labels import ISSUE_CLASSES, SENTIMENT_CLASSES, URGENCY_CLASSES
from capstone.models.baselines import pack_prediction
from capstone.models.corn import corn_class_probs, corn_loss
from capstone.pii import mask_pii

MODEL_ID = "Davlan/afro-xlmr-base"
TINY_HIDDEN = 32
TINY_LAYERS = 4
TINY_HEADS = 4
TINY_VOCAB = 128


class CharTokenizer:
    """Byte tokenizer for the random tiny encoder. No download."""

    def __init__(self, max_length: int = 64) -> None:
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.vocab_size = TINY_VOCAB
        self.model_max_length = max_length

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        limit = max_length or self.model_max_length
        rows: list[list[int]] = []
        for text in texts:
            ids = [self.bos_token_id]
            piece = text if not truncation else text[: max(0, limit - 2)]
            for char in piece:
                ids.append(4 + (ord(char) % (self.vocab_size - 4)))
            ids.append(self.eos_token_id)
            rows.append(ids[:limit])
        width = max(len(row) for row in rows) if padding else max(len(row) for row in rows)
        input_ids = []
        mask = []
        for row in rows:
            padded = row + [self.pad_token_id] * (width - len(row))
            input_ids.append(padded)
            mask.append([1] * len(row) + [0] * (width - len(row)))
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": mask}

    def save(self, directory: Path) -> None:
        payload = {
            "type": "char",
            "vocab_size": self.vocab_size,
            "max_length": self.model_max_length,
        }
        (directory / "char_tokenizer.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> CharTokenizer:
        payload = json.loads((directory / "char_tokenizer.json").read_text(encoding="utf-8"))
        tokenizer = cls(max_length=int(payload.get("max_length", 64)))
        tokenizer.vocab_size = int(payload.get("vocab_size", TINY_VOCAB))
        return tokenizer


class MultiTaskEncoder(nn.Module):
    """Shared encoder plus issue, sentiment, and CORN urgency heads."""

    def __init__(self, encoder: XLMRobertaModel, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.issue = nn.Linear(hidden_size, len(ISSUE_CLASSES))
        self.sentiment = nn.Linear(hidden_size, len(SENTIMENT_CLASSES))
        self.urgency = nn.Linear(hidden_size, len(URGENCY_CLASSES) - 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return self.issue(pooled), self.sentiment(pooled), self.urgency(pooled)


def build_encoder(tiny: bool, model_name: str = MODEL_ID) -> tuple[XLMRobertaModel, dict]:
    """Return the encoder and a JSON-serialisable config that rebuilds it."""
    if tiny:
        config = XLMRobertaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            intermediate_size=TINY_HIDDEN * 2,
            max_position_embeddings=128,
            type_vocab_size=1,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        return XLMRobertaModel(config), config.to_dict()
    encoder = XLMRobertaModel.from_pretrained(model_name)
    return encoder, encoder.config.to_dict()


def freeze_bottom(
    encoder: XLMRobertaModel,
    *,
    bottom_layers: int,
    freeze_embeddings: bool,
) -> int:
    """Freeze embeddings and the first ``bottom_layers`` encoder blocks.

    If the encoder has fewer layers than requested, the last block stays
    trainable so a tiny model can still take a gradient step. Returns how
    many blocks were frozen.
    """
    if freeze_embeddings:
        for parameter in encoder.embeddings.parameters():
            parameter.requires_grad = False
        pooler = getattr(encoder, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad = False
    layers = list(encoder.encoder.layer)
    limit = min(bottom_layers, max(0, len(layers) - 1))
    for index, layer in enumerate(layers):
        trainable = index >= limit
        for parameter in layer.parameters():
            parameter.requires_grad = trainable
    return limit


def masked_loss(
    issue_logits: torch.Tensor,
    sentiment_logits: torch.Tensor,
    urgency_logits: torch.Tensor,
    issue_y: torch.Tensor,
    sentiment_y: torch.Tensor,
    urgency_y: torch.Tensor,
    urgency_weight: torch.Tensor,
    class_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sum of the heads that have at least one labelled row in the batch.

    ``-100`` is the missing-label index. Urgency uses CORN, multiplied by the
    row weight and the urgency class weight.
    """
    loss = issue_logits.new_zeros(())
    if bool((issue_y != -100).any()):
        loss = loss + F.cross_entropy(
            issue_logits, issue_y, weight=class_weights["issue"], ignore_index=-100
        )
    if bool((sentiment_y != -100).any()):
        loss = loss + F.cross_entropy(
            sentiment_logits,
            sentiment_y,
            weight=class_weights["sentiment"],
            ignore_index=-100,
        )
    labelled = urgency_y != -100
    if bool(labelled.any()):
        loss = loss + corn_loss(
            urgency_logits[labelled],
            urgency_y[labelled],
            num_classes=len(URGENCY_CLASSES),
            sample_weight=urgency_weight[labelled],
            class_weight=class_weights["urgency"],
        )
    return loss


class MultiTaskPredictor:
    """CPU inference wrapper. ``predict`` masks PII before the encoder."""

    name = "afroxlmr"

    def __init__(
        self,
        model: MultiTaskEncoder,
        tokenizer,
        *,
        max_length: int,
        threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.threshold = threshold
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str], emergency_threshold: float | None = None) -> list[dict]:
        if not texts:
            return []
        threshold = self.threshold if emergency_threshold is None else emergency_threshold
        masked = [mask_pii(text) for text in texts]
        rows: list[dict] = []
        batch_size = 16
        with torch.no_grad():
            for start in range(0, len(masked), batch_size):
                chunk = masked[start : start + batch_size]
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                issue_logits, sentiment_logits, urgency_logits = self.model(
                    encoded["input_ids"].to(self.device),
                    encoded["attention_mask"].to(self.device),
                )
                issue_probs = torch.softmax(issue_logits, dim=1).cpu()
                sentiment_probs = torch.softmax(sentiment_logits, dim=1).cpu()
                urgency_probs = corn_class_probs(urgency_logits).cpu()
                for index in range(issue_probs.size(0)):
                    rows.append(
                        _row_from_probs(
                            issue_probs[index],
                            sentiment_probs[index],
                            urgency_probs[index],
                            threshold,
                        )
                    )
        return rows


def load_for_inference(directory: str | Path, threshold: float | None = None) -> MultiTaskPredictor:
    """Load a saved multi-task checkpoint. This is the function the API calls."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    encoder = XLMRobertaModel(XLMRobertaConfig.from_dict(config["encoder"]))
    model = MultiTaskEncoder(encoder, hidden_size=int(config["hidden_size"]))
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if config.get("tokenizer_type") == "char":
        tokenizer = CharTokenizer.load(directory)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory)
    chosen = threshold
    if chosen is None:
        threshold_path = directory / "threshold.json"
        if threshold_path.is_file():
            chosen = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        else:
            chosen = 0.5
    return MultiTaskPredictor(
        model,
        tokenizer,
        max_length=int(config["max_length"]),
        threshold=float(chosen),
    )


def save_inference_checkpoint(
    model: MultiTaskEncoder,
    tokenizer,
    config: dict,
    directory: Path,
) -> str:
    """Write state dict, config, tokenizer, and the state-dict SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.pt"
    torch.save(model.state_dict(), model_path)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if config.get("tokenizer_type") == "char":
        tokenizer.save(directory)
    else:
        tokenizer.save_pretrained(directory)
    digest = _sha256(model_path)
    (directory / "sha256.txt").write_text(f"{digest}  model.pt\n", encoding="utf-8")
    return digest


def _row_from_probs(
    issue_probs: torch.Tensor,
    sentiment_probs: torch.Tensor,
    urgency_probs: torch.Tensor,
    threshold: float,
) -> dict:
    issue = {label: float(issue_probs[index]) for index, label in enumerate(ISSUE_CLASSES)}
    sentiment = {
        label: float(sentiment_probs[index]) for index, label in enumerate(SENTIMENT_CLASSES)
    }
    urgency = {label: float(urgency_probs[index]) for index, label in enumerate(URGENCY_CLASSES)}
    issue_label = max(issue, key=issue.get)
    sentiment_label = max(sentiment, key=sentiment.get)
    urgency_label = max(urgency, key=urgency.get)
    return pack_prediction(
        issue_label,
        issue,
        sentiment_label,
        sentiment,
        urgency_label,
        urgency,
        urgency["emergency"] >= threshold,
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
