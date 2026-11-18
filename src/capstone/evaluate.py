"""Score every baseline and the multi-task model on the test split.

The emergency threshold is chosen on validation only, then applied once to
test. Bootstrap intervals use 1,000 resamples and seed 42. Paired intervals
compare each model with TF-IDF on the same resamples.

Writes ``reports/results.json`` and ``reports/results.md``. Mistakes go to
``data/private/errors.csv``, which is git-ignored, with a category hint and a
flag on the 30 mistakes set aside for the error taxonomy. Message text is not
written into the report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from capstone.models.baselines import (
    KeywordPredictor,
    load_majority,
    load_tfidf,
)
from capstone.models.data import LABEL_LISTS, head_splits
from capstone.models.metrics import (
    EMERGENCY_FALSE_RATE_CAP,
    bootstrap_ci,
    choose_emergency_threshold,
    emergency_rates,
    paired_bootstrap_diff,
    per_class_scores,
    quadratic_weighted_kappa,
)
from capstone.models.multitask import load_for_inference

TAXONOMY_PICKS = 30
_MODEL_ORDER = ("majority", "keyword", "tfidf", "afroxlmr")
URGENCY_ORDER = {"low": 0, "medium": 1, "emergency": 2}
MONEY = {"Airtel_Money_Transfer", "Airtel_Money_Reversal"}
NETWORK = {"Network_Issues", "Data_Bundle_Problems", "Router_WiFi_5G"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate heads on the test split.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--tfidf", type=Path, default=Path("artifacts/tfidf"))
    parser.add_argument("--transformer", type=Path, default=Path("artifacts/afroxlmr"))
    parser.add_argument("--baselines", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--errors", type=Path, default=Path("data/private/errors.csv"))
    parser.add_argument("--kind", choices=("harness", "data"), default="data")
    parser.add_argument("--cap", type=float, default=EMERGENCY_FALSE_RATE_CAP)
    args = parser.parse_args(argv)
    result = evaluate(
        label_dir=args.labels,
        manifest_dir=args.manifests,
        tfidf_dir=args.tfidf,
        transformer_dir=args.transformer,
        baselines_dir=args.baselines,
        out_dir=args.out,
        errors_path=args.errors,
        kind=args.kind,
        cap=args.cap,
    )
    if args.kind == "harness":
        print("HARNESS CHECK")
    else:
        print("EVALUATION")
    for head, payload in result["heads"].items():
        print(f"{head} n={payload['n']}")
    print(f"results={args.out / 'results.json'}")
    print(f"errors={args.errors}")
    return 0


def evaluate(
    *,
    label_dir: Path,
    manifest_dir: Path,
    tfidf_dir: Path,
    transformer_dir: Path,
    baselines_dir: Path,
    out_dir: Path,
    errors_path: Path,
    kind: str,
    cap: float = EMERGENCY_FALSE_RATE_CAP,
) -> dict:
    splits = head_splits(label_dir, manifest_dir)
    predictors, skipped = _predictors(tfidf_dir, transformer_dir, baselines_dir)
    thresholds = {}
    for name, predictor in predictors.items():
        thresholds[name] = _threshold_for(predictor, splits["urgency"]["val"], cap)
        _write_threshold(predictor, name, thresholds[name], tfidf_dir, transformer_dir)

    heads = {}
    error_rows: list[dict] = []
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for head, classes in LABEL_LISTS.items():
        test = splits[head]["test"]
        gold = [row.label for row in test]
        texts = [row.text for row in test]
        model_preds = {}
        model_flags = {}
        model_blocks = {}
        for name in _MODEL_ORDER:
            if name in skipped:
                model_blocks[name] = skipped[name]
                continue
            predictor = predictors[name]
            predictions = predictor.predict(
                texts, emergency_threshold=thresholds[name]["threshold"]
            )
            labels = [item[head]["label"] for item in predictions]
            model_preds[name] = labels
            if head == "urgency":
                model_flags[name] = [
                    "emergency" if item["urgency"]["emergency"] else "other" for item in predictions
                ]
            block = _score_head(head, gold, labels, predictions, list(classes))
            model_blocks[name] = block
            _save_confusion(
                figures / f"{head}_{name}.png", block["confusion"], list(classes), f"{head} {name}"
            )
            error_rows.extend(_errors(head, test, gold, predictions))
        heads[head] = {
            "n": len(test),
            "per_class_n": {name: sum(label == name for label in gold) for name in classes},
            "models": model_blocks,
            "paired_vs_tfidf": _paired(
                head, gold, model_preds, list(classes), model_flags if head == "urgency" else None
            ),
        }

    taxonomy = _mark_taxonomy(error_rows)
    payload = {
        "kind": kind,
        "banner": _banner(kind),
        "seed": 42,
        "bootstrap_resamples": 1000,
        "emergency_threshold": {
            "cap": cap,
            "by_model": thresholds,
        },
        "not_evaluated": skipped,
        "heads": heads,
        "taxonomy_picks": taxonomy,
        "figures": sorted(str(path.relative_to(out_dir)) for path in figures.glob("*.png")),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "results.md").write_text(_markdown(payload), encoding="utf-8")
    _write_errors(errors_path, error_rows)
    return payload


def _predictors(tfidf_dir: Path, transformer_dir: Path, baselines_dir: Path) -> tuple[dict, dict]:
    """Load every artefact that is on disk. A missing model is not a crash."""
    predictors: dict = {}
    skipped: dict = {}
    majority_path = baselines_dir / "majority.json"
    if majority_path.is_file():
        majority = load_majority(majority_path)
        predictors["majority"] = majority
        predictors["keyword"] = KeywordPredictor(majority.fitted)
    else:
        note = _not_evaluated(f"{majority_path} is absent")
        skipped["majority"] = note
        skipped["keyword"] = note
    tfidf_ready = all(
        (tfidf_dir / name).is_file()
        for name in ("issue.joblib", "sentiment.joblib", "urgency.joblib", "labels.json")
    )
    if tfidf_ready:
        predictors["tfidf"] = load_tfidf(tfidf_dir)
    else:
        skipped["tfidf"] = _not_evaluated(f"{tfidf_dir} is absent")
    transformer = _load_transformer(transformer_dir)
    if isinstance(transformer, dict):
        skipped["afroxlmr"] = transformer
    else:
        predictors["afroxlmr"] = transformer
    if not predictors:
        raise RuntimeError("no model artefacts found to evaluate")
    return predictors, skipped


def _not_evaluated(reason: str) -> dict:
    return {"status": "not evaluated", "reason": reason}


def _load_transformer(directory: Path):
    directory = Path(directory)
    if not (directory / "model.pt").is_file():
        reason = "artifacts/afroxlmr is absent"
        if directory.name != "afroxlmr":
            reason = f"{directory} is absent"
        return _not_evaluated(reason)
    try:
        return load_for_inference(directory)
    except Exception as exc:
        return _not_evaluated(f"{directory} could not be loaded ({type(exc).__name__})")


def _scored(model: dict) -> bool:
    return model.get("status") != "not evaluated"


def _threshold_for(predictor, val_rows, cap: float) -> dict:
    if not val_rows:
        raise RuntimeError("urgency validation split is empty; cannot choose a threshold")
    predictions = predictor.predict([row.text for row in val_rows], emergency_threshold=0.5)
    gold = [row.label for row in val_rows]
    probs = [item["urgency"]["probs"]["emergency"] for item in predictions]
    return choose_emergency_threshold(gold, probs, cap=cap)


def _write_threshold(
    predictor, name: str, chosen: dict, tfidf_dir: Path, transformer_dir: Path
) -> None:
    predictor.threshold = chosen["threshold"]
    target = None
    if name == "tfidf":
        target = tfidf_dir / "threshold.json"
    elif name == "afroxlmr":
        target = transformer_dir / "threshold.json"
    if target is not None:
        target.write_text(json.dumps(chosen, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score_head(head, gold, labels, predictions, classes) -> dict:
    per_class, macro, matrix = per_class_scores(gold, labels, classes)

    def macro_metric(gold_s, pred_s):
        return per_class_scores(gold_s, pred_s, classes)[1]

    block = {
        "n": len(gold),
        "per_class": per_class,
        "confusion": matrix,
        "macro_f1": bootstrap_ci(gold, labels, macro_metric),
    }
    if head == "urgency":
        flags = [bool(item["urgency"]["emergency"]) for item in predictions]
        recall, false_rate, n_pos, n_neg = emergency_rates(gold, flags)
        block["qwk"] = bootstrap_ci(
            gold,
            labels,
            lambda gold_s, pred_s: quadratic_weighted_kappa(gold_s, pred_s, classes),
        )
        block["emergency_recall"] = {
            "estimate": recall,
            "n_emergency": n_pos,
            **_flag_ci(gold, flags, which="recall"),
        }
        block["false_emergency_rate"] = {
            "estimate": false_rate,
            "n_non_emergency": n_neg,
            **_flag_ci(gold, flags, which="false_rate"),
        }
    return block


def _flag_ci(gold: list[str], flags: list[bool], which: str) -> dict:
    pred = ["emergency" if flag else "other" for flag in flags]

    def metric(gold_s, pred_s):
        decoded = [item == "emergency" for item in pred_s]
        recall, false_rate, n_pos, n_neg = emergency_rates(gold_s, decoded)
        if which == "recall":
            return None if n_pos == 0 else recall
        return None if n_neg == 0 else false_rate

    interval = bootstrap_ci(gold, pred, metric)
    return {"ci95": interval["ci95"], "resamples_used": interval["resamples_used"]}


def _paired(
    head: str,
    gold: list[str],
    model_preds: dict[str, list[str]],
    classes: list[str],
    flags: dict[str, list[str]] | None,
) -> dict:
    if "tfidf" not in model_preds:
        return {}
    reference = model_preds["tfidf"]

    def macro_metric(gold_s, pred_s):
        return per_class_scores(gold_s, pred_s, classes)[1]

    paired = {}
    for name, predicted in model_preds.items():
        if name == "tfidf":
            continue
        entry = {"macro_f1_diff": paired_bootstrap_diff(gold, predicted, reference, macro_metric)}
        if head == "urgency":
            entry["qwk_diff"] = paired_bootstrap_diff(
                gold,
                predicted,
                reference,
                lambda gold_s, pred_s: quadratic_weighted_kappa(gold_s, pred_s, classes),
            )
            if flags is not None:
                entry["emergency_recall_diff"] = paired_bootstrap_diff(
                    gold,
                    flags[name],
                    flags["tfidf"],
                    _recall_from_flags,
                )
        paired[name] = entry
    return paired


def _recall_from_flags(gold: list[str], pred: list[str]) -> float | None:
    decoded = [item == "emergency" for item in pred]
    recall, _false_rate, n_pos, _n_neg = emergency_rates(gold, decoded)
    if n_pos == 0:
        return None
    return recall


def _errors(head: str, examples, gold: list[str], predictions: list[dict]) -> list[dict]:
    rows = []
    for example, truth, prediction in zip(examples, gold, predictions, strict=True):
        label = prediction[head]["label"]
        if label == truth:
            continue
        probs = prediction[head].get("probs") or {}
        confidence = float(probs.get(label, prediction[head].get("prob", 0.0)))
        rows.append(
            {
                "head": head,
                "file": example.file,
                "row_id": example.row_id,
                "gold": truth,
                "pred": label,
                "confidence": f"{confidence:.6f}",
                "category_hint": _hint(head, truth, label),
                "taxonomy_pick": "false",
                "text": example.text,
            }
        )
    return rows


def _hint(head: str, gold: str, pred: str) -> str:
    if head == "urgency":
        if gold == "emergency" and pred != "emergency":
            return "emergency_miss"
        if gold != "emergency" and pred == "emergency":
            return "false_emergency"
        if abs(URGENCY_ORDER[gold] - URGENCY_ORDER[pred]) == 1:
            return "urgency_adjacent"
        return "urgency_skip"
    if head == "sentiment":
        return "sentiment_flip"
    pair = {gold, pred}
    if pair == MONEY:
        return "issue_money_mixup"
    if pair <= NETWORK:
        return "issue_network_family"
    return "issue_other"


def _mark_taxonomy(rows: list[dict]) -> int:
    ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    for row in ranked[:TAXONOMY_PICKS]:
        row["taxonomy_pick"] = "true"
    return min(TAXONOMY_PICKS, len(rows))


def _write_errors(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "head",
        "file",
        "row_id",
        "gold",
        "pred",
        "confidence",
        "category_hint",
        "taxonomy_pick",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion(path: Path, matrix: list[list[int]], classes: list[str], title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), labels=classes, rotation=90, fontsize=7)
    axis.set_yticks(range(len(classes)), labels=classes, fontsize=7)
    axis.set_xlabel("predicted")
    axis.set_ylabel("gold")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _banner(kind: str) -> str:
    if kind == "harness":
        return (
            "HARNESS CHECK. These numbers are from the synthetic sample. "
            "They are not modelling results."
        )
    return (
        "Test-split evaluation. The emergency threshold was chosen on validation only. "
        "Not a training loss."
    )


def _markdown(payload: dict) -> str:
    lines = ["# Evaluation", "", f"> {payload['banner']}", ""]
    by_model = payload["emergency_threshold"]["by_model"]
    if "tfidf" in by_model:
        rule = by_model["tfidf"]["rule"]
    elif by_model:
        rule = next(iter(by_model.values()))["rule"]
    else:
        rule = "No model was loaded, so no validation threshold was chosen."
    lines.append(f"Emergency threshold rule: {rule}")
    lines.append("")
    lines.append("| model | threshold | val recall | val false-emergency rate | met cap |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for name, chosen in payload["emergency_threshold"]["by_model"].items():
        lines.append(
            f"| {name} | {chosen['threshold']:.2f} | {chosen['val_emergency_recall']:.3f} | "
            f"{chosen['val_false_emergency_rate']:.3f} | {chosen['meets_cap']} |"
        )
    for head, block in payload["heads"].items():
        lines.extend(["", f"## {head}", "", f"n = {block['n']}", ""])
        lines.append("Class support (gold n):")
        lines.append("")
        for name, count in block["per_class_n"].items():
            lines.append(f"- {name}: {count}")
        lines.extend(["", "| model | macro-F1 | 95% CI |", "| --- | ---: | --- |"])
        for name, model in block["models"].items():
            if not _scored(model):
                lines.append(f"| {name} | not evaluated | |")
                continue
            estimate = model["macro_f1"]["estimate"]
            low, high = model["macro_f1"]["ci95"]
            lines.append(f"| {name} | {estimate:.3f} | {_fmt_ci(low, high)} |")
        if head == "urgency":
            lines.extend(
                [
                    "",
                    "| model | QWK | emergency recall | false-emergency rate | "
                    "n emergency | n other |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, model in block["models"].items():
                if not _scored(model):
                    lines.append(f"| {name} | not evaluated | | | | |")
                    continue
                lines.append(
                    f"| {name} | {model['qwk']['estimate']:.3f} | "
                    f"{model['emergency_recall']['estimate']:.3f} | "
                    f"{model['false_emergency_rate']['estimate']:.3f} | "
                    f"{model['emergency_recall']['n_emergency']} | "
                    f"{model['false_emergency_rate']['n_non_emergency']} |"
                )
        lines.extend(["", "Per-class F1 (n is the gold support):", ""])
        class_names = list(block["per_class_n"])
        header = "| model | " + " | ".join(class_names) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---:" for _ in class_names) + " |")
        for name, model in block["models"].items():
            if not _scored(model):
                blanks = " | ".join(["not evaluated"] * len(class_names))
                lines.append(f"| {name} | {blanks} |")
                continue
            cells = []
            for class_name in class_names:
                stats = model["per_class"][class_name]
                cells.append(f"{stats['f1']:.2f} (n={stats['n']})")
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
        lines.extend(["", "Paired difference versus TF-IDF (model minus TF-IDF), macro-F1:", ""])
        if not block["paired_vs_tfidf"]:
            lines.append("- not evaluated (TF-IDF was not scored)")
        for name, diff in block["paired_vs_tfidf"].items():
            estimate = diff["macro_f1_diff"]["estimate"]
            low, high = diff["macro_f1_diff"]["ci95"]
            lines.append(f"- {name}: {estimate:.3f}, 95% CI {_fmt_ci(low, high)}")
    lines.extend(["", f"Taxonomy picks: {payload['taxonomy_picks']} (cap {TAXONOMY_PICKS}).", ""])
    return "\n".join(lines)


def _fmt_ci(low, high) -> str:
    if low is None or high is None:
        return "n/a"
    return f"[{low:.3f}, {high:.3f}]"


if __name__ == "__main__":
    raise SystemExit(main())
