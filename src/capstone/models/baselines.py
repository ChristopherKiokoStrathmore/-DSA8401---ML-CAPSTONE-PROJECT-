"""Majority class, an urgency keyword heuristic, and TF-IDF logistic regression.

C for the logistic regression is chosen on the validation split only. The
fitted pipeline is then trained on the training split alone, so validation
stays available for the emergency threshold. Test rows are never passed in.

The saved TF-IDF files are sklearn Pipelines. ``predict_proba`` accepts raw
text and masks PII first, which is the format the API loads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from capstone.labels import ISSUE_CLASSES, URGENCY_CLASSES
from capstone.models.data import LABEL_LISTS, Example, head_splits
from capstone.models.metrics import quadratic_weighted_kappa
from capstone.pii import mask_pii, mask_texts

C_GRID: tuple[float, ...] = (0.25, 1.0, 4.0)
HEADS = ("issue", "sentiment", "urgency")

# Substring rules for the urgency heuristic. English, Swahili, and Sheng,
# matched on the masked lowercased text. Not learned from customer messages.
EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "emergency",
    "urgent",
    "dharura",
    "haraka",
    "stolen",
    "imeibiwa",
    "fraud",
    "scam",
    "police",
    "hospital",
    "died",
    "dying",
    "blocked",
    "imefungwa",
    "sim swap",
    "cannot withdraw",
    "can't withdraw",
    "akaunti imefungwa",
    "line blocked",
)
MEDIUM_KEYWORDS: tuple[str, ...] = (
    "refund",
    "reverse",
    "complaint",
    "not working",
    "haiendi",
    "haijaingia",
    "mbaya",
    "no signal",
    "down",
    "bundle",
)
ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Airtel_Money_Reversal": ("reverse", "reversal", "wrong person"),
    "Airtel_Money_Transfer": ("transfer", "till", "sent money", "payment"),
    "App_Rewards": ("reward", "points", "app"),
    "Data_Bundle_Problems": ("bundle", "1gb", "data"),
    "Network_Issues": ("signal", "network", "mtandao", "calls"),
    "Router_WiFi_5G": ("router", "wifi", "5g"),
    "SIM_Line_Services": ("sim", "line", "puk"),
    "Product_Enquiry": ("which", "offer", "tariff"),
    "Complaint_General": ("complaint", "service"),
    "Non_Actionable": ("thank", "asante", "fine now"),
}
NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "not",
    "no ",
    "fail",
    "bad",
    "mbaya",
    "complaint",
    "stolen",
    "down",
    "haija",
    "haiend",
)


def pack_prediction(
    issue_label: str,
    issue_probs: dict[str, float],
    sentiment_label: str,
    sentiment_probs: dict[str, float],
    urgency_label: str,
    urgency_probs: dict[str, float],
    emergency: bool,
) -> dict:
    """The record ``load_for_inference().predict`` and the baselines both return."""
    return {
        "issue": {"label": issue_label, "probs": issue_probs},
        "sentiment": {
            "label": sentiment_label,
            "prob": float(sentiment_probs[sentiment_label]),
            "probs": sentiment_probs,
        },
        "urgency": {
            "label": urgency_label,
            "probs": urgency_probs,
            "emergency": bool(emergency),
        },
    }


def fit_majority(examples: list[Example], classes: tuple[str, ...]) -> dict:
    counts = {name: 0 for name in classes}
    for example in examples:
        counts[example.label] = counts.get(example.label, 0) + 1
    total = max(1, sum(counts.values()))
    probs = {name: counts.get(name, 0) / total for name in classes}
    label = max(classes, key=lambda name: (probs[name], -classes.index(name)))
    return {"label": label, "probs": probs, "n_train": len(examples)}


class MajorityPredictor:
    """Constant prediction: the training mode, with the training frequencies as probs."""

    name = "majority"

    def __init__(self, fitted: dict[str, dict], threshold: float = 0.5) -> None:
        self.fitted = fitted
        self.threshold = threshold

    def predict(self, texts: list[str], emergency_threshold: float | None = None) -> list[dict]:
        threshold = self.threshold if emergency_threshold is None else emergency_threshold
        rows = []
        for _ in texts:
            urgency_probs = self.fitted["urgency"]["probs"]
            sentiment_probs = self.fitted["sentiment"]["probs"]
            rows.append(
                pack_prediction(
                    self.fitted["issue"]["label"],
                    self.fitted["issue"]["probs"],
                    self.fitted["sentiment"]["label"],
                    sentiment_probs,
                    self.fitted["urgency"]["label"],
                    urgency_probs,
                    urgency_probs["emergency"] >= threshold,
                )
            )
        return rows


class KeywordPredictor:
    """Urgency from an emergency/medium keyword list. Other heads use keywords, else majority."""

    name = "keyword"

    def __init__(self, majority: dict[str, dict], threshold: float = 0.5) -> None:
        self.majority = majority
        self.threshold = threshold

    def predict(self, texts: list[str], emergency_threshold: float | None = None) -> list[dict]:
        threshold = self.threshold if emergency_threshold is None else emergency_threshold
        return [self._one(mask_pii(text), threshold) for text in texts]

    def _one(self, text: str, threshold: float) -> dict:
        lowered = text.lower()
        urgency_probs = _urgency_keyword_probs(lowered)
        urgency_label = max(URGENCY_CLASSES, key=lambda name: urgency_probs[name])
        issue_probs, issue_label = _issue_keyword(lowered, self.majority["issue"])
        sentiment_probs, sentiment_label = _sentiment_keyword(lowered, self.majority["sentiment"])
        return pack_prediction(
            issue_label,
            issue_probs,
            sentiment_label,
            sentiment_probs,
            urgency_label,
            urgency_probs,
            urgency_probs["emergency"] >= threshold,
        )


class TfidfPredictor:
    """Three fitted pipelines. Probabilities follow ``labels.json`` order."""

    name = "tfidf"

    def __init__(
        self, pipelines: dict[str, Pipeline], labels: dict[str, list[str]], threshold: float = 0.5
    ) -> None:
        self.pipelines = pipelines
        self.labels = labels
        self.threshold = threshold

    def predict(self, texts: list[str], emergency_threshold: float | None = None) -> list[dict]:
        if not texts:
            return []
        threshold = self.threshold if emergency_threshold is None else emergency_threshold
        issue = _probs_for(self.pipelines["issue"], texts, self.labels["issue"])
        sentiment = _probs_for(self.pipelines["sentiment"], texts, self.labels["sentiment"])
        urgency = _probs_for(self.pipelines["urgency"], texts, self.labels["urgency"])
        rows = []
        for index in range(len(texts)):
            issue_label = max(issue[index], key=issue[index].get)
            sentiment_label = max(sentiment[index], key=sentiment[index].get)
            urgency_label = max(urgency[index], key=urgency[index].get)
            rows.append(
                pack_prediction(
                    issue_label,
                    issue[index],
                    sentiment_label,
                    sentiment[index],
                    urgency_label,
                    urgency[index],
                    urgency[index]["emergency"] >= threshold,
                )
            )
        return rows


def train_baselines(
    label_dir: Path,
    manifest_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    kind: str = "data",
) -> dict:
    """Fit all baselines. Returns the selection record. Does not read test rows."""
    splits = head_splits(label_dir, manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    majority = {}
    pipelines = {}
    labels = {}
    selection: dict = {
        "kind": kind,
        "seed": seed,
        "chosen_on": "validation",
        "c_grid": list(C_GRID),
        "heads": {},
    }
    for head in HEADS:
        train = splits[head]["train"]
        val = splits[head]["val"]
        if len(train) < 2 or len({row.label for row in train}) < 2:
            raise RuntimeError(f"{head} train split needs at least two classes to fit TF-IDF")
        classes = LABEL_LISTS[head]
        majority[head] = fit_majority(train, classes)
        pipeline, report = _select_c(head, train, val, classes, seed)
        pipelines[head] = pipeline
        labels[head] = list(classes)
        selection["heads"][head] = report
        joblib.dump(pipeline, out_dir / f"{head}.joblib")
    (out_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    baseline_dir = out_dir.parent / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "majority.json").write_text(
        json.dumps(majority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection


def load_tfidf(directory: Path, threshold: float = 0.5) -> TfidfPredictor:
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    pipelines = {head: joblib.load(directory / f"{head}.joblib") for head in HEADS}
    return TfidfPredictor(pipelines, labels, threshold=threshold)


def load_majority(path: Path, threshold: float = 0.5) -> MajorityPredictor:
    return MajorityPredictor(json.loads(path.read_text(encoding="utf-8")), threshold=threshold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit majority, keyword, and TF-IDF baselines.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tfidf"))
    parser.add_argument("--kind", choices=("harness", "data"), default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    selection = train_baselines(
        args.labels, args.manifests, args.out, seed=args.seed, kind=args.kind
    )
    if args.kind == "harness":
        print("HARNESS CHECK")
    else:
        print("BASELINES")
    for head, report in selection["heads"].items():
        print(
            f"{head} chosen_c={report['chosen_c']} val_{report['metric']}={report['val_score']:.4f}"
        )
    print(f"out={args.out}")
    return 0


def _select_c(
    head: str,
    train: list[Example],
    val: list[Example],
    classes: tuple[str, ...],
    seed: int,
) -> tuple[Pipeline, dict]:
    metric_name = "qwk" if head == "urgency" else "macro_f1"
    texts_train = [row.text for row in train]
    y_train = [row.label for row in train]
    weights = [row.weight for row in train] if head == "urgency" else None
    texts_val = [row.text for row in val]
    y_val = [row.label for row in val]
    scores: dict[str, float] = {}
    best_c = C_GRID[0]
    best_score = float("-inf")
    for candidate in C_GRID:
        pipeline = _pipeline(candidate, seed)
        _fit(pipeline, texts_train, y_train, weights)
        predicted = pipeline.predict(texts_val).tolist()
        score = _val_score(head, y_val, predicted, classes)
        scores[str(candidate)] = score
        if score > best_score:
            best_score = score
            best_c = candidate
    final = _pipeline(best_c, seed)
    _fit(final, texts_train, y_train, weights)
    return final, {
        "metric": metric_name,
        "chosen_c": best_c,
        "val_score": best_score,
        "val_scores": scores,
        "chosen_on": "validation",
    }


def _pipeline(c_value: float, seed: int) -> Pipeline:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    features = FeatureUnion([("word", word), ("char", char)])
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline(
        [
            ("mask", FunctionTransformer(mask_texts, validate=False)),
            ("features", features),
            ("clf", classifier),
        ]
    )


def _fit(
    pipeline: Pipeline, texts: list[str], labels: list[str], weights: list[float] | None
) -> None:
    if weights is None:
        pipeline.fit(texts, labels)
        return
    pipeline.fit(texts, labels, clf__sample_weight=np.asarray(weights, dtype=np.float64))


def _val_score(head: str, gold: list[str], pred: list[str], classes: tuple[str, ...]) -> float:
    if head == "urgency":
        return quadratic_weighted_kappa(gold, pred, list(classes))
    from sklearn.metrics import f1_score

    return float(f1_score(gold, pred, labels=list(classes), average="macro", zero_division=0))


def _probs_for(
    pipeline: Pipeline, texts: list[str], label_order: list[str]
) -> list[dict[str, float]]:
    probabilities = pipeline.predict_proba(texts)
    known = list(pipeline.named_steps["clf"].classes_)
    rows = []
    for row in probabilities:
        by_label = {label: 0.0 for label in label_order}
        for index, label in enumerate(known):
            by_label[str(label)] = float(row[index])
        rows.append(by_label)
    return rows


def _urgency_keyword_probs(text: str) -> dict[str, float]:
    if any(keyword in text for keyword in EMERGENCY_KEYWORDS):
        return {"low": 0.05, "medium": 0.15, "emergency": 0.80}
    if any(keyword in text for keyword in MEDIUM_KEYWORDS):
        return {"low": 0.15, "medium": 0.70, "emergency": 0.15}
    return {"low": 0.70, "medium": 0.25, "emergency": 0.05}


def _issue_keyword(text: str, majority: dict) -> tuple[dict[str, float], str]:
    hits = [
        name
        for name in ISSUE_CLASSES
        if any(keyword in text for keyword in ISSUE_KEYWORDS.get(name, ()))
    ]
    if not hits:
        return dict(majority["probs"]), majority["label"]
    label = hits[0]
    probs = {name: 0.02 for name in ISSUE_CLASSES}
    probs[label] = 1.0 - 0.02 * (len(ISSUE_CLASSES) - 1)
    return probs, label


def _sentiment_keyword(text: str, majority: dict) -> tuple[dict[str, float], str]:
    if any(keyword in text for keyword in NEGATIVE_KEYWORDS):
        return {"negative": 0.85, "not_negative": 0.15}, "negative"
    return dict(majority["probs"]), majority["label"]


if __name__ == "__main__":
    raise SystemExit(main())
