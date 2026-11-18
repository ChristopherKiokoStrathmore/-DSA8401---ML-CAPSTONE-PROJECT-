"""Load the Phase 3 artefacts and normalise their predictions.

The transformer contract, implemented by Phase 3, is::

    capstone.models.multitask.load_for_inference(directory) -> model
    model.predict(texts: list[str]) -> list[dict]

Each dict is::

    {
        "issue": {"label": str, "probs": {label: float, ...}},
        "sentiment": {"label": str, "prob": float},
        "urgency": {
            "level": "low" | "medium" | "emergency",
            "probs": {"low": float, "medium": float, "emergency": float},
            "emergency": bool,
        },
    }

A flat dict that already uses the public field names (``issue``,
``sentiment``, and ``urgency`` as strings) is accepted too. This module
imports ``capstone.models.multitask`` only when ``MODEL_BACKEND=transformer``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from capstone.labels import ISSUE_CLASSES, SENTIMENT_CLASSES, URGENCY_CLASSES

_URGENCY_SCORE = {"low": 0.0, "medium": 0.5, "emergency": 1.0}
_TFIDF_FILES = ("issue.joblib", "sentiment.joblib", "urgency.joblib", "labels.json")


class ModelNotReady(RuntimeError):
    """The configured backend cannot serve. The API stays up and returns 503."""


class InferenceError(RuntimeError):
    """A loaded model returned a payload this API cannot publish."""


class Predictor(Protocol):
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict already-masked texts. Do not log ``texts``."""


@dataclass(frozen=True)
class LoadedModel:
    predictor: Predictor
    labels: dict[str, list[str]]
    device: str
    checkpoint: str


def canonical_labels() -> dict[str, list[str]]:
    return {
        "issue": list(ISSUE_CLASSES),
        "sentiment": list(SENTIMENT_CLASSES),
        "urgency": list(URGENCY_CLASSES),
    }


def load_backend(backend: str, tfidf_dir: Path, transformer_dir: Path) -> LoadedModel:
    if backend == "tfidf":
        return load_tfidf(tfidf_dir)
    if backend == "transformer":
        return load_transformer(transformer_dir)
    raise ModelNotReady(f"unknown MODEL_BACKEND {backend!r}; expected 'tfidf' or 'transformer'")


def load_tfidf(directory: Path) -> LoadedModel:
    missing = [name for name in _TFIDF_FILES if not (directory / name).is_file()]
    if missing:
        raise ModelNotReady(
            "tfidf backend unavailable: missing " + ", ".join(missing) + f" under {directory}"
        )
    try:
        import joblib
    except ImportError as exc:
        raise ModelNotReady("tfidf backend unavailable: joblib is not installed") from exc

    labels = _read_labels(directory / "labels.json")
    pipelines = {}
    for head in ("issue", "sentiment", "urgency"):
        path = directory / f"{head}.joblib"
        try:
            pipe = joblib.load(path)
        except Exception as exc:
            raise ModelNotReady(
                f"tfidf backend unavailable: could not read {path.name} ({type(exc).__name__})"
            ) from exc
        if not hasattr(pipe, "predict_proba"):
            raise ModelNotReady(f"tfidf backend unavailable: {path.name} has no predict_proba")
        found = {str(label) for label in _pipeline_classes(pipe)}
        if found != set(labels[head]):
            raise ModelNotReady(
                f"tfidf backend unavailable: {path.name} classes do not match labels.json"
            )
        pipelines[head] = pipe
    return LoadedModel(
        predictor=TfidfPredictor(pipelines, labels),
        labels=labels,
        device="cpu",
        checkpoint=str(directory),
    )


def load_transformer(directory: Path) -> LoadedModel:
    """Import Phase 3 only here. A missing module is a 503, not a crash."""
    try:
        from capstone.models.multitask import load_for_inference
    except ImportError as exc:
        raise ModelNotReady(
            "transformer backend unavailable: capstone.models.multitask could not be imported"
        ) from exc
    try:
        model = load_for_inference(directory)
    except Exception as exc:
        raise ModelNotReady(
            "transformer backend unavailable: capstone.models.multitask.load_for_inference "
            f"failed ({type(exc).__name__})"
        ) from exc
    if not hasattr(model, "predict"):
        raise ModelNotReady("transformer backend unavailable: model has no predict method")
    device = getattr(model, "device", "cpu")
    labels = _labels_from_model(model)
    return LoadedModel(
        predictor=TransformerPredictor(model),
        labels=labels,
        device=str(device),
        checkpoint=str(directory),
    )


class TfidfPredictor:
    def __init__(self, pipelines: dict[str, Any], labels: dict[str, list[str]]) -> None:
        self._pipelines = pipelines
        self._labels = labels

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        issue_rows = self._pipelines["issue"].predict_proba(texts)
        sentiment_rows = self._pipelines["sentiment"].predict_proba(texts)
        urgency_rows = self._pipelines["urgency"].predict_proba(texts)
        predictions: list[dict[str, Any]] = []
        for index in range(len(texts)):
            issue_probs = _proba_map(self._pipelines["issue"], issue_rows[index])
            sentiment_probs = _proba_map(self._pipelines["sentiment"], sentiment_rows[index])
            urgency_probs = _proba_map(self._pipelines["urgency"], urgency_rows[index])
            issue = _top(issue_probs, self._labels["issue"])
            sentiment = _top(sentiment_probs, self._labels["sentiment"])
            urgency = _top(urgency_probs, self._labels["urgency"])
            predictions.append(
                {
                    "issue": issue,
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "urgency_score": _urgency_score(urgency_probs),
                    "confidence": {
                        "issue": issue_probs[issue],
                        "sentiment": sentiment_probs[sentiment],
                    },
                }
            )
        return predictions


class TransformerPredictor:
    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        try:
            raw = self._model.predict(list(texts))
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"transformer predict failed ({type(exc).__name__})") from exc
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise InferenceError("transformer predict returned the wrong number of rows")
        return [to_public(item) for item in raw]


def to_public(item: Any) -> dict[str, Any]:
    """Map one model row onto the public prediction schema."""
    if not isinstance(item, dict):
        raise InferenceError("prediction was not an object")
    issue = item.get("issue")
    sentiment = item.get("sentiment")
    urgency = item.get("urgency")
    if isinstance(issue, str) and isinstance(sentiment, str) and isinstance(urgency, str):
        return _public_flat(item)
    if isinstance(issue, dict) or isinstance(urgency, dict):
        return _public_rich(item)
    raise InferenceError("prediction did not match the inference contract")


def _public_flat(item: dict[str, Any]) -> dict[str, Any]:
    confidence = item.get("confidence") or {}
    if not isinstance(confidence, dict):
        confidence = {}
    score = item.get("urgency_score")
    return {
        "issue": str(item["issue"]),
        "sentiment": str(item["sentiment"]),
        "urgency": str(item["urgency"]),
        "urgency_score": None if score is None else float(score),
        "confidence": {
            "issue": _optional_float(confidence.get("issue")),
            "sentiment": _optional_float(confidence.get("sentiment")),
        },
    }


def _public_rich(item: dict[str, Any]) -> dict[str, Any]:
    issue_obj = item.get("issue") if isinstance(item.get("issue"), dict) else {}
    sentiment_obj = item.get("sentiment") if isinstance(item.get("sentiment"), dict) else {}
    urgency_obj = item.get("urgency") if isinstance(item.get("urgency"), dict) else {}

    issue_probs = _float_map(issue_obj.get("probs"))
    sentiment_probs = _float_map(sentiment_obj.get("probs"))
    urgency_probs = _float_map(urgency_obj.get("probs"))

    issue_label = issue_obj.get("label")
    if not isinstance(issue_label, str) or issue_label == "":
        if not issue_probs:
            raise InferenceError("issue prediction has no label")
        issue_label = _top(issue_probs, list(issue_probs))

    sentiment_label = sentiment_obj.get("label")
    if not isinstance(sentiment_label, str) or sentiment_label == "":
        if not sentiment_probs:
            raise InferenceError("sentiment prediction has no label")
        sentiment_label = _top(sentiment_probs, list(sentiment_probs))

    urgency_label = urgency_obj.get("level", urgency_obj.get("label"))
    if not isinstance(urgency_label, str) or urgency_label == "":
        if urgency_obj.get("emergency") is True:
            urgency_label = "emergency"
        elif urgency_probs:
            known = [label for label in URGENCY_CLASSES if label in urgency_probs]
            urgency_label = _top(urgency_probs, known or list(urgency_probs))
        else:
            raise InferenceError("urgency prediction has no level")

    sentiment_conf = _optional_float(sentiment_obj.get("prob"))
    if sentiment_conf is None and sentiment_probs:
        sentiment_conf = float(sentiment_probs.get(sentiment_label, max(sentiment_probs.values())))
    issue_conf = None
    if issue_probs:
        issue_conf = float(issue_probs.get(issue_label, max(issue_probs.values())))

    if urgency_probs:
        score = _urgency_score(urgency_probs)
    else:
        score = _URGENCY_SCORE.get(str(urgency_label))
    return {
        "issue": str(issue_label),
        "sentiment": str(sentiment_label),
        "urgency": str(urgency_label),
        "urgency_score": None if score is None else float(score),
        "confidence": {"issue": issue_conf, "sentiment": sentiment_conf},
    }


def _read_labels(path: Path) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelNotReady(
            f"tfidf backend unavailable: labels.json is not valid JSON ({type(exc).__name__})"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelNotReady("tfidf backend unavailable: labels.json must be an object")
    labels: dict[str, list[str]] = {}
    for head in ("issue", "sentiment", "urgency"):
        values = payload.get(head)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ModelNotReady(
                f"tfidf backend unavailable: labels.json {head} must be a list of strings"
            )
        labels[head] = list(values)
    if labels["urgency"] != ["low", "medium", "emergency"]:
        raise ModelNotReady(
            "tfidf backend unavailable: urgency labels must be low, medium, emergency in that order"
        )
    return labels


def _labels_from_model(model: Any) -> dict[str, list[str]]:
    raw = getattr(model, "labels", None)
    if not isinstance(raw, dict):
        return canonical_labels()
    labels: dict[str, list[str]] = {}
    for head in ("issue", "sentiment", "urgency"):
        values = raw.get(head)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            return canonical_labels()
        labels[head] = list(values)
    if labels["urgency"] != ["low", "medium", "emergency"]:
        return canonical_labels()
    return labels


def _pipeline_classes(pipe: Any) -> list[Any]:
    """Class order for a Phase 3 sklearn Pipeline, or any estimator with classes_."""
    if hasattr(pipe, "classes_"):
        return list(pipe.classes_)
    steps = getattr(pipe, "named_steps", None) or {}
    classifier = steps.get("clf")
    if classifier is not None and hasattr(classifier, "classes_"):
        return list(classifier.classes_)
    raise ModelNotReady("tfidf backend unavailable: pipeline has no classes_")


def _proba_map(pipe: Any, row: Any) -> dict[str, float]:
    classes = [str(label) for label in _pipeline_classes(pipe)]
    return {label: float(row[index]) for index, label in enumerate(classes)}


def _top(probs: dict[str, float], order: list[str]) -> str:
    ranked = [label for label in order if label in probs]
    if not ranked:
        raise InferenceError("prediction is missing class probabilities")
    return max(ranked, key=lambda label: (probs[label], -order.index(label)))


def _urgency_score(probs: dict[str, float]) -> float:
    return float(sum(probs.get(label, 0.0) * weight for label, weight in _URGENCY_SCORE.items()))


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    mapped: dict[str, float] = {}
    for key, prob in value.items():
        number = _optional_float(prob)
        if number is None:
            continue
        mapped[str(key)] = number
    return mapped


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number
