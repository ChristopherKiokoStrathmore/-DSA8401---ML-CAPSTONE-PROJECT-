"""Train a tiny TF-IDF fixture from the synthetic label files.

This is a harness for CI and ``make serve-smoke``. It is not the Phase 3
model and its predictions are not a quality result. Phase 3 writes the same
filenames under ``artifacts/tfidf/``; this command is what fills them when
those artefacts are not on disk yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capstone.labels import (
    ISSUE_CLASSES,
    SENTIMENT_CLASSES,
    URGENCY_CLASSES,
    load_label_dir,
)
from capstone.serving.backends import ModelNotReady

_HEADS = (
    ("issue", "issue", ISSUE_CLASSES),
    ("sentiment", "sentiment", SENTIMENT_CLASSES),
    ("urgency", "urgency", URGENCY_CLASSES),
)


def train_fixture(labels_dir: Path, out_dir: Path) -> Path:
    """Fit three pipelines on the sample tables and write them to ``out_dir``."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rows = load_label_dir(labels_dir)
    tables = {
        "issue": [(row.text, row.issue) for row in rows if row.table == "issue" and row.issue],
        "sentiment": [
            (row.text, row.sentiment)
            for row in rows
            if row.table == "sentiment_hand" and row.sentiment
        ],
        "urgency": [
            (row.text, row.urgency) for row in rows if row.table == "urgency_train" and row.urgency
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_payload: dict[str, list[str]] = {}
    for head, _attr, classes in _HEADS:
        pairs = tables[head]
        present = {label for _text, label in pairs}
        missing = [label for label in classes if label not in present]
        if missing:
            raise ModelNotReady(
                f"fixture cannot train {head}: sample is missing {', '.join(missing)}"
            )
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs"),
                ),
            ]
        )
        texts = [text for text, _label in pairs]
        targets = [label for _text, label in pairs]
        pipeline.fit(texts, targets)
        joblib.dump(pipeline, out_dir / f"{head}.joblib")
        labels_payload[head] = list(classes)
    (out_dir / "labels.json").write_text(
        json.dumps(labels_payload, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the smoke TF-IDF fixture.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/sample/labels"),
        help="Directory of synthetic label files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/tfidf"),
        help="Where to write the three pipelines and labels.json.",
    )
    args = parser.parse_args(argv)
    out = train_fixture(args.labels, args.out)
    print(f"wrote tfidf fixture to {out}")


if __name__ == "__main__":
    main()
