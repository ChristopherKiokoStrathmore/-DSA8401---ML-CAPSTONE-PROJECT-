"""Join split manifests back to label rows. Manifests do not store message text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from capstone.duplicates import normalise_message
from capstone.labels import (
    ISSUE_CLASSES,
    SENTIMENT_CLASSES,
    URGENCY_CLASSES,
    LabelRow,
    load_label_dir,
)

HEADS: tuple[str, ...] = ("issue", "sentiment", "urgency")
SPLITS: tuple[str, ...] = ("train", "val", "test")
LABEL_LISTS: dict[str, tuple[str, ...]] = {
    "issue": ISSUE_CLASSES,
    "sentiment": SENTIMENT_CLASSES,
    "urgency": URGENCY_CLASSES,
}


@dataclass(frozen=True)
class Example:
    """One labelled row for a single head."""

    text: str
    label: str
    weight: float
    file: str
    row_id: int


@dataclass(frozen=True)
class MultiExample:
    """One training text. A head is None when this row has no label for it."""

    text: str
    issue: str | None
    sentiment: str | None
    urgency: str | None
    weight: float


def load_manifests(manifest_dir: Path) -> dict[str, dict]:
    loaded = {}
    for head in HEADS:
        path = Path(manifest_dir) / f"{head}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest {path}")
        loaded[head] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def head_splits(label_dir: Path, manifest_dir: Path) -> dict[str, dict[str, list[Example]]]:
    """Return ``head -> split -> examples`` using only that head's manifest rows."""
    rows = {(row.file_name, row.row_id): row for row in load_label_dir(label_dir)}
    manifests = load_manifests(manifest_dir)
    splits: dict[str, dict[str, list[Example]]] = {}
    for head in HEADS:
        splits[head] = {}
        for split in SPLITS:
            examples: list[Example] = []
            seen: set[tuple[str, int]] = set()
            for record in manifests[head]["splits"][split]:
                key = (record["file"], int(record["row_id"]))
                if key in seen:
                    continue
                seen.add(key)
                row = rows.get(key)
                if row is None:
                    raise KeyError(f"{head} manifest points at missing row {key}")
                label = _label_of(head, row)
                weight = float(row.weight) if head == "urgency" and row.weight is not None else 1.0
                examples.append(
                    Example(
                        text=row.text,
                        label=label,
                        weight=weight,
                        file=row.file_name,
                        row_id=row.row_id,
                    )
                )
            splits[head][split] = examples
    return splits


def merged_split(splits: dict[str, dict[str, list[Example]]], split: str) -> list[MultiExample]:
    """Merge the three heads on normalised text inside one split.

    A text that is in the issue train split and the sentiment train split becomes
    one row with both labels. A text that is only in one head trains only that
    head. Labels are never taken from a different split, so a test label cannot
    attach to a training row.
    """
    by_text: dict[str, dict] = {}
    order: list[str] = []
    for head in HEADS:
        for example in splits[head][split]:
            key = normalise_message(example.text)
            if key not in by_text:
                by_text[key] = {
                    "text": example.text,
                    "issue": None,
                    "sentiment": None,
                    "urgency": None,
                    "weight": 1.0,
                }
                order.append(key)
            slot = by_text[key]
            current = slot[head]
            if current is None:
                slot[head] = example.label
            if head == "urgency":
                slot["weight"] = example.weight
    return [
        MultiExample(
            text=by_text[key]["text"],
            issue=by_text[key]["issue"],
            sentiment=by_text[key]["sentiment"],
            urgency=by_text[key]["urgency"],
            weight=float(by_text[key]["weight"]),
        )
        for key in order
    ]


def class_weight_vector(labels: list[str], classes: tuple[str, ...]) -> list[float]:
    """Inverse-frequency weights with mean 1. Unused classes get weight 0."""
    counts = {name: 0 for name in classes}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    present = [name for name in classes if counts[name] > 0]
    total = sum(counts.values())
    raw = []
    for name in classes:
        if counts[name] == 0:
            raw.append(0.0)
        else:
            raw.append(total / (len(present) * counts[name]))
    mean = sum(raw) / max(1, len(present))
    if mean == 0:
        return [1.0 for _ in classes]
    return [value / mean for value in raw]


def _label_of(head: str, row: LabelRow) -> str:
    value = getattr(row, head)
    if not value:
        raise KeyError(f"{row.file_name}:{row.row_id} has no {head} label")
    return str(value)
