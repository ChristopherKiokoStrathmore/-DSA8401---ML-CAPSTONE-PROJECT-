"""Load the seven label files into one canonical row shape.

The loader only accepts this directory layout. Real exports use longer names;
rename them before pointing the loader at the directory. A byte-identical
second copy of the issue file must be dropped rather than passed in.

    issue.csv                text, label
    sentiment_hand.csv       text, sentiment          (hand labels, primary)
    sentiment_platform.csv   text, sentiment          (platform-derived, agreement only)
    sentiment_test.csv       row_id, message, your_sentiment
    urgency_train.csv        base_idx, text, urg, source, weight
    urgency_test.csv         base_idx, text, urg, source, weight
    gemini_relabeled.csv     text, issue, sentiment, urgency

Texts are already masked. This module does not mask them again.

Manifest ``row_id`` is the file's own id: the ``row_id`` column, else
``base_idx``, else the 0-based data-row index. Issue and the two 1,500-row
sentiment files have no id column, so they use that index.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

ISSUE_CLASSES: tuple[str, ...] = (
    "Airtel_Money_Reversal",
    "Airtel_Money_Transfer",
    "App_Rewards",
    "Complaint_General",
    "Data_Bundle_Problems",
    "Network_Issues",
    "Non_Actionable",
    "Product_Enquiry",
    "Router_WiFi_5G",
    "SIM_Line_Services",
)
SENTIMENT_CLASSES: tuple[str, ...] = ("negative", "not_negative")
URGENCY_CLASSES: tuple[str, ...] = ("low", "medium", "emergency")

# Real filenames from the label-file schema, and the names this loader reads.
FILE_MAP: dict[str, str] = {
    "issue_retagged_clean_1500.csv": "issue.csv",
    "sentiment_judgment_1500.csv": "sentiment_hand.csv",
    "sentiment_binary_1500.csv": "sentiment_platform.csv",
    "sentiment_test_150.csv": "sentiment_test.csv",
    "urgency_train.csv": "urgency_train.csv",
    "urgency_test.csv": "urgency_test.csv",
    "gemini_relabeled.csv": "gemini_relabeled.csv",
}

FILES: dict[str, str] = {
    "issue": "issue.csv",
    "sentiment_hand": "sentiment_hand.csv",
    "sentiment_platform": "sentiment_platform.csv",
    "sentiment_test": "sentiment_test.csv",
    "urgency_train": "urgency_train.csv",
    "urgency_test": "urgency_test.csv",
    "gemini": "gemini_relabeled.csv",
}

_SOURCE_ALIASES = {
    "gold": "gold",
    "distilled": "distilled",
    "silver": "distilled",
    "llm silver": "distilled",
}


class SchemaError(ValueError):
    """A label file is missing or a value is outside the known label set."""


@dataclass
class LabelRow:
    table: str
    file_name: str
    row_id: int
    global_id: int
    text: str
    issue: str | None = None
    sentiment: str | None = None
    urgency: str | None = None
    source: str | None = None
    weight: float | None = None
    dup_group_id: int = -1


def load_label_dir(directory: Path) -> list[LabelRow]:
    """Read every fixed filename. Missing files and unknown labels raise."""
    directory = Path(directory)
    missing = [name for name in FILES.values() if not (directory / name).is_file()]
    if missing:
        raise SchemaError("missing label files: " + ", ".join(missing))

    rows: list[LabelRow] = []
    rows.extend(_load_issue(directory / FILES["issue"]))
    rows.extend(_load_sentiment(directory / FILES["sentiment_hand"], "sentiment_hand", "sentiment"))
    rows.extend(
        _load_sentiment(directory / FILES["sentiment_platform"], "sentiment_platform", "sentiment")
    )
    rows.extend(
        _load_sentiment(directory / FILES["sentiment_test"], "sentiment_test", "your_sentiment")
    )
    rows.extend(_load_urgency(directory / FILES["urgency_train"], "urgency_train"))
    rows.extend(_load_urgency(directory / FILES["urgency_test"], "urgency_test"))
    rows.extend(_load_gemini(directory / FILES["gemini"]))
    for global_id, row in enumerate(rows):
        row.global_id = global_id
    return rows


def canonical_issue(value: str) -> str:
    key = re.sub(r"[\s\-]+", "_", value.strip()).lower()
    for name in ISSUE_CLASSES:
        if name.lower() == key:
            return name
    raise SchemaError(f"unknown issue label {value!r}")


def canonical_sentiment(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in SENTIMENT_CLASSES:
        return key
    raise SchemaError(f"unknown sentiment label {value!r}")


def canonical_urgency(value: str) -> str:
    word = value.strip().lower()
    if word in URGENCY_CLASSES:
        return word
    try:
        number = float(word)
    except ValueError as exc:
        raise SchemaError(f"unknown urgency label {value!r}") from exc
    if abs(number - 0.0) < 1e-9:
        return "low"
    if abs(number - 0.5) < 1e-9:
        return "medium"
    if abs(number - 1.0) < 1e-9:
        return "emergency"
    raise SchemaError(f"urgency {value!r} is outside 0, 0.5, 1")


def canonical_source(value: str) -> str:
    key = value.strip().lower()
    if key in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[key]
    raise SchemaError(f"unknown urgency source {value!r}")


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SchemaError(f"{path.name} has no header")
        rows: list[dict[str, str]] = []
        for raw in reader:
            item: dict[str, str] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                item[key.strip().lstrip("\ufeff").lower()] = (value or "").strip()
            rows.append(item)
        return rows


def _require(row: dict[str, str], column: str, file_name: str, index: int) -> str:
    if column not in row or row[column] == "":
        raise SchemaError(f"{file_name} row {index} is missing {column}")
    return row[column]


def _load_issue(path: Path) -> list[LabelRow]:
    loaded = []
    for index, row in enumerate(read_table(path)):
        text = _require(row, "text", path.name, index)
        loaded.append(
            LabelRow(
                table="issue",
                file_name=path.name,
                row_id=index,
                global_id=-1,
                text=text,
                issue=canonical_issue(_require(row, "label", path.name, index)),
            )
        )
    return loaded


def _load_sentiment(path: Path, table: str, label_column: str) -> list[LabelRow]:
    text_column = "message" if table == "sentiment_test" else "text"
    loaded = []
    for index, row in enumerate(read_table(path)):
        text = _require(row, text_column, path.name, index)
        if "row_id" in row and row["row_id"] != "":
            row_id = int(float(row["row_id"]))
        else:
            row_id = index
        loaded.append(
            LabelRow(
                table=table,
                file_name=path.name,
                row_id=row_id,
                global_id=-1,
                text=text,
                sentiment=canonical_sentiment(_require(row, label_column, path.name, index)),
            )
        )
    return loaded


def _load_urgency(path: Path, table: str) -> list[LabelRow]:
    loaded = []
    for index, row in enumerate(read_table(path)):
        text = _require(row, "text", path.name, index)
        if row.get("base_idx"):
            row_id = int(float(row["base_idx"]))
        else:
            row_id = index
        weight = float(_require(row, "weight", path.name, index))
        loaded.append(
            LabelRow(
                table=table,
                file_name=path.name,
                row_id=row_id,
                global_id=-1,
                text=text,
                urgency=canonical_urgency(_require(row, "urg", path.name, index)),
                source=canonical_source(_require(row, "source", path.name, index)),
                weight=weight,
            )
        )
    return loaded


def _load_gemini(path: Path) -> list[LabelRow]:
    loaded = []
    for index, row in enumerate(read_table(path)):
        text = _require(row, "text", path.name, index)
        loaded.append(
            LabelRow(
                table="gemini",
                file_name=path.name,
                row_id=index,
                global_id=-1,
                text=text,
                issue=canonical_issue(_require(row, "issue", path.name, index)),
                sentiment=canonical_sentiment(_require(row, "sentiment", path.name, index)),
                urgency=canonical_urgency(_require(row, "urgency", path.name, index)),
            )
        )
    return loaded
