"""Load the Airtel workbook once, check the schema, mask PII, and audit it.

The modelling table is written as parquet. Raw message text is not kept.
``reports/data_audit.md`` contains aggregate counts only: no message text, no
tag values, no notes.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

from capstone.duplicates import DICE_THRESHOLD, assign_dup_group_ids, normalise_message
from capstone.pii import mask_pii

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Channel",
    "Type",
    "Posted on",
    "Text",
    "Last action",
    "Action date",
    "Sentiment",
    "Tags",
    "Notes",
)

# Observed range of the real export. Dates outside it are counted, not dropped.
STUDY_START = pd.Timestamp("2025-11-01")
STUDY_END_EXCLUSIVE = pd.Timestamp("2026-04-18")

OUTPUT_COLUMNS: tuple[str, ...] = (
    "row_id",
    "Channel",
    "Channel_norm",
    "Type",
    "Type_norm",
    "Posted on",
    "Action date",
    "Text",
    "text_norm",
    "char_len",
    "Last action",
    "Sentiment",
    "Tags",
    "Notes",
    "dup_group_id",
)

_UNNAMED = re.compile(r"^Unnamed: \d+$")
_PAREN = re.compile(r"\s*\([^)]*\)")
_SPACE = re.compile(r"\s+")

# Airtel inbox channels collapse to a short id. Parenthetical contact details
# are stripped before the lookup, so the phone-number variants match the page.
_CHANNEL_ALIASES = {
    "airtel kenya": "airtel_kenya",
    "@airtel_ke": "airtel_ke",
    "@airtel_kenya": "airtel_kenya_handle",
    "airtel kenya twitter search": "airtel_kenya_twitter_search",
}

_INCLUSION_RULES: tuple[str, ...] = (
    "empty",
    "non_linguistic",
    "automated_notice",
    "ticket_acknowledgement",
    "payment_or_bundle_receipt",
    "dear_customer_template",
    "airtel_money_receipt",
    "deleted_or_unavailable",
)

_AUTOMATED = re.compile(
    r"(?i)\b(?:this is an automated message|automated message|auto-?reply|"
    r"please do not reply|do not reply to this (?:message|number|sms))\b"
)
_ACK = re.compile(
    r"(?i)\byour (?:request|ticket|query|complaint) (?:has been|was) "
    r"(?:received|logged|registered|created)\b"
)
_PAYMENT = re.compile(
    r"(?i)(?:\b(?:transaction|payment|recharge|purchase|subscription) "
    r"(?:was |is )?(?:successful|completed)\b|"
    r"\byou have successfully (?:subscribed|purchased|recharged|received)\b)"
)
_DEAR = re.compile(r"(?i)^dear (?:customer|subscriber)\b")
_DEAR_BODY = re.compile(r"(?i)\b(?:successfully|reference number|has been processed)\b")
_AIRTEL_MONEY = re.compile(
    r"(?i)\bairtel money\b.{0,80}\b(?:you have received|new balance|sent successfully)\b",
    re.DOTALL,
)
_DELETED = re.compile(
    r"(?i)\b(?:this message was deleted|message deleted|this message is unavailable|"
    r"unsupported message type)\b"
)


class SchemaError(ValueError):
    """The workbook is missing a column the pipeline requires."""


def normalise_channel(value: object) -> str | None:
    """Map channel variants onto one id. Originals are stored separately."""
    if _missing(value):
        return None
    text = str(value).strip().lstrip("'\"").strip()
    text = _PAREN.sub("", text)
    text = _SPACE.sub(" ", text).strip().lower()
    if text in _CHANNEL_ALIASES:
        return _CHANNEL_ALIASES[text]
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or None


def normalise_type(value: object) -> str | None:
    """Casefold message types so 'Message' and 'message' match."""
    if _missing(value):
        return None
    return _SPACE.sub(" ", str(value).strip().lower())


def exclusion_reason(text: str | None) -> str | None:
    """Return the inclusion-rule name that drops ``text``, or None to keep it.

    Rules are applied to the original text, before PII masking, and the first
    match wins. A phone number alone is non-linguistic. A sentence that merely
    mentions Airtel Money is kept.
    """
    if text is None or text.strip() == "":
        return "empty"
    stripped = text.strip()
    if not any(char.isalpha() for char in stripped):
        return "non_linguistic"
    if _AUTOMATED.search(stripped):
        return "automated_notice"
    if _ACK.search(stripped):
        return "ticket_acknowledgement"
    if _PAYMENT.search(stripped):
        return "payment_or_bundle_receipt"
    if _DEAR.search(stripped) and _DEAR_BODY.search(stripped):
        return "dear_customer_template"
    if _AIRTEL_MONEY.search(stripped):
        return "airtel_money_receipt"
    if _DELETED.search(stripped):
        return "deleted_or_unavailable"
    return None


def load_table(path: Path) -> pd.DataFrame:
    """Read an xlsx workbook or a csv file once."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise SchemaError(f"unsupported file type {suffix!r}; expected .xlsx or .csv")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def prepare_frame(
    frame: pd.DataFrame,
    *,
    threshold: float = DICE_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Validate, mask, filter, and assign duplicate groups.

    The returned frame contains kept rows only. The dict is the aggregate
    audit (no message text).
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError("schema check failed; missing columns: " + ", ".join(missing))

    working = frame.copy()
    working.insert(0, "row_id", list(range(len(working))))

    overflow = [column for column in working.columns if _UNNAMED.match(str(column))]
    known = set(REQUIRED_COLUMNS) | {"row_id"} | set(overflow)
    extra = [column for column in working.columns if column not in known]

    overflow_counts = {column: int(working[column].notna().sum()) for column in overflow}
    extra_counts = {column: int(working[column].notna().sum()) for column in extra}

    missingness = {
        column: _missingness(working[column])
        for column in list(REQUIRED_COLUMNS) + overflow + extra
    }
    posted_stats = _date_stats(working["Posted on"])
    action_stats = _date_stats(working["Action date"])
    sentiment_stats = _sentiment_stats(working["Sentiment"])

    originals = working["Text"].map(_as_text)
    lengths = originals.map(lambda text: None if text is None else len(text.strip()))
    reasons = originals.map(exclusion_reason)
    inclusion_counts = {rule: int((reasons == rule).sum()) for rule in _INCLUSION_RULES}
    inclusion_counts["kept"] = int(reasons.isna().sum())

    length_all = _length_stats([int(length) for length in lengths.tolist() if pd.notna(length)])

    working["Channel_norm"] = working["Channel"].map(normalise_channel)
    working["Type_norm"] = working["Type"].map(normalise_type)
    working["Posted on"] = pd.to_datetime(working["Posted on"], errors="coerce")
    working["Action date"] = pd.to_datetime(working["Action date"], errors="coerce")
    working["Sentiment"] = pd.to_numeric(working["Sentiment"], errors="coerce")
    working["char_len"] = lengths
    working["Text"] = originals.map(_mask_optional)
    working["Tags"] = working["Tags"].map(_mask_optional)
    working["Notes"] = working["Notes"].map(_mask_optional)
    working["text_norm"] = working["Text"].map(
        lambda text: "" if text is None else normalise_message(text)
    )

    kept = working.loc[reasons.isna()].copy()
    dup = assign_dup_group_ids(
        [int(row_id) for row_id in kept["row_id"].tolist()],
        [str(text) for text in kept["text_norm"].tolist()],
        threshold=threshold,
    )
    kept["dup_group_id"] = dup.group_ids

    repeat_stats = _repeat_stats(kept["text_norm"].tolist(), dup.group_ids)
    length_kept = _length_stats(
        [int(length) for length in kept["char_len"].tolist() if pd.notna(length)]
    )

    audit = {
        "rows_in": int(len(working)),
        "rows_out": int(len(kept)),
        "missingness": missingness,
        "overflow": overflow_counts,
        "extra_dropped": extra_counts,
        "dates": {"Posted on": posted_stats, "Action date": action_stats},
        "sentiment": sentiment_stats,
        "channels": _category_counts(working["Channel"]),
        "channels_norm": _category_counts(working["Channel_norm"]),
        "types": _category_counts(working["Type"]),
        "types_norm": _category_counts(working["Type_norm"]),
        "last_action": _category_counts(working["Last action"]),
        "inclusion": inclusion_counts,
        "lengths_all_nonempty": length_all,
        "lengths_kept": length_kept,
        "duplicates": repeat_stats,
        "near_dup_buckets_capped": dup.buckets_capped,
        "unique_texts_kept": dup.unique_texts,
    }
    modelled = kept.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)
    return modelled, audit


def render_audit(audit: dict, *, kind: str) -> str:
    """Render the audit markdown. ``kind`` is ``harness`` or ``data``."""
    if kind == "harness":
        banner = (
            "HARNESS CHECK. This audit describes the synthetic sample (or another "
            "smoke input). The numbers are not modelling results and they are not "
            "the real workbook."
        )
    else:
        banner = (
            "DATA AUDIT. Aggregate statistics only. This is not a modelling result. "
            "No message text is included."
        )
    lines = [
        "# Data audit",
        "",
        f"> {banner}",
        "",
        "## Size",
        "",
        f"- rows in: {audit['rows_in']}",
        f"- rows kept: {audit['rows_out']}",
        "",
        "## Missingness",
        "",
        "| column | missing | missing % |",
        "| --- | ---: | ---: |",
    ]
    for column, stats in audit["missingness"].items():
        lines.append(f"| {column} | {stats['missing']} | {stats['missing_pct']:.2f} |")
    lines.extend(["", "## Overflow and unexpected columns", ""])
    lines.append("Unnamed overflow columns are counted here and then dropped.")
    lines.append("Any other unexpected column is counted and dropped as well.")
    lines.extend(["", "| column | non-null | treatment |", "| --- | ---: | --- |"])
    if not audit["overflow"] and not audit["extra_dropped"]:
        lines.append("| (none) | 0 |  |")
    for column, count in audit["overflow"].items():
        lines.append(f"| {column} | {count} | dropped overflow |")
    for column, count in audit["extra_dropped"].items():
        lines.append(f"| {column} | {count} | dropped unexpected |")

    lines.extend(["", "## Dates", ""])
    lines.append(
        "The study window is 2025-11-01 through 2026-04-17 inclusive. "
        "Out-of-window dates are counted and kept."
    )
    lines.extend(
        [
            "",
            "| column | missing | unparseable | before window | after window |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for column, stats in audit["dates"].items():
        lines.append(
            f"| {column} | {stats['missing']} | {stats['unparseable']} | "
            f"{stats['before_window']} | {stats['after_window']} |"
        )

    lines.extend(["", "## Sentiment column (platform score, not our label)", ""])
    sentiment = audit["sentiment"]
    lines.append(f"- non-numeric values: {sentiment['invalid']}")
    lines.append(f"- numeric values outside -1, 0, 1: {sentiment['out_of_domain']}")
    lines.append("- value counts:")
    for label, count in sentiment["counts"]:
        lines.append(f"  - {label}: {count}")

    lines.extend(["", "## Channel and type", ""])
    lines.append("Originals are kept beside the normalised values.")
    for title, key in (
        ("Channel", "channels"),
        ("Channel normalised", "channels_norm"),
        ("Type", "types"),
        ("Type normalised", "types_norm"),
        ("Last action", "last_action"),
    ):
        lines.extend(["", f"### {title}", ""])
        for label, count in audit[key]:
            lines.append(f"- {label}: {count}")

    lines.extend(["", "## Inclusion", ""])
    lines.append("The first matching rule drops the row. Counts partition the input.")
    for rule in _INCLUSION_RULES:
        lines.append(f"- {rule}: {audit['inclusion'][rule]}")
    lines.append(f"- kept: {audit['inclusion']['kept']}")

    duplicates = audit["duplicates"]
    lines.extend(
        [
            "",
            "## Duplicates",
            "",
            "Exact duplicates share a normalised masked string. Near-duplicates "
            "share a `dup_group_id` when character 3-gram Dice is at least 0.80 "
            "after exact texts are collapsed. Group id is the smallest source row id.",
            "",
            f"- unique normalised texts kept: {audit['unique_texts_kept']}",
            f"- texts that occur more than once: {duplicates['repeated_texts']}",
            f"- rows belonging to a repeated text: {duplicates['rows_in_repeated_texts']}",
            f"- extra copies beyond the first: {duplicates['extra_copies']}",
            f"- near-duplicate groups (more than one distinct text): "
            f"{duplicates['near_groups']}",
            f"- rows in a multi-row duplicate group: {duplicates['rows_in_multi_groups']}",
            f"- candidate buckets that hit the size cap: {audit['near_dup_buckets_capped']}",
            "",
            "Repeat-size histogram (group size: number of distinct texts). "
            "The text itself is not shown.",
            "",
        ]
    )
    if duplicates["repeat_histogram"]:
        for size, count in duplicates["repeat_histogram"]:
            lines.append(f"- size {size}: {count}")
    else:
        lines.append("- (no repeated text)")

    lines.extend(["", "## Length", ""])
    lines.append("Character length of the stripped original text, before masking.")
    lines.extend(["", "### All non-empty texts", ""])
    lines.extend(_length_lines(audit["lengths_all_nonempty"]))
    lines.extend(["", "### Kept texts", ""])
    lines.extend(_length_lines(audit["lengths_kept"]))
    lines.append("")
    return "\n".join(lines)


def run(
    source: Path,
    out_dir: Path,
    audit_path: Path,
    *,
    kind: str = "data",
    threshold: float = DICE_THRESHOLD,
) -> dict:
    """Load ``source``, write parquet and the audit, and return the audit dict."""
    frame = load_table(source)
    modelled, audit = prepare_frame(frame, threshold=threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "messages.parquet"
    modelled.to_parquet(parquet_path, index=False)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(render_audit(audit, kind=kind), encoding="utf-8")
    audit["parquet_path"] = str(parquet_path)
    audit["audit_path"] = str(audit_path)
    audit["kind"] = kind
    return audit


def print_summary(audit: dict) -> None:
    """Print aggregate counts only."""
    kind = audit.get("kind", "data")
    if kind == "harness":
        print("HARNESS CHECK")
    else:
        print("DATA AUDIT")
    print(f"rows_in={audit['rows_in']}")
    print(f"rows_out={audit['rows_out']}")
    for rule in _INCLUSION_RULES:
        print(f"inclusion_{rule}={audit['inclusion'][rule]}")
    print(f"inclusion_kept={audit['inclusion']['kept']}")
    duplicates = audit["duplicates"]
    print(f"repeated_texts={duplicates['repeated_texts']}")
    print(f"extra_copies={duplicates['extra_copies']}")
    print(f"near_groups={duplicates['near_groups']}")
    print(f"parquet={audit['parquet_path']}")
    print(f"audit={audit['audit_path']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load an Airtel workbook, mask PII, and write parquet plus an audit."
    )
    parser.add_argument("input", type=Path, help="xlsx or csv workbook")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/private"),
        help="directory for messages.parquet (default: data/private)",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("reports/data_audit.md"),
        help="markdown audit path",
    )
    parser.add_argument(
        "--audit-kind",
        choices=("harness", "data"),
        default="data",
        help="harness labels smoke numbers; data is the real-workbook wording",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DICE_THRESHOLD,
        help="Dice threshold for near-duplicate groups",
    )
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1
    try:
        audit = run(
            args.input,
            args.out,
            args.audit,
            kind=args.audit_kind,
            threshold=args.threshold,
        )
    except SchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(audit)
    return 0


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _as_text(value: object) -> str | None:
    if _missing(value):
        return None
    return str(value)


def _mask_optional(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return mask_pii(str(value))


def _missingness(series: pd.Series) -> dict:
    missing = int(series.isna().sum())
    total = int(len(series))
    pct = 0.0 if total == 0 else (100.0 * missing / total)
    return {"missing": missing, "missing_pct": pct}


def _date_stats(series: pd.Series) -> dict:
    parsed = pd.to_datetime(series, errors="coerce")
    missing = int(series.isna().sum())
    unparseable = int((series.notna() & parsed.isna()).sum())
    before = int((parsed.notna() & (parsed < STUDY_START)).sum())
    after = int((parsed.notna() & (parsed >= STUDY_END_EXCLUSIVE)).sum())
    return {
        "missing": missing,
        "unparseable": unparseable,
        "before_window": before,
        "after_window": after,
    }


def _sentiment_stats(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = int((series.notna() & numeric.isna()).sum())
    out_of_domain = int((numeric.notna() & ~numeric.isin([-1.0, 0.0, 1.0])).sum())
    labels = []
    for value, count in numeric.value_counts(dropna=True).sort_index().items():
        labels.append((str(float(value)), int(count)))
    nulls = int(numeric.isna().sum())
    labels.append(("null", nulls))
    return {"invalid": invalid, "out_of_domain": out_of_domain, "counts": labels}


def _category_counts(series: pd.Series) -> list[tuple[str, int]]:
    counts = series.fillna("(null)").astype(str).value_counts()
    items = [(str(label), int(count)) for label, count in counts.items()]
    items.sort(key=lambda item: (-item[1], item[0]))
    return items


def _length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(lengths)

    def quantile(q: float) -> int:
        pos = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return int(ordered[pos])

    return {
        "n": len(ordered),
        "min": int(ordered[0]),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": int(ordered[-1]),
        "mean": sum(ordered) / len(ordered),
    }


def _length_lines(stats: dict) -> list[str]:
    if stats["n"] == 0:
        return ["- (no texts)"]
    return [
        f"- n: {stats['n']}",
        f"- min: {stats['min']}",
        f"- p25: {stats['p25']}",
        f"- p50: {stats['p50']}",
        f"- p75: {stats['p75']}",
        f"- p95: {stats['p95']}",
        f"- max: {stats['max']}",
        f"- mean: {stats['mean']:.1f}",
    ]


def _repeat_stats(texts: list[str], group_ids: list[int]) -> dict:
    sizes: dict[str, int] = {}
    for text in texts:
        sizes[text] = sizes.get(text, 0) + 1
    repeated = {text: size for text, size in sizes.items() if size > 1}
    histogram: dict[int, int] = {}
    for size in repeated.values():
        histogram[size] = histogram.get(size, 0) + 1
    histogram_items = sorted(histogram.items())

    rows_of_group: dict[int, int] = {}
    texts_of_group: dict[int, set[str]] = {}
    for text, group_id in zip(texts, group_ids, strict=True):
        rows_of_group[group_id] = rows_of_group.get(group_id, 0) + 1
        texts_of_group.setdefault(group_id, set()).add(text)
    near_groups = sum(1 for group_texts in texts_of_group.values() if len(group_texts) > 1)
    rows_in_multi = sum(size for size in rows_of_group.values() if size > 1)
    extra = sum(size - 1 for size in repeated.values())
    return {
        "repeated_texts": len(repeated),
        "rows_in_repeated_texts": sum(repeated.values()),
        "extra_copies": extra,
        "near_groups": near_groups,
        "rows_in_multi_groups": rows_in_multi,
        "repeat_histogram": histogram_items,
    }


if __name__ == "__main__":
    raise SystemExit(main())
