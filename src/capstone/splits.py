"""Grouped, stratified splits and the gold-registry removal.

Duplicate groups are computed on the label texts with the same near-duplicate
function as the workbook pipeline. The label files do not carry the workbook
``dup_group_id``; joining them back to the export is a later step. Group ids
here are the smallest global row id in the component, so they are unique
across files.

Gold is the urgency test file plus the sentiment test file. Exact and
near-duplicate matches of that gold are removed from every train and
validation pool. Gemini is never a training source. It is loaded so the
overlap with urgency test can be counted.

Issue is a 70/15/15 split of ``issue.csv`` (seed 42), then gold rows are
dropped from train and val only. Sentiment train/val come from the hand file
after gold removal, in the 70:15 ratio (the held-out 15% is the separate
sentiment test file). Urgency test is the fixed gold file. Urgency validation
is carved from the remaining pure-gold groups inside the train file, targeting
60 rows (one third when fewer than 60 clean gold rows exist). Source and
weight stay on the urgency records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from capstone.duplicates import assign_dup_group_ids, normalise_message
from capstone.labels import FILES, LabelRow, load_label_dir

SEED = 42
URGENCY_VAL_ROWS = 60
ISSUE_FRACTIONS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
# The 15% test slice of 70/15/15 is the separate hand-annotated file.
SENTIMENT_FRACTIONS = (("train", 70 / 85), ("val", 15 / 85))


def build_manifests(
    label_dir: Path,
    *,
    seed: int = SEED,
    urgency_val_rows: int = URGENCY_VAL_ROWS,
    kind: str = "data",
) -> dict[str, dict]:
    """Return issue, sentiment, urgency, and registry manifests. No message text."""
    label_dir = Path(label_dir)
    rows = load_label_dir(label_dir)
    _attach_groups(rows)
    gold_groups = {
        row.dup_group_id for row in rows if row.table in {"urgency_test", "sentiment_test"}
    }
    file_hashes = {name: _sha256(label_dir / name) for name in FILES.values()}
    issue = _issue_manifest(rows, gold_groups, seed, file_hashes, kind)
    sentiment = _sentiment_manifest(rows, gold_groups, seed, file_hashes, kind)
    urgency = _urgency_manifest(rows, gold_groups, seed, urgency_val_rows, file_hashes, kind)
    registry = _registry(rows, gold_groups, file_hashes, kind)
    return {
        "issue": issue,
        "sentiment": sentiment,
        "urgency": urgency,
        "registry": registry,
    }


def write_manifests(manifests: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in manifests.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build grouped label splits and manifests.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/manifests"))
    parser.add_argument("--kind", choices=("harness", "data"), default="data")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--urgency-val-rows", type=int, default=URGENCY_VAL_ROWS)
    args = parser.parse_args(argv)
    manifests = build_manifests(
        args.labels,
        seed=args.seed,
        urgency_val_rows=args.urgency_val_rows,
        kind=args.kind,
    )
    write_manifests(manifests, args.out)
    if args.kind == "harness":
        print("HARNESS CHECK")
    else:
        print("LABEL SPLITS")
    for head in ("issue", "sentiment", "urgency"):
        counts = manifests[head]["counts"]
        print(
            f"{head} train={counts['train']} val={counts['val']} "
            f"test={counts['test']} removed_gold={counts['removed_gold']}"
        )
    overlaps = manifests["registry"]["overlaps"]
    print("urgency_test_exact_in_issue=" f"{overlaps['urgency_test_exact_in_issue']}")
    print("urgency_test_exact_in_gemini=" f"{overlaps['urgency_test_exact_in_gemini']}")
    print(f"manifests={args.out}")
    return 0


def _attach_groups(rows: list[LabelRow]) -> None:
    result = assign_dup_group_ids(
        [row.global_id for row in rows],
        [normalise_message(row.text) for row in rows],
    )
    for row, group_id in zip(rows, result.group_ids, strict=True):
        row.dup_group_id = int(group_id)


def _issue_manifest(
    rows: list[LabelRow],
    gold_groups: set[int],
    seed: int,
    file_hashes: dict[str, str],
    kind: str,
) -> dict:
    pool = [row for row in rows if row.table == "issue"]
    assignment = _assign(pool, lambda row: row.issue or "", ISSUE_FRACTIONS, seed)
    splits: dict[str, list[LabelRow]] = {"train": [], "val": [], "test": []}
    removed: list[LabelRow] = []
    for row in pool:
        split = assignment[row.dup_group_id]
        if split in {"train", "val"} and row.dup_group_id in gold_groups:
            removed.append(row)
        else:
            splits[split].append(row)
    return _manifest(
        head="issue",
        kind=kind,
        seed=seed,
        proportions={"train": 0.70, "val": 0.15, "test": 0.15},
        note=(
            "Grouped stratified 70/15/15 on issue.csv, seed 42. "
            "Gold matches are then removed from train and val."
        ),
        files={"issue.csv": file_hashes["issue.csv"]},
        splits=splits,
        removed=removed,
    )


def _sentiment_manifest(
    rows: list[LabelRow],
    gold_groups: set[int],
    seed: int,
    file_hashes: dict[str, str],
    kind: str,
) -> dict:
    hand = [row for row in rows if row.table == "sentiment_hand"]
    removed = [row for row in hand if row.dup_group_id in gold_groups]
    pool = [row for row in hand if row.dup_group_id not in gold_groups]
    assignment = _assign(pool, lambda row: row.sentiment or "", SENTIMENT_FRACTIONS, seed)
    splits = {
        "train": [row for row in pool if assignment[row.dup_group_id] == "train"],
        "val": [row for row in pool if assignment[row.dup_group_id] == "val"],
        "test": [row for row in rows if row.table == "sentiment_test"],
    }
    return _manifest(
        head="sentiment",
        kind=kind,
        seed=seed,
        proportions={"train": 70 / 85, "val": 15 / 85, "test": None},
        note=(
            "Train and val are the hand file after gold removal, split 70:15 "
            "(the non-test part of 70/15/15). Test is sentiment_test.csv, kept whole."
        ),
        files={
            "sentiment_hand.csv": file_hashes["sentiment_hand.csv"],
            "sentiment_test.csv": file_hashes["sentiment_test.csv"],
        },
        splits=splits,
        removed=removed,
    )


def _urgency_manifest(
    rows: list[LabelRow],
    gold_groups: set[int],
    seed: int,
    target_val_rows: int,
    file_hashes: dict[str, str],
    kind: str,
) -> dict:
    train_file = [row for row in rows if row.table == "urgency_train"]
    removed = [row for row in train_file if row.dup_group_id in gold_groups]
    remaining = [row for row in train_file if row.dup_group_id not in gold_groups]
    grouped: dict[int, list[LabelRow]] = defaultdict(list)
    for row in remaining:
        grouped[row.dup_group_id].append(row)
    pure_gold: list[LabelRow] = []
    train_rows: list[LabelRow] = []
    for members in grouped.values():
        if all(member.source == "gold" for member in members):
            pure_gold.extend(members)
        else:
            train_rows.extend(members)
    fraction = _urgency_val_fraction(len(pure_gold), target_val_rows)
    if pure_gold and fraction > 0:
        assignment = _assign(
            pure_gold,
            lambda row: row.urgency or "",
            (("val", fraction), ("train", 1.0 - fraction)),
            seed,
        )
        val_rows = [row for row in pure_gold if assignment[row.dup_group_id] == "val"]
        train_rows.extend(row for row in pure_gold if assignment[row.dup_group_id] == "train")
    else:
        val_rows = []
        train_rows.extend(pure_gold)
    splits = {
        "train": train_rows,
        "val": val_rows,
        "test": [row for row in rows if row.table == "urgency_test"],
    }
    return _manifest(
        head="urgency",
        kind=kind,
        seed=seed,
        proportions={"val_target_rows": target_val_rows, "val_fraction_of_clean_gold": fraction},
        note=(
            "Test is the fixed urgency gold file. Validation is carved from "
            "pure-gold groups that do not near-duplicate that test, targeting "
            f"{target_val_rows} rows (one third when fewer clean gold rows exist). "
            "Source and weight are kept. Mixed gold/distilled groups stay in train."
        ),
        files={
            "urgency_train.csv": file_hashes["urgency_train.csv"],
            "urgency_test.csv": file_hashes["urgency_test.csv"],
        },
        splits=splits,
        removed=removed,
        keep_training_fields=True,
    )


def _registry(
    rows: list[LabelRow],
    gold_groups: set[int],
    file_hashes: dict[str, str],
    kind: str,
) -> dict:
    by_table: dict[str, list[LabelRow]] = defaultdict(list)
    for row in rows:
        by_table[row.table].append(row)
    urgency_test = by_table["urgency_test"]
    sentiment_test = by_table["sentiment_test"]
    banner = (
        "HARNESS CHECK. Overlap counts are from the synthetic labels, not the real files."
        if kind == "harness"
        else "Overlap counts recomputed from the label files. Not a modelling result."
    )
    return {
        "banner": banner,
        "kind": kind,
        "gemini_excluded_from_training": True,
        "gold_registry": {
            "urgency_test_rows": len(urgency_test),
            "sentiment_test_rows": len(sentiment_test),
            "definition": "urgency_test.csv plus sentiment_test.csv",
        },
        "source_files": file_hashes,
        "overlaps": {
            "urgency_test_exact_in_issue": _exact(urgency_test, by_table["issue"]),
            "urgency_test_exact_in_sentiment_hand": _exact(
                urgency_test, by_table["sentiment_hand"]
            ),
            "urgency_test_exact_in_gemini": _exact(urgency_test, by_table["gemini"]),
            "urgency_test_exact_in_urgency_train": _exact(urgency_test, by_table["urgency_train"]),
            "sentiment_test_exact_in_issue": _exact(sentiment_test, by_table["issue"]),
            "sentiment_test_exact_in_sentiment_hand": _exact(
                sentiment_test, by_table["sentiment_hand"]
            ),
            "urgency_test_near_in_issue": _near(urgency_test, by_table["issue"]),
            "urgency_test_near_in_gemini": _near(urgency_test, by_table["gemini"]),
            "sentiment_test_near_in_issue": _near(sentiment_test, by_table["issue"]),
        },
        "gold_group_count": len(gold_groups),
        "schema_reported_exact_overlaps": {
            "urgency_test_in_issue_and_sentiment_1500": 9,
            "urgency_test_in_gemini_relabeled": 23,
            "urgency_train_vs_urgency_test": 0,
            "status": (
                "Cited from the label-file schema (18 Nov 2026). "
                "The overlaps object above is what this run recomputed. "
                "On a real directory those recomputed counts replace the citation."
            ),
        },
    }


def _manifest(
    *,
    head: str,
    kind: str,
    seed: int,
    proportions: dict,
    note: str,
    files: dict[str, str],
    splits: dict[str, list[LabelRow]],
    removed: list[LabelRow],
    keep_training_fields: bool = False,
) -> dict:
    encoded = {
        name: [_record(row, keep_training_fields) for row in _sorted(members)]
        for name, members in splits.items()
    }
    removed_encoded = [_record(row, keep_training_fields) for row in _sorted(removed)]
    _reject_shared_groups(encoded)
    return {
        "head": head,
        "kind": kind,
        "seed": seed,
        "proportions": proportions,
        "note": note,
        "source_files": files,
        "counts": {
            "train": len(encoded["train"]),
            "val": len(encoded["val"]),
            "test": len(encoded["test"]),
            "removed_gold": len(removed_encoded),
        },
        "splits": encoded,
        "removed_gold": removed_encoded,
    }


def _record(row: LabelRow, keep_training_fields: bool) -> dict:
    item = {
        "dup_group_id": row.dup_group_id,
        "file": row.file_name,
        "row_id": row.row_id,
        "text_sha256": hashlib.sha256(normalise_message(row.text).encode()).hexdigest(),
    }
    if keep_training_fields:
        item["source"] = row.source
        item["urgency"] = row.urgency
        item["weight"] = row.weight
    return item


def _sorted(rows: list[LabelRow]) -> list[LabelRow]:
    return sorted(rows, key=lambda row: (row.file_name, row.row_id, row.global_id))


def _reject_shared_groups(splits: dict[str, list[dict]]) -> None:
    seen: dict[int, str] = {}
    for name, records in splits.items():
        for record in records:
            previous = seen.get(record["dup_group_id"])
            if previous is not None and previous != name:
                raise RuntimeError(
                    f"dup_group_id {record['dup_group_id']} is in both {previous} and {name}"
                )
            seen[record["dup_group_id"]] = name


def _assign(
    rows: list[LabelRow],
    label_of,
    fractions: tuple[tuple[str, float], ...],
    seed: int,
) -> dict[int, str]:
    """Map each duplicate group to one split. Ties go to the earlier split name."""
    grouped: dict[int, list[LabelRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dup_group_id].append(row)
    by_stratum: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for group_id, members in grouped.items():
        counts = Counter(label_of(member) for member in members)
        top = max(counts.values())
        stratum = sorted(label for label, count in counts.items() if count == top)[0]
        by_stratum[stratum].append((group_id, len(members)))
    rng = random.Random(seed)
    order = [name for name, _ in fractions]
    targets_by_name = dict(fractions)
    assignment: dict[int, str] = {}
    for stratum in sorted(by_stratum):
        bucket = by_stratum[stratum]
        rng.shuffle(bucket)
        total = sum(size for _, size in bucket)
        targets = {name: total * targets_by_name[name] for name in order}
        counts = {name: 0 for name in order}
        for group_id, size in bucket:
            choice = _behind(order, targets, counts)
            assignment[group_id] = choice
            counts[choice] += size
    return assignment


def _behind(order: list[str], targets: dict[str, float], counts: dict[str, int]) -> str:
    best = order[0]
    best_deficit = targets[best] - counts[best]
    for name in order[1:]:
        deficit = targets[name] - counts[name]
        if deficit > best_deficit:
            best = name
            best_deficit = deficit
    return best


def _urgency_val_fraction(n_gold_rows: int, target: int) -> float:
    """Share of clean gold rows that should land in validation.

    At or above ``target`` rows, aim for that many. Below it, keep one third
    so a small file still has gold in both train and validation.
    """
    if n_gold_rows <= 0:
        return 0.0
    if n_gold_rows == 1:
        return 1.0
    if n_gold_rows < target:
        return 1.0 / 3.0
    return target / n_gold_rows


def _exact(gold_rows: list[LabelRow], other: list[LabelRow]) -> int:
    norms = {normalise_message(row.text) for row in other}
    return sum(1 for row in gold_rows if normalise_message(row.text) in norms)


def _near(gold_rows: list[LabelRow], other: list[LabelRow]) -> int:
    """Gold rows that share a group with the other file but not the exact text."""
    other_norms = {normalise_message(row.text) for row in other}
    other_groups = {row.dup_group_id for row in other}
    return sum(
        1
        for row in gold_rows
        if row.dup_group_id in other_groups and normalise_message(row.text) not in other_norms
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
