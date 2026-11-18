"""Recompute sentiment agreement between the hand labels and the platform file.

The hand file is the training source. The platform file is not used for
training. Both are already masked. Comparison is on the normalised masked
text, one label per distinct text (majority if a file disagrees with itself).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from capstone.duplicates import normalise_message
from capstone.labels import FILES, SchemaError, load_label_dir

HAND_TABLE = "sentiment_hand"
PLATFORM_TABLE = "sentiment_platform"


def agreement_and_kappa(left: list[str], right: list[str]) -> dict:
    """Raw agreement and Cohen's kappa for paired labels."""
    if len(left) != len(right):
        raise ValueError("label lists must be the same length")
    if not left:
        raise ValueError("no paired labels")
    n = len(left)
    names = sorted(set(left) | set(right))
    matrix = {(row, col): 0 for row in names for col in names}
    for gold, other in zip(left, right, strict=True):
        matrix[(gold, other)] += 1
    po = sum(matrix[(name, name)] for name in names) / n
    pe = 0.0
    for name in names:
        row_rate = sum(matrix[(name, col)] for col in names) / n
        col_rate = sum(matrix[(row, name)] for row in names) / n
        pe += row_rate * col_rate
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)
    table = {row: {col: matrix[(row, col)] for col in names} for row in names}
    return {
        "n": n,
        "agreement": po,
        "kappa": kappa,
        "labels": names,
        "confusion": table,
    }


def sentiment_agreement(label_dir: Path) -> dict:
    """Join the hand and platform files and return agreement, kappa, and counts."""
    rows = load_label_dir(label_dir)
    hand, hand_conflicts = _one_label(rows, HAND_TABLE, "sentiment")
    platform, platform_conflicts = _one_label(rows, PLATFORM_TABLE, "sentiment")
    shared = sorted(set(hand) & set(platform))
    if not shared:
        raise SchemaError("hand and platform sentiment files share no texts")
    stats = agreement_and_kappa(
        [hand[text] for text in shared],
        [platform[text] for text in shared],
    )
    stats["hand_only"] = len(set(hand) - set(platform))
    stats["platform_only"] = len(set(platform) - set(hand))
    stats["within_file_conflicts"] = hand_conflicts + platform_conflicts
    stats["hand_counts"] = dict(Counter(hand.values()))
    stats["platform_counts"] = dict(Counter(platform.values()))
    return stats


def write_agreement(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute hand vs platform sentiment agreement.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/agreement.json"))
    parser.add_argument("--kind", choices=("harness", "data"), default="data")
    args = parser.parse_args(argv)
    stats = sentiment_agreement(args.labels)
    stats["kind"] = args.kind
    stats["files"] = {
        "hand": FILES["sentiment_hand"],
        "platform": FILES["sentiment_platform"],
    }
    if args.kind == "harness":
        stats["banner"] = "HARNESS CHECK. Agreement on the synthetic labels. Not a result."
    else:
        stats["banner"] = "Recomputed from the label files. Not a modelling result."
    write_agreement(stats, args.out)
    if args.kind == "harness":
        print("HARNESS CHECK")
    print(f"n={stats['n']}")
    print(f"agreement={stats['agreement']:.6f}")
    print(f"kappa={stats['kappa']:.6f}")
    print(f"out={args.out}")
    return 0


def _one_label(rows: list, table: str, field: str) -> tuple[dict[str, str], int]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.table != table:
            continue
        grouped[normalise_message(row.text)].append(getattr(row, field))
    chosen: dict[str, str] = {}
    conflicts = 0
    for text, labels in grouped.items():
        if len(set(labels)) > 1:
            conflicts += 1
        chosen[text] = Counter(labels).most_common(1)[0][0]
    return chosen, conflicts


if __name__ == "__main__":
    raise SystemExit(main())
