"""Fail when a test row shares an exact or near-duplicate with train or val.

The check reads manifests, not raw text. A shared ``dup_group_id`` is a
near-duplicate (exact copies are in the same group too). A shared
``text_sha256`` is an exact normalised match. Either one is leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADS = ("issue", "sentiment", "urgency")


class LeakError(Exception):
    """At least one test row collides with train or validation."""

    def __init__(self, leaks: list[dict]) -> None:
        self.leaks = leaks
        super().__init__(f"{len(leaks)} leakage collision(s)")


def find_leaks(manifest_dir: Path) -> list[dict]:
    """Return one record per colliding train/val row. Empty means clean."""
    leaks: list[dict] = []
    for head in HEADS:
        path = Path(manifest_dir) / f"{head}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        test = manifest["splits"]["test"]
        test_groups = {item["dup_group_id"] for item in test}
        test_hashes = {item["text_sha256"] for item in test}
        for split in ("train", "val"):
            for item in manifest["splits"][split]:
                via = []
                if item["dup_group_id"] in test_groups:
                    via.append("dup_group_id")
                if item["text_sha256"] in test_hashes:
                    via.append("text_sha256")
                if via:
                    leaks.append(
                        {
                            "head": head,
                            "split": split,
                            "file": item["file"],
                            "row_id": item["row_id"],
                            "dup_group_id": item["dup_group_id"],
                            "via": via,
                        }
                    )
    return leaks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check split manifests for leakage.")
    parser.add_argument("--manifests", type=Path, required=True)
    args = parser.parse_args(argv)
    leaks = find_leaks(args.manifests)
    if leaks:
        print(f"leakage_collisions={len(leaks)}")
        for leak in leaks[:20]:
            print(
                f"{leak['head']} {leak['split']} file={leak['file']} "
                f"row_id={leak['row_id']} via={','.join(leak['via'])}"
            )
        return 1
    print("leakage=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
