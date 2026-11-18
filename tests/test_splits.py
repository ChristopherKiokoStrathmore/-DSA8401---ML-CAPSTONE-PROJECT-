"""Grouped splits, gold removal, and the urgency validation carve."""

import hashlib

import pytest

from capstone.labels import SchemaError, load_label_dir
from capstone.sample_labels import (
    GEMINI_OVERLAP,
    ISSUE_OVERLAP,
    NEAR_KEEP_A,
    NEAR_KEEP_B,
    PLANTED_EXACT,
    STABLE_DUP,
    URGENCY_NEAR_TRAIN,
    write_label_dir,
)
from capstone.splits import build_manifests


def test_loader_rejects_unknown_issue(tmp_path) -> None:
    write_label_dir(tmp_path)
    issue = tmp_path / "issue.csv"
    issue.write_text("text,label\nhello there,Not_A_Class\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="unknown issue"):
        load_label_dir(tmp_path)


def test_sample_splits_keep_groups_and_drop_gold(tmp_path) -> None:
    write_label_dir(tmp_path)
    manifests = build_manifests(tmp_path, kind="harness")
    issue = manifests["issue"]
    sentiment = manifests["sentiment"]
    urgency = manifests["urgency"]
    registry = manifests["registry"]

    loaded = [row for row in load_label_dir(tmp_path) if row.table == "issue"]
    by_text: dict[str, list[int]] = {}
    for row in loaded:
        by_text.setdefault(row.text, []).append(row.row_id)

    def placed(row_id: int) -> str | None:
        for name in ("train", "val", "test"):
            if any(
                item["row_id"] == row_id and item["file"] == "issue.csv"
                for item in issue["splits"][name]
            ):
                return name
        if any(item["row_id"] == row_id for item in issue["removed_gold"]):
            return "removed"
        return None

    stable = by_text[STABLE_DUP]
    assert len(stable) == 2
    assert placed(stable[0]) == placed(stable[1])
    assert placed(stable[0]) in {"train", "val", "test"}

    keep = by_text[NEAR_KEEP_A] + by_text[NEAR_KEEP_B]
    assert len({placed(row_id) for row_id in keep}) == 1
    assert placed(keep[0]) in {"train", "val", "test"}

    for row_id in by_text[PLANTED_EXACT] + by_text[ISSUE_OVERLAP]:
        assert placed(row_id) in {"test", "removed"}

    counts = issue["counts"]
    total = counts["train"] + counts["val"] + counts["test"]
    assert counts["train"] / total == pytest.approx(0.70, abs=0.12)
    assert counts["val"] / total == pytest.approx(0.15, abs=0.12)
    assert counts["test"] / total == pytest.approx(0.15, abs=0.12)
    assert counts["removed_gold"] > 0

    # Every issue class reaches train. Groups are per message, not per class.
    train_ids = {item["row_id"] for item in issue["splits"]["train"]}
    train_labels = {row.issue for row in loaded if row.row_id in train_ids}
    assert train_labels == set(row.issue for row in loaded)

    # Sentiment test is the whole file. Hand gold is not in train or val.
    test_ids = {item["row_id"] for item in sentiment["splits"]["test"]}
    assert 5099 in test_ids and 5100 in test_ids
    hand_removed = {item["row_id"] for item in sentiment["removed_gold"]}
    hand_train = {item["row_id"] for item in sentiment["splits"]["train"]}
    assert by_text[PLANTED_EXACT][0] in hand_removed
    assert not (hand_removed & hand_train)

    # Urgency test is the fixed file, including the planted overlaps.
    urgency_test_text = {
        row.text for row in load_label_dir(tmp_path) if row.table == "urgency_test"
    }
    assert ISSUE_OVERLAP in urgency_test_text
    assert GEMINI_OVERLAP in urgency_test_text
    assert urgency["counts"]["test"] == len(urgency_test_text)
    val_sources = {item["source"] for item in urgency["splits"]["val"]}
    assert val_sources == {"gold"}
    assert urgency["counts"]["val"] > 0
    train_ids_u = {item["row_id"] for item in urgency["splits"]["train"]}
    val_ids_u = {item["row_id"] for item in urgency["splits"]["val"]}
    near_train = next(
        row.row_id
        for row in load_label_dir(tmp_path)
        if row.table == "urgency_train" and row.text == URGENCY_NEAR_TRAIN
    )
    assert near_train not in train_ids_u
    assert near_train not in val_ids_u
    assert any(
        item["weight"] == 1.0 and item["source"] == "gold" for item in urgency["splits"]["train"]
    )
    assert any(
        item["weight"] == 0.5 and item["source"] == "distilled"
        for item in urgency["splits"]["train"]
    )

    assert registry["gemini_excluded_from_training"] is True
    assert registry["overlaps"]["urgency_test_exact_in_issue"] >= 1
    assert registry["overlaps"]["urgency_test_exact_in_gemini"] >= 1
    assert registry["overlaps"]["urgency_test_exact_in_urgency_train"] == 0
    assert registry["kind"] == "harness"
    assert "HARNESS CHECK" in registry["banner"]

    # Gemini rows are not listed in any head split.
    for manifest in (issue, sentiment, urgency):
        for split in manifest["splits"].values():
            assert all(item["file"] != "gemini_relabeled.csv" for item in split)

    digest = hashlib.sha256((tmp_path / "issue.csv").read_bytes()).hexdigest()
    assert issue["source_files"]["issue.csv"] == digest
