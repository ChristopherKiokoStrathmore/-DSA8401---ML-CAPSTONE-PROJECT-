"""Leakage check over manifests. The sample must be clean. A planted collision must fail."""

import json
from pathlib import Path

import pytest

from capstone.leakage import LeakError, find_leaks, main
from capstone.sample_labels import write_label_dir
from capstone.splits import build_manifests, write_manifests


def test_sample_manifests_have_no_leakage(tmp_path: Path) -> None:
    write_label_dir(tmp_path / "labels")
    manifests = build_manifests(tmp_path / "labels", kind="harness")
    write_manifests(manifests, tmp_path / "manifests")
    assert find_leaks(tmp_path / "manifests") == []
    assert main(["--manifests", str(tmp_path / "manifests")]) == 0


def test_checker_fails_when_a_test_group_is_in_train(tmp_path: Path) -> None:
    write_label_dir(tmp_path / "labels")
    manifests = build_manifests(tmp_path / "labels", kind="harness")
    stolen = manifests["issue"]["splits"]["test"][0]
    manifests["issue"]["splits"]["train"].append(dict(stolen))
    write_manifests(manifests, tmp_path / "manifests")
    leaks = find_leaks(tmp_path / "manifests")
    assert leaks
    assert leaks[0]["head"] == "issue"
    assert "dup_group_id" in leaks[0]["via"]
    assert main(["--manifests", str(tmp_path / "manifests")]) == 1
    with pytest.raises(LeakError):
        raise LeakError(leaks)


def test_checked_in_labels_match_generator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "sample" / "labels"
    write_label_dir(tmp_path)
    for name in (
        "issue.csv",
        "sentiment_hand.csv",
        "sentiment_platform.csv",
        "sentiment_test.csv",
        "urgency_train.csv",
        "urgency_test.csv",
        "gemini_relabeled.csv",
    ):
        expected = tmp_path / name
        checked_in = root / name
        assert checked_in.is_file(), f"missing {checked_in}"
        assert checked_in.read_bytes() == expected.read_bytes()


def test_manifest_round_trip_has_no_message_text(tmp_path: Path) -> None:
    write_label_dir(tmp_path / "labels")
    manifests = build_manifests(tmp_path / "labels", kind="harness")
    blob = json.dumps(manifests)
    assert "please help" not in blob
    assert "my line has been down" not in blob
