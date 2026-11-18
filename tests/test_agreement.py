"""Cohen's kappa on a known table, and the hand-vs-platform join."""

from capstone.agreement import agreement_and_kappa, sentiment_agreement
from capstone.sample_labels import write_label_dir


def test_kappa_matches_a_hand_computed_table() -> None:
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no.
    left = ["yes"] * 25 + ["no"] * 25
    right = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    stats = agreement_and_kappa(left, right)
    assert stats["n"] == 50
    assert stats["agreement"] == 0.70
    assert abs(stats["kappa"] - 0.40) < 1e-9


def test_sample_agreement_is_between_the_two_files(tmp_path) -> None:
    write_label_dir(tmp_path)
    stats = sentiment_agreement(tmp_path)
    assert stats["n"] > 100
    assert 0.0 < stats["agreement"] < 1.0
    assert stats["kappa"] < stats["agreement"]
    assert set(stats["hand_counts"]) == {"negative", "not_negative"}
    assert set(stats["platform_counts"]) == {"negative", "not_negative"}
