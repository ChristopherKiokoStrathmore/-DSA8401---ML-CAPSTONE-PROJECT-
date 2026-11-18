"""Exact copies and near copies must share dup_group_id. Unrelated texts must not."""

from capstone.duplicates import assign_dup_group_ids, normalise_message


def test_normalise_collapses_case_and_space() -> None:
    assert normalise_message("  Call   [PHONE]\u00a0Now ") == "call [phone] now"


def test_exact_and_near_share_min_row_id() -> None:
    row_ids = [10, 3, 7, 8]
    texts = [
        "my line has been down since morning please help",
        "my line has been down since morning please help",
        "my line has been down since morning please help!",
        "the router has no 5g signal at all tonight",
    ]
    result = assign_dup_group_ids(row_ids, texts)
    assert result.group_ids[0] == result.group_ids[1] == result.group_ids[2] == 3
    assert result.group_ids[3] != 3
    assert result.unique_texts == 3


def test_one_character_typo_is_near() -> None:
    texts = [
        "bundle haijaingia tangu jana asubuhi",
        "bundle haijaingia tangu jana asubui",
    ]
    result = assign_dup_group_ids([1, 2], texts)
    assert result.group_ids == [1, 1]


def test_unrelated_texts_stay_apart() -> None:
    texts = [
        "please reverse the wrong payment of kes 200 today",
        "router ya 5g haionyeshi ishara kabisa",
    ]
    result = assign_dup_group_ids([4, 9], texts)
    assert result.group_ids == [4, 9]


def test_group_ids_are_deterministic() -> None:
    texts = ["hello there friend", "hello there friend!", "something else entirely here"]
    first = assign_dup_group_ids([1, 2, 3], texts)
    second = assign_dup_group_ids([1, 2, 3], texts)
    assert first == second


def test_empty_input() -> None:
    result = assign_dup_group_ids([], [])
    assert result.group_ids == []
    assert result.unique_texts == 0


def test_large_shared_prefix_is_capped_not_quadratic_crash() -> None:
    texts = [f"aaaaaaaaaaaa-{index:04d}-zzzz" for index in range(220)]
    result = assign_dup_group_ids(list(range(220)), texts)
    assert len(result.group_ids) == 220
    assert result.buckets_capped >= 1
