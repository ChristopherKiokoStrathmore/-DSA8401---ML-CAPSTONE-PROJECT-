"""Exact and near-duplicate group ids for short messages.

Exact duplicates are collapsed first: the same normalised text is one node.
Near-duplicates are found only among those unique strings.

Candidate pairs come from two cheap indexes that scale to a few hundred
thousand short rows on one CPU:

* MinHash LSH over character 3-grams (32 permutations, 8 bands of 4, seed 42).
* Shared prefixes of 6 and of 12 characters.

A candidate is kept when the Dice coefficient of its character 3-gram sets is
at least ``DICE_THRESHOLD`` (0.80). Connected components are unioned. Every
row in a component receives ``dup_group_id`` equal to the smallest source
row id in that component, so exact copies and near-copies share one id.

Buckets larger than ``MAX_BUCKET`` are split by length and then compared only
inside a small length-neighbour window. That cap is recorded in
``DupResult.buckets_capped``. It is the safeguard that keeps a huge repeated
prefix from turning into an all-pairs comparison.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

NGRAM = 3
NUM_PERM = 32
BANDS = 8
ROWS_PER_BAND = 4
DICE_THRESHOLD = 0.80
PREFIX_LENGTHS = (6, 12)
MAX_BUCKET = 200
NEIGHBOUR_WINDOW = 40
PERM_SEED = 42
PRIME = 2_147_483_647  # 2^31 - 1

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DupResult:
    """Group id per input row, in the same order as the inputs."""

    group_ids: list[int]
    buckets_capped: int
    unique_texts: int


def normalise_message(text: str) -> str:
    """Casefold and collapse whitespace. Placeholders stay in the string."""
    cleaned = text.replace("\u00a0", " ").lower()
    return _WHITESPACE.sub(" ", cleaned).strip()


def assign_dup_group_ids(
    row_ids: list[int],
    texts: list[str],
    *,
    threshold: float = DICE_THRESHOLD,
    seed: int = PERM_SEED,
) -> DupResult:
    """Return one group id per row. Empty input returns an empty result."""
    if len(row_ids) != len(texts):
        raise ValueError("row_ids and texts must have the same length")
    if not row_ids:
        return DupResult(group_ids=[], buckets_capped=0, unique_texts=0)

    order: list[str] = []
    rows_of: dict[str, list[int]] = defaultdict(list)
    for row_id, text in zip(row_ids, texts, strict=True):
        if text not in rows_of:
            order.append(text)
        rows_of[text].append(int(row_id))

    n_unique = len(order)
    parent = list(range(n_unique))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    perms_a, perms_b = _permutations(NUM_PERM, seed)
    signatures = [signature(text, perms_a, perms_b) for text in order]
    lengths = [len(text) for text in order]
    pairs, capped = _candidate_pairs(order, signatures, lengths)

    grams_cache: dict[int, set[str]] = {}

    def grams(index: int) -> set[str]:
        cached = grams_cache.get(index)
        if cached is None:
            cached = shingle_set(order[index])
            grams_cache[index] = cached
        return cached

    for left, right in pairs:
        if dice(grams(left), grams(right)) >= threshold:
            union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    index_of = {text: index for index, text in enumerate(order)}
    for text, ids in rows_of.items():
        members[find(index_of[text])].extend(ids)
    group_of_root = {root: min(ids) for root, ids in members.items()}

    group_ids = [group_of_root[find(index_of[text])] for text in texts]
    return DupResult(
        group_ids=group_ids,
        buckets_capped=capped,
        unique_texts=n_unique,
    )


def shingle_set(text: str, n: int = NGRAM) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def dice(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return (2.0 * overlap) / (len(left) + len(right))


def signature(text: str, perms_a: list[int], perms_b: list[int]) -> tuple[int, ...]:
    """MinHash signature. An empty string gets the maximum sentinel."""
    width = len(perms_a)
    values = shingle_set(text)
    if not values:
        return tuple([PRIME] * width)
    sig = [PRIME] * width
    for gram in values:
        hashed = _fnv(gram)
        for index, (coeff_a, coeff_b) in enumerate(zip(perms_a, perms_b, strict=True)):
            mixed = (coeff_a * hashed + coeff_b) % PRIME
            if mixed < sig[index]:
                sig[index] = mixed
    return tuple(sig)


def _permutations(count: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    coeffs_a: list[int] = []
    coeffs_b: list[int] = []
    used: set[int] = set()
    while len(coeffs_a) < count:
        coeff = rng.randrange(1, PRIME)
        if coeff in used:
            continue
        used.add(coeff)
        coeffs_a.append(coeff)
        coeffs_b.append(rng.randrange(0, PRIME))
    return coeffs_a, coeffs_b


def _fnv(text: str) -> int:
    hashed = 2166136261
    for char in text:
        hashed ^= ord(char)
        hashed = (hashed * 16777619) & 0xFFFFFFFF
    return hashed


def _candidate_pairs(
    texts: list[str],
    signatures: list[tuple[int, ...]],
    lengths: list[int],
) -> tuple[list[tuple[int, int]], int]:
    buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, sig in enumerate(signatures):
        for band in range(BANDS):
            start = band * ROWS_PER_BAND
            chunk = sig[start : start + ROWS_PER_BAND]
            buckets[("lsh", band, chunk)].append(index)
        for length in PREFIX_LENGTHS:
            buckets[("prefix", length, texts[index][:length])].append(index)

    pairs: set[tuple[int, int]] = set()
    capped = 0
    for members in buckets.values():
        unique_members = list(dict.fromkeys(members))
        if len(unique_members) < 2:
            continue
        found, was_capped = _pairs_in_bucket(unique_members, lengths)
        capped += was_capped
        pairs.update(found)
    return sorted(pairs), capped


def _pairs_in_bucket(members: list[int], lengths: list[int]) -> tuple[list[tuple[int, int]], int]:
    if len(members) <= MAX_BUCKET:
        return [(left, right) for left, right in combinations(sorted(members), 2)], 0

    capped = 1
    found: list[tuple[int, int]] = []
    stripes: dict[int, list[int]] = defaultdict(list)
    for index in members:
        stripes[lengths[index] // 8].append(index)
    for stripe in stripes.values():
        if len(stripe) <= MAX_BUCKET:
            found.extend(combinations(sorted(stripe), 2))
            continue
        capped += 1
        ordered = sorted(stripe, key=lambda index: (lengths[index], index))
        for pos, left in enumerate(ordered):
            for right in ordered[pos + 1 : pos + 1 + NEIGHBOUR_WINDOW]:
                pair = (left, right) if left < right else (right, left)
                found.append(pair)
    return found, capped
