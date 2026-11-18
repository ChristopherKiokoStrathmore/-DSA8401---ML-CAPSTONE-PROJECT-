"""Metrics, the validation emergency threshold, and bootstrap intervals.

Nothing in this module reads a test set. Callers pass the arrays they chose.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

# Policy cap, not tuned on test. About one false emergency in five
# non-emergencies is the most the queue flag is still worth paging on.
EMERGENCY_FALSE_RATE_CAP = 0.20
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def per_class_scores(
    gold: list[str], pred: list[str], classes: list[str]
) -> tuple[dict[str, dict], float, list[list[int]]]:
    """Precision, recall, F1, and support for every class, plus macro-F1.

    Macro-F1 is the unweighted mean over the full class list. A class with no
    gold rows contributes 0 and is reported with n=0.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        pred,
        labels=classes,
        average=None,
        zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(classes):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "n": int(support[index]),
        }
    macro = float(np.mean([per_class[name]["f1"] for name in classes])) if classes else 0.0
    matrix = confusion_matrix(gold, pred, labels=classes).tolist()
    return per_class, macro, matrix


def quadratic_weighted_kappa(gold: list[str], pred: list[str], classes: list[str]) -> float:
    if not gold:
        return 0.0
    mapping = {name: index for index, name in enumerate(classes)}
    gold_i = [mapping[label] for label in gold]
    pred_i = [mapping[label] for label in pred]
    if len(set(gold_i)) < 2 and gold_i == pred_i:
        return 1.0
    return float(cohen_kappa_score(gold_i, pred_i, weights="quadratic"))


def emergency_rates(gold: list[str], flags: list[bool]) -> tuple[float, float, int, int]:
    """Emergency recall and false-emergency rate.

    False-emergency rate is the share of non-emergency gold rows flagged
    emergency. Recall is undefined when the gold set has no emergencies; the
    returned recall is then 0 and ``n_emergency`` is 0.
    """
    positives = [flag for label, flag in zip(gold, flags, strict=True) if label == "emergency"]
    negatives = [flag for label, flag in zip(gold, flags, strict=True) if label != "emergency"]
    recall = float(sum(positives) / len(positives)) if positives else 0.0
    false_rate = float(sum(negatives) / len(negatives)) if negatives else 0.0
    return recall, false_rate, len(positives), len(negatives)


def choose_emergency_threshold(
    gold: list[str],
    emergency_probs: list[float],
    *,
    cap: float = EMERGENCY_FALSE_RATE_CAP,
) -> dict:
    """Pick a threshold on validation probabilities only.

    Among thresholds whose false-emergency rate is at most ``cap``, take the
    highest emergency recall. Ties go to the lower false-emergency rate, then
    to the higher threshold. If none meet the cap, take the lowest
    false-emergency rate instead and record that the cap was missed.
    """
    if len(gold) != len(emergency_probs):
        raise ValueError("gold and probabilities must be the same length")
    grid = [step / 20 for step in range(1, 20)]
    scored = []
    for threshold in grid:
        flags = [prob >= threshold for prob in emergency_probs]
        recall, false_rate, n_pos, n_neg = emergency_rates(gold, flags)
        scored.append(
            {
                "threshold": threshold,
                "recall": recall,
                "false_emergency_rate": false_rate,
                "meets_cap": false_rate <= cap,
                "n_emergency": n_pos,
                "n_non_emergency": n_neg,
            }
        )
    feasible = [row for row in scored if row["meets_cap"]]
    pool = feasible or scored
    # max recall, then min false rate, then max threshold
    chosen = max(
        pool,
        key=lambda row: (row["recall"], -row["false_emergency_rate"], row["threshold"]),
    )
    return {
        "threshold": chosen["threshold"],
        "val_emergency_recall": chosen["recall"],
        "val_false_emergency_rate": chosen["false_emergency_rate"],
        "meets_cap": chosen["meets_cap"],
        "cap": cap,
        "fallback_lowest_false_rate": not feasible,
        "rule": (
            "On validation only: maximise emergency recall subject to "
            f"false-emergency rate <= {cap:.2f}. "
            "Ties: lower false-emergency rate, then a higher threshold. "
            "If no threshold meets the cap, use the lowest false-emergency rate."
        ),
    }


def apply_emergency_flag(probs: list[float], threshold: float) -> list[bool]:
    return [prob >= threshold for prob in probs]


def bootstrap_ci(
    gold: list[str],
    pred: list[str],
    metric,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile 95% interval of ``metric(gold, pred)`` under row resampling."""
    raw_estimate = metric(gold, pred)
    estimate = None if raw_estimate is None else float(raw_estimate)
    if not gold or estimate is None:
        return {"estimate": estimate, "ci95": [estimate, estimate], "resamples_used": 0}
    rng = np.random.default_rng(seed)
    gold_a = np.asarray(gold, dtype=object)
    pred_a = np.asarray(pred, dtype=object)
    samples = []
    n = len(gold)
    for _ in range(resamples):
        index = rng.integers(0, n, size=n)
        value = metric(gold_a[index].tolist(), pred_a[index].tolist())
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        samples.append(float(value))
    if not samples:
        return {"estimate": estimate, "ci95": [None, None], "resamples_used": 0}
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "resamples_used": len(samples),
    }


def paired_bootstrap_diff(
    gold: list[str],
    pred_a: list[str],
    pred_b: list[str],
    metric,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """CI for metric(A) - metric(B) on shared resamples. B is the reference."""
    left = metric(gold, pred_a)
    right = metric(gold, pred_b)
    estimate = None if left is None or right is None else float(left - right)
    if not gold or estimate is None:
        return {"estimate": estimate, "ci95": [estimate, estimate], "resamples_used": 0}
    rng = np.random.default_rng(seed)
    gold_a = np.asarray(gold, dtype=object)
    a = np.asarray(pred_a, dtype=object)
    b = np.asarray(pred_b, dtype=object)
    samples = []
    n = len(gold)
    for _ in range(resamples):
        index = rng.integers(0, n, size=n)
        gold_s = gold_a[index].tolist()
        left = metric(gold_s, a[index].tolist())
        right = metric(gold_s, b[index].tolist())
        if left is None or right is None:
            continue
        diff = float(left - right)
        if np.isnan(diff):
            continue
        samples.append(diff)
    if not samples:
        return {"estimate": estimate, "ci95": [None, None], "resamples_used": 0}
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "resamples_used": len(samples),
    }
