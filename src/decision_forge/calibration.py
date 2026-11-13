"""Post-hoc temperature scaling calibration, reliability/risk-coverage metrics, and order-flip checks."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize_scalar

from decision_forge.constants import RISK_COVERAGE_THRESHOLDS
from decision_forge.prompts import shuffle_item_options
from decision_forge.types import CalibrationReport, DecisionItem

if TYPE_CHECKING:
    from decision_forge.engine import DecisionEngine


def bucket_for_item(item: DecisionItem) -> str:
    """Assign a DecisionItem to its calibration bucket from CALIBRATION_BUCKETS."""
    if item.primitive == "noul":
        return "noul"
    if item.primitive == "score":
        return "score"
    n_opts = len(item.options)
    if n_opts <= 4:
        return "choice:small"
    if n_opts <= 8:
        return "choice:mid"
    return "choice:large"


def _log_softmax_scaled(logits: np.ndarray, temp: float) -> np.ndarray:
    scaled = logits / temp
    max_val = float(np.max(scaled))
    shifted = scaled - max_val
    lse = float(np.log(np.sum(np.exp(shifted))))
    return shifted - lse


def _softmax_scaled(logits: np.ndarray, temp: float) -> np.ndarray:
    return np.exp(_log_softmax_scaled(logits, temp))


def _mean_nll_subset(
    logits_arrs: list[np.ndarray],
    gold_indices: list[int],
    indices: list[int],
    temp: float,
) -> float:
    if not indices:
        return 0.0
    total = 0.0
    for idx in indices:
        log_probs = _log_softmax_scaled(logits_arrs[idx], temp)
        total -= float(log_probs[gold_indices[idx]])
    return total / len(indices)


def _fit_temp_subset(
    logits_arrs: list[np.ndarray],
    gold_indices: list[int],
    indices: list[int],
) -> float:
    if not indices:
        return 1.0
    res = minimize_scalar(
        lambda log_t: _mean_nll_subset(logits_arrs, gold_indices, indices, math.exp(log_t)),
        bounds=(-2.5, 2.5),
        method="bounded",
    )
    cand_t = float(math.exp(float(res.x)))
    cand_nll = _mean_nll_subset(logits_arrs, gold_indices, indices, cand_t)
    unit_nll = _mean_nll_subset(logits_arrs, gold_indices, indices, 1.0)
    if unit_nll <= cand_nll:
        return 1.0
    return cand_t


def _compute_ece_and_bins(
    probs_list: list[np.ndarray],
    gold_indices: list[int],
    num_bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    n = len(probs_list)
    if n == 0:
        return 0.0, []

    confs = np.array([float(np.max(p)) for p in probs_list], dtype=np.float64)
    preds = np.array([int(np.argmax(p)) for p in probs_list], dtype=np.int64)
    corrects = (preds == np.asarray(gold_indices, dtype=np.int64)).astype(np.float64)

    ece = 0.0
    bins: list[dict[str, Any]] = []
    for m in range(num_bins):
        lower = m / num_bins
        upper = (m + 1) / num_bins
        if m == 0:
            in_bin = (confs >= lower) & (confs <= upper)
        else:
            in_bin = (confs > lower) & (confs <= upper)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confs[in_bin]))
            avg_acc = float(np.mean(corrects[in_bin]))
            gap = abs(avg_acc - avg_conf)
            ece += (count / n) * gap
        else:
            avg_conf = 0.0
            avg_acc = 0.0
            gap = 0.0

        bins.append(
            {
                "bin_index": m,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "avg_confidence": float(avg_conf),
                "accuracy": float(avg_acc),
                "gap": float(gap),
            }
        )
    return float(ece), bins


def _compute_brier(probs_list: list[np.ndarray], gold_indices: list[int]) -> float:
    if not probs_list:
        return 0.0
    total = 0.0
    for probs, gold in zip(probs_list, gold_indices):
        sq_sum = float(np.sum(probs * probs))
        total += sq_sum - 2.0 * float(probs[gold]) + 1.0
    return total / len(probs_list)


def _compute_risk_coverage(
    probs_list: list[np.ndarray],
    gold_indices: list[int],
) -> list[dict[str, Any]]:
    n = len(probs_list)
    if n == 0:
        return []

    confs = np.array([float(np.max(p)) for p in probs_list], dtype=np.float64)
    preds = np.array([int(np.argmax(p)) for p in probs_list], dtype=np.int64)
    corrects = (preds == np.asarray(gold_indices, dtype=np.int64)).astype(np.float64)

    order = np.argsort(-confs, kind="mergesort")
    sorted_confs = confs[order]
    sorted_corrects = corrects[order]

    curve: list[dict[str, Any]] = []
    for target_cov in RISK_COVERAGE_THRESHOLDS:
        k = max(1, min(n, math.ceil(float(target_cov) * n)))
        sel_acc = float(np.mean(sorted_corrects[:k]))
        curve.append(
            {
                "threshold": float(target_cov),
                "target_coverage": float(target_cov),
                "coverage": float(k / n),
                "count": int(k),
                "accuracy": sel_acc,
                "risk": float(1.0 - sel_acc),
                "min_confidence": float(sorted_confs[k - 1]),
            }
        )
    return curve


def fit_temperature_scaling(
    logits_list: list[np.ndarray],
    gold_indices: list[int],
    buckets: list[str] | None = None,
    per_bucket: bool = True,
    min_bucket_samples: int = 30,
    num_bins: int = 10,
) -> CalibrationReport:
    """Fit global and optional per-bucket temperatures via NLL minimization."""
    if len(logits_list) != len(gold_indices):
        raise ValueError(
            f"logits_list length ({len(logits_list)}) != gold_indices length ({len(gold_indices)})."
        )
    if buckets is not None and len(buckets) != len(logits_list):
        raise ValueError(
            f"buckets length ({len(buckets)}) != logits_list length ({len(logits_list)})."
        )

    n = len(logits_list)
    if n == 0:
        return CalibrationReport(
            global_temperature=1.0,
            bucket_temperatures={},
            val_samples=0,
            accuracy=0.0,
            nll_before=0.0,
            nll_after=0.0,
            ece_10_before=0.0,
            ece_10_after=0.0,
            ece_15_before=0.0,
            ece_15_after=0.0,
            brier_before=0.0,
            brier_after=0.0,
            order_flip_rate=None,
            reliability_bins=[],
            risk_coverage=[],
        )

    logits_arrs: list[np.ndarray] = []
    for i, raw in enumerate(logits_list):
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Empty logits array at index {i}.")
        g = int(gold_indices[i])
        if not (0 <= g < arr.size):
            raise ValueError(f"gold_idx {g} out of bounds for {arr.size} options at index {i}.")
        logits_arrs.append(arr)

    all_indices = list(range(n))
    global_temp = _fit_temp_subset(logits_arrs, gold_indices, all_indices)

    bucket_temps: dict[str, float] = {}
    if per_bucket and buckets is not None:
        grouped: dict[str, list[int]] = {}
        for idx, b in enumerate(buckets):
            grouped.setdefault(b, []).append(idx)
        for b_name, idxs in sorted(grouped.items()):
            if len(idxs) >= min_bucket_samples:
                bucket_temps[b_name] = _fit_temp_subset(logits_arrs, gold_indices, idxs)

    probs_before: list[np.ndarray] = []
    probs_after: list[np.ndarray] = []
    nll_before = 0.0
    nll_after = 0.0
    correct = 0

    for i, arr in enumerate(logits_arrs):
        g = gold_indices[i]
        if int(np.argmax(arr)) == g:
            correct += 1

        lp_before = _log_softmax_scaled(arr, 1.0)
        nll_before -= float(lp_before[g])
        probs_before.append(np.exp(lp_before))

        t_eff = (
            bucket_temps.get(buckets[i], global_temp)
            if (per_bucket and buckets is not None)
            else global_temp
        )
        lp_after = _log_softmax_scaled(arr, t_eff)
        nll_after -= float(lp_after[g])
        probs_after.append(np.exp(lp_after))

    ece_10_before, _ = _compute_ece_and_bins(probs_before, gold_indices, 10)
    ece_10_after, bins_10 = _compute_ece_and_bins(probs_after, gold_indices, 10)
    ece_15_before, _ = _compute_ece_and_bins(probs_before, gold_indices, 15)
    ece_15_after, _ = _compute_ece_and_bins(probs_after, gold_indices, 15)
    if num_bins != 10:
        _, reliability_bins = _compute_ece_and_bins(probs_after, gold_indices, num_bins)
    else:
        reliability_bins = bins_10

    brier_before = _compute_brier(probs_before, gold_indices)
    brier_after = _compute_brier(probs_after, gold_indices)
    risk_coverage = _compute_risk_coverage(probs_after, gold_indices)

    return CalibrationReport(
        global_temperature=float(global_temp),
        bucket_temperatures=bucket_temps,
        val_samples=n,
        accuracy=float(correct / n),
        nll_before=float(nll_before / n),
        nll_after=float(nll_after / n),
        ece_10_before=ece_10_before,
        ece_10_after=ece_10_after,
        ece_15_before=ece_15_before,
        ece_15_after=ece_15_after,
        brier_before=float(brier_before),
        brier_after=float(brier_after),
        order_flip_rate=None,
        reliability_bins=reliability_bins,
        risk_coverage=risk_coverage,
    )


def measure_order_flip_rate(
    engine: DecisionEngine,
    items: list[DecisionItem],
    num_shuffles: int = 3,
    seed: int = 42,
) -> float:
    """Measure how often option-order shuffling changes the predicted option key."""
    if not items or num_shuffles <= 0:
        return 0.0

    rng = random.Random(seed)
    base_preds = engine.predict_batch(items)
    base_choices = [p.choice for p in base_preds]

    flips = 0
    total = len(items) * num_shuffles
    for _ in range(num_shuffles):
        shuffled = [shuffle_item_options(it, rng) for it in items]
        shuf_preds = engine.predict_batch(shuffled)
        for base_c, shuf_p in zip(base_choices, shuf_preds):
            if shuf_p.choice != base_c:
                flips += 1

    return float(flips / total)
