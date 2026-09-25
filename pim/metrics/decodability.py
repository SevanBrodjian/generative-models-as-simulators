"""Probe Skill = 1 - probe loss / trivial-predictor loss, one axis for regression and classification.

1 is perfect, 0 is the trivial predictor (the train mean, or the single most common class pooled over
all cells; always from the train split), below 0 is an overfit probe. Against the train mean it is R².
"""

from __future__ import annotations

import numpy as np

__all__ = ["probe_skill_regression", "r2", "probe_skill_from_stats", "insample_gap_from_stats"]


def probe_skill_from_stats(stats: dict) -> float:
    """Probe Skill from a fit's stats dict: ``r2`` for regression, ``1 - error_rate /
    majority_class_error_rate`` for classification, the trivial predictor being the single most
    common class pooled over all cells of the fit's train split."""
    if "r2" in stats:
        return float(stats["r2"])
    return float(1.0 - stats["error_rate"] / stats["majority_class_error_rate"])


def insample_gap_from_stats(stats: dict) -> float:
    """In-sample minus held-out Probe Skill (the overfit check); NaN without an in-sample stat."""
    if "r2" in stats:
        return float(stats.get("r2_insample", np.nan)) - float(stats["r2"])
    if "error_rate_insample" not in stats:
        return float("nan")
    return float((stats["error_rate"] - stats["error_rate_insample"])
                 / stats["majority_class_error_rate"])


def probe_skill_regression(pred: np.ndarray, y: np.ndarray,
                           train_mean: np.ndarray) -> float:
    """1 - SSE(probe) / SSE(train-mean predictor): R² against the train mean."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    denom = ((y - np.asarray(train_mean, float)) ** 2).sum()
    if denom <= 0:
        raise ValueError("trivial predictor has zero error: the target is constant on this "
                         "split, so skill is undefined")
    return float(1.0 - ((pred - y) ** 2).sum() / denom)


def r2(pred: np.ndarray, y: np.ndarray, train_mean: np.ndarray) -> float:
    """R² against the train mean; the same number as ``probe_skill_regression``."""
    return probe_skill_regression(pred, y, train_mean)
