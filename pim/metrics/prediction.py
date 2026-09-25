"""Prediction quality: a model's held-out loss against its instance's Bayes floor, in training-loss units.

The floor (``runs/_baselines/<env>/<instance>/bayes_floor.json``) is the Bayes-optimal predictor's
loss on the same history: exact on Othello, a sampled bracket on Rayworld (``floor_estimate``).
"""
from __future__ import annotations

import math

import numpy as np


def next_frame_mse(pred, target) -> np.ndarray:
    """(N, T, R) predicted vs true next frames -> (N,) MSE per sequence over positions and rays
    (``pim.training.train.mse_next_obs`` kept per sequence)."""
    p, y = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or p.ndim != 3:
        raise ValueError(f"expected matching (N, T, R) arrays, got {p.shape} and {y.shape}")
    return ((p - y) ** 2).mean(axis=(1, 2))


def log_softmax(logits) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def next_token_ce(logits, target, ignore_index: int | None = None) -> np.ndarray:
    """(N, T, V) logits vs (N, T) integer targets -> (N,) mean cross-entropy per sequence in nats
    (``pim.training.train.ce_next_move`` kept per sequence; ``ignore_index`` positions are skipped)."""
    lp = log_softmax(logits)
    y = np.asarray(target, np.int64)
    keep = np.ones_like(y, bool) if ignore_index is None else (y != ignore_index)
    nll = -np.take_along_axis(lp, np.where(keep, y, 0)[..., None], axis=-1)[..., 0]
    return (nll * keep).sum(1) / np.maximum(keep.sum(1), 1)


def expected_frame(probs, frames, drop: tuple[int, ...] = (0,)) -> tuple[np.ndarray, np.ndarray]:
    """``probs`` (..., V) over a frame vocabulary with ``frames`` (V, R) -> (mean frame (..., R), dropped
    mass (...)), so a token model is scored by MSE. Ids in ``drop`` (undefined frames) are removed and
    the rest renormalized."""
    p = np.array(probs, np.float64, copy=True)
    fr = np.array(frames, np.float64, copy=True)
    dropped = p[..., list(drop)].sum(-1) if drop else np.zeros(p.shape[:-1])
    for d in drop:
        p[..., d] = 0.0
        fr[d] = 0.0                                   # the reserved id decodes to NaN
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-300)
    return p @ fr, dropped


def mean_se(per_sequence) -> tuple[float, float]:
    """(mean, standard error of the mean over sequences)."""
    a = np.asarray(per_sequence, np.float64)
    return float(a.mean()), (float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan"))


def floor_estimate(floor: dict, objective: str) -> tuple[float, float] | None:
    """(value, ±): the floor as one number. Exact: (value, 0). Sampled: the bracket's midpoint,
    ± half its width plus the larger of the two ends' standard errors over sequences."""
    b = (floor or {}).get(objective)
    if not b:
        return None
    if "exact" in b:
        return float(b["exact"]), 0.0
    lo, hi = float(b["lo"]), float(b["hi"])
    se = max(float(b.get("lo_se") or 0.0), float(b.get("hi_se") or 0.0))
    return (lo + hi) / 2, abs(hi - lo) / 2 + se


def excess_estimate(loss: float, estimate: tuple[float, float] | None) -> tuple[float, float, float] | None:
    """(loss - floor, ±, (loss - floor) / floor). The ± is the floor's: the loss is taken on the
    same sequences, so their sampling noise is largely shared."""
    if estimate is None or loss is None:
        return None
    v, pm = estimate
    return float(loss - v), float(pm), float((loss - v) / v) if v else float("nan")


def gap_closed(loss: float, trivial: float | None, floor: float | None) -> float:
    """(trivial - loss) / (trivial - floor): 1 at the Bayes floor, 0 at the trivial predictor.
    Unit-free, so comparable across environments."""
    if loss is None or trivial is None or floor is None or not trivial > floor:
        return float("nan")
    return float((trivial - loss) / (trivial - floor))

