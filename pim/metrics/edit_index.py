"""The Edit Index and Edit Fidelity formulas, shared by the two constructions.

``zone_editability`` scores a predicted frame against the two worlds' clean renders on the rays
where they differ; ``set_editability`` scores a predicted distribution against uniform
distributions over each world's set of outcomes. Both call ``edit_index_per_case``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["masked_rmse_per_case", "edit_index_per_case", "fidelity_ratio_from",
           "case_stats", "ratio_ci95"]


def masked_rmse_per_case(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,) RMSE between ``pred`` and ``ref`` over each case's ``mask``; NaN where the mask is empty."""
    out = np.full(len(pred), np.nan)
    for i in range(len(pred)):
        m = mask[i]
        if m.any():
            out[i] = np.sqrt(((pred[i, m] - ref[i, m]) ** 2).mean())
    return out


def edit_index_per_case(pred: np.ndarray, ref_edited: np.ndarray, ref_unedited: np.ndarray,
                        support: np.ndarray) -> np.ndarray:
    """(N,) Edit Index ``(d_uned - d_edit) / (d_uned + d_edit)``, ``d = RMSE(pred, ref)`` over ``support``:
    +1 the output is the edited world, -1 the unedited one, about 0 equidistant or far from both.
    NaN where the support is empty or both distances vanish; average with ``nanmean``."""
    d_e = masked_rmse_per_case(pred, ref_edited, support)
    d_u = masked_rmse_per_case(pred, ref_unedited, support)
    with np.errstate(invalid="ignore", divide="ignore"):
        ei = (d_u - d_e) / (d_u + d_e)
    ei[~(d_u + d_e > 1e-12)] = np.nan
    return ei


def fidelity_ratio_from(rmse_edited: float, rmse_unsteered: float, eps: float = 1e-12) -> float:
    """The fidelity ratio: RMSE(edited prediction, edited-world truth) / RMSE(unedited prediction,
    same truth) at the edit step. Above 1 the edit left the output further from the truth than
    no edit. Absolute, where the Edit Index is relative."""
    return float(rmse_edited) / max(float(rmse_unsteered), eps)


FIDELITY_GUARD = 0.0     # Edit Fidelity below this: the edit degraded the prediction (ratio > 1)


def fidelity(ratio):
    """Edit Fidelity = 1 - fidelity ratio: 1 reproduces the edited world exactly, 0 is no better
    than no edit, below 0 is degraded. ``scores.json`` stores the ratio; tables report this.
    A ratio interval ``[lo, hi]`` becomes ``[1 - hi, 1 - lo]``."""
    return 1.0 - ratio


# Case-level spread: how much an arm's aggregates move when the bench cases are resampled
# (bootstrap with a fixed seed). This is estimation noise on one model, not the seed spread
# the tables report (``pim.metrics.replicates``).

N_BOOT = 1000


def case_stats(per_case, *, n_boot: int = N_BOOT, seed: int = 0, prefix: str = "") -> dict:
    """Case-level spread of a per-case metric averaged by ``nanmean``: SD (ddof 1), standard error,
    number of scored cases and a percentile-bootstrap 95% interval of the mean, NaN cases dropped.
    Keys are prefixed (``<prefix>case_sd``, ...)."""
    v = np.asarray(per_case, float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    out = {f"{prefix}case_sd": float("nan"), f"{prefix}case_se": float("nan"), f"{prefix}n_cases": n,
           f"{prefix}ci95_lo": float("nan"), f"{prefix}ci95_hi": float("nan")}
    if n < 2:
        return out
    out[f"{prefix}case_sd"] = float(v.std(ddof=1))
    out[f"{prefix}case_se"] = float(v.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    out[f"{prefix}ci95_lo"], out[f"{prefix}ci95_hi"] = float(lo), float(hi)
    return out


def ratio_ci95(num_per_case, den_per_case, *, root: bool = False, n_boot: int = N_BOOT,
               seed: int = 0, eps: float = 1e-12) -> tuple[float, float]:
    """Percentile-bootstrap 95% interval of a ratio ``agg(num) / agg(den)``, the two resampled as
    pairs. ``agg`` is the mean, or the root of the mean with ``root`` (a ratio of RMSEs from
    per-case mean squared errors, as on Rayworld). Cases with a NaN on either side are dropped."""
    a = np.asarray(num_per_case, float)
    b = np.asarray(den_per_case, float)
    if a.shape != b.shape:
        raise ValueError(f"per-case arrays differ in shape: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    na, nb = a[idx].mean(axis=1), b[idx].mean(axis=1)
    if root:
        na, nb = np.sqrt(na), np.sqrt(nb)
    r = na / np.maximum(nb, eps)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)
