"""The linear probe (Li et al. §3.1): one affine map on the standardized activation.

Regression: ``y = (A z + b) * y_std + y_mean`` with ``z = (h - x_mean) / x_std``, solved in closed
form. Classification: ``softmax(W z)`` per tile, trained by the same Adam loop as the MLP probe.
"""

from __future__ import annotations

import numpy as np

from pim.probes.base import FIT_BATCH, FIT_EPOCHS, FIT_LR, WorldStateProbe, fit_probe


def fit_linear(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_classes: int | None = None,
    epochs: int = FIT_EPOCHS,
    lr: float = FIT_LR,
    batch: int = FIT_BATCH,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[WorldStateProbe, dict]:
    """Fit the linear probe. Returns ``(probe, stats)``: ``r2`` / ``per_dim_r2`` (regression) or
    ``error_rate`` / ``per_tile_error_rate`` (classification), with in-sample counterparts."""
    return fit_probe(x_tr, y_tr, x_te, y_te, hidden=None, n_classes=n_classes,
                     epochs=epochs, lr=lr, batch=batch, device=device, seed=seed)
