"""The MLP-128 probe (Li et al. §3.2): one hidden layer of width 128, and the MLP-vs-linear check.

The regression loss is taken in standardized target space, so every output dimension counts equally.
"""

from __future__ import annotations

import numpy as np

from pim.probes.base import (  # noqa: F401 (CANONICAL_HIDDEN is re-exported)
    CANONICAL_HIDDEN, FIT_BATCH, FIT_EPOCHS, FIT_LR, WorldStateProbe, fit_probe)


def fit_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    hidden: int = CANONICAL_HIDDEN,
    n_classes: int | None = None,
    epochs: int = FIT_EPOCHS,
    lr: float = FIT_LR,
    batch: int = FIT_BATCH,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[WorldStateProbe, dict]:
    """Fit the MLP probe. Returns ``(probe, stats)`` as ``fit_linear``."""
    return fit_probe(x_tr, y_tr, x_te, y_te, hidden=hidden, n_classes=n_classes,
                     epochs=epochs, lr=lr, batch=batch, device=device, seed=seed)


class ProbeSanityError(AssertionError):
    """An MLP probe scored worse than a linear one on held-out data."""


def check_probe_sanity(lin: dict, mlp: dict, *, tol: float = 0.01, strict: bool = True,
                       label: str = "", log=print) -> dict:
    """Check that the held-out MLP probe's Probe Skill is not below the linear probe's at any point
    (``lin`` / ``mlp``: residual point -> ``(probe, stats)``); an MLP can represent the linear map.
    Returns the report (``r2_*`` keys hold the skill); raises ``ProbeSanityError`` when ``strict``."""
    from pim.metrics.decodability import insample_gap_from_stats, probe_skill_from_stats

    rows, bad = [], []
    for ell in sorted(set(lin) & set(mlp)):
        sl, sm = lin[ell][1], mlp[ell][1]
        r_lin, r_mlp = probe_skill_from_stats(sl), probe_skill_from_stats(sm)
        gap_mlp = insample_gap_from_stats(sm)
        gap_lin = insample_gap_from_stats(sl)
        row = {"point": ell, "r2_linear": r_lin, "r2_mlp": r_mlp,
               "mlp_minus_linear": r_mlp - r_lin,
               "insample_gap_mlp": gap_mlp, "insample_gap_linear": gap_lin}
        rows.append(row)
        if r_mlp < r_lin - tol:
            bad.append(row)
    report = {"label": label, "tol": tol, "rows": rows, "n_violations": len(bad)}
    if log:
        worst = max(rows, key=lambda r: r["insample_gap_mlp"]) if rows else None
        if worst is not None:
            log(f"    probe sanity{' [' + label + ']' if label else ''}: "
                f"{len(bad)}/{len(rows)} points where MLP < linear; "
                f"worst MLP in-sample gap {worst['insample_gap_mlp']:+.4f} "
                f"@ point {worst['point']}")
    if bad and strict:
        det = "\n".join(f"      point {r['point']}: linear {r['r2_linear']:+.4f} > "
                        f"MLP {r['r2_mlp']:+.4f} (by {-r['mlp_minus_linear']:.4f}), "
                        f"MLP in-sample gap {r['insample_gap_mlp']:+.4f}" for r in bad)
        raise ProbeSanityError(
            f"MLP probe beaten by linear probe at {len(bad)} residual point(s)"
            f"{' for ' + label if label else ''}: the MLP is fitting the probe training "
            f"set, not the representation. Refit with more probe sequences (>=10k) before "
            f"trusting any decodability number from it.\n{det}")
    return report
