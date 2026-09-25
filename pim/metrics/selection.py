"""Which number a table reports, selected from the arms and probe skills stored in ``scores.json``.

A probe is reported at its best residual point; an editor at its highest Edit Index among arms with
fidelity ratio <= 1, or, if none, at its lowest fidelity ratio (``within_guard`` False).
"""
from __future__ import annotations

import numpy as np

GUARD = 1.0          # a fidelity ratio above this: the edit degraded the prediction


def best_point(values) -> tuple[float, int]:
    """(maximum over residual points, its index); ``(nan, -1)`` when no value is finite."""
    v = np.asarray(values, float)
    if v.size == 0 or not np.isfinite(v).any():
        return float("nan"), -1
    return float(np.nanmax(v)), int(np.nanargmax(v))


def arms_of(arms: list[dict], editor: str) -> list[dict]:
    """An editor's arms. Labels are "PI[zspace]", "GS@L0", "IM" on Rayworld and "PI", "GS", "IM" on Othello."""
    return [a for a in arms if a["editor"] == editor or a["editor"].startswith(editor + "[")
            or a["editor"].startswith(editor + "@")]


def best_arm(arms: list[dict], editor: str, key: str, guard: float | None = GUARD) -> dict | None:
    """The arm an editor is reported at: max ``key`` among arms with ``fidelity_ratio <= guard``;
    if none is inside the guard, the arm with the lowest ``fidelity_ratio`` (max ``key`` overall
    if no arm carries a ratio). Returns a copy with ``within_guard``. ``guard=None``: plain argmax."""
    def ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    sub = [a for a in arms_of(arms, editor) if ok(a.get(key))]
    if not sub:
        return None
    rated = [a for a in sub if ok(a.get("fidelity_ratio"))]
    inside = [a for a in rated if guard is not None and a["fidelity_ratio"] <= guard]
    if inside:
        pick = max(inside, key=lambda a: a[key])
    elif guard is not None and rated:
        pick = min(rated, key=lambda a: a["fidelity_ratio"])
    else:
        pick = max(sub, key=lambda a: a[key])
    return {**pick, "within_guard": bool(inside)}


def best_arm_by_fidelity(arms: list[dict], editor: str, key: str) -> dict | None:
    """The arm with the lowest fidelity ratio among an editor's arms that carry ``key``, i.e. the
    write that brings the output closest to the edited world. ``within_guard``: ratio <= ``GUARD``."""
    def ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    sub = [a for a in arms_of(arms, editor) if ok(a.get(key)) and ok(a.get("fidelity_ratio"))]
    if not sub:
        return None
    pick = min(sub, key=lambda a: a["fidelity_ratio"])
    return {**pick, "within_guard": bool(pick["fidelity_ratio"] <= GUARD)}
