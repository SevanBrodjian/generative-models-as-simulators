"""The spread over a seed-replicate set: SD, mean and t-based 95% interval per run and block.

A replicate row is a flat dict of the quantities a table shows for one (replicate run, block),
plus ``parent``, ``basis``, ``steps`` and ``seed``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Student-t 0.975 quantiles by degrees of freedom (n - 1); beyond 30 the normal quantile. A 95%
# interval on the mean of n seeds is mean ± t * SD / sqrt(n) (2.48 SD at n = 3). An unlisted df
# takes the next lower listed df's quantile (the wider interval).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df: int) -> float:
    if df in _T975:
        return _T975[df]
    if df > 30:
        return 1.960
    lower = max(k for k in _T975 if k < df)
    return _T975[lower]


def ci95_halfwidth(values) -> float:
    """t-based 95% half-width of the mean over a replicate set (NaN below n = 2)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return float("nan")
    return float(t975(n - 1) * v.std(ddof=1) / np.sqrt(n))


def pool_replicates(rep_rows: list[dict], cols: list[str], *, pool_budgets: bool = False,
                    budget_tolerance: float = 0.10) -> dict:
    """Per (parent, basis): ``n``, ``steps``, ``seeds``, ``dropped_steps`` and, per column, SD (ddof 1),
    ``_mean``, ``_ci95``, ``_values``. Only replicates within ``budget_tolerance`` of each other in
    ``steps`` pool (largest such set, ties to the larger budget) unless ``pool_budgets``."""
    if not rep_rows:
        return {}
    R = pd.DataFrame(rep_rows)
    if "steps" not in R:
        R["steps"] = np.nan
    R["steps"] = R["steps"].fillna(-1).astype(int)
    out = {}
    for (parent, basis), g in R.groupby(["parent", "basis"]):
        dropped: list[int] = []
        if not pool_budgets:
            groups: list[list] = []
            for _, r in g.sort_values("steps").iterrows():
                if groups and r["steps"] <= groups[-1][0]["steps"] * (1 + budget_tolerance):
                    groups[-1].append(r)
                else:
                    groups.append([r])
            chosen = max(groups, key=lambda grp: (len(grp), max(r["steps"] for r in grp)))
            dropped = sorted({int(r["steps"]) for grp in groups if grp is not chosen for r in grp})
            g = pd.DataFrame(chosen)
        if len(g) < 2:
            continue
        g = g.sort_values("seed") if "seed" in g else g
        have = [c for c in cols if c in g and g[c].notna().sum() >= 2]
        out[(parent, basis)] = {
            "n": int(len(g)), "steps": sorted({int(x) for x in g["steps"]}),
            "seeds": sorted(int(x) for x in g["seed"]) if "seed" in g else [],
            "dropped_steps": dropped, "pooled_budgets": bool(pool_budgets),
            **{c: float(g[c].std(ddof=1)) for c in have},
            **{f"{c}_mean": float(g[c].mean()) for c in cols if c in g},
            **{f"{c}_ci95": ci95_halfwidth(g[c].dropna().to_numpy()) for c in have},
            **{f"{c}_values": [float(x) for x in g[c]] for c in cols if c in g}}
    return out
