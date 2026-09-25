"""The ``scores.json`` block: which probe-target blocks a run gets, and the shape of one.

``probe_block`` is the schema the table code reads. A Rayworld block is one regression basis or
one categorical / snapped target of the run's own instance; each is its own table row.
"""
import json

import numpy as np

from pim.environments.rayworld import arms as rwa
from pim.environments.rayworld.grid_target import categorical_target, snapped_target
from pim.metrics.decodability import probe_skill_from_stats
from pim.scoring.runs import REPO

# PI and GS read the probes; IM and IM-NN write the state
EDITORS_SCORED = ("PI", "GS", "IM", "IM-NN")
IM_VERSION = "1.0"


def top_arm(recs, key="edit_index"):
    """The record with the highest ``key``, scalars only (the ``best`` fields; the tables apply
    the guarded selection rule of ``pim.metrics.selection`` themselves)."""
    b = max(recs, key=lambda r: r[key])
    return {k: v for k, v in b.items() if np.isscalar(v)}


def by_point(probes):
    """A probe set's stats in residual-point order, independent of the dict's insertion order."""
    return [probes[ell][1] for ell in sorted(probes)]


def run_config(run_id: str) -> dict:
    return json.loads((REPO / "runs" / run_id / "config.json").read_text())


def rw_bases_for(instance, s):
    """The regression bases a Rayworld instance is scored in; the first also keys its categorical
    probes."""
    return tuple(s.get("rw_bases_by_instance", {}).get(f"rayworld/{instance}", s["rw_bases"]))


def extra_targets_of(run_id, s):
    """A run's extra probe targets: its own ``rw_extra_targets`` entry (a seed replicate is listed
    under its own id, never inherited from its parent)."""
    return tuple(s["rw_extra_targets"].get(run_id, ()))


def rayworld_blocks(run_id, s):
    """[(block key, target, basis)]: one block per regression basis, then one per extra target
    (keyed by the target, its probes fitted in the instance's first basis)."""
    bases = rw_bases_for(run_config(run_id)["data"]["instance"], s)
    blocks = [(basis, s["rw_target"], basis) for basis in bases]
    blocks += [(t, t, bases[0]) for t in extra_targets_of(run_id, s)]
    return blocks


def rw_probe_recipe(target, instance, s) -> dict:
    """``fit_probes`` keywords for a Rayworld target at the SETTINGS sizes: ``rw_probe_seqs``
    sequences for a regression target, ``rw_cat_probe_seqs`` for ``rw_cat_probe_epochs`` epochs
    for a categorical one."""
    return rwa.probe_recipe(target, instance, n_seq=s["rw_probe_seqs"], cat_n_seq=s["rw_cat_probe_seqs"],
                            cat_epochs=s["rw_cat_probe_epochs"])


def rw_block_setup(target, s):
    """(categorical target or None, dim sets, (PI alphas, GS alphas)) for a Rayworld target."""
    cat = categorical_target(target)
    if cat is not None:
        return cat, ("all",), (s["rw_grid_alpha_pi"], s["rw_grid_alpha_gs"])
    snap = snapped_target(target)
    if snap is not None and snap.base == "pos":
        return None, ("all",), (s["rw_alpha_pi"], s["rw_alpha_gs"])
    return None, s["rw_edit_dims"], (s["rw_alpha_pi"], s["rw_alpha_gs"])


def in_scope(scope, instance: str, target: str) -> bool:
    """Whether a SETTINGS scope ``{"instances": (...), "targets": (...)}`` lists both this Rayworld
    instance and this target."""
    c = scope or {}
    return f"rayworld/{instance}" in c.get("instances", ()) and target in c.get("targets", ())


def cat_inverse_in_scope(instance: str, target: str, s) -> bool:
    """Whether a categorical Rayworld block gets an IM arm (SETTINGS ``rw_cat_im``); elsewhere a
    categorical block carries none."""
    return in_scope(s.get("rw_cat_im"), instance, target)


def attach_inverse(blocks: dict, arms_by_key: dict, stats: dict, ei_key: str = "edit_index") -> None:
    """Replace each block's IM / IM-NN arms with ``arms_by_key``, set their ``best`` entries and
    record the inverse map's fit (``nn_r2`` only where a retrieval bank gave a finite one)."""
    from pim.probes.inverse import INVERSE_EPOCHS, INVERSE_HIDDEN, RETRIEVAL_K
    for key, recs in arms_by_key.items():
        blk = blocks[key]
        blk["arms"] = [r for r in blk["arms"] if r["editor"] not in ("IM", "IM-NN")]
        blk["arms"] += [{k: v for k, v in r.items() if np.isscalar(v)} for r in recs]
        for ed in ("IM", "IM-NN"):
            sub = [r for r in blk["arms"] if r["editor"] == ed]
            blk["best"][ed] = top_arm(sub, ei_key) if sub else None
            for d in blk.get("best_by_dims", {}):
                blk["best_by_dims"][d][ed] = blk["best"][ed]
        blk["inverse_map"] = {"g_r2": stats["g_r2"], "g_rmse": stats["g_rmse"], "hidden": INVERSE_HIDDEN,
                              "epochs": INVERSE_EPOCHS, "k": RETRIEVAL_K, "version": IM_VERSION}
        nn = stats.get("nn_r2")
        if nn is not None and np.isfinite(np.asarray(nn, dtype=float)).all():
            blk["inverse_map"]["nn_r2"] = nn


def probe_block(lin, mlp, sanity, u, arms, dimsets, *, target, basis, kind, n_classes, recipe,
                alphas, selection, ei_key="edit_index", extra=None):
    """One probe-target block: the target, Probe Skill per residual point, the MLP-vs-linear
    sanity report, the unedited scorecard, the top arm per editor (and per dim set) and every arm."""
    def pick(ed, dims=None):
        # "PI" owns "PI[zspace]" and "GS" owns "GS@L0", but "IM" must not own "IM-NN"
        sub = [r for r in arms if (r["editor"] == ed or r["editor"].startswith(ed + "[") or r["editor"].startswith(ed + "@"))
               and (dims is None or r.get("dims", "all") == dims)]
        return top_arm(sub, ei_key) if sub else None
    return {
        "target": target, "basis": basis, "kind": kind, "n_classes": n_classes,
        "probe_recipe": dict(recipe or {}),
        "alphas": {"PI": list(alphas[0]), "GS": list(alphas[1])},
        "bench_selection": selection,                 # None = the first n cases
        "unedited": {k: v for k, v in u.items() if np.isscalar(v)},
        "probe_skill_linear": [probe_skill_from_stats(st) for st in by_point(lin)],
        "probe_skill_mlp": [probe_skill_from_stats(st) for st in by_point(mlp)],
        "probe_perdim_linear": [st.get("per_dim_r2") for st in by_point(lin)],
        "probe_perdim_mlp": [st.get("per_dim_r2") for st in by_point(mlp)],
        "probe_sanity": sanity,
        "best": {ed: pick(ed) for ed in EDITORS_SCORED},
        "best_by_dims": {d: {ed: pick(ed, d) for ed in EDITORS_SCORED} for d in dimsets},
        "arms": [{k: v for k, v in r.items() if np.isscalar(v)} for r in arms],
        **(extra or {}),
    }
