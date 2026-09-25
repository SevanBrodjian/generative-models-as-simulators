"""The Rayworld editability bench: the edit cases, their probe targets and ray zones.

The cases are mid-sequence teleports from the instance's edits split, warmed into a model
state; the zones come from ``pim.metrics.zone_editability``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.environments import layout
from pim.environments.rayworld.grid_target import (
    CategoricalTarget, FactorizedTarget, categorical_target, selection_target, snapped_target)
from pim.metrics.zone_editability import build_edit_zones

N_OBJ, EF, K_ROLL, SEED = 2, 20, 15, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_REPO = layout.REPO


def _instance(instance: str | None) -> str:
    return layout.DEFAULT_INSTANCE["rayworld"] if instance is None else instance


def _edit_set(instance: str | None) -> tuple[Path, Path]:
    """(edits.h5, selection.json) of an instance (None: the default instance)."""
    inst = _instance(instance)
    return layout.edits_file("rayworld", inst), layout.edits_selection("rayworld", inst)


@dataclass
class Bench:
    """The edits split, warmed into a model state, with everything an arm is scored against."""

    obs: np.ndarray  # (N, T, R) observations
    gt_roll: np.ndarray  # (N, K, R) clean post-edit ground truth from the edit frame
    zones: object  # target / ghost / differing ray masks
    tgt: torch.Tensor  # (N, d_out) the probe target the edit asks for
    change_mask: torch.Tensor  # (N, d_out) bool, the edited object's dims only
    out_dims: list[int]  # the read-out rows the edit is allowed to move
    state: object  # the model state warmed on obs[:, :EF]
    n: int
    pos: np.ndarray | None = None          # (N, T, N_OBJ, 2) post-edit positions
    vel: np.ndarray | None = None          # (N, T, N_OBJ, 2)
    edit_object: np.ndarray | None = None  # (N,)
    sim: dict | None = None                # the instance's sim config
    kind: str = "regression"               # or "classification": tgt holds long labels
    cells: dict | None = None              # categorical move per case: {"A", "B", "cls"}
    moves: dict | None = None              # factorized move: {"tile", "old", "new"}, (N, F)
    selection: dict | None = None          # which cases were scored, when not the first n


def _to_basis(pos, vel, sim, basis_name):
    """World (x, y[, vx, vy]) to the named basis; None or "cartesian" is a no-op."""
    if basis_name in (None, "cartesian"):
        return pos, vel
    from pim.environments.rayworld.frustum import basis as fb

    return fb(pos, vel, sim, depth=basis_name)


def selection_path(instance: str | None = None) -> Path:
    """An instance's filtered case list (``selection.json`` beside its edits split, written by
    ``scripts/make_edit_selection.py``), shared by the frame and the token bench."""
    return _edit_set(instance)[1]


def grid_selection(n: int, grid: CategoricalTarget, instance: str | None = None
                   ) -> tuple[np.ndarray, dict]:
    """A categorical target's cases: the first ``n`` whose teleport changes cell (one that stays
    inside a cell asks for no change). Returns the indices and a record for ``scores.json``."""
    with h5py.File(_edit_set(instance)[0], "r") as f:
        pos = f["positions"][:, EF - 1: EF + 1, :N_OBJ, :].astype(np.float32)   # (M, 2, N_OBJ, 2)
        vel = f["velocities"][:, EF - 1, :N_OBJ, :].astype(np.float32)          # (M, N_OBJ, 2)
        eobj = f["edit_object"][:].astype(int)
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    # frame 1 of the slice becomes the pre-dynamics target state (see bench_from_arrays)
    ar = np.arange(len(pos))
    pre_dyn = pos[:, 0].copy()
    pre_dyn[ar, eobj] = pos[ar, 1, eobj] - vel[ar, eobj] * float(sim["dt"])
    pos[:, 1] = pre_dyn
    mv = grid.edit_cells(pos, eobj, ef=1, sim=sim)
    valid = np.where(mv["A"] != mv["B"])[0]
    sel = valid[:n]
    scanned = int(sel[-1]) + 1 if len(sel) else 0
    return sel, {"rule": f"first {n} cases whose teleport changes a {grid.name} cell "
                         f"(pre-dynamics target state)",
                 "n": int(len(sel)), "scanned": scanned,
                 "dropped_same_cell": int(scanned - len(sel))}


def bench_from_arrays(obs: np.ndarray, pos: np.ndarray, vel: np.ndarray, eobj: np.ndarray,
                      clean: np.ndarray, sim: dict, blink: np.ndarray | None, *, target: str,
                      basis_name: str, selection: dict | None = None, edit_frame: int = EF) -> dict:
    """The bench dict from arrays (targets, change masks, zones): ``obs`` / ``clean`` (n, T, R)
    fed and clean frames, ``pos`` / ``vel`` (n, T, N_OBJ, 2) post-edit, ``eobj`` (n,), ``sim``
    the sim dict, ``blink`` (n, T, N_OBJ) bool or None."""
    grid, snap = categorical_target(target), snapped_target(target)
    n = obs.shape[0]
    gt_roll = clean[:, EF: EF + K_ROLL, :]
    zones = build_edit_zones(pre_pos=pos[:, EF - 1], tgt_pos=pos[:, EF],
                             pre_vel=vel[:, EF - 1], edit_object=eobj, sim=sim,
                             n_obj=N_OBJ, traj_pos=pos[:, EF: EF + K_ROLL],
                             gt_edited_traj=gt_roll,
                             blink_visible=None if blink is None
                             else blink[:, EF - 1: EF + K_ROLL + 1])
    # write target = the pre-dynamics state: pos[EF] - v * dt for the edited object, else frame EF - 1
    dt = float(sim["dt"])
    ar1 = np.arange(n)
    pre_dyn = pos[:, EF - 1].copy()                                     # (n, N_OBJ, 2)
    pre_dyn[ar1, eobj] = pos[ar1, EF, eobj] - vel[ar1, EF - 1, eobj] * dt
    pos_t = pos.copy()
    pos_t[:, EF] = pre_dyn            # frame EF read as the pre-dynamics target state below
    cells = moves = None
    if isinstance(grid, FactorizedTarget):
        # only the edited object's tiles whose class changes are asked to move
        cur, _ = grid.label_frames(pos[:, EF - 1], sim)                  # (n, tiles) int64
        moves = grid.edit_moves(pos_t, eobj, EF, sim)
        ar = np.arange(n)[:, None]
        y = cur.copy()
        y[ar, moves["tile"]] = moves["new"]
        cm = np.zeros((n, grid.n_tiles_on(sim)), bool)
        cm[ar, moves["tile"]] = moves["new"] != moves["old"]
        out_dims = []
    elif grid is not None:
        # the object leaves cell A and appears in cell B; only those two cells change
        cur, _ = grid.label_frames(pos[:, EF - 1], sim)                  # (n, G) uint8
        cells = grid.edit_cells(pos_t, eobj, EF, sim)                    # B = the pre-dynamics cell
        ar = np.arange(n)
        y = cur.astype(np.int64)
        y[ar, cells["A"]] = 0
        y[ar, cells["B"]] = cells["cls"]
        cm = np.zeros((n, grid.n_cells(sim)), bool)
        cm[ar, cells["A"]] = True
        cm[ar, cells["B"]] = True
        out_dims = []            # per-case rows, not a shared set
    else:
        bp, bv = _to_basis(pre_dyn, vel[:, EF - 1], sim, basis_name)
        base = target
        if snap is not None:
            # the edit asks for the center of the new cell, in frustum coordinates
            if basis_name != "frustum":
                raise ValueError(f"{target} is defined in the frustum basis, got {basis_name!r}")
            bp, base = snap.snap(pre_dyn, sim).astype(np.float32), snap.base
        y = bp.reshape(n, -1)
        if base == "full":
            y = np.concatenate([y, bv.reshape(n, -1)], axis=1)
        # the change mask marks the edited object's dims only; every other dim is held
        d_out = y.shape[1]
        cm = np.zeros((n, d_out), bool)
        cm[np.arange(n), 2 * eobj] = True
        cm[np.arange(n), 2 * eobj + 1] = True
        if base == "full":
            cm[np.arange(n), 2 * N_OBJ + 2 * eobj] = True
            cm[np.arange(n), 2 * N_OBJ + 2 * eobj + 1] = True
        out_dims = sorted({int(i) for i in np.where(cm.any(0))[0]})
    assert int(edit_frame) == EF, f"edit_frame {edit_frame}: the bench assumes {EF}"
    return dict(obs=obs, pos=pos, vel=vel, edit_object=eobj, clean=clean, sim=sim,
                gt_roll=gt_roll, zones=zones, y=y, change_mask=cm, out_dims=out_dims, n=n,
                blink_visible=blink, kind="classification" if grid else "regression",
                cells=cells, moves=moves, selection=selection)


def bench_arrays(n: int = 192, target: str = "pos", basis_name: str = "cartesian", *,
                 select: np.ndarray | None = None, use_selection: bool = True,
                 instance: str | None = None) -> dict:
    """The edit cases' arrays and zones, model-free. Cases: ``select`` if given, else
    ``grid_selection`` for a categorical or snapped target, else the instance's
    ``selection.json`` (when ``use_selection``), else the first ``n``."""
    from pim.environments.rayworld.loading import load_edits

    edits_h5, _sp = _edit_set(instance)
    selection = None
    sel_target = selection_target(target)      # the partition, or a snapped target's partition
    if sel_target is not None and select is None:
        select, selection = grid_selection(n, sel_target, instance=instance)
    if select is None and use_selection:
        if _sp.exists():
            _sel = json.loads(_sp.read_text())
            select = np.asarray(_sel["select"], dtype=int)[:n]
            selection = {"file": str(_sp.relative_to(_REPO)) if _sp.is_relative_to(_REPO) else _sp.name,
                         "rule": _sel.get("rule"), "n": int(len(select)),
                         **{k: _sel[k] for k in ("min_rays", "pool", "stats") if k in _sel}}
    b = load_edits(edits_h5, n_obj_keep=N_OBJ)
    sl = slice(None, n) if select is None else np.asarray(select, dtype=int)
    obs = b.obs[sl].astype(np.float32)
    pos = b.positions[sl, :, :N_OBJ, :].astype(np.float32)
    eobj = b.edit_object[sl].astype(int)
    clean = b.clean_obs[sl].astype(np.float32)
    n = obs.shape[0]
    with h5py.File(b.h5_path, "r") as f:
        vel = f["velocities"][:, :, :N_OBJ, :].astype(np.float32)[sl]
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    blink = None if b.blink_visible is None else b.blink_visible[sl]
    return bench_from_arrays(obs, pos, vel, eobj, clean, sim, blink, target=target,
                             basis_name=basis_name, selection=selection,
                             edit_frame=int(getattr(b, "edit_frame", EF)))


def load_bench(model, n: int = 192, target: str = "pos", basis_name: str = "cartesian", *,
               select: np.ndarray | None = None, use_selection: bool = True,
               instance: str | None = None) -> Bench:
    """Warm ``model`` on the edits split and build the ground-truth zones (``bench_arrays``)."""
    a = bench_arrays(n, target, basis_name, select=select, use_selection=use_selection,
                     instance=instance)
    return bench_of(model, a)


def bench_of(model, a: dict) -> Bench:
    """Warm ``model`` on a bench dict's frames (``bench_arrays`` / ``bench_from_arrays``)."""
    state = model.state_from_obs(torch.from_numpy(a["obs"][:, :EF]).float().to(DEV))
    tgt = torch.from_numpy(a["y"]).to(DEV)
    tgt = tgt.long() if a["kind"] == "classification" else tgt.float()
    _long = lambda d: (None if d is None                                   # noqa: E731
                       else {k: torch.from_numpy(v).long().to(DEV) for k, v in d.items()})
    return Bench(a["obs"], a["gt_roll"], a["zones"], tgt,
                 torch.from_numpy(a["change_mask"]).to(DEV), a["out_dims"], state, a["n"],
                 pos=a["pos"], vel=a["vel"], edit_object=a["edit_object"], sim=a["sim"],
                 kind=a["kind"], cells=_long(a["cells"]), moves=_long(a["moves"]),
                 selection=a["selection"])


# the read-outs an edit drives; every edit writes the full state
DIM_SETS: dict[str, tuple[int, ...] | None] = {
    "all": None,                      # every read-out the probe has
}


def dim_idx(dims: str):
    """Name to read-out indices (None = all of them)."""
    if dims not in DIM_SETS:
        raise KeyError(f"dims must be one of {sorted(DIM_SETS)}, got {dims!r}")
    return DIM_SETS[dims]


def full_state_pair(pos: np.ndarray, vel: np.ndarray, edit_object: np.ndarray, sim: dict,
                    basis_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The full state in ``basis_name`` before the edit (frame EF - 1) and the pre-dynamics
    target state after it, per case: two (N, 4 * N_OBJ) arrays, the inverse map's inputs."""
    n = len(pos)
    ar = np.arange(n)
    eobj = np.asarray(edit_object, int)
    pre_dyn = pos[:, EF - 1].copy()
    pre_dyn[ar, eobj] = pos[ar, EF, eobj] - vel[ar, EF - 1, eobj] * float(sim["dt"])
    bp0, bv0 = _to_basis(pos[:, EF - 1], vel[:, EF - 1], sim, basis_name)
    bp1, bv1 = _to_basis(pre_dyn, vel[:, EF - 1], sim, basis_name)
    s_pre = np.concatenate([bp0.reshape(n, -1), bv0.reshape(n, -1)], 1).astype(np.float32)
    s_post = np.concatenate([bp1.reshape(n, -1), bv1.reshape(n, -1)], 1).astype(np.float32)
    return s_pre, s_post
