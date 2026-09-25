"""Categorical probe targets for Rayworld: the state read as labeled cells, Othello-style.

``grid-<nu>x<nd>``: a grid in (u, 1/y); ``appearance``: positions lighting the same run of rays;
``appearance-fac``: each disc's appearance bin as a center label and a length label (15 and 5
classes on 8-ray); ``pos@<partition>``: positions snapped to cell centers."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import numpy as np

from pim.environments.rayworld.frustum import lateral

N_CLASSES = 3                  # empty / object 0 / object 1
N_OBJ = 2
_GRID = re.compile(r"^grid-(\d+)x(\d+)$")
_APP = re.compile(r"^appearance$")
_FAC = re.compile(r"^(.+)-fac$")
_SNAP = re.compile(r"^pos@(.+)$")


def _require_single_observer(sim: dict) -> None:
    if int(sim.get("n_observers", 1)) > 1:
        raise ValueError("this categorical target is defined for single-observer instances only")


class CategoricalTarget:
    """A cell-indexed target: each cell is labeled {0 empty, 1 object 0, 2 object 1}, the nearer
    object winning a shared cell. ``cell_of`` is the family-specific part."""

    n_classes = N_CLASSES

    @property
    def name(self) -> str:                       # pragma: no cover
        raise NotImplementedError

    def n_cells(self, sim: dict) -> int:         # pragma: no cover
        raise NotImplementedError

    def n_tiles_on(self, sim: dict) -> int:
        """Width of the label vector the probe reads (the cells, for a cell-indexed target)."""
        return self.n_cells(sim)

    def n_classes_on(self, sim: dict) -> int:
        """Classes per tile on this instance."""
        return self.n_classes

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:   # pragma: no cover
        """(..., 2) world positions to (...) int cell indices in [0, n_cells)."""
        raise NotImplementedError

    def labels_from_cells(self, cells: np.ndarray, y_world: np.ndarray, sim: dict) -> np.ndarray:
        """(..., N_OBJ) cells and depths to (..., G) uint8 labels, the nearer object winning."""
        _require_single_observer(sim)
        g = self.n_cells(sim)
        lead = cells.shape[:-1]
        lab = np.zeros(lead + (g,), dtype=np.uint8)
        fl = lab.reshape(-1, g)
        fc, fy = cells.reshape(-1, N_OBJ), y_world.reshape(-1, N_OBJ)
        order = np.argsort(-fy, axis=1)                    # far first, so near overwrites
        rows = np.arange(fl.shape[0])
        for k in range(N_OBJ):
            j = order[:, k]
            fl[rows, fc[rows, j]] = (j + 1).astype(np.uint8)
        return lab

    def label_frames(self, pos: np.ndarray, sim: dict) -> tuple[np.ndarray, int]:
        """(..., N_OBJ, 2) positions to (..., G) uint8 labels, plus the number of frames in
        which two objects share a cell."""
        cells = self.cell_of(pos, sim)                     # (..., N_OBJ)
        fc = cells.reshape(-1, N_OBJ)
        conflicts = int((fc[:, 0] == fc[:, 1]).sum())
        return self.labels_from_cells(cells, pos[..., 1], sim), conflicts

    def edit_cells(self, pos: np.ndarray, edit_object: np.ndarray, ef: int,
                   sim: dict) -> dict:
        """Per case, the move a teleport asks for: ``A`` the edited object's cell at frame
        ``ef - 1``, ``B`` its cell at ``ef``, ``cls`` its class (object + 1); (N,) int64 each.
        ``pos`` is (N, T, N_OBJ, 2)."""
        idx = np.arange(len(edit_object))
        j = np.asarray(edit_object, dtype=int)
        return {"A": self.cell_of(pos[idx, ef - 1, j], sim),
                "B": self.cell_of(pos[idx, ef, j], sim),
                "cls": (j + 1).astype(np.int64)}

    def centroids(self, sim: dict) -> np.ndarray:
        """(cells, 2) frustum-basis center of every cell (cached per geometry)."""
        return _centroids_of(self, _numeric_sim_key(sim))

    def snap(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        """(..., 2) world positions to (..., 2) frustum coordinates of their cell's center."""
        return self.centroids(sim)[self.cell_of(pos, sim)]


# ── the product grid ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GridTarget(CategoricalTarget):
    """``nu`` lateral x ``nd`` depth cells, named ``grid-<nu>x<nd>``."""

    nu: int = 16
    nd: int = 8

    @property
    def name(self) -> str:
        return f"grid-{self.nu}x{self.nd}"

    @property
    def g(self) -> int:
        """Number of cells (independent of the instance)."""
        return self.nu * self.nd

    def n_cells(self, sim: dict) -> int:
        return self.g

    @classmethod
    def parse(cls, target: str) -> "GridTarget | None":
        """``"grid-16x8"`` to ``GridTarget(16, 8)``; any other name to None."""
        m = _GRID.match(str(target))
        return cls(int(m.group(1)), int(m.group(2))) if m else None

    def edges(self, sim: dict) -> tuple[np.ndarray, np.ndarray]:
        """Bin edges in (normalized u, 1/y) over the reachable region."""
        r = float(sim["radius"])
        u = np.linspace(-1.0, 1.0, self.nu + 1)
        d = np.linspace(1.0 / (float(sim["y_far"]) - r), 1.0 / (float(sim["y_near"]) + r),
                        self.nd + 1)
        return u, d

    @staticmethod
    def lateral_norm(pos: np.ndarray, sim: dict) -> np.ndarray:
        """u' = u / (1 - r / (scale * y)): the ray coordinate rescaled by a center's reach at
        that depth, so u' = +-1 is a disc touching the frustum wall."""
        scale = float(sim["x_far"]) / float(sim["y_far"])
        r = float(sim["radius"])
        reach = 1.0 - r / (scale * np.maximum(pos[..., 1], 1e-6))
        return lateral(pos, sim) / np.maximum(reach, 1e-6)

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        ue, de = self.edges(sim)
        u = self.lateral_norm(pos, sim)
        inv_y = 1.0 / np.maximum(pos[..., 1], 1e-6)
        iu = np.clip(np.searchsorted(ue, u, side="right") - 1, 0, self.nu - 1)
        idp = np.clip(np.searchsorted(de, inv_y, side="right") - 1, 0, self.nd - 1)
        return (iu * self.nd + idp).astype(np.int64)


# ── the appearance partition ──────────────────────────────────────────────────


def covered_rays(pos: np.ndarray, sim: dict) -> np.ndarray:
    """(..., 2) single-disc centers to (..., R_kept) bool: which kept rays that disc lights,
    by the renderer's own ray-disc test (discs inside the frustum never need the near clamp)."""
    p = np.asarray(pos)
    R = int(sim["obs_res"])
    scale = float(sim["x_far"]) / float(sim["y_far"])
    s = np.linspace(-1.0, 1.0, R)
    dx, dy = s * scale, np.ones(R)
    nrm = np.hypot(dx, dy)
    dx, dy = dx / nrm, dy / nrm
    cx, cy = p[..., 0:1], p[..., 1:2]                       # (..., 1), the caller's dtype
    b = dx * cx + dy * cy                                   # (..., R) float64
    # |c|^2 - r^2 in the positions' dtype, as the renderer computes it, or grazing rays flip
    r = np.asarray(sim["radius"], dtype=p.dtype if np.issubdtype(p.dtype, np.floating) else np.float64)
    C = (cx ** 2 + cy ** 2 - r ** 2).astype(np.float64)
    disc = b ** 2 - C
    t_front = b - np.sqrt(np.maximum(disc, 0.0))
    y_front = dy * t_front
    hit = (disc >= 0) & (t_front > 1e-9) & (y_front >= float(sim["y_near"])) & (y_front <= float(sim["y_far"]))
    return hit[..., 1:-1] if sim.get("drop_edge_rays", False) else hit


def _sim_key(sim: dict) -> tuple:
    return tuple(float(sim[k]) for k in ("radius", "y_near", "y_far", "x_far")) + \
        (int(sim["obs_res"]), bool(sim.get("drop_edge_rays", False)))


@functools.lru_cache(maxsize=None)
def _runs_of(sim_key: tuple, n_side: int = 500):
    """A deterministic dense sweep of the reachable region: the realizable (first, last) runs
    in cell order, and the number of kept rays. Cached per geometry."""
    r, y_near, y_far, x_far, R, drop = sim_key
    sim = {"radius": r, "y_near": y_near, "y_far": y_far, "x_far": x_far, "obs_res": R,
           "drop_edge_rays": drop}
    scale = x_far / y_far
    ys = np.linspace(y_near + r, y_far - r, n_side)
    xs = np.linspace(-1.0, 1.0, n_side)
    Y = np.repeat(ys[:, None], n_side, 1)
    X = xs[None, :] * (scale * Y - r)                       # inside the reach at each depth
    hit = covered_rays(np.stack([X, Y], -1), sim)          # (n, n, Rk)
    first = hit.argmax(-1)
    last = hit.shape[-1] - 1 - hit[..., ::-1].argmax(-1)
    any_ = hit.any(-1)
    assert any_.all(), "a reachable position lights no ray — the appearance partition needs r large enough"
    assert (hit.sum(-1) == last - first + 1).all(), "non-contiguous run"
    code = first * hit.shape[-1] + last
    runs = [(int(c // hit.shape[-1]), int(c % hit.shape[-1])) for c in np.unique(code)]
    runs.sort(key=lambda fl: (fl[1] - fl[0], fl[0]))        # short runs (far) first, then left to right
    return tuple(runs), hit.shape[-1]


def _nearest_run(runs: tuple, f: int, la: int) -> int:
    """Index of the realizable run nearest to (f, la): same center first, then the smallest
    |df| + |dl|, ties to the earlier run in cell order."""
    best, key = 0, None
    for i, (rf, rl) in enumerate(runs):
        k = (abs((rf + rl) - (f + la)), abs(rf - f) + abs(rl - la))
        if key is None or k < key:
            best, key = i, k
    return best


@dataclass(frozen=True)
class AppearanceTarget(CategoricalTarget):
    """The observation-exact partition: a cell is the run of rays a single disc lights."""

    factor_names = ("center", "length")

    @property
    def name(self) -> str:
        return "appearance"

    @classmethod
    def parse(cls, target: str) -> "AppearanceTarget | None":
        return cls() if _APP.match(str(target)) else None

    def runs(self, sim: dict) -> tuple:
        """The realizable (first, last) runs on this instance, in cell order."""
        _require_single_observer(sim)
        return _runs_of(_sim_key(sim))[0]

    def n_cells(self, sim: dict) -> int:
        return len(self.runs(sim))

    # positions per chunk of the ray-disc test: bounds the (..., R) float64 intermediates
    CHUNK = 1 << 18

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        _require_single_observer(sim)
        p = np.asarray(pos, np.float64)
        flat = p.reshape(-1, 2)
        if flat.shape[0] <= self.CHUNK:
            return self._cell_of_flat(flat, sim).reshape(p.shape[:-1])
        out = np.concatenate([self._cell_of_flat(flat[i: i + self.CHUNK], sim)
                              for i in range(0, flat.shape[0], self.CHUNK)])
        return out.reshape(p.shape[:-1])

    def _cell_of_flat(self, p: np.ndarray, sim: dict) -> np.ndarray:
        runs, rk = _runs_of(_sim_key(sim))
        hit = covered_rays(p, sim)
        first = hit.argmax(-1)
        last = rk - 1 - hit[..., ::-1].argmax(-1)
        code = first * rk + last
        table = np.full(rk * rk, -1, np.int64)
        for i, (f, la) in enumerate(runs):
            table[f * rk + la] = i
        run_idx = table[code]
        if (run_idx < 0).any():
            # a run the sweep did not see (float32 rounding): snap to the nearest realizable run
            miss = run_idx < 0
            if not hit[miss].any(-1).all():
                raise ValueError("a position lights no ray — outside the reachable region")
            snapped = np.array([_nearest_run(runs, int(c // rk), int(c % rk))
                                for c in np.unique(code[miss])])
            lut = dict(zip(np.unique(code[miss]).tolist(), snapped.tolist()))
            run_idx = run_idx.copy()
            run_idx[miss] = [lut[int(c)] for c in code[miss]]
        return run_idx

    def _factor_tables(self, sim: dict) -> tuple[dict, dict]:
        """{first + last: class}, {length: class} over the realizable runs (sorted)."""
        runs = self.runs(sim)
        centers = sorted({f + la for f, la in runs})
        lengths = sorted({la - f + 1 for f, la in runs})
        return ({c: i for i, c in enumerate(centers)}, {ln: i for i, ln in enumerate(lengths)})

    def factors_of(self, cells: np.ndarray, sim: dict) -> np.ndarray:
        """(...) cells to (..., 2) classes: (center class, length class)."""
        runs = np.asarray(self.runs(sim))                  # (G, 2)
        ct, lt = self._factor_tables(sim)
        c_cls = np.array([ct[f + la] for f, la in runs])
        l_cls = np.array([lt[la - f + 1] for f, la in runs])
        cells = np.asarray(cells, np.int64)
        return np.stack([c_cls[cells], l_cls[cells]], -1)

    def factor_sizes(self, sim: dict) -> tuple[int, ...]:
        ct, lt = self._factor_tables(sim)
        return (len(ct), len(lt))


# ── the factorized target: tile F*j + f is object j's factor f, over all factors' classes ──


@dataclass(frozen=True)
class FactorizedTarget(CategoricalTarget):
    """``appearance-fac``: the appearance cells read per object as categorical factors, each tile
    a softmax over every factor's classes laid side by side (center classes, then lengths)."""

    cat: CategoricalTarget = None

    @property
    def name(self) -> str:
        return f"{self.cat.name}-fac"

    @classmethod
    def parse(cls, target: str) -> "FactorizedTarget | None":
        m = _FAC.match(str(target))
        if not m:
            return None
        cat = AppearanceTarget.parse(m.group(1))
        return cls(cat) if cat is not None else None

    def n_cells(self, sim: dict) -> int:
        return self.cat.n_cells(sim)

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        return self.cat.cell_of(pos, sim)

    @property
    def n_factors(self) -> int:
        return len(self.cat.factor_names)

    @property
    def n_tiles(self) -> int:
        return N_OBJ * self.n_factors

    def n_factors_on(self, sim: dict) -> int:
        return len(self.cat.factor_sizes(sim))

    def n_tiles_on(self, sim: dict) -> int:
        return N_OBJ * self.n_factors_on(sim)

    def class_offsets(self, sim: dict) -> np.ndarray:
        """Where each factor's classes start on the shared class axis."""
        return np.concatenate([[0], np.cumsum(self.cat.factor_sizes(sim))[:-1]]).astype(np.int64)

    def n_classes_on(self, sim: dict) -> int:
        return int(sum(self.cat.factor_sizes(sim)))

    def factor_labels(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        """(..., N_OBJ, 2) positions to (..., N_OBJ * F) labels on the shared class axis."""
        cells = self.cat.cell_of(pos, sim)                          # (..., N_OBJ)
        fac = self.cat.factors_of(cells, sim) + self.class_offsets(sim)   # (..., N_OBJ, F)
        return fac.reshape(*fac.shape[:-2], -1).astype(np.int64)

    def label_frames(self, pos: np.ndarray, sim: dict) -> tuple[np.ndarray, int]:
        """Labels are per object, so two objects sharing a cell is not a conflict."""
        return self.factor_labels(pos, sim), 0

    def labels_from_cells(self, cells, y_world, sim):        # pragma: no cover
        raise NotImplementedError("a factorized target labels objects, not cells")

    def edit_cells(self, pos, edit_object, ef, sim):        # pragma: no cover
        raise NotImplementedError("a factorized target's edit is `edit_moves`, not a cell pair")

    def edit_moves(self, pos: np.ndarray, edit_object: np.ndarray, ef: int, sim: dict) -> dict:
        """Per case, the edited object's tiles and their classes before (frame ``ef - 1``) and
        after (frame ``ef``) the teleport: ``{"tile", "old", "new"}``, (N, F) int64 each."""
        j = np.asarray(edit_object, dtype=int)
        idx = np.arange(len(j))
        F = self.n_factors_on(sim)
        before = self.factor_labels(pos[:, ef - 1], sim).reshape(len(j), N_OBJ, F)[idx, j]
        after = self.factor_labels(pos[:, ef], sim).reshape(len(j), N_OBJ, F)[idx, j]
        tile = (j[:, None] * F + np.arange(F)[None, :]).astype(np.int64)
        return {"tile": tile, "old": before.astype(np.int64), "new": after.astype(np.int64)}


def categorical_target(target: str) -> "CategoricalTarget | None":
    """The categorical target a name denotes, or None for a regression target
    (``pos`` / ``full`` / ``pos@...``). Every branch in the pipeline goes through this."""
    return GridTarget.parse(target) or AppearanceTarget.parse(target) or FactorizedTarget.parse(target)


# ── the snapped regression target ─────────────────────────────────────────────


def frustum_to_world(f: np.ndarray, sim: dict) -> np.ndarray:
    """(..., 2) frustum coordinates (u, 1/y) to (..., 2) world (x, y)."""
    from pim.environments.rayworld.frustum import CANONICAL_DEPTH, fov_scale

    assert CANONICAL_DEPTH == "inv_y", "frustum_to_world inverts the inverse-depth axis only"
    f = np.asarray(f, np.float64)
    y = 1.0 / np.maximum(f[..., 1], 1e-9)
    x = f[..., 0] * fov_scale(sim) * y
    return np.stack([x, y], -1)


def _numeric_sim_key(sim: dict) -> tuple:
    return tuple((k, float(sim[k])) for k in sorted(sim) if isinstance(sim[k], (int, float, bool)))


@functools.lru_cache(maxsize=None)
def _centroids_of(target: "CategoricalTarget", sim_key: tuple, n_side: int = 500) -> np.ndarray:
    """(G, 2) cell centers in frustum coordinates: each cell's centroid under a uniform sweep of
    the reachable region, or its sweep point nearest the centroid when that falls outside."""
    from pim.environments.rayworld.frustum import basis as fb

    sim = dict(sim_key)
    r, y_near, y_far = float(sim["radius"]), float(sim["y_near"]), float(sim["y_far"])
    scale = float(sim["x_far"]) / y_far
    ys = np.linspace(y_near + r, y_far - r, n_side)
    xs = np.linspace(-1.0, 1.0, n_side)
    Y = np.repeat(ys[:, None], n_side, 1)
    X = xs[None, :] * (scale * Y - r)                        # the reachable region, uniformly
    P = np.stack([X, Y], -1).reshape(-1, 2)
    cells = target.cell_of(P, sim).reshape(-1)
    F = fb(P, None, sim, depth="frustum")[0]                 # (n, 2) frustum coordinates
    g = target.n_cells(sim)
    count = np.bincount(cells, minlength=g).astype(np.float64)
    if (count == 0).any():
        raise ValueError(f"{target.name}: cells {np.where(count == 0)[0].tolist()} hold no swept "
                         f"position — the sweep is too coarse for this partition")
    cen = np.stack([np.bincount(cells, F[:, k], minlength=g) for k in range(2)], -1) / count[:, None]
    back = target.cell_of(frustum_to_world(cen, sim), sim)
    for c in np.where(back != np.arange(g))[0]:              # non-convex cell: use its medoid
        idx = np.where(cells == c)[0]
        cen[c] = F[idx[np.argmin(((F[idx] - cen[c]) ** 2).sum(-1))]]
    return cen


@dataclass(frozen=True)
class SnappedTarget:
    """``pos@<categorical>``: positions snapped to their cell center under ``cat``, in the
    frustum basis. ``base`` is the underlying regression target (``"pos"``)."""

    base: str
    cat: CategoricalTarget

    @property
    def name(self) -> str:
        return f"{self.base}@{self.cat.name}"

    @classmethod
    def parse(cls, target: str) -> "SnappedTarget | None":
        m = _SNAP.match(str(target))
        if not m:
            return None
        cat = categorical_target(m.group(1))
        return cls("pos", cat) if cat is not None else None

    def n_cells(self, sim: dict) -> int:
        return self.cat.n_cells(sim)

    def snap(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        return self.cat.snap(pos, sim)


def snapped_target(target: str) -> "SnappedTarget | None":
    """The snapped regression target a name denotes (``pos@appearance``), or None."""
    return SnappedTarget.parse(target)


def selection_target(target: str) -> "CategoricalTarget | None":
    """The partition whose cell change defines a genuine edit under ``target``: the categorical
    target's partition, the snapped target's, or None for a plain regression target."""
    cat = categorical_target(target)
    if cat is not None:
        return cat.cat if isinstance(cat, FactorizedTarget) else cat
    sn = snapped_target(target)
    return sn.cat if sn is not None else None


def target_cells(target: str, sim: dict) -> int | None:
    """How many cells a target's partition has on this instance (None for ``pos`` / ``full``)."""
    sel = selection_target(target)
    return None if sel is None else sel.n_cells(sim)
