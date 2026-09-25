"""Blink: discs leave the observation (not the physics) for a run of frames. One disc is hidden at a
time, and a disc cannot start a new blackout on the frame it reappears. The frame before a blackout
and its last frame carry ``MARK_VALUE`` on an edge ray (ray 0 for disc 0, the last ray for disc 1),
stored in ``obs_id`` as ``marker_id(j)``."""

from __future__ import annotations

import numpy as np

from .config import SimConfig

MARK_VALUE = 0.5


def blink_enabled(cfg: SimConfig) -> bool:
    return float(getattr(cfg, "blink_prob", 0.0)) > 0.0


def marker_id(j: int) -> int:
    """The ``obs_id`` code of object ``j``'s marker ray (-2 for object 0, -3 for object 1)."""
    return -2 - int(j)


def marker_ray(j: int, n_rays: int) -> int:
    """Object 0 signals on the leftmost ray, object 1 on the rightmost."""
    return 0 if j == 0 else n_rays - 1


def blink_schedule(cfg: SimConfig, n_obj: int) -> np.ndarray | None:
    """(n_frames, n_obj) bool visibility, or None when blinking is off. Deterministic in
    ``cfg.seed`` (stream seed + 7); one object hidden at a time, none before ``blink_warmup``."""
    if not blink_enabled(cfg):
        return None
    if n_obj > 2:
        raise ValueError("blink markers are defined for at most 2 objects (one edge ray each)")
    rng = np.random.default_rng(int(cfg.seed) + 7)
    F = int(cfg.n_frames)
    vis = np.ones((F, n_obj), dtype=bool)
    remaining = np.zeros(n_obj, dtype=int)          # hidden frames still to serve
    p, mean, cap, warm = (float(cfg.blink_prob), float(cfg.blink_mean),
                          int(cfg.blink_max), int(cfg.blink_warmup))
    for t in range(F):
        for j in range(n_obj):                       # first serve the running blackouts
            if remaining[j] > 0:
                vis[t, j] = False
                remaining[j] -= 1
        for j in range(n_obj):                       # then consider new ones
            if t < max(warm, 1) or not vis[t - 1, j]:   # warm-up, or just reappeared
                continue
            if not vis[t].all():                     # some object is hidden this frame
                continue
            if rng.random() < p:
                length = min(cap, int(rng.geometric(1.0 / mean)))
                vis[t, j] = False
                remaining[j] = length - 1
    return vis


def paint_markers(hit_id: np.ndarray, obs_intensity: np.ndarray,
                  vis_now: np.ndarray, vis_next: np.ndarray | None) -> None:
    """In place: mark object ``j``'s edge ray when its visibility changes between this frame
    and the next (no next frame, no marker)."""
    if vis_next is None:
        return
    R = obs_intensity.shape[0]
    for j in range(vis_now.shape[0]):
        if bool(vis_now[j]) != bool(vis_next[j]):
            r = marker_ray(j, R)
            obs_intensity[r] = MARK_VALUE
            hit_id[r] = marker_id(j)
