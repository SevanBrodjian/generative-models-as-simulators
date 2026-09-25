"""The 1D observation: analytic ray casting of the discs from an observer at the origin.

Rays are uniform in tan(angle) with half-FOV atan(x_far / y_far); each returns the reflectivity
of the first disc it hits."""

from __future__ import annotations

import numpy as np

from .blink import paint_markers
from .config import SimConfig, obs_dim
from .observers import render_frame_multi
from .sim import Scene
from .soft_render import render_frame_soft, soft_enabled


def _fov_scale(cfg: SimConfig) -> float:
    """tan(half-FOV): the horizontal spread of the ray fan."""
    return cfg.x_far / cfg.y_far


def _keep(hit_depth, hit_id, obs_intensity, cfg: SimConfig):
    """Drop the two wall-aligned rays when the config asks for it; the kept rays are
    bit-identical to rays 1..obs_res-2 of the full cast."""
    if getattr(cfg, "drop_edge_rays", False):
        return hit_depth[1:-1], hit_id[1:-1], obs_intensity[1:-1]
    return hit_depth, hit_id, obs_intensity


def render_frame(
    positions: np.ndarray,  # (n_objects, 2)
    radii: np.ndarray,  # (n_objects,)
    reflectivities: np.ndarray,  # (n_objects,)
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
    visible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame: ``hit_depth`` (0 on a miss), ``hit_id`` (-1 on a miss) and ``obs_intensity``
    (hit reflectivity plus optional noise, clipped to [0, 1]). ``visible`` (blink) removes
    hidden objects from the cast; ``hit_id`` keeps the original indices."""
    if visible is not None and not bool(np.all(visible)):
        keep = np.flatnonzero(np.asarray(visible, dtype=bool))
        d, ids, inten = render_frame(positions[keep], radii[keep], reflectivities[keep],
                                     cfg, rng=rng)
        ids = ids.copy()
        hit = ids >= 0
        ids[hit] = keep[ids[hit]]
        return d, ids, inten

    if int(getattr(cfg, "n_observers", 1)) > 1:
        return render_frame_multi(positions, radii, reflectivities, cfg, rng=rng)

    if soft_enabled(cfg):
        return render_frame_soft(positions, radii, reflectivities, cfg, rng=rng)

    R = cfg.obs_res
    n = len(radii)

    # unit ray directions: s uniform in [-1, 1], d = (s * scale, 1) / norm
    s = np.linspace(-1.0, 1.0, R)  # (R,)
    scale = _fov_scale(cfg)
    dx = s * scale  # (R,)
    dy = np.ones(R)
    norm = np.hypot(dx, dy)
    dx /= norm
    dy /= norm

    hit_depth = np.zeros(R)
    hit_id = np.full(R, -1, dtype=int)
    obs_intensity = np.zeros(R)

    if n == 0:
        return _keep(hit_depth, hit_id, obs_intensity, cfg)

    cx = positions[:, 0]  # (N,)
    cy = positions[:, 1]  # (N,)

    # ray-circle intersection over R rays x N objects: t = (d.c) - sqrt((d.c)^2 - (|c|^2 - r^2))
    b = dx[:, None] * cx[None, :] + dy[:, None] * cy[None, :]  # (R, N)
    C = cx**2 + cy**2 - radii**2  # (N,)
    disc = b**2 - C[None, :]  # (R, N)

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))  # (R, N)
    t_front = b - sqrt_disc  # (R, N) near intersection
    t_back = b + sqrt_disc  # (R, N) far intersection
    hit_y_front = dy[:, None] * t_front  # (R, N)
    hit_y_back = dy[:, None] * t_back  # (R, N)

    # the front surface lies inside the frustum
    valid_front = (
        (disc >= 0)
        & (t_front > 1e-9)
        & (hit_y_front >= cfg.y_near)
        & (hit_y_front <= cfg.y_far)
    )
    # a disc straddling y_near is hit at the near plane; one entirely in front of it is not
    t_at_near = cfg.y_near / dy[:, None]  # (R, N)
    clamp_to_near = (
        (disc >= 0) & (hit_y_front < cfg.y_near) & (hit_y_back >= cfg.y_near)
    )

    t_eff = np.where(valid_front, t_front, np.where(clamp_to_near, t_at_near, np.inf))
    t_masked = t_eff  # (R, N)

    best_j = np.argmin(t_masked, axis=1)  # (R,)
    best_t = t_masked[np.arange(R), best_j]  # (R,)

    hit_mask = best_t < np.inf
    hit_depth[hit_mask] = dy[hit_mask] * best_t[hit_mask]
    hit_id[hit_mask] = best_j[hit_mask]
    obs_intensity[hit_mask] = reflectivities[best_j[hit_mask]]

    if cfg.obs_noise_std > 0 and rng is not None:
        obs_intensity += rng.normal(0.0, cfg.obs_noise_std, R)
        obs_intensity = np.clip(obs_intensity, 0.0, 1.0)

    return _keep(hit_depth, hit_id, obs_intensity, cfg)


def render_scene(scene: Scene, visible: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every frame of a scene: ``(obs_depth, obs_id, obs_intensity)``, each (n_frames, obs_dim).
    ``visible`` is a blink schedule: hidden objects are left out and the markers painted."""
    cfg = scene.config
    rng = np.random.default_rng(cfg.seed + 1)  # offset from the simulator's stream

    R = obs_dim(cfg)
    obs_depth = np.zeros((cfg.n_frames, R))
    obs_id = np.full((cfg.n_frames, R), -1, dtype=int)
    obs_intensity = np.zeros((cfg.n_frames, R))

    for f in range(cfg.n_frames):
        obs_depth[f], obs_id[f], obs_intensity[f] = render_frame(
            scene.positions[f], scene.radii, scene.reflectivities, cfg, rng=rng,
            visible=None if visible is None else visible[f],
        )
        if visible is not None:
            paint_markers(obs_id[f], obs_intensity[f], visible[f],
                          visible[f + 1] if f + 1 < cfg.n_frames else None)

    return obs_depth, obs_id, obs_intensity
