"""Several observers around a circular arena (the obs5 variant).

Observer k is the standard observer rotated by 2*pi*k/N about the frustum's depth midpoint
(observer 0 is the standard one); views are concatenated observer-major."""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import SimConfig


def pivot(cfg: SimConfig) -> np.ndarray:
    """The point the observers rotate about: the optical axis at the frustum's depth midpoint."""
    return np.array([0.0, 0.5 * (cfg.y_near + cfg.y_far)])


def region_radius(cfg: SimConfig) -> float:
    """Radius of the circular arena."""
    return 0.5 * (cfg.y_far - cfg.y_near)


def observer_angles(cfg: SimConfig) -> np.ndarray:
    """(N,) rotation of each observer about the pivot; observer 0 is the standard one."""
    n = int(getattr(cfg, "n_observers", 1))
    return 2.0 * np.pi * np.arange(n) / n


def observer_poses(cfg: SimConfig) -> np.ndarray:
    """(N, 3): each observer's position (x, y) and view angle phi from the +x axis
    (pi/2 for the standard observer). Each sits at distance (y_near + y_far)/2 from the pivot,
    looking at it."""
    c = pivot(cfg)
    d = c[1]
    out = []
    for a in observer_angles(cfg):
        off = d * np.array([np.sin(a), -np.cos(a)])      # the standard offset (0, -d), rotated
        out.append([c[0] + off[0], c[1] + off[1], np.pi / 2 + a])
    return np.array(out)


def to_observer_frame(positions: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """World coordinates (..., 2) to the observer's frame (observer at the origin looking +y).
    The identity, exactly, for observer 0."""
    a = float(pose[2]) - np.pi / 2
    p = np.asarray(positions, float) - np.asarray(pose[:2], float)
    if a == 0.0:
        return p
    ca, sa = np.cos(-a), np.sin(-a)
    x, y = p[..., 0], p[..., 1]
    return np.stack([ca * x - sa * y, sa * x + ca * y], -1)


def inside_region(positions: np.ndarray, radius: float, cfg: SimConfig) -> bool:
    """True when every disc is fully inside the circle at every frame; ``positions`` is
    (F, n, 2), as in ``sim.fully_in_frustum``."""
    d = np.linalg.norm(np.asarray(positions, float) - pivot(cfg), axis=-1)
    return bool((d + radius <= region_radius(cfg)).all())


def sample_position_region(rng: np.random.Generator, cfg: SimConfig, radius: float) -> tuple[float, float]:
    """One (x, y) uniform over the circle of radius ``region_radius - radius``; two
    ``rng.uniform`` draws, like ``sim.sample_position``."""
    rmax = region_radius(cfg) - radius
    r = rmax * np.sqrt(rng.uniform(0.0, 1.0))
    t = rng.uniform(0.0, 2.0 * np.pi)
    c = pivot(cfg)
    return float(c[0] + r * np.cos(t)), float(c[1] + r * np.sin(t))


def single_view_config(cfg: SimConfig) -> SimConfig:
    """The same config with one observer: what each view is rendered with."""
    return dataclasses.replace(cfg, n_observers=1)


def render_frame_multi(positions, radii, reflectivities, cfg: SimConfig, rng=None, visible=None):
    """``renderer.render_frame`` from every observer, concatenated observer-major:
    ``(hit_depth, hit_id, intensity)``, each of length N * R. Depths are in each observer's
    frame; ids are world object indices."""
    from .renderer import render_frame

    cfg1 = single_view_config(cfg)
    pos = np.asarray(positions, float)
    outs = [render_frame(to_observer_frame(pos, pose), radii, reflectivities, cfg1, rng=rng,
                         visible=visible) for pose in observer_poses(cfg)]
    return tuple(np.concatenate([o[j] for o in outs]) for j in range(3))
