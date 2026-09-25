"""Soft rendering for the smooth variant: each disc's image is a dome, not a plateau.

A ray's intensity is the reflectivity times ``(1 - (perp/r)^2) ** soft_profile_power``
(``perp`` its distance from the disc center), zero at the silhouette; occlusion stays exact."""

from __future__ import annotations

import numpy as np

from pim.environments.rayworld.config import SimConfig, check_supported

__all__ = ["soft_enabled", "render_frame_soft"]

_EPS = 1e-9
_FAR = 1e9  # stand-in for "no hit"; finite so it survives arithmetic


def soft_enabled(cfg: SimConfig) -> bool:
    """True when any soft-rendering knob is off its default."""
    return (
        getattr(cfg, "soft_edge", 0.0) > 0.0
        or getattr(cfg, "soft_shading", "flat") != "flat"
        or getattr(cfg, "soft_psf_sigma", 0.0) > 0.0
        or getattr(cfg, "soft_occlusion_temp", 0.0) > 0.0
    )


def _ray_dirs(cfg: SimConfig):
    """Unit ray directions, identical to ``renderer.render_frame``."""
    from pim.environments.rayworld.renderer import _fov_scale

    s = np.linspace(-1.0, 1.0, cfg.obs_res)
    dx = s * _fov_scale(cfg)
    dy = np.ones(cfg.obs_res)
    norm = np.hypot(dx, dy)
    return dx / norm, dy / norm


def render_frame_soft(
    positions: np.ndarray,
    radii: np.ndarray,
    reflectivities: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft render of one frame; same signature and returns as ``renderer.render_frame``."""
    from pim.environments.rayworld.renderer import render_frame

    if not soft_enabled(cfg):
        return render_frame(positions, radii, reflectivities, cfg, rng=rng)
    check_supported(cfg)

    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    dx, dy = _ray_dirs(cfg)

    # the float expressions below fix the stored smooth data bit for bit
    cx, cy = positions[:, 0], positions[:, 1]
    b_ = dx[:, None] * cx[None, :] + dy[:, None] * cy[None, :]
    perp2 = np.maximum(cx**2 + cy**2 - b_**2, 0.0)
    sq = np.sqrt(np.maximum(radii[None, :] ** 2 - perp2, 0.0))
    t_front, t_back = b_ - sq, b_ + sq
    hy_f, hy_b = dy[:, None] * t_front, dy[:, None] * t_back
    t_at_near = cfg.y_near / dy[:, None]

    visible = (t_front > _EPS) & (hy_f >= cfg.y_near) & (hy_f <= cfg.y_far)
    clamp_near = (hy_f < cfg.y_near) & (hy_b >= cfg.y_near)
    t_eff = np.where(visible, t_front, np.where(clamp_near, t_at_near, _FAR))
    gate = (t_eff < _FAR).astype(float)
    signed = radii[None, :] - np.sqrt(perp2)
    cos_n_d = sq / np.maximum(radii[None, :], _EPS)        # sqrt(1 - (perp/r)^2)
    alpha = gate * (signed > 0)
    shade = reflectivities[None, :] * cos_n_d ** (2.0 * float(cfg.soft_profile_power))

    dt = t_eff[..., None, :] - t_eff[..., :, None]
    front = (dt > 0).astype(float)
    eye = np.eye(alpha.shape[-1])
    keep = 1.0 - alpha[..., None, :] * front * (1.0 - eye)
    intensity = (alpha * shade * keep.prod(-1)).sum(-1)

    # depth and id use the strict hit test (discriminant >= 0), as the hard renderer does
    strict = (radii[None, :] ** 2 - perp2 >= 0) & (t_eff < _FAR)
    t_strict = np.where(strict, t_eff, _FAR)
    best = np.argmin(t_strict, axis=-1)
    hit = t_strict[np.arange(cfg.obs_res), best] < _FAR
    hit_depth = np.where(hit, dy * t_strict[np.arange(cfg.obs_res), best], 0.0)
    hit_id = np.where(hit, best, -1).astype(int)

    if cfg.obs_noise_std > 0 and rng is not None:
        intensity = np.clip(
            intensity + rng.normal(0.0, cfg.obs_noise_std, cfg.obs_res), 0.0, 1.0
        )
    else:
        intensity = np.clip(intensity, 0.0, 1.0)
    return hit_depth, hit_id, intensity
