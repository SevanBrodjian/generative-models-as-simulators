"""Simulation and rendering configuration for Rayworld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SimConfig:
    """All parameters of one scene; the observer sits at the origin looking along +y. Fields this
    release does not implement are kept, since stored configs rebuild with ``SimConfig(**d)``."""

    # reproducibility
    seed: int = 42

    # world geometry
    y_near: float = 3.0  # depth of the near plane
    y_far: float = 12.0  # depth of the far plane
    x_near: float = 1.5  # frustum half-width at y_near
    x_far: float = 6.0  # frustum half-width at y_far

    # objects
    n_objects: int | None = 3  # fixed count; None draws from [n_objects_min, n_objects_max]
    n_objects_min: int = 1
    n_objects_max: int = 5
    radius: float = 0.5
    speed_min: float = 0.05  # world units per frame
    speed_max: float = 0.12

    # dynamics; the three noise terms default to 0 (straight lines at constant speed)
    n_frames: int = 100
    dt: float = 1.0
    direction_noise_std: float = 0.0  # radians per step on the velocity angle
    speed_noise_std: float = 0.0  # multiplicative noise on speed per step
    position_noise_std: float = 0.0  # additive Gaussian on position per step

    # blink (blink.py); blink_prob = 0 turns it off
    blink_prob: float = 0.0  # per object and frame, probability of starting a blackout
    blink_mean: float = 6.0  # mean blackout length, Geometric(1/mean), capped at blink_max
    blink_max: int = 12
    blink_warmup: int = 3  # no blackout begins before this frame

    # 1D observation
    obs_res: int = 128  # rays cast
    drop_edge_rays: bool = False  # drop the two wall-aligned rays: obs_dim = obs_res - 2
    refl_min: float = 0.4  # reflectivity range; a ray returns the reflectivity it hits
    refl_max: float = 0.8
    refl_min_sep: float = 0.15  # minimum pairwise reflectivity separation when sampled
    fixed_reflectivities: bool = False  # evenly spaced in [refl_min, refl_max] instead
    obs_noise_std: float = 0.04  # additive Gaussian on every ray, clipped to [0, 1]

    # "bounce" off the frustum walls, "open" (drift out of view), or "wrap" (toroidal)
    boundary: Literal["bounce", "open", "wrap"] = "bounce"

    # scene generation
    always_in_frustum: bool = False  # reject trajectories that ever touch a frustum wall
    max_gen_attempts: int = 300
    collision_margin: float = 1.6  # minimum center distance = collision_margin * 2 * radius

    # soft rendering (soft_render.py): only soft_shading="power" with the other knobs at 0
    soft_edge: float = 0.0
    soft_shading: Literal["flat", "lambert", "power"] = "flat"
    soft_profile_power: float = 2.0
    soft_psf_sigma: float = 0.0
    soft_occlusion_temp: float = 0.0

    # several observers on a ring, and the circular arena (observers.py; open boundary only)
    n_observers: int = 1
    region: Literal["frustum", "circle"] = "frustum"

    # a top-down raster observation; not implemented here, kept so stored configs rebuild
    omni2d: bool = False
    omni2d_h: int = 48
    omni2d_w: int = 64


def check_supported(cfg: SimConfig) -> None:
    """Raise for a configuration this renderer does not implement."""
    if getattr(cfg, "omni2d", False):
        raise ValueError("omni2d rendering is not implemented")
    if (getattr(cfg, "soft_edge", 0.0) > 0.0 or getattr(cfg, "soft_psf_sigma", 0.0) > 0.0
            or getattr(cfg, "soft_occlusion_temp", 0.0) > 0.0
            or getattr(cfg, "soft_shading", "flat") not in ("flat", "power")):
        raise ValueError("soft rendering is implemented for soft_shading='power' with "
                         "soft_edge, soft_psf_sigma and soft_occlusion_temp at 0 only")


def obs_dim(cfg: SimConfig) -> int:
    """Flat observation width: kept rays per view times the number of views."""
    n_views = int(getattr(cfg, "n_observers", 1))
    if getattr(cfg, "drop_edge_rays", False):
        return n_views * (int(cfg.obs_res) - 2)
    return n_views * int(cfg.obs_res)
