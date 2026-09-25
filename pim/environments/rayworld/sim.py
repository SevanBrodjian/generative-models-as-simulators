"""Rayworld's simulator state and its dynamics: object positions, velocities and trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .observers import inside_region, sample_position_region

# RGB in [0, 1], one per object (up to 5), distinct on a dark background.
OBJECT_COLORS: list[tuple[float, float, float]] = [
    (0.00, 0.83, 1.00),  # cyan
    (1.00, 0.42, 0.42),  # coral
    (1.00, 0.85, 0.24),  # amber
    (0.42, 0.80, 0.47),  # green
    (0.78, 0.48, 1.00),  # violet
]


@dataclass
class Scene:
    """One simulated trajectory; ``positions`` is the full simulator state the renderer reads.
    ``velocities[f]`` is the velocity after boundary handling at frame f (moving into f + 1)."""

    positions: np.ndarray  # (n_frames, n_objects, 2)  —  [x, y] per object per frame
    velocities: np.ndarray  # (n_frames, n_objects, 2)  —  [vx, vy] per object per frame
    radii: np.ndarray  # (n_objects,)
    colors: np.ndarray  # (n_objects, 3)  —  RGB
    reflectivities: np.ndarray  # (n_objects,)  —  scalar in [refl_min, refl_max]
    config: SimConfig


def frustum_half_width(y: float | np.ndarray, cfg: SimConfig) -> float | np.ndarray:
    """Half-width of the frustum at depth y (linear interpolation)."""
    t = (y - cfg.y_near) / (cfg.y_far - cfg.y_near)
    return cfg.x_near + (cfg.x_far - cfg.x_near) * t


def compute_visibility(scene: "Scene") -> np.ndarray:
    """(n_frames, n_objects) bool: the disc at least partly overlaps the frustum (approximate
    at the near and far planes)."""
    cfg = scene.config
    pos = scene.positions  # (n_frames, n_objects, 2)
    r = scene.radii[None, :]  # (1, n_objects)

    x = pos[:, :, 0]  # (F, N)
    y = pos[:, :, 1]  # (F, N)

    in_y = (y + r > cfg.y_near) & (y - r < cfg.y_far)

    y_clamped = np.clip(y, cfg.y_near, cfg.y_far)
    x_lim = frustum_half_width(y_clamped, cfg)  # (F, N)
    in_x = np.abs(x) - r < x_lim

    return in_y & in_x


def fully_in_frustum(positions: np.ndarray, radius: float, cfg: SimConfig) -> bool:
    """True if every disc is fully inside the arena at every frame; ``positions`` is (F, n, 2)
    (a single point as ``p[None, None, :]``). Shared by the simulator, edits and interactive world."""
    if getattr(cfg, "region", "frustum") == "circle":
        return inside_region(positions, radius, cfg)
    x = positions[:, :, 0]  # (F, n)
    y = positions[:, :, 1]  # (F, n)

    in_y = (y - radius >= cfg.y_near) & (y + radius <= cfg.y_far)

    # frustum half-width at each object's depth, then subtract radius for clearance
    y_clamped = np.clip(y, cfg.y_near, cfg.y_far)
    x_lim = frustum_half_width(y_clamped, cfg) - radius
    in_x = np.abs(x) <= x_lim

    return bool((in_y & in_x).all())


def sample_position(rng: np.random.Generator, cfg: SimConfig, radius: float) -> tuple[float, float]:
    """One (x, y) with the disc fully inside the arena: depth first, then x at that depth.
    Two ``rng.uniform`` calls in this order; every generated dataset depends on it."""
    if getattr(cfg, "region", "frustum") == "circle":
        return sample_position_region(rng, cfg, radius)
    y = rng.uniform(cfg.y_near + radius, cfg.y_far - radius)
    x_lim = frustum_half_width(y, cfg) - radius
    x = rng.uniform(-x_lim, x_lim)
    return x, y


def _sample_reflectivities(
    rng: np.random.Generator, n: int, cfg: SimConfig
) -> np.ndarray:
    """Sample n reflectivities from [refl_min, refl_max] with pairwise separation ≥ refl_min_sep."""
    if n <= 1 or cfg.refl_min_sep <= 0.0:
        return rng.uniform(cfg.refl_min, cfg.refl_max, n)

    needed = cfg.refl_min_sep * (n - 1)
    available = cfg.refl_max - cfg.refl_min
    if needed > available + 1e-9:
        raise ValueError(
            f"refl_min_sep={cfg.refl_min_sep} * (n_objects-1={n-1}) = {needed:.3f} "
            f"exceeds refl range {cfg.refl_min}–{cfg.refl_max} = {available:.3f}"
        )

    for _ in range(1000):
        vals = rng.uniform(cfg.refl_min, cfg.refl_max, n)
        diffs = np.abs(vals[:, None] - vals[None, :])
        np.fill_diagonal(diffs, np.inf)
        if diffs.min() >= cfg.refl_min_sep - 1e-9:
            return vals

    raise RuntimeError(
        f"Could not sample {n} reflectivities with pairwise separation ≥ {cfg.refl_min_sep} "
        f"in [refl_min={cfg.refl_min}, refl_max={cfg.refl_max}] after 1000 attempts."
    )


def simulate(cfg: SimConfig) -> Scene:
    """A collision-free trajectory from ``cfg.seed``: rejects attempts where two objects come
    within ``collision_margin * 2 * radius`` (or, with ``always_in_frustum``, leave the arena),
    up to ``cfg.max_gen_attempts`` times with the same generator."""
    rng = np.random.default_rng(cfg.seed)
    if getattr(cfg, "region", "frustum") == "circle" and cfg.boundary != "open":
        raise ValueError('region="circle" has no walls: use boundary="open" (with '
                         "always_in_frustum for the stay-inside acceptance)")

    n = cfg.n_objects
    if n is None:
        n = int(rng.integers(cfg.n_objects_min, cfg.n_objects_max + 1))
    pair = getattr(cfg, "pair_separation", None)
    if pair is not None:
        if n != 2:
            raise ValueError("pair_separation needs exactly two objects")
        if cfg.boundary != "open":
            raise ValueError('pair_separation needs boundary="open" (a wall bounce would break the pair)')
        if pair < cfg.collision_margin * 2.0 * cfg.radius:
            raise ValueError(f"pair_separation {pair} is below the collision spacing "
                             f"{cfg.collision_margin * 2.0 * cfg.radius}; every draw would be rejected")

    if cfg.fixed_reflectivities:
        reflectivities = np.linspace(cfg.refl_min, cfg.refl_max, n)
    else:
        reflectivities = _sample_reflectivities(rng, n, cfg)
    colors = np.array(OBJECT_COLORS[:n], dtype=float)
    radii = np.full(n, cfg.radius)
    min_sep = cfg.collision_margin * 2.0 * cfg.radius

    for _ in range(cfg.max_gen_attempts):
        positions = np.zeros((cfg.n_frames, n, 2))
        velocities = np.zeros((cfg.n_frames, n, 2))
        vels = np.zeros((n, 2))

        for i in range(n):
            if pair is not None and i == 1:
                # rigid pair: object 1 at the fixed separation from object 0, same velocity
                theta = rng.uniform(0.0, 2.0 * np.pi)
                positions[0, 1] = positions[0, 0] + pair * np.array([np.cos(theta), np.sin(theta)])
                vels[1] = vels[0]
                continue
            positions[0, i] = sample_position(rng, cfg, cfg.radius)

            speed = rng.uniform(cfg.speed_min, cfg.speed_max)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            vels[i] = speed * np.array([np.cos(angle), np.sin(angle)])

        velocities[0] = vels

        init_collision = any(
            np.linalg.norm(positions[0, a] - positions[0, b]) < min_sep
            for a in range(n)
            for b in range(a + 1, n)
        )
        if init_collision:
            continue

        collision = False
        for f in range(1, cfg.n_frames):
            if cfg.direction_noise_std > 0 or cfg.speed_noise_std > 0:
                speeds = np.hypot(vels[:, 0], vels[:, 1])  # (n,)
                angles = np.arctan2(vels[:, 1], vels[:, 0])  # (n,)
                if cfg.direction_noise_std > 0:
                    angles += rng.normal(0.0, cfg.direction_noise_std, n)
                if cfg.speed_noise_std > 0:
                    speeds *= np.maximum(
                        0.0, 1.0 + rng.normal(0.0, cfg.speed_noise_std, n)
                    )
                    speeds = np.clip(speeds, cfg.speed_min * 0.3, cfg.speed_max * 2.0)
                vels[:, 0] = speeds * np.cos(angles)
                vels[:, 1] = speeds * np.sin(angles)

            new_pos = positions[f - 1] + vels * cfg.dt
            if cfg.position_noise_std > 0:
                new_pos = new_pos + rng.normal(0.0, cfg.position_noise_std, (n, 2))

            for i in range(n):
                x, y = new_pos[i]

                if cfg.boundary == "bounce":
                    y_lo = cfg.y_near + cfg.radius
                    y_hi = cfg.y_far - cfg.radius
                    if y < y_lo:
                        y = 2.0 * y_lo - y
                        vels[i, 1] = abs(vels[i, 1])
                    elif y > y_hi:
                        y = 2.0 * y_hi - y
                        vels[i, 1] = -abs(vels[i, 1])
                    y = np.clip(y, y_lo, y_hi)
                    x_lim = frustum_half_width(y, cfg) - cfg.radius
                    if x < -x_lim:
                        x = -2.0 * x_lim - x
                        vels[i, 0] = abs(vels[i, 0])
                    elif x > x_lim:
                        x = 2.0 * x_lim - x
                        vels[i, 0] = -abs(vels[i, 0])
                    x = np.clip(x, -x_lim, x_lim)

                elif cfg.boundary == "open":
                    pass

                elif cfg.boundary == "wrap":
                    y = cfg.y_near + (y - cfg.y_near) % (cfg.y_far - cfg.y_near)
                    x = -cfg.x_far + (x + cfg.x_far) % (2.0 * cfg.x_far)

                new_pos[i] = [x, y]

            positions[f] = new_pos
            velocities[f] = vels

            for a in range(n):
                for b in range(a + 1, n):
                    if np.linalg.norm(positions[f, a] - positions[f, b]) < min_sep:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                break

        if not collision and (
            not cfg.always_in_frustum or fully_in_frustum(positions, cfg.radius, cfg)
        ):
            return Scene(
                positions=positions,
                velocities=velocities,
                radii=radii,
                colors=colors,
                reflectivities=reflectivities,
                config=cfg,
            )

    raise RuntimeError(
        f"Could not generate a collision-free scene after {cfg.max_gen_attempts} attempts. "
        "Try fewer objects, a larger world, or a smaller collision_margin."
    )
