"""World coordinates to the frustum basis (u, 1/y): the coordinates the renderer samples.

u = x / (scale * y) is linear in the ray index; depth shows only through a disc's apparent
width, which goes as 1/y. Velocities are central differences of the map along the velocity."""

from __future__ import annotations

import numpy as np


def fov_scale(sim: dict) -> float:
    """tan(half-FOV), the quantity ``renderer._fov_scale`` uses."""
    return float(sim["x_far"]) / float(sim["y_far"])


# The depth coordinate "frustum" resolves to; the other entries are alternatives.
CANONICAL_DEPTH = "inv_y"

DEPTHS = {
    "y": lambda x, y, sim: y,                                   # axial depth
    "rho": lambda x, y, sim: np.hypot(x, y),                    # Euclidean range
    "inv_y": lambda x, y, sim: 1.0 / y,                         # inverse axial depth
    "inv_rho": lambda x, y, sim: 1.0 / np.hypot(x, y),          # inverse range
    "width": lambda x, y, sim: (                                # apparent half-width in ray units
        float(sim["radius"]) * np.hypot(x, y)
        / (fov_scale(sim) * np.maximum(y, 1e-6) ** 2)
    ),
}


def lateral(pos: np.ndarray, sim: dict) -> np.ndarray:
    """The ray coordinate u = x / (scale * y), linear in the ray index."""
    scale = fov_scale(sim)
    x, y = pos[..., 0], pos[..., 1]
    return x / (scale * np.where(np.abs(y) < 1e-6, 1e-6, y))


def basis(pos: np.ndarray, vel: np.ndarray | None, sim: dict, depth: str = "frustum"):
    """(u, g(x, y)) and their time derivatives for a named depth coordinate g
    (``"frustum"`` resolves to ``CANONICAL_DEPTH``). Returns ``(positions, velocities)``,
    velocities None when ``vel`` is None."""
    g = DEPTHS[CANONICAL_DEPTH if depth == "frustum" else depth]
    x, y = pos[..., 0], pos[..., 1]
    fpos = np.stack([lateral(pos, sim), g(x, y, sim)], axis=-1)
    if vel is None:
        return fpos, None
    h = 1e-5
    p_plus, p_minus = pos + vel * h, pos - vel * h
    f_plus = np.stack([lateral(p_plus, sim), g(p_plus[..., 0], p_plus[..., 1], sim)], -1)
    f_minus = np.stack([lateral(p_minus, sim), g(p_minus[..., 0], p_minus[..., 1], sim)], -1)
    return fpos, (f_plus - f_minus) / (2 * h)
