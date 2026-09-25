"""A step-able Rayworld for the interactive demo (``scripts/demos/play.py``).

``InteractiveWorld`` advances one frame per ``step`` under one action per object, in ``"shift"``
(position offsets) or ``"force"`` (momentum) dynamics, with optional death and restart rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .config import SimConfig
from .sim import OBJECT_COLORS, frustum_half_width, fully_in_frustum, sample_position

WallMode = Literal["bounce", "clamp"]
Dynamics = Literal["shift", "force"]


@dataclass
class InteractiveConfig:
    """Dynamics and episode parameters of ``InteractiveWorld``; geometry and rendering stay in
    ``SimConfig``. The defaults suit human play at about 15 fps."""

    dynamics: Dynamics = "force"

    # shift mode
    shift_scale: float = 0.25  # world units moved per unit action, per step
    base_drift_speed: float = 0.06  # constant base drift magnitude (0 = static base)

    # force mode
    mass: float = 1.0
    force_scale: float = (
        0.06  # accel = (force_scale·action + drift) / mass  [world/step^2]
    )
    drift_force_std: float = 0.015  # intrinsic random force per step (anti-freeze)
    friction: float = 0.02  # fractional velocity damping per step
    max_speed: float = 0.6  # velocity-magnitude clamp [world/step]
    init_speed: float = 0.28  # random initial speed at each (re)start

    # collisions and spawning
    collision_slack: float = (
        1.0  # contact distance = 2·radius·slack (1.0 = discs exactly touch)
    )
    spawn_clearance: float = 1.5  # spawn separation = 2·radius·clearance

    # boundaries and death
    wall_mode: WallMode = (
        "bounce"  # physical wall response (walls stay solid regardless of death)
    )
    death_on_collision: bool = False  # object–object contact ends the episode
    death_on_wall: bool = (
        False  # touching a frustum wall ends the episode (walls still bounce/clamp)
    )
    reset_on_death: bool = True  # restart from a fresh state on death
    reset_noise_frames: int = 0  # frames of pure-noise observation before the restart

    # generation
    max_reset_attempts: int = 300


class InteractiveWorld:
    """A stateful world stepped one frame at a time by per-object actions."""

    def __init__(
        self,
        sim: SimConfig,
        cfg: InteractiveConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.sim = sim
        self.cfg = cfg or InteractiveConfig()
        self.n = int(sim.n_objects if sim.n_objects is not None else sim.n_objects_max)
        self._contact = (
            2.0 * sim.radius * self.cfg.collision_slack
        )  # true object contact distance
        self._spawn_sep = 2.0 * sim.radius * self.cfg.spawn_clearance  # birth clearance

        # per-object identity, fixed across restarts
        self._radii = np.full(self.n, sim.radius, dtype=float)
        self._colors = np.array(OBJECT_COLORS[: self.n], dtype=float)
        # one generator seeded from `seed`, so a session is reproducible
        self._rng = np.random.default_rng(seed)
        if sim.fixed_reflectivities:
            self._refl = np.linspace(sim.refl_min, sim.refl_max, self.n)
        else:
            self._refl = self._rng.uniform(sim.refl_min, sim.refl_max, self.n)

        self.reset(seed=None)

    @property
    def positions(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def velocities(self) -> np.ndarray:
        return self._vel.copy()

    @property
    def radii(self) -> np.ndarray:
        return self._radii

    @property
    def reflectivities(self) -> np.ndarray:
        return self._refl

    @property
    def colors(self) -> np.ndarray:
        return self._colors

    @property
    def obs_res(self) -> int:
        return self.sim.obs_res

    @property
    def n_objects(self) -> int:
        return self.n

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Restart from a fresh random state; returns the first observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.alive = True
        self.deaths = 0
        self.frames_survived = 0
        self._reset_after = 0
        self._last_action = np.zeros((self.n, 2))
        self._sample_ic()
        return self._render()

    def _sample_ic(self) -> None:
        sim, n, r = self.sim, self.n, self.sim.radius
        for _ in range(self.cfg.max_reset_attempts):
            pos = np.zeros((n, 2))
            for i in range(n):
                pos[i] = sample_position(self._rng, sim, r)
            ok = all(
                np.linalg.norm(pos[a] - pos[b]) >= self._spawn_sep
                for a in range(n)
                for b in range(a + 1, n)
            )
            if not ok:
                continue
            speed = (
                self.cfg.init_speed
                if self.cfg.dynamics == "force"
                else self.cfg.base_drift_speed
            )
            ang = self._rng.uniform(0.0, 2.0 * np.pi, n)
            self._pos = pos
            self._vel = speed * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            return
        raise RuntimeError(
            f"InteractiveWorld: could not sample a non-overlapping IC for {n} objects "
            f"after {self.cfg.max_reset_attempts} attempts (try fewer objects / smaller radius)."
        )

    def step(self, actions: Any = None) -> tuple[np.ndarray, dict]:
        """Advance one frame under ``actions``: ``(n, 2)`` in [-1, 1], ``(n, 3)`` as
        ``[active, a1, a2]``, or ``None``. Returns ``(obs, info)``, info holding the events
        (collision, wall, death, restart), the applied action and the full state."""
        # after a death: noise frames, then the restart
        if not self.alive and self.cfg.reset_on_death:
            self.t += 1
            if self._reset_after > 0:
                self._reset_after -= 1
                obs = self._noise_obs()
                return obs, self._info(dying=True)
            self._sample_ic()
            self.alive = True
            self.frames_survived = 0
            obs = self._render()
            return obs, self._info(rebirth=True)
        if not self.alive:  # dead without automatic restart: hold the last frame
            return self._render(), self._info()

        a = self._coerce_actions(actions)
        self._last_action = a
        blocked = np.zeros(self.n, dtype=bool)
        wall_hit = np.zeros(self.n, dtype=bool)

        if self.cfg.dynamics == "shift":
            new_pos = self._pos + self._vel * self.sim.dt  # drift
            for i in range(self.n):
                cand = new_pos[i] + a[i] * self.cfg.shift_scale  # plus the action offset
                if self._accept(cand, i, new_pos):
                    new_pos[i] = cand
                else:
                    blocked[i] = True
            for i in range(self.n):
                new_pos[i], self._vel[i], wall_hit[i] = self._apply_wall(
                    new_pos[i], self._vel[i]
                )
            self._pos = new_pos
        else:  # "force"
            drift = self._rng.normal(0.0, self.cfg.drift_force_std, (self.n, 2))
            accel = (a * self.cfg.force_scale + drift) / self.cfg.mass
            self._vel = self._vel + accel
            self._vel *= 1.0 - self.cfg.friction
            spd = np.linalg.norm(self._vel, axis=1, keepdims=True)
            over = (spd > self.cfg.max_speed).ravel()
            if over.any():
                self._vel[over] = self._vel[over] / spd[over] * self.cfg.max_speed
            new_pos = self._pos + self._vel * self.sim.dt
            for i in range(self.n):
                new_pos[i], self._vel[i], wall_hit[i] = self._apply_wall(
                    new_pos[i], self._vel[i]
                )
            self._pos = new_pos

        pairs = [
            (a_, b_)
            for a_ in range(self.n)
            for b_ in range(a_ + 1, self.n)
            if np.linalg.norm(self._pos[a_] - self._pos[b_]) < self._contact
        ]
        collision = len(pairs) > 0

        self.t += 1
        died = (collision and self.cfg.death_on_collision) or (
            wall_hit.any() and self.cfg.death_on_wall
        )
        if died:
            self.deaths += 1
            self.alive = False
            self._reset_after = self.cfg.reset_noise_frames
        else:
            self.frames_survived += 1

        obs = self._render()
        return obs, self._info(
            collision=collision,
            collision_pairs=pairs,
            wall=wall_hit,
            blocked=blocked,
            died=died,
        )

    def _coerce_actions(self, actions: Any) -> np.ndarray:
        n = self.n
        if actions is None:
            return np.zeros((n, 2))
        a = np.asarray(actions, dtype=float)
        if a.shape == (n, 3):  # [active, a1, a2]
            a = a[:, 0:1] * a[:, 1:3]
        a = a.reshape(n, 2)
        return np.clip(a, -1.0, 1.0)

    def _accept(self, cand: np.ndarray, i: int, positions: np.ndarray) -> bool:
        """Shift-mode guard: the candidate stays inside the frustum and overlaps no object."""
        if not fully_in_frustum(np.asarray(cand)[None, None, :], self.sim.radius, self.sim):
            return False
        for j in range(self.n):
            if j != i and np.linalg.norm(cand - positions[j]) < self._contact:
                return False
        return True

    def _apply_wall(
        self, p: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        sim, r, mode = self.sim, self.sim.radius, self.cfg.wall_mode
        x, y = float(p[0]), float(p[1])
        vx, vy = float(v[0]), float(v[1])
        hit = False
        y_lo, y_hi = sim.y_near + r, sim.y_far - r
        if y < y_lo:
            y, vy = (
                (2.0 * y_lo - y, abs(vy)) if mode == "bounce" else (y_lo, max(vy, 0.0))
            )
            hit = True
        elif y > y_hi:
            y, vy = (
                (2.0 * y_hi - y, -abs(vy)) if mode == "bounce" else (y_hi, min(vy, 0.0))
            )
            hit = True
        y = float(np.clip(y, y_lo, y_hi))
        x_lim = float(frustum_half_width(y, sim)) - r
        if x < -x_lim:
            x, vx = (
                (-2.0 * x_lim - x, abs(vx))
                if mode == "bounce"
                else (-x_lim, max(vx, 0.0))
            )
            hit = True
        elif x > x_lim:
            x, vx = (
                (2.0 * x_lim - x, -abs(vx))
                if mode == "bounce"
                else (x_lim, min(vx, 0.0))
            )
            hit = True
        x = float(np.clip(x, -x_lim, x_lim))
        return np.array([x, y]), np.array([vx, vy]), hit

    def _render(self) -> np.ndarray:
        from .renderer import render_frame

        depth, hit_id, intensity = render_frame(
            self._pos, self._radii, self._refl, self.sim, self._rng
        )
        self._last_depth = depth
        self._last_id = hit_id
        self._last_intensity = intensity.astype(np.float32)
        return self._last_intensity

    def _noise_obs(self) -> np.ndarray:
        obs = self._rng.uniform(0.0, 1.0, self.sim.obs_res).astype(np.float32)
        self._last_depth = np.zeros(self.sim.obs_res)
        self._last_id = np.full(self.sim.obs_res, -1, dtype=int)
        self._last_intensity = obs
        return obs

    def _info(
        self,
        *,
        collision: bool = False,
        collision_pairs: list | None = None,
        wall: np.ndarray | None = None,
        blocked: np.ndarray | None = None,
        died: bool = False,
        dying: bool = False,
        rebirth: bool = False,
    ) -> dict:
        return {
            "t": self.t,
            "dynamics": self.cfg.dynamics,
            "alive": self.alive,
            "died": died,
            "dying": dying,
            "rebirth": rebirth,
            "collision": collision,
            "collision_pairs": collision_pairs or [],
            "wall": np.zeros(self.n, bool) if wall is None else wall,
            "blocked": np.zeros(self.n, bool) if blocked is None else blocked,
            "deaths": self.deaths,
            "frames_survived": self.frames_survived,
            "action": self._last_action.copy(),
            "positions": self._pos.copy(),
            "velocities": self._vel.copy(),
            "hit_id": self._last_id.copy(),
            "hit_depth": self._last_depth.copy(),
        }
