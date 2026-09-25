#!/usr/bin/env python
"""Play Rayworld with the keyboard (pim.environments.rayworld.interactive), or watch a scripted driver.

Left: the 2D world; right: the 1D observation waterfall; bottom: the keys pressed and a status panel.
Object 0: W/A/S/D. Object 1: arrow keys (or I/J/K/L). R reset, M shift/force dynamics,
C death on collision, B death on wall, SPACE pause, Q quit.

    python scripts/demos/play.py [--dynamics shift] [--driver avoid --save outputs/play.gif]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pim.environments.rayworld.config import SimConfig  # noqa: E402
from pim.environments.rayworld.interactive import InteractiveConfig, InteractiveWorld  # noqa: E402

# matplotlib key names -> (column, row, label) on the key overlay
WASD = {"w": (1, 2, "W"), "a": (0, 1, "A"), "s": (1, 1, "S"), "d": (2, 1, "D")}
ARROWS = {"up": (5, 2, "↑"), "left": (4, 1, "←"), "down": (5, 1, "↓"), "right": (6, 1, "→")}
IJKL_ALIAS = {"i": "up", "k": "down", "j": "left", "l": "right"}
KEY_OFF = "#20242e"


def keys_to_action(pressed: set[str], n: int) -> np.ndarray:
    """The held keys as an ``(n, 2)`` action: object 0 from WASD, object 1 from the arrows."""
    p = set(pressed)
    for k, tgt in IJKL_ALIAS.items():
        if k in p:
            p.add(tgt)
    a = np.zeros((n, 2))
    if n >= 1:
        a[0] = [("d" in p) - ("a" in p), ("w" in p) - ("s" in p)]
    if n >= 2:
        a[1] = [("right" in p) - ("left" in p), ("up" in p) - ("down" in p)]
    return a


class RandomDriver:
    """Smoothed (Ornstein-Uhlenbeck) random actions."""

    def __init__(self, n: int, seed: int = 0, theta: float = 0.15, sigma: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros((n, 2))
        self.theta, self.sigma = theta, sigma

    def act(self, world: InteractiveWorld) -> np.ndarray:
        self.a += -self.theta * self.a + self.sigma * self.rng.normal(0, 1, self.a.shape)
        return np.clip(self.a, -1, 1)


class HeuristicAvoidDriver:
    """Pushes each object away from the other objects and from the near and far walls."""

    def act(self, world: InteractiveWorld) -> np.ndarray:
        pos, n, sim = world.positions, world.n, world.sim
        a = np.zeros((n, 2))
        for i in range(n):
            f = np.zeros(2)
            for j in range(n):
                if j != i:
                    d = pos[i] - pos[j]
                    f += d / (np.linalg.norm(d) + 1e-6) ** 2 * 2.0     # inverse-square repulsion
            f[1] += 1.0 / (pos[i, 1] - sim.y_near + 0.3) - 1.0 / (sim.y_far - pos[i, 1] + 0.3)
            a[i] = f
        return np.clip(a / (np.abs(a).max() + 1e-6), -1, 1)


class Emulator:
    """The figure, its keyboard handlers and one ``update`` per animation frame."""

    def __init__(self, world: InteractiveWorld, driver=None, wf_rows: int = 100):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Polygon, Rectangle

        # free the keyboard: matplotlib's default key bindings (e.g. 's' = save) eat key releases
        for km in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[km] = []
        self.plt, self.world, self.driver = plt, world, driver        # driver None: the keyboard
        self.pressed: set[str] = set()
        self.paused = False
        self._flash, self._flash_t = "", 0
        self.wf = np.zeros((wf_rows, world.obs_res), dtype=np.float32)

        bg, edge, tick, text = "#0a0a14", "#5c677f", "#808a9d", "#a3adc2"
        self.text = text
        sim = world.sim
        fig = plt.figure(figsize=(6.5 + 8.4, 6.6), facecolor=bg)
        gs = GridSpec(2, 2, height_ratios=[5, 1.5], hspace=0.24, wspace=0.14,
                      left=0.055, right=0.975, top=0.92, bottom=0.06)
        self.fig = fig
        axw, axf = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axk, axs = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        for ax in (axw, axf, axk, axs):
            ax.set_facecolor(bg)
            for sp in ax.spines.values():
                sp.set_edgecolor(tick)
            ax.tick_params(colors=tick, labelsize=8)

        # the 2D world
        axw.set_xlim(-sim.x_far - 0.7, sim.x_far + 0.7)
        axw.set_ylim(sim.y_near - 0.7, sim.y_far + 0.7)
        axw.set_aspect("equal")
        axw.set_xlabel("x", color=text, fontsize=9)
        axw.set_ylabel("depth  y", color=text, fontsize=9)
        axw.set_title("2D world  (simulator state)", color=text, fontsize=11, pad=6)
        corners = np.array([[-sim.x_near, sim.y_near], [sim.x_near, sim.y_near],
                            [sim.x_far, sim.y_far], [-sim.x_far, sim.y_far]])
        axw.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=edge, linewidth=1.8, zorder=1))
        self.circles, self.refl_labels, self.vel_lines, self.act_lines = [], [], [], []
        for i in range(world.n):
            c = plt.Circle((0, 0), world.radii[i], facecolor=world.colors[i], zorder=3,
                           linewidth=1.2, edgecolor="white", alpha=0.95)
            axw.add_patch(c)
            self.circles.append(c)
            self.vel_lines.append(axw.plot([], [], color=world.colors[i], lw=1.6, alpha=0.5, zorder=2)[0])
            self.act_lines.append(axw.plot([], [], color="white", lw=2.4, alpha=0.9, zorder=5,
                                           solid_capstyle="round")[0])
            self.refl_labels.append(axw.text(0, 0, f"{world.reflectivities[i]:.2f}", ha="center",
                                             va="center", color="white", fontsize=7, fontweight="bold",
                                             zorder=6, fontfamily="monospace"))
        self.event_text = axw.text(0.5, 0.965, "", transform=axw.transAxes, ha="center", va="top",
                                   color="#FF5252", fontsize=13, fontweight="bold")

        # the observation waterfall
        axf.set_title("1D observation waterfall  (what the model sees)", color=text, fontsize=11, pad=6)
        axf.set_xlabel("ray", color=text, fontsize=9)
        axf.set_ylabel("time  (newest at bottom)", color=text, fontsize=9)
        self.wf_img = axf.imshow(self.wf, aspect="auto", origin="upper", cmap="gray", vmin=0, vmax=1,
                                 interpolation="nearest", extent=[0, world.obs_res, wf_rows, 0])
        axf.set_yticks([])

        # the key overlay
        axk.set_xlim(-0.7, 7.7)
        axk.set_ylim(0.3, 3.0)
        axk.set_aspect("equal")
        axk.set_xticks([])
        axk.set_yticks([])
        axk.set_title("keys pressed", color=text, fontsize=10, pad=4)
        self.key_patches = {}
        for keymap, col in ((WASD, world.colors[0]),
                            (ARROWS, world.colors[1] if world.n > 1 else world.colors[0])):
            for key, (cx, cy, label) in keymap.items():
                rect = Rectangle((cx - 0.44, cy - 0.44), 0.88, 0.88, facecolor=KEY_OFF,
                                 edgecolor=col, linewidth=2.0, zorder=2)
                axk.add_patch(rect)
                axk.text(cx, cy, label, ha="center", va="center", color=text, fontsize=11,
                         fontweight="bold", zorder=3)
                self.key_patches[key] = (rect, np.array(col))
        axk.text(1.0, 0.15, "object 0", ha="center", color=world.colors[0], fontsize=8)
        if world.n > 1:
            axk.text(5.0, 0.15, "object 1  (or IJKL)", ha="center", color=world.colors[1], fontsize=8)

        # the status panel
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title("status", color=text, fontsize=10, pad=4)
        self.status_text = axs.text(0.03, 0.92, "", transform=axs.transAxes, ha="left", va="top",
                                    color=text, fontsize=9.5, fontfamily="monospace", linespacing=1.5)
        fig.canvas.mpl_connect("key_press_event", self._on_press)
        fig.canvas.mpl_connect("key_release_event", self._on_release)

    def _flash_msg(self, msg: str) -> None:
        self._flash, self._flash_t = msg, 25            # shown for about 25 frames

    def _on_press(self, event):
        k, cfg = event.key, self.world.cfg
        if k is None:
            return
        if k in ("q", "escape"):
            self.plt.close(self.fig)
        elif k == " ":
            self.paused = not self.paused
        elif k == "r":
            self.world.reset()
            self.wf[:] = 0
            self.pressed.clear()
        elif k == "m":                                   # switch dynamics, keeping positions
            cfg.dynamics = "shift" if cfg.dynamics == "force" else "force"
            self.pressed.clear()
            self._flash_msg(f"mode → {cfg.dynamics}")
        elif k == "c":
            cfg.death_on_collision = not cfg.death_on_collision
            self._flash_msg(f"death-on-collision: {cfg.death_on_collision}")
        elif k == "b":
            cfg.death_on_wall = not cfg.death_on_wall
            self._flash_msg(f"death-on-wall: {cfg.death_on_wall}")
        else:
            self.pressed.add(k)

    def _on_release(self, event):
        self.pressed.discard(event.key)

    @staticmethod
    def _action_to_keys(action: np.ndarray, n: int, thr: float = 0.25) -> set[str]:
        """A driver's continuous action shown as key presses."""
        keys: set[str] = set()
        for i, (left, right, down, up) in enumerate((("a", "d", "s", "w"),
                                                     ("left", "right", "down", "up"))[:n]):
            keys |= {k for k, on in ((right, action[i, 0] > thr), (left, action[i, 0] < -thr),
                                     (up, action[i, 1] > thr), (down, action[i, 1] < -thr)) if on}
        return keys

    def update(self, _frame):
        world = self.world
        if not self.paused:
            action = (keys_to_action(self.pressed, world.n) if self.driver is None
                      else self.driver.act(world))
            obs, info = world.step(action)
            self.wf[:-1] = self.wf[1:]
            self.wf[-1] = obs
        else:
            info = world._info()
        self.wf_img.set_data(self.wf)
        act = info["action"]
        for i in range(world.n):
            p, v = info["positions"][i], info["velocities"][i]
            self.circles[i].center = (p[0], p[1])
            self.refl_labels[i].set_position((p[0], p[1]))
            self.vel_lines[i].set_data([p[0], p[0] + v[0] * 8.0], [p[1], p[1] + v[1] * 8.0])
            self.act_lines[i].set_data([p[0], p[0] + act[i, 0] * 1.4], [p[1], p[1] + act[i, 1] * 1.4])

        held = set(self.pressed if self.driver is None else self._action_to_keys(act, world.n))
        for kk, tgt in IJKL_ALIAS.items():
            if kk in held:
                held.add(tgt)
        for key, (rect, col) in self.key_patches.items():
            rect.set_facecolor(tuple(col) if key in held else KEY_OFF)

        msg = ""
        if info.get("dying"):
            msg = "• • •"
        elif info.get("rebirth"):
            msg = "rebirth"
        elif info.get("died"):
            msg = "✖ DEATH"
        elif info.get("collision"):
            msg = "collision"
        if self._flash_t > 0:                            # a mode or toggle message takes precedence
            self._flash_t -= 1
            msg = self._flash
        self.event_text.set_text(msg)
        dc = "on" if world.cfg.death_on_collision else "off"
        db = "on" if world.cfg.death_on_wall else "off"
        self.status_text.set_text(
            f"dynamics : {world.cfg.dynamics}\nframe    : {info['t']}\nalive    : {info['alive']}\n"
            f"survived : {info['frames_survived']}\ndeaths   : {info['deaths']}\n"
            f"death: coll {dc} · wall {db}\n{'PAUSED' if self.paused else ''}")
        return []

    def run(self, interval: int = 60):
        from matplotlib.animation import FuncAnimation

        self.fig.suptitle("Rayworld  —  R reset · M mode · C coll-death · B wall-death · SPACE pause · Q quit",
                          color=self.text, fontsize=12, y=0.975)
        self._anim = FuncAnimation(self.fig, self.update, frames=itertools.count(), interval=interval,
                                   blit=False, cache_frame_data=False)
        self.plt.show()

    def save(self, path: str, frames: int = 150, fps: int = 15, dpi: int = 110):
        from matplotlib.animation import FuncAnimation, PillowWriter

        self.fig.suptitle("Rayworld", color=self.text, fontsize=12, y=0.995)
        anim = FuncAnimation(self.fig, self.update, frames=range(frames), interval=1000 // fps,
                             blit=False, cache_frame_data=False)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        print(f"saved -> {path}")


def build_world(args) -> InteractiveWorld:
    sim = SimConfig(seed=args.seed, n_objects=args.n_objects, radius=0.5, obs_res=args.obs_res,
                    obs_noise_std=args.obs_noise, fixed_reflectivities=True, boundary="bounce")
    icfg = InteractiveConfig(dynamics=args.dynamics, death_on_collision=args.death_on_collision,
                             death_on_wall=args.death_on_wall, reset_on_death=True,
                             reset_noise_frames=args.reset_noise_frames, wall_mode="bounce")
    return InteractiveWorld(sim, icfg, seed=args.seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dynamics", choices=["shift", "force"], default="force")
    p.add_argument("--driver", choices=["human", "random", "avoid"], default="human")
    p.add_argument("--n-objects", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-res", type=int, default=128, help="rays")
    p.add_argument("--obs-noise", type=float, default=0.05, help="observation noise std")
    p.add_argument("--death-on-collision", action=argparse.BooleanOptionalAction, default=True,
                   help="object contact ends the episode")
    p.add_argument("--death-on-wall", action=argparse.BooleanOptionalAction, default=True,
                   help="touching a frustum wall ends the episode (walls still bounce)")
    p.add_argument("--reset-noise-frames", type=int, default=3, help="noise frames shown at a restart")
    p.add_argument("--interval", type=int, default=60, help="milliseconds between live frames")
    p.add_argument("--save", default=None, help="write a GIF here with a scripted driver instead of playing")
    p.add_argument("--frames", type=int, default=150, help="frames for --save")
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    world = build_world(args)
    driver = {"human": None, "random": RandomDriver(world.n, seed=args.seed),
              "avoid": HeuristicAvoidDriver()}[args.driver]
    if args.save:
        Emulator(world, driver=driver or HeuristicAvoidDriver()).save(args.save, frames=args.frames,
                                                                       fps=args.fps)
    else:
        Emulator(world, driver=driver).run(interval=args.interval)


if __name__ == "__main__":
    main()
