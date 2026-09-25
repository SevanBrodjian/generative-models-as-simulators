#!/usr/bin/env python
"""Simulate and animate one Rayworld scene: the 2D world beside the 1D ray observation over time.

    python scripts/demos/demo.py --seed 8 --n-objects 4 --fixed-reflectivities [--save outputs/demo.gif]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402

from pim.environments.rayworld.config import SimConfig  # noqa: E402
from pim.environments.rayworld.renderer import render_scene  # noqa: E402
from pim.environments.rayworld.sim import simulate  # noqa: E402
from pim.environments.rayworld.viz import animate_scene, save_animation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-objects", type=int, default=3,
                   help="more than 3 needs --fixed-reflectivities")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--obs-res", type=int, default=128, help="rays")
    p.add_argument("--boundary", choices=["bounce", "open", "wrap"], default="bounce")
    p.add_argument("--direction-noise", type=float, default=0.0, help="velocity angle noise per step (radians)")
    p.add_argument("--speed-noise", type=float, default=0.0, help="fractional speed noise per step")
    p.add_argument("--position-noise", type=float, default=0.0, help="position noise std per step")
    p.add_argument("--obs-noise-std", type=float, default=0.04, help="observation noise std")
    p.add_argument("--waterfall-mode", choices=["model", "human"], default="model",
                   help="model: grayscale intensity; human: color and depth")
    p.add_argument("--fixed-reflectivities", action="store_true", help="evenly spaced reflectivities")
    p.add_argument("--always-in-frustum", action="store_true",
                   help="reject trajectories that touch a frustum wall")
    p.add_argument("--save", default=None, help="write the animation here (.gif or .mp4) instead of showing it")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--interval", type=int, default=50, help="milliseconds between frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(seed=args.seed, n_objects=args.n_objects, n_frames=args.frames,
                    obs_res=args.obs_res, boundary=args.boundary,
                    direction_noise_std=args.direction_noise, speed_noise_std=args.speed_noise,
                    position_noise_std=args.position_noise, obs_noise_std=args.obs_noise_std,
                    fixed_reflectivities=args.fixed_reflectivities,
                    always_in_frustum=args.always_in_frustum)
    print(f"simulating seed={cfg.seed} objects={cfg.n_objects} frames={cfg.n_frames}")
    scene = simulate(cfg)
    obs_depth, obs_id, obs_intensity = render_scene(scene)
    anim = animate_scene(scene, obs_depth, obs_id, obs_intensity, interval=args.interval,
                         title=f"seed {cfg.seed}  ·  {scene.positions.shape[1]} objects",
                         waterfall_mode=args.waterfall_mode)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_animation(anim, str(out), fps=args.fps)
    else:
        plt.show()


if __name__ == "__main__":
    main()
