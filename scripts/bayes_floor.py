#!/usr/bin/env python
"""The Bayes floor of next-step prediction on an instance -> runs/_baselines/<env>/<instance>/bayes_floor.json.

Othello: exact (pim.environments.othello.bayes; about a minute per instance, CPU). Rayworld: estimated by posterior
sampling over the initial state (pim.environments.rayworld.bayes; minutes per instance on a GPU).

    python scripts/bayes_floor.py [--instance othello/standard rayworld/8-ray] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.layout import baselines_dir  # noqa: E402

PAPER_INSTANCES = ("othello/standard", "othello/adjacent-flip", "othello/adjacent-noflip",
                   "othello/standard-noflip", "rayworld/standard", "rayworld/blink",
                   "rayworld/128-ray", "rayworld/16-ray", "rayworld/8-ray", "rayworld/5-ray")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", nargs="+", default=list(PAPER_INSTANCES), metavar="ENV/INSTANCE")
    ap.add_argument("--n-seq", type=int, default=None,
                    help="sequences (default: every Othello test game, the first 1000 Rayworld)")
    ap.add_argument("--particles", type=int, default=512, help="Rayworld sampler particles")
    ap.add_argument("--sweeps", type=int, default=40, help="Rayworld MH sweeps per frame")
    ap.add_argument("--init-sweeps", type=int, default=500, help="Rayworld MH sweeps at frame 0")
    ap.add_argument("--exact-draws", type=int, default=20_000_000,
                    help="Rayworld draws of the exact position-0 check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="bayes_floor.json", help="file name under the baseline directory")
    ap.add_argument("--force", action="store_true", help="recompute a file already at the current version")
    a = ap.parse_args()
    for key in a.instance:
        env, _, inst = key.partition("/")
        if env == "othello":
            from pim.environments.othello import bayes as ob

            version = ob.FLOOR_VERSION
            run = lambda: ob.bayes_floor(inst, **({"n_games": a.n_seq} if a.n_seq else {}))  # noqa: E731
        elif env == "rayworld":
            from pim.environments.rayworld import bayes as rb

            version = rb.FLOOR_VERSION
            run = lambda: rb.bayes_floor(inst, n_seq=a.n_seq or rb.N_FLOOR_SEQ,  # noqa: E731
                                         particles=a.particles, sweeps=a.sweeps,
                                         init_sweeps=a.init_sweeps, exact_draws=a.exact_draws,
                                         seed=a.seed, device=a.device)
        else:
            raise SystemExit(f"{key!r}: give <env>/<instance> with env othello or rayworld")
        out = baselines_dir(env, inst) / a.out
        if out.exists() and not a.force and json.loads(out.read_text()).get("version") == version:
            print(f"{key}: {out.relative_to(REPO)} is current, skipped", flush=True)
            continue
        res = run()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1, default=float) + "\n")
        os.replace(tmp, out)
        print(f"{key}: wrote {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
