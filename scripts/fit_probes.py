#!/usr/bin/env python
"""Fit a Rayworld run's probes for an extra probe target (categorical, or snapped pos@<partition>).

The scorer only reads these fits from the probe cache, so run this before notebooks/master_eval.ipynb:
the run's probes go to runs/<run>/probes/, the floors (--random-init, --observation) to
runs/_baselines/rayworld/<instance>/probes/. A token run is probed on its token inputs.

    python scripts/fit_probes.py --run rayworld/8-ray --target appearance-fac [--random-init | --observation]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from pim.environments.layout import baselines_dir  # noqa: E402
from pim.environments.rayworld import arms as rwa  # noqa: E402
from pim.environments.rayworld.grid_target import categorical_target, snapped_target  # noqa: E402
from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.baselines import random_init_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# the fit sizes the scorer reads these probes at (notebooks/master_eval.ipynb SETTINGS)
N_SEQ = 30_000          # SETTINGS["rw_probe_seqs"]
CAT_N_SEQ = 200_000     # SETTINGS["rw_cat_probe_seqs"]
CAT_EPOCHS = 50         # SETTINGS["rw_cat_probe_epochs"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run id, e.g. rayworld/8-ray")
    p.add_argument("--target", required=True, help="e.g. appearance-fac, grid-16x8, pos@appearance")
    p.add_argument("--basis", default="frustum",
                   help="basis the probes are keyed under (the scorer's first basis)")
    p.add_argument("--families", nargs="+", default=("linear", "mlp"))
    group = p.add_mutually_exclusive_group()
    group.add_argument("--random-init", action="store_true",
                       help="fit the random-init floor (same architecture, untrained, seed 0)")
    group.add_argument("--observation", action="store_true",
                       help="fit the observation floor (right-aligned causal history)")
    p.add_argument("--n-seq", type=int, default=N_SEQ,
                   help="sequences of a regression target (pos@<partition>)")
    p.add_argument("--cat-n-seq", type=int, default=CAT_N_SEQ,
                   help="sequences of a categorical target (from probe_250k)")
    p.add_argument("--cat-epochs", type=int, default=CAT_EPOCHS, help="epochs of a categorical target")
    p.add_argument("--cache-dir", default=None, help="write the probes here instead")
    a = p.parse_args()

    run_dir = REPO / "runs" / a.run
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg["data"]["env"] != "rayworld":
        raise SystemExit("extra probe targets are Rayworld targets")
    if categorical_target(a.target) is None and snapped_target(a.target) is None:
        raise SystemExit(f"{a.target!r} is not an extra probe target; the scorer fits the "
                         f"regression probes itself")
    inst = cfg["data"]["instance"]
    recipe = rwa.probe_recipe(a.target, inst, n_seq=a.n_seq, cat_n_seq=a.cat_n_seq,
                              cat_epochs=a.cat_epochs)
    if categorical_target(a.target) is not None:
        note = f"rw_cat_probe_seqs = {a.cat_n_seq} and rw_cat_probe_epochs = {a.cat_epochs}"
    else:
        note = f"rw_probe_seqs = {a.n_seq}"

    model, info = load_checkpoint(run_dir / "best_model.pt", device="cpu")
    span = int(getattr(model, "state_span", 39))
    enc = {}
    if info.arch.endswith("_tokens"):
        from pim.environments.rayworld import token_bench as tkb
        from pim.environments.rayworld.tokens import FrameVocab

        encoder, tag = tkb.token_encoder(FrameVocab.load(run_dir / "vocab.npz"))
        enc = {"encoder": encoder, "encoder_tag": tag}
    floors = baselines_dir("rayworld", inst) / "probes"
    if a.random_init:
        model, cache_dir, what = (random_init_model(info.arch, info.model_config, seed=0, device=DEV),
                                  floors, f"random-init {info.arch}")
    elif a.observation:
        model, cache_dir, what = None, floors, "observation floor"
    else:
        model, cache_dir, what = model.to(DEV), run_dir / "probes", a.run
    if a.cache_dir:
        cache_dir = Path(a.cache_dir)
    print(f"{what} · {a.target} · basis {a.basis} · {recipe}", flush=True)
    print(f"note: notebooks/master_eval.ipynb reads these probes only when its SETTINGS have {note}",
          flush=True)

    for fam in a.families:
        t0 = time.time()
        if a.observation:
            _, st = rwa.observation_probes(target=a.target, family=fam, basis_name=a.basis,
                                           span=span, cache_dir=cache_dir, align="right",
                                           log=print, **recipe)
            print(f"  {fam}: skill {probe_skill_from_stats(st):+.4f}"
                  f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        else:
            fits = rwa.fit_probes(model, target=a.target, family=fam, basis_name=a.basis,
                                  cache_dir=cache_dir, log=print, **recipe, **enc)
            best = max(fits, key=lambda e: probe_skill_from_stats(fits[e][1]))
            print(f"  {fam}: best point {best} skill {probe_skill_from_stats(fits[best][1]):+.4f}"
                  f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
