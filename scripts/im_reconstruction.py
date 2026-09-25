#!/usr/bin/env python
"""IM reconstruction control on Othello: write g(pre-edit board) at each residual point, with no edit.

g is the run's cached inverse map, written into the latent state at the last position. The output's
error against the pre-edit legal set over the unedited model's own -> runs/<run>/im_reconstruction.json.

    python scripts/im_reconstruction.py [--run othello/standard]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VERSION = "1.0"
PROBE_GAMES = 20_000   # SETTINGS["oth_probe_games"] of notebooks/master_eval.ipynb (the IM's fit games)


def pre_boards(bench, rules: dict) -> np.ndarray:
    """(n_cases, 64) unedited boards in the mover's frame, as ``inverse_arms`` builds them."""
    from pim.environments.othello import case_targets
    from pim.environments.othello.data import canonical_vocab, tokens_and_labels

    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [None] * bench.n_cases
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    bd = tokens_and_labels(hist, **rules)
    s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(bench.n_cases)])
    cur, _ = case_targets(bench)
    assert (s_pre[np.arange(bench.n_cases), bench.pos_int] == cur).all(), "boards disagree with the bench"
    return s_pre


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="othello/standard", help="Othello run id")
    a = ap.parse_args()

    import torch

    from pim.environments.othello import arms as oa
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.bench import load_benchmark
    from pim.metrics.set_editability import move_fidelity_ratio, move_rmse
    from pim.models import load_checkpoint, n_points
    from pim.scoring.othello import _probe_games

    t0 = time.time()
    run_dir = REPO / "runs" / a.run
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg["data"]["env"] != "othello":
        raise SystemExit(f"{a.run} is not an Othello run")
    inst = cfg["data"]["instance"]
    rules = oc.rules_of(inst)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    bench = load_benchmark(inst)
    s_pre = pre_boards(bench, rules)
    uns = oa.unsteered_probs(model, bench)
    data = _probe_games(PROBE_GAMES, inst)
    fitted = []
    _, _, probs = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes",
                                  n_games=PROBE_GAMES, uns_probs=uns, return_probs=True,
                                  post_boards=s_pre, log=fitted.append)
    if fitted:
        print(f"  {len(fitted)} inverse maps were not in the probe cache and were fit", flush=True)
    points = list(range(n_points(model)))
    model_error = float(move_rmse(uns, bench.legal_pre))
    g_error = [float(move_rmse(probs[("IM", p)], bench.legal_pre)) for p in points]
    ratio = [float(move_fidelity_ratio(probs[("IM", p)], uns, bench.legal_pre)) for p in points]

    out = {"run": a.run, "points": points, "ratio": ratio, "model_error": model_error,
           "g_error": g_error, "n_cases": int(bench.n_cases), "version": VERSION}
    path = run_dir / "im_reconstruction.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, path)
    print(f"{a.run}: {bench.n_cases} cases, unedited error vs the pre-edit legal set {model_error:.4f}")
    print("point  error with g(pre-edit board)  / unedited")
    for p, e, r in zip(points, g_error, ratio):
        print(f"{p:>5}  {e:.4f}  {r:6.2f}x")
    print(f"wrote {path.relative_to(REPO)}  [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
