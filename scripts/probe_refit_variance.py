#!/usr/bin/env python
"""Probe-refit spread: how much Probe Skill and the editors move when only the probe's seed changes.

The linear probes (Rayworld: each --targets entry, frustum basis; Othello: mine/theirs) and, on
Othello, the inverse map are refit with seeds 0..N-1 (seed 0 is the scored fit) and PI, IM and IM-NN
are rerun; written to runs/<run>/variance.json under "probe_seeds". Othello takes one --seeds value.

    python scripts/probe_refit_variance.py --run rayworld/8-ray --targets appearance-fac --seeds 6
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

from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

# the scorer's settings (notebooks/master_eval.ipynb)
RW_PROBE_SEQS = 30_000      # SETTINGS["rw_probe_seqs"]
RW_BENCH_N = 1000           # SETTINGS["rw_bench_n"]
OTH_PROBE_GAMES = 20_000    # SETTINGS["oth_probe_games"]
BASIS = "frustum"           # SETTINGS["rw_bases"][0], the basis categorical probes are keyed under
EI = "edit_index_symdiff"


def _sd(x) -> float | None:
    return float(np.std(x, ddof=1)) if len(x) > 1 else None


def _alphas(arms: list[dict], editor: str) -> list[float]:
    """The alpha grid the run's scored arms swept for one editor."""
    return sorted({float(r["alpha"]) for r in arms if r["editor"].split("[")[0] == editor})


def _skill_summary(seeds: dict) -> dict:
    n = len(seeds)
    sk_best = np.array([s["skill_by_point"][s["best_point"]] for s in seeds.values()])
    sk_pts = np.array([s["skill_by_point"] for s in seeds.values()])          # (seeds, points)
    per_seed = sk_pts.std(axis=1, ddof=1)
    return {"best_skill": {"mean": float(sk_best.mean()), "sd": _sd(sk_best)},
            "best_point": {"values": [s["best_point"] for s in seeds.values()],
                           "n_distinct": len({s["best_point"] for s in seeds.values()})},
            "skill_sd_across_points_per_seed": {"mean": float(per_seed.mean()),
                                                "min": float(per_seed.min()),
                                                "max": float(per_seed.max())},
            "skill_sd_across_seeds_per_point": ([float(x) for x in sk_pts.std(axis=0, ddof=1)]
                                                if n > 1 else None)}


def _editor_summary(seeds: dict, names, labels) -> dict:
    out = {}
    for name in names:
        for label in labels:
            vals = [s["editors"].get(label, {}).get(name) for s in seeds.values()]
            vals = [v for v in vals if v]
            if not vals:
                continue
            ei = np.array([v["edit_index"] for v in vals])
            fd = np.array([v["fidelity_ratio"] for v in vals])
            out[f"{name}@{label}"] = {
                "edit_index": {"mean": float(ei.mean()), "sd": _sd(ei)},
                "fidelity_ratio": {"mean": float(fd.mean()), "sd": _sd(fd)},
                "points_used": sorted({v["point"] for v in vals})}
    return out


def rayworld(run_dir: Path, targets, n_seeds_per_target) -> dict:
    """``probe_seeds`` entries per target: linear probes refit per seed, PI swept at each seed's own
    best point, the seed-0 best point and (where scored) the run's best PI point."""
    from pim.environments.rayworld import arms as rwa
    from pim.environments.rayworld import bench as rwb
    from pim.environments.rayworld.grid_target import categorical_target

    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    scores = json.loads((run_dir / "scores.json").read_text())
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=rwa.DEV)
    model.eval()
    out, t0 = {}, time.time()
    for target, n_seeds in zip(targets, n_seeds_per_target):
        cat = categorical_target(target) is not None
        recipe = rwa.probe_recipe(target, inst, n_seq=RW_PROBE_SEQS)
        alphas = _alphas(scores["bases"][target if cat else BASIS]["arms"], "PI")
        b = rwb.load_bench(model, n=RW_BENCH_N, target=target, basis_name=BASIS, instance=inst)
        u = rwa.unsteered(model, b)
        canon = scores["bases"].get(target, {}).get("best", {})
        seeds = {}
        for seed in range(n_seeds):
            fits = rwa.fit_probes(model, target=target, family="linear", basis_name=BASIS,
                                  cache_dir=run_dir / "probes", log=None, seed=seed, **recipe)
            skill = [probe_skill_from_stats(fits[e][1]) for e in sorted(fits)]
            best = int(np.argmax(skill))
            if seed == 0:
                best0 = best
            rec = {"skill_by_point": skill, "best_point": best, "editors": {}}
            points = [("own_best", best), ("seed0_best", best0)]
            if canon.get("PI"):
                points.append(("canonical_edit_best", int(canon["PI"]["point"])))
            for label, pt in points:
                arms = rwa.pinv_arm(model, b, {pt: fits[pt]}, alphas, dims="all")
                for r in arms:
                    r["fidelity_ratio"] = rwa.fidelity_ratio(r, u)
                bb = max(arms, key=lambda r: r["edit_index"])
                rec["editors"][label] = {"PI": {"edit_index": bb["edit_index"],
                                                "fidelity_ratio": bb["fidelity_ratio"],
                                                "alpha": bb["alpha"], "dims": bb.get("dims"),
                                                "point": pt}}
            seeds[str(seed)] = rec
            e = rec["editors"]["own_best"]["PI"]
            print(f"  {target} seed {seed}: best point {best} skill {skill[best]:.4f} | PI "
                  f"{e['edit_index']:+.3f}/{e['fidelity_ratio']:.2f}  [{(time.time() - t0) / 60:.1f} min]",
                  flush=True)
        out[target] = {"summary": {"n_seeds": n_seeds, "recipe": recipe, **_skill_summary(seeds),
                                   "editors": _editor_summary(seeds, ("PI",), ("own_best", "seed0_best",
                                                                               "canonical_edit_best"))},
                       "seeds": seeds}
    return out


def othello(run_dir: Path, n_seeds: int, im_seeds: int, im_points: str) -> dict:
    """``probe_seeds["mine"]`` (linear probe grid refit per seed, PI at three points) and
    ``probe_seeds["inverse_map"]`` (the inverse map refit per seed, IM and IM-NN)."""
    from pim.environments.othello import arms as oa
    from pim.environments.othello import case_targets, load_benchmark
    from pim.environments.othello import corpus as oc
    from pim.metrics.set_editability import move_fidelity_ratio
    from pim.scoring.othello import _probe_games

    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    scores = json.loads((run_dir / "scores.json").read_text())
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=oa.DEV)
    model.eval()
    npnt = n_points(model)
    data = _probe_games(OTH_PROBE_GAMES, inst)
    bench = load_benchmark(inst)
    cur, tgt = case_targets(bench)
    uns = oa.unsteered_probs(model, bench)
    alphas = _alphas(scores["arms"], "PI")

    def canon_point(editor: str) -> int:
        """The run's best-edit point for one editor under the symmetric-difference Edit Index."""
        sub = [r for r in scores["arms"] if r["editor"] == editor
               or r["editor"].startswith(editor + "[") or r["editor"].startswith(editor + "@")]
        return int(max(sub, key=lambda r: r[EI])["point"])

    out, t0 = {}, time.time()
    if n_seeds:
        seeds = {}
        for seed in range(n_seeds):
            grid = oa.fit_probe_grid(model, data, families=("linear",), seed=seed,
                                     cache_dir=run_dir / "probes", log=None)
            lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
            skill = [probe_skill_from_stats(next(st for st in grid.stats
                                                 if st["point"] == p and st["family"] == "linear"))
                     for p in range(npnt)]
            best = int(np.argmax(skill))
            if seed == 0:
                best0 = best
            rec = {"skill_by_point": skill, "best_point": best, "editors": {}}
            for label, pt in (("own_best", best), ("seed0_best", best0),
                              ("canonical_edit_best", canon_point("PI"))):
                arms = []
                for al in alphas:
                    pr, card = oa.linear_arm(model, bench, lin, tgt, cur, alpha=al, points={pt})
                    arms.append({"alpha": al, "edit_index": card[EI],
                                 "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post)})
                rec["editors"][label] = {"PI": {**max(arms, key=lambda r: r["edit_index"]), "point": pt}}
            seeds[str(seed)] = rec
            e = rec["editors"]["own_best"]["PI"]
            print(f"  mine seed {seed}: best point {best} skill {skill[best]:.4f} | PI "
                  f"{e['edit_index']:+.3f}/{e['fidelity_ratio']:.2f}  [{(time.time() - t0) / 60:.1f} min]",
                  flush=True)
        u = oa.unsteered(model, bench)
        out["mine"] = {"summary": {"n_seeds": n_seeds, "families": ["linear"], "n_games": OTH_PROBE_GAMES,
                                   "canonical_point_by": EI, **_skill_summary(seeds),
                                   "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))},
                                   "edit_index": EI,
                                   "editors": _editor_summary(seeds, ("PI",), ("own_best", "seed0_best",
                                                                               "canonical_edit_best"))},
                       "seeds": seeds}
    if im_seeds:
        points = None if im_points == "all" else [canon_point("IM")]
        seeds = {}
        for seed in range(im_seeds):
            recs, st = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes",
                                       n_games=OTH_PROBE_GAMES, seed=seed, points=points,
                                       uns_probs=uns, log=None)
            rec = {"g_r2": st["g_r2"], "nn_r2": st.get("nn_r2"), "editors": {}}
            for name in ("IM", "IM-NN"):
                b = max((r for r in recs if r["editor"] == name), key=lambda r: r[EI])
                rec["editors"][name] = {"edit_index": b[EI], "fidelity_ratio": b.get("fidelity_ratio"),
                                        "point": b["point"], "g_r2": b["g_r2"]}
            seeds[str(seed)] = rec
            print(f"  inverse map seed {seed}: g R² {max(st['g_r2']):+.3f} | " + " ".join(
                f"{k} {v['edit_index']:+.3f}/{(v['fidelity_ratio'] or float('nan')):.2f} (pt {v['point']})"
                for k, v in rec["editors"].items()) + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        summ = {"n_seeds": im_seeds, "points": im_points, "n_games": OTH_PROBE_GAMES, "edit_index": EI,
                "canonical_point": canon_point("IM"), "canonical_point_by": EI, "editors": {}}
        for name in ("IM", "IM-NN"):
            vals = [x["editors"][name] for x in seeds.values()]
            ei = np.array([v["edit_index"] for v in vals], float)
            fd = np.array([np.nan if v["fidelity_ratio"] is None else v["fidelity_ratio"] for v in vals], float)
            gr = np.array([v["g_r2"] for v in vals], float)
            summ["editors"][name] = {"edit_index": {"mean": float(ei.mean()), "sd": _sd(ei)},
                                     "fidelity_ratio": {"mean": float(np.nanmean(fd)), "sd": _sd(fd)},
                                     "g_r2_at_arm": {"mean": float(gr.mean()), "sd": _sd(gr)},
                                     "points_used": sorted({v["point"] for v in vals})}
        nn = [x["nn_r2"] for x in seeds.values() if x["nn_r2"]]
        if nn:
            summ["nn_r2_max"] = {"mean": float(np.mean([max(x) for x in nn])),
                                 "sd": _sd([max(x) for x in nn])}
        out["inverse_map"] = {"summary": summ, "seeds": seeds}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="seeds per target (Rayworld: one per --targets entry; default 10 6; Othello: one number, default 10)")
    ap.add_argument("--targets", nargs="+", default=("full", "appearance-fac"), help="Rayworld probe targets")
    ap.add_argument("--im-seeds", type=int, default=None, help="Othello inverse-map seeds (default: --seeds; 0 skips)")
    ap.add_argument("--im-points", choices=("canonical", "all"), default="canonical",
                    help="Othello: refit the inverse map at the run's IM point, or at every point")
    ap.add_argument("--out", default="variance.json", help="file name under the run directory")
    a = ap.parse_args()
    run_dir = REPO / "runs" / a.run
    env = json.loads((run_dir / "config.json").read_text())["data"]["env"]
    if env == "rayworld":
        seeds = a.seeds or [10, 6][: len(a.targets)]
        if len(seeds) != len(a.targets):
            raise SystemExit("give one --seeds value per --targets entry")
        entries = rayworld(run_dir, a.targets, seeds)
    else:
        n = (a.seeds or [10])[0]
        entries = othello(run_dir, n, n if a.im_seeds is None else a.im_seeds, a.im_points)
    path = run_dir / a.out
    var = json.loads(path.read_text()) if path.exists() else {}
    var.setdefault("probe_seeds", {}).update(entries)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(var, indent=1, default=float) + "\n")
    os.replace(tmp, path)
    print(f"wrote {path.relative_to(REPO)} ({', '.join(entries)})")


if __name__ == "__main__":
    main()
