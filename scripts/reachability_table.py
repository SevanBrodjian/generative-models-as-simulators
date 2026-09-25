#!/usr/bin/env python
"""Each Othello editor's Table 2 result split by whether the edited board is legal (reachable).

Step 1 (per instance, CPU, cached): every edit case decided by exhaustive search
(pim.environments.othello.reachability) -> runs/_baselines/othello/<instance>/reachability.json.
Step 2 (per run, GPU): PI, GS and IM re-run at their Table 2 setting (pim.metrics.selection.best_arm),
checked against scores.json, then scored per subset -> runs/<run>/editability_by_reachability.json.

    python scripts/reachability_table.py [--runs othello/standard] [--classify-only]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.layout import baselines_dir  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello import reachability as rch  # noqa: E402
from pim.environments.othello.bench import cases_path, load_benchmark  # noqa: E402

VERSION = "1.0"
PAPER_RUNS = ["othello/standard", "othello/adjacent-flip", "othello/adjacent-noflip",
              "othello/standard-noflip"]
# the scorer's Othello settings (notebooks/master_eval.ipynb)
PROBE_GAMES = 20_000    # SETTINGS["oth_probe_games"]
GS_STEPS = 100          # SETTINGS["oth_gs_steps"]
GS_BETA = 0.2           # SETTINGS["oth_gs_beta"]
EI = "edit_index_symdiff"
STATUSES = ("reachable", "unreachable", "undecided")


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    os.replace(tmp, path)
    print(f"  wrote {path.relative_to(REPO)}", flush=True)


def _verdicts_path(instance: str) -> Path:
    return baselines_dir("othello", instance) / "reachability.json"


def classify(instance: str, budget: int, workers: int) -> dict:
    """Every edit case of ``instance`` decided; cached per instance and budget, resumable."""
    out = _verdicts_path(instance)
    cases = pickle.load(open(cases_path(instance), "rb"))
    if out.exists():
        prev = json.loads(out.read_text())
        if (prev.get("version") == rch.VERSION and prev.get("budget") == budget
                and prev.get("n_cases") == len(cases)):
            print(f"  {instance}: cached verdicts {prev['counts']}", flush=True)
            return prev
    rules = oc.rules_of(instance)
    part = out.with_suffix(".partial.jsonl")       # verdicts appended as they arrive
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            r = json.loads(line)
            if r.get("budget") == budget and r.get("version") == rch.VERSION:
                done[r["case"]] = r
        print(f"  {instance}: resuming, {len(done)} cases already decided", flush=True)
    args = [(i, [int(x) for x in c["history"]], int(c["pos_int"]), rules, budget)
            for i, c in enumerate(cases) if i not in done]
    t0, recs = time.time(), list(done.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool, open(part, "a") as fh:
        for k, r in enumerate(pool.imap_unordered(rch.decide_case, args, chunksize=1), 1):
            recs.append(r)
            fh.write(json.dumps({**r, "budget": budget, "version": rch.VERSION}) + "\n")
            fh.flush()
            if k % 100 == 0 or k == len(args):
                print(f"  {instance}: {len(recs)}/{len(cases)} decided {rch.summarize(recs)} "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    recs = [{k: v for k, v in r.items() if k not in ("budget", "version")} for r in recs]
    recs.sort(key=lambda r: r["case"])
    assert [r["case"] for r in recs] == list(range(len(cases))), "missing or duplicate cases"
    res = {"instance": instance, "version": rch.VERSION, "script_version": VERSION, "budget": budget,
           "rules": rules, "n_cases": len(cases), "counts": rch.summarize(recs),
           "definition": "legal = some legal game of the same length (passes only when forced) ends on "
                         "exactly the flipped board with the same player to move; exhaustive search, "
                         "witnesses replayed",
           "minutes": round((time.time() - t0) / 60, 1), "records": recs}
    _write(out, res)
    part.unlink(missing_ok=True)
    return res


def split_run(run: str, verdicts: dict, tol_index: float, tol_ratio: float) -> dict:
    """Each editor at its Table 2 setting on the full bench, per-case outputs split by verdict."""
    import torch

    from pim.environments.othello import arms as oa
    from pim.environments.othello import case_targets
    from pim.metrics.selection import GUARD, best_arm
    from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio
    from pim.models import load_checkpoint, n_points
    from pim.scoring.othello import _probe_games

    run_dir = REPO / "runs" / run
    S = json.loads((run_dir / "scores.json").read_text())
    inst = S["instance"]
    rules = oc.rules_of(inst)
    bench = load_benchmark(inst)
    cases = pickle.load(open(cases_path(inst), "rb"))
    assert list(bench.pos_int) == [int(c["pos_int"]) for c in cases], "bench and case file disagree"
    status = np.array([r["status"] for r in verdicts["records"]])
    assert len(status) == bench.n_cases

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    pre, post = bench.legal_pre, bench.legal_post
    cur, tgt = case_targets(bench)
    uns = oa.unsteered_probs(model, bench)
    data = _probe_games(PROBE_GAMES, inst)
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    npnt = n_points(model)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
    mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(npnt)}

    probs, arms = {}, {}
    for ed in ("PI", "GS", "IM"):
        a = best_arm(S["arms"], ed, EI, guard=GUARD)
        pt, al = int(a["point"]), float(a["alpha"])
        if ed == "PI":
            pr, _ = oa.linear_arm(model, bench, lin, tgt, cur, alpha=al, points={pt})
        elif ed == "GS":
            pr, _ = oa.grad_steer_arm(model, bench, mlp, pt, alpha=al, n_steps=GS_STEPS,
                                      beta=GS_BETA, target_labels=tgt)
        else:
            _, _, by = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes",
                                       n_games=PROBE_GAMES, points=[pt], uns_probs=uns,
                                       log=None, return_probs=True)
            pr = by[("IM", pt)]
        full_ei = float(np.nanmean(edit_index_legal(pr, pre, post, "symdiff")))
        full_ratio = float(move_fidelity_ratio(pr, uns, post))
        d_ei, d_ratio = full_ei - float(a[EI]), full_ratio - float(a["fidelity_ratio"])
        ok = abs(d_ei) <= tol_index and abs(d_ratio) <= tol_ratio
        print(f"  {ed}: point {pt} alpha {al:g} | full bench EI {full_ei:+.4f} (scores.json {a[EI]:+.4f}) "
              f"ratio {full_ratio:.4f} (scores.json {a['fidelity_ratio']:.4f}) {'ok' if ok else 'MISMATCH'}",
              flush=True)
        if not ok:
            raise SystemExit(f"{run} {ed}: the re-run arm does not reproduce scores.json "
                             f"(dEI {d_ei:+.4f}, dratio {d_ratio:+.4f}); nothing written")
        probs[ed] = pr
        arms[ed] = {"point": pt, "alpha": al, "within_guard": bool(a.get("within_guard", True)),
                    "full_bench": {"index": full_ei, "fidelity": 1.0 - full_ratio},
                    "scores_json": {"index": float(a[EI]), "fidelity": 1.0 - float(a["fidelity_ratio"])}}

    ei_uns = np.asarray(edit_index_legal(uns, pre, post, "symdiff"), float)
    ei_ed = {ed: np.asarray(edit_index_legal(p, pre, post, "symdiff"), float) for ed, p in probs.items()}
    split = {}
    for s in STATUSES:
        idx = np.where(status == s)[0]
        row = {"n": int(len(idx))}
        if len(idx):
            row["unedited_index"] = float(np.nanmean(ei_uns[idx]))
            row["editors"] = {}
            for ed in probs:
                x = ei_ed[ed][idx]
                row["editors"][ed] = {
                    "index": float(np.nanmean(x)),
                    "index_se": (float(np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum()))
                                 if len(idx) > 1 else float("nan")),
                    "fidelity": 1.0 - float(move_fidelity_ratio(probs[ed][idx], uns[idx],
                                                                [post[i] for i in idx]))}
        split[s] = row
    res = {"run": run, "instance": inst, "version": VERSION,
           "reachability": {"file": str(_verdicts_path(inst).relative_to(REPO)),
                            "version": verdicts["version"], "budget": verdicts["budget"],
                            "counts": verdicts["counts"]},
           "edit_index": "symmetric difference (legal-set)",
           "selection": "pim.metrics.selection.best_arm, as Table 2",
           "arms": arms, "split": split}
    _write(run_dir / "editability_by_reachability.json", res)
    return res


def _fmt(x, sign=True):
    return "--" if x is None or not np.isfinite(x) else (f"{x:+.2f}" if sign else f"{x:.2f}")


def summary(results: list[dict]) -> None:
    print("\nvariant               target        n   unedited |  PI index/fid |  GS index/fid |  IM index/fid")
    for r in results:
        for s, lab in (("reachable", "legal"), ("unreachable", "illegal"), ("undecided", "undecided")):
            row = r["split"][s]
            if s == "undecided" and row["n"] == 0:
                continue
            e = row.get("editors", {})
            cells = "  ".join(f"{_fmt(e[ed]['index'])} / {_fmt(e[ed]['fidelity'], False)}" if ed in e
                              else "  -- / --  " for ed in ("PI", "GS", "IM"))
            print(f"{r['instance']:20} {lab:10} {row['n']:5}   {_fmt(row.get('unedited_index'))}   |  {cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=PAPER_RUNS, help="Othello run ids")
    ap.add_argument("--budget", type=int, default=5_000_000,
                    help="search nodes per case before it is undecided")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--classify-only", action="store_true", help="step 1 only (no GPU)")
    ap.add_argument("--tol-index", type=float, default=0.01, help="allowed Edit Index mismatch")
    ap.add_argument("--tol-ratio", type=float, default=0.02, help="allowed fidelity ratio mismatch")
    a = ap.parse_args()
    t0, results = time.time(), []
    for run in a.runs:
        inst = json.loads((REPO / "runs" / run / "config.json").read_text())["data"]["instance"]
        print(f"{run} ({inst})", flush=True)
        v = classify(inst, a.budget, a.workers)
        if not a.classify_only:
            results.append(split_run(run, v, a.tol_index, a.tol_ratio))
    if results:
        summary(results)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
