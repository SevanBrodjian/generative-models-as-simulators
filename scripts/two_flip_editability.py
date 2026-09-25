#!/usr/bin/env python
"""Two-token Othello edits: are the editors effective once the flipped board is legal?

Each bench case keeps its flipped token s and pairs it with a token t of the opposite color, so
each color's count is preserved; the first t whose two-token board is reachable (legal) and the
first that is unreachable (illegal) are kept, decided exactly (pim.environments.othello.reachability).
Cases are tried in order until --n have both partners (a case without both is dropped, with the
reason recorded); --pool extends the bench with further cases built the same way. PI, GS and IM
then edit the single flip and both pairs -> runs/<run>/two_flip_editability.json.

    python scripts/make_othello_edits.py --instance adjacent-noflip --n 3000
    python scripts/two_flip_editability.py --run othello/adjacent-noflip --pool 3000 [--no-legal]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.counterfactual import replay  # noqa: E402
from pim.environments.othello.reachability import search, state_of  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

VERSION = "1.0"
GROUPS = ("single", "legal", "illegal")
# the scorer's Othello settings (notebooks/master_eval.ipynb)
PROBE_GAMES = 20_000    # SETTINGS["oth_probe_games"]
GS_STEPS = 100          # SETTINGS["oth_gs_steps"]
GS_BETA = 0.2           # SETTINGS["oth_gs_beta"]


def _case_pairs(args: tuple) -> dict | None:
    """One case: its first reachable and first unreachable partner, in an order seeded by the case
    index (with ``require_legal=False``, every partner is searched and counted)."""
    i, c, rules, budget, require_legal = args
    h, s = [int(x) for x in c["history"]], int(c["pos_int"])
    b, w, mover = state_of(replay(h, rules))
    opp = w if (b >> s) & 1 else b                              # partners: the other color's tokens
    partners = [t for t in range(64) if (opp >> t) & 1]
    rng = np.random.default_rng(i)
    legal = illegal = None
    counts = {"reachable": 0, "unreachable": 0, "undecided": 0}
    for t in [partners[j] for j in rng.permutation(len(partners))]:
        tgt = (b ^ (1 << s) ^ (1 << t), w ^ (1 << s) ^ (1 << t), mover)
        assert bin(tgt[0]).count("1") == bin(b).count("1") and bin(tgt[1]).count("1") == bin(w).count("1")
        v = search(tgt, len(h), rules, order={m: k for k, m in enumerate(h)}, budget=budget)
        counts[v.status] += 1
        if v.status == "reachable" and legal is None:
            wb = replay(v.witness, rules)
            assert wb is not None and state_of(wb) == tgt, "witness failed the vendored engine's replay"
            legal = {"t": t, "witness": v.witness}
        elif v.status == "unreachable" and illegal is None:
            illegal = {"t": t}
        if require_legal and legal and illegal:
            break
    kept = bool(illegal and (legal or not require_legal))
    why = None if kept else ("no unreachable partner" if not illegal else "no reachable partner")
    if not kept and not legal and counts["undecided"]:
        why += " (some searches undecided)"
    return {"case": i, "s": s, "legal": legal, "illegal": illegal, "partners_searched": counts,
            "kept": kept, "dropped_because": why}


def find_pairs(cases: list[dict], rules: dict, n: int, budget: int, require_legal: bool = True,
               workers: int = 1) -> tuple[list[dict], list[dict]]:
    """The first ``n`` qualifying bench cases, and every case tried with the reason a dropped one was
    dropped, in bench order (the pool returns the serial result)."""
    from multiprocessing import Pool

    args = [(i, c, rules, budget, require_legal) for i, c in enumerate(cases)]
    out, tried = [], []
    with Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap(_case_pairs, args, chunksize=1):
            tried.append(r)
            if r["kept"]:
                out.append(r)
                if len(out) >= n:
                    pool.terminate()
                    break
    return out, tried


def _grid(arms: list[dict], editor: str) -> tuple[list[float], list[int]]:
    """The (alphas, points) the run's scored arms swept for one editor."""
    sub = [r for r in arms if r["editor"] == editor]
    return sorted({float(r["alpha"]) for r in sub}), sorted({int(r["point"]) for r in sub})


def main() -> None:
    import torch

    from pim.environments.othello import arms as oa
    from pim.environments.othello import case_targets
    from pim.environments.othello.bench import load_benchmark
    from pim.environments.othello.data import MINE, THEIRS, canonical_vocab, tokens_and_labels
    from pim.metrics.selection import GUARD, best_arm
    from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio
    from pim.models import load_checkpoint, n_points
    from pim.scoring.othello import _probe_games

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="othello/adjacent-noflip", help="Othello run id")
    ap.add_argument("--n", type=int, default=1000,
                    help="cases to keep (each gives one single flip, one legal and one illegal pair); the default tries the whole bench")
    ap.add_argument("--budget", type=int, default=2_000_000, help="search nodes per partner before it is skipped")
    ap.add_argument("--no-legal", action="store_true",
                    help="variant without legal pairs: search every partner, keep single flip and illegal pair")
    ap.add_argument("--workers", type=int, default=16, help="processes for the pair search")
    ap.add_argument("--pool", type=int, default=1000,
                    help="cases to draw from: the 1000-case bench, or a larger set made by "
                         "scripts/make_othello_edits.py --n <pool> (same recipe and seed, so it begins with the bench)")
    a = ap.parse_args()
    groups = ("single", "illegal") if a.no_legal else GROUPS
    two = [g for g in groups if g != "single"]
    t0 = time.time()
    run_dir = REPO / "runs" / a.run
    S = json.loads((run_dir / "scores.json").read_text())
    inst = S["instance"]
    rules = oc.rules_of(inst)
    cases = pickle.load(open(cases_path(inst), "rb"))
    if a.pool != len(cases):
        from pim.environments.layout import othello_cases_file

        pool = pickle.load(open(othello_cases_file(inst, a.pool), "rb"))
        assert len(pool) == a.pool and all(x["history"] == y["history"] and x["pos_int"] == y["pos_int"]
                                           for x, y in zip(cases, pool)), "the pool does not begin with the bench"
        cases = pool
    pairs, tried = find_pairs(cases, rules, a.n, a.budget, require_legal=not a.no_legal, workers=a.workers)
    dropped = {}
    for r in tried:
        if not r["kept"]:
            dropped[r["dropped_because"]] = dropped.get(r["dropped_because"], 0) + 1
    searched = {k: sum(p_["partners_searched"][k] for p_ in pairs) for k in ("reachable", "unreachable", "undecided")}
    print(f"{len(pairs)} cases kept, from the first {pairs[-1]['case'] + 1} cases; partners searched "
          f"{searched} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if a.no_legal and searched["reachable"]:
        print(f"  WARNING: --no-legal, but {searched['reachable']} reachable pairs exist on this variant", flush=True)

    # one bench holding every group: the same histories once per group, the edit differing per group
    sub = [cases[p["case"]] for p in pairs]
    bench1 = benchmark_from_cases(sub, **rules)
    n = len(sub)
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [[int(x) for x in c["history"]] for c in sub]
    bd = tokens_and_labels([[itos[canonical_vocab()[m]] for m in h] for h in hist], **rules)
    s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(n)])
    cur, tgt = case_targets(bench1)

    def swap(board, sq):
        board[sq] = THEIRS if board[sq] == MINE else MINE

    def legal_set(h, squares):
        b = OthelloBoardState(**rules)
        b.update(h, prt=False)
        for q in squares:
            b.state[q // 8, q % 8] *= -1
        return sorted(b.get_valid_moves())

    boards, legal_post = {}, {}
    for g in groups:
        bb = s_pre.copy()
        sets = []
        for i, p in enumerate(pairs):
            sq = [p["s"]] if g == "single" else [p["s"], p[g]["t"]]
            for q in sq:
                swap(bb[i], q)
            sets.append(legal_set(hist[i], sq))
        boards[g], legal_post[g] = bb, sets
    ref = s_pre.copy()
    ref[np.arange(n), bench1.pos_int] = tgt
    assert (boards["single"] == ref).all(), "single-flip boards differ from inverse_arms' own construction"
    assert legal_post["single"] == [list(x) for x in bench1.legal_post], "single-flip legal sets differ from the bench's"

    all_cases = sub * len(groups)
    bench = benchmark_from_cases(all_cases, **rules)
    bench = dataclasses.replace(bench, legal_post=[x for g in groups for x in legal_post[g]])
    post_all = np.concatenate([boards[g] for g in groups])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    uns = oa.unsteered_probs(model, bench)
    data = _probe_games(PROBE_GAMES, inst)
    _, stats, probs_by = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes",
                                         n_games=PROBE_GAMES, uns_probs=uns, log=None,
                                         return_probs=True, post_boards=post_all)
    pre_all = bench.legal_pre
    res = {g: {"unedited_index": None, "points": []} for g in groups}
    for k, g in enumerate(groups):
        sl = slice(k * n, (k + 1) * n)
        pre_g, post_g = pre_all[sl], legal_post[g]
        res[g]["unedited_index"] = float(np.nanmean(edit_index_legal(uns[sl], pre_g, post_g, "symdiff")))
        for ell in sorted({pt for (ed, pt) in probs_by if ed == "IM"}):
            pr = probs_by[("IM", ell)][sl]
            ei = np.asarray(edit_index_legal(pr, pre_g, post_g, "symdiff"), float)
            res[g]["points"].append({"point": int(ell), "index": float(np.nanmean(ei)),
                                     "index_se": float(np.nanstd(ei, ddof=1) / np.sqrt(np.isfinite(ei).sum())),
                                     "fidelity": 1.0 - float(move_fidelity_ratio(pr, uns[sl], post_g))})

    # PI and GS on the same cases, over every setting the run's scored arms swept
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    npnt = n_points(model)
    lin = {q: grid.probes[("mine", "linear", "sequence", q)] for q in range(npnt)}
    mlp = {q: grid.probes[("mine", "mlp", "sequence", q)] for q in range(npnt)}
    alpha_pi, _ = _grid(S["arms"], "PI")
    alpha_gs, gs_layers = _grid(S["arms"], "GS")
    t2 = {ed: best_arm(S["arms"], ed, "edit_index_symdiff", guard=GUARD) for ed in ("PI", "GS", "IM")}
    full = load_benchmark(inst)                     # the single-token path must reproduce scores.json
    fcur, ftgt = case_targets(full)
    funs = oa.unsteered_probs(model, full)
    for ed in ("PI", "GS"):
        pt, al = int(t2[ed]["point"]), float(t2[ed]["alpha"])
        pr = (oa.linear_arm(model, full, lin, ftgt, fcur, alpha=al, points={pt})[0] if ed == "PI" else
              oa.grad_steer_arm(model, full, mlp, pt, alpha=al, n_steps=GS_STEPS, beta=GS_BETA,
                                target_labels=ftgt)[0])
        e = float(np.nanmean(edit_index_legal(pr, full.legal_pre, full.legal_post, "symdiff")))
        r = float(move_fidelity_ratio(pr, funs, full.legal_post))
        assert abs(e - t2[ed]["edit_index_symdiff"]) < 0.01 and abs(r - t2[ed]["fidelity_ratio"]) < 0.02, \
            f"{ed}: the single-token path does not reproduce scores.json ({e:+.4f} vs {t2[ed]['edit_index_symdiff']:+.4f})"
        print(f"  check {ed} at the Table 2 setting (pt {pt}, a {al:g}) on the full bench: {e:+.4f} / "
              f"ratio {r:.4f} = scores.json", flush=True)
    b1 = benchmark_from_cases(sub, **rules)
    b2 = dataclasses.replace(benchmark_from_cases(sub * len(two), **rules),
                             legal_post=[x for g in two for x in legal_post[g]])
    cur1, tgt1 = case_targets(b1)
    cur2, tgt2 = case_targets(b2)
    t_sq = np.array([pr_[g]["t"] for g in two for pr_ in pairs], dtype=np.int64)
    cur_t = np.concatenate([s_pre] * len(two))[np.arange(len(two) * n), t_sq]
    assert set(np.unique(cur_t)) <= {MINE, THEIRS}, "a partner cell is empty"
    tgt_t = np.where(cur_t == MINE, THEIRS, MINE)
    uns1, uns2 = oa.unsteered_probs(model, b1), oa.unsteered_probs(model, b2)
    parts = {"single": (slice(0, n), uns1, b1.legal_pre, legal_post["single"])}
    for k, g in enumerate(two):
        sl = slice(k * n, (k + 1) * n)
        parts[g] = (sl, uns2[sl], b2.legal_pre[sl], legal_post[g])
    arms = {g: [] for g in groups}

    def add(ed, pt, al, pr1, pr2):
        for g, (sl, u, pre_g, post_g) in parts.items():
            pr = pr1 if g == "single" else pr2[sl]
            ei = np.asarray(edit_index_legal(pr, pre_g, post_g, "symdiff"), float)
            arms[g].append({"editor": ed, "point": int(pt), "alpha": float(al),
                            "edit_index_symdiff": float(np.nanmean(ei)),
                            "index_se": float(np.nanstd(ei, ddof=1) / np.sqrt(np.isfinite(ei).sum())),
                            "fidelity_ratio": float(move_fidelity_ratio(pr, u, post_g))})
    for pt in range(npnt):
        for al in alpha_pi:
            add("PI", pt, al, oa.linear_arm(model, b1, lin, tgt1, cur1, alpha=al, points={pt})[0],
                oa.linear_arm(model, b2, lin, tgt2, cur2, alpha=al, points={pt},
                              second=(t_sq, cur_t, tgt_t))[0])
    print(f"  PI swept [{(time.time() - t0) / 60:.1f} min]", flush=True)
    for ls in gs_layers:
        for al in alpha_gs:
            add("GS", ls, al, oa.grad_steer_arm(model, b1, mlp, ls, alpha=al, n_steps=GS_STEPS,
                                                beta=GS_BETA, target_labels=tgt1)[0],
                oa.grad_steer_arm(model, b2, mlp, ls, alpha=al, n_steps=GS_STEPS, beta=GS_BETA,
                                  target_labels=tgt2, second=(t_sq, tgt_t))[0])
    print(f"  GS swept [{(time.time() - t0) / 60:.1f} min]", flush=True)
    for g in groups:
        for q in res[g]["points"]:
            arms[g].append({"editor": "IM", "point": q["point"], "alpha": 1.0, "edit_index_symdiff": q["index"],
                            "index_se": q["index_se"], "fidelity_ratio": 1.0 - q["fidelity"]})
    reported = {g: {ed: best_arm(arms[g], ed, "edit_index_symdiff", guard=GUARD) for ed in ("PI", "GS", "IM")}
                for g in groups}
    at_t2 = {g: {ed: next(x for x in arms[g] if x["editor"] == ed and x["point"] == int(t2[ed]["point"])
                          and x["alpha"] == float(t2[ed]["alpha"])) for ed in ("PI", "GS", "IM")}
             for g in groups}

    out = {"run": a.run, "instance": inst, "groups_run": list(groups), "partners_searched": searched,
           "version": VERSION, "n_cases": n, "case_pool": len(cases), "cases_tried": len(tried), "cases_dropped": dropped, "budget": a.budget,
           "editor": "IM (canonical inverse_arms, post_boards)",
           "reported": reported, "at_table2_setting": at_t2,
           "table2_setting": {ed: {"point": int(t2[ed]["point"]), "alpha": float(t2[ed]["alpha"])}
                              for ed in ("PI", "GS", "IM")},
           "arms_by_group": arms,
           "edit_index": "symmetric difference (legal-set)", "groups": res, "g_r2": stats["g_r2"],
           "pairs": pairs, "minutes": round((time.time() - t0) / 60, 1)}
    path = run_dir / "two_flip_editability.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1, default=float) + "\n")
    os.replace(tmp, path)
    print(f"wrote {path.relative_to(REPO)}")
    print(f"\nIM, n = {n} cases per group, Edit Index (symdiff) ± SE / Edit Fidelity")
    print("point | " + " | ".join(f"{g:^24}" for g in groups))
    print("unedited " + "  ".join(f"| {res[g]['unedited_index']:+.3f}{'':18}" for g in groups))
    for j in range(len(res["single"]["points"])):
        print(f"{res['single']['points'][j]['point']:>5}    " + "  ".join(
            f"| {res[g]['points'][j]['index']:+.3f} ± {res[g]['points'][j]['index_se']:.3f} / "
            f"{res[g]['points'][j]['fidelity']:+.2f}" for g in groups))
    for title, tab in (("each group at the setting Table 2's rule picks within that group", reported),
                       ("each group at the run's Table 2 setting", at_t2)):
        print(f"\n{title}: index ± SE / Edit Fidelity (point, alpha)")
        for g in groups:
            print(f"  {g:8} " + "   ".join(
                f"{ed} {tab[g][ed]['edit_index_symdiff']:+.3f} ± {tab[g][ed]['index_se']:.3f} / "
                f"{1 - tab[g][ed]['fidelity_ratio']:+.2f} (pt {tab[g][ed]['point']}, a {tab[g][ed]['alpha']:g})"
                for ed in ("PI", "GS", "IM")))


if __name__ == "__main__":
    main()
