"""The appendix Othello edit figures: one bench case per variant, the four variants across.

Rows: the pre-edit board shaded uniformly over its legal moves (Unedited GT), the flipped board
shaded over its legal moves (Edited GT), then the model's next-move distribution after each editor's
write (PI, GS, IM), shaded fully at 2% probability. Outlined: the flipped tile and every square whose
legality the flip changes (the squares the Edit Index scores), cyan before the edit, magenta after it.
Each editor is written at the setting Table 2 reports (``pim.metrics.selection.best_arm``), with the
probes and inverse maps from each run's cache. Seed ``k`` draws one case per variant from
``numpy.random.default_rng(k)``.

    python scripts/figures/qualitative_othello.py [--seeds 1 2 3 4 5] [--recompute]
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import style
import torch
from matplotlib.gridspec import GridSpec

from pim.environments.othello import arms as oa
from pim.environments.othello import case_targets, load_benchmark
from pim.environments.othello import corpus as oc
from pim.environments.othello.data import canonical_vocab, tokens_and_labels
from pim.metrics.selection import best_arm
from pim.metrics.set_editability import uniform_over_legal
from pim.models import load_checkpoint

plt = style.plt
VARIANTS = [("Standard", "standard"), ("Adjacent Flip", "adjacent-flip"), ("Adjacent NoFlip", "adjacent-noflip"),
            ("Standard NoFlip", "standard-noflip")]
SEEDS = (1, 2, 3, 4, 5)
EDITORS = ("PI", "GS", "IM")
KEY = "edit_index_symdiff"              # the Othello Edit Index Table 2 reports
PROBE_GAMES, GS_STEPS, GS_BETA = 20_000, 100, 0.2    # the scorer's probe corpus and GS settings
CACHE = style.CACHE / "qualitative_othello.pkl"
CONDITIONS = ("Unedited GT", "Edited GT") + EDITORS


def boards(bench, rules: dict) -> np.ndarray:
    """(n_cases, 64) pre-edit boards (0 white, 1 empty, 2 black), each history replayed under the rules."""
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [None] * bench.n_cases
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    bd = tokens_and_labels(hist, **rules)
    return np.stack([bd.labels[i, len(hist[i]) - 1] for i in range(bench.n_cases)])


def probe_games(inst: str, rules: dict):
    """The first ``PROBE_GAMES`` games of the instance's probe split, labeled: the probes' fit data."""
    tok, ln = oc.load(oc.build(oc.N_TRAIN_GAMES, log=lambda s: None, only=("probe",), instance=inst)["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:n]] for row, n in zip(tok[:PROBE_GAMES], ln[:PROBE_GAMES])],
                             **rules)


@torch.no_grad()
def compute(inst: str) -> dict:
    """Every bench case of one variant: the boards, legal sets and each editor's next-move distribution."""
    run_dir = style.REPO / "runs" / "othello" / inst
    scores = json.loads((run_dir / "scores.json").read_text())
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=oa.DEV)
    model.eval()
    data = probe_games(inst, rules)
    bench = load_benchmark(inst)
    cur, tgt = case_targets(bench)
    pre = boards(bench, rules)
    ar = np.arange(bench.n_cases)
    assert (pre[ar, bench.pos_int] != 1).all(), "an edited tile must be occupied"
    post = pre.copy()
    post[ar, bench.pos_int] = 2 - pre[ar, bench.pos_int]
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(model.n_layers + 1)}
    mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(model.n_layers + 1)}
    arms = {ed: best_arm(scores["arms"], ed, KEY) for ed in EDITORS}
    pi, gs, im = arms["PI"], arms["GS"], arms["IM"]
    probs = {"PI": oa.linear_arm(model, bench, lin, tgt, cur, alpha=pi["alpha"], points={pi["point"]})[0],
             "GS": oa.grad_steer_arm(model, bench, mlp, gs["point"], alpha=gs["alpha"], n_steps=GS_STEPS,
                                     beta=GS_BETA, target_labels=tgt)[0]}
    _, _, pb = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes", n_games=PROBE_GAMES,
                               points=[im["point"]], log=None, return_probs=True)
    probs["IM"] = pb[("IM", im["point"])]
    print(f"  othello/{inst}: settings {({e: (a['point'], a['alpha']) for e, a in arms.items()})}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"board_pre": pre, "board_post": post, "pos": bench.pos_int.copy(),
            "legal_pre": [list(map(int, x)) for x in bench.legal_pre],
            "legal_post": [list(map(int, x)) for x in bench.legal_post], "probs": probs,
            "settings": {e: (int(a["point"]), float(a["alpha"])) for e, a in arms.items()}}


def variants(names, recompute: bool = False) -> dict:
    """{instance: compute(instance)}, from ``outputs/cache/`` where present, else computed (GPU)."""
    have = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    missing = [inst for inst in names if recompute or inst not in have]
    for inst in missing:
        have[inst] = compute(inst)
    if missing:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(have))
    return {inst: have[inst] for inst in names}


def marked(col: dict, i: int) -> list[int]:
    """The flipped tile plus every square whose legality the flip changes."""
    return sorted({int(col["pos"][i])} | (set(col["legal_pre"][i]) ^ set(col["legal_post"][i])))


def draw_case(ax, col: dict, i: int, cond: str, lw: float) -> None:
    """One board of one condition, with its outline."""
    if cond == "Unedited GT":
        style.board(ax, col["board_pre"][i], uniform_over_legal(col["legal_pre"][i]), marked=marked(col, i),
                    mark_color=style.ORIGIN_C, lw=lw)
        return
    p = uniform_over_legal(col["legal_post"][i]) if cond == "Edited GT" else col["probs"][cond][i]
    style.board(ax, col["board_post"][i], p, marked=marked(col, i), mark_color=style.DEST_C, lw=lw)


def draw(cols: dict, picks: dict, seed: int) -> None:
    """Variants across, conditions down."""
    names = [n for n, _ in VARIANTS]
    fig = plt.figure(figsize=(1.55 * len(names) + 1.6, 1.55 * len(CONDITIONS) + 0.9))
    gs = GridSpec(len(CONDITIONS), len(names), figure=fig, left=0.11, right=0.995, top=0.92, bottom=0.01,
                  wspace=0.08, hspace=0.08)
    for c, (name, inst) in enumerate(VARIANTS):
        for r, cond in enumerate(CONDITIONS):
            ax = fig.add_subplot(gs[r, c])
            draw_case(ax, cols[inst], picks[inst], cond, lw=2.56)
            if r == 0:
                ax.set_title(name, fontsize=13, pad=6, color=style.TEXT)
            if c == 0:
                y = (ax.get_position().y0 + ax.get_position().y1) / 2
                fig.text(ax.get_position().x0 - 0.008, y, cond, ha="right", va="center", fontsize=11,
                         color=style.TEXT, fontweight="bold" if cond == "Edited GT" else "normal")
    style.mark_key(fig, markersize=9, fontsize=10, borderpad=0.4, loc="upper left", bbox_to_anchor=(0.005, 0.995))
    style.save(fig, f"appendix/othello_qualitative/othello_edits_seed{seed}_cols", dpi=170)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="case-draw seeds, one figure each")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached distributions")
    a = ap.parse_args()
    cols = variants([inst for _, inst in VARIANTS], a.recompute)
    for s in a.seeds:
        rng = np.random.default_rng(s)
        picks = {inst: int(rng.choice(len(cols[inst]["pos"]), size=1, replace=False)[0]) for _, inst in VARIANTS}
        print(f"seed {s}: cases {picks}")
        draw(cols, picks, s)


if __name__ == "__main__":
    main()
