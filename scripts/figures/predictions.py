"""The two appendix prediction figures: the trained models predict well before any edit.

Rayworld (six variants in two bands of three): for two of the first 32 bench cases per variant
(``numpy.random.default_rng(0)``), the last 8 observed frames above a dashed line and 15 steps below
it of the unedited world (Ground truth), the model's free-run (Prediction) and their difference.
Othello (four variants): three bench cases per variant (one ``numpy.random.default_rng(0)`` shared in
column order), each board shaded by its true legal moves above the same board shaded by the model's
next-move distribution (fully at 2% probability).

    python scripts/figures/predictions.py
"""
from __future__ import annotations

import argparse

import numpy as np
import style
import torch
from qualitative_othello import boards

from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello import load_benchmark
from pim.environments.rayworld import arms as rwa
from pim.environments.rayworld import bench as rwb
from pim.environments.rayworld.bench import EF, K_ROLL
from pim.metrics.set_editability import uniform_over_legal
from pim.models import load_checkpoint

plt = style.plt
RW_PANELS = [("a", "standard"), ("b", "blink"), ("c", "128-ray"), ("d", "16-ray"), ("e", "8-ray"), ("f", "5-ray")]
OTH_PANELS = [("a", "standard"), ("b", "adjacent-flip"), ("c", "adjacent-noflip"), ("d", "standard-noflip")]
N_BENCH, N_PER, N_OTH, SEED = 32, 2, 3, 0
N_CTX, PER_BAND = 8, 3
ROWS = [("Ground truth", "gt"), ("Prediction", "pred"), ("Difference", None)]
# Rayworld geometry, inches
LEFT, RIGHT, TOP, BOTTOM, BAND_GAP = 0.46, 0.44, 0.20, 0.30, 0.30
COL_GAP, PAIR_GAP, ROW_GAP, FRAME_H = 0.06, 0.18, 0.10, 0.034
PDF_PPI = 600                           # keeps every one of the 128 rays in the PDF's rasters


@torch.no_grad()
def rayworld_panel(inst: str) -> dict:
    """The drawn cases' observed history, the unedited world's clean continuation and the free-run."""
    model, _ = load_checkpoint(style.REPO / "runs" / "rayworld" / inst / "best_model.pt", device=rwb.DEV)
    model.eval()
    b = rwb.load_bench(model, n=N_BENCH, target="full", basis_name="cartesian", instance=inst)
    cases = sorted(int(i) for i in np.random.default_rng(SEED).choice(b.n, N_PER, replace=False))
    roll = rwa.unsteered_rollout(model, b)
    del model
    torch.cuda.empty_cache()
    return {"cases": cases, "context": b.obs[cases, EF - N_CTX:EF], "gt": b.zones.gt_unedited_traj[cases, :K_ROLL],
            "pred": roll[cases, :K_ROLL]}


def rayworld_figure(panels: list):
    width = style.TEXT_WIDTH_IN
    bands = [panels[i:i + PER_BAND] for i in range(0, len(panels), PER_BAND)]
    ncol = N_PER * PER_BAND
    panel_w = (width - LEFT - RIGHT - COL_GAP * (ncol - PER_BAND) - PAIR_GAP * (PER_BAND - 1)) / ncol
    panel_h = FRAME_H * (N_CTX + K_ROLL)
    band_h = TOP + len(ROWS) * panel_h + (len(ROWS) - 1) * ROW_GAP
    height = len(bands) * band_h + (len(bands) - 1) * BAND_GAP + BOTTOM
    fig = plt.figure(figsize=(width, height), dpi=PDF_PPI)
    for bi, band in enumerate(bands):
        y0 = bi * (band_h + BAND_GAP)
        x = LEFT
        for letter, d in band:
            fig.text(x / width, 1 - (y0 + 0.5 * TOP) / height, f"({letter})", ha="left", va="center", fontsize=9,
                     fontweight="bold", color=style.TEXT)
            for k in range(N_PER):
                for r, (_, key) in enumerate(ROWS):
                    ax = style.box(fig, x, y0 + TOP + r * (panel_h + ROW_GAP), panel_w, panel_h)
                    if key is None:     # the error below the line; the observed frames above it are not predictions
                        err = style.error(d["pred"][k], d["gt"][k])
                        style.waterfall(ax, np.zeros((N_CTX, err.shape[1])), err, diff=True)
                    else:
                        style.waterfall(ax, d["context"][k], d[key][k])
                x += panel_w + COL_GAP
            x += PAIR_GAP - COL_GAP
        for r, (label, key) in enumerate(ROWS):
            y = y0 + TOP + r * (panel_h + ROW_GAP)
            style.arrow(fig, LEFT - 0.12, y, LEFT - 0.12, y + panel_h)
            yc = 1 - (y + panel_h / 2) / height
            fig.text((LEFT - 0.19) / width, yc, "time", rotation=90, ha="center", va="center", fontsize=8,
                     color=style.TEXT)
            fig.text((LEFT - 0.36) / width, yc, label, rotation=90, ha="center", va="center", fontsize=9,
                     color=style.TEXT)
            if key is None:
                style.colorbar(style.box(fig, width - RIGHT + 0.07, y, 0.07, panel_h), "prediction − truth",
                               labelpad=3, length=2, pad=3.5)
    y_bot = len(bands) * band_h + (len(bands) - 1) * BAND_GAP
    style.arrow(fig, LEFT, y_bot + 0.10, LEFT + panel_w, y_bot + 0.10)
    fig.text((LEFT + panel_w / 2) / width, 1 - (y_bot + 0.20) / height, "ray", ha="center", va="center",
             fontsize=8, color=style.TEXT)
    style.line_key(fig, LEFT + panel_w + COL_GAP + 0.08, y_bot + 0.10, [("h", style.EDIT_LINE, "free-run start")])
    return fig


@torch.no_grad()
def othello_panels() -> list:
    """(letter, [(board, legal-move shading, model shading)] per drawn case) per variant."""
    rng = np.random.default_rng(SEED)
    out = []
    for letter, inst in OTH_PANELS:
        bench = load_benchmark(inst)
        ids = sorted(int(i) for i in rng.choice(bench.n_cases, N_OTH, replace=False))
        model, _ = load_checkpoint(style.REPO / "runs" / "othello" / inst / "best_model.pt", device=oa.DEV)
        model.eval()
        probs = oa.unsteered_probs(model, bench)
        pre = boards(bench, oc.rules_of(inst))
        out.append((letter, [(pre[i], uniform_over_legal([int(s) for s in bench.legal_pre[i]]), probs[i])
                             for i in ids]))
        print(f"  othello/{inst}: cases {ids}", flush=True)
        del model
        torch.cuda.empty_cache()
    return out


def othello_figure(panels: list):
    """Variants across; per case a (legal moves, model) pair of rows, the cases stacked."""
    W, cg, rg, bg, top, left = style.TEXT_WIDTH_IN, 0.12, 0.12, 0.30, 0.2, 0.62
    s = (W - left - (len(panels) - 1) * cg) / len(panels)
    block = 2 * s + rg
    H = top + N_OTH * block + (N_OTH - 1) * bg
    fig = plt.figure(figsize=(W, H))
    for c, (letter, cases) in enumerate(panels):
        x0 = left + c * (s + cg)
        fig.text(x0 / W, 1 - 0.5 * top / H, f"({letter})", ha="left", va="center", fontsize=9, fontweight="bold")
        for e, (squares, legal, model) in enumerate(cases):
            for r, (label, probs) in enumerate((("legal moves", legal), ("model", model))):
                y0 = H - top - e * (block + bg) - (r + 1) * s - r * rg
                style.board(fig.add_axes([x0 / W, y0 / H, s / W, s / H]), squares, probs)
                if c == 0:
                    fig.text((left - 0.08) / W, (y0 + s / 2) / H, label, ha="right", va="center", fontsize=8)
    return fig


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    panels = []
    for letter, inst in RW_PANELS:
        d = rayworld_panel(inst)
        print(f"  rayworld/{inst}: cases {d['cases']}", flush=True)
        panels.append((letter, d))
    style.save(rayworld_figure(panels), "appendix/rayworld_predictions")
    style.save(othello_figure(othello_panels()), "appendix/othello_predictions")


if __name__ == "__main__":
    main()
