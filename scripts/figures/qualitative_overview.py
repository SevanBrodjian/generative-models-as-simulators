"""The main-text qualitative edit figure: (a) Rayworld, (b) Othello, side by side.

(a) Four columns: the Standard model editing the continuous state on two worlds (seeds 0 and 2), and
the 128-ray and 5-ray models editing the categorical appearance bins on one shared world (seed 1).
Rows: the last 8 observed frames, the true next frame without and with the teleport, and each editor's
next-step prediction above its error against the edited truth. (b) One bench case per Othello variant:
the pre-edit and flipped boards shaded over their legal moves, then the model's next-move distribution
after PI and IM, cropped to a 5 x 5 window around the squares the flip changes. Settings, predictions
and caches are those of ``qualitative_rayworld.py`` and ``qualitative_othello.py``.
The seeds are the first three whose teleport changes at least two rays of the 5-ray frame by at least
0.2. The Othello cases are fixed bench cases in which at least three squares change legality within a
5 x 5 window, picked for per-case Edit Indices close to their variant's means.

    python scripts/figures/qualitative_overview.py [--recompute]
"""
from __future__ import annotations

import argparse

import qualitative_othello as qo
import qualitative_rayworld as qr
import style
from matplotlib.gridspec import GridSpec

plt = style.plt
# (column title, model, instance, seed, block)
RW_COLUMNS = [("Ex. 1", "Standard", "standard", 0, "cont"), ("Ex. 2", "Standard", "standard", 2, "cont"),
              ("Ex. 3", "128-ray", "128-ray", 1, "cat"), ("Ex. 3", "5-ray", "5-ray", 1, "cat")]
BLOCK_NAME = {"cont": "continuous", "cat": "categorical"}
OTH_CASES = {"standard": 342, "adjacent-flip": 261, "adjacent-noflip": 39}
OTH_NAMES = {"standard": "Standard", "adjacent-flip": "Adjacent\nFlip", "adjacent-noflip": "Adjacent\nNoFlip"}
OTH_ROWS = ("Unedited GT", "Edited GT", "PI", "IM")
EDITORS = ("PI", "GS", "IM")

WIDTH, KEY_GAP, GUTTER, LETTER = 6.10, 0.56, 0.52, 11.5   # inches; GUTTER holds both panels' row labels; pt
# (a): row heights in strip units; gutters in inches
CTX, STRIP, ERR, GAP, GAP_GT, GAP_EDIT = 0.42, 1.0, 0.7, 0.3, 0.65, 0.9
RIGHT, TOP, BOTTOM = 0.40, 0.46, 0.03
# (b): board side, gaps, margins (inches)
BOARD, COL_GAP, ROW_GAP, PI_GAP, OTH_TOP, PAD = 0.60, 0.16, 0.05, 0.10, 0.30, 0.03


def rayworld_rows(n_ctx: int) -> list:
    rows = [("context", n_ctx * CTX), ("gap", GAP), ("unedited_gt", STRIP), ("gap", GAP_GT),
            ("edited_gt", STRIP), ("gap", GAP_EDIT)]
    for ed in EDITORS:
        rows += [(f"{ed}:pred", STRIP), (f"{ed}:err", ERR), ("gap", GAP)]
    return rows[:-1]


def runs(values) -> list:
    """(value, first index, last index) of each run of equal consecutive values."""
    out = []
    for i, v in enumerate(values):
        if out and out[-1][0] == v:
            out[-1][2] = i
        else:
            out.append([v, i, i])
    return out


def rayworld_panel(F, cols: list) -> None:
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    rows = rayworld_rows(len(cols[0]["context"]))
    gs = GridSpec(len(rows), len(cols), figure=F, height_ratios=[h for _, h in rows], left=GUTTER / W,
                  right=1 - RIGHT / W, top=1 - TOP / H, bottom=BOTTOM / H, wspace=0.13, hspace=0.0)
    for c, ((title, _, _, _, blk), col) in enumerate(zip(RW_COLUMNS, cols)):
        for r, (kind, _) in enumerate(rows):
            if kind == "gap":
                continue
            ax = F.add_subplot(gs[r, c])
            ed, _, sub = kind.partition(":")
            if not sub:
                style.strip(ax, col[kind])
            elif col[blk][ed] is None:
                ax.set_axis_off()
                continue
            elif sub == "pred":
                style.strip(ax, col[blk][ed])
            else:
                style.strip(ax, style.error(col[blk][ed], col["edited_gt"]), diff=True)
            if kind != "context":
                style.locators(ax, col["origin_x"], col["dest_x"], lw=0.8)
            if r == 0:
                ax.set_title(title, pad=3, fontsize=7, color=style.TEXT)
            if c == len(cols) - 1 and not sub:           # time runs down the history, t is the predicted frame
                marks = [(1 - 0.5 / len(col["context"]), "0"), (0.5 / len(col["context"]), "t-1")] \
                    if kind == "context" else [(0.5, "t")]
                for y, s in marks:
                    ax.annotate(s, xy=(1, y), xycoords="axes fraction", xytext=(3, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=7, color=style.TEXT)
            if c == 0 and sub != "err":
                label = {"context": "History", "unedited_gt": "Unedited\nGT", "edited_gt": "Edited\nGT"}.get(kind, ed)
                y = (1 - ERR / STRIP) / 2 if sub else 0.5       # an editor's label centers on prediction + error
                ax.annotate(label, xy=(0, y), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=style.TEXT, linespacing=0.95,
                            fontweight="bold" if kind == "edited_gt" else "normal")
    y = gs[0, 0].get_position(F).y1 + 0.18 / H
    xc = lambda c0, c1: (gs[0, c0].get_position(F).x0 + gs[0, c1].get_position(F).x1) / 2  # noqa: E731
    for blk, c0, c1 in runs([BLOCK_NAME[c[4]] for c in RW_COLUMNS]):
        F.text(xc(c0, c1), y + 0.135 / H, blk, ha="center", va="bottom", fontsize=8, color=style.TEXT)
    for model, c0, c1 in runs([c[1] for c in RW_COLUMNS]):
        F.text(xc(c0, c1), y, model, ha="center", va="bottom", fontsize=8, color=style.TEXT)
    r0 = next(r for r, (k, _) in enumerate(rows) if k.endswith(":pred"))
    r1 = max(r for r, (k, _) in enumerate(rows) if k.endswith(":err"))
    b0, b1 = gs[r0, -1].get_position(F), gs[r1, -1].get_position(F)
    style.colorbar(F.add_axes([b0.x1 + 0.07 / W, b1.y0, 0.05 / W, b0.y1 - b1.y0]), r"prediction$_t$ − truth$_t$")
    F.text(0.02 / W, 1 - 0.02 / H, "(a)", ha="left", va="top", fontsize=LETTER, fontweight="bold", color=style.TEXT)


def window(col: dict, i: int, side: int = 5) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1): a side x side crop centered on the outlined squares, kept on the board."""
    rc = [divmod(s, 8) for s in qo.marked(col, i)]
    out = []
    for lo, hi in ((min(r for r, _ in rc) - 1, max(r for r, _ in rc) + 1),
                   (min(c for _, c in rc) - 1, max(c for _, c in rc) + 1)):
        lo, hi = max(0, lo), min(7, hi)
        lo = min(max(lo - (side - (hi - lo + 1)) // 2, 0), 8 - side)
        out += [lo, lo + side - 1]
    return tuple(out)


def othello_size() -> tuple[float, float]:
    n = len(OTH_CASES)
    return (GUTTER + n * BOARD + (n - 1) * COL_GAP + PAD,
            OTH_TOP + len(OTH_ROWS) * BOARD + 2 * ROW_GAP + PI_GAP + PAD)


def othello_panel(F, cols: dict) -> None:
    W, H = F.bbox.width / F.dpi, F.bbox.height / F.dpi
    y = H - OTH_TOP
    for r, cond in enumerate(OTH_ROWS):
        y -= BOARD + (0 if r == 0 else (PI_GAP if cond == "PI" else ROW_GAP))
        for c, (inst, i) in enumerate(OTH_CASES.items()):
            ax = F.add_axes([(GUTTER + c * (BOARD + COL_GAP)) / W, y / H, BOARD / W, BOARD / H])
            qo.draw_case(ax, cols[inst], i, cond, lw=1.0)
            r0, r1, c0, c1 = window(cols[inst], i)
            ax.set_xlim(c0, c1 + 1)
            ax.set_ylim(7 - r1, 8 - r0)
            if r == 0:
                ax.set_title(OTH_NAMES[inst], pad=5, fontsize=8, color=style.TEXT, linespacing=1.25)
            if c == 0:
                label = {"Unedited GT": "Unedited\nGT", "Edited GT": "Edited\nGT"}.get(cond, cond)
                ax.annotate(label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=style.TEXT, linespacing=0.95,
                            fontweight="bold" if cond == "Edited GT" else "normal")
    F.text(0.02 / W, 1 - 0.02 / H, "(b)", ha="left", va="top", fontsize=LETTER, fontweight="bold", color=style.TEXT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="ignore the cached predictions")
    a = ap.parse_args()
    rw = qr.columns([(inst, seed) for _, _, inst, seed, _ in RW_COLUMNS], a.recompute)
    oth = qo.variants(list(OTH_CASES), a.recompute)
    wb, h = othello_size()
    fig = plt.figure(figsize=(WIDTH, h))
    fa, fk, fb = fig.subfigures(1, 3, width_ratios=[WIDTH - wb - KEY_GAP, KEY_GAP, wb], wspace=0.0)
    rayworld_panel(fa, [rw[(inst, seed)] for _, _, inst, seed, _ in RW_COLUMNS])
    style.mark_key(fk, fontsize=7.5, markersize=4.5, loc="center", ncol=1)
    othello_panel(fb, oth)
    style.save(fig, "qualitative_edits_overview")


if __name__ == "__main__":
    main()
