"""Shared look of the paper figures: fonts, colors, output paths and the three drawing primitives
(an observation strip, a waterfall with its free-run, an Othello board).

Importing it selects the Agg backend and puts the repository root on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FIGURES = REPO / "outputs" / "figures"
CACHE = REPO / "outputs" / "cache"

from pim.environments.rayworld.viz import BG_HEX as DARK_BG  # noqa: E402

EDIT_LINE = "#e8ecf5"                   # the edit-frame line, neutral beside the colored locators
TARGET_C, GHOST_C = "#00b050", "#ff3b30"
# signed error (prediction - truth): under-prediction red, over-prediction green, zero the background
DIFF_CMAP = LinearSegmentedColormap.from_list("pim_diff", [GHOST_C, DARK_BG, TARGET_C])

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,                 # TrueType: the PDF text stays text
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
})

TEXT_WIDTH_IN = 5.5                     # the paper's text width
TEXT, FRAME = "black", "#6f6f6f"        # text; the thin border of an observation panel
ORIGIN_C, DEST_C = "#00bcd4", "#ff4fa3"  # cyan: before the edit; magenta: after it
EDITOR_COLORS = {"PI": "#0072B2", "GS": "#D55E00", "IM": "#009E73"}   # Okabe-Ito, one color per editor
BOARD_GREEN, BOARD_LINE, BOARD_TINT = "#33a852", "#1e1e1e", "#ffe600"
TINT_SCALE, TINT_GAMMA, DISC_R = 0.02, 0.6, 0.38   # a square is fully tinted at 2% probability


def save(fig, rel_stem: str, dpi: int = 300) -> Path:
    """``outputs/figures/<rel_stem>.pdf`` and a PNG preview, cropped to the content."""
    stem = FIGURES / rel_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"-> {stem.relative_to(REPO)}.pdf", flush=True)
    return stem


def frame(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor(FRAME)


def strip(ax, img, *, diff: bool = False) -> None:
    """Observation frames (rows = time) in gray on 0..1, or a signed error on the fixed +-1 map."""
    img = np.asarray(img)
    img = img[None] if img.ndim == 1 else img
    if diff:
        ax.imshow(img, cmap=DIFF_CMAP, vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest")
    else:
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor(DARK_BG)
    frame(ax)


def error(pred, truth) -> np.ndarray:
    """Prediction minus truth, the prediction clipped to the drawn range [0, 1] first."""
    return np.clip(np.asarray(pred, float), 0.0, 1.0) - np.asarray(truth, float)


def locators(ax, origin_x: float, dest_x: float, lw: float, alpha: float = 0.95) -> None:
    """Vertical lines at the edited disc's ray centers before (cyan) and after (magenta) the edit."""
    for x, c in ((origin_x, ORIGIN_C), (dest_x, DEST_C)):
        if np.isfinite(x):
            ax.axvline(x, color=c, lw=lw, alpha=alpha)


def waterfall(ax, context, body=None, origin_x=np.nan, dest_x=np.nan, *, diff: bool = False,
              edit_lw: float = 0.8, loc_lw: float = 0.7) -> None:
    """Context frames above a dashed line and ``body`` (a free-run, or its error) below it."""
    img = context if body is None else np.concatenate([context, body], axis=0)
    strip(ax, img, diff=diff)
    if body is not None:
        ax.axhline(len(context) - 0.5, color=EDIT_LINE, lw=edit_lw, ls=(0, (3, 2)))
    locators(ax, origin_x, dest_x, loc_lw, alpha=1.0)


def box(fig, x: float, y: float, w: float, h: float):
    """Axes at ``x``, ``y`` inches from the top left of ``fig``."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def arrow(fig, x0: float, y0: float, x1: float, y1: float) -> None:
    W, H = fig.get_size_inches()
    fig.add_artist(FancyArrowPatch((x0 / W, 1 - y0 / H), (x1 / W, 1 - y1 / H), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=5, lw=0.6, color=TEXT,
                                   shrinkA=0, shrinkB=0))


def line_key(fig, x: float, y: float, entries, *, edit_lw: float = 0.8, loc_lw: float = 0.7) -> None:
    """A dark swatch carrying each line as the panels draw it (``"h"`` dashed, ``"v"`` solid), labeled."""
    W, H = fig.get_size_inches()
    for kind, color, label in entries:
        ax = box(fig, x, y, 0.22, 0.11)
        ax.set_facecolor(DARK_BG)
        frame(ax)
        if kind == "h":
            ax.axhline(0.5, color=color, lw=edit_lw, ls=(0, (3, 2)))
        else:
            ax.axvline(0.5, color=color, lw=loc_lw)
        t = fig.text((x + 0.27) / W, 1 - (y + 0.055) / H, label, ha="left", va="center", fontsize=8, color=TEXT)
        x += 0.27 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + 0.14


def board(ax, squares, probs, *, marked=(), mark_color=None, lw: float = 1.6) -> None:
    """An 8 x 8 Othello board (0 white, 1 empty, 2 black; row-major), each square tinted yellow by
    ``probs``, and the ``marked`` squares outlined in ``mark_color``."""
    g, t = np.array(to_rgb(BOARD_GREEN)), np.array(to_rgb(BOARD_TINT))
    for sq in range(64):
        r, c = divmod(sq, 8)
        a = min(1.0, float(probs[sq]) / TINT_SCALE) ** TINT_GAMMA if probs[sq] > 0 else 0.0
        ax.add_patch(Rectangle((c, 7 - r), 1, 1, facecolor=(1 - a) * g + a * t, edgecolor=BOARD_LINE, linewidth=0.4))
        if squares[sq] != 1:
            ax.add_patch(Circle((c + 0.5, 7 - r + 0.5), DISC_R, facecolor="black" if squares[sq] == 2 else "white",
                                edgecolor="#333333", linewidth=0.5, zorder=3))
    for sq in marked:
        r, c = divmod(int(sq), 8)
        ax.add_patch(Rectangle((c, 7 - r), 1, 1, facecolor="none", edgecolor=mark_color, linewidth=lw, zorder=5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def mark_key(owner, *, fontsize: float, markersize: float, borderpad: float = 0.0, **kw):
    """The cyan pre-edit / magenta post-edit key; ``kw`` goes to ``owner.legend``."""
    handles = [Line2D([], [], marker="o", linestyle="none", color=c, markersize=markersize)
               for c in (ORIGIN_C, DEST_C)]
    return owner.legend(handles=handles, labels=["pre-edit", "post-edit"], fontsize=fontsize, handlelength=0.8,
                        handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0, borderpad=borderpad, **kw)


def colorbar(cax, label: str, *, labelsize: float = 7, fontsize: float = 8, labelpad: float = 2,
             length: float = 1.5, pad: float = 1.5, width: float = 0.5) -> None:
    """The +-1 signed-error scale."""
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=DIFF_CMAP, norm=matplotlib.colors.Normalize(-1.0, 1.0),
                                          orientation="vertical")
    cb.set_ticks([-1.0, 0.0, 1.0])
    cb.ax.tick_params(labelsize=labelsize, length=length, pad=pad, width=width, colors=TEXT)
    cb.outline.set_edgecolor(FRAME)
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=fontsize, labelpad=labelpad, color=TEXT)
