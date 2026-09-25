"""Editability across residual points on standard Othello and standard Rayworld (appendix figure).

Top row: the Edit Index of each editor's reported setting among the arms at each point (GS at its start
point; hollow where no arm at that point has Edit Fidelity >= 0). Bottom row: the inverse map's R² and the
MLP probe's skill per point. Values come from ``pim.figures.tables.by_point``.

    python scripts/figures/editability_by_point.py      # outputs/figures/appendix/editability_over_res_point.{pdf,png}
"""
from __future__ import annotations

import argparse

import style as st  # fonts and the Agg backend; puts the repository root on sys.path

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from pim.figures import tables as T  # noqa: E402

RUNS = {"Othello": "othello/standard", "Rayworld": "rayworld/standard"}
MARK = {"PI": "o", "GS": "s", "IM": "^"}
GRAY, ZERO, SPINE = "#7f7f7f", "#c8c8c8", "#555555"
LEGEND_GAP = 0.18            # inches between the x-axis label and the legend row


def style_ax(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor(SPINE)
    ax.tick_params(labelsize=8, labelcolor=st.TEXT, color=SPINE, length=2.5)
    ax.set_xticks(range(9))
    ax.set_xlim(-0.35, 8.35)


def draw_edit_index(ax, d) -> bool:
    """Each editor's Edit Index per point; hollow markers outside the fidelity cutoff. Returns whether any is hollow."""
    ax.axhline(0, color=ZERO, lw=0.6, zorder=0)
    hollow_any = False
    for ed in T.EDITORS:
        sub = d[d[f"{ed} EI"].notna()]
        c = st.EDITOR_COLORS[ed]
        ax.plot(sub.index, sub[f"{ed} EI"], color=c, marker=MARK[ed], ms=3.2, lw=1.0, zorder=3)
        out = sub[~sub[f"{ed} within_guard"].astype(bool)]
        if len(out):
            hollow_any = True
            ax.plot(out.index, out[f"{ed} EI"], ls="none", marker=MARK[ed], ms=3.2, mfc="white", mec=c, zorder=4)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    style_ax(ax)
    return hollow_any


def draw_skill(ax, d) -> None:
    ax.plot(d.index, d["g_r2"], color=st.EDITOR_COLORS["IM"], marker="d", ms=2.6, lw=0.9, zorder=3, clip_on=False)
    ax.plot(d.index, d["skill_MLP"], color=GRAY, marker="o", ms=2.6, lw=0.9, zorder=3, clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    style_ax(ax)


def legend_below(fig, handles) -> None:
    """A legend on the bottom edge with ``LEGEND_GAP`` inches above it; the axes take the region above."""
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0.0,
                     ncol=len(handles), handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    strip = leg.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi + LEGEND_GAP
    frac = strip / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fig, axes = plt.subplots(2, len(RUNS), squeeze=False, sharex=True, sharey="row", layout="constrained",
                             figsize=(st.TEXT_WIDTH_IN, 3.2), gridspec_kw=dict(height_ratios=[1.75, 1]))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    hollow = False
    for j, (title, run) in enumerate(RUNS.items()):
        d = T.by_point(run)
        hollow |= draw_edit_index(axes[0, j], d)
        axes[0, j].set_title(title, pad=3)
        draw_skill(axes[1, j], d)
        axes[1, j].set_xlabel("Residual point")
    axes[0, 0].set_ylabel("Edit Index")
    axes[1, 0].set_ylabel("Skill")
    handles = [Line2D([], [], color=st.EDITOR_COLORS[e], marker=MARK[e], ms=3.2, lw=1.0, label=e) for e in T.EDITORS]
    if hollow:
        handles.append(Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=st.TEXT,
                              label="Edit Fidelity < 0"))
    handles += [Line2D([], [], color=st.EDITOR_COLORS["IM"], marker="d", ms=2.6, lw=0.9, label="inverse map R²"),
                Line2D([], [], color=GRAY, marker="o", ms=2.6, lw=0.9, label="MLP probe skill")]
    legend_below(fig, handles)
    st.save(fig, "appendix/editability_over_res_point")


if __name__ == "__main__":
    main()
