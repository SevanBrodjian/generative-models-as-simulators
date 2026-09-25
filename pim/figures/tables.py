"""The paper's tables, read from the shipped artifacts and drawn as heat-colored figure tables.

One ``table_*`` function per paper table; the other public functions return the numbers the text quotes.
Inputs: ``runs/<env>/<variant>/scores.json`` (seed replicates in ``runs/<env>/<variant>__seed*/``), the
floors in ``runs/_baselines/<env>/<instance>/`` and the per-run analysis files. Arm selection comes from
``pim.metrics.selection``, seed spread from ``pim.metrics.replicates``, loss floors from ``pim.metrics.prediction``.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pim.environments import layout
from pim.metrics.edit_index import FIDELITY_GUARD, fidelity
from pim.metrics.prediction import excess_estimate, floor_estimate, gap_closed
from pim.metrics.replicates import pool_replicates, t975
from pim.metrics.selection import arms_of, best_arm, best_arm_by_fidelity, best_point

RUNS = layout.REPO / "runs"
EDITORS = ("PI", "GS", "IM")
EDITORS_ALL = ("PI", "GS", "IM", "IM-NN")
OTH_BLOCK = "mine/theirs"            # Othello's probe target, the board relative to the player to move
OTH_EI = "edit_index_symdiff"        # Othello's Edit Index: the legal-set construction on the symmetric difference
RW_BLOCK = "cartesian"               # the continuous Rayworld state every table reports
CAT_BLOCK = "appearance-fac"         # the categorical Rayworld target of the main tables
# the scores.json blocks a table may read; "frustum" is scored but not reported
REPORTED_BLOCKS = (RW_BLOCK, CAT_BLOCK, "appearance", "grid-6x5", "grid-10x3", "grid-16x8", "pos@appearance")
TARGET_LABEL = {RW_BLOCK: "Position, regression", "appearance": "Appearance", CAT_BLOCK: "Appearance, factorized",
                "grid-6x5": "Grid 6x5", "grid-10x3": "Grid 10x3", "grid-16x8": "Grid 16x8",
                "pos@appearance": "Snapped regression"}
VARIANT_NOTE = {"rayworld/standard": "128 rays", "rayworld/blink": "128 rays", "rayworld/128-ray": "big discs"}
N_RAY_FAMILY = ("rayworld/128-ray", "rayworld/16-ray", "rayworld/8-ray", "rayworld/5-ray")
# the runs whose appearance-fac block the tables report; standard and blink carry one for the figures only
CAT_RUNS = N_RAY_FAMILY + ("rayworld/8-ray-tokens",)
DAGGER_SD = 0.1                      # a cell whose seed SD exceeds this carries a dagger
LANDS = 0.25                         # an editor "lands" when its seed-mean Edit Index is at least this
SEED_COLS = ["skill_LIN", "skill_MLP", "unedited"] + [f"{e} {m}" for m in ("EI", "fid") for e in EDITORS_ALL]


# ── reading the artifacts ────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def read_json(run_id: str, name: str = "scores.json") -> dict:
    return _json(str(RUNS / run_id / name))


def replicates(run_id: str) -> list[str]:
    """The seed replicates of a run, ``<env>/<variant>__seed<k>``, in seed order."""
    env, variant = run_id.split("/", 1)
    return [f"{env}/{p.parent.name}" for p in sorted((RUNS / env).glob(f"{variant}__seed*/scores.json"))]


def baselines(env: str, instance: str, name: str = "baselines.json") -> dict | None:
    p = layout.baselines_dir(env, instance) / name
    return _json(str(p)) if p.exists() else None


def blocks(s: dict, run: str | None = None) -> dict[str, dict]:
    """The reported blocks of one ``scores.json``: Othello's board target, Rayworld's ``REPORTED_BLOCKS``
    (with ``run``, a (main) run id, ``CAT_BLOCK`` only where ``run`` is in ``CAT_RUNS``)."""
    if s["env"] == "othello":
        return {OTH_BLOCK: {"probe_skill_linear": s["probe_skill"]["mine|linear|sequence"],
                            "probe_skill_mlp": s["probe_skill"]["mine|mlp|sequence"],
                            "unedited": s["unedited"], "arms": s["arms"], "inverse_map": s.get("inverse_map") or {}}}
    return {k: s["bases"][k] for k in REPORTED_BLOCKS
            if k in s["bases"] and (run is None or k != CAT_BLOCK or run in CAT_RUNS)}


def ei_key(env: str) -> str:
    return OTH_EI if env == "othello" else "edit_index"


def pick(arms: list[dict], editor: str, key: str, select: str = "index") -> dict | None:
    """The arm a table reports: the paper's rule (``select="index"``) or the highest fidelity."""
    if select == "fidelity":
        return best_arm_by_fidelity(arms, editor, key)
    return best_arm(arms, editor, key)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def block_row(blk: dict, key: str, select: str = "index") -> dict:
    """The numbers one (run, block) contributes: best-point Probe Skill, the unedited index, each editor's
    reported arm with its Edit Index and Edit Fidelity, and the two latent R² at the IM arm's point."""
    arms = blk.get("arms", [])
    row = {"skill_LIN": best_point(blk["probe_skill_linear"])[0], "skill_MLP": best_point(blk["probe_skill_mlp"])[0],
           "unedited": blk["unedited"].get(key, np.nan)}
    for ed in EDITORS_ALL:
        a = pick(arms, ed, key, select)
        row[f"{ed} EI"] = a[key] if a else np.nan
        row[f"{ed} fid"] = fidelity(a["fidelity_ratio"]) if a and not _nan(a.get("fidelity_ratio")) else np.nan
        row[f"{ed} arm"] = a
    pt = int(row["IM arm"]["point"]) if row["IM arm"] else None
    inv = blk.get("inverse_map") or {}
    for name in ("g_r2", "nn_r2"):
        vals = inv.get(name)
        row[f"{name}@IM"] = vals[pt] if vals and pt is not None and pt < len(vals) else np.nan
        row[f"{name} max"] = best_point(vals)[0] if vals else np.nan
    return row


@dataclass
class Frames:
    df: pd.DataFrame                   # one row per (run, block), in run order
    rep_sd: dict                       # (run, block) -> pim.metrics.replicates.pool_replicates entry
    reps: pd.DataFrame                 # one row per (replicate, block), with the replicate's arms
    select: str = "index"
    runs: list = field(default_factory=list)

    def row(self, run: str, block: str) -> dict:
        r = self.df[(self.df["run"] == run) & (self.df["block"] == block)]
        if r.empty:
            raise KeyError(f"no ({run}, {block}) row")
        return r.iloc[0].to_dict()

    def sd(self, run: str, block: str, col: str) -> float:
        return self.rep_sd.get((run, block), {}).get(col, np.nan)


def collect(runs: list[str], select: str = "index", *, pool_budgets: bool = False,
            budget_tolerance: float = 0.10) -> Frames:
    """Every listed run's reported blocks and, per block, the spread over its seed replicates (pooled at a
    matched training budget unless ``pool_budgets``). ``select``: ``"index"`` (the paper's rule) or ``"fidelity"``."""
    if select not in ("index", "fidelity"):
        raise ValueError(f"select must be 'index' or 'fidelity', got {select!r}")
    rows, rep_rows = [], []
    for run in runs:
        s = read_json(run)
        env, key = s["env"], ei_key(s["env"])
        for name, blk in blocks(s, run).items():
            rows.append({"run": run, "env": env, "instance": s["instance"], "arch": s["arch"], "block": name,
                         **block_row(blk, key, select)})
        for rep in replicates(run):
            cfg = read_json(rep, "config.json").get("replicate", {})
            for name, blk in blocks(read_json(rep), run).items():
                rep_rows.append({"run": rep, "parent": run, "basis": name, "steps": cfg.get("steps", np.nan),
                                 "seed": cfg.get("seed", -1), **block_row(blk, key, select)})
    rep_sd = pool_replicates(rep_rows, SEED_COLS, pool_budgets=pool_budgets, budget_tolerance=budget_tolerance)
    return Frames(pd.DataFrame(rows), rep_sd, pd.DataFrame(rep_rows), select, list(runs))


def floor_cells(env: str, instance: str, arch: str, block: str) -> dict:
    """The observation floor (right-aligned, large corpus) and the random-init floor of one (instance, arch, block)."""
    b = ((baselines(env, instance) or {}).get("archs", {}).get(arch, {}).get("bases", {}).get(block, {}))
    obs = b.get("observation_right_large") or b.get("observation_right") or {}
    rnd = b.get("random_init") or {}
    return {"obs_LIN": obs.get("linear", {}).get("skill", np.nan), "obs_MLP": obs.get("mlp", {}).get("skill", np.nan),
            "rand_LIN": rnd.get("linear", {}).get("skill", np.nan), "rand_MLP": rnd.get("mlp", {}).get("skill", np.nan)}


@lru_cache(maxsize=None)
def bins(instance: str, target: str) -> int | None:
    """How many cells a categorical target has on a Rayworld instance (None for a regression target)."""
    import h5py

    from pim.environments.rayworld.grid_target import target_cells

    if target == RW_BLOCK:
        return None
    with h5py.File(layout.edits_file("rayworld", instance)) as f:
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    return target_cells(target, sim)


def variant(run: str) -> str:
    return run.split("/", 1)[1]


def row_label(r: dict) -> str:
    """A row's variant name with the paper's note: ray and disc description, or the bin count of a categorical row."""
    if r["block"] in (OTH_BLOCK, RW_BLOCK):
        note = VARIANT_NOTE.get(r["run"])
    else:
        note = f"{bins(r['instance'], r['block'])} bins"
    return variant(r["run"]) + (f" ({note})" if note else "")


# ── drawing ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Col:
    key: str                           # the column of the values frame
    group: str                         # upper header, spanning consecutive columns with the same group
    label: str                         # lower header
    kind: str                          # color scale, a key of SCALES
    fmt: str = ".3f"


SCALES = {
    "skill": ("Greens", Normalize(0.0, 1.0)),
    "index": ("RdYlGn", Normalize(-1.0, 1.0)),
    "fid": ("RdYlGn", TwoSlopeNorm(vmin=-2.0, vcenter=FIDELITY_GUARD, vmax=1.0)),
    "sd": ("Oranges", Normalize(0.0, 0.25)),
    "excess": ("YlOrRd", Normalize(0.0, 0.25)),
    "plain": (None, None),
}
WIDTH = {"plain": 1.05, "excess": 1.05}  # column width in inches by kind (default CELL_W)
CELL_W, ROW_H, CHAR_W = 0.66, 0.25, 0.066
INK, RULE, PLAIN_BG, BLANK_BG = "#172239", "#172239", "#f4f3ee", "#ffffff"


@dataclass
class Panel:
    title: str
    cols: list
    text: pd.DataFrame                  # index (section, row label); one string per drawn cell
    shade: pd.DataFrame                 # the value each cell is colored by (NaN: no color)


@dataclass
class Table:
    """One paper table: its numbers (``values``), the strings as printed (``text``) and the figure."""
    name: str
    values: pd.DataFrame
    text: pd.DataFrame
    fig: Figure

    def _repr_png_(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        return buf.getvalue()

    def __repr__(self) -> str:
        return f"<{self.name}: {len(self.values)} rows>"

    def save(self, path) -> None:
        self.fig.savefig(path, bbox_inches="tight")


def _fmt(v, spec: str) -> str:
    """A number as the paper prints it; a signed or unsigned zero prints without a minus sign."""
    if _nan(v):
        return "—"
    if spec == "d":
        return str(int(v))
    s = format(v, spec)
    if re.fullmatch(r"-0\.0*", s):
        s = ("+" if spec.startswith("+") else "") + s[1:]
    return s


def texts(values: pd.DataFrame, cols: list[Col], daggers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every drawn cell as a string, with a dagger where ``daggers`` is True."""
    out = pd.DataFrame(index=values.index, columns=[c.key for c in cols], dtype=object)
    for c in cols:
        for i, v in zip(values.index, values[c.key]):
            dg = daggers is not None and c.key in daggers and bool(daggers.at[i, c.key])
            out.at[i, c.key] = _fmt(v, c.fmt) + ("†" if dg else "")
    return out


def _cell_color(kind: str, v):
    cmap, norm = SCALES[kind]
    if cmap is None:
        return PLAIN_BG
    if _nan(v):
        return BLANK_BG
    return colormaps[cmap](float(np.clip(norm(v), 0.0, 1.0)))


def _ink_for(bg) -> str:
    if isinstance(bg, str):
        return INK
    r, g, b = bg[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.45 else INK


def _draw_panel(fig: Figure, rect, p: Panel, sec_w: float, lab_w: float, height: float, width: float) -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    y = 0.0
    if p.title:
        ax.text(0.0, y + 0.18, p.title, fontsize=9.5, color=INK, ha="left", va="center", weight="bold")
        y += 0.34
    x0 = sec_w + lab_w
    xs, x = [], x0
    for c in p.cols:
        xs.append(x)
        x += WIDTH.get(c.kind, CELL_W)
    x_end = x
    ax.plot([0, x_end], [y, y], color=RULE, lw=1.2)
    groups, j = [], 0
    while j < len(p.cols):
        k = j
        while k + 1 < len(p.cols) and p.cols[k + 1].group == p.cols[j].group:
            k += 1
        groups.append((j, k))
        j = k + 1
    for j, k in groups:
        xa, xb = xs[j], xs[k] + WIDTH.get(p.cols[k].kind, CELL_W)
        if p.cols[j].group:
            ax.text((xa + xb) / 2, y + ROW_H / 2, p.cols[j].group, ha="center", va="center", fontsize=8.5, color=INK)
            ax.plot([xa + 0.04, xb - 0.04], [y + ROW_H - 0.02] * 2, color=RULE, lw=0.6)
    y += ROW_H
    for c, xc in zip(p.cols, xs):
        ax.text(xc + WIDTH.get(c.kind, CELL_W) / 2, y + ROW_H / 2, c.label, ha="center", va="center", fontsize=8,
                color=INK)
    y += ROW_H
    ax.plot([0, x_end], [y, y], color=RULE, lw=0.9)
    sections = list(dict.fromkeys(p.text.index.get_level_values(0)))
    for si, sec in enumerate(sections):
        sub = p.text.loc[[sec]]
        if si:
            ax.plot([0, x_end], [y, y], color=RULE, lw=0.6)
        if sec:
            ax.text(0.04, y + ROW_H * len(sub) / 2, sec, ha="left", va="center", fontsize=8, color=INK)
        for (_, lab), trow in sub.iterrows():
            ax.text(sec_w, y + ROW_H / 2, lab, ha="left", va="center", fontsize=8, color=INK)
            for c, xc in zip(p.cols, xs):
                w = WIDTH.get(c.kind, CELL_W)
                bg = _cell_color(c.kind, p.shade.at[(sec, lab), c.key])
                ax.add_patch(Rectangle((xc, y), w, ROW_H, facecolor=bg, edgecolor="white", lw=0.8))
                ax.text(xc + w / 2, y + ROW_H / 2, trow[c.key], ha="center", va="center", fontsize=7.6,
                        color=_ink_for(bg))
            y += ROW_H
    ax.plot([0, x_end], [y, y], color=RULE, lw=1.2)


def draw(panels: list[Panel]) -> Figure:
    """Figure tables in the paper's layout: section and variant labels on the left, a two-level header,
    each cell colored on its column's scale."""
    sec_w = max((len(s) for p in panels for s in p.text.index.get_level_values(0)), default=0) * CHAR_W
    sec_w = sec_w + 0.15 if sec_w else 0.0
    lab_w = max(len(s) for p in panels for s in p.text.index.get_level_values(1)) * CHAR_W + 0.2
    width = sec_w + lab_w + max(sum(WIDTH.get(c.kind, CELL_W) for c in p.cols) for p in panels) + 0.05
    heights = [(0.34 if p.title else 0.0) + ROW_H * (2 + len(p.text)) + 0.08 for p in panels]
    gap = 0.25
    total = sum(heights) + gap * (len(panels) - 1)
    fig = Figure(figsize=(width, total))
    top = total
    for p, h in zip(panels, heights):
        _draw_panel(fig, [0, (top - h) / total, 1, h / total], p, sec_w, lab_w, h, width)
        top -= h + gap
    return fig


def _table(name: str, title: str, values: pd.DataFrame, cols: list[Col], daggers=None, text=None,
           shade=None) -> Table:
    text = texts(values, cols, daggers) if text is None else text
    shade = values[[c.key for c in cols]] if shade is None else shade
    return Table(name, values, text, draw([Panel(title, cols, text, shade)]))


def _frame(rows: list[tuple[str, str, dict]]) -> pd.DataFrame:
    """(section, row label, values) triples as a frame indexed by (section, row)."""
    idx = pd.MultiIndex.from_tuples([(s, lab) for s, lab, _ in rows], names=["section", "row"])
    return pd.DataFrame([v for _, _, v in rows], index=idx)


# ── column sets ──────────────────────────────────────────────────────────────


DEC_COLS = [Col("obs_LIN", "Observation", "Lin.", "skill"), Col("obs_MLP", "Observation", "MLP", "skill"),
            Col("rand_LIN", "Random init", "Lin.", "skill"), Col("rand_MLP", "Random init", "MLP", "skill"),
            Col("skill_LIN", "Trained", "Lin.", "skill"), Col("skill_MLP", "Trained", "MLP", "skill"),
            Col("g_r2 max", "Inverse Map MLP", "Best Point", "skill"),
            Col("g_r2@IM", "Inverse Map MLP", "Edit Point", "skill")]
EDIT_COLS = [Col("unedited", "Unedited", "Index", "index", "+.2f")] + [
    Col(f"{e} {m}", e, lab, kind, fmt) for e in EDITORS
    for m, lab, kind, fmt in (("EI", "Index", "index", "+.2f"), ("fid", "Fid.", "fid", ".2f"))]
SPREAD_COLS = [Col("skill_LIN", "Probe Skill", "Lin.", "sd"), Col("skill_MLP", "Probe Skill", "MLP", "sd")] + [
    Col(f"{e} {m}", e, lab, "sd") for e in EDITORS for m, lab in (("EI", "Index"), ("fid", "Fid."))]
SKILL_EDIT_COLS = [Col("skill_LIN", "Probe Skill", "Lin.", "skill"), Col("skill_MLP", "Probe Skill", "MLP", "skill")] \
    + EDIT_COLS[1:]
MAIN_SECTIONS = (("Othello", "othello", OTH_BLOCK), ("Rayworld (continuous)", "rayworld", RW_BLOCK),
                 ("Rayworld (categorical)", "rayworld", CAT_BLOCK))


def main_rows(F: Frames, sections=MAIN_SECTIONS) -> list[tuple[str, str, dict]]:
    """(section, label, row) for the Table 2 layout: Othello, Rayworld continuous, Rayworld categorical."""
    out = []
    for sec, env, block in sections:
        for r in F.df[(F.df["env"] == env) & (F.df["block"] == block)].to_dict("records"):
            out.append((sec, row_label(r), r))
    return out


def _decodability_values(F: Frames, rows) -> pd.DataFrame:
    keys = ["skill_LIN", "skill_MLP", "g_r2 max", "g_r2@IM"]
    return _frame([(s, lab, {"run": r["run"], "block": r["block"],
                             **floor_cells(r["env"], r["instance"], r["arch"], r["block"]),
                             **{k: r[k] for k in keys}}) for s, lab, r in rows])


def _edit_values(rows) -> pd.DataFrame:
    return _frame([(s, lab, {"run": r["run"], "block": r["block"], **{c.key: r[c.key] for c in EDIT_COLS}})
                   for s, lab, r in rows])


def _daggers(F: Frames, values: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({c.key: [F.sd(run, blk, c.key) > DAGGER_SD for run, blk in zip(values["run"], values["block"])]
                         for c in cols}, index=values.index)


# ── main text ────────────────────────────────────────────────────────────────


def table_decodability(F: Frames) -> Table:
    """tab:decodability: Probe Skill of the observation, random-init and trained probes (each at its best
    residual point) and the inverse map's R² at its best point and at the IM arm's point."""
    rows = main_rows(F, (("Othello", "othello", OTH_BLOCK), ("Rayworld", "rayworld", RW_BLOCK)))
    return _table("tab:decodability", "Decodability", _decodability_values(F, rows), DEC_COLS)


def table_editability(F: Frames) -> Table:
    """tab:editability: the unedited index and each editor's Edit Index and Edit Fidelity at the paper's
    setting; a dagger where the seed SD exceeds ``DAGGER_SD``."""
    values = _edit_values(main_rows(F))
    return _table("tab:editability", "Editability", values, EDIT_COLS, daggers=_daggers(F, values, EDIT_COLS[1:]))


def im_vs_nn_gain(F: Frames) -> pd.DataFrame:
    """IM's Edit Index relative to the nearest-neighbor control on each Rayworld variant, and the mean gain."""
    R = F.df[(F.df["env"] == "rayworld") & (F.df["block"] == RW_BLOCK)]
    out = pd.DataFrame({"IM": R["IM EI"].values, "IM-NN": R["IM-NN EI"].values}, index=R["run"].values)
    out["gain"] = out["IM"] / out["IM-NN"] - 1
    out.loc["mean"] = [np.nan, np.nan, out["gain"].mean()]
    return out


def seed_sd_maxima(F: Frames, lands: float = LANDS) -> pd.Series:
    """The largest seed SD of any trained probe's Probe Skill, of the Edit Index of an editor that lands
    (seed-mean Edit Index >= ``lands``), and the number of editor cells whose SD exceeds ``DAGGER_SD``."""
    skill, edit, over = [], [], []
    for _, _, r in main_rows(F):
        v = F.rep_sd.get((r["run"], r["block"]), {})
        skill += [(v[c], f"{r['run']} {r['block']} {c}") for c in ("skill_LIN", "skill_MLP") if c in v]
        for e in EDITORS:
            if f"{e} EI" in v and v[f"{e} EI_mean"] >= lands:
                edit.append((v[f"{e} EI"], f"{r['run']} {r['block']} {e}"))
            over += [f"{r['run']} {r['block']} {e} {m}" for m in ("EI", "fid") if v.get(f"{e} {m}", 0) > DAGGER_SD]
    s, e = max(skill), max(edit)
    return pd.Series({"Probe Skill SD, max": s[0], "Probe Skill SD, max at": s[1],
                      "landing Edit Index SD, max": e[0], "landing Edit Index SD, max at": e[1],
                      f"cells with SD > {DAGGER_SD}": len(over),
                      f"cells with SD > {DAGGER_SD}, which": "; ".join(over)})


def flip_rates(runs: list[str]) -> pd.DataFrame:
    """Per Othello variant: tokens flipped per move and per game in its held-out test games
    (``runs/_baselines/othello/<instance>/corpus_stats.json``)."""
    out = {}
    for run in runs:
        d = _json(str(layout.baselines_dir("othello", read_json(run)["instance"]) / "corpus_stats.json"))
        out[variant(run)] = {k: d[k] for k in ("flips_per_move", "flips_per_game", "n_games", "n_moves")}
    return pd.DataFrame.from_dict(out, orient="index")


def pi_landing(F: Frames, alpha: float = 1.0) -> Table:
    """The linear probe's readout error after PI's write at ``alpha`` (1 = the exact step), per Rayworld
    run (continuous state) and residual point."""
    rows = []
    for run in (r for r in F.runs if r.startswith("rayworld/")):
        arms = [a for a in arms_of(blocks(read_json(run))[RW_BLOCK]["arms"], "PI") if a["alpha"] == alpha]
        err = {int(a["point"]): a["readout_err_after"] for a in arms}
        rows.append(("", variant(run), {"run": run, **{f"point {p}": err[p] for p in sorted(err)}}))
    V = _frame(rows)
    cols = [Col(c, "Residual point", c.split()[1], "plain", ".1e") for c in V.columns if c != "run"]
    return _table("pi_landing", f"Linear-probe readout error after PI's write at α = {alpha:g}", V, cols)


# ── appendix: predictive loss ────────────────────────────────────────────────


READINGS = {"othello": ("moves",), "rayworld": ("frames", "tokens", "expected-frame")}
READING_LABEL = {"tokens": "next-token CE (nats per frame)", "expected-frame": "mean frame (MSE ×10⁻³)"}


def prediction_values(runs: list[str]) -> pd.DataFrame:
    """Per (run, reading): the trivial predictor's loss, the Bayes floor (value ± its uncertainty), the model's
    held-out loss on the floor's sequences, the excess over the floor and the share of the gap closed."""
    rows = []
    for run in runs:
        s = read_json(run)
        floor = baselines(s["env"], s["instance"], "bayes_floor.json")
        readings = s.get("prediction", {}).get("readings", {})
        for reading in (r for r in READINGS[s["env"]] if r in readings):
            rd = readings[reading]
            obj = rd["objective"]
            est = floor_estimate(floor, obj)
            ex = excess_estimate(rd["loss_paired"], est)
            tv = ((floor or {}).get("trivial") or {}).get(obj) or {}
            triv = tv.get("value", np.nan)
            if reading in READING_LABEL:
                sec, lab = f"Rayworld ({variant(run)})", READING_LABEL[reading]
            else:
                sec = "Othello (nats per move)" if s["env"] == "othello" else "Rayworld (MSE ×10⁻³)"
                lab = row_label({"run": run, "block": RW_BLOCK})
            rows.append((sec, lab, {"run": run, "reading": reading, "objective": obj, "trivial": triv,
                                    "floor": est[0] if est else np.nan, "floor_pm": est[1] if est else np.nan,
                                    "loss": rd["loss_paired"], "excess": ex[0] if ex else np.nan,
                                    "excess_pm": ex[1] if ex else np.nan, "excess_rel": ex[2] if ex else np.nan,
                                    "gap_closed": gap_closed(rd["loss_paired"], triv, est[0]) if est else np.nan,
                                    "n_paired": rd.get("n_paired")}))
    return _frame(rows)


def table_predictive_skill(runs: list[str]) -> Table:
    """tab:predictive_skill: held-out loss between the trivial predictor and the Bayes floor. Rayworld frame
    losses are shown ×10³; ± is the floor's uncertainty (none where the floor is exact)."""
    V = prediction_values(runs)
    cols = [Col("trivial", "", "Trivial", "plain"), Col("floor", "", "Bayes floor", "plain"),
            Col("loss", "", "Model", "plain"), Col("excess", "", "Excess", "excess")]
    text = pd.DataFrame(index=V.index, columns=[c.key for c in cols], dtype=object)
    for i, r in V.iterrows():
        k, d = (1e3, 2) if r["objective"] == "mse" else (1.0, 3)

        def pm(v, u):
            return _fmt(v * k, f".{d}f") + ("" if _nan(u) or u == 0 else f" ± {_fmt(u * k, f'.{d}f')}")
        text.loc[i] = [_fmt(r["trivial"] * k, f".{d - 1 if r['objective'] == 'mse' else d}f"),
                       pm(r["floor"], r["floor_pm"]), _fmt(r["loss"] * k, f".{d}f"), pm(r["excess"], r["excess_pm"])]
    shade = V[["trivial", "floor", "loss"]].assign(excess=V["excess_rel"])
    return _table("tab:predictive_skill", "Predictive loss", V, cols, text=text, shade=shade)


def gap_closed_min(runs: list[str]) -> pd.Series:
    """The share of the gap from the trivial predictor to the Bayes floor each model closes, and the minimum."""
    V = prediction_values(runs)
    s = pd.Series(V["gap_closed"].values, index=[f"{r} {d}" for r, d in zip(V["run"], V["reading"])])
    s.loc["min"] = s.min()
    return s


# ── appendix: seed spread ────────────────────────────────────────────────────


def table_seed_spread(F: Frames) -> Table:
    """tab:seed_spread: SD over the seed replicates (pooled at a matched budget) of each trained probe's
    Probe Skill and each editor's Edit Index and Edit Fidelity; a dagger above ``DAGGER_SD``."""
    rows = main_rows(F)
    V = _frame([(s, lab, {"run": r["run"], "block": r["block"],
                          "n": F.rep_sd.get((r["run"], r["block"]), {}).get("n", 0),
                          "steps": "/".join(str(x) for x in F.rep_sd.get((r["run"], r["block"]), {}).get("steps", [])),
                          **{c.key: F.sd(r["run"], r["block"], c.key) for c in SPREAD_COLS}}) for s, lab, r in rows])
    daggers = V[[c.key for c in SPREAD_COLS]] > DAGGER_SD
    budget = ", ".join(sorted({f"n = {n} at {st}" for n, st in zip(V["n"], V["steps"])}))
    return _table("tab:seed_spread", f"Seed standard deviation ({budget} steps)", V, SPREAD_COLS, daggers=daggers)


def ci_multiplier(n: int = 3) -> float:
    """The half-width of a t-based 95% interval on the mean of ``n`` seeds, in units of their SD."""
    return t975(n - 1) / np.sqrt(n)


def im_steps(F: Frames, runs=N_RAY_FAMILY) -> pd.DataFrame:
    """Each step of IM's Edit Index as the rays coarsen, over the seed means, in units of the two members'
    combined SD (sqrt(s1² + s2²)) and of the larger member SD."""
    out = []
    for block in (RW_BLOCK, CAT_BLOCK):
        for a, b in zip(runs[:-1], runs[1:]):
            va, vb = F.rep_sd[(a, block)], F.rep_sd[(b, block)]
            step = vb["IM EI_mean"] - va["IM EI_mean"]
            out.append({"block": block, "from": variant(a), "to": variant(b), "step": step,
                        "SD from": va["IM EI"], "SD to": vb["IM EI"],
                        "step / combined SD": step / np.hypot(va["IM EI"], vb["IM EI"]),
                        "step / larger SD": step / max(va["IM EI"], vb["IM EI"])})
    return pd.DataFrame(out)


def replicate_arms(F: Frames, run: str, editor: str, block: str | None = None) -> pd.DataFrame:
    """The arm an editor is reported at on the main run and on each seed replicate: point, step size,
    Edit Index, Edit Fidelity and whether any arm passed the fidelity cutoff."""
    block = block or (OTH_BLOCK if run.startswith("othello") else RW_BLOCK)
    rows = [("main", F.row(run, block))]
    if not F.reps.empty:
        R = F.reps[(F.reps["parent"] == run) & (F.reps["basis"] == block)].sort_values("seed")
        rows += [(f"seed {int(r['seed'])}", r) for r in R.to_dict("records")]
    out = []
    for name, r in rows:
        a = r[f"{editor} arm"] or {}
        out.append({"member": name, "point": a.get("point"), "alpha": a.get("alpha"), "EI": r[f"{editor} EI"],
                    "fid": r[f"{editor} fid"], "within_guard": a.get("within_guard")})
    return pd.DataFrame(out).set_index("member")


def dagger_cells(F: Frames) -> pd.DataFrame:
    """Each cell whose seed SD exceeds ``DAGGER_SD``, with the setting each seed reports it at."""
    out = []
    for _, _, r in main_rows(F):
        v = F.rep_sd.get((r["run"], r["block"]), {})
        for e in EDITORS:
            for m in ("EI", "fid"):
                if v.get(f"{e} {m}", 0) > DAGGER_SD:
                    A = replicate_arms(F, r["run"], e, r["block"]).drop(index="main")
                    out.append({"run": r["run"], "block": r["block"], "cell": f"{e} {m}", "SD": v[f"{e} {m}"],
                                "mean": v[f"{e} {m}_mean"],
                                "seed settings": "; ".join(f"pt{p} α{a:g}" for p, a in zip(A["point"], A["alpha"])),
                                "seed EI": " / ".join(f"{x:+.2f}" for x in A["EI"]),
                                "seed fid": " / ".join(f"{x:.2f}" for x in A["fid"]),
                                "seeds outside the cutoff": int((~A["within_guard"].astype(bool)).sum())})
    return pd.DataFrame(out)


def seed_means_vs_main(F: Frames) -> pd.DataFrame:
    """Per editor cell outside the daggered ones: the seed mean at 512k against the main run's value, and
    the largest difference for the Edit Index and for Edit Fidelity (last two rows)."""
    out = []
    for _, _, r in main_rows(F):
        v = F.rep_sd.get((r["run"], r["block"]), {})
        for e in EDITORS:
            for m in ("EI", "fid"):
                c = f"{e} {m}"
                if c in v and v[c] <= DAGGER_SD:
                    out.append({"cell": f"{r['run']} {r['block']} {c}", "metric": m, "main": r[c],
                                "seed mean": v[f"{c}_mean"], "diff": v[f"{c}_mean"] - r[c]})
    D = pd.DataFrame(out).set_index("cell")
    for m in ("EI", "fid"):
        sub = D[D["metric"] == m]
        i = sub["diff"].abs().idxmax()
        D.loc[f"max |diff|, {m}: {i}"] = [m, sub.at[i, "main"], sub.at[i, "seed mean"], abs(sub.at[i, "diff"])]
    return D


def fixed_setting_check(F: Frames, cells=(("othello/adjacent-flip", "PI"), ("othello/adjacent-noflip", "PI"),
                                          ("othello/adjacent-noflip", "IM"))) -> pd.DataFrame:
    """Each replicate read at the main run's reported point and step size instead of its own selection:
    the per-seed Edit Index and Edit Fidelity there and the seed SD of that Edit Index."""
    out = []
    for run, editor in cells:
        block = OTH_BLOCK if run.startswith("othello") else RW_BLOCK
        main = F.row(run, block)[f"{editor} arm"]
        key = ei_key(run.split("/")[0])
        rep_rows = []
        for rep in replicates(run):
            cfg = read_json(rep, "config.json").get("replicate", {})
            same = ("point", "alpha", "dims")
            arms = [a for a in arms_of(blocks(read_json(rep))[block]["arms"], editor)
                    if all(a.get(k) == main.get(k) for k in same)]
            a = arms[0] if arms else {}
            rep_rows.append({"parent": run, "basis": block, "steps": cfg.get("steps"), "seed": cfg.get("seed"),
                             "EI": a.get(key, np.nan), "fid": fidelity(a["fidelity_ratio"]) if a else np.nan})
        pooled = pool_replicates(rep_rows, ["EI", "fid"]).get((run, block), {})
        out.append({"run": run, "editor": editor, "point": main["point"], "alpha": main["alpha"],
                    "seed EI": " / ".join(f"{x:+.2f}" for x in pooled.get("EI_values", [])),
                    "seed fid": " / ".join(f"{x:.2f}" for x in pooled.get("fid_values", [])),
                    "EI SD": pooled.get("EI", np.nan), "n": pooled.get("n", 0)})
    return pd.DataFrame(out)


def probe_refit_spread(linear=(("othello/standard", "mine"), ("othello/adjacent-flip__seed0", "mine"),
                               ("othello/adjacent-flip__seed1", "mine"), ("othello/adjacent-flip__seed2", "mine"),
                               ("rayworld/8-ray", CAT_BLOCK)),
                       inverse=("othello/standard", "othello/adjacent-flip__seed0", "othello/adjacent-flip__seed1",
                                "othello/adjacent-flip__seed2")) -> pd.DataFrame:
    """SD over probe refits on a fixed model (``variance.json``): the linear probe's best Probe Skill and PI's
    Edit Index at the main run's edit point, and IM's Edit Index over inverse-map refits; the maxima last."""
    out = []
    for run, target in linear:
        s = read_json(run, "variance.json")["probe_seeds"][target]["summary"]
        out.append({"refit": "linear probe", "run": run, "target": target, "n_seeds": s["n_seeds"],
                    "Probe Skill SD": s["best_skill"]["sd"],
                    "Edit Index SD": s["editors"]["PI@canonical_edit_best"]["edit_index"]["sd"]})
    for run in inverse:
        s = read_json(run, "variance.json")["probe_seeds"]["inverse_map"]["summary"]
        out.append({"refit": "inverse map", "run": run, "target": "", "n_seeds": s["n_seeds"],
                    "Probe Skill SD": np.nan, "Edit Index SD": s["editors"]["IM"]["edit_index"]["sd"]})
    D = pd.DataFrame(out)
    for kind in ("linear probe", "inverse map"):
        sub = D[D["refit"] == kind]
        D.loc[len(D)] = {"refit": f"{kind}, max", "run": "", "target": "", "n_seeds": sub["n_seeds"].max(),
                         "Probe Skill SD": sub["Probe Skill SD"].max(), "Edit Index SD": sub["Edit Index SD"].max()}
    return D


# ── appendix: reachable vs unreachable targets ───────────────────────────────


LEGAL_COLS = [Col("n", "", "n", "plain", "d")] + EDIT_COLS


def reachability_counts(runs: list[str]) -> pd.DataFrame:
    """Per Othello variant: how many bench cases have a reachable, unreachable or undecided target board."""
    return pd.DataFrame({variant(r): baselines("othello", read_json(r)["instance"], "reachability.json")["counts"]
                         for r in runs}).T


def table_legal_illegal(runs: list[str]) -> Table:
    """tab:legal_illegal: each editor at its main-table setting, on the cases whose target board some legal
    game reaches (legal) and on those no game reaches (illegal); undecided cases are left out."""
    rows = []
    for run in runs:
        e = read_json(run, "editability_by_reachability.json")
        counts = baselines("othello", e["instance"], "reachability.json")["counts"]
        for target, split in (("legal", "reachable"), ("illegal", "unreachable")):
            d = e["split"][split]
            eds = d.get("editors", {})
            v = {"run": run, "n": counts[split], "unedited": d.get("unedited_index", np.nan)}
            for ed in EDITORS:
                v[f"{ed} EI"] = eds.get(ed, {}).get("index", np.nan)
                v[f"{ed} fid"] = eds.get(ed, {}).get("fidelity", np.nan)
            rows.append((variant(run), target, v))
    return _table("tab:legal_illegal", "Reachable (legal) vs unreachable (illegal) targets", _frame(rows), LEGAL_COLS)


def table_two_flip(runs: list[str]) -> Table:
    """tab:two_flip: edits that flip two tokens of opposite color, split by whether a legal game reaches the
    result; each editor at the arm the paper's rule picks within the group."""
    rows = []
    for run in runs:
        d = read_json(run, "two_flip_editability.json")
        for target in ("legal", "illegal"):
            ran = target in d["groups_run"]
            rep = d["reported"].get(target, {}) if ran else {}
            v = {"run": run, "n": d["n_cases"] if ran else 0,
                 "unedited": d["groups"][target]["unedited_index"] if ran else np.nan}
            for ed in EDITORS:
                a = rep.get(ed)
                v[f"{ed} EI"] = a[OTH_EI] if a else np.nan
                v[f"{ed} fid"] = fidelity(a["fidelity_ratio"]) if a else np.nan
                v[f"{ed} SE"] = a["index_se"] if a else np.nan
            rows.append((variant(run), target, v))
    return _table("tab:two_flip", "Two-token edits", _frame(rows), LEGAL_COLS)


def two_flip_numbers(runs: list[str]) -> pd.DataFrame:
    """Per variant: the number of cases and the largest standard error of an Edit Index in the table
    (legal and illegal groups)."""
    out = {}
    for run in runs:
        d = read_json(run, "two_flip_editability.json")
        se = [a["index_se"] for g in ("legal", "illegal") for a in d["reported"].get(g, {}).values()]
        out[variant(run)] = {"n_cases": d["n_cases"], "max Edit Index SE": max(se)}
    return pd.DataFrame.from_dict(out, orient="index")


# ── appendix: the token interface ────────────────────────────────────────────


def _tokens_rows(F: Frames) -> list[tuple[str, str, dict]]:
    out = []
    for sec, block in (("Continuous", RW_BLOCK), ("Categorical", CAT_BLOCK)):
        for r in F.df[F.df["block"] == block].to_dict("records"):
            out.append((sec, f"{r['instance']} {'tokens' if r['arch'].endswith('_tokens') else 'frames'}", r))
    return out


def table_tokens_decodability(F: Frames) -> Table:
    """tab:tokens_decodability: Table 1's columns for the 8-ray frame model and the 8-ray token model."""
    return _table("tab:tokens_decodability", "Decodability, frame and token interface",
                  _decodability_values(F, _tokens_rows(F)), DEC_COLS)


def mean_frame_row(blk: dict) -> dict:
    """A token model's editability read from its mean frame: each arm's ray-zone Edit Index and fidelity
    ratio of the expected frame, arms selected by the paper's rule on that readout."""
    key = "zone_edit_index_expected"
    arms = [{**a, "fidelity_ratio": a["fidelity_ratio_expected"]} for a in blk["arms"] if key in a]
    row = {"unedited": blk["unedited"][key]}
    for ed in EDITORS:
        a = best_arm(arms, ed, key)
        row[f"{ed} EI"] = a[key] if a else np.nan
        row[f"{ed} fid"] = fidelity(a["fidelity_ratio"]) if a else np.nan
    return row


def table_tokens_editability(F: Frames) -> Table:
    """tab:tokens_editability: the frame model, the token model scored on its next-frame distribution
    (frame-set construction, marked *), and the token model's mean frame scored as a frame model."""
    rows = []
    for sec, lab, r in _tokens_rows(F):
        v = {"run": r["run"], "block": r["block"], **{c.key: r[c.key] for c in EDIT_COLS}}
        if not r["arch"].endswith("_tokens"):
            rows.append((sec, lab, v))
            continue
        rows.append((sec, f"{lab}, frame set*", v))
        blk = read_json(r["run"])["bases"][r["block"]]
        rows.append((sec, f"{lab}, mean frame", {"run": r["run"], "block": r["block"], **mean_frame_row(blk)}))
    return _table("tab:tokens_editability", "Editability, frame and token interface", _frame(rows), EDIT_COLS)


def tokens_numbers(F: Frames) -> pd.Series:
    """The largest trained Probe Skill difference between the token and frame models."""
    V = _decodability_values(F, _tokens_rows(F))
    diffs = []
    for sec in dict.fromkeys(V.index.get_level_values(0)):
        sub = V.loc[sec]
        diffs += list((sub.iloc[1][["skill_LIN", "skill_MLP"]] - sub.iloc[0][["skill_LIN", "skill_MLP"]]).abs())
    return pd.Series({"trained Probe Skill |tokens - frames|, max": max(diffs)})


# ── appendix: further tables ─────────────────────────────────────────────────


def table_additional_rw(F: Frames) -> Table:
    """tab:additional_rw: (a) Table 1's columns and (b) Table 2's columns for the listed Rayworld variants
    (continuous state)."""
    rows = [("", variant(r["run"]), r) for r in F.df[F.df["block"] == RW_BLOCK].to_dict("records")]
    dec, edit = _decodability_values(F, rows), _edit_values(rows)
    panels = [Panel("(a) Probe Skill", DEC_COLS, texts(dec, DEC_COLS), dec[[c.key for c in DEC_COLS]]),
              Panel("(b) Edit Index and Edit Fidelity", EDIT_COLS, texts(edit, EDIT_COLS),
                    edit[[c.key for c in EDIT_COLS]])]
    values = pd.concat({"a": dec, "b": edit}, names=["panel"])
    text = pd.concat({"a": panels[0].text, "b": panels[1].text}, names=["panel"])
    return Table("tab:additional_rw", values, text, draw(panels))


def by_point(run: str) -> pd.DataFrame:
    """Per residual point: the MLP probe's skill, the inverse map's R², and each editor's reported arm among
    the arms at that point (the paper's rule; GS at its start point)."""
    s = read_json(run)
    block = OTH_BLOCK if s["env"] == "othello" else RW_BLOCK
    blk, key = blocks(s)[block], ei_key(s["env"])
    out = []
    for p in range(int(s["n_points"])):
        row = {"point": p, "skill_MLP": blk["probe_skill_mlp"][p], "g_r2": blk["inverse_map"]["g_r2"][p]}
        for ed in EDITORS:
            a = best_arm([a for a in arms_of(blk["arms"], ed) if a["point"] == p], ed, key)
            row[f"{ed} EI"] = a[key] if a else np.nan
            row[f"{ed} fid"] = fidelity(a["fidelity_ratio"]) if a else np.nan
            row[f"{ed} alpha"] = a["alpha"] if a else np.nan
            row[f"{ed} within_guard"] = a["within_guard"] if a else False
        out.append(row)
    return pd.DataFrame(out).set_index("point")


def table_im_by_point(runs=("othello/standard", "rayworld/standard")) -> Table:
    """tab:im_by_point: at every residual point, the MLP probe's skill, the inverse map's R² and the Edit
    Index and Edit Fidelity of the inverse-map overwrite written there."""
    parts, cols = [], []
    for run in runs:
        d = by_point(run)[["skill_MLP", "g_r2", "IM EI", "IM fid"]]
        d.columns = [f"{run}|{c}" for c in d.columns]
        parts.append(d)
        grp = f"{run.split('/')[0].capitalize()} {variant(run)}"
        cols += [Col(f"{run}|skill_MLP", grp, "Probe Skill", "skill"),
                 Col(f"{run}|g_r2", grp, "Inverse R²", "skill"),
                 Col(f"{run}|IM EI", grp, "IM Index", "index", "+.2f"),
                 Col(f"{run}|IM fid", grp, "IM Fid.", "fid", ".2f")]
    V = pd.concat(parts, axis=1)
    V.index = pd.MultiIndex.from_tuples([("", f"{p}" + (" (embedding)" if p == 0 else "")) for p in V.index],
                                        names=["section", "row"])
    return _table("tab:im_by_point", "Inverse map by residual point", V, cols)


def im_reconstruction(run: str = "othello/standard") -> pd.DataFrame:
    """Writing g of the unedited board at each residual point, with no edit (``im_reconstruction.json``):
    the output's error against the pre-edit legal set, the model's own error and their ratio."""
    d = read_json(run, "im_reconstruction.json")
    return pd.DataFrame({"g error": d["g_error"], "model error": d["model_error"], "ratio": d["ratio"]},
                        index=pd.Index(d["points"], name="point"))


def table_categorical(F: Frames, run: str = "rayworld/8-ray") -> Table:
    """tab:categorical: every probe target of one Rayworld model (regression, appearance bins, evenly spaced
    grids, snapped regression) with its Probe Skill and each editor's Edit Index and Edit Fidelity."""
    rows = []
    for block in TARGET_LABEL:
        if not ((F.df["run"] == run) & (F.df["block"] == block)).any():
            continue
        r = F.row(run, block)
        n = bins(r["instance"], block)
        rows.append(("", TARGET_LABEL[block] + (f" ({n})" if n else ""),
                     {"run": run, "block": block, **{c.key: r[c.key] for c in SKILL_EDIT_COLS}}))
    return _table("tab:categorical", f"Probe targets on {run}", _frame(rows), SKILL_EDIT_COLS)


NN_COLS = [Col("g_r2@IM", "Latent R²", "IM", "skill"), Col("nn_r2@IM", "Latent R²", "NN", "skill"),
           Col("IM EI", "Inverse Map", "Index", "index", "+.2f"), Col("IM fid", "Inverse Map", "Fid.", "fid", ".2f"),
           Col("IM-NN EI", "Nearest Neighbor", "Index", "index", "+.2f"),
           Col("IM-NN fid", "Nearest Neighbor", "Fid.", "fid", ".2f")]


def table_im_vs_nn(F: Frames) -> Table:
    """tab:im_vs_nn: the inverse map against the nearest-neighbor control: both latent R² at the IM arm's
    point and each editor's Edit Index and Edit Fidelity."""
    rows = main_rows(F, (("Othello", "othello", OTH_BLOCK), ("Rayworld", "rayworld", RW_BLOCK)))
    V = _frame([(s, lab, {"run": r["run"], "block": r["block"], **{c.key: r[c.key] for c in NN_COLS}})
                for s, lab, r in rows])
    return _table("tab:im_vs_nn", "Inverse map vs nearest neighbor", V, NN_COLS)


def table_fidelity_selected(F: Frames) -> Table:
    """tab:fidelity_selected: Table 2 with each editor at its highest Edit Fidelity (``collect(select="fidelity")``)."""
    if F.select != "fidelity":
        raise ValueError('table_fidelity_selected needs collect(..., select="fidelity")')
    return _table("tab:fidelity_selected", "Editability, highest-fidelity setting", _edit_values(main_rows(F)),
                  EDIT_COLS)


def fidelity_rule_shift(F: Frames, F_fid: Frames) -> pd.DataFrame:
    """Per row: IM's Edit Index under the paper's rule and under the highest-fidelity rule, and whether PI's and
    GS's step size shrinks; the largest IM change last."""
    out = []
    for (s, lab, a), (_, _, b) in zip(main_rows(F), main_rows(F_fid)):
        row = {"row": f"{s}: {lab}", "IM EI": a["IM EI"], "IM EI (fidelity rule)": b["IM EI"],
               "IM change": b["IM EI"] - a["IM EI"]}
        for e in ("PI", "GS"):
            row[f"{e} alpha"] = a[f"{e} arm"]["alpha"]
            row[f"{e} alpha (fidelity rule)"] = b[f"{e} arm"]["alpha"]
        out.append(row)
    D = pd.DataFrame(out).set_index("row")
    i = D["IM change"].abs().idxmax()
    D.loc[f"max |IM change|: {i}"] = [np.nan, np.nan, abs(D.at[i, "IM change"])] + [np.nan] * 4
    return D
