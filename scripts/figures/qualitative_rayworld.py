"""The appendix Rayworld edit figures: one teleport per figure, seen through every variant.

Each seed generates one two-disc world with a teleport at the edit frame (under the coarse-ray
geometry, so the world is valid for every variant) and renders it with each variant's own renderer.
Per column: the last 8 observed frames, the true next frame without and with the teleport, then each
editor's next-step prediction above its error against the edited truth, first through the continuous
state (Cartesian block) and then through the categorical appearance bins (appearance-fac block). Each
editor is drawn at the setting Table 2 reports (``pim.metrics.selection.best_arm``); a block with no
setting for an editor leaves its cell blank. Probes and inverse maps are read from each run's cache.
The default seeds are the first five, after the main-text figure's 0-2, whose teleport changes at
least two rays of the 5-ray frame by at least 0.2.

    python scripts/figures/qualitative_rayworld.py [--seeds 5 7 9 10 12] [--recompute]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle

import h5py
import numpy as np
import style
import torch
from matplotlib.gridspec import GridSpec

from pim.editors.inverse import inverse_overwrite
from pim.environments import layout
from pim.environments.rayworld import arms as rwa
from pim.environments.rayworld import bench as rwb
from pim.environments.rayworld.bench import EF, N_OBJ, full_state_pair
from pim.environments.rayworld.blink import blink_schedule
from pim.environments.rayworld.config import SimConfig
from pim.environments.rayworld.edits_dataset import _generate_one_edit
from pim.environments.rayworld.renderer import render_scene
from pim.environments.rayworld.sim import Scene
from pim.metrics.selection import best_arm
from pim.models import load_checkpoint
from pim.probes.inverse import encode_categorical_state

plt = style.plt
VARIANTS = [("Standard", "standard"), ("Blink", "blink"), ("16-ray", "16-ray"), ("8-ray", "8-ray"),
            ("5-ray", "5-ray")]
SEEDS = (5, 7, 9, 10, 12)
BLOCKS = {"cont": ("cartesian", "full", "cartesian"),              # (scores.json block, target, basis)
          "cat": ("appearance-fac", "appearance-fac", "frustum")}
EDITORS = ("PI", "GS", "IM")
CONTEXT = 8
PROBE_SEQS = 30_000                     # the scorer's regression probe corpus size
CACHE = style.CACHE / "qualitative_rayworld.pkl"


# ── the scenario: one world, rendered by each variant ─────────────────────────


def sim_config(inst: str) -> SimConfig:
    """The instance's simulator config, as stored with its edit split."""
    with h5py.File(layout.edits_file("rayworld", inst), "r") as f:
        return SimConfig(**json.loads(f.attrs["config_json"])["dataset"]["sim"])


def scenario_config() -> SimConfig:
    """The largest-disc geometry among the variants, so the world fits every one of them."""
    return max((sim_config(inst) for _, inst in VARIANTS), key=lambda c: c.radius)


def scenario(seed: int, base: SimConfig) -> dict:
    """One teleport case from the edit-set generator; positions after the edit."""
    r = _generate_one_edit((seed, base, N_OBJ, EF, True, 2000))
    return {"pos": r["positions"][:, :N_OBJ].astype(np.float64), "vel": r["velocities"][:, :N_OBJ].astype(np.float64),
            "refl": r["reflectivities"][:N_OBJ].astype(np.float64), "colors": r["colors"][:N_OBJ].astype(np.float64),
            "edit_object": int(r["edit_object"])}


def render(sc: dict, cfg: SimConfig, seed: int):
    """(observed, clean, blink schedule) frames of the scenario under ``cfg``; a blink schedule is
    forced visible from the frame before the edit on, so the edit can be seen."""
    cfg = dataclasses.replace(cfg, seed=int(seed), n_objects=N_OBJ)
    scene = Scene(positions=sc["pos"], velocities=sc["vel"], radii=np.full(N_OBJ, cfg.radius),
                  colors=sc["colors"], reflectivities=sc["refl"], config=cfg)
    vis = blink_schedule(cfg, N_OBJ)
    if vis is not None:
        vis = vis.copy()
        vis[EF - 1:] = True
    obs = render_scene(scene, visible=vis)[2]
    clean = render_scene(dataclasses.replace(scene, config=dataclasses.replace(cfg, obs_noise_std=0.0)),
                         visible=vis)[2]
    return obs.astype(np.float32), clean.astype(np.float32), vis


def ray_center(mask) -> float:
    idx = np.flatnonzero(mask)
    return float(idx.mean()) if idx.size else float("nan")


# ── predictions ────────────────────────────────────────────────────────────


def editor_settings(scores: dict) -> dict:
    """{block: {editor: arm or None}} at the Table 2 selection; a block absent from scores.json has none."""
    out = {}
    for key, (blk, _, _) in BLOCKS.items():
        arms = scores["bases"].get(blk, {}).get("arms", [])
        out[key] = {ed: best_arm(arms, ed, "edit_index") for ed in EDITORS}
    return out


def inverse_map(model, cache_dir, inst: str, key: str, point: int):
    """(g, n_classes) at ``point`` for the block: the continuous map of the full state, or the
    categorical map of the block's labels plus the Cartesian velocity."""
    blk, target, basis = BLOCKS[key]
    kw = rwa.probe_recipe(target, inst, n_seq=PROBE_SEQS)
    # no retrieval bank: a cached map then reads no probe corpus
    gen = rwa.iter_inverse_maps(model, basis_name=basis, target=target, points=[point], cache_dir=cache_dir,
                                bank=False, log=None, **kw)
    _, g, _, st = next(gen)
    gen.close()
    return g, st.get("n_classes")


@torch.no_grad()
def write_im(model, b, g, n_classes, point: int, categorical: bool) -> np.ndarray:
    """The next frame after writing g(post-edit state) at ``point``."""
    _, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, "cartesian")
    s_post = torch.from_numpy(s_post).to(rwb.DEV)
    if categorical:
        s_post = encode_categorical_state(b.tgt, s_post[:, 2 * N_OBJ:], n_classes)
    rwa.as_activations(model, point)
    return model.decode_with_edit(b.state, point, inverse_overwrite(g, s_post))[0].cpu().numpy()


def predict(inst: str, seeds, base: SimConfig) -> dict:
    """{seed: column} for one variant: context, both ground truths, the locators, and every editor's
    next frame per block (None where the block has no setting for the editor)."""
    run_dir = style.REPO / "runs" / "rayworld" / inst
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=rwb.DEV)
    model.eval()
    scores = json.loads((run_dir / "scores.json").read_text())
    settings = editor_settings(scores)
    cache_dir = run_dir / "probes"
    cfg = sim_config(inst)
    tools = {}
    for key, (blk, target, basis) in BLOCKS.items():
        s = settings[key]
        if all(a is None for a in s.values()):
            print(f"  note: rayworld/{inst} has no {blk} block in scores.json; its cells are left blank")
            continue
        kw = rwa.probe_recipe(target, inst, n_seq=PROBE_SEQS)
        probes = {fam: rwa.fit_probes(model, target=target, family=fam, basis_name=basis, cache_dir=cache_dir,
                                      log=None, require_cached=True, **kw) for fam in ("linear", "mlp")}
        im = None if s["IM"] is None else inverse_map(model, cache_dir, inst, key, s["IM"]["point"])
        tools[key] = (probes, im)
    out = {}
    for seed in seeds:
        sc = scenario(seed, base)
        obs, clean, vis = render(sc, cfg, seed)
        sim = dataclasses.asdict(dataclasses.replace(cfg, seed=int(seed), n_objects=N_OBJ))
        col = {"context": obs[EF - CONTEXT:EF], "edited_gt": clean[EF]}
        for key, (blk, target, basis) in BLOCKS.items():
            n1 = lambda x: np.asarray(x)[None]  # noqa: E731
            a = rwb.bench_from_arrays(n1(obs), n1(sc["pos"]).astype(np.float32), n1(sc["vel"]).astype(np.float32),
                                      np.array([sc["edit_object"]]), n1(clean), sim, None if vis is None else n1(vis),
                                      target=target, basis_name=basis)
            if key == "cont":
                z = a["zones"]
                col |= {"unedited_gt": z.gt_unedited[0], "origin_x": ray_center(z.ghost[0]),
                        "dest_x": ray_center(z.target[0])}
            frames = dict.fromkeys(EDITORS)
            if key in tools:
                b = rwb.bench_of(model, a)
                (probes, im), s = tools[key], settings[key]
                if s["PI"] is not None:
                    pt = s["PI"]["point"]
                    frames["PI"] = rwa.pinv_rollout(model, b, probes["linear"][pt][0], pt, s["PI"]["alpha"],
                                                    dims=s["PI"].get("dims", "all"))[0, 0]
                if s["GS"] is not None:
                    frames["GS"] = rwa.grad_steer_rollout(model, b, probes["mlp"], s["GS"]["point"], s["GS"]["alpha"],
                                                          dims=s["GS"].get("dims", "all"))[0, 0]
                if im is not None:
                    frames["IM"] = write_im(model, b, *im, s["IM"]["point"], key == "cat")
            col[key] = frames
        col["settings"] = {k: {ed: None if a is None else (int(a["point"]), float(a["alpha"])) for ed, a in s.items()}
                           for k, s in settings.items()}
        out[seed] = col
    print(f"  rayworld/{inst}: settings {out[seeds[0]]['settings']}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def columns(pairs, recompute: bool = False) -> dict:
    """{(instance, seed): column}, from ``outputs/cache/`` where present, else computed (GPU)."""
    have = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    todo = {}
    for inst, seed in pairs:
        if recompute or (inst, seed) not in have:
            todo.setdefault(inst, []).append(seed)
    if todo:
        base = scenario_config()
        for inst, seeds in todo.items():
            for seed, col in predict(inst, seeds, base).items():
                have[(inst, seed)] = col
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(have))
    return {p: have[p] for p in pairs}


# ── the appendix figure ─────────────────────────────────────────────────────

STRIP, CTX_ROW, ERR, GAP, BIGGAP, COL_W = 1.0, 0.55, 0.75, 0.22, 0.8, 3.3


def draw(cols: list, seed: int) -> None:
    """Variants as columns; rows: context, the two ground truths, and prediction + error per editor."""
    def edit_rows(key):
        rows = []
        for ed in EDITORS:
            rows += [(key, ed, "pred", STRIP), (key, ed, "err", ERR), ("gap", None, None, GAP)]
        return rows[:-1]
    rows = ([("context", None, None, CONTEXT * CTX_ROW), ("gap", None, None, GAP),
             ("unedited_gt", None, None, STRIP), ("gap", None, None, GAP), ("edited_gt", None, None, STRIP),
             ("gap", None, None, BIGGAP)] + edit_rows("cont") + [("gap", None, None, BIGGAP)] + edit_rows("cat"))
    heights = [r[3] for r in rows]
    fig = plt.figure(figsize=(COL_W * len(cols) + 1.6, 0.40 * sum(heights) + 1.2))
    gs = GridSpec(len(rows), len(cols), figure=fig, height_ratios=heights, left=0.15, right=0.945, top=0.94,
                  bottom=0.01, wspace=0.10, hspace=0.0)
    first = {}
    for c, (name, col) in enumerate(cols):
        for r, (kind, ed, sub, _) in enumerate(rows):
            if kind == "gap":
                continue
            ax = fig.add_subplot(gs[r, c])
            if kind in ("context", "unedited_gt", "edited_gt"):
                style.strip(ax, col[kind])
            elif col[kind][ed] is None:
                ax.set_axis_off()       # no setting for this editor in this block
            elif sub == "pred":
                style.strip(ax, col[kind][ed])
            else:
                style.strip(ax, style.error(col[kind][ed], col["edited_gt"]), diff=True)
            if kind == "context":
                ax.set_title(name, fontsize=24, pad=8, color=style.TEXT)
            elif ax.axison:
                style.locators(ax, col["origin_x"], col["dest_x"], lw=1.3)
            if c == 0:
                first[(kind, ed, sub)] = ax
    x_lab = first[("context", None, None)].get_position().x0 - 0.006
    labels = {"context": f"last {CONTEXT} frames", "unedited_gt": "Unedited GT", "edited_gt": "Edited GT"}
    for (kind, ed, sub), ax in first.items():
        if sub == "err":
            continue
        low = first.get((kind, ed, "err"), ax)     # an editor's label is centered on prediction + error
        y = (low.get_position().y0 + ax.get_position().y1) / 2
        fig.text(x_lab, y, labels.get(kind, ed), ha="right", va="center", fontsize=17, color=style.TEXT,
                 fontweight="bold" if kind == "edited_gt" else "normal")
    x_line = x_lab - 0.040
    for key, text in (("cont", "Continuous\npositions"), ("cat", "Categorical\npositions")):
        y0 = first[(key, EDITORS[-1], "err")].get_position().y0
        y1 = first[(key, EDITORS[0], "pred")].get_position().y1
        fig.add_artist(plt.Line2D([x_line, x_line], [y0, y1], transform=fig.transFigure, color=style.TEXT, lw=0.9))
        fig.text(x_line - 0.030, (y0 + y1) / 2, text, ha="center", va="center", rotation=90, fontsize=18,
                 color=style.TEXT)
    y0 = first[("cat", EDITORS[-1], "err")].get_position().y0
    y1 = first[("cont", EDITORS[0], "pred")].get_position().y1
    style.colorbar(fig.add_axes([0.958, y0, 0.010, y1 - y0]), "red = under-prediction, green = over",
                   labelsize=12, fontsize=14, labelpad=6, length=2, pad=3.5, width=0.6)
    style.save(fig, f"appendix/rayworld_qualitative/qualitative_edits_seed{seed}_paired", dpi=170)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="scenario seeds, one figure each")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached predictions")
    a = ap.parse_args()
    data = columns([(inst, s) for _, inst in VARIANTS for s in a.seeds], a.recompute)
    for s in a.seeds:
        draw([(name, data[(inst, s)]) for name, inst in VARIANTS], s)


if __name__ == "__main__":
    main()
