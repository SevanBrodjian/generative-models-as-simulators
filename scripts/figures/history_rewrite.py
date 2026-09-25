"""Persistent edits on the Standard Rayworld model by rewriting the history (appendix figure and numbers).

The edited state keeps the moved disc's velocity, so integrating it backward gives the counterfactual
state at every earlier frame. The history is rebuilt one frame at a time: the model reads the frames
rewritten so far, its latent state at the last position is overwritten with g(counterfactual state) at
the inverse map's residual point, and its prediction becomes the next frame (the first frame is kept).
Four 15-step rollouts on the first 32 bench cases are scored with the ray-zone Edit Index: unedited, a
single inverse-map write at the edit frame, the rewritten history alone, and the rewritten history
with the same write. The figure draws three of the cases whose teleport moves the disc by at least 20
rays (``numpy.random.default_rng(0)``); the numbers are printed and saved beside the figure.

    python scripts/figures/history_rewrite.py
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import style
import torch

from pim.editors.inverse import inverse_overwrite
from pim.environments import layout
from pim.environments.rayworld import arms as rwa
from pim.environments.rayworld import bench as rwb
from pim.environments.rayworld.bench import EF, K_ROLL, N_OBJ, full_state_pair
from pim.environments.rayworld.config import SimConfig
from pim.environments.rayworld.renderer import render_frame
from pim.metrics.selection import best_arm
from pim.models import load_checkpoint
from pim.models.protocol import free_run
from qualitative_rayworld import ray_center

plt = style.plt
INSTANCE, BLOCK, N_CASES, PROBE_SEQS = "standard", "cartesian", 32, 30_000
N_CTX, MIN_DISP, SEED = 8, 20.0, 0      # history frames drawn; eligible teleports move >= MIN_DISP rays
COLUMNS = [("Ground truth", "gt"), ("Unedited", "unedited"), ("Single-point edit", "IM"),
           ("History rewrite", "hist+IM")]
FRAME_H, COL_GAP, ROW_GAP = 0.042, 0.07, 0.11                  # inches
LEFT, RIGHT, TOP, BOTTOM = 0.30, 0.02, 0.20, 0.30
PDF_PPI = 600                           # keeps every one of the 128 rays in the PDF's rasters


@torch.no_grad()
def compute() -> dict:
    run_dir = style.REPO / "runs" / "rayworld" / INSTANCE
    scores = json.loads((run_dir / "scores.json").read_text())
    pt = int(best_arm(scores["bases"][BLOCK]["arms"], "IM", "edit_index")["point"])
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=rwb.DEV)
    model.eval()
    b = rwb.load_bench(model, n=N_CASES, target="full", basis_name=BLOCK, instance=INSTANCE)

    # the counterfactual history: the edited disc integrated backward from the write's target state
    dt, ar, eo = float(b.sim["dt"]), np.arange(b.n), b.edit_object
    pos_cf, vel_cf = b.pos[:, :EF].copy(), b.vel[:, :EF].copy()
    v = b.vel[ar, EF - 1, eo]
    target = b.pos[ar, EF, eo] - v * dt
    for t in range(EF):
        pos_cf[ar, t, eo] = target - (EF - 1 - t) * v * dt
        vel_cf[ar, t, eo] = v
    s_cf = np.concatenate([pos_cf.reshape(b.n, EF, -1), vel_cf.reshape(b.n, EF, -1)], -1).astype(np.float32)
    _, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, BLOCK)
    assert np.allclose(s_cf[:, EF - 1], s_post, atol=1e-4), "the history must end in the write's target state"

    # the simulator's clean render of the counterfactual history
    sel = np.asarray(json.loads(layout.edits_selection("rayworld", INSTANCE).read_text())["select"], int)[:N_CASES]
    with h5py.File(layout.edits_file("rayworld", INSTANCE), "r") as f:
        radii, refl = f["radii"][:][sel, :N_OBJ], f["reflectivities"][:][sel, :N_OBJ]
        cfg = SimConfig(**json.loads(f.attrs["config_json"])["dataset"]["sim"])
    cf_render = np.stack([np.stack([render_frame(pos_cf[i, t], radii[i], refl[i], cfg)[2] for t in range(EF)])
                          for i in range(b.n)]).astype(np.float32)

    gen = rwa.iter_inverse_maps(model, basis_name=BLOCK, points=[pt], cache_dir=run_dir / "probes", bank=False,
                                log=None, **rwa.probe_recipe("full", INSTANCE, n_seq=PROBE_SEQS))
    _, g, _, _ = next(gen)
    gen.close()
    S = torch.from_numpy(s_cf).to(rwb.DEV)
    obs_cf = torch.from_numpy(b.obs[:, :EF]).to(rwb.DEV).clone()
    for t in range(EF - 1):
        st = model.state_from_obs(obs_cf[:, :t + 1])
        obs_cf[:, t + 1] = model.decode_with_edit(st, pt, inverse_overwrite(g, S[:, t]))
    h_post = inverse_overwrite(g, torch.from_numpy(s_post).to(rwb.DEV))
    st_cf = model.state_from_obs(obs_cf)
    rolls = {"unedited": rwa.unsteered_rollout(model, b),
             "IM": model.rollout_with_edit(b.state, pt, h_post, K_ROLL).cpu().numpy(),
             "hist+IM": model.rollout_with_edit(st_cf, pt, h_post, K_ROLL).cpu().numpy(),
             "hist": free_run(model, model.decode(st_cf), model.advance(st_cf, model.decode(st_cf)),
                              K_ROLL).cpu().numpy()}
    obs_cf = obs_cf.cpu().numpy()
    cards = {k: rwa.score(model, b, r) for k, r in rolls.items()}
    for k in cards:
        cards[k]["fidelity_ratio"] = 1.0 if k == "unedited" else rwa.fidelity_ratio(cards[k], cards["unedited"])
    rmse = lambda x: float(np.sqrt(((x[:, 1:] - cf_render[:, 1:]) ** 2).mean()))  # noqa: E731
    numbers = {"run": f"rayworld/{INSTANCE}", "block": BLOCK, "im_point": pt, "n_cases": int(b.n),
               "arms": {k: {"edit_index": c["edit_index"], "edit_fidelity": 1.0 - c["fidelity_ratio"],
                            "edit_index_step1": c["edit_index_by_step"][1],
                            "edit_index_step14": c["edit_index_by_step"][14]} for k, c in cards.items()},
               "history_rmse_vs_counterfactual_render": {"original": rmse(b.obs[:, :EF]), "rewritten": rmse(obs_cf)}}
    return {"numbers": numbers, "context": b.obs[:, :EF], "gt": b.gt_roll, **rolls,
            "origin_x": np.array([ray_center(b.zones.ghost[i]) for i in range(b.n)]),
            "dest_x": np.array([ray_center(b.zones.target[i]) for i in range(b.n)])}


def figure(d: dict, cases: list):
    """Rows = cases; every column shows the same history above the dashed edit frame, its rollout below."""
    width = style.TEXT_WIDTH_IN
    panel_w = (width - LEFT - RIGHT - COL_GAP * (len(COLUMNS) - 1)) / len(COLUMNS)
    panel_h = FRAME_H * (N_CTX + K_ROLL)
    height = TOP + len(cases) * panel_h + (len(cases) - 1) * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(width, height), dpi=PDF_PPI)
    for r, case in enumerate(cases):
        y = TOP + r * (panel_h + ROW_GAP)
        for c, (name, key) in enumerate(COLUMNS):
            ax = style.box(fig, LEFT + c * (panel_w + COL_GAP), y, panel_w, panel_h)
            style.waterfall(ax, d["context"][case, EF - N_CTX:EF], d[key][case, :K_ROLL],
                            d["origin_x"][case], d["dest_x"][case])
            if r == 0:
                ax.set_title(name, pad=3)
        style.arrow(fig, LEFT - 0.12, y, LEFT - 0.12, y + panel_h)
        fig.text((LEFT - 0.19) / width, 1 - (y + panel_h / 2) / height, "time", rotation=90, ha="center",
                 va="center", fontsize=8, color=style.TEXT)
    y_bot = TOP + len(cases) * panel_h + (len(cases) - 1) * ROW_GAP
    style.arrow(fig, LEFT, y_bot + 0.10, LEFT + panel_w, y_bot + 0.10)
    fig.text((LEFT + panel_w / 2) / width, 1 - (y_bot + 0.20) / height, "ray", ha="center", va="center",
             fontsize=8, color=style.TEXT)
    style.line_key(fig, LEFT + panel_w + COL_GAP + 0.08, y_bot + 0.10,
                   [("h", style.EDIT_LINE, "edit frame"), ("v", style.ORIGIN_C, "origin position"),
                    ("v", style.DEST_C, "destination position")])
    return fig


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    d = compute()
    n = d["numbers"]
    print(f"rayworld/{INSTANCE}, {n['n_cases']} cases, IM at residual point {n['im_point']}")
    print(f"{'rollout':<10}{'Edit Index':>12}{'Fidelity':>10}{'step 1':>9}{'step 14':>9}")
    for k, a in n["arms"].items():
        print(f"{k:<10}{a['edit_index']:>+12.2f}{a['edit_fidelity']:>10.2f}{a['edit_index_step1']:>+9.2f}"
              f"{a['edit_index_step14']:>+9.2f}")
    h = n["history_rmse_vs_counterfactual_render"]
    print(f"history RMSE against the counterfactual render: original {h['original']:.3f}, "
          f"rewritten {h['rewritten']:.3f}")
    disp = np.abs(d["dest_x"] - d["origin_x"])
    eligible = [int(i) for i in np.flatnonzero(np.nan_to_num(disp, nan=-1.0) >= MIN_DISP)]
    cases = sorted(int(i) for i in np.random.default_rng(SEED).choice(eligible, 3, replace=False))
    print(f"drawn cases {cases} (of {len(eligible)} moving the disc by >= {MIN_DISP:g} rays)")
    stem = style.save(figure(d, cases), "appendix/history_rewrite")
    stem.with_suffix(".json").write_text(json.dumps({**n, "drawn_cases": cases}, indent=1))


if __name__ == "__main__":
    main()
