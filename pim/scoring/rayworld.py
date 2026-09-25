"""The Rayworld scorers: frame models (ray-zone Edit Index over a rollout) and frames-as-tokens
models (frame-set Edit Index on the next-frame distribution). Wiring over
``pim.environments.rayworld``; the bench and probe corpus are always the run's own instance.
"""
import json

from pim.environments.rayworld import arms as rwa
from pim.environments.rayworld import bench as rwb
from pim.environments.rayworld import token_bench as tkb
from pim.environments.rayworld.tokens import FrameVocab
from pim.probes.inverse import CATEGORICAL_STATE
from pim.probes.mlp import check_probe_sanity
from pim.scoring.blocks import (attach_inverse, cat_inverse_in_scope, probe_block, rayworld_blocks,
                                rw_block_setup, rw_probe_recipe)
from pim.scoring.runs import REPO


def inverse_rayworld(model, blocks: dict, benches: dict, ucards: dict, inst: str, probe_dir, s,
                     tokens=None) -> None:
    """IM / IM-NN for the blocks in ``benches``; ``ucards`` holds each block's unsteered card
    (frames) or unsteered probabilities (tokens, with ``tokens=(vocab, arrays)``).

    The inverse map inverts the state the block's own probes read: a regression block's
    continuous full state in its basis (IM and IM-NN); a categorical block's one-hot labels plus
    Cartesian velocity, fitted with the target's forward-probe recipe (IM only, where
    ``cat_inverse_in_scope``)."""
    reg = [k for k in benches if blocks[k]["kind"] == "regression"]
    for k in (k for k in benches if blocks[k]["kind"] != "regression"):
        if not cat_inverse_in_scope(inst, blocks[k]["target"], s):
            continue
        recipe = rw_probe_recipe(blocks[k]["target"], inst, s)
        if tokens is None:
            arms, st = rwa.inverse_arms(model, {k: benches[k]}, basis_name=blocks[k]["basis"],
                                        target=blocks[k]["target"], unsteered_cards={k: ucards[k]},
                                        cache_dir=probe_dir, log=None, **recipe)
        else:
            vocab, arrays = tokens
            arms, st = tkb.inverse_arms(model, {k: benches[k]}, {k: arrays[k]}, vocab,
                                        basis_name=blocks[k]["basis"], target=blocks[k]["target"],
                                        uns={k: ucards[k]}, cache_dir=probe_dir, log=None, **recipe)
        attach_inverse(blocks, arms, st, ei_key="edit_index")
        blocks[k]["inverse_map"].update({"state": CATEGORICAL_STATE, "epochs": recipe["epochs"],
                                         "n_seq": recipe["n_seq"], "g_r2_insample": st.get("g_r2_insample")})
        im = blocks[k]["best"]["IM"]
        print(f"    {k}: IM[categorical state] {im['edit_index']:+.4f}/{im['fidelity_ratio']:.2f} (pt {im['point']})  "
              f"g R² max {max(st['g_r2']):+.3f}", flush=True)
    recipe = rw_probe_recipe("full", inst, s)
    for basis in sorted({blocks[k]["basis"] for k in reg}):
        keys = [k for k in reg if blocks[k]["basis"] == basis]
        if tokens is None:
            arms, st = rwa.inverse_arms(model, {k: benches[k] for k in keys}, basis_name=basis,
                                        unsteered_cards={k: ucards[k] for k in keys}, cache_dir=probe_dir,
                                        log=None, **recipe)
        else:
            vocab, arrays = tokens
            arms, st = tkb.inverse_arms(model, {k: benches[k] for k in keys}, {k: arrays[k] for k in keys}, vocab,
                                        basis_name=basis, uns={k: ucards[k] for k in keys}, cache_dir=probe_dir,
                                        log=None, **recipe)
        attach_inverse(blocks, arms, st, ei_key="edit_index")
        for k in keys:
            im = blocks[k]["best"]["IM"]
            print(f"    {k}: IM {im['edit_index']:+.4f}/{im['fidelity_ratio']:.2f} (pt {im['point']})  "
                  f"IM-NN {blocks[k]['best']['IM-NN']['edit_index']:+.4f}  g R² max {max(st['g_r2']):+.3f}", flush=True)


def _fit(model, target, basis, probe_dir, recipe, cat, key, **enc):
    """(linear, MLP) probe sets; a categorical target's probes must already be cached (fit them
    with ``scripts/fit_probes.py``), so a miss returns None and the block is skipped."""
    try:
        return tuple(rwa.fit_probes(model, target=target, family=fam, basis_name=basis, cache_dir=probe_dir,
                                    log=None, require_cached=cat is not None, **enc, **recipe)
                     for fam in ("linear", "mlp"))
    except RuntimeError as e:
        print(f"    {key}: SKIPPED ({str(e).splitlines()[0][:90]})", flush=True)
        return None


def score_rayworld(model, run_dir, s, only=None) -> dict:
    """Every block of ``rayworld_blocks`` for a frame model (or only the keys in ``only``)."""
    probe_dir = run_dir / "probes"
    run_id = f"{run_dir.parent.name}/{run_dir.name}"
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    out = {"probe_dir": f"runs/{run_id}/probes", "instance": inst, "target": s["rw_target"],
           "edit_dims": list(s["rw_edit_dims"]), "bases": {}}
    benches, ucards = {}, {}
    for key, target, basis in rayworld_blocks(run_id, s):
        if only is not None and key not in only:
            continue
        cat, dimsets, (a_pi, a_gs) = rw_block_setup(target, s)
        recipe = rw_probe_recipe(target, inst, s)
        fits = _fit(model, target, basis, probe_dir, recipe, cat, key)
        if fits is None:
            continue
        lin, mlp = fits
        b = rwb.load_bench(model, n=s["rw_bench_n"], target=target, basis_name=basis, instance=inst)
        sanity = check_probe_sanity(lin, mlp, strict=False, log=print, label=key)
        u = rwa.unsteered(model, b)
        arms = []
        for dims in dimsets:
            arms += rwa.pinv_arm(model, b, lin, a_pi, dims=dims)
            arms += rwa.grad_steer_arm(model, b, mlp, s["gs_layers"], a_gs,
                                       n_steps=s["rw_gs_steps"], beta=s["rw_gs_beta"], dims=dims)
        for r in arms:
            r["fidelity_ratio"] = rwa.fidelity_ratio(r, u)
        block = probe_block(lin, mlp, sanity, u, arms, dimsets, target=target, basis=basis,
                            kind=b.kind, n_classes=next(iter(lin.values()))[0].n_classes if cat else None,
                            recipe=recipe, alphas=(a_pi, a_gs), selection=b.selection)
        out["bases"][key] = block
        benches[key], ucards[key] = b, u
        pi = block["best"]["PI"]
        print(f"    {key}: skill lin {max(block['probe_skill_linear']):+.4f}"
              f"  PI {pi['edit_index']:+.4f}/{pi['fidelity_ratio']:.2f}"
              f" (dims={pi['dims']})", flush=True)
    if benches:
        inverse_rayworld(model, out["bases"], benches, ucards, inst, probe_dir, s)
    return out


def score_rayworld_tokens(model, run_dir, s, only=None) -> dict:
    """Every block of ``rayworld_blocks`` for a frames-as-tokens model: probes read token inputs,
    and the Edit Index is the frame-set construction at the edit frame (+1 = the edited world's
    frame, -1 = the unedited one), guarded by the move-fidelity ratio."""
    probe_dir = run_dir / "probes"
    run_id = f"{run_dir.parent.name}/{run_dir.name}"
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    vocab = FrameVocab.load(run_dir / "vocab.npz")        # the run's own vocabulary
    enc, tag = tkb.token_encoder(vocab)
    sel_path = tkb.selection_path(instance=inst)
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else None
    out = {"probe_dir": f"runs/{run_id}/probes", "instance": inst, "target": s["rw_target"],
           "edit_dims": list(s["rw_edit_dims"]), "repr": "tokens", "vocab_size": int(vocab.size),
           "ei_construction": "frame-set",
           "bench_selection": ({"file": str(sel_path.relative_to(REPO)), "rule": sel["rule"],
                                "n": sel["n"], "min_rays": sel["min_rays"],
                                "pool": sel["pool"], "stats": sel["stats"]}
                               if sel else "first-n (no selection file)"),
           "bases": {}}
    benches, ucards, arrays = {}, {}, {}
    for key, target, basis in rayworld_blocks(run_id, s):
        if only is not None and key not in only:
            continue
        cat, dimsets, (a_pi, a_gs) = rw_block_setup(target, s)
        recipe = rw_probe_recipe(target, inst, s)
        fits = _fit(model, target, basis, probe_dir, recipe, cat, key, encoder=enc, encoder_tag=tag)
        if fits is None:
            continue
        lin, mlp = fits
        tb = tkb.load_token_bench(vocab, n=s["rw_bench_n"], target=target, basis_name=basis, instance=inst)
        sanity = check_probe_sanity(lin, mlp, strict=False, log=print, label=key)
        uns, u = tkb.unsteered(model, tb)
        arms = []
        for dims in dimsets:
            arms += tkb.pinv_arm(model, tb, lin, a_pi, uns, dims=dims)
            arms += tkb.grad_steer_arm(model, tb, mlp, s["gs_layers"], a_gs, uns,
                                       n_steps=s["rw_gs_steps"], beta=s["rw_gs_beta"], dims=dims)
        block = probe_block(lin, mlp, sanity, u, arms, dimsets, target=target, basis=basis,
                            kind=tb.kind, n_classes=next(iter(lin.values()))[0].n_classes if cat else None,
                            recipe=recipe, alphas=(a_pi, a_gs),
                            selection=tb.selection if tb.selection else out["bench_selection"],
                            extra={"n_cases_kept": int(tb.keep.sum())})   # cases whose two frames differ
        out["bases"][key] = block
        benches[key], ucards[key] = tb, uns
        arrays[key] = rwb.bench_arrays(s["rw_bench_n"], target, basis, instance=inst)
        pi = block["best"]["PI"]
        print(f"    {key}: skill lin {max(block['probe_skill_linear']):+.4f}"
              f"  unedited {block['unedited']['edit_index']:+.4f}"
              f"  PI {pi['edit_index']:+.4f}/{pi['fidelity_ratio']:.2f} (dims={pi['dims']})"
              f"  cases kept {block['n_cases_kept']}/{tb.n}", flush=True)
    if benches:
        inverse_rayworld(model, out["bases"], benches, ucards, inst, probe_dir, s, tokens=(vocab, arrays))
    return out
