"""The Othello scorer: mine/theirs probes (held out by sequence), PI / GS / IM / IM-NN over the
instance's own edit cases, legal-set Edit Index. Wiring over ``pim.environments.othello``; the
corpus splits, rules and bench all come from the run's ``data.instance``.
"""
import json

from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello import case_targets, load_benchmark
from pim.environments.othello.data import canonical_vocab, tokens_and_labels
from pim.metrics.decodability import probe_skill_from_stats
from pim.metrics.set_editability import move_fidelity_ci95, move_fidelity_ratio
from pim.models import n_points
from pim.scoring.blocks import EDITORS_SCORED, IM_VERSION

PROBE_SOURCES = {
    "PI": "mine|linear|sequence",
    "GS": "mine|mlp|sequence (targets: mine/theirs via case_targets)",
}


def _probe_games(n, instance="standard"):
    """The first ``n`` games of the instance's probe split, labeled under its rules."""
    tok, ln = oc.load(oc.build(oc.N_TRAIN_GAMES, log=lambda s: None, only=("probe",),
                               instance=instance)["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:L]]
                              for row, L in zip(tok[:n], ln[:n])], **oc.rules_of(instance))


def _guarded(pr, card, uns_probs, bench, **rec) -> dict:
    return {**rec, "fidelity_ratio": move_fidelity_ratio(pr, uns_probs, bench.legal_post),
            **move_fidelity_ci95(pr, uns_probs, bench.legal_post),
            **{k: v for k, v in card.items() if isinstance(v, (int, float))}}


def othello_arms(model, bench, lin, mlp, tgt, cur, uns_probs, alphas, s):
    """PI at every residual point and GS from every start layer, each arm with its guard."""
    a_pi, a_gs = alphas
    arms_out = []
    for ell in range(n_points(model)):
        for a in a_pi:
            pr, card = oa.linear_arm(model, bench, lin, tgt, cur, alpha=a, points={ell})
            arms_out.append(_guarded(pr, card, uns_probs, bench, editor="PI", point=ell, alpha=a))
    # GS targets must be in the probes' mine/theirs frame, not absolute color
    for ls in s["oth_gs_layers"]:
        for a in a_gs:
            pr, card = oa.grad_steer_arm(model, bench, mlp, ls, alpha=a, n_steps=s["oth_gs_steps"],
                                         beta=s["oth_gs_beta"], target_labels=tgt)
            arms_out.append(_guarded(pr, card, uns_probs, bench, editor="GS", point=ls, alpha=a))
    for r in arms_out:
        r["edit_index"] = r["edit_index_union"]
    return arms_out


def score_othello(model, run_dir, s, only=None) -> dict:
    """The run's one block ("mine/theirs", stored at the top level); ``only`` is accepted for
    symmetry with the Rayworld scorers."""
    probe_dir = run_dir / "probes"
    run_id = f"{run_dir.parent.name}/{run_dir.name}"
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    data = _probe_games(s["oth_probe_games"], inst)
    bench = load_benchmark(inst)
    cur, tgt = case_targets(bench)
    u = oa.unsteered(model, bench)
    uns_probs = oa.unsteered_probs(model, bench)   # the guard's reference
    npnt = n_points(model)
    out = {"instance": inst, "rules": rules,
           "bench": f"{bench.n_cases} single-tile flips at a fixed 20-move prefix, from {inst}'s own edits games",
           "probe_dir": f"runs/{run_id}/probes", "bases": {}}
    if only is not None and "mine/theirs" not in only:
        return out
    # held-out gates on the test split (the generator's Bayes floor is exact)
    tok, ln = oc.load(oc.build(oc.N_TRAIN_GAMES, log=lambda s_: None, only=("test",),
                               instance=inst)["test"])
    g = oa.gates(model, tok[: s["oth_gates_games"]], ln[: s["oth_gates_games"]], log=None, **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=probe_dir, log=None)
    skill = {}
    for st in grid.stats:
        skill.setdefault((st["target"], st["family"], st["split"]), []).append(probe_skill_from_stats(st))
    lin_mine = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
    mlp_mine = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(npnt)}
    arms_out = othello_arms(model, bench, lin_mine, mlp_mine, tgt, cur, uns_probs,
                            (s["oth_alpha_pi"], s["oth_alpha_gs"]), s)
    im, im_st = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=probe_dir,
                                n_games=s["oth_probe_games"], uns_probs=uns_probs, log=None)
    for r in im:
        r["edit_index"] = r["edit_index_union"]
    out["inverse_map"] = {"g_r2": im_st["g_r2"], "g_rmse": im_st["g_rmse"], "version": IM_VERSION,
                          "nn_r2": im_st.get("nn_r2")}
    arms_out += im

    def best(ed):
        sub = [r for r in arms_out if r["editor"] == ed]
        return max(sub, key=lambda r: r["edit_index_union"]) if sub else None
    out |= {
        "gates": g,
        "probe_sources": PROBE_SOURCES,
        "probe_skill": {"|".join(k): v for k, v in skill.items()},
        "probe_stats": [{k: v for k, v in st.items() if not isinstance(v, list)} for st in grid.stats],
        "unedited": {**{k: v for k, v in u.items() if isinstance(v, (int, float))}, "fidelity_ratio": 1.0},
        "best": {ed: best(ed) for ed in EDITORS_SCORED},
        "arms": arms_out,
    }
    return out
