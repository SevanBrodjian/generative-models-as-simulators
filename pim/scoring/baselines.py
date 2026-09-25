"""The two decodability floors per (environment, instance, architecture), written to
``runs/_baselines/<env>/<instance>/baselines.json``: the same probes fitted to the causal input
history (observation) and to the same architecture at random init (seed 0, never trained).

Observation floors come in four forms: the history left-aligned or aligned to the present
(``_right``), each on the canonical probe corpus or the large one (``_large``: Rayworld 250k
sequences, Othello 170k games, 50 epochs). Every other choice (corpus, split by sequence,
targets, bases, families) matches the run probes. Both floors depend on the architecture: the
observation probe reads the history the model consumes (``state_span``).
"""
import json
import time

import torch

from pim.environments import layout
from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.rayworld import arms as rwa
from pim.environments.rayworld.grid_target import categorical_target
from pim.metrics.decodability import insample_gap_from_stats, probe_skill_from_stats
from pim.models import load_checkpoint
from pim.probes.baselines import random_init_model
from pim.scoring.blocks import by_point, extra_targets_of, in_scope, rw_bases_for, rw_probe_recipe
from pim.scoring.othello import _probe_games
from pim.scoring.runs import DEV

BASELINE_SEED = 0
BASELINE_VERSION = "1.0"   # independent of the eval version: an editor change leaves floors valid
LARGE = {"rw_n_seq": 250_000, "oth_split": "probe_large", "epochs": 50}
# (block name, history alignment, epochs; None = the canonical corpus and fit length)
OBS_KINDS = (("observation", "left", None), ("observation_large", "left", LARGE["epochs"]),
             ("observation_right", "right", None), ("observation_right_large", "right", LARGE["epochs"]))


def _pack(st):
    """Probe Skill plus the overfit check (``insample_gap`` = train minus held-out skill)."""
    return {"skill": probe_skill_from_stats(st), "insample_gap": insample_gap_from_stats(st),
            "d_in": st.get("d_in"), "n_train_rows": st.get("n_train_rows")}


def _agg(per_pt):
    """A random-init model's best residual point, as every model row reports its best point."""
    best = max(range(len(per_pt)), key=lambda i: per_pt[i]["skill"])
    return {**per_pt[best], "point": best, "per_point": per_pt}


def _rw_encoder(inst, arch):
    """A frames-as-tokens architecture's probes read token inputs: its encoder and cache tag."""
    if not arch.endswith("_tokens"):
        return {}
    from pim.environments.rayworld import token_bench as tkb
    from pim.environments.rayworld.tokens import FrameVocab
    enc, tag = tkb.token_encoder(FrameVocab.load(layout.tokens_dir(inst) / "vocab.npz"))
    return {"encoder": enc, "encoder_tag": tag}


def floor_targets_for(runs, env, inst, arch, s) -> list:
    """The extra probe targets that get floors: every run's extra target on this (env, instance,
    arch) that the SETTINGS scope ``rw_floor_targets`` lists for this instance."""
    if env != "rayworld":
        return []
    out = []
    for r in runs:
        if (r["env"], r["instance"], r["arch"]) == (env, inst, arch):
            out += [t for t in extra_targets_of(r["id"], s)
                    if in_scope(s.get("rw_floor_targets"), inst, t) and t not in out]
    return out


def _rw_regression_floors(rand, span, inst, target, basis, enc, s) -> dict:
    """All four observation floors and the random-init floor for one regression target in one
    basis, fitted inline."""
    pdir = layout.baselines_dir("rayworld", inst) / "probes"
    probe_small, probe_large = {"instance": inst, "size": "120k"}, {"instance": inst, "size": "250k"}
    blk = {k: {} for k, _, _ in OBS_KINDS}
    blk["random_init"] = {}
    for fam in ("linear", "mlp"):
        for kind, align, epochs in OBS_KINDS:
            is_large = epochs is not None
            if is_large and not layout.probe_file("rayworld", inst, "250k").exists():
                continue
            _, st = rwa.observation_probes(
                target=target, n_seq=LARGE["rw_n_seq"] if is_large else s["rw_probe_seqs"],
                family=fam, basis_name=basis, span=span,
                probe=probe_large if is_large else probe_small, cache_dir=pdir,
                log=None, epochs=epochs, align=align)
            blk[kind][fam] = {**_pack(st), **({"n_seq": LARGE["rw_n_seq"], "epochs": epochs}
                                               if is_large else {})}
        fits = rwa.fit_probes(rand, target=target, n_seq=s["rw_probe_seqs"], family=fam,
                              basis_name=basis, probe=probe_small, cache_dir=pdir, log=None, **enc)
        blk["random_init"][fam] = _agg([_pack(st) for st in by_point(fits)])
        orr = blk["observation_right_large"].get(fam, {}).get("skill", float("nan"))
        print(f"    {target}/{basis}/{fam}: obs {blk['observation'][fam]['skill']:+.4f}"
              f"  obs_right_large {orr:+.4f}  random-init "
              f"{blk['random_init'][fam]['skill']:+.4f}", flush=True)
    return blk


def _rw_categorical_floors(rand, span, inst, arch, target, basis, enc, s) -> dict | None:
    """The right-aligned large observation floor and the random-init floor of a categorical
    target, from cached probes only (``scripts/fit_probes.py``); None when they are not fitted."""
    pdir = layout.baselines_dir("rayworld", inst) / "probes"
    rec = rw_probe_recipe(target, inst, s)
    blk = {"random_init": {}, "observation_right_large": {}}
    try:
        for fam in ("linear", "mlp"):
            _, st = rwa.observation_probes(target=target, family=fam, basis_name=basis, span=span,
                                           cache_dir=pdir, log=None, align="right",
                                           require_cached=True, **rec)
            blk["observation_right_large"][fam] = {**_pack(st), "n_seq": rec["n_seq"], "epochs": rec["epochs"]}
            fits = rwa.fit_probes(rand, target=target, family=fam, basis_name=basis,
                                  cache_dir=pdir, log=None, require_cached=True, **rec, **enc)
            blk["random_init"][fam] = _agg([_pack(st) for st in by_point(fits)])
            print(f"    {arch}/{target}/{fam}: obs_right_large "
                  f"{blk['observation_right_large'][fam]['skill']:+.4f}  random-init "
                  f"{blk['random_init'][fam]['skill']:+.4f}", flush=True)
    except RuntimeError as e:
        print(f"    {arch}/{target}: floors SKIPPED ({str(e).splitlines()[0][:80]})", flush=True)
        return None
    return blk


def score_baseline_targets(inst, arch, model_config, targets, s) -> dict:
    """Both floors for a Rayworld instance's extra probe targets: a snapped regression target
    (``pos@<partition>``) fitted inline, a categorical target from cached probes."""
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    basis0 = rw_bases_for(inst, s)[0]
    enc = _rw_encoder(inst, arch)
    out = {}
    for target in targets:
        if categorical_target(target) is None:
            out[target] = _rw_regression_floors(rand, span, inst, target, basis0, enc, s)
            continue
        blk = _rw_categorical_floors(rand, span, inst, arch, target, basis0, enc, s)
        if blk is not None:
            out[target] = blk
    del rand
    return out


def score_baseline_bases(inst, arch, model_config, bases, s) -> dict:
    """The regression floors of the canonical target in each of ``bases`` (Rayworld)."""
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    enc = _rw_encoder(inst, arch)
    out = {b: _rw_regression_floors(rand, span, inst, s["rw_target"], b, enc, s) for b in bases}
    del rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _othello_floors(inst, arch, model_config, s) -> dict:
    """The mine/theirs floors of an Othello instance: random-init probe grid and all four
    observation floors."""
    pdir = layout.baselines_dir("othello", inst) / "probes"
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    rules = oc.rules_of(inst)
    data = _probe_games(s["oth_probe_games"], inst)
    grid = oa.fit_probe_grid(rand, data, cache_dir=pdir, log=None)
    paths = oc.build(only=(LARGE["oth_split"],), log=lambda s_: None, instance=inst)
    data_large = oc.probe_data(paths[LARGE["oth_split"]], **rules)   # labels cached beside the corpus
    blk = {k: {} for k, _, _ in OBS_KINDS}
    blk["random_init"] = {}
    for fam in ("linear", "mlp"):
        for kind, align, epochs in OBS_KINDS:
            is_large = epochs is not None
            _, st = oa.observation_probes(data_large if is_large else data, family=fam,
                                          seed=BASELINE_SEED, cache_dir=pdir, log=None,
                                          epochs=epochs, align=align)
            blk[kind][fam] = {**_pack(st), **({"n_seq": int(len(data_large.tokens)), "epochs": epochs}
                                               if is_large else {})}
        pts = [x for x in grid.stats if x["target"] == "mine"
               and x["family"] == fam and x["split"] == "sequence"]
        blk["random_init"][fam] = _agg([_pack(x) for x in pts])
        print(f"    {arch}/mine/{fam}: obs {blk['observation'][fam]['skill']:+.4f}"
              f"  obs_large {blk['observation_large'][fam]['skill']:+.4f}"
              f"  obs_right_large {blk['observation_right_large'][fam]['skill']:+.4f}"
              f"  random-init {blk['random_init'][fam]['skill']:+.4f}", flush=True)
    del rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"span": span, "bases": {"mine/theirs": blk}}


def score_baselines_arch(runs, env, inst, arch, model_config, s) -> dict:
    """Both floors for one (environment, instance, architecture)."""
    if env == "othello":
        return _othello_floors(inst, arch, model_config, s)
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    del rand
    block = {"span": span, "bases": score_baseline_bases(inst, arch, model_config, rw_bases_for(inst, s), s)}
    block["bases"].update(score_baseline_targets(inst, arch, model_config,
                                                 floor_targets_for(runs, env, inst, arch, s), s))
    return block


def score_all_baselines(runs, s, dry_run=False) -> list:
    """Fill in every floor a ``baselines.json`` lacks, for every (env, instance, arch) among
    ``runs``. ``dry_run`` only reports what would be fitted. Returns the to-do list."""
    todo_all = []
    instances = {}
    for r in runs:
        spec = instances.setdefault((r["env"], r["instance"]), {})
        if r["arch"] not in spec:
            _m, info = load_checkpoint(r["dir"] / "best_model.pt", device="cpu")
            spec[info.arch] = info.model_config
            del _m

    for (env, inst), archs in instances.items():
        bp = layout.baselines_dir(env, inst) / "baselines.json"
        prev = json.loads(bp.read_text()) if bp.exists() else {}
        fresh = prev.get("baseline_version") == BASELINE_VERSION
        have = set(prev.get("archs", {})) if fresh else set()
        todo = [a for a in archs if a not in have]
        todo_targets, todo_bases = {}, {}
        for a in have:
            got = prev["archs"][a]["bases"]
            ts = [t for t in floor_targets_for(runs, env, inst, a, s) if t not in got]
            bs = [b for b in rw_bases_for(inst, s) if b not in got] if env == "rayworld" else []
            if ts:
                todo_targets[a] = ts
            if bs:
                todo_bases[a] = bs
        label = f"{env}/{inst}"
        if not todo and not todo_targets and not todo_bases:
            print(f"skip  baselines {label}  (archs {sorted(have)})")
            continue
        todo_all.append({"env": env, "instance": inst, "archs": todo, "targets": todo_targets,
                         "bases": todo_bases})
        if dry_run:
            print(f"WOULD fit baselines {label}: archs {todo} extra targets {todo_targets} bases {todo_bases}")
            continue
        t0 = time.time()
        print(f"\n=== baselines for {label}: archs {todo} extra targets {todo_targets} "
              f"bases {todo_bases} ===", flush=True)
        out = prev if fresh else {"instance": inst, "env": env, "seed": BASELINE_SEED, "archs": {}}
        for arch in todo:
            out["archs"][arch] = score_baselines_arch(runs, env, inst, arch, archs[arch], s)
        added = 0
        for arch, bs in todo_bases.items():
            new = score_baseline_bases(inst, arch, archs[arch], bs, s)
            out["archs"][arch]["bases"].update(new)
            added += len(new)
        for arch, ts in todo_targets.items():
            new = score_baseline_targets(inst, arch, archs[arch], ts, s)
            out["archs"][arch]["bases"].update(new)
            added += len(new)
        if not todo and not added:
            print(f"    nothing new for {label} (categorical floor probes not fitted yet)")
            continue
        out |= {"baseline_version": BASELINE_VERSION, "minutes": round((time.time() - t0) / 60, 1)}
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(out, indent=1, default=float))
        print(f"    wrote runs/_baselines/{label}/baselines.json  [{out['minutes']} min]", flush=True)
    print("\nall baselines present")
    return todo_all
