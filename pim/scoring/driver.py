"""Score every run that needs it: a run without ``scores.json`` (or at another eval version) is
scored in full; a current run that lacks a block the settings ask for gets only that block added.
Nothing already present is recomputed.
"""
import json
import os
import time
from pathlib import Path

import torch

from pim.models import load_checkpoint, n_points
from pim.scoring.blocks import rayworld_blocks
from pim.scoring.othello import score_othello
from pim.scoring.rayworld import score_rayworld, score_rayworld_tokens
from pim.scoring.runs import DEV

EVAL_VERSION = "1.0"
EVAL_VERSION_BY_ENV = {"othello": "1.0", "rayworld": "1.0"}
SCORED_ENVS = ("rayworld", "othello")


def eval_version(r) -> str:
    """The eval version a run's ``scores.json`` must carry to count as current."""
    return EVAL_VERSION_BY_ENV.get(r["env"], EVAL_VERSION)


def scorer_for(r) -> tuple:
    """By what the model emits and where it lives: Othello (legal-set Edit Index), a Rayworld
    token model (frame-set), or a Rayworld frame model (ray-zone)."""
    if r["env"] == "othello":
        return score_othello, "othello"
    if r["arch"].endswith("_tokens"):
        return score_rayworld_tokens, "rayworld/tokens"
    return score_rayworld, "rayworld"


def _write_scores(sp: Path, scores: dict) -> None:
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scores, indent=1, default=float))
    os.replace(tmp, sp)


def missing_blocks(r, prev, s) -> list:
    """Rayworld probe-target blocks the settings ask of a run that its ``scores.json`` lacks
    (an Othello run has one block, always present once scored)."""
    if r["env"] != "rayworld":
        return []
    return [k for k, _, _ in rayworld_blocks(r["id"], s) if k not in prev.get("bases", {})]


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _complete(r, prev, s, dry_run) -> dict | None:
    """Add what a current ``scores.json`` lacks; None when nothing is missing."""
    sp = r["dir"] / "scores.json"
    missing = missing_blocks(r, prev, s)
    if not missing:
        print(f"skip  {r['id']}  (scored at {prev['eval_version']})")
        return None
    todo = {"run": r["id"], "action": "add", "blocks": missing}
    if dry_run:
        print(f"WOULD add to {r['id']}: blocks {missing}")
        return todo
    scorer, _ = scorer_for(r)
    model, _ = load_checkpoint(r["dir"] / "best_model.pt", device=DEV)
    print(f"\n=== adding blocks {missing} to {r['id']} ===", flush=True)
    add = scorer(model, r["dir"], s, only=missing)
    if add["bases"]:
        prev.setdefault("bases", {}).update(add["bases"])
        _write_scores(sp, prev)
        print(f"    wrote runs/{r['id']}/scores.json  +{sorted(add['bases'])}", flush=True)
    else:
        print("    nothing added (categorical probes not fitted yet)", flush=True)
    del model
    _empty_cache()
    return todo


def score_all(runs, s, eval_version=eval_version, dry_run=False) -> list:
    """Score every run in ``runs`` whose ``scores.json`` is missing, at another eval version, or
    lacks a block the settings ask for. ``dry_run`` only reports what would be done. Returns the
    to-do list."""
    todo_all = []
    for r in runs:
        sp = r["dir"] / "scores.json"
        if sp.exists():
            prev = json.loads(sp.read_text())
            if prev.get("eval_version") == eval_version(r):
                todo = _complete(r, prev, s, dry_run)
                if todo:
                    todo_all.append(todo)
                continue
            print(f"stale {r['id']}  ({prev.get('eval_version')} -> {eval_version(r)})")
        if r["env"] not in SCORED_ENVS:
            print(f"skip  {r['id']}  (env {r['env']!r} has no scorer)")
            continue
        todo_all.append({"run": r["id"], "action": "score"})
        if dry_run:
            print(f"WOULD score {r['id']}  ({'stale' if sp.exists() else 'unscored'})")
            continue
        t0 = time.time()
        scorer, label = scorer_for(r)
        print(f"\n=== scoring {r['id']}  ({r['arch']} on {label}) ===", flush=True)
        model, info = load_checkpoint(r["dir"] / "best_model.pt", device=DEV)
        scores = scorer(model, r["dir"], s)
        scores = {"run": r["id"], "arch": info.arch, "env": r["env"], "instance": r["instance"],
                  "val_loss": info.val_loss, "n_points": n_points(model),
                  "eval_version": eval_version(r), "minutes": round((time.time() - t0) / 60, 1),
                  **scores}
        _write_scores(sp, scores)
        print(f"    wrote runs/{r['id']}/scores.json  [{scores['minutes']} min]", flush=True)
        del model
        _empty_cache()
    print("\nall runs scored")
    return todo_all
