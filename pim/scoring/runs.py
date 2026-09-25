"""Which runs the scorer sees: every ``runs/<env>/<variant>`` holding a checkpoint."""
import json
from pathlib import Path

import torch

from pim.environments import layout

REPO = layout.REPO
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def training_complete(run_dir: Path, cfg: dict) -> bool:
    """True once the last step in ``metrics.jsonl`` reaches ``train.steps`` (or either is absent)."""
    target = int(cfg.get("train", {}).get("steps", 0) or 0)
    mp = run_dir / "metrics.jsonl"
    if not target or not mp.exists():
        return True
    last = 0
    for line in mp.read_text().splitlines():
        if line.strip():
            last = max(last, int(json.loads(line).get("step", 0)))
    return last >= target


def scan_runs(root: Path = REPO / "runs") -> list[dict]:
    """One row per run: id ``<env>/<variant>``, its directory, arch, env, instance, scored.

    Directories starting with ``_`` (``runs/_baselines``) are skipped, and so is a run still
    training: scoring its interim checkpoint would mark the finished run as already scored."""
    rows = []
    for env_dir in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_")):
        for run_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists() or not (run_dir / "best_model.pt").exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            run_id = f"{env_dir.name}/{run_dir.name}"
            if not training_complete(run_dir, cfg):
                print(f"skip  {run_id}  (still training)")
                continue
            rows.append({
                "id": run_id, "dir": run_dir,
                "arch": cfg.get("arch", "?"),
                "env": cfg.get("data", {}).get("env", env_dir.name),
                "instance": cfg.get("data", {}).get("instance", "?"),
                "n_params": cfg.get("n_params"),
                "scored": (run_dir / "scores.json").exists(),
            })
    return rows
