#!/usr/bin/env python
"""Add the `prediction` block (held-out predictive loss) to scored runs' scores.json.

The block is pim.environments.prediction.score_run: the run's training objective on its instance's
held-out split, per sequence. Only that block is written; a run already at PRED_VERSION is skipped.

    python scripts/score_prediction.py [--runs rayworld/8-ray othello/standard] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.prediction import PRED_VERSION, score_run  # noqa: E402


def scored_runs() -> list[str]:
    """Run ids of every scored run under runs/<env>/ (baselines excluded)."""
    root = REPO / "runs"
    return sorted(str(p.parent.relative_to(root)) for env in ("othello", "rayworld")
                  for p in (root / env).glob("*/scores.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run ids (default: every scored run)")
    ap.add_argument("--force", action="store_true", help="rescore runs already at PRED_VERSION")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    for run in a.runs or scored_runs():
        sp = REPO / "runs" / run / "scores.json"
        s = json.loads(sp.read_text())
        if s.get("prediction", {}).get("version") == PRED_VERSION and not a.force:
            continue
        t0 = time.time()
        block = score_run(sp.parent, device=a.device)
        s = json.loads(sp.read_text())                   # re-read so the write window is short
        s["prediction"] = block
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(s, indent=1, default=float) + "\n")
        os.replace(tmp, sp)
        line = "  ".join(f"{k} {v['loss']:.5f}" for k, v in block["readings"].items())
        print(f"  {run}: {line}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
