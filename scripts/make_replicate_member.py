#!/usr/bin/env python
"""Lay out a run's own checkpoint at the replicates' step budget as its seed-0 member, runs/<run>__seed0/.

ckpt/step_<step>.pt is copied to best_model.pt with the validation loss nearest that step, and
config.json gains a `replicate` block, so seed 0 pools with seeds 1 and 2 at a matched budget.

    python scripts/make_replicate_member.py --run rayworld/8-ray --step 512000
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id, e.g. rayworld/8-ray")
    ap.add_argument("--step", type=int, default=512_000, help="a saved checkpoint step")
    ap.add_argument("--force", action="store_true", help="overwrite an existing member")
    a = ap.parse_args()

    src = REPO / "runs" / a.run
    cfg = json.loads((src / "config.json").read_text())
    seed = int(cfg["train"].get("seed", 0))
    dst = src.parent / f"{src.name}__seed{seed}"
    if (dst / "best_model.pt").exists() and not a.force:
        sys.exit(f"{dst.relative_to(REPO)} exists (--force to overwrite)")
    ckpt = src / "ckpt" / f"step_{a.step:09d}.pt"
    if not ckpt.exists():
        saved = sorted(int(p.stem.split("_")[1]) for p in (src / "ckpt").glob("step_*.pt"))
        sys.exit(f"no checkpoint at step {a.step:,}; saved steps: {saved}")
    vals = {int(r["step"]): float(r["val_loss"])
            for r in map(json.loads, (src / "metrics.jsonl").read_text().splitlines())
            if r.get("val_loss") is not None}
    near = min(vals, key=lambda v: abs(v - a.step))

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck["val_loss"], ck["val_loss_step"], ck["arch"] = vals[near], near, cfg["arch"]
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst / "best_model.pt")
    member = dict(cfg)
    member["replicate"] = {"of": a.run, "seed": seed, "steps": a.step, "checkpoint": True,
                           "source": str(ckpt.relative_to(REPO)), "val_loss_from_step": near,
                           "note": "the main run's own checkpoint at the replicates' step budget"}
    (dst / "config.json").write_text(json.dumps(member, indent=1) + "\n")
    print(f"{dst.relative_to(REPO)}: step {a.step:,}, val {vals[near]:.6f} (from step {near:,})")


if __name__ == "__main__":
    main()
