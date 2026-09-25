#!/usr/bin/env python
"""Build a Rayworld instance's 20M-sequence training corpus in datasets/rayworld/<instance>/train/.

40 seed-strided shards (scripts/generate_dataset.py) packed into obs.f32 + meta.h5, every seed checked
against the other splits' ranges, corpus.json written; resumable. --shards / --shard-n: a smaller prefix.

    python scripts/build_rayworld_corpus.py --instance 8-ray [--shards 2 --shard-n 4000]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.rayworld import bigcorpus as bc  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="standard", choices=sorted(bc.INSTANCES))
    p.add_argument("--concurrent", type=int, default=4, help="shards generated at once")
    p.add_argument("--workers", type=int, default=8, help="processes per shard")
    p.add_argument("--shards", type=int, default=None,
                   help=f"shards to build (default {bc.N_SHARDS}); fewer for quick checks")
    p.add_argument("--shard-n", type=int, default=None,
                   help=f"sequences per shard (default {bc.SHARD_N:,}); fewer for quick checks "
                        "(a frame model needs 6,144 sequences in all)")
    a = p.parse_args(argv)

    size = {k: v for k, v in (("n_shards", a.shards), ("shard_n", a.shard_n)) if v is not None}
    bc.use_instance(a.instance, **size)
    print(f"{bc.INSTANCE}: corpus -> {bc.OUT.relative_to(REPO)}, base seed {bc.BASE_SEED:,}",
          flush=True)
    bc.OUT.mkdir(parents=True, exist_ok=True)
    if not bc.obs_path().exists():                  # sparse allocation, filled shard by shard
        np.memmap(bc.obs_path(), dtype=np.float32, mode="w+",
                  shape=(bc.N_TOTAL, bc.FRAMES, bc.OBS_RES)).flush()
    todo = [k for k in range(bc.N_SHARDS) if not (bc.OUT / f"_done_{k:03d}").exists()]
    print(f"{len(todo)} of {bc.N_SHARDS} shards to generate", flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.concurrent) as ex:
        futs = {ex.submit(bc.generate_shard, k, a.workers): k for k in todo}
        for fut in cf.as_completed(futs):
            fut.result()
            bc.strip_shard(futs[fut])
    info = bc.verify()
    (bc.OUT / "corpus.json").write_text(json.dumps(
        {**info, "shard_n": bc.SHARD_N, "n_shards": bc.N_SHARDS, "base_seed": bc.BASE_SEED,
         "seed_stride": bc.SEED_STRIDE, "sim_flags": bc.SIM_FLAGS,
         "train_n": bc.TRAIN_N, "val_n": bc.VAL_N,
         "instance": bc.INSTANCE, "n_frames": bc.FRAMES, "obs_dim": bc.OBS_RES}, indent=1) + "\n")
    print("corpus complete", flush=True)


if __name__ == "__main__":
    main()
