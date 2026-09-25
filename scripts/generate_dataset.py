#!/usr/bin/env python
"""Generate Rayworld sequences: one split of an instance, or a four-split suite into a directory.

With --role and a registered --instance, the simulator flags, seeds and size come from the
instance registry (pim.environments.rayworld.bigcorpus): the released split is regenerated exactly.

    python scripts/generate_dataset.py --instance 8-ray --role eval     # also edits, probe --size 120k|250k
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments import layout  # noqa: E402
from pim.environments.rayworld import bigcorpus as bc  # noqa: E402
from pim.environments.rayworld.config import SimConfig  # noqa: E402
from pim.environments.rayworld.dataset import DatasetConfig, generate_dataset  # noqa: E402
from pim.environments.rayworld.edits_dataset import (  # noqa: E402
    EditDatasetConfig,
    generate_edits_dataset,
)

# each split of a registered instance: (seed range in bc.SEED_RANGES, offset into it, size)
SPLITS = {"eval": ("eval suite", 200_000_000, 10_000),
          "edits": ("eval suite", 300_000_000, 10_000),
          "probe_120k": ("probe", 200, 120_000),
          "probe_250k": ("probe_large", 200, 250_000)}
EDITS_N = {"blink": 20_000}
EDITS_FLAGS = ["--edit-frame", "20", "--edit-always-in-frustum"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", nargs="?", default=None,
                   help="four-split mode: write train/val/test/edits .h5 here (must be empty)")

    g = p.add_argument_group("one split of an instance")
    g.add_argument("--role", choices=["eval", "edits", "probe"], default=None)
    g.add_argument("--instance", default=None, help="datasets/rayworld/<INSTANCE>")
    g.add_argument("--size", default="120k", choices=["120k", "250k"], help="probe corpus size")
    g.add_argument("--n", type=int, default=None, help="samples (default: the registry's)")

    g = p.add_argument_group("four-split sizes")
    g.add_argument("--n-train", type=int, default=100_000)
    g.add_argument("--n-val", type=int, default=10_000)
    g.add_argument("--n-test", type=int, default=10_000)
    g.add_argument("--n-edits", type=int, default=5_000)

    g = p.add_argument_group("simulator")
    g.add_argument("--n-objects", type=int, default=2)
    g.add_argument("--frames", type=int, default=40)
    g.add_argument("--obs-res", type=int, default=128, help="rays cast")
    g.add_argument("--drop-edge-rays", action="store_true",
                   help="drop the two frustum-wall rays (obs width obs-res - 2)")
    g.add_argument("--radius", type=float, default=0.5, help="disc radius")
    g.add_argument("--boundary", choices=["bounce", "open", "wrap"], default="open")
    g.add_argument("--position-noise", type=float, default=0.0, help="position noise std per step")
    g.add_argument("--obs-noise-std", type=float, default=0.0, help="observation noise std")
    g.add_argument("--fixed-reflectivities", action="store_true",
                   help="evenly spaced reflectivities")
    g.add_argument("--always-in-frustum", action="store_true",
                   help="reject trajectories that touch a frustum wall")
    g.add_argument("--blink-prob", type=float, default=0.0, help="blackout start probability")
    g.add_argument("--blink-mean", type=float, default=6.0, help="mean blackout length")
    g.add_argument("--blink-max", type=int, default=12, help="blackout length cap")
    g.add_argument("--blink-warmup", type=int, default=3, help="first frame a blackout can start")
    g.add_argument("--soft-shading", choices=["flat", "power"], default="flat",
                   help="disc profile: flat, or the smooth power dome")
    g.add_argument("--soft-profile-power", type=float, default=2.0, help="power-dome exponent")
    g.add_argument("--n-observers", type=int, default=1, help="observers on a ring")
    g.add_argument("--region", choices=["frustum", "circle"], default="frustum",
                   help="arena of the discs")

    g = p.add_argument_group("edits")
    g.add_argument("--edit-frame", type=int, default=-1, help="frame of the edit (-1: frames // 2)")
    g.add_argument("--edit-always-in-frustum", action="store_true",
                   help="reject edits that leave the frustum")
    g.add_argument("--max-edit-attempts", type=int, default=50)

    g = p.add_argument_group("seeds and storage")
    g.add_argument("--seed", type=int, default=None,
                   help="first seed (default: the registry's; four-split mode: 0)")
    g.add_argument("--n-workers", type=int, default=4, help="processes (0: single process)")
    g.add_argument("--write-batch", type=int, default=512)
    g.add_argument("--compression-level", type=int, default=4)
    return p


def _split_name(a) -> str:
    return f"probe_{a.size}" if a.role == "probe" else a.role


def parse_args(argv=None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else list(argv)
    p = _parser()
    a = p.parse_args(argv)
    if a.role and a.instance in bc.INSTANCES:
        # the registry's flags first, so any flag given on the command line overrides them
        a = p.parse_args(bc.INSTANCES[a.instance]["sim_flags"]
                         + (EDITS_FLAGS if a.role == "edits" else []) + argv)
        rng, offset, n = SPLITS[_split_name(a)]
        if a.seed is None:
            a.seed = bc.SEED_RANGES[a.instance][rng][0] + offset
        if a.n is None:
            a.n = EDITS_N.get(a.instance, n) if a.role == "edits" else n
    return a


def _sim(a) -> SimConfig:
    return SimConfig(n_objects=a.n_objects, n_frames=a.frames, obs_res=a.obs_res,
                     drop_edge_rays=a.drop_edge_rays, radius=a.radius, boundary=a.boundary,
                     position_noise_std=a.position_noise, obs_noise_std=a.obs_noise_std,
                     fixed_reflectivities=a.fixed_reflectivities,
                     always_in_frustum=a.always_in_frustum, blink_prob=a.blink_prob,
                     blink_mean=a.blink_mean, blink_max=a.blink_max, blink_warmup=a.blink_warmup,
                     soft_shading=a.soft_shading, soft_profile_power=a.soft_profile_power,
                     n_observers=a.n_observers, region=a.region)


def _storage(a) -> dict:
    return dict(n_workers=a.n_workers, write_batch=a.write_batch,
                compression_level=a.compression_level)


def _edits_config(a, sim, n, seed) -> tuple[EditDatasetConfig, dict]:
    frame = a.edit_frame if a.edit_frame >= 0 else a.frames // 2
    cfg = EditDatasetConfig(n_samples=n, sim=sim, base_seed=seed, edit_frame=frame,
                            edit_always_in_frustum=a.edit_always_in_frustum,
                            max_edit_attempts=a.max_edit_attempts, **_storage(a))
    return cfg, {"n_samples": n, "base_seed": seed, "edit_frame": frame,
                 "edit_always_in_frustum": a.edit_always_in_frustum,
                 "max_edit_attempts": a.max_edit_attempts}


def _generation(a) -> dict:
    return {"n_workers": a.n_workers, "write_batch": a.write_batch, "compression": "gzip",
            "compression_level": a.compression_level}


def one_split(a) -> None:
    """Write one split into the instance's directory, with its .json manifest beside it."""
    if not a.instance or a.n is None or a.seed is None:
        raise SystemExit("--role needs --instance (registered, or with --n and --seed)")
    inst = a.instance
    if a.role == "probe":
        target, manifest = (layout.probe_file("rayworld", inst, a.size),
                            layout.probe_manifest("rayworld", inst, a.size))
    elif a.role == "eval":
        target, manifest = layout.eval_file("rayworld", inst), layout.eval_manifest("rayworld", inst)
    else:
        target, manifest = layout.edits_file("rayworld", inst), layout.edits_manifest("rayworld", inst)
    if target.exists():
        raise SystemExit(f"{target.relative_to(REPO)} exists; refusing to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    sim = _sim(a)
    t0 = time.perf_counter()
    print(f"{inst} {a.role} -> {target.relative_to(REPO)}  (n={a.n:,}, seed {a.seed:,})", flush=True)
    if a.role == "edits":
        cfg, split = _edits_config(a, sim, a.n, a.seed)
        meta = generate_edits_dataset(cfg, target)
    else:
        cfg = DatasetConfig(n_samples=a.n, sim=sim, base_seed=a.seed, **_storage(a))
        meta = generate_dataset(cfg, target)
        split = {"n_samples": a.n, "base_seed": a.seed}
        if a.role == "probe":
            split["size"] = a.size
            split["holdout"] = ("internal: pim.environments.rayworld.arms.fit_probes draws a "
                                "seeded 80/20 split BY SEQUENCE inside this file (seed 0)")
    manifest.write_text(json.dumps({
        "role": a.role, "file": target.name, "sim": dataclasses.asdict(sim),
        "splits": {a.role: split}, "generation": _generation(a), "schema": meta.get("schema"),
    }, indent=2) + "\n")
    print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)


def four_splits(a) -> None:
    """Train, val, test and edits splits on consecutive seed ranges starting at --seed."""
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    sim, seed = _sim(a), a.seed or 0
    seeds = {"train": seed, "val": seed + a.n_train, "test": seed + a.n_train + a.n_val,
             "edits": seed + a.n_train + a.n_val + a.n_test}
    sizes = {"train": a.n_train, "val": a.n_val, "test": a.n_test, "edits": a.n_edits}
    splits, schema = {}, None
    for name in ("train", "val", "test"):
        meta = generate_dataset(DatasetConfig(n_samples=sizes[name], sim=sim,
                                              base_seed=seeds[name], **_storage(a)),
                                out / f"{name}.h5")
        schema = schema or meta["schema"]
        splits[name] = {"n_samples": sizes[name], "base_seed": seeds[name]}
    cfg, splits["edits"] = _edits_config(a, sim, a.n_edits, seeds["edits"])
    generate_edits_dataset(cfg, out / "edits.h5")
    (out / "dataset.json").write_text(json.dumps({
        "sim": dataclasses.asdict(sim), "splits": splits, "generation": _generation(a),
        "schema": schema}, indent=2) + "\n")


def main() -> None:
    a = parse_args()
    if a.role:
        one_split(a)
    elif a.output_dir:
        four_splits(a)
    else:
        raise SystemExit("give an output directory, or --role with --instance")


if __name__ == "__main__":
    main()
