"""The 20M-sequence training corpus of a Rayworld instance, and the instance registry.

``python scripts/build_rayworld_corpus.py --instance <inst>`` generates 40 shards, strips each into
``obs.f32`` + ``meta.h5``, and ``verify()`` checks seeds against every other split's range."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FULL_SHARD_N, FULL_N_SHARDS = 500_000, 40   # shard size below the 1M retry step of simulate_with_retry
SEED_STRIDE = 500_000_000            # shard k starts at base_seed + k * SEED_STRIDE
FRAMES = 40
# the current corpus size and kept rays; rebound by use_instance()
SHARD_N, N_SHARDS = FULL_SHARD_N, FULL_N_SHARDS
N_TOTAL = SHARD_N * N_SHARDS                 # 20,000,000
OBS_RES = 128
VAL_N = N_TOTAL // 10
TRAIN_N = N_TOTAL - VAL_N

# ``scripts/generate_dataset.py`` flags of each instance's simulator
_COMMON_FLAGS = ["--n-objects", "2", "--frames", str(FRAMES),
                 "--boundary", "open", "--fixed-reflectivities", "--always-in-frustum"]
_NO_NOISE = ["--position-noise", "0.0", "--obs-noise-std", "0.0"]
_RAYS_128 = ["--obs-res", "128"]
# the N-ray family: radius 1.0, N + 2 rays cast and the wall rays dropped
_RAYS_8 = ["--obs-res", "10", "--drop-edge-rays", "--radius", "1.0",
           "--max-edit-attempts", "2000"]
_RAYS_5 = ["--obs-res", "7", "--drop-edge-rays", "--radius", "1.0", "--max-edit-attempts", "2000"]
_RAYS_16 = ["--obs-res", "18", "--drop-edge-rays", "--radius", "1.0", "--max-edit-attempts", "2000"]
_RAYS_128R = ["--obs-res", "130", "--drop-edge-rays", "--radius", "1.0", "--max-edit-attempts", "2000"]
_BLINK = ["--blink-prob", "0.05", "--blink-mean", "7", "--blink-max", "12", "--blink-warmup", "3"]
_SMOOTH = ["--soft-shading", "power", "--soft-profile-power", "2.0"]
_OBS5 = ["--n-observers", "5", "--region", "circle"]

# ── the instance registry ─────────────────────────────────────────────────────

INSTANCES = {
    # 128 rays, disc radius 0.5
    "standard": {"base_seed": 30_000_000_000, "obs_dim": 128,
                 "sim_flags": _COMMON_FLAGS + _RAYS_128 + _NO_NOISE},
    # standard with blackouts (blink.py)
    "blink": {"base_seed": 110_000_000_000, "obs_dim": 128,
              "sim_flags": _COMMON_FLAGS + _RAYS_128 + _NO_NOISE + _BLINK},
    # standard with the smooth power-dome disc profile (soft_render.py)
    "smooth": {"base_seed": 200_000_000_000, "obs_dim": 128,
               "sim_flags": _COMMON_FLAGS + _RAYS_128 + _NO_NOISE + _SMOOTH},
    "128-ray": {"base_seed": 300_000_000_000, "obs_dim": 128,
                "sim_flags": _COMMON_FLAGS + _RAYS_128R + _NO_NOISE},
    "16-ray": {"base_seed": 270_000_000_000, "obs_dim": 16,
               "sim_flags": _COMMON_FLAGS + _RAYS_16 + _NO_NOISE},
    "8-ray": {"base_seed": 60_000_000_000, "obs_dim": 8,
              "sim_flags": _COMMON_FLAGS + _RAYS_8 + _NO_NOISE},
    "5-ray": {"base_seed": 160_000_000_000, "obs_dim": 5,
              "sim_flags": _COMMON_FLAGS + _RAYS_5 + _NO_NOISE},
    # 8-ray seen by five observers around a circular arena (observers.py): 5 x 8 rays
    "obs5": {"base_seed": 230_000_000_000, "obs_dim": 40,
             "sim_flags": _COMMON_FLAGS + _RAYS_8 + _NO_NOISE + _OBS5},
}

# seed range [lo, hi) of every split of every instance ("eval suite": the eval and edits splits)
_B = 1_000_000_000
SEED_RANGES = {
    "standard": {"train": (30 * _B, 50 * _B), "eval suite": (52 * _B, 52_400_000_000),
                 "probe": (950 * _B, 951 * _B), "probe_large": (970 * _B, 971 * _B)},
    "8-ray": {"train": (60 * _B, 80 * _B), "eval suite": (85 * _B, 85_400_000_000),
              "probe": (980 * _B, 981 * _B), "probe_large": (990 * _B, 991 * _B)},
    "blink": {"train": (110 * _B, 130 * _B), "eval suite": (135 * _B, 135_400_000_000),
              "probe": (1000 * _B, 1001 * _B), "probe_large": (1010 * _B, 1011 * _B)},
    "5-ray": {"train": (160 * _B, 180 * _B), "eval suite": (185 * _B, 185_400_000_000),
              "probe": (1020 * _B, 1021 * _B), "probe_large": (1030 * _B, 1031 * _B)},
    "smooth": {"train": (200 * _B, 220 * _B), "eval suite": (225 * _B, 225_400_000_000),
               "probe": (1040 * _B, 1041 * _B), "probe_large": (1050 * _B, 1051 * _B)},
    "obs5": {"train": (230 * _B, 250 * _B), "eval suite": (255 * _B, 255_400_000_000),
             "probe": (1060 * _B, 1061 * _B), "probe_large": (1070 * _B, 1071 * _B)},
    "16-ray": {"train": (270 * _B, 290 * _B), "eval suite": (295 * _B, 295_400_000_000),
               "probe": (1080 * _B, 1081 * _B), "probe_large": (1090 * _B, 1091 * _B)},
    "128-ray": {"train": (300 * _B, 320 * _B), "eval suite": (325 * _B, 325_400_000_000),
                "probe": (1100 * _B, 1101 * _B), "probe_large": (1110 * _B, 1111 * _B)},
}
# seed ranges of data outside this release; every corpus is still checked against them
RESERVED = [(0, 120_000), (3_000_000, 3_950_000), (10_000_000, 19_800_000_000),
            (900 * _B, 901 * _B), (960 * _B, 961 * _B)]


def forbidden(inst: str) -> list[tuple[int, int, str]]:
    """Every seed range an instance's training corpus must avoid: all other splits."""
    out = [(lo, hi, "reserved") for lo, hi in RESERVED]
    for name, splits in SEED_RANGES.items():
        for split, (lo, hi) in splits.items():
            if not (name == inst and split == "train"):
                out.append((lo, hi, f"{name} {split}"))
    return sorted(out)


for _inst, _spec in INSTANCES.items():
    assert _spec["base_seed"] == SEED_RANGES[_inst]["train"][0], _inst
    _spec["forbidden"] = forbidden(_inst)


def train_dir(inst: str) -> Path:
    from pim.environments.layout import train_dir as _train_dir

    return _train_dir("rayworld", inst)


def _spec(inst: str) -> dict:
    if inst not in INSTANCES:
        raise KeyError(f"unknown instance {inst!r}; registered: {sorted(INSTANCES)}")
    return INSTANCES[inst]


# the module-level instance every function below reads; switch it with use_instance()
INSTANCE = "standard"
OUT = train_dir(INSTANCE)
BASE_SEED = INSTANCES[INSTANCE]["base_seed"]
SIM_FLAGS = INSTANCES[INSTANCE]["sim_flags"]
FORBIDDEN = INSTANCES[INSTANCE]["forbidden"]


def use_instance(inst: str, *, n_shards: int | None = None, shard_n: int | None = None) -> None:
    """Point every function in this module at one instance's paths, seeds and flags, and at a corpus
    of the first ``n_shards`` shards of ``shard_n`` sequences each (None: the full 40 x 500,000).
    Shard k always starts at seed ``base_seed + k * SEED_STRIDE``, so a small corpus is a prefix."""
    global INSTANCE, OUT, BASE_SEED, SIM_FLAGS, FORBIDDEN, OBS_RES
    global SHARD_N, N_SHARDS, N_TOTAL, VAL_N, TRAIN_N
    spec = _spec(inst)
    n_shards = FULL_N_SHARDS if n_shards is None else int(n_shards)
    shard_n = FULL_SHARD_N if shard_n is None else int(shard_n)
    if not (1 <= n_shards <= FULL_N_SHARDS and 1 <= shard_n <= FULL_SHARD_N):
        raise ValueError(f"n_shards must be in [1, {FULL_N_SHARDS}] and shard_n in [1, {FULL_SHARD_N:,}], "
                         f"got {n_shards} and {shard_n:,}")
    SHARD_N, N_SHARDS = shard_n, n_shards
    N_TOTAL = SHARD_N * N_SHARDS
    VAL_N = N_TOTAL // 10
    TRAIN_N = N_TOTAL - VAL_N
    INSTANCE = inst
    OBS_RES = int(spec.get("obs_dim", 128))
    OUT = train_dir(inst)
    BASE_SEED = spec["base_seed"]
    SIM_FLAGS = spec["sim_flags"]
    FORBIDDEN = spec["forbidden"]


def shard_seed(k: int) -> int:
    return BASE_SEED + k * SEED_STRIDE


def obs_path() -> Path:
    return OUT / "obs.f32"


def corpus_n() -> int:
    """Sequences in the current instance's corpus: ``n`` of ``train/corpus.json`` once the corpus
    is built, else ``N_TOTAL``."""
    f = OUT / "corpus.json"
    return int(json.loads(f.read_text())["n"]) if f.exists() else N_TOTAL


def _shard_dir(k: int) -> Path:
    return OUT / f"_shard_{k:03d}"


def generate_shard(k: int, workers: int = 8, log=print) -> None:
    """One shard through the shared generator. Idempotent: a stripped shard is skipped."""
    if (OUT / f"_done_{k:03d}").exists():
        log(f"  shard {k:03d} already done")
        return
    d = _shard_dir(k)
    if d.exists():
        shutil.rmtree(d)
    cmd = [sys.executable, str(REPO / "scripts" / "generate_dataset.py"), str(d),
           "--n-train", str(SHARD_N), "--n-val", "100", "--n-test", "100", "--n-edits", "100",
           *SIM_FLAGS, "--seed", str(shard_seed(k)), "--n-workers", str(workers),
           "--compression-level", "0"]
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO),
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def strip_shard(k: int, log=print) -> None:
    """Append a shard's observations to the flat memmap and its metadata to meta.h5, then
    delete the shard (``obs_depth`` and ``obs_id`` are not kept)."""
    if (OUT / f"_done_{k:03d}").exists():
        return
    src = _shard_dir(k) / "train.h5"
    lo = k * SHARD_N
    obs = np.memmap(obs_path(), dtype=np.float32, mode="r+",
                    shape=(N_TOTAL, FRAMES, OBS_RES))
    with h5py.File(src, "r") as f:
        n = f["obs_intensity"].shape[0]
        assert n == SHARD_N, f"shard {k} has {n} != {SHARD_N}"
        for i in range(0, n, 50_000):                       # chunked so RAM stays flat
            j = min(i + 50_000, n)
            obs[lo + i: lo + j] = f["obs_intensity"][i:j]
        meta = {kk: f[kk][:] for kk in ("positions", "velocities", "seeds",
                                        "reflectivities", "radii")}
    obs.flush()
    del obs
    with h5py.File(OUT / "meta.h5", "a") as g:
        for kk, v in meta.items():
            if kk not in g:
                g.create_dataset(kk, shape=(N_TOTAL, *v.shape[1:]), dtype=v.dtype,
                                 chunks=(min(4096, N_TOTAL), *v.shape[1:]))
            g[kk][lo: lo + SHARD_N] = v
    shutil.rmtree(_shard_dir(k))
    (OUT / f"_done_{k:03d}").touch()
    log(f"  shard {k:03d} stripped  (seeds {shard_seed(k):,}..{shard_seed(k) + SHARD_N:,})")


def verify(log=print) -> dict:
    """No duplicate seeds, no seed in another split's range, and the obs file fully written."""
    n = corpus_n()
    with h5py.File(OUT / "meta.h5", "r") as g:
        s = g["seeds"][:]
    assert len(s) == n, f"{len(s):,} seeds, expected {n:,}"
    dups = len(s) - len(np.unique(s))
    assert dups == 0, f"{dups:,} duplicate seeds — a retry collided, the corpus is not i.i.d."
    for lo, hi, name in FORBIDDEN:
        n_bad = int(((s >= lo) & (s < hi)).sum())
        assert n_bad == 0, f"{n_bad:,} seeds collide with {name} [{lo:,},{hi:,})"
    sz = obs_path().stat().st_size
    want = n * FRAMES * OBS_RES * 4
    assert sz == want, f"obs.f32 is {sz:,} bytes, expected {want:,}"
    out = {"n": int(n), "duplicates": 0, "seed_min": int(s.min()), "seed_max": int(s.max()),
           "obs_bytes": sz, "disjoint_from": [name for _, _, name in FORBIDDEN]}
    log(f"  VERIFIED {n:,} sequences, seeds {s.min():,}..{s.max():,}, "
        f"0 duplicates, disjoint from every other split, obs.f32 {sz / 1e9:.1f} GB")
    return out


def open_obs(mode: str = "r") -> np.memmap:
    """The (n, FRAMES, OBS_RES) frame memmap, n = ``corpus_n()``."""
    return np.memmap(obs_path(), dtype=np.float32, mode=mode,
                     shape=(corpus_n(), FRAMES, OBS_RES))

