"""Dataset generation: (observation sequence, simulator state) pairs written to HDF5.

Arrays are padded to ``max_objects`` (``n_objects`` holds the true count); each file's
``config_json`` attribute stores the full configuration and a field-by-field schema."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .blink import MARK_VALUE, blink_enabled, blink_schedule
from .config import SimConfig, check_supported, obs_dim
from .renderer import render_scene
from .sim import Scene, compute_visibility, simulate
from .soft_render import soft_enabled


@dataclass
class DatasetConfig:
    """Configuration of one dataset generation run."""

    n_samples: int = 100_000
    sim: SimConfig = field(default_factory=SimConfig)
    base_seed: int = 0
    n_workers: int = 4  # 0 runs in a single process
    write_batch: int = 512  # samples buffered in RAM before each HDF5 write
    hdf5_chunk: int = 64  # HDF5 chunk size along the sample axis
    compression: str = "gzip"
    compression_level: int = 4


def simulate_with_retry(base_cfg: SimConfig, seed: int) -> tuple[Scene, SimConfig]:
    """``simulate`` at ``seed``; if it gives up, retry at ``seed + attempt * 1_000_000``
    (up to 10 attempts). The seed law every generator shares; ``bigcorpus``'s shard stride
    is sized to it."""
    cfg = dataclasses.replace(base_cfg, seed=int(seed))
    for attempt in range(10):
        try:
            if attempt:
                cfg = dataclasses.replace(cfg, seed=int(seed) + attempt * 1_000_000)
            return simulate(cfg), cfg
        except RuntimeError:
            if attempt == 9:
                raise
    raise AssertionError("unreachable")


def pack_sample(scene: Scene, cfg: SimConfig, max_obj: int) -> dict:
    """Render ``scene`` and pack every field under its HDF5 name, padded to ``max_obj`` objects.
    ``obs_clean`` is stored only under soft rendering, where ids do not determine it."""
    bvis = blink_schedule(cfg, scene.positions.shape[1])   # None unless blinking
    obs_depth, obs_id, obs_intensity = render_scene(scene, visible=bvis)
    obs_clean = None
    if soft_enabled(cfg):
        obs_clean = render_scene(
            dataclasses.replace(scene, config=dataclasses.replace(cfg, obs_noise_std=0.0)),
            visible=bvis,
        )[2].astype(np.float32)
    vis = compute_visibility(scene)  # (n_frames, n)
    n = scene.positions.shape[1]

    pos_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    vel_out = np.zeros((cfg.n_frames, max_obj, 2), dtype=np.float32)
    col_out = np.zeros((max_obj, 3), dtype=np.float32)
    radii_out = np.zeros((max_obj,), dtype=np.float32)
    refl_out = np.zeros((max_obj,), dtype=np.float32)
    vis_out = np.zeros((cfg.n_frames, max_obj), dtype=bool)

    pos_out[:, :n] = scene.positions.astype(np.float32)
    vel_out[:, :n] = scene.velocities.astype(np.float32)
    col_out[:n] = scene.colors.astype(np.float32)
    radii_out[:n] = scene.radii.astype(np.float32)
    refl_out[:n] = scene.reflectivities.astype(np.float32)
    vis_out[:, :n] = vis
    blink_out = None
    if bvis is not None:                      # padding counts as visible
        blink_out = np.ones((cfg.n_frames, max_obj), dtype=bool)
        blink_out[:, :n] = bvis

    return {
        "obs_intensity": obs_intensity.astype(np.float32),
        **({"obs_clean": obs_clean} if obs_clean is not None else {}),
        "obs_depth": obs_depth.astype(np.float32),
        "obs_id": obs_id.astype(np.int8),
        "is_visible": vis_out,
        **({"blink_visible": blink_out} if blink_out is not None else {}),
        "positions": pos_out,
        "velocities": vel_out,
        "colors": col_out,
        "radii": radii_out,
        "reflectivities": refl_out,
        "n_objects": np.uint8(n),
        "seeds": np.int64(cfg.seed),
    }


def _generate_one(args: tuple[int, SimConfig, int]) -> dict:
    """One sample; runs in worker processes (module level so it pickles)."""
    seed, base_cfg, max_obj = args
    scene, cfg = simulate_with_retry(base_cfg, seed)
    return pack_sample(scene, cfg, max_obj)


def common_layout(sim: SimConfig, max_obj: int) -> list[tuple[str, tuple, str, str]]:
    """``(name, tail shape, dtype, chunk kind)`` for every field ``pack_sample`` emits, in
    on-disk creation order. Chunk kind ``"sample"`` is ``min(C, N)`` samples per chunk,
    ``"scalar"`` is ``min(C * F, N)``."""
    F, R = sim.n_frames, obs_dim(sim)
    rows = [("obs_intensity", (F, R), "float32", "sample")]
    if soft_enabled(sim):
        rows.append(("obs_clean", (F, R), "float32", "sample"))
    rows += [
        ("obs_depth", (F, R), "float32", "sample"),
        ("obs_id", (F, R), "int8", "sample"),
        ("is_visible", (F, max_obj), "bool", "sample"),
        *([("blink_visible", (F, max_obj), "bool", "sample")] if blink_enabled(sim) else []),
        ("positions", (F, max_obj, 2), "float32", "sample"),
        ("velocities", (F, max_obj, 2), "float32", "sample"),
        ("colors", (max_obj, 3), "float32", "sample"),
        ("radii", (max_obj,), "float32", "sample"),
        ("reflectivities", (max_obj,), "float32", "sample"),
        ("n_objects", (), "uint8", "scalar"),
        ("seeds", (), "int64", "scalar"),
    ]
    return rows


def create_datasets(hf: h5py.File, layout, n_samples: int, n_frames: int, chunk: int,
                    compression: str, compression_level: int) -> None:
    N, F, C = n_samples, n_frames, chunk
    kw = dict(compression=compression, compression_opts=compression_level)
    for name, tail, dtype, kind in layout:
        c0 = min(C, N) if kind == "sample" else min(C * F, N)
        hf.create_dataset(name, (N, *tail), dtype=dtype, chunks=(c0, *tail), **kw)


def write_batch(hf: h5py.File, batch: list[dict], start: int) -> None:
    end = start + len(batch)
    for name in batch[0]:
        hf[name][start:end] = np.stack([s[name] for s in batch])


def generate_h5(h5_path: Path, *, n_samples: int, n_workers: int, write_batch_size: int,
                config_json: str, layout, n_frames: int, chunk: int, compression: str,
                compression_level: int, worker, args) -> None:
    """The generation loop: a worker pool over ``args``, batched writes into ``h5_path``.
    Refuses to overwrite. Shared by the plain and the edits generator."""
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists — refusing to overwrite.")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    chunksize = max(1, write_batch_size // max(1, n_workers))
    pool = mp.Pool(n_workers) if n_workers > 0 else None
    try:
        iterator = (pool.imap(worker, args, chunksize=chunksize) if pool is not None
                    else map(worker, args))
        written = 0
        batch: list[dict] = []
        with h5py.File(h5_path, "w") as hf:
            hf.attrs["config_json"] = config_json
            create_datasets(hf, layout, n_samples, n_frames, chunk, compression, compression_level)
            t0 = time.perf_counter()
            with tqdm(total=n_samples, unit="sample", dynamic_ncols=True,
                      desc=f"generating → {h5_path.name}") as pbar:
                for sample in iterator:
                    batch.append(sample)
                    pbar.update(1)
                    if len(batch) >= write_batch_size:
                        write_batch(hf, batch, written)
                        written += len(batch)
                        batch = []
                if batch:
                    write_batch(hf, batch, written)
                    written += len(batch)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    elapsed = time.perf_counter() - t0
    size_mb = h5_path.stat().st_size / 1e6
    print(f"  {n_samples:,} samples  |  {elapsed:.1f}s  ({n_samples / elapsed:.0f} samples/s)  |  "
          f"{size_mb:.1f} MB  →  {h5_path}")


def reconstruct_clean_obs(
    obs_id: np.ndarray,
    reflectivities: np.ndarray,
) -> np.ndarray:
    """Noiseless intensities from stored ids, exact for the flat renderer: the hit object's
    reflectivity, 0 on a miss, ``MARK_VALUE`` on a blink marker ray. ``obs_id`` is (T, R) or
    (N, T, R), ``reflectivities`` (max_obj,) or (N, max_obj)."""
    clean = np.zeros(obs_id.shape, dtype=np.float32)
    hit = obs_id >= 0
    if obs_id.ndim == 2:  # single sample (T, R)
        clean[hit] = reflectivities[obs_id[hit].astype(np.intp)]
    else:  # batched (N, T, R)
        n_idx = np.broadcast_to(
            np.arange(obs_id.shape[0], dtype=np.intp)[:, None, None], obs_id.shape
        )
        clean[hit] = reflectivities[n_idx[hit], obs_id[hit].astype(np.intp)]
    clean[obs_id <= -2] = MARK_VALUE
    return clean


def generate_dataset(dcfg: DatasetConfig, h5_path: str | Path) -> dict:
    """Generate a dataset into ``h5_path`` (parent created; an existing file raises
    ``FileExistsError``). Returns the metadata stored in the file's ``config_json``."""
    h5_path = Path(h5_path)
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists — refusing to overwrite.")
    check_supported(dcfg.sim)

    max_obj = (
        dcfg.sim.n_objects if dcfg.sim.n_objects is not None else dcfg.sim.n_objects_max
    )
    R = obs_dim(dcfg.sim)
    obs_desc = f"1D perspective scan, obs_res={R}"

    meta = {
        "dataset": dataclasses.asdict(dcfg),
        "schema": {
            "_observation":  obs_desc,
            "obs_intensity": f"float32  (N, n_frames={dcfg.sim.n_frames}, R={R})  — noisy intensity in [0,1]; 0=background",
            "obs_depth":     f"float32  (N, n_frames, R={R})  — depth of first hit; 0=miss",
            "obs_id":        f"int8     (N, n_frames, R={R})  — object index, -1=miss",
            "is_visible":     f"bool     (N, n_frames, max_objects={max_obj})  — partial frustum overlap per object",
            "positions":      f"float32  (N, n_frames, max_objects={max_obj}, 2)  — (x, y)",
            "velocities":     "float32  (N, n_frames, max_objects, 2)  — (vx, vy)",
            "colors":         "float32  (N, max_objects, 3)  — RGB, zero-padded",
            "radii":          f"float32  (N, max_objects={max_obj})  — per-object radius, zero-padded",
            "reflectivities": f"float32  (N, max_objects={max_obj})  — per-object reflectivity, zero-padded",
            "n_objects":      "uint8    (N,)  — true object count per sample",
            "seeds":          "int64    (N,)  — RNG seed per sample",
            "_clean_obs_note": "Clean (noiseless) obs can be reconstructed via reconstruct_clean_obs(obs_id, reflectivities) — no extra storage needed.",
        },
    }
    seeds = dcfg.base_seed + np.arange(dcfg.n_samples, dtype=np.int64)
    generate_h5(h5_path, n_samples=dcfg.n_samples, n_workers=dcfg.n_workers,
                write_batch_size=dcfg.write_batch, config_json=json.dumps(meta, indent=2),
                layout=common_layout(dcfg.sim, max_obj), n_frames=dcfg.sim.n_frames,
                chunk=dcfg.hdf5_chunk, compression=dcfg.compression,
                compression_level=dcfg.compression_level,
                worker=_generate_one, args=[(int(s), dcfg.sim, max_obj) for s in seeds])
    return meta
