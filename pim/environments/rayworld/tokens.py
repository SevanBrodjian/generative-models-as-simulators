"""Frames as tokens, for the 8-ray token model.

On a noiseless instance with fixed reflectivities a frame is one of K**R patterns; ``FrameVocab``
gives each occurring pattern an id (1..V in ascending code order, 0 = ``UNK``)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

UNK = 0


def h5_splits(instance: str) -> tuple[tuple[str, Path], ...]:
    """(name, h5 file) of every small split that joins the vocabulary: the probe corpus, the
    held-out eval sequences and the edits split."""
    from pim.environments import layout

    return (("probe", layout.probe_file("rayworld", instance, "120k")),
            ("test", layout.eval_file("rayworld", instance)),
            ("edits", layout.edits_file("rayworld", instance)))


def frame_codes(frames, levels: np.ndarray) -> np.ndarray:
    """(..., R) ray values to (...) integer pattern codes, base K = len(levels)."""
    x = np.asarray(frames, np.float32)
    R = x.shape[-1]
    flat = x.reshape(-1, R)
    idx = np.clip(np.searchsorted(levels, flat), 0, len(levels) - 1)
    if not np.array_equal(levels[idx], flat):
        bad = flat[(levels[idx] != flat).any(1)][:3]
        raise ValueError(f"frame values off the level grid {levels.tolist()}: {bad.tolist()}")
    w = (len(levels) ** np.arange(R)).astype(np.int64)
    return (idx.astype(np.int64) * w).sum(1).reshape(x.shape[:-1])


@dataclass
class FrameVocab:
    levels: np.ndarray       # (K,) sorted distinct ray values
    frames: np.ndarray       # (V+1, R) float32; row UNK is NaN
    code_to_id: np.ndarray   # (K**R,) int16; 0 = UNK
    counts: np.ndarray       # (V+1,) int64 occurrences in the corpus the vocabulary is built from

    @property
    def size(self) -> int:           # the model's vocabulary size (V + 1)
        return len(self.frames)

    def save(self, path: Path) -> None:
        np.savez(path, levels=self.levels, frames=self.frames,
                 code_to_id=self.code_to_id, counts=self.counts)

    @classmethod
    def load(cls, path: Path) -> "FrameVocab":
        z = np.load(path)
        return cls(z["levels"], z["frames"], z["code_to_id"], z["counts"])


def vocab_from_counts(code_counts: np.ndarray, levels: np.ndarray, obs_dim: int) -> FrameVocab:
    """Ids 1..V for the codes that occur, in ascending code order; 0 = UNK."""
    K = len(levels)
    codes = np.nonzero(code_counts)[0]
    frames = np.full((len(codes) + 1, obs_dim), np.nan, np.float32)
    digits = (codes[:, None] // (K ** np.arange(obs_dim))[None, :]) % K
    frames[1:] = levels[digits]
    code_to_id = np.zeros(K ** obs_dim, np.int16)
    code_to_id[codes] = np.arange(1, len(codes) + 1, dtype=np.int16)
    counts = np.zeros(len(codes) + 1, np.int64)
    counts[1:] = code_counts[codes]
    return FrameVocab(np.asarray(levels, np.float32), frames, code_to_id, counts)


def encode(frames, vocab: FrameVocab) -> np.ndarray:
    """(..., R) frames to (...) int16 ids; a pattern outside the vocabulary is UNK."""
    return vocab.code_to_id[frame_codes(frames, vocab.levels)]


def load_tokens(tdir: Path):
    """(train tokens (N, T) int16 memmap, lengths (N,) int8, FrameVocab, meta)."""
    tdir = Path(tdir)
    meta = json.loads((tdir / "meta.json").read_text())
    N, T = meta["n_train"], meta["n_frames"]
    tok = np.memmap(tdir / "train.i16", np.int16, mode="r", shape=(N, T))
    ln = np.full(N, T, np.int8)
    return tok, ln, FrameVocab.load(tdir / "vocab.npz"), meta


def tokenize_instance(instance_dir: Path, chunk: int = 250_000, levels=None, log=print) -> dict:
    """Build the vocabulary over every split and write ``tokens/`` (``train.i16``, ``vocab.npz``,
    ``meta.json``) under ``instance_dir`` = ``layout.instance_root("rayworld", <instance>)``.
    The small splits join the vocabulary but are encoded from their h5 at read time."""
    from pim.environments import layout

    t0 = time.time()
    inst_dir = Path(instance_dir)
    train = layout.train_dir("rayworld", inst_dir.name)
    inst = json.loads((train / "corpus.json").read_text())   # written by scripts/build_rayworld_corpus.py
    T, R, N = int(inst["n_frames"]), int(inst["obs_dim"]), int(inst["n"])
    obs = np.memmap(train / "obs.f32", np.float32, mode="r", shape=(N, T, R))
    out = layout.tokens_dir(inst_dir.name)
    out.mkdir(exist_ok=True)
    if levels is None:
        levels = np.unique(np.asarray(obs[: min(N, 20_000)]))
    levels = np.asarray(levels, np.float32)
    K = len(levels)
    # pass 1 stores raw pattern codes (< K**R) in the int16 file before remapping to ids
    assert K ** R < 2 ** 15, (f"K**R = {K ** R:,} patterns overflow the int16 code/id files "
                              f"(max 32,767); widen the dtype before tokenizing this instance")
    log(f"{inst['instance']}: N={N:,} T={T} R={R}; levels {levels.tolist()} -> "
        f"{K ** R:,} possible patterns", flush=True)

    # pass 1: train frames to codes (int16 file, remapped in pass 2) and counts
    counts = np.zeros(K ** R, np.int64)
    per_split_counts = {}
    mm = np.memmap(out / "train.i16", np.int16, mode="w+", shape=(N, T))
    tr_counts = np.zeros(K ** R, np.int64)
    for i in range(0, N, chunk):
        codes = frame_codes(obs[i: i + chunk], levels)
        mm[i: i + chunk] = codes.astype(np.int16)
        tr_counts += np.bincount(codes.ravel(), minlength=K ** R)
        if (i // chunk) % 10 == 0:
            log(f"  train {i + len(codes):>12,}/{N:,}  distinct so far {int((tr_counts > 0).sum())}"
                f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    mm.flush()
    per_split_counts["train"] = tr_counts
    counts += tr_counts
    # the small splits join the vocabulary (counted, not stored)
    splits = h5_splits(inst_dir.name)
    for name, p in splits:
        with h5py.File(p) as h:
            x = h["obs_intensity"][:]
        per_split_counts[name] = np.bincount(frame_codes(x, levels).ravel(), minlength=K ** R)
        counts += per_split_counts[name]

    vocab = vocab_from_counts(counts, levels, R)
    vocab.save(out / "vocab.npz")
    # pass 2: codes to ids in place
    for i in range(0, N, chunk):
        mm[i: i + chunk] = vocab.code_to_id[mm[i: i + chunk].astype(np.int64)]
    mm.flush()
    del mm

    in_vocab = counts > 0
    meta = {
        "instance": inst["instance"],
        "sources": {n: str(p.relative_to(layout.REPO)) if p.is_relative_to(layout.REPO) else p.name
                    for n, p in splits},
        "n_train": N, "n_frames": T, "obs_dim": R, "levels": levels.tolist(),
        "possible_patterns": int(K ** R), "vocab_size": int(vocab.size),
        "unk_id": UNK, "id_order": "ascending pattern code; ids 1..V, 0 = UNK",
        "vocab_built_from": ["train"] + [n for n, _ in splits],
        "distinct_frames": {k: int((v > 0).sum()) for k, v in per_split_counts.items()},
        "frames_only_outside_train": int((in_vocab & (tr_counts == 0)).sum()),
        "frame_occurrences_outside_train_vocab": {
            k: int(v[in_vocab & (tr_counts == 0)].sum()) for k, v in per_split_counts.items()
            if k != "train"},
        "files": {"train": "train.i16 (N, T) int16 memmap",
                  "vocab": "vocab.npz (levels, frames (V+1, R), code_to_id, counts)"},
        "small_splits": "not stored as tokens; encode their h5 at read time (tokens.encode)",
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done: vocab {vocab.size} (incl. UNK), distinct per split {meta['distinct_frames']}, "
        f"frames only outside train {meta['frames_only_outside_train']}  [{meta['minutes']} min]",
        flush=True)
    return meta
