"""Othello instances (their rules) and their corpora: disjoint index ranges of one seed.

A game is a pure function of its index (``data._one_game``), so each split is an index range
(``TRAIN_LO`` ... ``EDITS_LO`` below); ``verify_splits`` checks that they are disjoint."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import numpy as np

from pim.environments.othello import data as od

SEED = 0
MAXLEN = od.MAXLEN      # 60
# flip: whether a placed disc recolors the discs it encloses.
# placement: "enclosure" (Othello's legality rule) or "adjacent" (the move must touch one of
# the mover's own discs in the 8-neighborhood).
INSTANCES = {
    "standard": {"flip": True, "placement": "enclosure"},
    "standard-noflip": {"flip": False, "placement": "enclosure"},
    "adjacent-noflip": {"flip": False, "placement": "adjacent"},
    "adjacent-flip": {"flip": True, "placement": "adjacent"},
}


def corpus_dir(instance: str = "standard", split: str = "train") -> Path:
    """The directory holding one of an instance's splits (``train`` | ``test`` | ``probe`` |
    ``probe_large`` | ``edits``)."""
    from pim.environments.layout import othello_split_dir

    if instance not in INSTANCES:
        raise KeyError(f"unknown othello instance {instance!r}; registered: {sorted(INSTANCES)}")
    return othello_split_dir(instance, split)


def flip_of(instance: str = "standard") -> bool:
    """The recoloring rule of an instance; every replay (labels, legal sets, bench) must use it."""
    return INSTANCES[instance]["flip"]


def placement_of(instance: str = "standard") -> str:
    """The placement rule of an instance: "enclosure" | "adjacent"."""
    return INSTANCES[instance].get("placement", "enclosure")


def rules_of(instance: str = "standard") -> dict:
    """Both rules as keyword arguments, for ``OthelloBoardState(**rules_of(inst))`` and every
    replaying helper (``tokens_and_labels``, ``legal_sets``, ``synthesize_cases``, ...)."""
    return {"flip": flip_of(instance), "placement": placement_of(instance)}


TRAIN_LO, N_TRAIN_GAMES = 0, 20_000_000
TEST_LO, TEST_N = 90_000_000, 10_000             # held-out gates, Bayes floor
PROBE_LO, PROBE_N = 91_000_000, 20_000           # probe and inverse-map fits
PROBE_LARGE_LO, PROBE_LARGE_N = 92_000_000, 170_000   # the large observation floor
EDITS_LO, EDITS_N = 93_000_000, 10_000           # the games the edit cases are cut from


def _generate(lo: int, n: int, chunk: int = 500_000, n_workers: int | None = None,
              log=print, flip: bool = True, placement: str = "enclosure") -> tuple[np.ndarray, np.ndarray]:
    """Games ``[lo, lo + n)`` as int8 tokens, converted chunk by chunk (a list of 20M games
    would not fit in memory; the int8 array is 1.2 GB)."""
    n_workers = n_workers or multiprocessing.cpu_count()
    stoi = od.canonical_vocab()
    tok = np.zeros((n, MAXLEN), np.int8)
    ln = np.zeros(n, np.int8)
    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            args = [(i, SEED, flip, placement) for i in range(lo + c0, lo + c1)]
            for j, g in enumerate(pool.imap(od._one_game, args, chunksize=256)):
                m = g[:MAXLEN]
                ln[c0 + j] = len(m)
                tok[c0 + j, : len(m)] = [stoi[s] for s in m]
            el = time.time() - t0
            log(f"    {c1:>10,}/{n:,}  {c1 / el:>8,.0f} games/s  "
                f"eta {(n - c1) / max(c1 / el, 1) / 60:5.1f} min")
    return tok, ln


def _regen_row(index: int, seed: int, flip: bool, stoi: dict,
               placement: str = "enclosure") -> tuple[np.ndarray, int]:
    g = od._one_game((index, seed, flip, placement))[:MAXLEN]
    row = np.zeros(MAXLEN, np.int8)
    row[: len(g)] = [stoi[s] for s in g]
    return row, len(g)


def verify_splits(paths: dict[str, Path], n_check: int = 8, log=print) -> dict[str, tuple[int, int]]:
    """Check a set of split files: their index ranges ``[lo, lo + n)`` are pairwise disjoint,
    and at ``n_check`` sampled rows (plus the first and last) the game regenerated from the
    recorded ``lo`` / ``seed`` / rules equals the stored row. Raises ``AssertionError``;
    returns the ranges. (Identical games at different indices are allowed: short games recur.)
    """
    stoi = od.canonical_vocab()
    rng = np.random.default_rng(0)
    ranges: dict[str, tuple[int, int]] = {}
    for name, p in paths.items():
        z = np.load(p)
        tok, ln, lo, seed = z["tokens"], z["lengths"], int(z["lo"]), int(z["seed"])
        # a file without a rule field was generated under the standard rule
        flip = bool(z["flip"]) if "flip" in z.files else True
        placement = str(z["placement"]) if "placement" in z.files else "enclosure"
        n = len(tok)
        ranges[name] = (lo, lo + n)
        idx = sorted({0, n - 1, *rng.integers(0, n, n_check).tolist()})
        for j in idx:
            row, length = _regen_row(lo + j, seed, flip, stoi, placement)
            assert np.array_equal(row, tok[j]) and int(ln[j]) == length, (
                f"{name}: stored row {j} is not the game at index {lo + j} (seed {seed}, "
                f"flip {flip}) — the recorded lo/seed/flip do not describe {p.name}")
    names = list(ranges)
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            (a0, a1), (b0, b1) = ranges[a_], ranges[b_]
            assert a1 <= b0 or b1 <= a0, (
                f"index ranges overlap: {a_} [{a0:,}, {a1:,}) vs {b_} [{b0:,}, {b1:,}) — "
                f"held-out data would be training data")
    if log:
        log("  splits verified: " + ", ".join(f"{k} [{lo:,}, {hi:,})" for k, (lo, hi) in ranges.items())
            + f"; {n_check}+2 regenerated rows per split bit-identical")
    return ranges


def build(n_train: int = N_TRAIN_GAMES, log=print, only: tuple[str, ...] | None = None,
          instance: str = "standard") -> dict[str, Path]:
    """Generate (or reuse) the named splits of an instance; returns their paths.

    CPU only; games are generated in parallel, one process per core.
    """
    flip, placement = flip_of(instance), placement_of(instance)
    out = {}
    plan = [("train", TRAIN_LO, n_train), ("test", TEST_LO, TEST_N), ("probe", PROBE_LO, PROBE_N),
            ("probe_large", PROBE_LARGE_LO, PROBE_LARGE_N), ("edits", EDITS_LO, EDITS_N)]
    if only is not None:
        plan = [x for x in plan if x[0] in only]
    for name, lo, n in plan:
        cache = corpus_dir(instance, name)
        p = cache / f"{name}_{n}.npz"
        out[name] = p
        if p.exists():
            log(f"  {name:<6} {n:>10,} games — cached")
            continue
        # a larger file at the same lo holds this split as a prefix
        bigger = sorted((q for q in cache.glob(f"{name}_*.npz")
                         if q.stem.split("_")[-1].isdigit()
                         and int(q.stem.split("_")[-1]) >= n),
                        key=lambda q: int(q.stem.split("_")[-1]))
        if bigger:
            out[name] = bigger[0]
            log(f"  {name:<6} {n:>10,} games — prefix of {bigger[0].name}")
            continue
        cache.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tok, ln = _generate(lo, n, log=log, flip=flip, placement=placement)
        np.savez(p, tokens=tok, lengths=ln, lo=lo, seed=SEED, flip=flip, placement=placement,
                 instance=instance)
        log(f"  {name:<6} {n:>10,} games in {time.time() - t0:6.1f}s  "
            f"({n / (time.time() - t0):,.0f}/s, {p.stat().st_size / 1e6:.0f} MB)  "
            f"mean length {ln.mean():.1f}")
    return out


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["tokens"], z["lengths"]


def probe_data(path: Path, n: int | None = None, flip: bool = True, placement: str = "enclosure"):
    """``tokens_and_labels`` for the first ``n`` games of a split, cached beside it as
    ``<stem>_labels_<n>.npz`` (labeling 170k games takes about 10 min)."""
    import dataclasses

    from pim.environments.othello.data import ProbeData, canonical_vocab, tokens_and_labels

    path = Path(path)
    tok, ln = load(path)
    n = len(tok) if n is None else min(n, len(tok))
    cache = path.with_name(f"{path.stem}_labels_{n}.npz")
    if cache.exists():
        z = np.load(cache)
        return ProbeData(**{k: z[k] for k in z.files})
    itos = {v: k for k, v in canonical_vocab().items()}
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n], ln[:n])],
                             flip=flip, placement=placement)
    np.savez(cache, **{f.name: getattr(data, f.name) for f in dataclasses.fields(data)})
    return data
