"""Every path under ``datasets/`` and ``runs/_baselines/``; no other module spells one.

An instance lives at ``datasets/<env>/<instance>/{train,probe,eval,edits,tokens}/``. Probe-cache
keys are logical (``probe_key``), never filesystem paths.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
BASELINES = REPO / "runs" / "_baselines"
CLASSES = ("rayworld", "othello")
# the shipped instances; the path helpers accept any name
INSTANCES = {
    "rayworld": ("standard", "blink", "128-ray", "16-ray", "8-ray", "5-ray", "smooth", "obs5", "pair"),
    "othello": ("standard", "adjacent-flip", "adjacent-noflip", "standard-noflip"),
}
DEFAULT_INSTANCE = {"rayworld": "standard", "othello": "standard"}
RW_PROBE_SIZES = ("120k", "250k")
# othello split name -> its directory; "edits" holds the games the edit cases are cut from
OTH_ROLE = {"train": "train", "test": "eval", "probe": "probe", "probe_large": "probe",
            "edits": "edits"}

__all__ = [
    "REPO", "DATASETS", "BASELINES", "CLASSES", "INSTANCES", "DEFAULT_INSTANCE",
    "RW_PROBE_SIZES", "OTH_ROLE", "instance_root", "train_dir", "probe_dir", "probe_file",
    "probe_manifest", "eval_dir", "eval_file", "eval_manifest", "edits_dir", "edits_file",
    "edits_manifest", "edits_selection", "othello_split_dir", "othello_split_file",
    "othello_cases_file", "tokens_dir", "baselines_dir", "probe_key",
]


def _check(cls: str) -> str:
    if cls not in CLASSES:
        raise KeyError(f"unknown environment {cls!r}; one of {CLASSES}")
    return cls


def instance_root(cls: str, inst: str) -> Path:
    return DATASETS / _check(cls) / inst


def train_dir(cls: str, inst: str) -> Path:
    """The training corpus directory."""
    if cls == "othello":
        return othello_split_dir(inst, "train")
    return instance_root(_check(cls), inst) / "train"


def probe_dir(cls: str, inst: str) -> Path:
    """The probe-corpus directory; ``probe_file`` names a specific Rayworld corpus."""
    if cls == "othello":
        return othello_split_dir(inst, "probe")
    return instance_root(_check(cls), inst) / "probe"


def probe_file(cls: str, inst: str, size: str = "120k") -> Path:
    """A Rayworld probe corpus: ``probe/probe_<size>.h5`` (120k for regression targets, 250k for
    categorical targets and the large observation floors)."""
    if cls != "rayworld":
        raise ValueError("probe_file is the rayworld form; othello uses othello_split_file")
    if size not in RW_PROBE_SIZES:
        raise KeyError(f"probe size must be one of {RW_PROBE_SIZES}, got {size!r}")
    return instance_root(cls, inst) / "probe" / f"probe_{size}.h5"


def probe_manifest(cls: str, inst: str, size: str = "120k") -> Path:
    """The manifest beside a Rayworld probe corpus (its ``sim`` config and split record)."""
    return probe_file(cls, inst, size).with_suffix(".json")


def eval_dir(cls: str, inst: str) -> Path:
    if cls == "othello":
        return othello_split_dir(inst, "test")
    return instance_root(_check(cls), inst) / "eval"


def eval_file(cls: str, inst: str) -> Path:
    """Rayworld's held-out sequences, ``eval/test.h5``."""
    if cls != "rayworld":
        raise ValueError("eval_file is the rayworld form; othello uses othello_split_file")
    return instance_root(cls, inst) / "eval" / "test.h5"


def eval_manifest(cls: str, inst: str) -> Path:
    if cls != "rayworld":
        raise ValueError("eval_manifest is the rayworld form")
    return instance_root(cls, inst) / "eval" / "test.json"


def edits_dir(cls: str, inst: str) -> Path:
    """The edit bench directory, ``edits/``."""
    return instance_root(_check(cls), inst) / "edits"


def edits_file(cls: str, inst: str, *, n_cases: int = 1000) -> Path:
    """Rayworld ``edits/edits.h5``; Othello ``edits/cases_<n_cases>.pkl``."""
    d = edits_dir(cls, inst)
    return d / "edits.h5" if cls == "rayworld" else d / f"cases_{n_cases}.pkl"


def edits_manifest(cls: str, inst: str, *, n_cases: int = 1000) -> Path:
    d = edits_dir(cls, inst)
    return d / "edits.json" if cls == "rayworld" else d / f"cases_{n_cases}.json"


def edits_selection(cls: str, inst: str) -> Path:
    """A Rayworld instance's filtered case list, ``edits/selection.json`` (callers test
    ``.exists()``)."""
    return edits_dir(cls, inst) / "selection.json"


def othello_split_dir(inst: str, name: str) -> Path:
    if name not in OTH_ROLE:
        raise KeyError(f"othello split must be one of {sorted(OTH_ROLE)}, got {name!r}")
    return instance_root("othello", inst) / OTH_ROLE[name]


def othello_split_file(inst: str, name: str, n: int) -> Path:
    return othello_split_dir(inst, name) / f"{name}_{n}.npz"


def othello_cases_file(inst: str, n_cases: int = 1000) -> Path:
    """The instance's edit cases, cut from its own ``edits`` games."""
    return edits_file("othello", inst, n_cases=n_cases)


def tokens_dir(inst: str) -> Path:
    return instance_root("rayworld", inst) / "tokens"


def baselines_dir(cls: str, inst: str) -> Path:
    """An instance's floors and their probes: ``runs/_baselines/<env>/<instance>/``."""
    return BASELINES / _check(cls) / inst


def probe_key(cls: str, inst: str, size: str = "120k") -> tuple[str, str]:
    """The ``(data, split)`` fields of a Rayworld probe-cache key, ``("<env>/<inst>",
    "probe_<size>")``. Part of every cached probe's filename hash: do not change the format."""
    if size not in RW_PROBE_SIZES:
        raise KeyError(f"probe size must be one of {RW_PROBE_SIZES}, got {size!r}")
    return f"{_check(cls)}/{inst}", f"probe_{size}"
