"""Reading an instance's edits split back into numpy arrays.

``clean_obs`` is the stored ``obs_clean`` when present (soft rendering), else reconstructed
exactly from ``(obs_id, reflectivities)``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from pim.environments.rayworld.dataset import reconstruct_clean_obs


@dataclass
class EditsData:
    """The edits split: observations, post-edit positions and the edit metadata."""

    obs: np.ndarray  # (N, T, R)
    clean_obs: np.ndarray  # (N, T, R)
    positions: np.ndarray  # (N, T, max_obj, 2)
    colors: np.ndarray  # (N, max_obj, 3)
    edit_frame: int  # the same for every sample (read from row 0)
    edit_object: np.ndarray  # (N,)
    edit_op: np.ndarray  # (N,)
    edit_value: np.ndarray  # (N, 2)
    h5_path: str
    T_frames: int
    obs_res: int
    blink_visible: np.ndarray | None = None  # (N, T, n_obj) bool, blink only

    @property
    def n_samples(self) -> int:
        return self.obs.shape[0]


def _blink_visible(f, n_obj):
    """The split's blink schedule, or None."""
    if "blink_visible" in f:
        return f["blink_visible"][:, :, :n_obj].astype(bool)
    return None


def _clean_obs(f, obs_id, reflectivities):
    """Noiseless observations: stored if present, else reconstructed exactly."""
    if "obs_clean" in f:
        return f["obs_clean"][:].astype(np.float32)
    return reconstruct_clean_obs(obs_id, reflectivities)


def load_edits(h5_path: str | Path, *, n_obj_keep: int | None = None) -> EditsData:
    """Load an edits split; ``n_obj_keep`` keeps only the first objects along the object axis."""
    with h5py.File(h5_path, "r") as f:
        T = f["obs_intensity"].shape[1]
        R = f["obs_intensity"].shape[2]
        obs = f["obs_intensity"][:].astype(np.float32)
        max_obj = f["positions"].shape[2]
        n_obj = n_obj_keep if n_obj_keep is not None else max_obj
        positions = f["positions"][:, :, :n_obj, :].astype(np.float32)
        colors = f["colors"][:, :n_obj, :].astype(np.float32)
        obs_id = f["obs_id"][:].astype(np.int8)
        reflectivities = f["reflectivities"][:].astype(np.float32)
        edit_frame = int(f["edit_frame"][0])
        edit_object = f["edit_object"][:]
        edit_op = f["edit_op"][:]
        edit_value = f["edit_value"][:]
        clean_obs = _clean_obs(f, obs_id, reflectivities)
        blink = _blink_visible(f, n_obj)
    return EditsData(
        blink_visible=blink,
        obs=obs,
        clean_obs=clean_obs,
        positions=positions,
        colors=colors,
        edit_frame=edit_frame,
        edit_object=edit_object,
        edit_op=edit_op,
        edit_value=edit_value,
        h5_path=str(h5_path),
        T_frames=T,
        obs_res=R,
    )
