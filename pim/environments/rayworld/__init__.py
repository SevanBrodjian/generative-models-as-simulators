"""Rayworld: discs moving in a 2D world, observed through a 1D fan of rays.

Each variant (standard, blink, 128-ray, 16-ray, 8-ray, 5-ray, smooth, obs5) is an instance with
its own simulator settings and seed ranges, registered in ``bigcorpus.INSTANCES``."""

from pim.environments.rayworld.config import SimConfig
from pim.environments.rayworld.dataset import (
    DatasetConfig,
    generate_dataset,
    reconstruct_clean_obs,
)
from pim.environments.rayworld.edits_dataset import EditDatasetConfig, generate_edits_dataset
from pim.environments.rayworld.loading import EditsData, load_edits
from pim.environments.rayworld.renderer import render_frame, render_scene
from pim.environments.rayworld.sim import (
    OBJECT_COLORS,
    Scene,
    compute_visibility,
    frustum_half_width,
    simulate,
)

__all__ = [
    "SimConfig",
    "Scene",
    "simulate",
    "compute_visibility",
    "frustum_half_width",
    "OBJECT_COLORS",
    "render_frame",
    "render_scene",
    "DatasetConfig",
    "generate_dataset",
    "reconstruct_clean_obs",
    "EditDatasetConfig",
    "generate_edits_dataset",
    "EditsData",
    "load_edits",
]
