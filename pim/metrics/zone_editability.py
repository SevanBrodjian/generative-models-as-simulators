"""Editability of a predicted frame (Rayworld): the ray-zone Edit Index and the fidelity ratio.

At the edit step both worlds are rendered, ``gt_edited`` and ``gt_unedited``; the Edit Index asks
which one the prediction is nearer on the rays where they differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim.metrics.edit_index import case_stats, edit_index_per_case, fidelity_ratio_from, ratio_ci95

DIFF_EPS = (
    1e-3  # intensity difference at which the two worlds differ on a ray
)


@dataclass
class EditZones:
    """The two ground-truth worlds at the edit frame, plus the ray zones ((N, R) unless noted)."""

    gt_edited: np.ndarray    # clean render with the edited disc at its target
    gt_unedited: np.ndarray  # clean render where the disc continued along its own velocity
    target: np.ndarray       # bool: rays the edited disc occupies in gt_edited
    ghost: np.ndarray        # bool: rays it occupied before the edit and vacates
    collateral: np.ndarray   # bool: rays the other disc occupies in gt_edited
    differing: np.ndarray    # bool: rays where the two worlds differ (the Edit Index's support)
    teleport: np.ndarray     # (N,) edit distance in sim units
    gt_unedited_traj: np.ndarray | None = None  # (N, K, R) the unedited world rolled forward
    differing_traj: np.ndarray | None = None    # (N, K, R) bool, the per-step support


def sim_config_from(sim: dict, n_obj: int):
    """The ``SimConfig`` for clean reference renders, inheriting the dataset's rendering settings
    (``sim`` is the dataset's stored sim dict) with ``obs_noise_std=0``."""
    from pim.environments.rayworld.sim import SimConfig

    return SimConfig(
        seed=0,
        y_near=sim["y_near"],
        y_far=sim["y_far"],
        x_near=sim["x_near"],
        x_far=sim["x_far"],
        n_objects=n_obj,
        radius=sim["radius"],
        n_frames=1,
        dt=float(sim["dt"]),
        obs_res=sim["obs_res"],
        drop_edge_rays=sim.get("drop_edge_rays", False),
        refl_min=sim["refl_min"],
        refl_max=sim["refl_max"],
        fixed_reflectivities=True,
        obs_noise_std=0.0,
        boundary="open",
        always_in_frustum=False,
        soft_edge=sim.get("soft_edge", 0.0),
        soft_shading=sim.get("soft_shading", "flat"),
        soft_profile_power=sim.get("soft_profile_power", 2.0),
        n_observers=sim.get("n_observers", 1),
        region=sim.get("region", "frustum"),
        soft_psf_sigma=sim.get("soft_psf_sigma", 0.0),
        soft_occlusion_temp=sim.get("soft_occlusion_temp", 0.0),
    )


def build_edit_zones(
    *,
    pre_pos: np.ndarray,          # (N, n_obj, 2) positions at frame ef-1
    tgt_pos: np.ndarray,          # (N, n_obj, 2) positions at frame ef, the edited disc moved
    pre_vel: np.ndarray,          # (N, n_obj, 2) velocities at frame ef-1
    edit_object: np.ndarray,      # (N,) index of the edited disc
    sim: dict,                    # the dataset's config["dataset"]["sim"]
    n_obj: int = 2,
    traj_pos: np.ndarray | None = None,        # (N, K, n_obj, 2) true positions over the rollout
    gt_edited_traj: np.ndarray | None = None,  # (N, K, R) clean edited frames over the rollout
    blink_visible: np.ndarray | None = None,   # (N, K+2, n_obj) Blink visibility over ef-1 .. ef+K
) -> EditZones:
    """Render both ground-truth worlds at the edit frame ef and derive the ray zones; with
    ``traj_pos`` and ``gt_edited_traj`` the unedited world is rolled forward too. Blink references
    get the same blackouts, so a hidden edited disc leaves no differing rays (NaN at that step)."""
    from pim.environments.rayworld.blink import paint_markers
    from pim.environments.rayworld.config import obs_dim
    from pim.environments.rayworld.renderer import render_frame as _render_frame

    def render_frame(p, rad_, refl_, cfg_, *, case=0, frame=0):
        """``renderer.render_frame`` with case ``case``'s Blink state at schedule index ``frame``."""
        if blink_visible is None:
            return _render_frame(p, rad_, refl_, cfg_)
        v = blink_visible[case]
        d, ids, inten = _render_frame(p, rad_, refl_, cfg_, visible=v[frame])
        paint_markers(ids, inten, v[frame], v[frame + 1] if frame + 1 < v.shape[0] else None)
        return d, ids, inten

    n = len(pre_pos)
    # the zones assume exactly two discs: the edited one and the other
    assert n_obj == 2, f"build_edit_zones assumes n_obj == 2 (one edited, one 'other'); got {n_obj}"
    dt = float(sim["dt"])
    cfg = sim_config_from(sim, n_obj)
    refl = np.linspace(sim["refl_min"], sim["refl_max"], n_obj).astype(np.float32)
    rad = np.full(n_obj, sim["radius"], np.float32)
    R = obs_dim(cfg)                 # rays kept (obs_res - 2 when the edge rays are dropped)

    # the unedited world: the edited disc continues along its own velocity; the other disc
    # sits at its true frame-ef position
    uned_pos = tgt_pos.copy()
    idx = np.arange(n)
    k = edit_object.astype(int)
    uned_pos[idx, k] = pre_pos[idx, k] + pre_vel[idx, k] * dt

    gt_edited = np.zeros((n, R), np.float32)
    gt_unedited = np.zeros((n, R), np.float32)
    id_edited = np.full((n, R), -1, np.int64)
    id_pre = np.full((n, R), -1, np.int64)
    for i in range(n):
        _, ide, inte = render_frame(tgt_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=1)
        _, _, intu = render_frame(uned_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=1)
        _, idp, _ = render_frame(pre_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=0)
        gt_edited[i], gt_unedited[i] = inte, intu
        id_edited[i], id_pre[i] = ide, idp

    other = 1 - k
    target = id_edited == k[:, None]
    ghost = (id_pre == k[:, None]) & (id_edited != k[:, None])
    collateral = id_edited == other[:, None]
    differing = np.abs(gt_edited - gt_unedited) > DIFF_EPS

    # roll the unedited world forward; the other disc follows its true trajectory
    uned_traj = diff_traj = None
    if traj_pos is not None and gt_edited_traj is not None:
        K = traj_pos.shape[1]
        uned_traj = np.zeros((n, K, R), np.float32)
        for s_ in range(K):
            step_pos = traj_pos[:, s_].copy()
            step_pos[idx, k] = pre_pos[idx, k] + pre_vel[idx, k] * dt * (s_ + 1)
            for i in range(n):
                _, _, inten = render_frame(
                    step_pos[i].astype(np.float32), rad, refl, cfg, case=i, frame=1 + s_
                )
                uned_traj[i, s_] = inten
        diff_traj = np.abs(gt_edited_traj - uned_traj) > DIFF_EPS

    return EditZones(
        gt_edited=gt_edited,
        gt_unedited=gt_unedited,
        target=target,
        ghost=ghost,
        collateral=collateral,
        differing=differing,
        teleport=np.linalg.norm(tgt_pos[idx, k] - pre_pos[idx, k], axis=-1),
        gt_unedited_traj=uned_traj,
        differing_traj=diff_traj,
    )


def _index_from(pred, gt_edit, gt_uned, mask) -> float:
    """Mean Edit Index for one frame: the two clean renders as references, the differing rays as support."""
    return float(np.nanmean(edit_index_per_case(pred, gt_edit, gt_uned, mask)))


def edit_index_by_step(
    roll: np.ndarray, zones: EditZones, gt_traj: np.ndarray
) -> list[float]:
    """Edit Index at every rollout step against the unedited world rolled forward (step 0 is
    ``edit_index``). Needs ``build_edit_zones(traj_pos=..., gt_edited_traj=...)``."""
    if zones.gt_unedited_traj is None:
        return []
    K = min(roll.shape[1], zones.gt_unedited_traj.shape[1])
    return [
        _index_from(
            roll[:, s],
            gt_traj[:, s],
            zones.gt_unedited_traj[:, s],
            zones.differing_traj[:, s],
        )
        for s in range(K)
    ]


def zone_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """RMSE between two (N, R) observation arrays over an (N, R) boolean ray mask."""
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((pred - gt) ** 2)[mask].mean()))


def edit_index(pred: np.ndarray, zones: EditZones) -> float:
    """Mean per-case ray-zone Edit Index in [-1, +1], each edit weighted equally."""
    return _index_from(pred, zones.gt_edited, zones.gt_unedited, zones.differing)


def edit_scorecard(
    roll: np.ndarray,
    zones: EditZones,
    gt_traj: np.ndarray,
) -> dict:
    """Every number an edit arm reports on a frame model, from its free run ``roll`` (N, K, R; step 0 is
    the edit frame), ``zones`` and the clean edited frames ``gt_traj`` (N, K, R). The fidelity ratio
    needs the unedited card too: ``fidelity_ratio(card, unedited_card)``."""
    p0 = roll[:, 0]
    step = [
        float(np.sqrt(((roll[:, s] - gt_traj[:, s]) ** 2).mean()))
        for s in range(roll.shape[1])
    ]
    allm = np.ones_like(zones.target)
    ei_case = edit_index_per_case(p0, zones.gt_edited, zones.gt_unedited, zones.differing)
    return dict(
        edit_index=float(np.nanmean(ei_case)),
        edit_index_by_step=edit_index_by_step(roll, zones, gt_traj),
        edit_frame_rmse=zone_rmse(p0, zones.gt_edited, allm),
        target_rmse=zone_rmse(p0, zones.gt_edited, zones.target),
        ghost_rmse=zone_rmse(p0, zones.gt_edited, zones.ghost),
        collateral_rmse=zone_rmse(p0, zones.gt_edited, zones.collateral),
        gt_traj_rmse=float(np.mean(step)),
        step_rmse_to_gt=step,
        # per-case lists never become scalar arm fields in scores.json
        **case_stats(ei_case, prefix="edit_index_"),
        mse_per_case=((p0 - zones.gt_edited) ** 2).mean(axis=1).tolist(),
    )


def fidelity_ci95(card: dict, unsteered_card: dict) -> dict:
    """Percentile-bootstrap 95% interval of ``fidelity_ratio`` over the bench cases
    (``edit_index.ratio_ci95`` with ``root``); empty when a card lacks its per-case MSEs."""
    a, b = card.get("mse_per_case"), unsteered_card.get("mse_per_case")
    if a is None or b is None:
        return {}
    lo, hi = ratio_ci95(a, b, root=True)
    return {"fidelity_ci95_lo": lo, "fidelity_ci95_hi": hi}


def fidelity_ratio(card: dict, unsteered_card: dict) -> float:
    """The fidelity ratio of a frame model: whole-frame RMSE of the edited prediction over that of
    the unedited prediction, both against the edited world, at the edit step. Above 1 is degraded."""
    return fidelity_ratio_from(card["edit_frame_rmse"], unsteered_card["edit_frame_rmse"])

