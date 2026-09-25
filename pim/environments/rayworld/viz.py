"""Animation of a Rayworld scene for the demos, and the dark palette the figures share.

Left panel: the 2D world; right panel: the observation waterfall, in ``"model"`` (grayscale) or
``"human"`` (colored by object, brightness by inverse depth) mode."""

from __future__ import annotations

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from .config import SimConfig, obs_dim
from .sim import Scene

# the dark palette the demos and figures share
BG = np.array([0.04, 0.04, 0.08])  # dark navy, RGB in [0, 1]
BG_HEX = "#0a0a14"
FRUSTUM_EDGE = "#5c677f"
TICK_COLOR = "#808a9d"
TEXT_COLOR = "#a3adc2"
_TRAIL_LEN = 25  # frames of trajectory trail to show
_TRAIL_ALPHA = 0.28
_DEPTH_GAMMA = 0.6  # brightness = 1 − 0.75·norm^γ  (near=1, far=0.25)


def _trail_t0(positions_i: np.ndarray, t0: int, f: int, cfg: SimConfig) -> int:
    """Trail start index, moved past any wrap-around jump."""
    if cfg.boundary != "wrap" or t0 >= f:
        return t0
    seg = positions_i[t0 : f + 1]
    diffs = np.abs(np.diff(seg, axis=0))
    wraps = (diffs[:, 0] > cfg.x_far) | (diffs[:, 1] > (cfg.y_far - cfg.y_near) * 0.5)
    if wraps.any():
        return t0 + int(np.where(wraps)[0][-1]) + 1
    return t0


def _brightness(depth: np.ndarray, cfg: SimConfig) -> np.ndarray:
    norm = (depth - cfg.y_near) / (cfg.y_far - cfg.y_near)
    norm = np.clip(norm, 0.0, 1.0)
    return 1.0 - 0.75 * (norm**_DEPTH_GAMMA)


def make_waterfall(
    obs_depth: np.ndarray,  # (n_frames, obs_res)
    obs_id: np.ndarray,  # (n_frames, obs_res) int
    obs_intensity: np.ndarray,  # (n_frames, obs_res) float in [0, 1]
    scene: Scene,
    mode: Literal["model", "human"] = "model",
) -> np.ndarray:
    """The RGBA waterfall image, (n_frames, obs_res, 4): grayscale intensity (``"model"``) or
    colored by object with brightness by inverse depth (``"human"``)."""
    cfg = scene.config
    n_frames, obs_res = obs_depth.shape

    img = np.zeros((n_frames, obs_res, 4))
    img[:, :, :3] = BG
    img[:, :, 3] = 1.0

    if mode == "model":
        gray = np.clip(obs_intensity, 0.0, 1.0)  # (T, R)
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray
    else:  # "human"
        brt = _brightness(obs_depth, cfg)  # (n_frames, obs_res)
        for obj_id, color in enumerate(scene.colors):
            mask = obs_id == obj_id
            for c in range(3):
                img[:, :, c] = np.where(
                    mask,
                    np.clip(color[c] * brt, 0.0, 1.0),
                    img[:, :, c],
                )

    return img


def animate_scene(
    scene: Scene,
    obs_depth: np.ndarray,
    obs_id: np.ndarray,
    obs_intensity: np.ndarray,
    interval: int = 50,
    title: Optional[str] = None,
    waterfall_mode: Literal["model", "human"] = "model",
) -> FuncAnimation:
    """A matplotlib ``FuncAnimation`` of the scene; ``interval`` is milliseconds per frame."""
    cfg = scene.config
    n_frames = cfg.n_frames
    n_obj = scene.positions.shape[1]

    waterfall = make_waterfall(
        obs_depth, obs_id, obs_intensity, scene, mode=waterfall_mode
    )

    fig = plt.figure(figsize=(12, 5.5), facecolor=BG_HEX)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.10, wspace=0.12)
    ax_w = fig.add_subplot(1, 2, 1)  # 2D environment
    ax_f = fig.add_subplot(1, 2, 2)  # waterfall

    for ax in (ax_w, ax_f):
        ax.set_facecolor(BG_HEX)
        for spine in ax.spines.values():
            spine.set_edgecolor(TICK_COLOR)
        ax.tick_params(colors=TICK_COLOR, labelsize=9)

    mx, my = 0.7, 0.7
    ax_w.set_xlim(-cfg.x_far - mx, cfg.x_far + mx)
    ax_w.set_ylim(cfg.y_near - my, cfg.y_far + my)
    ax_w.set_aspect("equal")
    ax_w.set_xlabel("x", color=TEXT_COLOR, fontsize=10)
    ax_w.set_ylabel("depth  y", color=TEXT_COLOR, fontsize=10)
    ax_w.set_title(
        "2D environment  (simulator state)", color=TEXT_COLOR, fontsize=11, pad=8
    )

    # frustum outline
    corners = np.array(
        [
            [-cfg.x_near, cfg.y_near],
            [cfg.x_near, cfg.y_near],
            [cfg.x_far, cfg.y_far],
            [-cfg.x_far, cfg.y_far],
        ]
    )
    ax_w.add_patch(
        Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=FRUSTUM_EDGE,
            linewidth=1.8,
            zorder=1,
        )
    )
    ax_w.text(
        0,
        cfg.y_near - 0.15,
        "near",
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=8.5,
        fontfamily="monospace",
    )
    ax_w.text(
        0,
        cfg.y_far + 0.15,
        "far",
        ha="center",
        va="bottom",
        color=TEXT_COLOR,
        fontsize=8.5,
        fontfamily="monospace",
    )

    # circles + trails + reflectivity labels
    circles, trails, refl_labels = [], [], []
    for i in range(n_obj):
        x0, y0 = scene.positions[0, i]
        refl = scene.reflectivities[i]

        circ = plt.Circle(
            (x0, y0),
            scene.radii[i],
            facecolor=scene.colors[i],
            zorder=3,
            linewidth=1.2,
            edgecolor="white",
            alpha=0.95,
        )
        ax_w.add_patch(circ)
        circles.append(circ)

        (trail,) = ax_w.plot(
            [x0],
            [y0],
            color=scene.colors[i],
            linewidth=1.5,
            alpha=_TRAIL_ALPHA,
            zorder=2,
            solid_capstyle="round",
        )
        trails.append(trail)

        # reflectivity label centered inside the circle
        lbl = ax_w.text(
            x0,
            y0,
            f"{refl:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            fontfamily="monospace",
            fontweight="bold",
            zorder=4,
        )
        refl_labels.append(lbl)

    frame_text = ax_w.text(
        -cfg.x_far - mx + 0.2,
        cfg.y_far + my - 0.15,
        "frame   0",
        color=TEXT_COLOR,
        fontsize=9,
        fontfamily="monospace",
        va="top",
    )

    mode_tag = (
        "model  (intensity)" if waterfall_mode == "model" else "human  (color+depth)"
    )
    wf_display = np.zeros((n_frames, obs_dim(cfg), 4))
    wf_display[:, :, :3] = BG
    wf_display[:, :, 3] = 1.0

    wf_img = ax_f.imshow(
        wf_display,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[0, obs_dim(cfg), n_frames, 0],
    )
    ax_f.set_xlabel("scan position", color=TEXT_COLOR, fontsize=10)
    ax_f.set_ylabel("frame", color=TEXT_COLOR, fontsize=10)
    ax_f.set_title(
        f"1D observation  ({mode_tag})", color=TEXT_COLOR, fontsize=11, pad=8
    )
    ax_f.set_xlim(0, obs_dim(cfg))
    ax_f.set_ylim(n_frames, 0)

    # horizontal highlight that tracks the current frame on the waterfall
    (frame_line,) = ax_f.plot(
        [0, obs_dim(cfg)],
        [0.5, 0.5],
        color="white",
        linewidth=1.4,
        alpha=0.45,
        zorder=5,
    )

    if title:
        fig.suptitle(title, color=TEXT_COLOR, fontsize=12, y=0.975)

    def update(f: int):
        for i, (circ, trail, lbl) in enumerate(zip(circles, trails, refl_labels)):
            pos = scene.positions[f, i]
            circ.center = pos
            lbl.set_position(pos)
            t0 = _trail_t0(scene.positions[:, i], max(0, f - _TRAIL_LEN), f, cfg)
            trail.set_data(
                scene.positions[t0 : f + 1, i, 0],
                scene.positions[t0 : f + 1, i, 1],
            )

        frame_text.set_text(f"frame  {f:3d}")

        wf_display[: f + 1] = waterfall[: f + 1]
        wf_img.set_data(wf_display)

        frame_line.set_ydata([f + 0.5, f + 0.5])

        return circles + trails + refl_labels + [frame_text, wf_img, frame_line]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    return anim


def save_animation(
    anim: FuncAnimation,
    path: str,
    fps: int = 20,
    dpi: int = 110,
) -> None:
    """Save an animation: GIF through Pillow, anything else through ffmpeg."""
    if path.endswith(".gif"):
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    else:
        anim.save(path, fps=fps, dpi=dpi, extra_args=["-vcodec", "libx264"])
    print(f"saved → {path}")
