"""The training loop: one loop and one recipe (``TrainConfig``) for every run, two objectives.

AdamW (lr 1e-3, weight decay 1e-4), gradient clip 1.0, batch 256, 2,000 warmup steps then a constant
rate, seed 0. The ``DataSource`` supplies MSE on the next frame or cross-entropy on the next token.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

IGNORE = -100  # CE ignore index for padded positions


@dataclass
class TrainConfig:
    """The training recipe; the defaults are the configuration every run in the paper used."""

    steps: int
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_schedule: str = "constant"  # "constant" | "cosine"
    warmup_steps: int = 2_000
    ckpt_base: int = 1_000  # log-spaced ckpts at base, 2·base, …; MUST be >= 1 (see below)
    val_every: int = 5_000
    seed: int = 0

    def __post_init__(self):
        # the log-spaced schedule doubles from ckpt_base, so 0 would never advance
        if self.ckpt_base < 1:
            raise ValueError("ckpt_base must be >= 1")


@dataclass
class DataSource:
    """What an environment supplies to the loop."""

    batches: Iterator          # infinite iterator of collated training batches
    loss_fn: Callable          # (model, batch) -> scalar loss
    validate: Callable         # (model) -> validation loss
    steps_per_epoch: float
    meta: dict = field(default_factory=dict)      # recorded into config.json
    skip: Callable[[int], None] | None = None     # advance n batches, so a resumed run sees the same batches


def mse_next_obs(model, x: torch.Tensor) -> torch.Tensor:
    """(B, T, R) frames -> MSE on the next frame at every position (input ``x[:, :-1]``,
    target ``x[:, 1:]``)."""
    return F.mse_loss(model(x[:, :-1]), x[:, 1:])


def xy_tokens(tok: torch.Tensor, ln: torch.Tensor, block: int):
    """Next-token (input, target) pairs over a right-padded batch; padded targets are IGNORE."""
    x = tok[:, :block].long()
    y = tok[:, 1: block + 1].long()
    pos = torch.arange(block, device=tok.device)[None, :]
    y = y.masked_fill(pos >= (ln[:, None].long() - 1), IGNORE)
    return x, y


def ce_next_move(model, batch, block: int = 59) -> torch.Tensor:
    """batch = (tok, ln) int tensors -> padded cross-entropy on the next token. ``block`` is the
    model's input length (59: the first 59 moves of a 60-move game)."""
    tok, ln = batch
    x, y = xy_tokens(tok, ln, block)
    lg = model.logits(x)
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=IGNORE)


def _ckpt_schedule(cfg: TrainConfig, steps_per_epoch: float) -> set[int]:
    """Checkpoint steps: log-spaced (ckpt_base, 2 * ckpt_base, ...) plus every epoch."""
    ck = set()
    s = cfg.ckpt_base
    while s < cfg.steps:
        ck.add(s)
        s *= 2
    for e in range(1, int(cfg.steps / steps_per_epoch) + 1):
        ck.add(int(round(e * steps_per_epoch)))
    ck.add(cfg.steps)
    return {c for c in ck if 0 < c <= cfg.steps}


def train(model, source: DataSource, cfg: TrainConfig, run_dir: str | Path, *,
          arch: str, model_config: dict, device: str = "cuda", log=print,
          resume: bool = False, extra_config: dict | None = None) -> dict:
    """Train into ``run_dir``: config.json, metrics.jsonl, best_model.pt, ckpt/step_*.pt and the
    resumable ckpt/latest.pt. ``resume=True`` continues from latest.pt up to ``cfg.steps``, fast-
    forwarding the batch stream with ``source.skip`` when the source has one."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    latest = run_dir / "ckpt" / "latest.pt"
    start_step, best, hist, t_off, resumed, exact = 1, float("inf"), [], 0.0, [], True
    if latest.exists():
        if not resume:
            raise SystemExit(f"{run_dir} already holds a resumable training state ({latest.name}); "
                             f"pass --resume to continue it, or choose a new --run")
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_step, best, hist, t_off = int(ck["step"]) + 1, float(ck["best"]), list(ck["hist"]), float(ck["elapsed_s"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if ck["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng"]["cuda"]])
        np.random.set_state(ck["rng"]["numpy"])
        if source.skip is not None:
            source.skip(int(ck["step"]))
        else:
            exact = False
            log("  warning: source has no skip(); batch order after the resume differs from an uninterrupted run")
        resumed = list(ck.get("resumed", [])) + [{"from_step": int(ck["step"]), "to_steps": cfg.steps,
                                                  "batch_order_exact": exact}]
        log(f"{run_dir.name}: RESUMED at step {start_step:,} (best val so far {best:.6f}), training to {cfg.steps:,}")
    elif resume:
        log(f"{run_dir.name}: --resume given but no {latest.name} yet; starting fresh")
    (run_dir / "config.json").write_text(json.dumps({
        "arch": arch, "model": model_config, "train": dataclasses.asdict(cfg),
        "data": source.meta, "n_params": n_par,
        "steps_per_epoch": source.steps_per_epoch,
        "epochs": cfg.steps / source.steps_per_epoch,
        **({"resumed": resumed} if resumed else {}),
        # `replicate` {"of": "<run id>", "seed", "steps"} marks a seed replicate of a main run
        **(extra_config or {}),
    }, indent=2))

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        if cfg.lr_schedule == "constant":
            return 1.0
        prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    ck_steps = _ckpt_schedule(cfg, source.steps_per_epoch)
    log(f"{run_dir.name}: {n_par:,} params · arch {arch} · {cfg.steps:,} steps "
        f"({cfg.steps / source.steps_per_epoch:.2f} epochs) · {len(ck_steps)} checkpoints")

    def save(path: Path, step: int, va: float | None):
        torch.save({"arch": arch, "step": step, "model_state": model.state_dict(),
                    "model_config": model_config,
                    "train_config": dataclasses.asdict(cfg),
                    "val_loss": va, "epoch": step / source.steps_per_epoch}, path)

    def save_latest(step: int):
        """The resumable state, written to a temp file and renamed so a crash keeps the previous one."""
        (run_dir / "ckpt").mkdir(exist_ok=True)
        tmp = latest.with_suffix(".tmp")
        torch.save({"arch": arch, "step": step, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(), "model_config": model_config,
                    "train_config": dataclasses.asdict(cfg), "best": best, "hist": hist,
                    "elapsed_s": t_off + time.perf_counter() - t0, "resumed": resumed,
                    "rng": {"torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "numpy": np.random.get_state()}}, tmp)
        tmp.replace(latest)

    t0 = time.perf_counter()
    if start_step > cfg.steps:
        log(f"{run_dir.name}: already at step {start_step - 1:,} >= {cfg.steps:,}; nothing to do")
        best_step = min(hist, key=lambda r: r["val_loss"])["step"] if hist else -1
        return {"best_val": best, "best_step": best_step, "minutes": t_off / 60}
    model.train()
    for step in range(start_step, cfg.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = cfg.lr * lr_at(step)
        batch = next(source.batches)
        loss = source.loss_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.val_every == 0 or step == cfg.steps:
            model.eval()
            va = source.validate(model)
            model.train()
            rec = {"step": step, "train_loss": float(loss.item()), "val_loss": float(va),
                   "lr": opt.param_groups[0]["lr"],
                   "elapsed_s": round(t_off + time.perf_counter() - t0, 1)}
            hist.append(rec)
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            mark = ""
            if va < best:
                best, mark = va, "  *"
                save(run_dir / "best_model.pt", step, va)
            save_latest(step)
            log(f"  step {step:>9,}/{cfg.steps:,}  train {loss.item():.6f}  "
                f"val {va:.6f}{mark}  [{(t_off + time.perf_counter() - t0) / 60:.1f} min]")

        # checkpoints run on their own cadence, outside the validation branch
        if step in ck_steps:
            (run_dir / "ckpt").mkdir(exist_ok=True)
            save(run_dir / "ckpt" / f"step_{step:09d}.pt", step,
                 hist[-1]["val_loss"] if hist else None)
            log(f"  [ckpt] step {step:,} (epoch {step / source.steps_per_epoch:.2f})")

    save_latest(cfg.steps)
    best_step = min(hist, key=lambda r: r["val_loss"])["step"] if hist else -1
    out = {"best_val": best, "best_step": best_step,
           "minutes": (t_off + time.perf_counter() - t0) / 60}
    log(f"done {run_dir.name}: best val {best:.6f} at step {best_step:,}/{cfg.steps:,} "
        f"· {out['minutes']:.1f} min")
    return out
