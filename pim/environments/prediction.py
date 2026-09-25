"""A run's held-out predictive loss: the ``prediction`` block of its ``scores.json``.

Each reading is the run's training objective on the instance's held-out split, kept per sequence
so it carries a standard error and pairs with the instance's Bayes floor on the first sequences.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.environments import layout
from pim.metrics import prediction as mp

PRED_VERSION = "1.0"


def _reading(objective: str, unit: str, per_seq: np.ndarray, n_paired: int, **extra) -> dict:
    loss, se = mp.mean_se(per_seq)
    lp, sp = mp.mean_se(per_seq[:n_paired])
    return {"objective": objective, "unit": unit, "loss": loss, "se": se, "n_sequences": int(len(per_seq)),
            "loss_paired": lp, "se_paired": sp, "n_paired": int(min(n_paired, len(per_seq))), **extra}


def _eval_frames(instance: str) -> np.ndarray:
    with h5py.File(layout.eval_file("rayworld", instance), "r") as f:
        return f["obs_intensity"][:].astype(np.float32)


@torch.no_grad()
def rayworld_frames(model, instance: str, n_paired: int, device: str, batch: int = 256) -> dict:
    """A frame model: next-frame MSE per ray."""
    obs = _eval_frames(instance)
    span = int(getattr(model, "state_span", obs.shape[1] - 1))
    x = torch.from_numpy(obs[:, : span + 1])
    per = [mp.next_frame_mse(model(x[i: i + batch, :-1].to(device)).float().cpu().numpy(), x[i: i + batch, 1:].numpy())
           for i in range(0, len(x), batch)]
    return {"frames": _reading("mse", "MSE (intensity² per ray)", np.concatenate(per), n_paired)}


@torch.no_grad()
def rayworld_tokens(model, vocab, instance: str, n_paired: int, device: str, batch: int = 256) -> dict:
    """A frames-as-tokens model, read two ways: next-frame cross-entropy ("tokens") and the MSE of
    the softmax's mean frame ("expected-frame")."""
    from pim.environments.rayworld.tokens import UNK, encode

    obs = _eval_frames(instance)
    tok = encode(obs, vocab).astype(np.int64)
    if (tok == UNK).any():
        raise ValueError(f"{instance}: a held-out frame is outside the run's vocabulary")
    block = obs.shape[1] - 1
    ce, mse, unk, hit, n = [], [], 0.0, 0, 0
    for i in range(0, len(tok), batch):
        t = torch.from_numpy(tok[i: i + batch]).to(device)
        lg = model.logits(t[:, :block]).float().cpu().numpy()                      # (B, block, V)
        y = tok[i: i + batch, 1: block + 1]
        ce.append(mp.next_token_ce(lg, y))
        hit, n = hit + int((lg.argmax(-1) == y).sum()), n + y.size
        mean_frame, dropped = mp.expected_frame(np.exp(mp.log_softmax(lg)), vocab.frames, drop=(UNK,))
        mse.append(mp.next_frame_mse(mean_frame, obs[i: i + batch, 1: block + 1]))
        unk += float(dropped.sum())
    return {"tokens": _reading("ce", "CE (nats per frame)", np.concatenate(ce), n_paired, top1=hit / n),
            "expected-frame": _reading("mse", "MSE (intensity² per ray)", np.concatenate(mse), n_paired,
                                       unk_mass_mean=unk / n)}


def othello_moves(scores: dict) -> dict:
    """Othello: next-move cross-entropy, taken from the run's held-out ``gates``."""
    g = scores["gates"]
    return {"moves": {"objective": "ce", "unit": "CE (nats per move)", "loss": g["ce"], "se": None,
                      "n_sequences": g["n_games"], "loss_paired": g["ce"], "se_paired": None,
                      "n_paired": g["n_games"], "n_positions": g["n_positions"], "top1": g["top1_acc"],
                      "source": "scores.json gates (arms.gates on the test split)"}}


def score_run(run_dir: Path, device: str | None = None) -> dict:
    """The ``prediction`` block for one scored run (its ``scores.json`` must exist)."""
    from pim.models import load_checkpoint

    run_dir = Path(run_dir)
    s = json.loads((run_dir / "scores.json").read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if s["env"] == "othello":
        readings, split = othello_moves(s), "test"
    else:
        from pim.environments.rayworld.bayes import N_FLOOR_SEQ

        model, info = load_checkpoint(run_dir / "best_model.pt", device=device)
        model.eval()
        if info.arch.endswith("_tokens"):
            from pim.environments.rayworld.tokens import FrameVocab

            readings = rayworld_tokens(model, FrameVocab.load(run_dir / "vocab.npz"), s["instance"], N_FLOOR_SEQ, device)
        else:
            readings = rayworld_frames(model, s["instance"], N_FLOOR_SEQ, device)
        split = "eval/test.h5"
    return {"version": PRED_VERSION, "instance": s["instance"], "split": split, "readings": readings}
