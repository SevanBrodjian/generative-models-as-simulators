"""Othello data: vocabulary, index-seeded synthetic games, board-state labels, activation harvest.

Conventions follow Li et al.: the model reads the first 59 moves of a game; the activation at
position t pairs with the board after move t; board encoding 0 = white, 1 = blank, 2 = black.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from pim.environments.othello.vendor.othello import OthelloBoardState, get_ood_game

# The four center squares are occupied at t=0 and are never a legal move, so they carry no
# token and are padded back in when move probabilities are laid out on the board.
CENTER = (27, 28, 35, 36)
N_TILES = 64
N_CLASSES = 3  # white / blank / black
MAXLEN = N_TILES - len(CENTER)   # 60: a game can fill every non-center square
T_MODEL = MAXLEN - 1             # 59: the model's input block
BLANK, MINE, THEIRS = 0, 1, 2    # the mine/theirs frame (`ProbeData.mine`, the bench targets)


# ── vocabulary ────────────────────────────────────────────────────────────────


def canonical_vocab() -> dict[int, int]:
    """``{board square -> token}`` over the 60 playable squares, with the pad symbol -100 at token 0."""
    squares = [s for s in range(N_TILES) if s not in CENTER]
    return {-100: 0, **{sq: i + 1 for i, sq in enumerate(sorted(squares))}}


VOCAB = len(canonical_vocab())   # 61: the 60 playable squares + the pad token 0


# ── synthetic games ───────────────────────────────────────────────────────────


def _one_game(args) -> list[int]:
    """One game from ``(index, seed[, flip[, placement]])``, uniform over legal moves at every step.

    Seeding per game index makes a corpus a pure function of ``seed`` and the index range,
    independent of worker scheduling."""
    i, seed, *rest = args
    flip = rest[0] if rest else True
    placement = rest[1] if len(rest) > 1 else "enclosure"
    random.seed(seed * 1_000_003 + i)
    return get_ood_game(i, flip=flip, placement=placement)


# ── tokens and board-state labels ─────────────────────────────────────────────


@dataclass
class ProbeData:
    tokens: np.ndarray  # (N, 59) int64, pad = 0
    labels: np.ndarray  # (N, 59, 64) int8 — white 0 / blank 1 / black 2
    mine: np.ndarray  # (N, 59, 64) int8 — blank 0 / mine 1 / theirs 2
    mask: np.ndarray  # (N, 59) bool — True where the position is a real move
    lengths: np.ndarray  # (N,) int


def tokens_and_labels(games: list[list[int]], flip: bool = True,
                      placement: str = "enclosure") -> ProbeData:
    """Tokenize the first 59 moves of each game and label the board after every move.

    Replay with the rules that generated the games (``corpus.rules_of(instance)``)."""
    stoi = canonical_vocab()
    n, T = len(games), T_MODEL
    tokens = np.zeros((n, T), np.int64)
    labels = np.full((n, T, N_TILES), 1, np.int8)
    mine = np.zeros((n, T, N_TILES), np.int8)
    mask = np.zeros((n, T), bool)
    lengths = np.zeros(n, int)

    for i, g in enumerate(games):
        moves = g[:T]
        lengths[i] = len(moves)
        tokens[i, : len(moves)] = [stoi[s] for s in moves]
        mask[i, : len(moves)] = True
        board = OthelloBoardState(flip=flip, placement=placement)
        for t, mv in enumerate(moves):
            board.umpire(mv)
            st = (board.state + 1).flatten().astype(np.int8)  # white 0 / blank 1 / black 2
            labels[i, t] = st
            # "mine" = the player about to move; next_hand_color is not move parity because of passes
            nxt = 2 if board.next_hand_color > 0 else 0
            mine[i, t] = np.where(st == 1, BLANK, np.where(st == nxt, MINE, THEIRS))
    return ProbeData(tokens, labels, mine, mask, lengths)


def flatten_rows(data: ProbeData, target: str = "state") -> tuple[np.ndarray, np.ndarray]:
    """(sequence index per row, label per row) over the real positions. ``target``: "state"
    (absolute color) or "mine" (mine/theirs), both 3-way."""
    if target not in ("state", "mine"):
        raise ValueError(f"target must be 'state' or 'mine', got {target!r}")
    y = data.labels if target == "state" else data.mine
    seq_idx = np.repeat(np.arange(len(data.tokens))[:, None], data.tokens.shape[1], 1)
    return seq_idx[data.mask], y[data.mask]


# ── activation harvest ────────────────────────────────────────────────────────


@torch.no_grad()
def harvest_point(model, tokens: np.ndarray, point: int, batch: int = 512) -> np.ndarray:
    """(N, 59, d_model) residual stream at one residual point."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(tokens), batch):
        idx = torch.from_numpy(tokens[i : i + batch]).to(dev)
        rs = model.residual_stack(idx)
        out.append(rs[point].float().cpu().numpy())
    return np.concatenate(out, 0)


# ── logits → board ────────────────────────────────────────────────────────────


def move_probs(outputs: torch.Tensor) -> torch.Tensor:
    """(..., 61) logits → (..., 60) next-move probabilities: drop the pad logit, softmax."""
    return torch.softmax(outputs[..., 1:], dim=-1)


def board_probs(outputs: torch.Tensor) -> np.ndarray:
    """(B, 61) logits → (B, 64) next-move probabilities on the board (zeros on the center squares)."""
    p = move_probs(outputs)
    pad = torch.zeros(len(p), 2, device=p.device, dtype=p.dtype)
    out = torch.cat([p[:, :27], pad, p[:, 27:33], pad, p[:, 33:]], dim=1)
    return out.float().cpu().numpy()
