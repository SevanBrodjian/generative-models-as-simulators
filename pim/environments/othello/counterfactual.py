"""Game replay helpers: the board after a move list, and how often a rule set recolors discs."""
from __future__ import annotations

import numpy as np

from pim.environments.othello.vendor.othello import OthelloBoardState


def replay(h, rules):
    """The board after the move list ``h`` under ``rules`` (``corpus.rules_of``), or None if illegal."""
    b = OthelloBoardState(**rules)
    try:
        b.update(h, prt=False)
    except AssertionError:
        return None
    return b


def flips_per_move(tokens: np.ndarray, lengths: np.ndarray, rules: dict) -> dict:
    """Discs recolored per move and per game over a set of tokenized games."""
    from pim.environments.othello.data import canonical_vocab

    itos = {v: k for k, v in canonical_vocab().items()}
    n_flipped = n_moves = 0
    for row, L in zip(tokens, lengths):
        b = OthelloBoardState(**rules)
        for t in range(int(L)):
            before = b.state.copy()
            b.umpire(itos[int(row[t])])
            n_flipped += int(((before != 0) & (before != b.state)).sum())
            n_moves += 1
    return {"n_games": int(len(tokens)), "n_moves": int(n_moves), "n_flipped": int(n_flipped),
            "flips_per_move": n_flipped / n_moves, "flips_per_game": n_flipped / len(tokens)}
