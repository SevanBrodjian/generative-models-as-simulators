"""The exact Bayes floor of next-move prediction on an Othello instance.

The generator draws each move uniformly from the legal set, so the Bayes-optimal cross-entropy
at a position is exactly log |legal|. Positions are counted as ``arms.gates`` counts them
(position t predicts move t+1; a position with no legal move is skipped). Written to
``runs/_baselines/othello/<instance>/bayes_floor.json`` by ``scripts/bayes_floor.py``.
"""
from __future__ import annotations

import math

import numpy as np

from pim.environments.othello import corpus as oc
from pim.environments.othello.arms import legal_sets

FLOOR_VERSION = "1.0"
N_GAMES = 10_000            # the test games the gates read (SETTINGS["oth_gates_games"])


def exact_ce_floor(tokens: np.ndarray, lengths: np.ndarray, flip: bool = True,
                   placement: str = "enclosure") -> tuple[float, float, int]:
    """(mean log |legal|, mean 1/|legal| (the best achievable top-1), n positions)."""
    legal = legal_sets(tokens, lengths, flip, placement)
    ce, top1, n = 0.0, 0.0, 0
    for g, L in zip(legal, lengths):
        for t in range(int(L) - 1):
            if g[t]:
                ce += math.log(len(g[t]))
                top1 += 1.0 / len(g[t])
                n += 1
    return ce / n, top1 / n, n


def trivial_ce(tokens: np.ndarray, lengths: np.ndarray, fit_tokens: np.ndarray, fit_lengths: np.ndarray,
               flip: bool = True, placement: str = "enclosure", alpha: float = 0.5) -> tuple[float, int]:
    """The best constant next-move distribution (the move frequencies of ``fit_tokens``, with
    add-``alpha`` smoothing over the 60 squares), scored on the positions ``exact_ce_floor`` counts."""
    cnt = np.zeros(61)
    for row, L in zip(fit_tokens, fit_lengths):
        cnt += np.bincount(row[: int(L)], minlength=61)[:61]
    logp = np.log((cnt[1:] + alpha) / (cnt[1:].sum() + 60 * alpha))   # token k is index k - 1
    legal = legal_sets(tokens, lengths, flip, placement)
    ce, n = 0.0, 0
    for g, row, L in zip(legal, tokens, lengths):
        for t in range(int(L) - 1):
            if g[t]:
                ce -= logp[int(row[t + 1]) - 1]
                n += 1
    return ce / n, n


def bayes_floor(instance: str, n_games: int = N_GAMES) -> dict:
    tok, ln = oc.load(oc.build(oc.N_TRAIN_GAMES, log=lambda s: None, only=("test",), instance=instance)["test"])
    tok, ln = tok[:n_games], ln[:n_games]
    ce, top1, n = exact_ce_floor(tok, ln, **oc.rules_of(instance))
    ftok, fln = oc.load(oc.build(oc.N_TRAIN_GAMES, log=lambda s: None, only=("probe",), instance=instance)["probe"])
    tce, tn = trivial_ce(tok, ln, ftok, fln, **oc.rules_of(instance))
    assert tn == n
    return {"instance": instance, "version": FLOOR_VERSION,
            "method": "exact: the generator is uniform over the legal set, so the floor is E[log |legal|]",
            "split": "test", "n_sequences": int(len(tok)), "sequences": f"the first {len(tok)}", "n_positions": int(n),
            "ce": {"exact": ce, "unit": "CE (nats per move)"}, "bayes_top1": top1,
            "trivial": {"fit_on": "the instance's probe games", "n_sequences": int(len(tok)), "mse": None,
                        "ce": {"value": tce, "se": None, "definition": "the constant next-move distribution = the probe "
                               "games' move frequencies (add-0.5 smoothing); log 60 = 4.094 would be the uniform one"}}}
