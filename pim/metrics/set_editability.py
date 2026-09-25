"""Editability of a predicted distribution: each world is a set of tokens, its reference the uniform
distribution over the set. Othello: the legal moves before and after the edit (the generator's exact
distribution). The Rayworld token model: the one frame each world renders (the frame-set Edit Index).
"""

from __future__ import annotations

import numpy as np

from pim.metrics.edit_index import (case_stats, edit_index_per_case, fidelity_ratio_from,
                                    masked_rmse_per_case, ratio_ci95)

N_TILES = 64  # 8x8 board; probability vectors over squares are laid out row-major

__all__ = ["N_TILES", "li_error", "uniform_over_legal", "edit_index_legal",
           "move_scorecard", "move_rmse", "move_fidelity_ratio"]


def li_error(probs: np.ndarray, legal: list[list[int]]) -> np.ndarray:
    """Li et al. §4.2 error: the top-N predictions against the N legal moves, false positives
    plus false negatives, ``2 * (N - overlap)``. NaN where the legal set is empty."""
    out = np.full(len(probs), np.nan)
    for i, L in enumerate(legal):
        if not L:
            continue
        top = set(np.argsort(-probs[i])[: len(L)].tolist())
        out[i] = 2 * (len(L) - len(top & set(L)))
    return out


def uniform_over_legal(legal: list[int], n_tiles: int = N_TILES) -> np.ndarray:
    """(n_tiles,) uniform distribution over ``legal``: the reference for one world.

    ``n_tiles`` defaults to the 64 Othello squares; callers pass the vocabulary size.
    """
    v = np.zeros(n_tiles, np.float32)
    if legal:
        v[list(legal)] = 1.0 / len(legal)
    return v


def edit_index_legal(
    probs: np.ndarray,
    legal_pre: list[list[int]],
    legal_post: list[list[int]],
    support: str = "union",
) -> np.ndarray:
    """(n,) legal-set Edit Index against uniform references over ``legal_pre`` and ``legal_post``, on the
    squares legal in either world (``"union"``) or only those whose legality changed (``"symdiff"``,
    the reported construction)."""
    n, V = probs.shape
    ref_uned = np.stack([uniform_over_legal(L, V) for L in legal_pre])
    ref_edit = np.stack([uniform_over_legal(L, V) for L in legal_post])
    supp = np.zeros((n, V), bool)                                          # where they differ
    for i, (L0, L1) in enumerate(zip(legal_pre, legal_post)):
        s0, s1 = set(L0), set(L1)
        supp[i, list(s0 | s1 if support == "union" else s0 ^ s1)] = True
    return edit_index_per_case(probs, ref_edit, ref_uned, supp)


def move_scorecard(
    probs: np.ndarray,
    legal_pre: list[list[int]],
    legal_post: list[list[int]],
) -> dict:
    """Every number an edit arm reports on a distribution model: Li error against both worlds,
    both Edit Index supports with their case-level spread, legal mass, and the per-case RMSE
    against the edited world that ``move_fidelity_ci95`` resamples."""
    e_post = li_error(probs, legal_post)
    e_pre = li_error(probs, legal_pre)
    ei_u = edit_index_legal(probs, legal_pre, legal_post, "union")
    ei_s = edit_index_legal(probs, legal_pre, legal_post, "symdiff")
    return {
        "li_error_vs_post": float(np.nanmean(e_post)),
        "li_error_vs_pre": float(np.nanmean(e_pre)),
        "edit_index_union": float(np.nanmean(ei_u)),
        "edit_index_symdiff": float(np.nanmean(ei_s)),
        "legal_mass": float(
            np.mean([probs[i, L].sum() for i, L in enumerate(legal_post) if L])
        ),
        "n_scored": int(np.isfinite(e_post).sum()),
        "li_error_vs_post_per_case": e_post.tolist(),
        "edit_index_union_per_case": ei_u.tolist(),
        # per-case lists never become scalar arm fields in scores.json
        **case_stats(ei_u, prefix="edit_index_union_"),
        **case_stats(ei_s, prefix="edit_index_symdiff_"),
        "rmse_post_per_case": move_rmse_per_case(probs, legal_post).tolist(),
    }


def move_rmse(probs: np.ndarray, legal: list[list[int]]) -> float:
    """Mean per-case RMSE between the predicted distribution and uniform-over-legal, over the
    whole vocabulary so the fidelity ratio sees damage outside the Edit Index's support."""
    return float(np.nanmean(move_rmse_per_case(probs, legal)))


def move_rmse_per_case(probs: np.ndarray, legal: list[list[int]]) -> np.ndarray:
    """(n_cases,): the distances ``move_rmse`` averages; NaN where the legal set is empty."""
    ref = np.stack([uniform_over_legal(L, probs.shape[1]) for L in legal])
    mask = np.array([[bool(L)] * probs.shape[1] for L in legal])   # all V squares, or none
    return masked_rmse_per_case(probs, ref, mask)


def move_fidelity_ratio(probs_edited: np.ndarray, probs_unsteered: np.ndarray,
                        legal_post: list[list[int]]) -> float:
    """The fidelity ratio of a distribution model: ``move_rmse`` of the edited prediction over
    that of the unedited prediction, both against the edited world. Above 1 is degraded.

    RMSE rather than ``li_error``, which is a top-N set mismatch blind to probability mass.
    """
    return fidelity_ratio_from(move_rmse(probs_edited, legal_post),
                               move_rmse(probs_unsteered, legal_post))


def move_fidelity_ci95(probs_edited: np.ndarray, probs_unsteered: np.ndarray,
                       legal_post: list[list[int]]) -> dict:
    """Percentile-bootstrap 95% interval of ``move_fidelity_ratio`` over the bench cases
    (``edit_index.ratio_ci95``), as ``fidelity_ci95_lo`` / ``fidelity_ci95_hi``."""
    lo, hi = ratio_ci95(move_rmse_per_case(probs_edited, legal_post),
                        move_rmse_per_case(probs_unsteered, legal_post))
    return {"fidelity_ci95_lo": lo, "fidelity_ci95_hi": hi}
