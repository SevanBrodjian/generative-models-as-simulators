"""The Othello edit bench: single-tile color flips on real game prefixes.

Each case is a game prefix plus one occupied square whose color the editor must flip. Every
instance synthesizes its own cases from its held-out edits games (``synthesize_cases``,
``scripts/make_othello_edits.py``).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pim.environments.othello.data import CENTER, MINE, THEIRS, canonical_vocab
from pim.environments.othello.vendor.othello import OthelloBoardState


@dataclass
class Benchmark:
    tokens: list[np.ndarray]  # per bucket, (B, L) int64
    case_ids: list[np.ndarray]  # per bucket, indices into the case list
    pos_int: np.ndarray  # (N,) intervened square
    new_class: np.ndarray  # (N,) requested absolute color, `2 - ori_color`
    legal_pre: list[list[int]]
    legal_post: list[list[int]]
    cur_lab: np.ndarray  # (N,) the intervened tile's current label, mine/theirs frame
    tgt_lab: np.ndarray  # (N,) the label the edit asks for (the flip of cur_lab)

    @property
    def n_cases(self) -> int:
        return len(self.pos_int)


def benchmark_from_cases(cases: list[dict], flip: bool = True, placement: str = "enclosure") -> Benchmark:
    """A Benchmark from ``{history, pos_int, ori_color}`` cases, replayed under the given rules.

    Cases are bucketed by history length: the editors write ``x[:, -1]``, so every row of a
    batch must end its real moves at the same index.
    """
    stoi = canonical_vocab()

    pos_int = np.array([c["pos_int"] for c in cases], int)
    new_class = np.array([int(2 - c["ori_color"]) for c in cases], int)
    legal_pre, legal_post = [], []
    cur = np.zeros(len(cases), np.int64)
    for i, (c, sq, new) in enumerate(zip(cases, pos_int, new_class)):
        pre = OthelloBoardState(flip=flip, placement=placement)
        pre.update(c["history"], prt=False)
        legal_pre.append(sorted(pre.get_valid_moves()))
        # the player to move does not change, so in the mover's frame the tile reads MINE iff
        # its color is the next hand's, and the flip is exactly MINE <-> THEIRS
        nxt = 2 if pre.next_hand_color > 0 else 0
        cur[i] = MINE if c["ori_color"] == nxt else THEIRS
        post = OthelloBoardState(flip=flip, placement=placement)
        post.update(c["history"], prt=False)
        post.state[sq // 8, sq % 8] = new - 1
        legal_post.append(sorted(post.get_valid_moves()))
    tgt = np.where(cur == MINE, THEIRS, MINE)

    by_len: dict[int, list[int]] = {}
    for i, c in enumerate(cases):
        by_len.setdefault(len(c["history"]), []).append(i)
    toks, ids = [], []
    for L in sorted(by_len):
        members = np.array(by_len[L], int)
        ids.append(members)
        toks.append(np.array([[stoi[s] for s in cases[i]["history"]] for i in members],
                             np.int64))
    return Benchmark(toks, ids, pos_int, new_class, legal_pre, legal_post, cur, tgt)


def cases_path(instance: str) -> Path:
    """The instance's edit cases file under ``datasets/othello/<instance>/edits/``."""
    from pim.environments.layout import othello_cases_file

    return othello_cases_file(instance)


def load_benchmark(instance: str = "standard") -> Benchmark:
    """The instance's edit cases, bucketed and replayed with the instance's rules."""
    from pim.environments.othello.corpus import rules_of

    with open(cases_path(instance), "rb") as f:
        return benchmark_from_cases(pickle.load(f), **rules_of(instance))


def synthesize_cases(histories: list[list[int]], n: int, length_counts: dict[int, int],
                     seed: int = 0, flip: bool = True, placement: str = "enclosure",
                     log=print) -> tuple[list[dict], dict]:
    """Edit cases from held-out games: a game prefix plus one occupied, non-center square
    flipped to the opposite color, rejected if the flip leaves the legal set unchanged or
    empties it. ``length_counts`` sets how many cases each prefix length gets (rescaled to
    ``n``; the paper uses ``{20: n}``). Returns the cases and a manifest.
    """
    rng = np.random.default_rng(seed)
    tot = sum(length_counts.values())
    quota = {L: int(round(n * c / tot)) for L, c in sorted(length_counts.items())}
    cases, stats = [], {}
    for L, want in quota.items():
        pool = np.array([i for i, h in enumerate(histories) if len(h) > L])
        got = tried = rej_same = rej_empty = 0
        for g in rng.permutation(pool):
            if got >= want:
                break
            tried += 1
            h = list(histories[g][:L])
            board = OthelloBoardState(flip=flip, placement=placement)
            board.update(h, prt=False)
            pre = sorted(board.get_valid_moves())
            if not pre:
                continue
            occ = [sq for sq in range(64) if board.state[sq // 8, sq % 8] != 0 and sq not in CENTER]
            for sq in rng.permutation(occ):
                sq = int(sq)
                ori = 0.0 if board.state[sq // 8, sq % 8] < 0 else 2.0
                post = OthelloBoardState(flip=flip, placement=placement)
                post.update(h, prt=False)
                post.state[sq // 8, sq % 8] = int(2 - ori) - 1
                legal_post = sorted(post.get_valid_moves())
                if not legal_post:
                    rej_empty += 1
                    continue
                if legal_post == pre:
                    rej_same += 1
                    continue
                cases.append({"history": h, "pos_int": sq, "ori_color": ori, "game": int(g)})
                got += 1
                break
        stats[int(L)] = {"want": want, "n": got, "games_tried": tried, "pool": int(len(pool)),
                         "rejected_same_legal": rej_same, "rejected_empty_legal": rej_empty}
        if log:
            log(f"  prefix {L:2d}: {got}/{want} cases from {tried} games "
                f"(rejected same-legal {rej_same}, empty {rej_empty})", flush=True)
    manifest = {"n_cases": len(cases), "seed": seed, "flip": flip, "placement": placement,
                "length_quota": quota,
                "recipe": "prefix of a held-out game; one uniformly random occupied non-center "
                          "square flipped to the opposite color; rejected if the legal set is "
                          "unchanged or empty",
                "stats": stats}
    return cases, manifest


def case_targets(bench: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    """Per case: the intervened tile's current and target label in mine/theirs coordinates."""
    return bench.cur_lab, bench.tgt_lab
