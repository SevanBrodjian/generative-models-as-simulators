"""Exact reachability of an edit case's flipped board: does a legal game of the same length under the
instance's rules end on exactly that board, same player to move? A depth-first search over a bitboard
copy of the vendored engine returns ``reachable`` (with a witness game, replayed through the engine),
``unreachable`` (search exhausted) or ``undecided`` (node budget ran out)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pim.environments.othello.counterfactual import replay

VERSION = "1.0"
FULL = (1 << 64) - 1
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# the squares along each ray from each square, nearest first
_RAYS = [[[(r + dr * k) * 8 + (c + dc * k) for k in range(1, 8) if 0 <= r + dr * k < 8 and 0 <= c + dc * k < 8]
          for dr, dc in _DIRS] for r in range(8) for c in range(8)]
# the 8-neighborhood of each square as a bitboard
_NBR = [sum(1 << ((r + dr) * 8 + (c + dc)) for dr, dc in _DIRS if 0 <= r + dr < 8 and 0 <= c + dc < 8)
        for r in range(8) for c in range(8)]
# per square, the 4 lines through it as (squares outward on one side, squares outward on the other)
_LINES = [[([(r + dr * k) * 8 + (c + dc * k) for k in range(1, 8) if 0 <= r + dr * k < 8 and 0 <= c + dc * k < 8],
            [(r - dr * k) * 8 + (c - dc * k) for k in range(1, 8) if 0 <= r - dr * k < 8 and 0 <= c - dc * k < 8])
           for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))] for r in range(8) for c in range(8)]

# (black, white, mover): bitboards and 1 = black / -1 = white to move, the vendor's start position
START = ((1 << 28) | (1 << 35), (1 << 27) | (1 << 36), 1)


def _captures(sq: int, own: int, opp: int) -> int:
    """The discs a disc placed on ``sq`` encloses (the vendor's ``_captures`` scan) as a bitboard."""
    out = 0
    for ray in _RAYS[sq]:
        buf = 0
        for s in ray:
            b = 1 << s
            if opp & b:
                buf |= b
            elif own & b:
                out |= buf
                break
            else:
                break
    return out


def _legal(sq: int, own: int, opp: int, placement: str) -> bool:
    if placement == "adjacent":
        return bool(_NBR[sq] & own)
    return _captures(sq, own, opp) != 0


def play(state: tuple, sq: int, rules: dict) -> tuple:
    """The vendor's ``umpire``: the mover plays ``sq`` if legal for them, else the opponent does (a pass)."""
    black, white, mover = state
    own, opp = (black, white) if mover == 1 else (white, black)
    color = mover
    if not _legal(sq, own, opp, rules["placement"]):
        color = -mover
        own, opp = opp, own
        if not _legal(sq, own, opp, rules["placement"]):
            raise ValueError(f"square {sq} is legal for neither player")
    fl = _captures(sq, own, opp) if rules["flip"] else 0
    own, opp = own | fl | (1 << sq), opp & ~fl
    b, w = (own, opp) if color == 1 else (opp, own)
    return b, w, -color


def valid_moves(state: tuple, rules: dict, allowed: int = FULL) -> list[int]:
    """The vendor's ``get_valid_moves`` (the mover's moves, else the opponent's after a forced
    pass), restricted to the squares in ``allowed``. Returns [] when the mover has a regular move
    outside ``allowed``: no pass is permitted then, and every move leaves ``allowed``."""
    black, white, mover = state
    own, opp = (black, white) if mover == 1 else (white, black)
    empty = ~(black | white) & FULL
    reg = [s for s in range(64) if (allowed >> s) & 1 and (empty >> s) & 1 and _legal(s, own, opp, rules["placement"])]
    if reg:
        return reg
    rest = empty & ~allowed
    if any((rest >> s) & 1 and _legal(s, own, opp, rules["placement"]) for s in range(64)):
        return []
    return [s for s in range(64) if (allowed >> s) & 1 and (empty >> s) & 1 and _legal(s, opp, own, rules["placement"])]


def state_of(board) -> tuple:
    """(black, white, mover) of a vendor ``OthelloBoardState``."""
    st = board.state.flatten()
    return (sum(1 << i for i in range(64) if st[i] == 1), sum(1 << i for i in range(64) if st[i] == -1),
            int(board.next_hand_color))


def _flippable(q: int, empty_tgt: int, occ: int) -> bool:
    """Necessary condition for the disc on ``q`` to be captured later: some line through it has an
    empty target square on one side, reached through target squares only, and a target square right
    beyond it on the other."""
    for side_a, side_b in _LINES[q]:
        for side_p, side_x in ((side_a, side_b), (side_b, side_a)):
            if not side_x or not (occ >> side_x[0]) & 1:
                continue
            for s in side_p:
                if not (occ >> s) & 1:
                    break
                if (empty_tgt >> s) & 1:
                    return True
    return False


@dataclass
class Verdict:
    status: str                              # "reachable" | "unreachable" | "undecided"
    nodes: int
    witness: list[int] | None = field(default=None)


class _Budget(Exception):
    pass


def search(target: tuple, n_moves: int, rules: dict, *, order: dict | None = None,
           budget: int = 2_000_000) -> Verdict:
    """Exhaustive search for a game of ``n_moves`` moves from the start position that ends in the
    state ``target`` = (black, white, mover). ``order`` ranks squares for move ordering only."""
    # a game ending on the target places one disc on each of its occupied squares, so only those are tried
    occ = target[0] | target[1]
    tb, tw = target[0], target[1]
    order = order or {}
    dead: set = set()
    path: list[int] = []
    nodes = 0

    def dfs(state: tuple, depth: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise _Budget
        if depth == n_moves:
            return state == target
        # the occupied count fixes the depth, so the position alone keys a failure
        key = (state[0] << 65) | (state[1] << 1) | (state[2] > 0)
        if key in dead:
            return False
        for s in sorted(valid_moves(state, rules, occ), key=lambda x: order.get(x, 99)):
            nxt = play(state, s, rules)
            if not rules["flip"]:
                # a placed disc keeps its color forever, so it must be placed with its target color
                if ((nxt[0] >> s) & 1) != ((tb >> s) & 1):
                    continue
            else:
                # every disc with the wrong color must still be flippable
                empty_tgt = occ & ~(nxt[0] | nxt[1])
                wrong = (nxt[0] & tw) | (nxt[1] & tb)
                ok = True
                while wrong:
                    q = (wrong & -wrong).bit_length() - 1
                    wrong &= wrong - 1
                    if not _flippable(q, empty_tgt, occ):
                        ok = False
                        break
                if not ok:
                    continue
            path.append(s)
            if dfs(nxt, depth + 1):
                return True
            path.pop()
        dead.add(key)
        return False

    try:
        found = dfs(START, 0)
    except _Budget:
        return Verdict("undecided", nodes)
    return Verdict("reachable", nodes, list(path)) if found else Verdict("unreachable", nodes)


def flipped_target(history: list[int], square: int, rules: dict) -> tuple:
    """The edit case's target state: the prefix's board with ``square``'s color flipped, same mover."""
    b0 = replay(history, rules)
    if b0 is None:
        raise ValueError("the history is not a legal game under these rules")
    black, white, mover = state_of(b0)
    bit = 1 << square
    if not ((black | white) & bit):
        raise ValueError(f"square {square} is empty: nothing to flip")
    return black ^ bit, white ^ bit, mover


def decide(history: list[int], square: int, rules: dict, *, budget: int = 2_000_000) -> Verdict:
    """Is the edit case (``history``, flip ``square``) reachable? A reachable verdict's witness is
    replayed through the vendored engine and must land exactly on the target."""
    target = flipped_target(history, square, rules)
    v = search(target, len(history), rules, order={m: i for i, m in enumerate(history)}, budget=budget)
    if v.status == "reachable":
        b = replay(v.witness, rules)
        if b is None or state_of(b) != target:
            raise AssertionError("witness failed the vendored engine's replay")
    return v


def decide_case(args: tuple) -> dict:
    """Pool-friendly wrapper: (index, history, square, rules, budget) -> a JSON-ready record."""
    i, history, square, rules, budget = args
    v = decide(list(history), int(square), rules, budget=budget)
    return {"case": int(i), "status": v.status, "nodes": int(v.nodes), "witness": v.witness}


def summarize(records: list[dict]) -> dict:
    """Counts per verdict."""
    st = np.array([r["status"] for r in records])
    return {k: int((st == k).sum()) for k in ("reachable", "unreachable", "undecided")}
