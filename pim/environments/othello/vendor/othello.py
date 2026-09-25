"""Othello board rules and the synthetic-game generator.

Modified from Li et al.'s othello_world (https://github.com/likenneth/othello_world,
data/othello.py; MIT, Copyright (c) 2023 Kenneth Li; see LICENSE here). Changes: the
`flip` and `placement` rule parameters (with the `_captures` / `_legal` split they need),
and removal of the championship-data loader, plotting, and unused helpers.
"""

import random

import numpy as np

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]


def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])


eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


def get_ood_game(_, flip=True, placement="enclosure"):
    # a uniformly random legal move at every step, drawn from the global `random` state
    tbr = []
    ab = OthelloBoardState(flip=flip, placement=placement)
    possible_next_steps = ab.get_valid_moves()
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        ab.update([next_step, ])
        possible_next_steps = ab.get_valid_moves()
    return tbr


class OthelloBoardState():
    # 1 is black, -1 is white
    def __init__(self, board_size = 8, flip=True, placement="enclosure"):
        self.board_size = board_size * board_size
        # flip=False: enclosed discs are not recolored; legality, passes and game end are unchanged.
        # placement: "enclosure" (a move must enclose an opponent disc) or "adjacent" (a move must
        # touch one of the mover's own discs in the 8-neighborhood).
        self.flip = flip
        if placement not in ("enclosure", "adjacent"):
            raise ValueError(f"placement must be enclosure|adjacent, got {placement!r}")
        self.placement = placement
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []

    def update(self, moves, prt=False):
        # takes a new move or new moves and update state
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()

    def _captures(self, r, c, color):
        """The discs a `color` disc at (r, c) would recolor under the enclosure rule."""
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        return tbf

    def _legal(self, r, c, color):
        """(legal, discs to recolor) for `color` placing on the empty square (r, c)."""
        if self.placement == "adjacent":
            for direction in eights:
                rr, cc = r + direction[0], c + direction[1]
                if 0 <= rr < 8 and 0 <= cc < 8 and self.state[rr, cc] == color:
                    # adjacency decides legality; with flip on, the placed disc still recolors
                    # what it encloses
                    return True, (self._captures(r, c, color) if self.flip else [])
            return False, []
        tbf = self._captures(r, c, color)
        return len(tbf) > 0, tbf

    def umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        color = self.next_hand_color
        ok, tbf = self._legal(r, c, color)
        if not ok:  # means one hand is forfeited
            color *= -1
            self.next_hand_color *= -1
            ok, tbf = self._legal(r, c, color)
        if not ok:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"

        self.age += 1
        if self.flip:
            for ff in tbf:
                self.state[ff[0], ff[1]] *= -1
                self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)

    def __print__(self, ):
        print("-"*20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
            print(" ".join([a[k]] + tbp))
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-"*20)

    def tentative_move(self, move):
        # tentatively put a piece, do nothing to state
        # returns 0 if this is not a move at all: occupied or both player have to forfeit
        # return 1 if regular move
        # return 2 if forfeit happens but the opponent can drop piece at this place
        r, c = move // 8, move % 8
        if not self.state[r, c] == 0:
            return 0
        color = self.next_hand_color
        if self._legal(r, c, color)[0]:
            return 1
        # means one hand is forfeited
        if self._legal(r, c, -color)[0]:
            return 2
        return 0

    def get_valid_moves(self, ):
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
            else:
                pass
        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
