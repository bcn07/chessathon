"""The python-chess engine: alpha-beta search with PeSTO evaluation, ~20 knps.

Ready the instant it is imported, which is why it exists: agent.py plays it while the numba
engine compiles, and falls back to it if the native module ever fails to load.

Search   iterative deepening, fail-soft negamax with alpha-beta, a transposition table that
         survives between moves, null-move pruning, late move reductions, a check extension and
         a captures-only quiescence search. Moves are ordered TT move, MVV-LVA captures,
         promotions, killers, history.
Eval     PeSTO tapered piece values and piece-square tables, passed pawns, endgame mop-up,
         bishop pair and tempo.
Time     a soft budget decides whether another iteration starts, a hard budget aborts one.
State    agent.py rebuilds the game from the FENs it is handed and passes repetition keys in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import chess

# ------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)
# Search ordering keeps the starting engine's coarse exchange values. Evaluation uses PeSTO's
# separately tuned middlegame and endgame values below.
PIECE_VALUE = (0, 100, 320, 330, 500, 900, 0)
MG_VALUE = (0, 82, 337, 365, 477, 1025, 0)
EG_VALUE = (0, 94, 281, 297, 512, 936, 0)
PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)
PHASE_TOTAL = 24
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 30, 50
TEMPO = 10

# Ronald Friederich's PeSTO tables, as published by the Chess Programming Wiki. Tables read like
# a diagram, a8 first and h1 last. White pieces look them up mirrored.
# fmt: off
_PAWN_MG = (
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_PAWN_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_KNIGHT_MG = (
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
)
_KNIGHT_EG = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)
_BISHOP_MG = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)
_BISHOP_EG = (
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
)
_ROOK_MG = (
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
)
_ROOK_EG = (
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
)
_QUEEN_MG = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)
_QUEEN_EG = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
)
_KING_MG = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)
_KING_EG = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
)
# fmt: on


def _tables(diagram: tuple[int, ...], value: int) -> tuple[list[int], list[int]]:
    """Per-square scores with material folded in, as (black table, white table)."""
    black = [value + diagram[square] for square in range(64)]
    white = [value + diagram[square ^ 56] for square in range(64)]
    return black, white


def _build(
    diagrams: tuple[tuple[int, ...], ...], values: tuple[int, ...]
) -> list[list[list[int]]]:
    tables: list[list[list[int]]] = [[[]] * 7, [[]] * 7]
    for piece_type, diagram in enumerate(diagrams, start=1):
        tables[chess.BLACK][piece_type], tables[chess.WHITE][piece_type] = _tables(
            diagram, values[piece_type]
        )
    return tables


PST_MG = _build(
    (_PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_MG), MG_VALUE
)
PST_EG = _build(
    (_PAWN_EG, _KNIGHT_EG, _BISHOP_EG, _ROOK_EG, _QUEEN_EG, _KING_EG), EG_VALUE
)

# Bonuses start at a pawn's home rank and grow as it approaches promotion. The front-span mask
# includes its own and adjacent files, so no runtime attack generation is needed.
PASSED_PAWN_MG = (0, 0, 5, 10, 20, 35, 60, 0)
PASSED_PAWN_EG = (0, 5, 10, 25, 50, 90, 150, 0)


def _passed_masks() -> list[list[int]]:
    masks = [[0] * 64 for _ in range(2)]
    for color in (chess.WHITE, chess.BLACK):
        for square in chess.SQUARES:
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            ahead = range(rank_index + 1, 8) if color else range(rank_index)
            mask = 0
            for front_rank in ahead:
                for front_file in range(max(0, file_index - 1), min(8, file_index + 2)):
                    mask |= chess.BB_SQUARES[chess.square(front_file, front_rank)]
            masks[color][square] = mask
    return masks


PASSED_MASKS = _passed_masks()

# The classic mop-up formula uses distance from the four centre squares and proximity between
# kings. Its small score only breaks otherwise aimless winning-endgame choices.
MOP_EDGE = tuple(
    round(
        4.7
        * (
            abs(chess.square_file(square) - 3.5)
            + abs(chess.square_rank(square) - 3.5)
        )
    )
    for square in chess.SQUARES
)


def _mop_up(board: chess.Board) -> int:
    """White-relative guidance when one king is nearly bare against a heavy piece."""
    for weak in (chess.WHITE, chess.BLACK):
        weak_extra = board.occupied_co[weak] & ~board.kings
        if weak_extra & board.pawns:
            continue
        if weak_extra and (
            weak_extra.bit_count() != 1 or not weak_extra & (board.knights | board.bishops)
        ):
            continue
        strong = not weak
        if not board.occupied_co[strong] & (board.rooks | board.queens):
            continue
        weak_king = board.king(weak)
        strong_king = board.king(strong)
        assert weak_king is not None and strong_king is not None
        king_distance = abs(chess.square_file(weak_king) - chess.square_file(strong_king)) + abs(
            chess.square_rank(weak_king) - chess.square_rank(strong_king)
        )
        bonus = MOP_EDGE[weak_king] + round(1.6 * (14 - king_distance))
        return bonus if strong == chess.WHITE else -bonus
    return 0


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns from the side to move's point of view."""
    mg = eg = phase = 0
    for color in (chess.WHITE, chess.BLACK):
        tables_mg = PST_MG[color]
        tables_eg = PST_EG[color]
        passed_masks = PASSED_MASKS[color]
        enemy_pawns = board.pieces_mask(PAWN, not color)
        side_mg = side_eg = 0
        for piece_type in range(1, 7):
            bb = board.pieces_mask(piece_type, color)
            if not bb:
                continue
            phase += PHASE_WEIGHT[piece_type] * bb.bit_count()
            if piece_type == BISHOP and bb & (bb - 1):
                side_mg += BISHOP_PAIR_MG
                side_eg += BISHOP_PAIR_EG
            table_mg = tables_mg[piece_type]
            table_eg = tables_eg[piece_type]
            while bb:
                square = bb.bit_length() - 1
                bb ^= 1 << square
                side_mg += table_mg[square]
                side_eg += table_eg[square]
                if piece_type == PAWN and not enemy_pawns & passed_masks[square]:
                    relative_rank = chess.square_rank(square)
                    if color == chess.BLACK:
                        relative_rank = 7 - relative_rank
                    side_mg += PASSED_PAWN_MG[relative_rank]
                    side_eg += PASSED_PAWN_EG[relative_rank]
        if color:
            mg += side_mg
            eg += side_eg
        else:
            mg -= side_mg
            eg -= side_eg
    phase = min(phase, PHASE_TOTAL)
    tapered = mg * phase + eg * (PHASE_TOTAL - phase)
    # Truncate toward zero as PeSTO's C implementation does. Python's // would round negative
    # values down and introduce a one-centipawn colour asymmetry.
    score = tapered // PHASE_TOTAL if tapered >= 0 else -((-tapered) // PHASE_TOTAL)
    score += _mop_up(board)
    return (score if board.turn else -score) + TEMPO


# ------------------------------------------------------------------------------------------
# Search
# ------------------------------------------------------------------------------------------

INF = 1_000_000
MATE = 100_000
MATE_BOUND = MATE - 1_000  # anything beyond this is a forced mate, and its distance matters
MAX_PLY = 64
MAX_DEPTH = 40

# Selective search margins (centipawns) and the late-move-reduction table. Futility margins are
# deliberately conservative; the LMR table is floor(ln(depth) * ln(index + 1) / 2.25), at least 1.
ASPIRATION_MARGIN = 30
REVERSE_FUTILITY_MARGIN = 120
FUTILITY_MARGIN = 120
LMR_MAX_MOVES = 256
LMR_REDUCTIONS = tuple(
    tuple(
        max(1, int(math.log(max(depth, 2)) * math.log(index + 1) / 2.25))
        for index in range(LMR_MAX_MOVES)
    )
    for depth in range(MAX_DEPTH + 2)
)

EXACT, LOWER, UPPER = 0, 1, 2

# A draw is worth a little less than nothing to us: shuffling into a repetition in an equal
# position throws away the chance to outplay a weaker opponent. Even plies are ours to move.
CONTEMPT = 30

ORDER_TT = 10_000_000
ORDER_CAPTURE = 1_000_000
ORDER_PROMOTION = 900_000
ORDER_KILLER = 800_000
HISTORY_CAP = 700_000

RANK_7 = (chess.BB_RANK_2, chess.BB_RANK_7)  # the rank a pawn promotes from, by colour
RANK_8 = (chess.BB_RANK_1, chess.BB_RANK_8)


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    score: int
    depth: int
    nodes: int
    elapsed_ms: float


def _captured(board: chess.Board, move: chess.Move) -> int:
    """Piece type taken by move, or 0. En passant lands on an empty square."""
    victim = board.piece_type_at(move.to_square)
    if victim:
        return victim
    if move.to_square == board.ep_square and board.piece_type_at(move.from_square) == PAWN:
        return PAWN
    return 0


def _mvv_lva(board: chess.Board, move: chess.Move, victim: int) -> int:
    attacker = board.piece_type_at(move.from_square) or 0
    score = ORDER_CAPTURE + PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker] // 10
    if move.promotion:
        score += PIECE_VALUE[move.promotion]
    return score


class Searcher:
    def __init__(self) -> None:
        self.tt: dict[Any, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history = [[0] * 4096, [0] * 4096]
        self.keys: list[Any] = []
        self.nodes = 0
        self.stop = False
        self.abort = False  # set from another thread to end a ponder search promptly
        self.deadline = 0.0
        self.root_move: chess.Move | None = None
        self.root_score = 0

    # -- driver ------------------------------------------------------------------------------

    def think(
        self,
        board: chess.Board,
        soft_ms: float,
        hard_ms: float,
        keys: list[Any] | None = None,
        max_depth: int = MAX_DEPTH,
    ) -> SearchResult:
        """Search board for about soft_ms, never past hard_ms, and return the best move found.

        max_depth caps the iterative deepening, which makes a search reproducible for benchmarks.
        """
        start = time.perf_counter()
        self.deadline = start + hard_ms / 1000.0
        self.keys = list(keys) if keys else []
        self.nodes = 0
        self.stop = False
        self.abort = False
        for table in self.history:
            for index in range(4096):
                table[index] >>= 1
        if len(self.tt) > 1_500_000:
            self.tt.clear()

        moves = self._ordered(board, list(board.legal_moves), None, 0)
        if not moves:
            raise ValueError("no legal moves")
        self.root_move = moves[0]
        self.root_score = 0
        depth_reached = 0
        if len(moves) > 1:
            iteration_started = start
            previous_ms = 0.0
            previous_move = self.root_move
            previous_score = 0
            target_ms = soft_ms
            for depth in range(1, max_depth + 1):
                score = self._root_search(board, depth)
                if self.stop:
                    break
                depth_reached = depth
                self.root_score = score
                # Same guard as the native searcher: a mate score read from the table at a
                # shallow depth is not a proven line, and stopping on it makes the engine
                # shuffle instead of mating. Keep iterating until the depth covers the
                # announced distance to mate.
                if abs(score) >= MATE_BOUND and depth >= MATE - abs(score):
                    break
                # An unstable root deserves more time: the best move just changed, or the score
                # fell. Spend up to the hard limit, but only when the search is telling us to.
                if depth >= 4 and (self.root_move != previous_move or score < previous_score - 40):
                    target_ms = min(hard_ms, soft_ms * 1.6)
                previous_move = self.root_move
                previous_score = score
                now = time.perf_counter()
                this_ms = (now - iteration_started) * 1000.0
                elapsed_ms = (now - start) * 1000.0
                # The next iteration costs a few times this one; do not start it if it would
                # land us well past the soft budget. The ratio is clamped because tiny early
                # iterations make it noise.
                growth = 3.0
                if previous_ms >= 5.0:
                    growth = min(5.0, max(2.0, this_ms / previous_ms))
                if elapsed_ms + this_ms * growth * 0.5 >= target_ms:
                    break
                iteration_started = now
                previous_ms = this_ms
        assert self.root_move is not None
        elapsed = (time.perf_counter() - start) * 1000.0
        return SearchResult(self.root_move, self.root_score, depth_reached, self.nodes, elapsed)

    def _root_search(self, board: chess.Board, depth: int) -> int:
        """One iteration through an aspiration window around the previous score."""
        if depth <= 2 or abs(self.root_score) >= MATE_BOUND:
            return self._search(board, depth, -INF, INF, 0)
        centre = self.root_score
        score = 0
        # A second miss opens the window fully, which bounds the overhead on a big swing.
        for margin in (ASPIRATION_MARGIN, ASPIRATION_MARGIN * 2, INF):
            alpha = -INF if margin == INF else centre - margin
            beta = INF if margin == INF else centre + margin
            score = self._search(board, depth, alpha, beta, 0)
            if self.stop or alpha < score < beta:
                break
        return score

    # -- negamax -------------------------------------------------------------------------------

    def _search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 255 == 0 and (self.abort or time.perf_counter() >= self.deadline):
            self.stop = True
        if self.stop:
            return 0
        if ply >= MAX_PLY:
            return evaluate(board)

        key = board._transposition_key()
        if ply > 0:
            halfmove = board.halfmove_clock
            if halfmove >= 100 or (halfmove >= 4 and key in self.keys[-halfmove:]):
                return _draw_score(ply)
            if board.is_insufficient_material():
                return _draw_score(ply)

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply)

        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if ply > 0 and tt_depth >= depth:
                score = _from_tt(tt_score, ply)
                if (
                    tt_flag == EXACT
                    or (tt_flag == LOWER and score >= beta)
                    or (tt_flag == UPPER and score <= alpha)
                ):
                    return score

        self.keys.append(key)

        # Null move: if passing still beats beta, a real move will too.
        if (
            depth >= 3
            and ply > 0
            and not in_check
            and beta < MATE_BOUND
            and board.occupied_co[board.turn] & ~(board.pawns | board.kings)
        ):
            reduction = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            score = -self._search(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1)
            board.pop()
            if self.stop:
                self.keys.pop()
                return 0
            if score >= beta:
                self.keys.pop()
                return score

        moves = self._ordered(board, list(board.legal_moves), tt_move, ply)
        if not moves:
            self.keys.pop()
            return -MATE + ply if in_check else _draw_score(ply)

        # Static pruning. Reverse futility: at a zero-window non-PV node a comfortably winning
        # static score already proves the bound. Moves were generated first, so stalemate stays
        # a draw. Futility: near the horizon, quiet moves that cannot lift a poor static score
        # back to alpha are skipped (never the first move, captures, promotions or checks).
        reverse_futility = (
            depth <= 3 and not in_check and beta - alpha == 1 and -MATE_BOUND < beta < MATE_BOUND
        )
        futility = depth <= 2 and not in_check
        static_eval = evaluate(board) if reverse_futility or futility else None
        if reverse_futility:
            assert static_eval is not None
            if static_eval - REVERSE_FUTILITY_MARGIN * depth >= beta:
                self.keys.pop()
                return static_eval

        best_score = -INF
        best_move: chess.Move | None = None
        flag = UPPER
        side = board.turn
        killers = self.killers[ply]
        can_reduce = depth >= 3 and not in_check
        for index, move in enumerate(moves):
            quiet = not move.promotion and not _captured(board, move)
            # gives_check is not cheap; only futility and LMR need it, and only for quiet moves.
            gives_check = False
            if quiet and index > 0 and (static_eval is not None or (can_reduce and index >= 3)):
                gives_check = board.gives_check(move)
            if (
                index > 0
                and quiet
                and static_eval is not None
                and -MATE_BOUND < alpha < MATE_BOUND
                and static_eval + FUTILITY_MARGIN * depth <= alpha
                and not gives_check
            ):
                continue
            reduction = 0
            if can_reduce and index >= 3 and quiet:
                reduction = LMR_REDUCTIONS[min(depth, MAX_DEPTH + 1)][min(index, LMR_MAX_MOVES - 1)]
                if move == killers[0] or move == killers[1]:
                    reduction -= 1
                if gives_check and reduction > 0:
                    reduction -= 1
            board.push(move)
            if index == 0:
                score = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Principal variation search: later moves get a null-window proof; a reduced move
                # that improves alpha is restored to full depth, then confirmed full-window.
                score = -self._search(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                if reduction and score > alpha and not self.stop:
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta and not self.stop:
                    score = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if self.stop:
                break
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    flag = EXACT
                    # Only a move that raised alpha is proven better at the root; in a failed
                    # aspiration pass every score is a bound, and must not change the answer.
                    if ply == 0:
                        self.root_move = move
                    if score >= beta:
                        flag = LOWER
                        if quiet:
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            table = self.history[side]
                            slot = move.from_square * 64 + move.to_square
                            table[slot] += depth * depth
                            if table[slot] > HISTORY_CAP:
                                for i in range(4096):
                                    table[i] >>= 1
                        break
        self.keys.pop()
        if not self.stop:
            self.tt[key] = (depth, _to_tt(best_score, ply), flag, best_move)
        return best_score

    # -- quiescence ----------------------------------------------------------------------------

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 255 == 0 and (self.abort or time.perf_counter() >= self.deadline):
            self.stop = True
        if self.stop:
            return 0
        if ply >= MAX_PLY:
            return evaluate(board)

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            stand_pat = -INF
            scored = [(self._quiet_order(board, move), move) for move in moves]
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            scored = []
            for move in board.generate_legal_captures():
                victim = _captured(board, move)
                # Delta pruning: a capture that cannot lift us back to alpha is not worth a node.
                if stand_pat + PIECE_VALUE[victim] + 200 <= alpha and not move.promotion:
                    continue
                scored.append((_mvv_lva(board, move, victim), move))
            side = board.turn
            promoters = board.pawns & board.occupied_co[side] & RANK_7[side]
            if promoters:
                for move in board.generate_legal_moves(promoters, RANK_8[side]):
                    if move.promotion == QUEEN and not board.piece_type_at(move.to_square):
                        scored.append((ORDER_PROMOTION, move))
        scored.sort(key=lambda item: item[0], reverse=True)

        best = stand_pat
        for _, move in scored:
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if self.stop:
                return 0
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        break
        return best

    # -- ordering ------------------------------------------------------------------------------

    def _ordered(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        killers = self.killers[ply]
        history = self.history[board.turn]
        scored = []
        for move in moves:
            if move == tt_move:
                score = ORDER_TT
            else:
                victim = _captured(board, move)
                if victim:
                    score = _mvv_lva(board, move, victim)
                elif move.promotion:
                    score = ORDER_PROMOTION + PIECE_VALUE[move.promotion]
                elif move == killers[0]:
                    score = ORDER_KILLER
                elif move == killers[1]:
                    score = ORDER_KILLER - 1
                else:
                    score = history[move.from_square * 64 + move.to_square]
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def _quiet_order(self, board: chess.Board, move: chess.Move) -> int:
        victim = _captured(board, move)
        if victim:
            return _mvv_lva(board, move, victim)
        return self.history[board.turn][move.from_square * 64 + move.to_square]


def _draw_score(ply: int) -> int:
    """A draw from the point of view of the side to move at this ply."""
    return -CONTEMPT if ply % 2 == 0 else CONTEMPT


def _to_tt(score: int, ply: int) -> int:
    """Mate scores are stored relative to the node, not the root."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


# ------------------------------------------------------------------------------------------
# Time management
# ------------------------------------------------------------------------------------------

INCREMENT_MS = 500  # the event's increment; it lands after we move, so it is not in time_left
RESERVE_MS = 150  # never planned to be spent, covers the runner's wire and pop-out latency


def budget(time_left_ms: int) -> tuple[float, float]:
    """(soft, hard) budgets in ms. Soft is the target, hard aborts the search."""
    usable = max(time_left_ms - RESERVE_MS, 10)
    soft = usable / 40.0 + INCREMENT_MS * 0.4
    hard = min(soft * 2.5, usable * 0.2)
    soft = min(soft, hard)
    return soft, hard




SEARCHER = Searcher()


def analyse(fen: str, ms: float, max_depth: int = MAX_DEPTH) -> SearchResult:
    """Search one position with a fresh searcher; for benchmarks, not for play."""
    return Searcher().think(chess.Board(fen), ms, ms, None, max_depth)
