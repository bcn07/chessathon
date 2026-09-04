"""The numba engine: PeSTO evaluation and alpha-beta search on fastboard's bitboard state.

Python owns only the iterative-deepening driver and time budgeting. Evaluation, move ordering,
quiescence, repetition checks, the transposition table and the recursive search all run in
compiled numba code. Importing this module compiles everything (20-40 s); agent.py does that in
a background thread and plays pyengine until it is ready.

Written by Codex from Claude's brief (work/numba-search), reviewed and adapted by Claude:
external abort via stats[1] (polled at every node), nogil on the search entry so a pondering
thread releases the GIL, and Searcher.ponder().
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

# LLVM's maximum optimization spends more import budget on this large recursive kernel than it
# recovers during a game. Level 1 keeps compile time down; agent.py sets it before numba loads.
os.environ.setdefault("NUMBA_OPT", "1")

import chess
import numba
import numpy as np

import fastboard as fb

# --------------------------------------------------------------------------------------
# PeSTO evaluation
# --------------------------------------------------------------------------------------

# Native piece types are PNBRQK = 0..5. These are the published PeSTO material values and
# piece-square tables. Table rows are written a8..h1; white looks up a vertically mirrored
# square, while black uses the table directly.
MG_VALUE = np.array((82, 337, 365, 477, 1025, 0), dtype=np.int16)
EG_VALUE = np.array((94, 281, 297, 512, 936, 0), dtype=np.int16)
PHASE_WEIGHT = np.array((0, 1, 1, 2, 4, 0), dtype=np.int16)
PHASE_TOTAL = 24
TEMPO = 10

# fmt: off
_MG_PAWN = (
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_EG_PAWN = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_MG_KNIGHT = (
   -167, -89, -34, -49,  61, -97, -15,-107,
    -73, -41,  72,  36,  23,  62,   7, -17,
    -47,  60,  37,  65,  84, 129,  73,  44,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -13,   4,  16,  13,  28,  19,  21,  -8,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
   -105, -21, -58, -33, -17, -28, -19, -23,
)
_EG_KNIGHT = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)
_MG_BISHOP = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)
_EG_BISHOP = (
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
)
_MG_ROOK = (
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
)
_EG_ROOK = (
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
)
_MG_QUEEN = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)
_EG_QUEEN = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
)
_MG_KING = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)
_EG_KING = (
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


def _build_eval_tables() -> tuple[np.ndarray, np.ndarray]:
    mg = np.zeros((12, 64), dtype=np.int16)
    eg = np.zeros((12, 64), dtype=np.int16)
    mg_source = (_MG_PAWN, _MG_KNIGHT, _MG_BISHOP, _MG_ROOK, _MG_QUEEN, _MG_KING)
    eg_source = (_EG_PAWN, _EG_KNIGHT, _EG_BISHOP, _EG_ROOK, _EG_QUEEN, _EG_KING)
    for color in (fb.WHITE, fb.BLACK):
        for piece_type in range(6):
            piece = color * 6 + piece_type
            for square in range(64):
                table_square = square ^ 56 if color == fb.WHITE else square
                mg[piece, square] = MG_VALUE[piece_type] + mg_source[piece_type][table_square]
                eg[piece, square] = EG_VALUE[piece_type] + eg_source[piece_type][table_square]
    return mg, eg


def _build_passed_masks() -> np.ndarray:
    masks = np.zeros((2, 64), dtype=np.uint64)
    for color in (fb.WHITE, fb.BLACK):
        for square in range(64):
            file_index, rank_index = square & 7, square >> 3
            ranks = range(rank_index + 1, 8) if color == fb.WHITE else range(rank_index)
            mask = 0
            for rank_to in ranks:
                for file_to in range(max(0, file_index - 1), min(8, file_index + 2)):
                    mask |= 1 << (rank_to * 8 + file_to)
            masks[color, square] = np.uint64(mask)
    return masks


MG_PST, EG_PST = _build_eval_tables()
PASSED_MASKS = _build_passed_masks()
PASSED_MG = np.array((0, 5, 10, 20, 35, 60, 100, 0), dtype=np.int16)
PASSED_EG = np.array((0, 10, 20, 35, 60, 100, 160, 0), dtype=np.int16)


@numba.njit
def _mop_up(state: np.ndarray, winner: int, loser: int) -> int:
    """Reward driving a bare king to an edge with the winning king nearby."""
    winner_offset = winner * 6
    material = state[winner_offset + fb.QUEEN] | state[winner_offset + fb.ROOK]
    minors = state[winner_offset + fb.KNIGHT] | state[winner_offset + fb.BISHOP]
    minor_count = 0
    scan = minors
    while scan:
        scan &= scan - fb.U64_ONE
        minor_count += 1
    if material == 0 and minor_count < 2:
        return 0
    winner_king = fb.lsb_square(state[winner_offset + fb.KING])
    loser_king = fb.lsb_square(state[loser * 6 + fb.KING])
    loser_file, loser_rank = loser_king & 7, loser_king >> 3
    edge_distance = min(loser_file, 7 - loser_file, loser_rank, 7 - loser_rank)
    king_distance = abs((winner_king & 7) - loser_file) + abs((winner_king >> 3) - loser_rank)
    return (3 - edge_distance) * 18 + (14 - king_distance) * 4


@numba.njit
def evaluate_state(state: np.ndarray) -> int:
    """Static centipawn score relative to the native state's side to move."""
    mg = 0
    eg = 0
    phase = 0
    for piece in range(12):
        color = piece // 6
        piece_type = piece % 6
        sign = 1 if color == fb.WHITE else -1
        pieces = state[piece]
        while pieces:
            square = fb.lsb_square(pieces)
            pieces &= pieces - fb.U64_ONE
            mg += sign * int(MG_PST[piece, square])
            eg += sign * int(EG_PST[piece, square])
            phase += int(PHASE_WEIGHT[piece_type])
            if piece_type == fb.PAWN:
                enemy_pawns = state[(1 - color) * 6 + fb.PAWN]
                if enemy_pawns & PASSED_MASKS[color, square] == 0:
                    rank_index = int(square) >> 3
                    relative_rank = np.int64(
                        rank_index if color == fb.WHITE else 7 - rank_index
                    )
                    mg += sign * int(PASSED_MG[relative_rank])
                    eg += sign * int(PASSED_EG[relative_rank])
    if phase > PHASE_TOTAL:
        phase = PHASE_TOTAL
    score = (mg * phase + eg * (PHASE_TOTAL - phase)) // PHASE_TOTAL
    if state[fb.BLACK_OCC] == state[fb.BK]:
        score += _mop_up(state, fb.WHITE, fb.BLACK)
    if state[fb.WHITE_OCC] == state[fb.WK]:
        score -= _mop_up(state, fb.BLACK, fb.WHITE)
    relative = score if int(state[fb.SIDE]) == fb.WHITE else -score
    return relative + TEMPO


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns from a python-chess board's side to move."""
    return int(evaluate_state(fb.from_fen(board.fen(en_passant="fen"))))


# --------------------------------------------------------------------------------------
# Numba search
# --------------------------------------------------------------------------------------

INF = 1_000_000
MATE = 100_000
MATE_BOUND = MATE - 1_000
MAX_PLY = 96
MAX_DEPTH = 40
MAX_GAME_HISTORY = 512

EXACT, LOWER, UPPER = 0, 1, 2
CONTEMPT = 30

ORDER_TT = 10_000_000
ORDER_CAPTURE = 1_000_000
ORDER_PROMOTION = 900_000
ORDER_KILLER = 800_000
HISTORY_CAP = 700_000
SEARCH_VALUE = np.array((100, 320, 330, 500, 900, 20_000), dtype=np.int32)

TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    score: int
    depth: int
    nodes: int
    elapsed_ms: float


@numba.njit
def _draw_score(ply: int) -> int:
    return -CONTEMPT if ply % 2 == 0 else CONTEMPT


@numba.njit
def _to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@numba.njit
def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


@numba.njit
def _canonical_key(state: np.ndarray) -> np.uint64:
    """Remove a raw EP hash unless the side to move has a legal en-passant capture."""
    ep_square = int(state[fb.EP_SQUARE])
    key = state[fb.ZOBRIST]
    if ep_square == fb.NO_SQUARE:
        return key
    color = int(state[fb.SIDE])
    captured_square = ep_square - 8 if color == fb.WHITE else ep_square + 8
    if not 0 <= captured_square < 64:
        return key ^ fb.ZOBRIST_EP[ep_square]
    captured_bit = fb.U64_ONE << np.uint64(captured_square)
    if state[(1 - color) * 6 + fb.PAWN] & captured_bit == 0:
        return key ^ fb.ZOBRIST_EP[ep_square]
    candidates = state[color * 6 + fb.PAWN] & fb.PAWN_ATTACKS[1 - color, ep_square]
    while candidates:
        from_square = fb.lsb_square(candidates)
        candidates &= candidates - fb.U64_ONE
        move = fb.encode_move(
            from_square,
            ep_square,
            0,
            fb.FLAG_CAPTURE | fb.FLAG_EN_PASSANT,
        )
        undo = fb.make_move(state, move)
        legal = not fb.is_in_check(state, color)
        fb.unmake_move(state, move, undo)
        if legal:
            return key
    return key ^ fb.ZOBRIST_EP[ep_square]


@numba.njit
def _is_insufficient(state: np.ndarray) -> bool:
    if state[fb.WP] | state[fb.BP] | state[fb.WR] | state[fb.BR] | state[fb.WQ] | state[fb.BQ]:
        return False
    knights = state[fb.WN] | state[fb.BN]
    bishops = state[fb.WB] | state[fb.BB]
    minors = knights | bishops
    count = 0
    scan = minors
    while scan:
        scan &= scan - fb.U64_ONE
        count += 1
    if count <= 1:
        return True
    if knights:
        return False
    square_color = -1
    scan = bishops
    while scan:
        square = fb.lsb_square(scan)
        scan &= scan - fb.U64_ONE
        color = ((square & 7) + (square >> 3)) & 1
        if square_color < 0:
            square_color = color
        elif square_color != color:
            return False
    return True


@numba.njit
def _is_draw(state: np.ndarray, ply: int, rep_keys: np.ndarray, rep_count: int) -> bool:
    halfmove = int(state[fb.HALFMOVE])
    if halfmove >= 100 or _is_insufficient(state):
        return True
    if halfmove < 4:
        return False
    key = _canonical_key(state)
    earliest = max(0, rep_count - halfmove)
    # Equal positions have equal side to move, so only every second historical ply can match.
    index = rep_count - 2
    while index >= earliest:
        if rep_keys[index] == key:
            return True
        index -= 2
    return False


@numba.njit
def _piece_type_at(state: np.ndarray, square: int) -> int:
    bit = fb.U64_ONE << np.uint64(square)
    for piece_type in range(6):
        if (state[piece_type] | state[6 + piece_type]) & bit:
            return piece_type
    return fb.NO_PIECE


@numba.njit
def _captured_type(state: np.ndarray, move: np.uint32) -> int:
    if int(move) & fb.FLAG_EN_PASSANT:
        return fb.PAWN
    if int(move) & fb.FLAG_CAPTURE:
        return _piece_type_at(state, fb.move_to(move))
    return fb.NO_PIECE


@numba.njit
def _move_score(
    state: np.ndarray,
    move: np.uint32,
    tt_move: np.uint32,
    ply: int,
    killers: np.ndarray,
    history: np.ndarray,
) -> int:
    if move == tt_move and tt_move != 0:
        return ORDER_TT
    victim = _captured_type(state, move)
    promotion = fb.move_promotion(move)
    if victim != fb.NO_PIECE:
        attacker = _piece_type_at(state, fb.move_from(move))
        score = ORDER_CAPTURE + int(SEARCH_VALUE[victim]) * 10
        if attacker != fb.NO_PIECE:
            score -= int(SEARCH_VALUE[attacker]) // 10
        if promotion:
            score += int(SEARCH_VALUE[promotion])
        return score
    if promotion:
        return ORDER_PROMOTION + int(SEARCH_VALUE[promotion])
    if move == killers[ply, 0]:
        return ORDER_KILLER
    if move == killers[ply, 1]:
        return ORDER_KILLER - 1
    slot = fb.move_from(move) * 64 + fb.move_to(move)
    return int(history[int(state[fb.SIDE]), slot])


@numba.njit
def _order_moves(
    state: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    count: int,
    tt_move: np.uint32,
    ply: int,
    killers: np.ndarray,
    history: np.ndarray,
) -> None:
    """Insertion sort is quick for chess-sized lists and needs no temporary allocation."""
    for index in range(count):
        scores[index] = _move_score(state, moves[index], tt_move, ply, killers, history)
    for index in range(1, count):
        move = moves[index]
        score = scores[index]
        previous = index - 1
        while previous >= 0 and scores[previous] < score:
            moves[previous + 1] = moves[previous]
            scores[previous + 1] = scores[previous]
            previous -= 1
        moves[previous + 1] = move
        scores[previous + 1] = score


@numba.njit
def _store_killer(
    move: np.uint32,
    depth: int,
    ply: int,
    side: int,
    killers: np.ndarray,
    history: np.ndarray,
) -> None:
    if killers[ply, 0] != move:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = move
    slot = fb.move_from(move) * 64 + fb.move_to(move)
    value = int(history[side, slot]) + depth * depth
    history[side, slot] = min(value, HISTORY_CAP)


@numba.njit
def _make_null(state: np.ndarray) -> tuple[np.uint64, np.uint64, np.uint64]:
    old_ep = state[fb.EP_SQUARE]
    old_halfmove = state[fb.HALFMOVE]
    old_hash = state[fb.ZOBRIST]
    key = old_hash
    if int(old_ep) != fb.NO_SQUARE:
        key ^= fb.ZOBRIST_EP[int(old_ep)]
    state[fb.EP_SQUARE] = fb.NO_SQUARE
    state[fb.HALFMOVE] = old_halfmove + fb.U64_ONE
    state[fb.SIDE] = 1 - state[fb.SIDE]
    state[fb.ZOBRIST] = key ^ fb.ZOBRIST_SIDE
    return old_ep, old_halfmove, old_hash


@numba.njit
def _unmake_null(
    state: np.ndarray, undo: tuple[np.uint64, np.uint64, np.uint64]
) -> None:
    old_ep, old_halfmove, old_hash = undo
    state[fb.SIDE] = 1 - state[fb.SIDE]
    state[fb.EP_SQUARE] = old_ep
    state[fb.HALFMOVE] = old_halfmove
    state[fb.ZOBRIST] = old_hash


@numba.njit
def _quiesce(
    state: np.ndarray,
    alpha: int,
    beta: int,
    ply: int,
    rep_keys: np.ndarray,
    rep_count: int,
    killers: np.ndarray,
    history: np.ndarray,
    move_buffers: np.ndarray,
    score_buffers: np.ndarray,
    stats: np.ndarray,
    node_limit: int,
) -> int:
    stats[0] += 1
    if stats[1] or stats[0] >= node_limit:
        stats[1] = 1
        return 0
    if ply >= MAX_PLY:
        return evaluate_state(state)
    if ply > 0 and _is_draw(state, ply, rep_keys, rep_count):
        return _draw_score(ply)

    in_check = fb.is_in_check(state, int(state[fb.SIDE]))
    stand_pat = -INF if in_check else evaluate_state(state)
    if not in_check:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    rep_keys[rep_count] = _canonical_key(state)
    moves = move_buffers[ply]
    scores = score_buffers[ply]
    count = fb.generate_pseudo_into(state, moves)
    _order_moves(state, moves, scores, count, np.uint32(0), ply, killers, history)
    moving_color = int(state[fb.SIDE])
    legal_count = 0
    best = stand_pat
    for index in range(count):
        move = moves[index]
        victim = _captured_type(state, move)
        promotion = fb.move_promotion(move)
        if not in_check and victim == fb.NO_PIECE and promotion != fb.QUEEN:
            continue
        if (
            not in_check
            and victim != fb.NO_PIECE
            and promotion == 0
            and stand_pat + int(SEARCH_VALUE[victim]) + 200 <= alpha
        ):
            continue
        undo = fb.make_move(state, move)
        if fb.is_in_check(state, moving_color):
            fb.unmake_move(state, move, undo)
            continue
        legal_count += 1
        score = -_quiesce(
            state,
            -beta,
            -alpha,
            ply + 1,
            rep_keys,
            rep_count + 1,
            killers,
            history,
            move_buffers,
            score_buffers,
            stats,
            node_limit,
        )
        fb.unmake_move(state, move, undo)
        if stats[1]:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if score >= beta:
                    break
    if in_check and legal_count == 0:
        return -MATE + ply
    return best


@numba.njit
def _negamax(
    state: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    rep_keys: np.ndarray,
    rep_count: int,
    tt_keys: np.ndarray,
    tt_depth: np.ndarray,
    tt_scores: np.ndarray,
    tt_flags: np.ndarray,
    tt_moves: np.ndarray,
    tt_mask: int,
    killers: np.ndarray,
    history: np.ndarray,
    move_buffers: np.ndarray,
    score_buffers: np.ndarray,
    stats: np.ndarray,
    node_limit: int,
) -> int:
    stats[0] += 1
    if stats[1] or stats[0] >= node_limit:
        stats[1] = 1
        return 0
    if ply >= MAX_PLY:
        return evaluate_state(state)
    if ply > 0 and _is_draw(state, ply, rep_keys, rep_count):
        return _draw_score(ply)

    in_check = fb.is_in_check(state, int(state[fb.SIDE]))
    if in_check:
        depth += 1
    if depth <= 0:
        return _quiesce(
            state,
            alpha,
            beta,
            ply,
            rep_keys,
            rep_count,
            killers,
            history,
            move_buffers,
            score_buffers,
            stats,
            node_limit,
        )

    raw_key = state[fb.ZOBRIST]
    slot = int(raw_key & np.uint64(tt_mask))
    tt_move = np.uint32(0)
    if tt_depth[slot] >= 0 and tt_keys[slot] == raw_key:
        tt_move = tt_moves[slot]
        if tt_depth[slot] >= depth:
            tt_score = _from_tt(int(tt_scores[slot]), ply)
            tt_flag = int(tt_flags[slot])
            if (
                tt_flag == EXACT
                or (tt_flag == LOWER and tt_score >= beta)
                or (tt_flag == UPPER and tt_score <= alpha)
            ):
                return tt_score

    rep_keys[rep_count] = _canonical_key(state)
    side = int(state[fb.SIDE])
    if (
        depth >= 3
        and ply > 0
        and not in_check
        and beta < MATE_BOUND
        and state[fb.WHITE_OCC + side]
        & ~(state[side * 6 + fb.PAWN] | state[side * 6 + fb.KING])
    ):
        reduction = 3 if depth >= 6 else 2
        null_undo = _make_null(state)
        null_score = -_negamax(
            state,
            depth - 1 - reduction,
            -beta,
            -beta + 1,
            ply + 1,
            rep_keys,
            rep_count + 1,
            tt_keys,
            tt_depth,
            tt_scores,
            tt_flags,
            tt_moves,
            tt_mask,
            killers,
            history,
            move_buffers,
            score_buffers,
            stats,
            node_limit,
        )
        _unmake_null(state, null_undo)
        if stats[1]:
            return 0
        if null_score >= beta:
            return null_score

    moves = move_buffers[ply]
    scores = score_buffers[ply]
    count = fb.generate_pseudo_into(state, moves)
    _order_moves(state, moves, scores, count, tt_move, ply, killers, history)
    alpha_original = alpha
    best_score = -INF
    best_move = np.uint32(0)
    legal_count = 0
    for index in range(count):
        move = moves[index]
        quiet = _captured_type(state, move) == fb.NO_PIECE and fb.move_promotion(move) == 0
        undo = fb.make_move(state, move)
        if fb.is_in_check(state, side):
            fb.unmake_move(state, move, undo)
            continue
        child_in_check = fb.is_in_check(state, int(state[fb.SIDE]))
        reduction = 0
        if depth >= 3 and legal_count >= 3 and quiet and not in_check and not child_in_check:
            reduction = 2 if depth >= 6 and legal_count >= 6 else 1
        if legal_count == 0:
            score = -_negamax(
                state,
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
                rep_keys,
                rep_count + 1,
                tt_keys,
                tt_depth,
                tt_scores,
                tt_flags,
                tt_moves,
                tt_mask,
                killers,
                history,
                move_buffers,
                score_buffers,
                stats,
                node_limit,
            )
        else:
            score = -_negamax(
                state,
                depth - 1 - reduction,
                -alpha - 1,
                -alpha,
                ply + 1,
                rep_keys,
                rep_count + 1,
                tt_keys,
                tt_depth,
                tt_scores,
                tt_flags,
                tt_moves,
                tt_mask,
                killers,
                history,
                move_buffers,
                score_buffers,
                stats,
                node_limit,
            )
            if score > alpha and score < beta and not stats[1]:
                score = -_negamax(
                    state,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    rep_keys,
                    rep_count + 1,
                    tt_keys,
                    tt_depth,
                    tt_scores,
                    tt_flags,
                    tt_moves,
                    tt_mask,
                    killers,
                    history,
                    move_buffers,
                    score_buffers,
                    stats,
                    node_limit,
                )
        fb.unmake_move(state, move, undo)
        if stats[1]:
            return 0
        legal_count += 1
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
            if alpha >= beta:
                if quiet:
                    _store_killer(move, depth, ply, side, killers, history)
                break

    if legal_count == 0:
        return -MATE + ply if in_check else _draw_score(ply)
    flag = UPPER
    if best_score >= beta:
        flag = LOWER
    elif best_score > alpha_original:
        flag = EXACT
    tt_keys[slot] = raw_key
    tt_depth[slot] = depth
    tt_scores[slot] = _to_tt(best_score, ply)
    tt_flags[slot] = flag
    tt_moves[slot] = best_move
    return best_score


@numba.njit(nogil=True)
def search_root(
    state: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    rep_keys: np.ndarray,
    rep_count: int,
    tt_keys: np.ndarray,
    tt_depth: np.ndarray,
    tt_scores: np.ndarray,
    tt_flags: np.ndarray,
    tt_moves: np.ndarray,
    tt_mask: int,
    killers: np.ndarray,
    history: np.ndarray,
    move_buffers: np.ndarray,
    score_buffers: np.ndarray,
    stats: np.ndarray,
    node_limit: int,
) -> tuple[int, np.uint32, bool]:
    """Search one completed iteration, aborting exactly at the caller's node limit."""
    stats[0] += 1
    if stats[1] or stats[0] >= node_limit:
        stats[1] = 1
        return 0, np.uint32(0), True
    raw_key = state[fb.ZOBRIST]
    slot = int(raw_key & np.uint64(tt_mask))
    tt_move = np.uint32(0)
    if tt_depth[slot] >= 0 and tt_keys[slot] == raw_key:
        tt_move = tt_moves[slot]

    rep_keys[rep_count] = _canonical_key(state)
    side = int(state[fb.SIDE])
    in_check = fb.is_in_check(state, side)
    effective_depth = depth + 1 if in_check else depth
    moves = move_buffers[0]
    scores = score_buffers[0]
    count = fb.generate_pseudo_into(state, moves)
    _order_moves(state, moves, scores, count, tt_move, 0, killers, history)
    alpha_original = alpha
    best_score = -INF
    best_move = np.uint32(0)
    legal_count = 0
    for index in range(count):
        move = moves[index]
        undo = fb.make_move(state, move)
        if fb.is_in_check(state, side):
            fb.unmake_move(state, move, undo)
            continue
        if legal_count == 0:
            score = -_negamax(
                state,
                effective_depth - 1,
                -beta,
                -alpha,
                1,
                rep_keys,
                rep_count + 1,
                tt_keys,
                tt_depth,
                tt_scores,
                tt_flags,
                tt_moves,
                tt_mask,
                killers,
                history,
                move_buffers,
                score_buffers,
                stats,
                node_limit,
            )
        else:
            score = -_negamax(
                state,
                effective_depth - 1,
                -alpha - 1,
                -alpha,
                1,
                rep_keys,
                rep_count + 1,
                tt_keys,
                tt_depth,
                tt_scores,
                tt_flags,
                tt_moves,
                tt_mask,
                killers,
                history,
                move_buffers,
                score_buffers,
                stats,
                node_limit,
            )
            if score > alpha and score < beta and not stats[1]:
                score = -_negamax(
                    state,
                    effective_depth - 1,
                    -beta,
                    -alpha,
                    1,
                    rep_keys,
                    rep_count + 1,
                    tt_keys,
                    tt_depth,
                    tt_scores,
                    tt_flags,
                    tt_moves,
                    tt_mask,
                    killers,
                    history,
                    move_buffers,
                    score_buffers,
                    stats,
                    node_limit,
                )
        fb.unmake_move(state, move, undo)
        if stats[1]:
            return 0, best_move, True
        legal_count += 1
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
            if alpha >= beta:
                break
    if legal_count == 0:
        terminal = -MATE if in_check else _draw_score(0)
        return terminal, np.uint32(0), False
    flag = UPPER
    if best_score >= beta:
        flag = LOWER
    elif best_score > alpha_original:
        flag = EXACT
    tt_keys[slot] = raw_key
    tt_depth[slot] = effective_depth
    tt_scores[slot] = _to_tt(best_score, 0)
    tt_flags[slot] = flag
    tt_moves[slot] = best_move
    return best_score, best_move, False


# --------------------------------------------------------------------------------------
# Python driver and time management
# --------------------------------------------------------------------------------------

INCREMENT_MS = 500
RESERVE_MS = 150


def budget(time_left_ms: int) -> tuple[float, float]:
    """Return soft and hard wall-time targets in milliseconds."""
    usable = max(time_left_ms - RESERVE_MS, 10)
    soft = usable / 40.0 + INCREMENT_MS * 0.4
    hard = min(soft * 2.5, usable * 0.2)
    return min(soft, hard), hard


class Searcher:
    def __init__(self, tt_bits: int = TT_BITS) -> None:
        size = 1 << tt_bits
        self.tt_keys = np.zeros(size, dtype=np.uint64)
        self.tt_depth = np.full(size, -1, dtype=np.int8)
        self.tt_scores = np.zeros(size, dtype=np.int32)
        self.tt_flags = np.zeros(size, dtype=np.int8)
        self.tt_moves = np.zeros(size, dtype=np.uint32)
        self.tt_mask = size - 1
        self.killers = np.zeros((MAX_PLY + 2, 2), dtype=np.uint32)
        self.history = np.zeros((2, 4096), dtype=np.int32)
        self.move_buffers = np.zeros((MAX_PLY + 2, fb.MAX_MOVES), dtype=np.uint32)
        self.score_buffers = np.zeros((MAX_PLY + 2, fb.MAX_MOVES), dtype=np.int32)
        self.rep_keys = np.zeros(MAX_GAME_HISTORY + MAX_PLY + 4, dtype=np.uint64)
        self.stats = np.zeros(2, dtype=np.int64)
        self.nodes = 0
        self.nps = 300_000.0

    def _iteration(
        self,
        state: np.ndarray,
        depth: int,
        alpha: int,
        beta: int,
        rep_count: int,
        node_limit: int,
    ) -> tuple[int, np.uint32, bool]:
        return search_root(
            state,
            depth,
            alpha,
            beta,
            self.rep_keys,
            rep_count,
            self.tt_keys,
            self.tt_depth,
            self.tt_scores,
            self.tt_flags,
            self.tt_moves,
            self.tt_mask,
            self.killers,
            self.history,
            self.move_buffers,
            self.score_buffers,
            self.stats,
            node_limit,
        )

    def think(
        self,
        board: chess.Board,
        soft_ms: float,
        hard_ms: float,
        keys: list[int] | None = None,
        max_depth: int = MAX_DEPTH,
    ) -> SearchResult:
        start = time.perf_counter()
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("no legal moves")
        if len(legal) == 1:
            return SearchResult(legal[0], 0, 0, 0, (time.perf_counter() - start) * 1000.0)

        state = fb.from_fen(board.fen(en_passant="fen"))
        history_keys = keys or []
        room = MAX_GAME_HISTORY
        trimmed = history_keys[-room:]
        rep_count = len(trimmed)
        if rep_count:
            self.rep_keys[:rep_count] = np.asarray(trimmed, dtype=np.uint64)
        self.stats[:] = 0
        self.history >>= 1

        best_move = legal[0]
        best_score = 0
        depth_reached = 0
        previous_iteration_ms = 0.0
        previous_score = 0
        for depth in range(1, max_depth + 1):
            now = time.perf_counter()
            elapsed_ms = (now - start) * 1000.0
            remaining_ms = hard_ms - elapsed_ms
            if remaining_ms <= 0:
                break
            iteration_start = now
            iteration_nodes = int(self.stats[0])
            # A node counter is deterministic and cheap inside Numba. Re-estimating its wall-time
            # conversion after every completed iteration makes the cap adapt to each position.
            allowance = max(128, int(remaining_ms * self.nps * 0.78 / 1000.0))
            node_limit = int(self.stats[0]) + allowance
            self.stats[1] = 0
            if depth >= 4:
                alpha = max(-INF, previous_score - 50)
                beta = min(INF, previous_score + 50)
            else:
                alpha, beta = -INF, INF
            score, native_move, aborted = self._iteration(
                state, depth, alpha, beta, rep_count, node_limit
            )
            if not aborted and (score <= alpha or score >= beta):
                score, native_move, aborted = self._iteration(
                    state, depth, -INF, INF, rep_count, node_limit
                )
            iteration_elapsed_ms = (time.perf_counter() - iteration_start) * 1000.0
            delta_nodes = int(self.stats[0]) - iteration_nodes
            if iteration_elapsed_ms >= 0.5 and delta_nodes > 0:
                observed = delta_nodes * 1000.0 / iteration_elapsed_ms
                self.nps = self.nps * 0.35 + observed * 0.65
            if aborted:
                break
            candidate = chess.Move.from_uci(fb.move_to_uci(native_move))
            if candidate not in board.legal_moves:
                raise ValueError(f"native search returned illegal move {candidate}")
            best_move = candidate
            best_score = int(score)
            previous_score = best_score
            depth_reached = depth
            if abs(best_score) >= MATE_BOUND:
                break
            total_elapsed_ms = (time.perf_counter() - start) * 1000.0
            growth = 3.0
            if previous_iteration_ms >= 0.5:
                growth = min(6.0, max(1.7, iteration_elapsed_ms / previous_iteration_ms))
            if total_elapsed_ms + iteration_elapsed_ms * growth * 0.55 >= soft_ms:
                break
            previous_iteration_ms = iteration_elapsed_ms

        self.nodes = int(self.stats[0])
        elapsed = (time.perf_counter() - start) * 1000.0
        return SearchResult(best_move, best_score, depth_reached, self.nodes, elapsed)

    def ponder(self, board: chess.Board, keys: list[int] | None = None) -> int:
        """Deepen on board until another thread sets stats[1], filling the shared TT."""
        legal = list(board.legal_moves)
        if not legal:
            return 0
        state = fb.from_fen(board.fen(en_passant="fen"))
        trimmed = (keys or [])[-MAX_GAME_HISTORY:]
        rep_count = len(trimmed)
        if rep_count:
            self.rep_keys[:rep_count] = np.asarray(trimmed, dtype=np.uint64)
        self.stats[:] = 0
        for depth in range(1, MAX_DEPTH + 1):
            score, _, aborted = self._iteration(state, depth, -INF, INF, rep_count, 1 << 62)
            if aborted or abs(score) >= MATE_BOUND:
                break
        self.nodes = int(self.stats[0])
        return self.nodes

    def abort(self) -> None:
        """Ask a running search (on any thread) to stop at its next node."""
        self.stats[1] = 1


def _warm_search() -> float:
    started = time.perf_counter()
    probe = Searcher(tt_bits=15)
    state = fb.from_fen(chess.STARTING_FEN)
    evaluate_state(state)
    _canonical_key(state)
    probe.stats[:] = 0
    probe._iteration(state, 4, -INF, INF, 0, 20_000)
    probe.tt_depth.fill(-1)
    probe.stats[:] = 0
    timed = time.perf_counter()
    probe._iteration(state, 8, -INF, INF, 0, 100_000)
    elapsed = max(time.perf_counter() - timed, 1e-6)
    global _INITIAL_NPS
    _INITIAL_NPS = max(50_000.0, min(5_000_000.0, float(probe.stats[0]) / elapsed))
    return time.perf_counter() - started


_INITIAL_NPS = 300_000.0
_WARMUP_SECONDS = _warm_search()
SEARCHER = Searcher()
SEARCHER.nps = _INITIAL_NPS




def _native_repetition_key(board: chess.Board) -> int:
    """The canonical repetition key of a python-chess board, for game-history lists."""
    return int(_canonical_key(fb.from_fen(board.fen(en_passant="fen"))))


def analyse(fen: str, ms: float, max_depth: int = MAX_DEPTH) -> SearchResult:
    """Search one position with a fresh searcher; for benchmarks, not for play."""
    searcher = Searcher()
    searcher.nps = _INITIAL_NPS
    return searcher.think(chess.Board(fen), ms, ms, None, max_depth)
