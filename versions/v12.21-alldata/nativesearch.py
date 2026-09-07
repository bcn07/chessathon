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

import math
import os
import time
from dataclasses import dataclass

# LLVM's maximum optimization spends more import budget on this large recursive kernel than it
# recovers during a game. Level 1 keeps compile time down; agent.py sets it before numba loads.
os.environ.setdefault("NUMBA_OPT", "3")

import chess
import numba
import numpy as np

import fastboard as fb
import nnue_eval as nn

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

# Mobility: safe destination squares per piece, counted against a baseline so a normally placed
# piece scores zero and only unusual freedom -- or a piece with nowhere to go -- moves the number.
# Knights and bishops skip squares an enemy pawn covers, because they cannot use them; rooks and
# queens count every square they see.  Centipawns per square, indexed by piece type PNBRQK, kept
# in arrays so the weights can be tuned without touching the loop.
MOBILITY_BASE = np.array((0, 4, 6, 7, 13, 0), dtype=np.int16)
MOBILITY_MG = np.array((0, 4, 5, 2, 1, 0), dtype=np.int16)
MOBILITY_EG = np.array((0, 4, 5, 4, 2, 0), dtype=np.int16)

NOT_FILE_A = np.uint64(0xFEFEFEFEFEFEFEFE)
NOT_FILE_H = np.uint64(0x7F7F7F7F7F7F7F7F)
FILE_A = np.uint64(0x0101010101010101)
FILE_B = np.uint64(0x0202020202020202)
# Multiplying a file's bits (gathered onto the a-file) by this diagonal collects ranks 2..7 into
# the top six bits; the b-file does the same for a diagonal or an anti-diagonal.  The classic
# "kindergarten" gathers: two shifts and a multiply replace fastboard's ray scan, which costs the
# evaluation far too much when it runs on every slider at every node.
FILE_GATHER = np.uint64(0x0080402010080400)

RANK_LINE, FILE_LINE, DIAGONAL_LINE, ANTIDIAGONAL_LINE = range(4)
_LINE_STEPS = (((1, 0), (-1, 0)), ((0, 1), (0, -1)), ((1, 1), (-1, -1)), ((1, -1), (-1, 1)))


def _line_squares(square: int, line: int) -> int:
    """The whole line through a square, that square included."""
    mask = 1 << square
    file_index, rank_index = square & 7, square >> 3
    for file_delta, rank_delta in _LINE_STEPS[line]:
        file_to, rank_to = file_index + file_delta, rank_index + rank_delta
        while 0 <= file_to < 8 and 0 <= rank_to < 8:
            mask |= 1 << (rank_to * 8 + file_to)
            file_to += file_delta
            rank_to += rank_delta
    return mask


def _line_attacks(square: int, line: int, occupied: int) -> int:
    """Attacks along one line, stopping on (and including) the first occupied square."""
    attacks = 0
    file_index, rank_index = square & 7, square >> 3
    for file_delta, rank_delta in _LINE_STEPS[line]:
        file_to, rank_to = file_index + file_delta, rank_index + rank_delta
        while 0 <= file_to < 8 and 0 <= rank_to < 8:
            bit = 1 << (rank_to * 8 + file_to)
            attacks |= bit
            if occupied & bit:
                break
            file_to += file_delta
            rank_to += rank_delta
    return attacks


def _line_key(square: int, line: int, occupied: int) -> int:
    """The six-bit table index; mirrors the compiled version in _rook_attacks/_bishop_attacks."""
    if line == RANK_LINE:
        return (occupied >> ((square >> 3) * 8 + 1)) & 63
    if line == FILE_LINE:
        column = (occupied >> (square & 7)) & int(FILE_A)
        return ((column * int(FILE_GATHER)) & 0xFFFFFFFFFFFFFFFF) >> 58
    mask = _DIAGONAL_MASKS[line - DIAGONAL_LINE][square]
    return (((occupied & mask) * int(FILE_B)) & 0xFFFFFFFFFFFFFFFF) >> 58


def _build_diagonal_masks() -> list[list[int]]:
    """Diagonal and anti-diagonal through each square, that square excluded."""
    masks = [[0] * 64, [0] * 64]
    for line in (DIAGONAL_LINE, ANTIDIAGONAL_LINE):
        for square in range(64):
            masks[line - DIAGONAL_LINE][square] = _line_squares(square, line) & ~(1 << square)
    return masks


_DIAGONAL_MASKS = _build_diagonal_masks()


def _build_line_attacks() -> np.ndarray:
    """LINE_ATTACKS[line, square, key]: every attack set a line can produce, by six-bit key.

    Enumerating the whole line rather than its inner squares covers the keys that occur in play
    (the piece's own square is occupied, and diagonal end squares reach the key too); the squares
    those extra bits describe cannot block, so the collision check below is what proves the key
    is a perfect hash for this line.
    """
    table = np.zeros((4, 64, 64), dtype=np.uint64)
    for line in range(4):
        for square in range(64):
            mask = _line_squares(square, line)
            seen: dict[int, int] = {}
            occupied = 0
            while True:
                key = _line_key(square, line, occupied)
                attacks = _line_attacks(square, line, occupied)
                if seen.get(key, attacks) != attacks:
                    raise RuntimeError(f"line {line} square {square}: key {key} collides")
                seen[key] = attacks
                table[line, square, key] = np.uint64(attacks)
                occupied = (occupied - mask) & mask  # next subset of mask
                if occupied == 0:
                    break
    return table


LINE_ATTACKS = _build_line_attacks()
DIAGONAL_MASK = np.array(_DIAGONAL_MASKS[0], dtype=np.uint64)
ANTIDIAGONAL_MASK = np.array(_DIAGONAL_MASKS[1], dtype=np.uint64)


@numba.njit(inline="always")
def _popcount(bitboard: np.uint64) -> int:
    """SWAR population count: numba exposes no ctpop intrinsic."""
    counts = bitboard - ((bitboard >> np.uint64(1)) & np.uint64(0x5555555555555555))
    counts = (counts & np.uint64(0x3333333333333333)) + (
        (counts >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    counts = (counts + (counts >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((counts * np.uint64(0x0101010101010101)) >> np.uint64(56))


@numba.njit(inline="always")
def _rook_attacks(square: int, occupied: np.uint64) -> np.uint64:
    """Equal to fb.rook_attacks, by two table lookups instead of a scan of four rays."""
    rank_key = int(
        (occupied >> (np.uint64(square >> 3) * np.uint64(8) + np.uint64(1))) & np.uint64(63)
    )
    column = (occupied >> np.uint64(square & 7)) & FILE_A
    file_key = int((column * FILE_GATHER) >> np.uint64(58))
    return LINE_ATTACKS[RANK_LINE, square, rank_key] | LINE_ATTACKS[FILE_LINE, square, file_key]


@numba.njit(inline="always")
def _bishop_attacks(square: int, occupied: np.uint64) -> np.uint64:
    """Equal to fb.bishop_attacks, by two table lookups instead of a scan of four rays."""
    diagonal_key = int(((occupied & DIAGONAL_MASK[square]) * FILE_B) >> np.uint64(58))
    anti_key = int(((occupied & ANTIDIAGONAL_MASK[square]) * FILE_B) >> np.uint64(58))
    return (
        LINE_ATTACKS[DIAGONAL_LINE, square, diagonal_key]
        | LINE_ATTACKS[ANTIDIAGONAL_LINE, square, anti_key]
    )


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
def _div_toward_zero(value: int, divisor: int) -> int:
    """Integer division that treats both colours alike (// rounds negatives away from zero)."""
    quotient = value // divisor
    if quotient < 0 and quotient * divisor != value:
        quotient += 1
    return quotient


@numba.njit
def evaluate_pesto(state: np.ndarray) -> int:
    """The hand-written v11 evaluation (PeSTO + passed pawns + mobility), kept as the reference
    the net is measured against; the search uses evaluate_state below."""
    mg = 0
    eg = 0
    phase = 0
    occupied = state[fb.ALL_OCC]
    # Squares each side's pawns cover, two shifts per colour, hoisted out of the piece loop.
    white_pawn_attacks = ((state[fb.WP] & NOT_FILE_A) << np.uint64(7)) | (
        (state[fb.WP] & NOT_FILE_H) << np.uint64(9)
    )
    black_pawn_attacks = ((state[fb.BP] & NOT_FILE_H) >> np.uint64(7)) | (
        (state[fb.BP] & NOT_FILE_A) >> np.uint64(9)
    )
    for piece in range(12):
        color = piece // 6
        piece_type = piece % 6
        sign = 1 if color == fb.WHITE else -1
        own = state[fb.WHITE_OCC + color]
        safe = ~own & ~(black_pawn_attacks if color == fb.WHITE else white_pawn_attacks)
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
            elif piece_type != fb.KING:
                # Mobility rides along with the piece loop: the loop already has the square and
                # the sign, and a second pass would re-walk every bitboard for them.
                if piece_type == fb.KNIGHT:
                    targets = fb.KNIGHT_ATTACKS[square] & safe
                elif piece_type == fb.BISHOP:
                    targets = _bishop_attacks(square, occupied) & safe
                elif piece_type == fb.ROOK:
                    targets = _rook_attacks(square, occupied) & ~own
                else:
                    targets = (
                        _bishop_attacks(square, occupied) | _rook_attacks(square, occupied)
                    ) & ~own
                count = _popcount(targets) - int(MOBILITY_BASE[piece_type])
                mg += sign * count * int(MOBILITY_MG[piece_type])
                eg += sign * count * int(MOBILITY_EG[piece_type])
    if phase > PHASE_TOTAL:
        phase = PHASE_TOTAL
    score = _div_toward_zero(mg * phase + eg * (PHASE_TOTAL - phase), PHASE_TOTAL)
    if state[fb.BLACK_OCC] == state[fb.BK]:
        score += _mop_up(state, fb.WHITE, fb.BLACK)
    if state[fb.WHITE_OCC] == state[fb.WK]:
        score -= _mop_up(state, fb.BLACK, fb.WHITE)
    relative = score if int(state[fb.SIDE]) == fb.WHITE else -score
    return relative + TEMPO


# --------------------------------------------------------------------------------------
# NNUE evaluation: weights/nnue.npz, a 768-256-1 net trained on the engine's own self-play labels
# --------------------------------------------------------------------------------------

# Loaded once at import; numba freezes the arrays into the compiled search as constants.
NNUE_W1, NNUE_B1, NNUE_W2, NNUE_B2 = nn.load_weights()
NNUE_W1_B1_W2_B2 = (NNUE_W1, NNUE_B1, NNUE_W2, NNUE_B2)


@numba.njit
def evaluate_state(state: np.ndarray) -> int:
    """Static score relative to the side to move (allocating wrapper for tools and tests)."""
    feats = np.empty(32, dtype=np.int64)
    acc = np.empty(NNUE_B1.shape[0], dtype=np.int16)
    return evaluate_state_into(state, feats, acc)


@numba.njit
def evaluate_state_into(state: np.ndarray, feats: np.ndarray, acc: np.ndarray) -> int:
    """Static centipawn score relative to the side to move: the net plus the mop-up term.

    The net replaces material, piece-square tables, passed pawns and mobility. Mop-up stays: it
    encodes the referee's draw rule for a bare king, which the training labels never express.
    ``feats``/``acc`` are the searcher's scratch buffers (no allocation per node)."""
    relative = (
        nn.nnue_evaluate_into(state, NNUE_W1, NNUE_B1, NNUE_W2, NNUE_B2, feats, acc)
        - nn.MOVER_BIAS
    )
    side = int(state[fb.SIDE])
    score = relative if side == fb.WHITE else -relative
    if state[fb.BLACK_OCC] == state[fb.BK]:
        score += _mop_up(state, fb.WHITE, fb.BLACK)
    if state[fb.WHITE_OCC] == state[fb.WK]:
        score -= _mop_up(state, fb.BLACK, fb.WHITE)
    return score if side == fb.WHITE else -score


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
# Quiet history with gravity: every update moves a score toward +/-HISTORY_MAX by a step that
# shrinks as it gets there, so scores stay bounded (and below the losing captures at
# ORDER_KILLER - 1000 + gain) without a cap, and old evidence fades as new evidence arrives.
# A cutoff gives its quiet move the bonus and every quiet tried before it the same-sized malus.
HISTORY_MAX = 16384
HISTORY_BONUS_CAP = 1500
# Singular extensions: at depth >= 6 with a deep-enough hash score that is not an upper bound.
SINGULAR_MIN_DEPTH = 6
SINGULAR_MARGIN = 2  # sbeta = hash score - MARGIN * depth
SINGULAR_DOUBLE_MARGIN = 20  # a fail-low this far below sbeta extends two plies (non-PV only)
# Losing-move pruning in the main search (not late-move pruning, which lost three times here):
# outside the PV, near the leaves, a quiet whose static exchange loses material or whose history
# is bad, and a capture that loses material outright, are skipped rather than searched.
SEE_QUIET_MARGIN = -70  # per ply of remaining depth
SEE_CAPTURE_MARGIN = -30  # per ply squared
HISTORY_PRUNE_MARGIN = -1024  # per ply, at depth <= 3
SEE_PRUNE_MAX_DEPTH = 8
SKIPPED_SCORE = -(1 << 30)  # marks a move the loop did not search (illegal or futile)
# Continuation history: indexed by (piece code * 64 + to-square) of the previous move and of the
# move being scored; the same gravity update as the main history. Correction history: per side
# and pawn structure (16-bit hash of the two pawn bitboards), an exponential average of
# (search score - static eval) in 1/CORR_SCALE cp, added to the static eval before pruning.
# The improving flag: whether our static score rose since our previous turn. A node that is
# improving can afford a larger reverse-futility margin and one ply less reduction.
CONT_SIZE = 12 * 64
CORR_ENTRIES = 1 << 16
CORR_SCALE = 256
CORR_MAX = 96 * CORR_SCALE

# Selective search. Margins in centipawns; the LMR table is floor(ln(depth) * ln(index + 1) / 2.25),
# at least one ply, indexed by depth and by the number of legal moves already searched.
REVERSE_FUTILITY_MARGIN = 120
FUTILITY_MARGIN = 120
LMR_TABLE = np.zeros((MAX_DEPTH + 2, 256), dtype=np.int8)
for _depth in range(MAX_DEPTH + 2):
    for _index in range(256):
        LMR_TABLE[_depth, _index] = max(
            1, int(math.log(max(_depth, 2)) * math.log(_index + 1) / 2.25)
        )
SEARCH_VALUE = np.array((100, 320, 330, 500, 900, 20_000), dtype=np.int32)

TT_BITS = 22
QS_STORE_STAND_PAT = False  # also record stand-pat fail-highs (compile-time constant for numba)
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
def _out_of_time(stats: np.ndarray) -> bool:
    """Wall-clock backstop for the node-based budget: stats[2] holds a perf_counter_ns deadline
    (0 = none). Checked every 16k nodes, so the cost of leaving compiled code is negligible, and
    a machine slower than the nodes/second estimate can no longer overrun the clock."""
    if stats[2] <= 0:
        return False
    with numba.objmode(now="int64"):
        now = time.perf_counter_ns()
    return now >= stats[2]


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
def _is_draw(
    state: np.ndarray, ply: int, rep_keys: np.ndarray, rep_count: int
) -> tuple[bool, np.uint64]:
    """(draw by rule or repetition, the node's canonical key) — the key is computed once here
    and stored by the caller, instead of twice per node."""
    key = _canonical_key(state)
    halfmove = int(state[fb.HALFMOVE])
    if halfmove >= 100 or _is_insufficient(state):
        return True, key
    if halfmove < 4:
        return False, key
    earliest = max(0, rep_count - halfmove)
    # Equal positions have equal side to move, so only every second historical ply can match.
    index = rep_count - 2
    while index >= earliest:
        if rep_keys[index] == key:
            return True, key
        index -= 2
    return False, key


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
def _least_valuable_attacker(
    state: np.ndarray, occupied: np.uint64, side: int, square: int
) -> int:
    """Cheapest piece of ``side`` still standing in ``occupied`` that attacks ``square``.

    Returns ``from_square | (piece_type << 6)``, or -1 when that side has no attacker left.
    Slider attacks are re-derived from ``occupied`` on every call, so a piece the swap-off has
    already consumed uncovers whatever stood behind it (the x-ray discovery SEE needs).
    ``SEARCH_VALUE`` rises with the piece-type index, so scanning P, N, B, R, Q, K in order
    already yields the least valuable attacker, and the king is only ever reached last.
    """
    offset = side * 6
    pawns = state[offset + fb.PAWN] & occupied & fb.PAWN_ATTACKS[1 - side, square]
    if pawns:
        return fb.lsb_square(pawns) | (fb.PAWN << 6)
    knights = state[offset + fb.KNIGHT] & occupied & fb.KNIGHT_ATTACKS[square]
    if knights:
        return fb.lsb_square(knights) | (fb.KNIGHT << 6)
    diagonal = fb.U64_ZERO
    if (state[offset + fb.BISHOP] | state[offset + fb.QUEEN]) & occupied:
        diagonal = fb.bishop_attacks(square, occupied)
    bishops = state[offset + fb.BISHOP] & occupied & diagonal
    if bishops:
        return fb.lsb_square(bishops) | (fb.BISHOP << 6)
    orthogonal = fb.U64_ZERO
    if (state[offset + fb.ROOK] | state[offset + fb.QUEEN]) & occupied:
        orthogonal = fb.rook_attacks(square, occupied)
    rooks = state[offset + fb.ROOK] & occupied & orthogonal
    if rooks:
        return fb.lsb_square(rooks) | (fb.ROOK << 6)
    queens = state[offset + fb.QUEEN] & occupied & (diagonal | orthogonal)
    if queens:
        return fb.lsb_square(queens) | (fb.QUEEN << 6)
    kings = state[offset + fb.KING] & occupied & fb.KING_ATTACKS[square]
    if kings:
        return fb.lsb_square(kings) | (fb.KING << 6)
    return -1


@numba.njit
def _see_gain(
    state: np.ndarray, occupied: np.uint64, side: int, square: int, standing: int
) -> int:
    """Material ``side`` wins by continuing the exchange on ``square``, in SEARCH_VALUE units.

    ``standing`` is what the piece now sitting on ``square`` is worth.  Recapturing is optional,
    so a side facing a losing continuation simply stops and the branch is worth nothing; that
    ``max(0, ...)`` is what makes the swap-off a negamax rather than a plain sum.
    """
    packed = _least_valuable_attacker(state, occupied, side, square)
    if packed < 0:
        return 0
    from_square = packed & 63
    piece_type = packed >> 6
    remaining = occupied & ~(fb.U64_ONE << np.uint64(from_square))
    # A king may only take the last defender: stepping onto a square the other side still
    # attacks is illegal, so the exchange ends here instead of continuing.
    if piece_type == fb.KING and _least_valuable_attacker(state, remaining, 1 - side, square) >= 0:
        return 0
    gain = standing - _see_gain(
        state, remaining, 1 - side, square, int(SEARCH_VALUE[piece_type])
    )
    return gain if gain > 0 else 0


@numba.njit
def see(state: np.ndarray, move: np.uint32) -> int:
    """Static exchange evaluation of ``move``, in SEARCH_VALUE units (a pawn is 100).

    Plays the move, then lets each side recapture with its least valuable attacker until one
    prefers to stop.  En passant scores the pawn behind the target; a promotion capture adds the
    promoted piece's value less a pawn, and leaves the promoted piece standing on the square.

    Deliberate simplifications, shared with the reference in ``test_see.py``: pins and
    discovered attacks are ignored (as in every SEE), and a pawn that reaches the last rank
    *during* the swap-off is counted as a pawn.
    """
    to_square = fb.move_to(move)
    from_square = fb.move_from(move)
    side = int(state[fb.SIDE])
    occupied = state[fb.ALL_OCC] & ~(fb.U64_ONE << np.uint64(from_square))
    if int(move) & fb.FLAG_EN_PASSANT:
        captured_square = to_square - 8 if side == fb.WHITE else to_square + 8
        occupied &= ~(fb.U64_ONE << np.uint64(captured_square))
        gain = int(SEARCH_VALUE[fb.PAWN])
    else:
        victim = _piece_type_at(state, to_square)
        gain = 0 if victim == fb.NO_PIECE else int(SEARCH_VALUE[victim])
    promotion = fb.move_promotion(move)
    if promotion:
        gain += int(SEARCH_VALUE[promotion]) - int(SEARCH_VALUE[fb.PAWN])
        standing = int(SEARCH_VALUE[promotion])
    else:
        mover = _piece_type_at(state, from_square)
        standing = 0 if mover == fb.NO_PIECE else int(SEARCH_VALUE[mover])
    return gain - _see_gain(state, occupied, 1 - side, to_square, standing)


@numba.njit
def _see_loses_material(state: np.ndarray, move: np.uint32) -> bool:
    """``see(state, move) < 0``, deciding the cheap cases without running the swap-off.

    Stopping after the first recapture is always available to us, so a capture is worth at least
    ``victim - attacker``: whenever the victim is the more valuable piece the move cannot lose
    material and the exchange never has to be walked.  That covers every en-passant capture and
    every capture-promotion (a promotion wins at least ``victim - pawn``), so those reach the
    swap-off only through ``see`` itself.
    """
    victim = _captured_type(state, move)
    if victim == fb.NO_PIECE:
        return False
    attacker = _piece_type_at(state, fb.move_from(move))
    if attacker == fb.NO_PIECE or SEARCH_VALUE[victim] >= SEARCH_VALUE[attacker]:
        return False
    return see(state, move) < 0


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
        # A capture that loses material drops below the killers but stays above every quiet:
        # it is still forcing, and history scores stay within +/-HISTORY_MAX.
        if attacker != fb.NO_PIECE and SEARCH_VALUE[victim] < SEARCH_VALUE[attacker]:
            gain = see(state, move)
            if gain < 0:
                return ORDER_KILLER - 1000 + gain
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
    cont_hist: np.ndarray,
    prev_idx: int,
) -> None:
    """Score every move once (same scores as _move_score plus the continuation history of the
    previous move when ``prev_idx`` >= 0; side and killers read once per node, the capture test
    from the move's flag bit)."""
    side = int(state[fb.SIDE])
    killer0 = killers[ply, 0]
    killer1 = killers[ply, 1]
    for index in range(count):
        move = moves[index]
        bits = int(move)
        if move == tt_move and tt_move != 0:
            scores[index] = ORDER_TT
            continue
        promotion = (bits >> fb.PROMOTION_SHIFT) & fb.PROMOTION_MASK
        if bits & fb.FLAG_CAPTURE:
            victim = _captured_type(state, move)
            attacker = _piece_type_at(state, bits & 63)
            if attacker != fb.NO_PIECE and SEARCH_VALUE[victim] < SEARCH_VALUE[attacker]:
                gain = see(state, move)
                if gain < 0:
                    scores[index] = ORDER_KILLER - 1000 + gain
                    continue
            score = ORDER_CAPTURE + int(SEARCH_VALUE[victim]) * 10
            if attacker != fb.NO_PIECE:
                score -= int(SEARCH_VALUE[attacker]) // 10
            if promotion:
                score += int(SEARCH_VALUE[promotion])
            scores[index] = score
        elif promotion:
            scores[index] = ORDER_PROMOTION + int(SEARCH_VALUE[promotion])
        elif move == killer0:
            scores[index] = ORDER_KILLER
        elif move == killer1:
            scores[index] = ORDER_KILLER - 1
        else:
            from_square = bits & 63
            to_square = (bits >> 6) & 63
            score = int(history[side, from_square * 64 + to_square])
            if prev_idx >= 0:
                piece = _piece_code(state, side, from_square)
                score += int(cont_hist[prev_idx, piece * 64 + to_square])
            scores[index] = score


@numba.njit
def _pick_next(moves: np.ndarray, scores: np.ndarray, start: int, count: int) -> None:
    """Swap the best-scoring move in [start, count) into position start."""
    best = start
    best_score = scores[start]
    for index in range(start + 1, count):
        if scores[index] > best_score:
            best = index
            best_score = scores[index]
    if best != start:
        moves[start], moves[best] = moves[best], moves[start]
        scores[start], scores[best] = scores[best], scores[start]


@numba.njit(inline="always")
def _history_bonus(depth: int) -> int:
    return min(16 * depth * depth + 32 * depth, HISTORY_BONUS_CAP)


@numba.njit(inline="always")
def _bump_history(history: np.ndarray, side: int, move: np.uint32, bonus: int) -> None:
    """Gravity update: the step shrinks as the score approaches +/-HISTORY_MAX."""
    slot = fb.move_from(move) * 64 + fb.move_to(move)
    value = int(history[side, slot])
    history[side, slot] = value + bonus - _div_toward_zero(value * abs(bonus), HISTORY_MAX)


@numba.njit(inline="always")
def _bump_table(table: np.ndarray, row: int, col: int, bonus: int) -> None:
    value = int(table[row, col])
    table[row, col] = value + bonus - _div_toward_zero(value * abs(bonus), HISTORY_MAX)


@numba.njit(inline="always")
def _piece_code(state: np.ndarray, side: int, square: int) -> int:
    """0-11 piece code of ``side``'s piece on ``square`` (pawn if the square is empty)."""
    piece_type = _piece_type_at(state, square)
    if piece_type < 0 or piece_type >= 6:
        piece_type = 0
    return side * 6 + piece_type


@numba.njit(inline="always")
def _nonpawn_slot(state: np.ndarray) -> int:
    """Hash of both sides' pieces other than pawns and kings: a second correction table, since
    the evaluation's error depends on the material left as well as on the pawn structure."""
    key = np.uint64(0)
    for piece in (fb.KNIGHT, fb.BISHOP, fb.ROOK, fb.QUEEN):
        key ^= (state[piece] + np.uint64(piece)) * np.uint64(0x9E3779B97F4A7C15)
        key ^= (state[6 + piece] + np.uint64(piece)) * np.uint64(0xC2B2AE3D27D4EB4F)
    return int(key >> np.uint64(48))


@numba.njit(inline="always")
def _pawn_slot(state: np.ndarray) -> int:
    key = state[fb.PAWN] * np.uint64(0x9E3779B97F4A7C15)
    key ^= state[6 + fb.PAWN] * np.uint64(0xC2B2AE3D27D4EB4F)
    return int(key >> np.uint64(48))


@numba.njit
def _store_killer(move: np.uint32, ply: int, killers: np.ndarray) -> None:
    if killers[ply, 0] != move:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = move


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
    state[fb.SIDE] = state[fb.SIDE] ^ fb.U64_ONE
    state[fb.ZOBRIST] = key ^ fb.ZOBRIST_SIDE
    return old_ep, old_halfmove, old_hash


@numba.njit
def _unmake_null(
    state: np.ndarray, undo: tuple[np.uint64, np.uint64, np.uint64]
) -> None:
    old_ep, old_halfmove, old_hash = undo
    state[fb.SIDE] = state[fb.SIDE] ^ fb.U64_ONE
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
    corr_hist: np.ndarray,
    cont_hist: np.ndarray,
    move_stack: np.ndarray,
    eval_feats: np.ndarray,
    eval_acc: np.ndarray,
    corr_np: np.ndarray,
    eval_stack: np.ndarray,
) -> int:
    stats[0] += 1
    if stats[1] or stats[0] >= node_limit or (stats[0] & 16383 == 0 and _out_of_time(stats)):
        stats[1] = 1
        return 0
    if ply >= MAX_PLY:
        return evaluate_state_into(state, eval_feats, eval_acc)
    draw, node_key = _is_draw(state, ply, rep_keys, rep_count)
    if ply > 0 and draw:
        return _draw_score(ply)

    # Transposition table in quiescence: captures transpose constantly, and every stored entry
    # (main search or an earlier quiescence visit) is at least as deep as this node, so any
    # bound that fits the window cuts. The stored move leads the capture ordering.
    raw_key = state[fb.ZOBRIST]
    slot = int(raw_key & np.uint64(tt_mask))
    tt_move = np.uint32(0)
    if tt_depth[slot] >= 0 and tt_keys[slot] == raw_key:
        tt_move = tt_moves[slot]
        tt_score = _from_tt(int(tt_scores[slot]), ply)
        tt_flag = int(tt_flags[slot])
        if (
            tt_flag == EXACT
            or (tt_flag == LOWER and tt_score >= beta)
            or (tt_flag == UPPER and tt_score <= alpha)
        ):
            return tt_score
    alpha_original = alpha

    in_check = fb.is_in_check(state, int(state[fb.SIDE]))
    stand_pat = -INF if in_check else evaluate_state_into(state, eval_feats, eval_acc)
    if not in_check:
        if stand_pat >= beta:
            if QS_STORE_STAND_PAT and tt_depth[slot] <= 0:
                tt_keys[slot] = raw_key
                tt_depth[slot] = 0
                tt_scores[slot] = _to_tt(stand_pat, ply)
                tt_flags[slot] = LOWER
                tt_moves[slot] = np.uint32(0)
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    rep_keys[rep_count] = node_key
    moves = move_buffers[ply]
    scores = score_buffers[ply]
    if in_check:
        count = fb.generate_pseudo_into(state, moves)
    else:
        count = fb.generate_captures_into(state, moves)
    _order_moves(state, moves, scores, count, tt_move, ply, killers, history, cont_hist, -1)
    moving_color = int(state[fb.SIDE])
    legal_count = 0
    best = stand_pat
    best_move = np.uint32(0)
    for index in range(count):
        _pick_next(moves, scores, index, count)
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
        # Losing captures cannot improve a quiescence score, and searching them is what makes
        # the tree explode.  Promotions and evasions stay in: both change the material picture
        # in ways the swap-off does not model.
        if (
            not in_check
            and victim != fb.NO_PIECE
            and promotion == 0
            and _see_loses_material(state, move)
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
            corr_hist,
            cont_hist,
            move_stack,
            eval_feats,
            eval_acc,
            corr_np,
            eval_stack,
        )
        fb.unmake_move(state, move, undo)
        if stats[1]:
            return 0
        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
                if score >= beta:
                    break
    flag = UPPER
    if in_check and legal_count == 0:
        best = -MATE + ply
        flag = EXACT
    elif best >= beta:
        flag = LOWER
    elif best > alpha_original:
        flag = EXACT
    # Depth 0 never evicts a main-search entry: only empty or quiescence slots are overwritten.
    if tt_depth[slot] <= 0:
        tt_keys[slot] = raw_key
        tt_depth[slot] = 0
        tt_scores[slot] = _to_tt(best, ply)
        tt_flags[slot] = flag
        tt_moves[slot] = best_move
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
    corr_hist: np.ndarray,
    cont_hist: np.ndarray,
    move_stack: np.ndarray,
    eval_feats: np.ndarray,
    eval_acc: np.ndarray,
    corr_np: np.ndarray,
    eval_stack: np.ndarray,
    excluded: np.uint32,
) -> int:
    """``excluded`` != 0 marks a singular-test search: that move is skipped and the node does not
    read or write the transposition table, prune with the null move, or reduce for a missing hash
    move, since none of those describe the position with a move removed."""
    stats[0] += 1
    if stats[1] or stats[0] >= node_limit or (stats[0] & 16383 == 0 and _out_of_time(stats)):
        stats[1] = 1
        return 0
    if ply >= MAX_PLY:
        return evaluate_state_into(state, eval_feats, eval_acc)
    draw, node_key = _is_draw(state, ply, rep_keys, rep_count)
    if ply > 0 and draw:
        return _draw_score(ply)
    if ply > 0:  # mate-distance pruning: no line from here can beat a mate already known
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

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
            corr_hist,
            cont_hist,
            move_stack,
            eval_feats,
            eval_acc,
            corr_np,
            eval_stack,
        )

    raw_key = state[fb.ZOBRIST]
    slot = int(raw_key & np.uint64(tt_mask))
    tt_move = np.uint32(0)
    tt_hit_depth = -1
    tt_hit_score = 0
    tt_hit_flag = EXACT
    if tt_depth[slot] >= 0 and tt_keys[slot] == raw_key:
        tt_move = tt_moves[slot]
        tt_hit_depth = int(tt_depth[slot])
        tt_hit_score = _from_tt(int(tt_scores[slot]), ply)
        tt_hit_flag = int(tt_flags[slot])
        if (
            excluded == 0
            and tt_hit_depth >= depth
            and (
                tt_hit_flag == EXACT
                or (tt_hit_flag == LOWER and tt_hit_score >= beta)
                or (tt_hit_flag == UPPER and tt_hit_score <= alpha)
            )
        ):
            return tt_hit_score

    if tt_move == 0 and depth >= 4 and not in_check and excluded == 0:
        depth -= 1  # internal iterative reduction: no hash move, search a ply shallower first

    if ply + 1 < MAX_PLY:  # the child inherits no stale killers from a sibling subtree
        killers[ply + 1, 0] = np.uint32(0)
        killers[ply + 1, 1] = np.uint32(0)
    rep_keys[rep_count] = node_key
    side = int(state[fb.SIDE])
    prev_idx = int(move_stack[ply - 1]) if ply > 0 else -1
    corr_slot = _pawn_slot(state)
    corr_np_slot = _nonpawn_slot(state)
    zero_window = beta - alpha == 1
    nmp_ok = (
        excluded == 0
        and depth >= 3
        and ply > 0
        and not in_check
        and beta < MATE_BOUND
        and state[fb.WHITE_OCC + side]
        & ~(state[side * 6 + fb.PAWN] | state[side * 6 + fb.KING])
        != 0
    )
    rfp_ok = (
        excluded == 0
        and depth <= 3
        and not in_check
        and zero_window
        and -MATE_BOUND < beta < MATE_BOUND
        and state[fb.WHITE_OCC + side] != state[side * 6 + fb.KING]
    )
    raw_static = -INF
    static_eval = -INF
    improving = False
    if nmp_ok or rfp_ok:
        raw_static = evaluate_state_into(state, eval_feats, eval_acc)
        static_eval = raw_static + _div_toward_zero(
            int(corr_hist[side, corr_slot]) + int(corr_np[side, corr_np_slot]), CORR_SCALE
        )
        eval_stack[ply] = static_eval
        if ply >= 2 and eval_stack[ply - 2] != -INF:
            improving = static_eval > eval_stack[ply - 2]
    if nmp_ok and static_eval >= beta:
        # a static score far above beta makes the null move safer: reduce more
        extra = (static_eval - beta) // 200
        if extra > 3:
            extra = 3
        reduction = 2 + depth // 6 + int(extra)
        move_stack[ply] = -1
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
            corr_hist,
            cont_hist,
            move_stack,
            eval_feats,
            eval_acc,
            corr_np,
            eval_stack,
 np.uint32(0),
        )
        _unmake_null(state, null_undo)
        if stats[1]:
            return 0
        if null_score >= beta:
            return null_score

    # Static pruning at non-PV nodes near the horizon. Reverse futility: a static score far
    # above beta proves the bound without a search. Futility (in the loop below): a quiet move
    # cannot lift a static score far below alpha. Neither fires in check, with mate bounds, or
    # for a bare king, whose stalemates are exactly what a static score gets wrong.
    if rfp_ok and static_eval - REVERSE_FUTILITY_MARGIN * (depth - improving) >= beta:
        return static_eval
    futile = (
        depth <= 2
        and static_eval != -INF
        and -MATE_BOUND < alpha < MATE_BOUND
        and static_eval + FUTILITY_MARGIN * depth <= alpha
    )

    # Singular extension. When the hash move is the only move that holds, the position turns on
    # it, so search it deeper. The test searches this node with that move excluded, at half depth
    # against a beta well below the hash score: a fail-low means nothing else comes close. Done
    # before the move list is generated, because the test search reuses move_buffers[ply].
    singular_extension = 0
    if (
        excluded == 0
        and depth >= SINGULAR_MIN_DEPTH
        and tt_move != 0
        and ply > 0
        and tt_hit_depth >= depth - 3
        and tt_hit_flag != UPPER
        and -MATE_BOUND < tt_hit_score < MATE_BOUND
    ):
        sbeta = tt_hit_score - SINGULAR_MARGIN * depth
        singular_score = _negamax(
            state,
            (depth - 1) // 2,
            sbeta - 1,
            sbeta,
            ply,
            rep_keys,
            rep_count,
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
            corr_hist,
            cont_hist,
            move_stack,
            eval_feats,
            eval_acc,
            corr_np,
            eval_stack,
 tt_move,
        )
        if stats[1]:
            return 0
        if singular_score < sbeta:
            singular_extension = 1
            if zero_window and singular_score < sbeta - SINGULAR_DOUBLE_MARGIN:
                singular_extension = 2  # by a distance: worth two plies outside the PV
        elif sbeta >= beta:
            return sbeta  # multicut: a second move already beats beta at reduced depth
        elif tt_hit_score >= beta:
            singular_extension = -1  # many moves fail high here; this one need not be extended

    moves = move_buffers[ply]
    scores = score_buffers[ply]
    count = fb.generate_pseudo_into(state, moves)
    _order_moves(state, moves, scores, count, tt_move, ply, killers, history, cont_hist, prev_idx)
    alpha_original = alpha
    best_score = -INF
    best_move = np.uint32(0)
    legal_count = 0
    can_reduce = depth >= 3 and not in_check
    prune_ok = zero_window and not in_check and depth <= SEE_PRUNE_MAX_DEPTH
    for index in range(count):
        _pick_next(moves, scores, index, count)
        move = moves[index]
        if move == excluded:
            scores[index] = SKIPPED_SCORE
            continue
        quiet = int(move) & (fb.FLAG_CAPTURE | (fb.PROMOTION_MASK << fb.PROMOTION_SHIFT)) == 0
        if futile and quiet and legal_count > 0:
            scores[index] = SKIPPED_SCORE
            continue
        if prune_ok and legal_count > 0 and best_score > -MATE_BOUND:
            # One exchange evaluation against one threshold. Written with a single `see` call
            # site on purpose: numba inlines it, and the two-call-site form cost 45% more compile
            # for exactly the same search.
            skip = False
            if quiet:
                slot_q = (int(move) & 63) * 64 + fb.move_to(move)
                threshold = SEE_QUIET_MARGIN * depth
                skip = depth <= 3 and int(history[side, slot_q]) < HISTORY_PRUNE_MARGIN * depth
            else:
                threshold = SEE_CAPTURE_MARGIN * depth * depth
            if not skip:
                skip = see(state, move) < threshold
            if skip:
                scores[index] = SKIPPED_SCORE
                continue
        undo = fb.make_move(state, move)
        if fb.is_in_check(state, side):
            fb.unmake_move(state, move, undo)
            scores[index] = SKIPPED_SCORE
            continue
        to_square = fb.move_to(move)
        move_stack[ply] = _piece_code(state, side, to_square) * 64 + to_square
        child_depth = depth - 1 + (singular_extension if move == tt_move else 0)
        reduction = 0
        if (
            can_reduce
            and legal_count >= 3
            and quiet
            and not fb.is_in_check(state, int(state[fb.SIDE]))
        ):
            reduction = int(LMR_TABLE[depth, legal_count])
            if move == killers[ply, 0] or move == killers[ply, 1]:
                reduction -= 1
            if improving:
                reduction -= 1
            # history-scaled: a quiet with a strong record is reduced up to two plies less, one
            # with a poor record up to two plies more
            reduction -= _div_toward_zero(
                int(history[side, fb.move_from(move) * 64 + fb.move_to(move)]), HISTORY_MAX // 2
            )
            if reduction < 0:
                reduction = 0
        if legal_count == 0:
            score = -_negamax(
                state,
                child_depth,
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
                corr_hist,
                cont_hist,
                move_stack,
                eval_feats,
                eval_acc,
                corr_np,
                eval_stack,
 np.uint32(0),
            )
        else:
            score = -_negamax(
                state,
                child_depth - reduction,
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
                corr_hist,
                cont_hist,
                move_stack,
                eval_feats,
                eval_acc,
                corr_np,
                eval_stack,
 np.uint32(0),
            )
            if reduction and score > alpha and not stats[1]:
                score = -_negamax(
                    state,
                    child_depth,
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
                    corr_hist,
                    cont_hist,
                    move_stack,
                    eval_feats,
                    eval_acc,
                    corr_np,
                    eval_stack,
 np.uint32(0),
                )
            if score > alpha and score < beta and not stats[1]:
                score = -_negamax(
                    state,
                    child_depth,
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
                    corr_hist,
                    cont_hist,
                    move_stack,
                    eval_feats,
                    eval_acc,
                    corr_np,
                    eval_stack,
 np.uint32(0),
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
                    _store_killer(move, ply, killers)
                    bonus = _history_bonus(depth)
                    _bump_history(history, side, move, bonus)
                    if prev_idx >= 0:
                        _bump_table(
                            cont_hist, prev_idx,
                            _piece_code(state, side, fb.move_from(move)) * 64 + fb.move_to(move),
                            bonus,
                        )
                    for earlier in range(index):
                        tried = moves[earlier]
                        if scores[earlier] != SKIPPED_SCORE and int(tried) & (
                            fb.FLAG_CAPTURE | (fb.PROMOTION_MASK << fb.PROMOTION_SHIFT)
                        ) == 0:
                            _bump_history(history, side, tried, -bonus)
                            if prev_idx >= 0:
                                _bump_table(
                                    cont_hist, prev_idx,
                                    _piece_code(state, side, fb.move_from(tried)) * 64
                                    + fb.move_to(tried),
                                    -bonus,
                                )
                break

    if legal_count == 0:
        if excluded != 0:
            return alpha  # only the excluded move was legal: not a mate, just nothing to try
        return -MATE + ply if in_check else _draw_score(ply)
    if excluded != 0:
        return best_score  # the table describes the position with every move available
    flag = UPPER
    if best_score >= beta:
        flag = LOWER
    elif best_score > alpha_original:
        flag = EXACT
    if (
        not in_check
        and raw_static != -INF
        and -MATE_BOUND < best_score < MATE_BOUND
        and int(best_move) & (fb.FLAG_CAPTURE | (fb.PROMOTION_MASK << fb.PROMOTION_SHIFT)) == 0
        and (
            flag == EXACT
            or (flag == LOWER and best_score > raw_static)
            or (flag == UPPER and best_score < raw_static)
        )
    ):
        # the eval was off by (best_score - raw_static) for this pawn structure: move the
        # correction toward it, faster from deeper searches
        weight = min(depth, 15) + 1
        value = int(corr_hist[side, corr_slot])
        value = _div_toward_zero(
            value * (CORR_SCALE - weight) + (best_score - raw_static) * CORR_SCALE * weight,
            CORR_SCALE,
        )
        corr_hist[side, corr_slot] = max(-CORR_MAX, min(CORR_MAX, value))
        value_np = _div_toward_zero(
            int(corr_np[side, corr_np_slot]) * (CORR_SCALE - weight)
            + (best_score - raw_static) * CORR_SCALE * weight,
            CORR_SCALE,
        )
        corr_np[side, corr_np_slot] = max(-CORR_MAX, min(CORR_MAX, value_np))
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
    corr_hist: np.ndarray,
    cont_hist: np.ndarray,
    move_stack: np.ndarray,
    eval_feats: np.ndarray,
    eval_acc: np.ndarray,
    corr_np: np.ndarray,
    eval_stack: np.ndarray,
) -> tuple[int, np.uint32, bool]:
    """Search one completed iteration, aborting exactly at the caller's node limit."""
    stats[0] += 1
    if stats[1] or stats[0] >= node_limit or (stats[0] & 16383 == 0 and _out_of_time(stats)):
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
    _order_moves(state, moves, scores, count, tt_move, 0, killers, history, cont_hist, -1)
    alpha_original = alpha
    best_score = -INF
    best_move = np.uint32(0)
    legal_count = 0
    for index in range(count):
        _pick_next(moves, scores, index, count)
        move = moves[index]
        undo = fb.make_move(state, move)
        if fb.is_in_check(state, side):
            fb.unmake_move(state, move, undo)
            continue
        to_square = fb.move_to(move)
        move_stack[0] = _piece_code(state, side, to_square) * 64 + to_square
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
                corr_hist,
                cont_hist,
                move_stack,
                eval_feats,
                eval_acc,
                corr_np,
                eval_stack,
 np.uint32(0),
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
                corr_hist,
                cont_hist,
                move_stack,
                eval_feats,
                eval_acc,
                corr_np,
                eval_stack,
 np.uint32(0),
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
                    corr_hist,
                    cont_hist,
                    move_stack,
                    eval_feats,
                    eval_acc,
                    corr_np,
                    eval_stack,
 np.uint32(0),
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


def budget(time_left_ms: int, fullmove: int = 1) -> tuple[float, float]:
    """Return soft and hard wall-time targets in milliseconds.

    The soft target divides the remaining time by an estimate of the moves still to come, which
    shrinks as the game goes on: a flat 1/40 left 55 s of 120 s unused in the first ladder game.
    The hard target bounds a single move. The platform suspends the process while the opponent
    thinks, so nothing is ever banked from the opponent's clock and the budget must stand alone."""
    usable = max(time_left_ms - RESERVE_MS, 10)
    # front-loaded since v12.6 (+44 +/- 22 at the real clock): ladder rounds 33 and 36 were lost on
    # 2-4 s thinks with 70-110 s on the clock; the 0.5 s increment carries the ending
    moves_to_go = max(12.0, 26.0 - 0.5 * fullmove)
    soft = usable / moves_to_go + INCREMENT_MS * 0.6
    hard = min(soft * 3.0, usable * 0.25)
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
        self.cont_hist = np.zeros((CONT_SIZE, CONT_SIZE), dtype=np.int32)
        self.corr_hist = np.zeros((2, CORR_ENTRIES), dtype=np.int32)
        self.corr_np = np.zeros((2, CORR_ENTRIES), dtype=np.int32)
        self.eval_stack = np.full(MAX_PLY + 4, -INF, dtype=np.int32)
        self.move_stack = np.full(MAX_PLY + 2, -1, dtype=np.int64)
        self.eval_feats = np.zeros(32, dtype=np.int64)  # nnue scratch, no allocation per node
        self.eval_acc = np.zeros(NNUE_B1.shape[0], dtype=np.int16)
        self.move_buffers = np.zeros((MAX_PLY + 2, fb.MAX_MOVES), dtype=np.uint32)
        self.score_buffers = np.zeros((MAX_PLY + 2, fb.MAX_MOVES), dtype=np.int32)
        self.rep_keys = np.zeros(MAX_GAME_HISTORY + MAX_PLY + 4, dtype=np.uint64)
        self.stats = np.zeros(3, dtype=np.int64)  # nodes, abort flag, wall-clock deadline (ns)
        self.stop_requested = False  # set by abort(); survives ponder()'s stats reset
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
            self.corr_hist,
            self.cont_hist,
            self.move_stack,
            self.eval_feats,
            self.eval_acc,
            self.corr_np,
            self.eval_stack,
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
        self.cont_hist >>= 1

        best_move = legal[0]
        best_score = 0
        depth_reached = 0
        previous_iteration_ms = 0.0
        previous_score = 0
        stable_iterations = 0
        effective_soft_ms = soft_ms
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
            self.stats[2] = time.perf_counter_ns() + int(remaining_ms * 1e6)
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
            stable_iterations = stable_iterations + 1 if candidate == best_move else 0
            score_drop = previous_score - int(score) if depth >= 5 else 0
            best_move = candidate
            best_score = int(score)
            previous_score = best_score
            depth_reached = depth
            if abs(best_score) >= MATE_BOUND:
                break
            # A score that fell sharply since the last iteration deserves more time (the move may
            # be about to change); a move unchanged for four iterations at a useful depth, less.
            if score_drop >= 60:
                effective_soft_ms = min(hard_ms, soft_ms * 1.8)
            elif depth >= 8 and stable_iterations >= 4:
                effective_soft_ms = min(effective_soft_ms, soft_ms * 0.6)
            total_elapsed_ms = (time.perf_counter() - start) * 1000.0
            growth = 3.0
            if previous_iteration_ms >= 0.5:
                growth = min(6.0, max(1.7, iteration_elapsed_ms / previous_iteration_ms))
            if total_elapsed_ms + iteration_elapsed_ms * growth * 0.55 >= effective_soft_ms:
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
        # abort() may have been called before we got here (a fast opponent reply); a reset must
        # not erase it, or this search would run with no limit while the main thread waits.
        if self.stop_requested:
            self.stats[1] = 1
        for depth in range(1, MAX_DEPTH + 1):
            if self.stop_requested:
                break
            score, _, aborted = self._iteration(state, depth, -INF, INF, rep_count, 1 << 62)
            if aborted or abs(score) >= MATE_BOUND:
                break
        self.nodes = int(self.stats[0])
        return self.nodes

    def abort(self) -> None:
        """Ask a running search (on any thread) to stop at its next node."""
        self.stop_requested = True
        self.stats[1] = 1

    def clear_abort(self) -> None:
        self.stop_requested = False
        self.stats[1] = 0


def _warm_search() -> float:
    started = time.perf_counter()
    probe = Searcher(tt_bits=15)
    state = fb.from_fen(chess.STARTING_FEN)
    evaluate_state(state)
    _canonical_key(state)
    # SEE is reached from _move_score below, but compile it explicitly so a future change to
    # ordering cannot quietly move its compilation into the first game move.
    exchange = fb.from_fen("r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
    capture = fb.find_legal_move(exchange, "b5c6")
    see(exchange, capture)
    _see_loses_material(exchange, capture)
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
