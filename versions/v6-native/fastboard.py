"""Numba-native bitboard state, legal move generation, and reversible moves.

The representation is a flat 21-word ``uint64`` array: twelve piece bitboards, white/black/all
occupancy, side, castling rights, en-passant square, both FEN clocks, and a Zobrist key.  Flat
arrays cross the Python/Numba boundary cheaply and are easy for a future search to copy into a
preallocated ply stack.

Knight, king, and pawn attacks are precomputed.  Bishops, rooks, and queens use direct ray scans.
That choice trades magic-bitboard peak speed for a small, auditable correctness surface.  Public
``generate_moves`` creates pseudo-legal moves, then filters them with reversible in-place moves
and a king attack test.  Castling transit-square safety is checked while creating the pseudo move.

Moves are packed into ``uint32`` as from(6), to(6), promotion(3), and flags.  Promotion codes are
1=knight, 2=bishop, 3=rook, 4=queen.  ``make_move`` updates every bitboard, occupancy, metadata,
and the Zobrist key incrementally; its fixed tuple is consumed by ``unmake_move``.
"""

from __future__ import annotations

import time
from typing import Final

import numba
import numpy as np

# Piece indexes: six white piece types, followed by the corresponding black types.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
WHITE, BLACK = 0, 1
WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK = range(12)

WHITE_OCC, BLACK_OCC, ALL_OCC = 12, 13, 14
SIDE, CASTLING, EP_SQUARE, HALFMOVE, FULLMOVE, ZOBRIST = 15, 16, 17, 18, 19, 20
STATE_SIZE: Final = 21
NO_SQUARE: Final = 64
NO_PIECE: Final = 12
MAX_MOVES: Final = 256

CASTLE_WHITE_KING = 1
CASTLE_WHITE_QUEEN = 2
CASTLE_BLACK_KING = 4
CASTLE_BLACK_QUEEN = 8

PROMOTION_SHIFT = 12
PROMOTION_MASK = 7
FLAG_CAPTURE = 1 << 15
FLAG_DOUBLE = 1 << 16
FLAG_EN_PASSANT = 1 << 17
FLAG_CASTLE = 1 << 18

U64_ZERO = np.uint64(0)
U64_ONE = np.uint64(1)
U64_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)


def _bit(square: int) -> np.uint64:
    return np.uint64(1 << square)


def _build_leaper_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    pawn = np.zeros((2, 64), dtype=np.uint64)
    knight_steps = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    king_steps = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    for square in range(64):
        file_index, rank_index = square & 7, square >> 3
        for df, dr in knight_steps:
            file_to, rank_to = file_index + df, rank_index + dr
            if 0 <= file_to < 8 and 0 <= rank_to < 8:
                knight[square] |= _bit(rank_to * 8 + file_to)
        for df, dr in king_steps:
            file_to, rank_to = file_index + df, rank_index + dr
            if 0 <= file_to < 8 and 0 <= rank_to < 8:
                king[square] |= _bit(rank_to * 8 + file_to)
        for color, rank_delta in ((WHITE, 1), (BLACK, -1)):
            rank_to = rank_index + rank_delta
            for file_delta in (-1, 1):
                file_to = file_index + file_delta
                if 0 <= file_to < 8 and 0 <= rank_to < 8:
                    pawn[color, square] |= _bit(rank_to * 8 + file_to)
    return knight, king, pawn


KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS = _build_leaper_tables()

# Multiplication maps the isolated least-significant bit to a unique six-bit table index.
_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_DEBRUIJN_INDEX = np.array(
    (
        0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4,
        62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5,
        63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6,
    ),
    dtype=np.uint8,
)


def _splitmix64(value: int) -> tuple[int, int]:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    mixed = value
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value, mixed ^ (mixed >> 31)


def _build_zobrist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.uint64]:
    seed = 0xC0DEC0FFEE123456

    def next_key() -> np.uint64:
        nonlocal seed
        seed, value = _splitmix64(seed)
        return np.uint64(value)

    pieces = np.empty((12, 64), dtype=np.uint64)
    castling = np.empty(16, dtype=np.uint64)
    ep = np.empty(64, dtype=np.uint64)
    for piece in range(12):
        for square in range(64):
            pieces[piece, square] = next_key()
    for rights in range(16):
        castling[rights] = next_key()
    for square in range(64):
        ep[square] = next_key()
    return pieces, castling, ep, next_key()


ZOBRIST_PIECES, ZOBRIST_CASTLING, ZOBRIST_EP, ZOBRIST_SIDE = _build_zobrist()

_CASTLING_MASK = np.full(64, 15, dtype=np.uint64)
_CASTLING_MASK[0] = 13  # a1 rook
_CASTLING_MASK[4] = 12  # e1 king
_CASTLING_MASK[7] = 14  # h1 rook
_CASTLING_MASK[56] = 7  # a8 rook
_CASTLING_MASK[60] = 3  # e8 king
_CASTLING_MASK[63] = 11  # h8 rook

_PIECE_FROM_SYMBOL = {
    "P": WP,
    "N": WN,
    "B": WB,
    "R": WR,
    "Q": WQ,
    "K": WK,
    "p": BP,
    "n": BN,
    "b": BB,
    "r": BR,
    "q": BQ,
    "k": BK,
}
_SYMBOL_FROM_PIECE = "PNBRQKpnbrqk"
_PROMOTION_TO_SYMBOL = " nbrq"


@numba.njit
def lsb_square(bitboard: np.uint64) -> int:
    """Return the square of a non-zero bitboard's least-significant set bit."""
    isolated = bitboard & (U64_ZERO - bitboard)
    index = int((isolated * _DEBRUIJN) >> np.uint64(58))
    return int(_DEBRUIJN_INDEX[index])


@numba.njit
def rook_attacks(square: int, occupied: np.uint64) -> np.uint64:
    """Ray-scan orthogonal attacks, including the first occupied square on each ray."""
    attacks = U64_ZERO
    file_index, rank_index = square & 7, square >> 3
    for file_delta, rank_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        file_to, rank_to = file_index + file_delta, rank_index + rank_delta
        while 0 <= file_to < 8 and 0 <= rank_to < 8:
            target = rank_to * 8 + file_to
            target_bit = U64_ONE << np.uint64(target)
            attacks |= target_bit
            if occupied & target_bit:
                break
            file_to += file_delta
            rank_to += rank_delta
    return attacks


@numba.njit
def bishop_attacks(square: int, occupied: np.uint64) -> np.uint64:
    """Ray-scan diagonal attacks, including the first occupied square on each ray."""
    attacks = U64_ZERO
    file_index, rank_index = square & 7, square >> 3
    for file_delta, rank_delta in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        file_to, rank_to = file_index + file_delta, rank_index + rank_delta
        while 0 <= file_to < 8 and 0 <= rank_to < 8:
            target = rank_to * 8 + file_to
            target_bit = U64_ONE << np.uint64(target)
            attacks |= target_bit
            if occupied & target_bit:
                break
            file_to += file_delta
            rank_to += rank_delta
    return attacks


@numba.njit
def encode_move(from_square: int, to_square: int, promotion: int = 0, flags: int = 0) -> np.uint32:
    return np.uint32(from_square | (to_square << 6) | (promotion << PROMOTION_SHIFT) | flags)


@numba.njit
def move_from(move: np.uint32) -> int:
    return int(move & np.uint32(63))


@numba.njit
def move_to(move: np.uint32) -> int:
    return int((move >> np.uint32(6)) & np.uint32(63))


@numba.njit
def move_promotion(move: np.uint32) -> int:
    return int((move >> np.uint32(PROMOTION_SHIFT)) & np.uint32(PROMOTION_MASK))


@numba.njit
def compute_zobrist(state: np.ndarray) -> np.uint64:
    """Recompute the key; tests use this to audit incremental updates."""
    key = ZOBRIST_CASTLING[int(state[CASTLING])]
    if int(state[SIDE]) == BLACK:
        key ^= ZOBRIST_SIDE
    ep_square = int(state[EP_SQUARE])
    if ep_square != NO_SQUARE:
        key ^= ZOBRIST_EP[ep_square]
    for piece in range(12):
        pieces = state[piece]
        while pieces:
            square = lsb_square(pieces)
            pieces &= pieces - U64_ONE
            key ^= ZOBRIST_PIECES[piece, square]
    return key


@numba.njit
def is_square_attacked(state: np.ndarray, square: int, by_color: int) -> bool:
    """Test attacks without constructing an attack map."""
    offset = by_color * 6
    if state[offset + PAWN] & PAWN_ATTACKS[1 - by_color, square]:
        return True
    if state[offset + KNIGHT] & KNIGHT_ATTACKS[square]:
        return True
    occupied = state[ALL_OCC]
    if (state[offset + BISHOP] | state[offset + QUEEN]) & bishop_attacks(square, occupied):
        return True
    if (state[offset + ROOK] | state[offset + QUEEN]) & rook_attacks(square, occupied):
        return True
    return bool(state[offset + KING] & KING_ATTACKS[square])


@numba.njit
def is_in_check(state: np.ndarray, color: int) -> bool:
    king = state[color * 6 + KING]
    if king == 0:
        return True
    return is_square_attacked(state, lsb_square(king), 1 - color)


@numba.njit
def _append_targets(
    moves: np.ndarray,
    count: int,
    from_square: int,
    targets: np.uint64,
    enemy: np.uint64,
) -> int:
    while targets:
        to_square = lsb_square(targets)
        targets &= targets - U64_ONE
        flags = FLAG_CAPTURE if enemy & (U64_ONE << np.uint64(to_square)) else 0
        moves[count] = encode_move(from_square, to_square, 0, flags)
        count += 1
    return count


@numba.njit
def _append_promotions(
    moves: np.ndarray,
    count: int,
    from_square: int,
    to_square: int,
    flags: int,
) -> int:
    for promotion in range(1, 5):
        moves[count] = encode_move(from_square, to_square, promotion, flags)
        count += 1
    return count


@numba.njit
def generate_pseudo_into(state: np.ndarray, moves: np.ndarray) -> int:
    """Write pseudo-legal moves into a caller-owned 256-entry buffer."""
    count = 0
    color = int(state[SIDE])
    offset = color * 6
    own = state[WHITE_OCC + color]
    enemy = state[BLACK_OCC - color]
    occupied = state[ALL_OCC]

    pawns = state[offset + PAWN]
    while pawns:
        from_square = lsb_square(pawns)
        pawns &= pawns - U64_ONE
        rank_index = from_square >> 3
        if color == WHITE:
            to_square = from_square + 8
            promotion_rank = rank_index == 6
            if to_square < 64 and not occupied & (U64_ONE << np.uint64(to_square)):
                if promotion_rank:
                    count = _append_promotions(moves, count, from_square, to_square, 0)
                else:
                    moves[count] = encode_move(from_square, to_square)
                    count += 1
                    double_to = from_square + 16
                    if rank_index == 1 and not occupied & (U64_ONE << np.uint64(double_to)):
                        moves[count] = encode_move(from_square, double_to, 0, FLAG_DOUBLE)
                        count += 1
        else:
            to_square = from_square - 8
            promotion_rank = rank_index == 1
            if to_square >= 0 and not occupied & (U64_ONE << np.uint64(to_square)):
                if promotion_rank:
                    count = _append_promotions(moves, count, from_square, to_square, 0)
                else:
                    moves[count] = encode_move(from_square, to_square)
                    count += 1
                    double_to = from_square - 16
                    if rank_index == 6 and not occupied & (U64_ONE << np.uint64(double_to)):
                        moves[count] = encode_move(from_square, double_to, 0, FLAG_DOUBLE)
                        count += 1

        captures = PAWN_ATTACKS[color, from_square] & enemy
        while captures:
            to_square = lsb_square(captures)
            captures &= captures - U64_ONE
            if promotion_rank:
                count = _append_promotions(
                    moves, count, from_square, to_square, FLAG_CAPTURE
                )
            else:
                moves[count] = encode_move(from_square, to_square, 0, FLAG_CAPTURE)
                count += 1

        ep_square = int(state[EP_SQUARE])
        if ep_square != NO_SQUARE and PAWN_ATTACKS[color, from_square] & (
            U64_ONE << np.uint64(ep_square)
        ):
            captured_square = ep_square - 8 if color == WHITE else ep_square + 8
            if state[(1 - color) * 6 + PAWN] & (U64_ONE << np.uint64(captured_square)):
                moves[count] = encode_move(
                    from_square,
                    ep_square,
                    0,
                    FLAG_CAPTURE | FLAG_EN_PASSANT,
                )
                count += 1

    pieces = state[offset + KNIGHT]
    while pieces:
        from_square = lsb_square(pieces)
        pieces &= pieces - U64_ONE
        count = _append_targets(
            moves, count, from_square, KNIGHT_ATTACKS[from_square] & ~own, enemy
        )

    pieces = state[offset + BISHOP]
    while pieces:
        from_square = lsb_square(pieces)
        pieces &= pieces - U64_ONE
        count = _append_targets(
            moves, count, from_square, bishop_attacks(from_square, occupied) & ~own, enemy
        )

    pieces = state[offset + ROOK]
    while pieces:
        from_square = lsb_square(pieces)
        pieces &= pieces - U64_ONE
        count = _append_targets(
            moves, count, from_square, rook_attacks(from_square, occupied) & ~own, enemy
        )

    pieces = state[offset + QUEEN]
    while pieces:
        from_square = lsb_square(pieces)
        pieces &= pieces - U64_ONE
        targets = (
            rook_attacks(from_square, occupied) | bishop_attacks(from_square, occupied)
        ) & ~own
        count = _append_targets(moves, count, from_square, targets, enemy)

    kings = state[offset + KING]
    if kings:
        from_square = lsb_square(kings)
        count = _append_targets(
            moves, count, from_square, KING_ATTACKS[from_square] & ~own, enemy
        )

    rights = int(state[CASTLING])
    opponent = 1 - color
    if color == WHITE and state[WK] & (U64_ONE << np.uint64(4)):
        if (
            rights & CASTLE_WHITE_KING
            and state[WR] & (U64_ONE << np.uint64(7))
            and not occupied
            & ((U64_ONE << np.uint64(5)) | (U64_ONE << np.uint64(6)))
            and not is_square_attacked(state, 4, opponent)
            and not is_square_attacked(state, 5, opponent)
            and not is_square_attacked(state, 6, opponent)
        ):
            moves[count] = encode_move(4, 6, 0, FLAG_CASTLE)
            count += 1
        if (
            rights & CASTLE_WHITE_QUEEN
            and state[WR] & (U64_ONE << np.uint64(0))
            and not occupied
            & (
                (U64_ONE << np.uint64(1))
                | (U64_ONE << np.uint64(2))
                | (U64_ONE << np.uint64(3))
            )
            and not is_square_attacked(state, 4, opponent)
            and not is_square_attacked(state, 3, opponent)
            and not is_square_attacked(state, 2, opponent)
        ):
            moves[count] = encode_move(4, 2, 0, FLAG_CASTLE)
            count += 1
    elif color == BLACK and state[BK] & (U64_ONE << np.uint64(60)):
        if (
            rights & CASTLE_BLACK_KING
            and state[BR] & (U64_ONE << np.uint64(63))
            and not occupied
            & ((U64_ONE << np.uint64(61)) | (U64_ONE << np.uint64(62)))
            and not is_square_attacked(state, 60, opponent)
            and not is_square_attacked(state, 61, opponent)
            and not is_square_attacked(state, 62, opponent)
        ):
            moves[count] = encode_move(60, 62, 0, FLAG_CASTLE)
            count += 1
        if (
            rights & CASTLE_BLACK_QUEEN
            and state[BR] & (U64_ONE << np.uint64(56))
            and not occupied
            & (
                (U64_ONE << np.uint64(57))
                | (U64_ONE << np.uint64(58))
                | (U64_ONE << np.uint64(59))
            )
            and not is_square_attacked(state, 60, opponent)
            and not is_square_attacked(state, 59, opponent)
            and not is_square_attacked(state, 58, opponent)
        ):
            moves[count] = encode_move(60, 58, 0, FLAG_CASTLE)
            count += 1
    return count


@numba.njit
def _generate_pseudo_moves(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Compatibility wrapper for the accepted allocating move-generator API."""
    moves = np.empty(MAX_MOVES, dtype=np.uint32)
    return moves, generate_pseudo_into(state, moves)


@numba.njit
def make_move(
    state: np.ndarray, move: np.uint32
) -> tuple[np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64]:
    """Apply ``move`` in place and return the fixed-size undo record."""
    from_square = move_from(move)
    to_square = move_to(move)
    promotion = move_promotion(move)
    flags = int(move) & (FLAG_CAPTURE | FLAG_DOUBLE | FLAG_EN_PASSANT | FLAG_CASTLE)
    color = int(state[SIDE])
    opponent = 1 - color
    offset = color * 6
    from_bit = U64_ONE << np.uint64(from_square)
    to_bit = U64_ONE << np.uint64(to_square)

    moving_piece = NO_PIECE
    for piece_type in range(6):
        if state[offset + piece_type] & from_bit:
            moving_piece = offset + piece_type
            break

    old_castling = state[CASTLING]
    old_ep = state[EP_SQUARE]
    old_halfmove = state[HALFMOVE]
    old_fullmove = state[FULLMOVE]
    old_hash = state[ZOBRIST]
    captured_piece = NO_PIECE

    key = old_hash ^ ZOBRIST_CASTLING[int(old_castling)]
    if int(old_ep) != NO_SQUARE:
        key ^= ZOBRIST_EP[int(old_ep)]
    key ^= ZOBRIST_PIECES[moving_piece, from_square]

    state[moving_piece] &= ~from_bit
    state[WHITE_OCC + color] &= ~from_bit
    state[ALL_OCC] &= ~from_bit

    captured_square = to_square
    if flags & FLAG_EN_PASSANT:
        captured_square = to_square - 8 if color == WHITE else to_square + 8
    captured_square_index = int(captured_square)
    captured_bit = U64_ONE << np.uint64(captured_square_index)
    if flags & FLAG_CAPTURE:
        for piece_type in range(6):
            piece = opponent * 6 + piece_type
            if state[piece] & captured_bit:
                captured_piece = piece
                state[piece] &= ~captured_bit
                state[WHITE_OCC + opponent] &= ~captured_bit
                state[ALL_OCC] &= ~captured_bit
                key ^= ZOBRIST_PIECES[piece, captured_square_index]
                break

    destination_piece = moving_piece
    if promotion:
        destination_piece = offset + promotion
    state[destination_piece] |= to_bit
    state[WHITE_OCC + color] |= to_bit
    state[ALL_OCC] |= to_bit
    key ^= ZOBRIST_PIECES[destination_piece, to_square]

    if flags & FLAG_CASTLE:
        if to_square == 6:
            rook_from, rook_to, rook_piece = 7, 5, WR
        elif to_square == 2:
            rook_from, rook_to, rook_piece = 0, 3, WR
        elif to_square == 62:
            rook_from, rook_to, rook_piece = 63, 61, BR
        else:
            rook_from, rook_to, rook_piece = 56, 59, BR
        rook_from_bit = U64_ONE << np.uint64(rook_from)
        rook_to_bit = U64_ONE << np.uint64(rook_to)
        state[rook_piece] ^= rook_from_bit | rook_to_bit
        state[WHITE_OCC + color] ^= rook_from_bit | rook_to_bit
        state[ALL_OCC] ^= rook_from_bit | rook_to_bit
        key ^= ZOBRIST_PIECES[rook_piece, rook_from]
        key ^= ZOBRIST_PIECES[rook_piece, rook_to]

    new_castling = int(old_castling & _CASTLING_MASK[from_square] & _CASTLING_MASK[to_square])
    state[CASTLING] = new_castling
    state[EP_SQUARE] = (from_square + to_square) // 2 if flags & FLAG_DOUBLE else NO_SQUARE
    if moving_piece % 6 == PAWN or captured_piece != NO_PIECE:
        state[HALFMOVE] = 0
    else:
        state[HALFMOVE] = old_halfmove + U64_ONE
    if color == BLACK:
        state[FULLMOVE] = old_fullmove + U64_ONE
    state[SIDE] = opponent

    key ^= ZOBRIST_CASTLING[new_castling]
    if int(state[EP_SQUARE]) != NO_SQUARE:
        key ^= ZOBRIST_EP[int(state[EP_SQUARE])]
    key ^= ZOBRIST_SIDE
    state[ZOBRIST] = key
    return (
        old_castling,
        old_ep,
        old_halfmove,
        old_fullmove,
        old_hash,
        np.uint64(captured_piece),
    )


@numba.njit
def unmake_move(
    state: np.ndarray,
    move: np.uint32,
    undo: tuple[np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64],
) -> None:
    """Restore the exact state preceding ``make_move``."""
    old_castling, old_ep, old_halfmove, old_fullmove, old_hash, captured_word = undo
    from_square = move_from(move)
    to_square = move_to(move)
    promotion = move_promotion(move)
    flags = int(move) & (FLAG_CAPTURE | FLAG_DOUBLE | FLAG_EN_PASSANT | FLAG_CASTLE)
    color = 1 - int(state[SIDE])
    offset = color * 6
    from_bit = U64_ONE << np.uint64(from_square)
    to_bit = U64_ONE << np.uint64(to_square)
    moving_piece = offset + PAWN if promotion else NO_PIECE
    destination_piece = offset + promotion if promotion else NO_PIECE
    if not promotion:
        for piece_type in range(6):
            piece = offset + piece_type
            if state[piece] & to_bit:
                moving_piece = piece
                destination_piece = piece
                break

    state[destination_piece] &= ~to_bit
    state[moving_piece] |= from_bit
    state[WHITE_OCC + color] ^= from_bit | to_bit
    state[ALL_OCC] ^= from_bit | to_bit

    if flags & FLAG_CASTLE:
        if to_square == 6:
            rook_from, rook_to, rook_piece = 7, 5, WR
        elif to_square == 2:
            rook_from, rook_to, rook_piece = 0, 3, WR
        elif to_square == 62:
            rook_from, rook_to, rook_piece = 63, 61, BR
        else:
            rook_from, rook_to, rook_piece = 56, 59, BR
        rook_from_bit = U64_ONE << np.uint64(rook_from)
        rook_to_bit = U64_ONE << np.uint64(rook_to)
        state[rook_piece] ^= rook_from_bit | rook_to_bit
        state[WHITE_OCC + color] ^= rook_from_bit | rook_to_bit
        state[ALL_OCC] ^= rook_from_bit | rook_to_bit

    captured_piece = int(captured_word)
    if captured_piece != NO_PIECE:
        captured_square = to_square
        if flags & FLAG_EN_PASSANT:
            captured_square = to_square - 8 if color == WHITE else to_square + 8
        captured_bit = U64_ONE << np.uint64(int(captured_square))
        state[captured_piece] |= captured_bit
        state[WHITE_OCC + (1 - color)] |= captured_bit
        state[ALL_OCC] |= captured_bit

    state[SIDE] = color
    state[CASTLING] = old_castling
    state[EP_SQUARE] = old_ep
    state[HALFMOVE] = old_halfmove
    state[FULLMOVE] = old_fullmove
    state[ZOBRIST] = old_hash


@numba.njit
def generate_moves(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a fixed-capacity array and the number of legal moves in its prefix."""
    pseudo, pseudo_count = _generate_pseudo_moves(state)
    legal = np.empty(MAX_MOVES, dtype=np.uint32)
    legal_count = 0
    moving_color = int(state[SIDE])
    for index in range(pseudo_count):
        move = pseudo[index]
        undo = make_move(state, move)
        if not is_in_check(state, moving_color):
            legal[legal_count] = move
            legal_count += 1
        unmake_move(state, move, undo)
    return legal, legal_count


@numba.njit
def perft(state: np.ndarray, depth: int) -> np.uint64:
    """Count legal leaf nodes while avoiding a second make/unmake filtering pass."""
    if depth == 0:
        return U64_ONE
    pseudo, pseudo_count = _generate_pseudo_moves(state)
    nodes = U64_ZERO
    moving_color = int(state[SIDE])
    for index in range(pseudo_count):
        move = pseudo[index]
        undo = make_move(state, move)
        if not is_in_check(state, moving_color):
            if depth == 1:
                nodes += U64_ONE
            else:
                nodes += perft(state, depth - 1)
        unmake_move(state, move, undo)
    return nodes


def from_fen(fen: str) -> np.ndarray:
    """Parse a standard (non-Chess960) FEN into the native state."""
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError(f"expected six FEN fields: {fen!r}")
    placement, turn, castling, ep_name, halfmove, fullmove = fields
    state = np.zeros(STATE_SIZE, dtype=np.uint64)
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"expected eight FEN ranks: {fen!r}")
    for fen_rank, text in enumerate(ranks):
        file_index = 0
        rank_index = 7 - fen_rank
        for symbol in text:
            if symbol.isdigit():
                file_index += int(symbol)
                continue
            if symbol not in _PIECE_FROM_SYMBOL or file_index >= 8:
                raise ValueError(f"invalid FEN placement: {placement!r}")
            piece = _PIECE_FROM_SYMBOL[symbol]
            state[piece] |= _bit(rank_index * 8 + file_index)
            file_index += 1
        if file_index != 8:
            raise ValueError(f"invalid FEN rank: {text!r}")

    state[WHITE_OCC] = np.bitwise_or.reduce(state[WP : WK + 1])
    state[BLACK_OCC] = np.bitwise_or.reduce(state[BP : BK + 1])
    state[ALL_OCC] = state[WHITE_OCC] | state[BLACK_OCC]
    if turn not in ("w", "b"):
        raise ValueError(f"invalid FEN side: {turn!r}")
    state[SIDE] = WHITE if turn == "w" else BLACK
    rights = 0
    if castling != "-":
        for symbol in castling:
            if symbol == "K":
                rights |= CASTLE_WHITE_KING
            elif symbol == "Q":
                rights |= CASTLE_WHITE_QUEEN
            elif symbol == "k":
                rights |= CASTLE_BLACK_KING
            elif symbol == "q":
                rights |= CASTLE_BLACK_QUEEN
            else:
                raise ValueError(f"invalid FEN castling field: {castling!r}")
    state[CASTLING] = rights
    state[EP_SQUARE] = NO_SQUARE if ep_name == "-" else parse_square(ep_name)
    state[HALFMOVE] = int(halfmove)
    state[FULLMOVE] = int(fullmove)
    state[ZOBRIST] = compute_zobrist(state)
    return state


def to_fen(state: np.ndarray) -> str:
    """Serialize all six state fields, preserving the exact en-passant square."""
    ranks: list[str] = []
    for rank_index in range(7, -1, -1):
        text = ""
        empty = 0
        for file_index in range(8):
            square = rank_index * 8 + file_index
            square_bit = _bit(square)
            piece_symbol = ""
            for piece in range(12):
                if state[piece] & square_bit:
                    piece_symbol = _SYMBOL_FROM_PIECE[piece]
                    break
            if piece_symbol:
                if empty:
                    text += str(empty)
                    empty = 0
                text += piece_symbol
            else:
                empty += 1
        if empty:
            text += str(empty)
        ranks.append(text)

    rights = int(state[CASTLING])
    castling = ""
    for bit, symbol in (
        (CASTLE_WHITE_KING, "K"),
        (CASTLE_WHITE_QUEEN, "Q"),
        (CASTLE_BLACK_KING, "k"),
        (CASTLE_BLACK_QUEEN, "q"),
    ):
        if rights & bit:
            castling += symbol
    ep_square = int(state[EP_SQUARE])
    return " ".join(
        (
            "/".join(ranks),
            "w" if int(state[SIDE]) == WHITE else "b",
            castling or "-",
            "-" if ep_square == NO_SQUARE else square_name(ep_square),
            str(int(state[HALFMOVE])),
            str(int(state[FULLMOVE])),
        )
    )


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"invalid square: {name!r}")
    return (ord(name[1]) - ord("1")) * 8 + ord(name[0]) - ord("a")


def square_name(square: int) -> str:
    if not 0 <= square < 64:
        raise ValueError(f"invalid square: {square}")
    return chr(ord("a") + (square & 7)) + chr(ord("1") + (square >> 3))


def move_to_uci(move: int | np.uint32) -> str:
    promotion = int(move_promotion(np.uint32(move)))
    suffix = _PROMOTION_TO_SYMBOL[promotion] if promotion else ""
    return square_name(int(move_from(np.uint32(move)))) + square_name(
        int(move_to(np.uint32(move)))
    ) + suffix


def legal_uci_moves(state: np.ndarray) -> list[str]:
    moves, count = generate_moves(state)
    return [move_to_uci(moves[index]) for index in range(count)]


def find_legal_move(state: np.ndarray, uci: str) -> np.uint32:
    """Resolve UCI to its fully flagged native move encoding."""
    moves, count = generate_moves(state)
    for index in range(count):
        if move_to_uci(moves[index]) == uci:
            return moves[index]
    raise ValueError(f"illegal move {uci!r} in {to_fen(state)!r}")


def warm_up() -> float:
    """Compile every public JIT entry point with its production argument types."""
    started = time.perf_counter()
    state = from_fen("8/8/8/8/8/8/4K3/6k1 w - - 0 1")
    occupied = state[ALL_OCC]
    lsb_square(state[WK])
    rook_attacks(0, occupied)
    bishop_attacks(0, occupied)
    encode_move(12, 20)
    sample = encode_move(12, 20)
    move_from(sample)
    move_to(sample)
    move_promotion(sample)
    compute_zobrist(state)
    is_square_attacked(state, 12, BLACK)
    is_in_check(state, WHITE)
    moves, count = generate_moves(state)
    if count:
        undo = make_move(state, moves[0])
        unmake_move(state, moves[0], undo)
    scratch = np.empty(MAX_MOVES, dtype=np.uint32)
    generate_pseudo_into(state, scratch)
    perft(state, 2)
    return time.perf_counter() - started


__all__ = [
    "ALL_OCC",
    "BLACK",
    "BLACK_OCC",
    "CASTLING",
    "EP_SQUARE",
    "FULLMOVE",
    "HALFMOVE",
    "MAX_MOVES",
    "SIDE",
    "STATE_SIZE",
    "WHITE",
    "WHITE_OCC",
    "ZOBRIST",
    "bishop_attacks",
    "compute_zobrist",
    "encode_move",
    "find_legal_move",
    "from_fen",
    "generate_moves",
    "generate_pseudo_into",
    "is_in_check",
    "is_square_attacked",
    "legal_uci_moves",
    "lsb_square",
    "make_move",
    "move_from",
    "move_promotion",
    "move_to",
    "move_to_uci",
    "parse_square",
    "perft",
    "rook_attacks",
    "square_name",
    "to_fen",
    "unmake_move",
    "warm_up",
]
