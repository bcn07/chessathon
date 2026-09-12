"""Side-to-move canonical board encoding shared by the NNUE pipeline, trainer and checks.

A position is stored as 27 bytes: an occupancy bitboard over *canonical* squares, sixteen bytes
of 4-bit piece codes (one per occupied square, least-significant square first), and an int16
centipawn score.  Canonical means "as if the side to move were White": squares are flipped
vertically for a black-to-move position and the piece codes are re-coloured, so a feature index
is simply ``code * 64 + square`` and one 768-input block describes the **mover's** perspective.

That block is the mover's view alone: it says nothing about where the opponent's king sits
relative to the opponent's own pieces (with king buckets it cannot even see the opponent's
bucket).  ``opponent_features_from_record`` is the second perspective a dual-perspective net
needs — the same function of the colour-swapped, vertically flipped position — and both index one
shared weight table.

The engine-side twin of ``features_from_record`` is ``nnue_eval.active_features``; the round-trip
is checked against python-chess in ``verify_encoding.py``.
"""

from __future__ import annotations

import numpy as np

# White-relative piece codes, matching fastboard's WP..BK ordering.
PIECE_FROM_SYMBOL = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}
SYMBOL_FROM_PIECE = "PNBRQKpnbrqk"

NUM_FEATURES = 768
MAX_PIECES = 32
SCORE_CLAMP = 2000

# King-bucketed inputs: the board is mirrored left-right when the side to move's king stands on
# files e-h (so every king sits on a-d), and the 768 features are offset by KING_BUCKET[king] * 768.
# Eight buckets: rank 1 split castled (a/b = h/g) vs centre (c/d = f/e), rank 2 the same, then
# ranks 3, 4, 5-6, 7-8. The engine's ``nnue_eval.py`` carries the same table; ``verify_encoding``
# checks the two agree. A net with a 768-row table uses no buckets.
KING_BUCKETS = 8
_BUCKET_BY_RANK_FILE = (
    (0, 0, 1, 1), (2, 2, 3, 3), (4, 4, 4, 4), (5, 5, 5, 5),
    (6, 6, 6, 6), (6, 6, 6, 6), (7, 7, 7, 7), (7, 7, 7, 7),
)
KING_BUCKET = np.array(
    [_BUCKET_BY_RANK_FILE[sq >> 3][(sq & 7) if (sq & 7) < 4 else 7 - (sq & 7)] for sq in range(64)],
    dtype=np.int64,
)


# A finer variant used by the 32-bucket nets (`train_kb32.py`): the same left-right mirror, then
# one bucket per (rank, file a-d) square, i.e. 8 x 4 = 32 buckets and a 24576-row input table.
# The 8-bucket table above stays the default; a net selects its table by ``w1.shape[0] // 768``.
KING_BUCKETS_32 = 32
KING_BUCKET_32 = np.array(
    [(sq >> 3) * 4 + ((sq & 7) if (sq & 7) < 4 else 7 - (sq & 7)) for sq in range(64)],
    dtype=np.int64,
)


def bucket_table(king_buckets: int) -> np.ndarray:
    """The king-bucket table for ``king_buckets`` inputs (32 = per-square, anything else = the
    default 8-bucket table)."""
    return KING_BUCKET_32 if king_buckets == KING_BUCKETS_32 else KING_BUCKET


def king_transform(king_square: int, king_buckets: int = KING_BUCKETS) -> tuple[int, int]:
    """(mirror xor, feature offset) for a side-to-move king on ``king_square``."""
    mirror = 7 if (king_square & 7) >= 4 else 0
    return mirror, int(bucket_table(king_buckets)[king_square ^ mirror]) * NUM_FEATURES

RECORD_DTYPE = np.dtype([("occ", "<u8"), ("nib", "u1", 16), ("score", "<i2")])

_DIGITS = frozenset("12345678")


def pack_placement(placement: str, black_to_move: bool) -> tuple[int, bytes]:
    """Return (occupancy, 16 nibble bytes) for a FEN placement field in canonical form.

    FEN ranks run 8..1; canonical squares run a1..h8 for white to move and (flipped) a8..h1 for
    black, so walking the ranks in the right order already yields ascending canonical squares
    and no sort is needed.
    """
    ranks = placement.split("/")
    if not black_to_move:
        ranks = ranks[::-1]
    occ = 0
    nib = bytearray(16)
    count = 0
    square = 0
    for text in ranks:
        for symbol in text:
            if symbol in _DIGITS:
                square += ord(symbol) - 48
                continue
            if count == MAX_PIECES:
                # The eval dump contains a handful of corrupt placements with 33+ pieces.
                raise ValueError(f"more than {MAX_PIECES} pieces: {placement!r}")
            code = PIECE_FROM_SYMBOL[symbol]
            if black_to_move:
                code = code - 6 if code >= 6 else code + 6
            occ |= 1 << square
            if count & 1:
                nib[count >> 1] |= code << 4
            else:
                nib[count >> 1] = code
            count += 1
            square += 1
    if square != 64:
        raise ValueError(f"invalid FEN placement: {placement!r}")
    return occ, bytes(nib)


def _pieces_from_record(occ: int, nib: np.ndarray) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    count = 0
    bits = int(occ)
    while bits:
        square = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        byte = int(nib[count >> 1])
        code = (byte >> 4) if (count & 1) else (byte & 15)
        pieces.append((code, square))
        count += 1
    return pieces


def features_from_record(occ: int, nib: np.ndarray, king_buckets: int = 1) -> list[int]:
    """Active feature indices (``code * 64 + square``, plus the king-bucket offset and left-right
    mirror when ``king_buckets`` > 1) for one packed record."""
    pieces = _pieces_from_record(occ, nib)
    mirror, offset = 0, 0
    if king_buckets > 1:
        king = [square for code, square in pieces if code == 5]
        if king:
            mirror, offset = king_transform(king[0], king_buckets)
    return [offset + code * 64 + (square ^ mirror) for code, square in pieces]


def opponent_pieces(pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The same position seen from the *non-mover's* side: swap the colour of every code and flip
    every square vertically. Applied to a canonical (side-to-move) piece list this yields the
    canonical piece list the record would hold if the other side were to move."""
    return [(code - 6 if code >= 6 else code + 6, square ^ 56) for code, square in pieces]


def opponent_features_from_record(
    occ: int, nib: np.ndarray, king_buckets: int = 1
) -> list[int]:
    """Active feature indices from the **non-mover's** perspective (P1 of a dual net).

    The records store the position canonically for the side to move, so P1 is the same function as
    ``features_from_record`` applied to the colour-swapped, vertically flipped position: codes are
    relative to the non-mover, squares are flipped, and the left-right mirror plus king-bucket
    offset come from the *non-mover's* king (code 5 after the swap). The mover's own view P0 stays
    exactly what ``features_from_record`` returns, so a dual net's two index sets are the same
    function of two different perspectives and share one weight table.
    """
    pieces = opponent_pieces(_pieces_from_record(occ, nib))
    mirror, offset = 0, 0
    if king_buckets > 1:
        king = [square for code, square in pieces if code == 5]
        if king:
            mirror, offset = king_transform(king[0], king_buckets)
    return [offset + code * 64 + (square ^ mirror) for code, square in pieces]


def bitboards_from_record(occ: int, nib: np.ndarray) -> np.ndarray:
    """Twelve canonical bitboards (own pawn..own king, enemy pawn..enemy king)."""
    boards = np.zeros(12, dtype=np.uint64)
    count = 0
    bits = int(occ)
    while bits:
        low = bits & -bits
        square = low.bit_length() - 1
        bits &= bits - 1
        byte = int(nib[count >> 1])
        code = (byte >> 4) if (count & 1) else (byte & 15)
        boards[code] |= np.uint64(1) << np.uint64(square)
        count += 1
    return boards
