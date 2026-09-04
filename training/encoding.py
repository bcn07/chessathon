"""Side-to-move canonical board encoding shared by the NNUE pipeline, trainer and checks.

A position is stored as 27 bytes: an occupancy bitboard over *canonical* squares, sixteen bytes
of 4-bit piece codes (one per occupied square, least-significant square first), and an int16
centipawn score.  Canonical means "as if the side to move were White": squares are flipped
vertically for a black-to-move position and the piece codes are re-coloured, so a feature index
is simply ``code * 64 + square`` and the same 768 inputs describe both perspectives.

The engine-side twin of ``features_from_record`` is ``nnue_eval.accumulate``; the round-trip is
checked against python-chess in ``test_encoding.py``.
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


def features_from_record(occ: int, nib: np.ndarray) -> list[int]:
    """Active feature indices (``code * 64 + square``) for one packed record."""
    features: list[int] = []
    count = 0
    bits = int(occ)
    while bits:
        square = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        byte = int(nib[count >> 1])
        code = (byte >> 4) if (count & 1) else (byte & 15)
        features.append(code * 64 + square)
        count += 1
    return features


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
