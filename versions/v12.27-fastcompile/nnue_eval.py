"""Quantised NNUE inference for the numba engine: numpy + numba only, no torch at runtime.

The weights live in ``weights/nnue.npz`` (the packager ships that directory with the zip).

The net is ``768 -> HIDDEN -> 1`` with a clipped-ReLU hidden layer.  Inputs are the 768
side-to-move-canonical features described in ``encoding.py``; because every input is 0 or 1 the
first layer is a sum of at most 32 weight rows, so a full recomputation costs
``pieces * HIDDEN`` int16 adds and needs no matrix multiply.

Fixed point: the first layer is scaled by ``QA`` (an accumulator value of ``QA`` means 1.0 after
the clipped ReLU), the output layer by ``QB``, so the int32 output is the network's value scaled
by ``QA * QB``.  ``CP_SCALE`` turns that value into centipawns (the net is trained so that
``sigmoid(value)`` is the win probability and ``value * CP_SCALE`` is the centipawn score).

The accumulator is int16.  ``quantise.py`` proves the worst-case accumulator over any legal piece
placement stays inside int16 before it writes the weights, so no saturation check is needed here.
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

import fastboard as fb

QA = 1024
QB = 1024
CP_SCALE = 400

# King-bucketed inputs (a net whose ``w1`` has more than 768 rows): the board is mirrored
# left-right when the side to move's king stands on files e-h, and the 768 features are offset by
# KING_BUCKET[king square] * 768. Same table as ``work/nnue/encoding.py``; eight buckets: rank 1
# split castled (a/b) vs centre (c/d), rank 2 the same, then ranks 3, 4, 5-6, 7-8.
_BUCKET_BY_RANK_FILE = (
    (0, 0, 1, 1), (2, 2, 3, 3), (4, 4, 4, 4), (5, 5, 5, 5),
    (6, 6, 6, 6), (6, 6, 6, 6), (7, 7, 7, 7), (7, 7, 7, 7),
)
KING_BUCKET = np.array(
    [_BUCKET_BY_RANK_FILE[sq >> 3][(sq & 7) if (sq & 7) < 4 else 7 - (sq & 7)] for sq in range(64)],
    dtype=np.int64,
)
NUM_FEATURES = 768
# Output buckets: the output layer is (buckets, hidden) and the bucket is chosen by how many
# pieces are on the board, so endgames and middlegames get their own output weights; a net with
# one bucket behaves exactly as before. SCRELU squares the clipped activation for a smoother,
# wider response at the same accumulator cost. Both are compile-time constants for numba.
BUCKET_DIVISOR = 4
SCRELU = True

# Tempo learned from the training labels. The labels are the engine's own search scores on
# self-play positions (mean +51 cp for the side to move, median +20), so the net learns a mover
# bonus of its own; `tempo_check.py` measures it and this constant brings it back to the +10 that
# the search's draw scores and pruning margins were tuned around.
MOVER_BIAS = 33  # measured 43.2 cp learned tempo, PeSTO has 10
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "nnue.npz"


def load_weights(
    path: Path | str = WEIGHTS_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (w1, b1, w2, b2) ready for ``nnue_evaluate``. The output layer is always shaped
    (buckets, hidden): a net trained without output buckets loads as one bucket."""
    with np.load(path) as data:
        w1 = np.ascontiguousarray(data["w1"], dtype=np.int16)
        b1 = np.ascontiguousarray(data["b1"], dtype=np.int16)
        w2 = np.asarray(data["w2"], dtype=np.int16)
        b2 = np.asarray(data["b2"], dtype=np.int32)
    if w2.ndim == 1:
        w2 = w2.reshape(1, -1)
    if b2.ndim == 0:
        b2 = b2.reshape(1)
    return w1, b1, np.ascontiguousarray(w2), np.ascontiguousarray(b2)


@numba.njit(inline="always")
def _king_transform(state: np.ndarray, stm: int, king_buckets: int) -> tuple[int, int]:
    """(mirror xor, feature offset) for the side to move's king; (0, 0) without buckets."""
    if king_buckets <= 1:
        return 0, 0
    king = int(fb.lsb_square(state[stm * 6 + fb.KING]))  # type: ignore[call-overload]
    if stm == fb.BLACK:
        king ^= 56
    mirror = 7 if (king & 7) >= 4 else 0
    return mirror, int(KING_BUCKET[king ^ mirror]) * NUM_FEATURES


@numba.njit
def active_features(state: np.ndarray, king_buckets: int) -> tuple[int, np.ndarray]:
    """The active feature indices for a fastboard state (used by tests, not by the search)."""
    out = np.empty(32, dtype=np.int64)
    count = 0
    stm = int(state[fb.SIDE])
    mirror, offset = _king_transform(state, stm, king_buckets)
    one = np.uint64(1)
    for piece in range(12):
        piece_type = piece % 6
        code = piece_type if piece // 6 == stm else piece_type + 6
        base = offset + code * 64
        bits = state[piece]
        while bits:
            square = int(fb.lsb_square(bits))  # type: ignore[call-overload]
            bits &= bits - one
            if stm == fb.BLACK:
                square ^= 56
            out[count] = base + (square ^ mirror)
            count += 1
    return count, out


@numba.njit
def nnue_evaluate(
    state: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
) -> int:
    """Static centipawn score relative to the state's side to move (allocating wrapper for
    tests and tools; the search calls ``nnue_evaluate_into`` with its own scratch buffers)."""
    feats = np.empty(32, dtype=np.int64)
    acc = np.empty(b1.shape[0], dtype=np.int16)
    return int(nnue_evaluate_into(state, w1, b1, w2, b2, feats, acc))


@numba.njit
def nnue_evaluate_into(
    state: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    feats: np.ndarray,
    acc: np.ndarray,
) -> int:
    """Static centipawn score relative to the state's side to move, using the caller's scratch
    ``feats`` (32 int64) and ``acc`` (hidden int16): two heap allocations fewer per node."""
    hidden = b1.shape[0]
    stm = int(state[fb.SIDE])
    mirror, offset = _king_transform(state, stm, w1.shape[0] // NUM_FEATURES)
    one = np.uint64(1)
    nf = 0
    for piece in range(12):
        piece_type = piece % 6
        code = piece_type if piece // 6 == stm else piece_type + 6
        base = offset + code * 64
        bits = state[piece]
        while bits:
            square = int(fb.lsb_square(bits))  # type: ignore[call-overload]
            bits &= bits - one
            if stm == fb.BLACK:
                square ^= 56
            feats[nf] = base + (square ^ mirror)
            nf += 1
    for j in range(hidden):
        acc[j] = b1[j]
    i = 0
    while i + 3 < nf:
        r1 = feats[i]
        r2 = feats[i + 1]
        r3 = feats[i + 2]
        r4 = feats[i + 3]
        for j in range(hidden):
            acc[j] += (w1[r1, j] + w1[r2, j]) + (w1[r3, j] + w1[r4, j])
        i += 4
    while i < nf:
        r1 = feats[i]
        for j in range(hidden):
            acc[j] += w1[r1, j]
        i += 1
    bucket = 0
    buckets = w2.shape[0]
    if buckets > 1:
        bucket = (nf - 2) // BUCKET_DIVISOR
        if bucket < 0:
            bucket = 0
        elif bucket >= buckets:
            bucket = buckets - 1
    total = np.int64(b2[bucket])
    for j in range(hidden):
        value = np.int64(acc[j])
        if value < 0:
            value = np.int64(0)
        elif value > QA:
            value = np.int64(QA)
        if SCRELU:
            value = value * value // QA
        total += value * np.int64(w2[bucket, j])
    return int(total) * CP_SCALE // (QA * QB)
