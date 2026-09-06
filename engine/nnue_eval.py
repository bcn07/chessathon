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

# Tempo learned from the training labels. The labels are the engine's own search scores on
# self-play positions (mean +51 cp for the side to move, median +20), so the net learns a mover
# bonus of its own; `tempo_check.py` measures it and this constant brings it back to the +10 that
# the search's draw scores and pruning margins were tuned around.
MOVER_BIAS = 15  # measured 25 cp learned tempo, PeSTO has 10
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "nnue.npz"


def load_weights(path: Path | str = WEIGHTS_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (w1, b1, w2, b2) as C-contiguous arrays ready for ``nnue_evaluate``."""
    with np.load(path) as data:
        w1 = np.ascontiguousarray(data["w1"], dtype=np.int16)
        b1 = np.ascontiguousarray(data["b1"], dtype=np.int16)
        w2 = np.ascontiguousarray(data["w2"], dtype=np.int16)
        b2 = np.int32(data["b2"])
    return w1, b1, w2, int(b2)


@numba.njit
def active_features(state: np.ndarray) -> tuple[int, np.ndarray]:
    """The active feature indices for a fastboard state (used by tests, not by the search)."""
    out = np.empty(32, dtype=np.int16)
    count = 0
    stm = int(state[fb.SIDE])
    one = np.uint64(1)
    for piece in range(12):
        piece_type = piece % 6
        code = piece_type if piece // 6 == stm else piece_type + 6
        base = code * 64
        bits = state[piece]
        while bits:
            square = int(fb.lsb_square(bits))
            bits &= bits - one
            if stm == fb.BLACK:
                square ^= 56
            out[count] = base + square
            count += 1
    return count, out


@numba.njit
def nnue_evaluate(
    state: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: int
) -> int:
    """Static centipawn score relative to the state's side to move."""
    hidden = b1.shape[0]
    acc = b1.copy()
    stm = int(state[fb.SIDE])
    one = np.uint64(1)
    for piece in range(12):
        piece_type = piece % 6
        code = piece_type if piece // 6 == stm else piece_type + 6
        base = code * 64
        bits = state[piece]
        while bits:
            square = int(fb.lsb_square(bits))
            bits &= bits - one
            if stm == fb.BLACK:
                square ^= 56
            row = w1[base + square]
            for j in range(hidden):
                acc[j] += row[j]
    total = np.int32(b2)
    for j in range(hidden):
        value = acc[j]
        if value < 0:
            value = 0
        elif value > QA:
            value = QA
        total += np.int32(value) * np.int32(w2[j])
    return int(total) * CP_SCALE // (QA * QB)
