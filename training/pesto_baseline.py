"""Score the hand-written PeSTO evaluation on the same held-out positions as the net.

Without this the net's centipawn MAE has nothing to be compared against: the held-out targets are
deep Stockfish scores, and a large residual may simply be what a static evaluation cannot know.
Packed records are rebuilt into fastboard states (canonical form, so the mover is White; castling
rights, en passant and the clocks are not stored and are left empty, which only disables the
fifty-move decay in `evaluate_pesto`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numba
import numpy as np

import _paths  # noqa: F401  (training/ and engine/ on sys.path)
import fastboard as fb
import nativesearch as ns


@numba.njit
def _pesto_scores(occ: np.ndarray, nib: np.ndarray) -> np.ndarray:
    out = np.empty(occ.shape[0], dtype=np.int32)
    state = np.zeros(fb.STATE_SIZE, dtype=np.uint64)
    one = np.uint64(1)
    for i in range(occ.shape[0]):
        for piece in range(12):
            state[piece] = np.uint64(0)
        bits = occ[i]
        count = 0
        while bits:
            square = fb.lsb_square(bits)
            bits &= bits - one
            byte = np.int64(nib[i, count >> 1])
            shift = 4 if count & 1 else 0
            code = (byte >> shift) & 15
            state[code] |= one << np.uint64(square)
            count += 1
        white = np.uint64(0)
        black = np.uint64(0)
        for piece in range(6):
            white |= state[piece]
            black |= state[piece + 6]
        state[fb.WHITE_OCC] = white
        state[fb.BLACK_OCC] = black
        state[fb.ALL_OCC] = occ[i]
        state[fb.SIDE] = np.uint64(fb.WHITE)
        state[fb.EP_SQUARE] = np.uint64(fb.NO_SQUARE)
        state[fb.CASTLING] = np.uint64(0)
        state[fb.HALFMOVE] = np.uint64(0)
        out[i] = ns.evaluate_pesto(state)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/data/positions.npy"))
    parser.add_argument("--holdout", type=int, default=1_000_000)
    args = parser.parse_args()

    records = np.load(args.data, mmap_mode="r")
    order = np.random.default_rng(20260904).permutation(len(records))[: args.holdout]
    order.sort()  # a sorted gather off the memmap, same set of positions as train.py
    occ = np.ascontiguousarray(records["occ"][order])
    nib = np.ascontiguousarray(records["nib"][order])
    target = np.ascontiguousarray(records["score"][order]).astype(np.float32)
    predicted = np.clip(_pesto_scores(occ, nib).astype(np.float32), -2000, 2000)
    quiet = np.abs(target) < 400
    print(f"held-out positions           {len(order):,}")
    print(f"PeSTO MAE (cp)               {np.abs(predicted - target).mean():.1f}")
    print(f"PeSTO MAE, |target|<400 (cp) {np.abs(predicted[quiet] - target[quiet]).mean():.1f}")
    print(f"PeSTO correlation            {np.corrcoef(predicted, target)[0, 1]:.4f}")
    print(f"PeSTO sign agreement         {np.mean((predicted > 0) == (target > 0)):.4f}")


if __name__ == "__main__":
    main()
