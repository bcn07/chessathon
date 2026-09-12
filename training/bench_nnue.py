"""Timing and cross-check for the shipped NNUE inference.

Times ``nnue_eval.nnue_evaluate`` and ``nativesearch.evaluate_pesto`` inside a compiled loop --
the number the search actually pays -- and again through the Python boundary, which is how
earlier reports quoted the ~235 ns PeSTO figure.  Also checks the numba integer path against the
independent numpy integer path in ``train.py`` so the shipped weights cannot silently disagree
with the ones the trainer measured.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import chess
import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fastboard as fb
import nativesearch as ns
import nnue_eval as nn


def sample_states(count: int, seed: int = 11) -> np.ndarray:
    rng = random.Random(seed)
    board = chess.Board()
    states = np.empty((count, fb.STATE_SIZE), dtype=np.uint64)
    filled = 0
    while filled < count:
        if board.is_game_over() or board.fullmove_number > 60:
            board = chess.Board()
            continue
        board.push(rng.choice(list(board.legal_moves)))
        states[filled] = fb.from_fen(board.fen(en_passant="fen"))
        filled += 1
    return states


@numba.njit(cache=True)
def _loop_nnue(states, reps, w1, b1, w2, b2):
    total = 0
    for _ in range(reps):
        for i in range(states.shape[0]):
            total += nn.nnue_evaluate(states[i], w1, b1, w2, b2)
    return total


@numba.njit(cache=True)
def _loop_pesto(states, reps):
    total = 0
    for _ in range(reps):
        for i in range(states.shape[0]):
            total += ns.evaluate_pesto(states[i])
    return total


@numba.njit(cache=True)
def _loop_full(states, reps):
    total = 0
    for _ in range(reps):
        for i in range(states.shape[0]):
            total += ns.evaluate_state(states[i])
    return total


def timed(function, states: np.ndarray, reps: int, *rest) -> float:
    function(states[:2], 1, *rest)
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        function(states, reps, *rest)
        best = min(best, time.perf_counter() - start)
    return best * 1e9 / (reps * len(states))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--tempo", action="store_true", help="measure the learned mover bonus")
    args = parser.parse_args()

    states = sample_states(args.positions)
    w1, b1, w2, b2 = nn.load_weights()

    from train import quantised_forward

    weights = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    flat_parts = []
    offsets = []
    cursor = 0
    for state in states:
        count, features = nn.active_features(state)
        flat_parts.append(features[:count].astype(np.int64))
        offsets.append(cursor)
        cursor += count
    flat = np.concatenate(flat_parts)
    reference = quantised_forward(weights, flat, np.array(offsets), len(states))
    native = np.array([nn.nnue_evaluate(state, w1, b1, w2, b2) for state in states])
    mismatch = int((reference != native).sum())
    print(f"numba vs numpy integer path: {mismatch} mismatches over {len(states)} positions")

    if args.tempo:
        # eval(P, white to move) + eval(P, black to move) is twice the tempo of a side-symmetric
        # evaluation, because one is the negation of the other apart from the mover bonus.
        net, pesto = [], []
        for state in states:
            placement = fb.to_fen(state).split()[0]
            white = fb.from_fen(f"{placement} w - - 0 1")
            black = fb.from_fen(f"{placement} b - - 0 1")
            net.append((nn.nnue_evaluate(white, w1, b1, w2, b2)
                        + nn.nnue_evaluate(black, w1, b1, w2, b2)) / 2)
            pesto.append((ns.evaluate_pesto(white) + ns.evaluate_pesto(black)) / 2)
        print(f"net tempo   mean {np.mean(net):7.1f} cp  median {np.median(net):7.1f}"
              f"  std {np.std(net):6.1f}")
        print(f"pesto tempo mean {np.mean(pesto):7.1f} cp  median {np.median(pesto):7.1f}")

    nnue_ns = timed(_loop_nnue, states, args.reps, w1, b1, w2, b2)
    pesto_ns = timed(_loop_pesto, states, args.reps)
    full_ns = timed(_loop_full, states, args.reps)
    print(f"compiled loop: nnue_evaluate      {nnue_ns:8.1f} ns/call")
    print(f"compiled loop: evaluate_pesto     {pesto_ns:8.1f} ns/call")
    print(f"compiled loop: evaluate_state     {full_ns:8.1f} ns/call  (net + draw heuristics)")
    print(f"ratio nnue/pesto: {nnue_ns / pesto_ns:.2f}x")

    # Warm both dispatchers first: the Python-facing signature is compiled on its first call,
    # and a one-second compile spread over the loop would swamp the measurement.
    nn.nnue_evaluate(states[0], w1, b1, w2, b2)
    ns.evaluate_pesto(states[0])
    start = time.perf_counter()
    for state in states:
        nn.nnue_evaluate(state, w1, b1, w2, b2)
    python_ns = (time.perf_counter() - start) * 1e9 / len(states)
    start = time.perf_counter()
    for state in states:
        ns.evaluate_pesto(state)
    python_pesto_ns = (time.perf_counter() - start) * 1e9 / len(states)
    print(f"from Python:   nnue_evaluate      {python_ns:8.1f} ns/call")
    print(f"from Python:   evaluate_pesto     {python_pesto_ns:8.1f} ns/call")


if __name__ == "__main__":
    main()
