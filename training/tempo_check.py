"""Measure the net's learned tempo and its cost, next to the hand-written evaluation.

    uv run python training/tempo_check.py

Tempo: eval(P, white to move) + eval(P, black to move) over random positions is twice the mover
bonus of a side-symmetric evaluation (PeSTO's is exactly 2 * TEMPO = 20)."""

from __future__ import annotations

import random
import time

import chess
import numpy as np

import _paths  # noqa: F401  (training/ and engine/ on sys.path)
import fastboard as fb
import nativesearch as ns
import nnue_eval as nn

WEIGHTS = ns.NNUE_W1_B1_W2_B2


def random_states(count: int) -> list[np.ndarray]:
    rng = random.Random(11)
    states: list[np.ndarray] = []
    while len(states) < count:
        board = chess.Board()
        for _ in range(rng.randint(6, 70)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            states.append(fb.from_fen(board.fen(en_passant="fen")))
    return states


def timed(function, sample: list[np.ndarray]) -> float:
    best = 1e9
    for _ in range(5):
        started = time.perf_counter()
        for state in sample:
            for _ in range(500):
                function(state)
        best = min(best, (time.perf_counter() - started) / (len(sample) * 500))
    return best * 1e9


def main() -> None:
    states = random_states(2000)
    net, pesto = [], []
    for state in states:
        flipped = state.copy()
        flipped[fb.SIDE] = 1 - flipped[fb.SIDE]
        net.append(nn.nnue_evaluate(state, *WEIGHTS) + nn.nnue_evaluate(flipped, *WEIGHTS))
        pesto.append(ns.evaluate_pesto(state) + ns.evaluate_pesto(flipped))
    print(f"net tempo   mean {np.mean(net) / 2:6.1f} cp  median {np.median(net) / 2:6.1f}")
    print(f"pesto tempo mean {np.mean(pesto) / 2:6.1f} cp  (MOVER_BIAS now {nn.MOVER_BIAS})")
    sample = states[:200]
    print(f"evaluate_state (net + mop-up) {timed(ns.evaluate_state, sample):6.0f} ns/call")
    print(f"evaluate_pesto                {timed(ns.evaluate_pesto, sample):6.0f} ns/call")


if __name__ == "__main__":
    main()
