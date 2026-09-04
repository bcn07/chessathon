"""Cross-check the packed encoding, the check filter and the numba feature extraction.

Random legal positions are generated with python-chess; for each one the packed record is
compared against a python-chess reference, ``parse_evals.in_check_mask`` against
``board.is_check()``, and ``nnue_eval.active_features`` (which reads a live fastboard state)
against ``encoding.features_from_record`` (which reads the training record).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoding import PIECE_FROM_SYMBOL, features_from_record, pack_placement
from nnue_eval import active_features
from parse_evals import in_check_mask

import fastboard as fb


def reference_features(board: chess.Board) -> set[int]:
    """Feature indices computed straight from python-chess, independent of the packer."""
    features = set()
    for square, piece in board.piece_map().items():
        code = PIECE_FROM_SYMBOL[piece.symbol()]
        if board.turn == chess.BLACK:
            code = code - 6 if code >= 6 else code + 6
            square ^= 56
        features.add(code * 64 + square)
    return features


def random_positions(count: int, seed: int = 7) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    board = chess.Board()
    while len(boards) < count:
        if board.is_game_over() or board.fullmove_number > 80:
            board = chess.Board()
            continue
        board.push(rng.choice(list(board.legal_moves)))
        boards.append(board.copy())
    return boards


def main() -> None:
    boards = random_positions(3000)
    occ = np.empty(len(boards), dtype=np.uint64)
    nib = np.empty((len(boards), 16), dtype=np.uint8)
    for index, board in enumerate(boards):
        packed_occ, packed_nib = pack_placement(
            board.board_fen(), board.turn == chess.BLACK
        )
        occ[index] = packed_occ
        nib[index] = np.frombuffer(packed_nib, dtype=np.uint8)

    for index, board in enumerate(boards):
        expected = reference_features(board)
        packed = set(features_from_record(int(occ[index]), nib[index]))
        assert packed == expected, f"packed features differ at {board.fen()}"
        state = fb.from_fen(board.fen(en_passant="fen"))
        count, buffer = active_features(state)
        assert set(int(v) for v in buffer[:count]) == expected, f"njit differ at {board.fen()}"

    mask = in_check_mask(occ, nib)
    expected_mask = np.array([b.is_check() for b in boards])
    assert np.array_equal(mask, expected_mask), "check filter disagrees with python-chess"
    print(f"ok: {len(boards)} positions, features + check filter match python-chess")
    print(f"    in check: {int(expected_mask.sum())}")


if __name__ == "__main__":
    main()
