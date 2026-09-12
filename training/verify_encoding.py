"""Cross-check the packed encoding, the check filter and the numba feature extraction.

Random legal positions are generated with python-chess; for each one the packed record is
compared against a python-chess reference, ``parse_evals.in_check_mask`` against
``board.is_check()``, and ``nnue_eval.active_features`` (which reads a live fastboard state)
against ``encoding.features_from_record`` (which reads the training record).
"""

from __future__ import annotations

import random

import chess
import numpy as np

import _paths  # noqa: F401  (training/ and engine/ on sys.path)
import fastboard as fb
from encoding import (
    KING_BUCKETS,
    PIECE_FROM_SYMBOL,
    features_from_record,
    king_transform,
    pack_placement,
)
from nnue_eval import active_features
from parse_evals import in_check_mask


def reference_features(board: chess.Board, king_buckets: int = 1) -> set[int]:
    """Feature indices computed straight from python-chess, independent of the packer."""
    features = set()
    king = board.king(board.turn)
    assert king is not None
    if board.turn == chess.BLACK:
        king ^= 56
    mirror, offset = king_transform(king) if king_buckets > 1 else (0, 0)
    for square, piece in board.piece_map().items():
        code = PIECE_FROM_SYMBOL[piece.symbol()]
        if board.turn == chess.BLACK:
            code = code - 6 if code >= 6 else code + 6
            square ^= 56
        features.add(offset + code * 64 + (square ^ mirror))
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
        state = fb.from_fen(board.fen(en_passant="fen"))
        for buckets in (1, KING_BUCKETS):
            expected = reference_features(board, buckets)
            packed = set(features_from_record(int(occ[index]), nib[index], buckets))
            assert packed == expected, f"packed features differ at {board.fen()} ({buckets})"
            count, buffer = active_features(state, buckets)
            assert set(int(v) for v in buffer[:count]) == expected, \
                f"njit differ at {board.fen()} ({buckets})"
        assert max(reference_features(board, KING_BUCKETS)) < KING_BUCKETS * 768

    mask = in_check_mask(occ, nib)
    expected_mask = np.array([b.is_check() for b in boards])
    assert np.array_equal(mask, expected_mask), "check filter disagrees with python-chess"
    print(f"ok: {len(boards)} positions, features (plain and {KING_BUCKETS} king buckets) + "
          "check filter match python-chess")
    print(f"    in check: {int(expected_mask.sum())}")


if __name__ == "__main__":
    main()
