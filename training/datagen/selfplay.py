"""Generate quiet, engine-labelled positions from self-play, for Texel tuning and NNUE training.

    uv run python work/datagen/selfplay.py --games 200 --seed 7 --out work/datagen/out/chunk_7.bin

One process, one core, deterministic for a seed. Each game starts from the initial position after
8-12 random plies, then the native engine plays both sides with a fixed node budget per move.
After ply 12 a position is kept when the side to move is not in check, the chosen move is not a
capture or promotion, and the quiescence score agrees with the static evaluation within
QUIET_MARGIN (the "quiet" test from the NNUE dataset study). Its label is the score of the search
that chose the move, from the mover's side, clipped to +-2000, plus the game result from the
mover's side (1 won, 0 drew, -1 lost). Records are the 27-byte packed board of
work/nnue/encoding.py followed by an int16 score and an int8 result, so both trainers read them.
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_OPT", "3")
ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path(os.environ.get("CHESSATHON_DATAGEN_ENGINE", ROOT)).resolve()  # engine to play with
sys.path.insert(0, str(ROOT / "work" / "nnue"))
sys.path.insert(0, str(ENGINE))

import chess  # noqa: E402
import encoding  # noqa: E402
import numpy as np  # noqa: E402

import fastboard as fb  # noqa: E402
import nativesearch as ns  # noqa: E402

RANDOM_PLIES = (6, 10)
OPENING_LIMIT = 600  # a game whose first search already sees a decided position is thrown away
NOISE_PLY_LIMIT = 24  # a random move now and then early on, for diversity; never later
NOISE_PROBABILITY = 0.03
QUIET_MARGIN = 60
MIN_PLY = 12
MAX_PLY = 300
SCORE_CLIP = 2000
NO_LIMIT = 1 << 62
RECORD = struct.Struct("<Q16shb")  # occ, nibbles, score (mover cp), result (mover): 27 bytes

Record = tuple[tuple[int, bytes], int, int]


class Player:
    """The native searcher driven by node counts instead of the clock, so output is reproducible."""

    def __init__(self, nodes: int) -> None:
        self.searcher = ns.Searcher()
        self.nodes = nodes
        self.keys: list[int] = []  # canonical repetition keys of the game so far

    def new_game(self) -> None:
        self.searcher.tt_depth.fill(-1)
        self.searcher.history.fill(0)
        self.searcher.killers.fill(0)
        self.keys = []

    def _load_history(self) -> int:
        trimmed = self.keys[-ns.MAX_GAME_HISTORY :]
        if trimmed:
            self.searcher.rep_keys[: len(trimmed)] = np.asarray(trimmed, dtype=np.uint64)
        return len(trimmed)

    def choose(self, board: chess.Board) -> tuple[chess.Move, int, int]:
        """Iterative deepening until the node budget is spent; returns move, score, static eval."""
        state = fb.from_fen(board.fen(en_passant="fen"))
        self.keys.append(int(ns._canonical_key(state)))
        rep_count = self._load_history()
        searcher = self.searcher
        searcher.history >>= 1
        searcher.stats[:] = 0
        best_move = np.uint32(0)
        best_score = 0
        for depth in range(1, ns.MAX_DEPTH + 1):
            searcher.stats[1] = 0
            limit = self.nodes if depth > 1 else NO_LIMIT  # depth 1 always completes
            score, move, aborted = searcher._iteration(
                state, depth, -ns.INF, ns.INF, rep_count, limit
            )
            if aborted:
                break
            best_move, best_score = move, int(score)
            if abs(best_score) >= ns.MATE_BOUND or searcher.stats[0] >= self.nodes:
                break
        candidate = chess.Move.from_uci(fb.move_to_uci(best_move))
        if candidate not in board.legal_moves:
            raise RuntimeError(f"native search returned illegal move {candidate} in {board.fen()}")
        return candidate, best_score, int(ns.evaluate_state(state))

    def quiescence(self, board: chess.Board) -> int:
        state = fb.from_fen(board.fen(en_passant="fen"))
        searcher = self.searcher
        searcher.stats[:] = 0
        rep_count = self._load_history()
        return int(
            ns._quiesce(
                state,
                -ns.INF,
                ns.INF,
                0,
                searcher.rep_keys,
                rep_count,
                searcher.tt_keys,
                searcher.tt_depth,
                searcher.tt_scores,
                searcher.tt_flags,
                searcher.tt_moves,
                searcher.tt_mask,
                searcher.killers,
                searcher.history,
                searcher.move_buffers,
                searcher.score_buffers,
                searcher.stats,
                NO_LIMIT,
            )
        )


def play_game(rng: random.Random, player: Player) -> list[Record]:
    board = chess.Board()
    for _ in range(rng.randint(*RANDOM_PLIES)):
        moves = list(board.legal_moves)
        if not moves:
            return []
        board.push(rng.choice(moves))
    player.new_game()
    kept: list[tuple[tuple[int, bytes], int, bool]] = []  # ((occ, nib), score, white_to_move)
    first = True
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLY:
        move, score, static = player.choose(board)
        if first and abs(score) > OPENING_LIMIT:
            return []
        first = False
        if (
            board.ply() >= MIN_PLY
            and not board.is_check()
            and not board.is_capture(move)
            and not move.promotion
            and abs(player.quiescence(board) - static) <= QUIET_MARGIN
        ):
            packed = encoding.pack_placement(board.board_fen(), not board.turn)
            kept.append((packed, max(-SCORE_CLIP, min(SCORE_CLIP, score)), board.turn))
        if board.ply() < NOISE_PLY_LIMIT and rng.random() < NOISE_PROBABILITY:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner
    records: list[Record] = []
    for packed, score, white_to_move in kept:
        result = 0 if winner is None else (1 if winner == white_to_move else -1)
        records.append((packed, score, result))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=20_000, help="node budget per move")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    player = Player(args.nodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total = 0
    plies = 0
    with open(args.out, "wb") as handle:
        for game in range(args.games):
            records = play_game(rng, player)
            plies += len(player.keys)
            for (occ, nib), score, result in records:
                handle.write(RECORD.pack(occ, nib, score, result))
            total += len(records)
            if game % 10 == 9:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"game {game + 1}/{args.games}: {total} positions from {plies} plies, "
                    f"{total / elapsed:.1f} positions/s",
                    flush=True,
                )
    print(
        f"done: {total} positions in {time.time() - started:.0f}s -> {args.out} (engine {ENGINE})"
    )


if __name__ == "__main__":
    main()
