"""Stockfish self-play data generation: diverse positions with strong labels.

    uv run python training/datagen/sfgen.py --games 200 --seed 1 --nodes 8000 \
        --engine tools/engines/stockfish --out results/datagen/sf1/chunk_0.bin

Each game starts from the initial position, plays 8-12 random plies to diversify the opening
(rejected and retried if that leaves either side more than 300 cp down), then Stockfish plays
itself at a fixed node count to the end. Every position is written with Stockfish's score for the
side to move and, once the game ends, the game result from that side's view -- the same 27-byte
record ``selfplay.py`` writes, so ``build_set.py`` reads a directory of these as a generation.

Positions where the side to move is in check, or where Stockfish's best move is a capture or a
promotion, are dropped: a static evaluation cannot learn them and they are what skewed the
Lichess corpus (REPORT.md). Mate scores are clipped to +/-2000 and those rows are dropped by
``build_set.py`` anyway.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nnue"))
from encoding import pack_placement

RECORD = struct.Struct("<Q16shb")  # occ, nibbles, score (mover cp), result (mover)
CLAMP = 2000
OPENING_BALANCE = 300  # reject a random opening that already loses this much
MAX_PLIES = 240


def random_opening(rng: random.Random, engine: chess.engine.SimpleEngine,
                   limit: chess.engine.Limit, book: list[str] | None) -> chess.Board:
    """A balanced opening position to play from.

    Without ``book``, 8-12 random plies from the initial position: broad coverage. With one (the
    ladder's own start positions), a random book entry plus 2-5 random plies, which keeps the
    data in the neighbourhood the competition actually starts from while still varying every
    game -- Stockfish is deterministic, so an unrandomised book entry would replay one game.
    Either way the position is rejected if a side is already lost.
    """
    while True:
        if book:
            board = chess.Board(rng.choice(book))
            plies = rng.randint(2, 5)
        else:
            board = chess.Board()
            plies = rng.randint(8, 12)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or not board.legal_moves:
            continue
        info = engine.analyse(board, limit)
        score = info["score"].pov(board.turn).score(mate_score=CLAMP)
        if score is not None and abs(score) <= OPENING_BALANCE:
            return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=8000)
    parser.add_argument("--engine", default="tools/engines/stockfish")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--book", type=Path, default=None,
                        help="json list of {'fen': ...} start positions (bench/platform_book.json)")
    args = parser.parse_args()

    book = None
    if args.book is not None:
        book = [entry["fen"] for entry in json.loads(args.book.read_text())]
        print(f"book: {len(book)} start positions from {args.book}")

    rng = random.Random(args.seed)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 32})
    limit = chess.engine.Limit(nodes=args.nodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    written = 0
    with args.out.open("wb") as handle:
        for game in range(args.games):
            board = random_opening(rng, engine, limit, book)
            pending: list[tuple[tuple[int, bytes], int, bool]] = []
            while not board.is_game_over(claim_draw=False) and len(board.move_stack) < MAX_PLIES:
                info = engine.analyse(board, limit)
                move = info.get("pv", [None])[0]
                if move is None:
                    break
                score = info["score"].pov(board.turn).score(mate_score=CLAMP)
                quiet = (
                    not board.is_check()
                    and not board.is_capture(move)
                    and move.promotion is None
                    and score is not None
                )
                if quiet:
                    packed = pack_placement(board.board_fen(), board.turn == chess.BLACK)
                    pending.append((packed, max(-CLAMP, min(CLAMP, int(score))), board.turn))
                board.push(move)
            outcome = board.outcome(claim_draw=True)
            winner = None if outcome is None else outcome.winner
            for (occ, nib), score, turn in pending:
                result = 0 if winner is None else (1 if winner == turn else -1)
                handle.write(RECORD.pack(occ, nib, score, result))
                written += 1
            if (game + 1) % 20 == 0:
                rate = written / (time.perf_counter() - started)
                print(f"game {game + 1}/{args.games}: {written:,} positions, {rate:.1f}/s",
                      flush=True)
    elapsed = time.perf_counter() - started
    engine.quit()
    print(f"wrote {args.out}: {written:,} positions in {elapsed:.0f}s "
          f"({written / max(elapsed, 1e-9):.1f}/s)")


if __name__ == "__main__":
    main()
