"""Build a large balanced opening book from our own gauntlet games.

    uv run python -m bench.build_book --out bench/book.json --target 1000

One position per game at a random ply in [8, 16] from ``results/*/games_*.pgn`` (real-clock
runs first), duplicates dropped (FEN without move counters), positions in check dropped, and only
positions the root engine scores within ``--max-cp`` at ``--ms`` per search kept. The gauntlet
plays every entry once with each colour, so the side to move does not need balancing.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]


def candidates(rng: random.Random, low: int, high: int) -> list[tuple[str, str]]:
    files = sorted(glob.glob(str(ROOT / "results/*-real/games_*.pgn")))
    files += sorted(
        f for f in glob.glob(str(ROOT / "results/*/games_*.pgn")) if "-real/" not in f
    )
    seen: set[str] = set()
    found: list[tuple[str, str]] = []
    for path in files:
        tag = Path(path).parent.name
        with open(path) as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                moves = list(game.mainline_moves())
                if len(moves) < high + 4:
                    continue
                ply = rng.randint(low, high)
                board = game.board()
                for move in moves[:ply]:
                    board.push(move)
                if board.is_check() or board.is_game_over():
                    continue
                key = " ".join(board.fen().split()[:4])
                if key in seen:
                    continue
                seen.add(key)
                found.append((f"{tag}/{game.headers.get('Round', '?')} ply {ply}", board.fen()))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "bench/book.json")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--max-cp", type=int, default=40)
    parser.add_argument("--ms", type=float, default=200)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = candidates(rng, args.min_ply, args.max_ply)
    rng.shuffle(pool)
    print(f"{len(pool):,} distinct candidate positions", flush=True)

    sys.path.insert(0, str(ROOT))
    import agent  # the root engine; compiles numba on import

    kept: list[dict[str, str | int]] = []
    scored = 0
    started = time.perf_counter()
    for name, fen in pool:
        if len(kept) >= args.target:
            break
        found = agent.analyse(fen, args.ms)
        scored += 1
        if found.score is None or abs(found.score) > args.max_cp:
            continue
        kept.append({"name": f"b{len(kept):04d} {name.split(' ply ')[1]}", "fen": fen,
                     "score": int(found.score), "source": name})
        if len(kept) % 100 == 0:
            print(f"  kept {len(kept)} of {scored} scored ({time.perf_counter() - started:.0f} s)",
                  flush=True)
    args.out.write_text(json.dumps(kept, indent=0) + "\n")
    print(f"wrote {args.out}: {len(kept)} positions (scored {scored}, "
          f"{len(kept) / max(scored, 1):.0%} within ±{args.max_cp} cp)")


if __name__ == "__main__":
    main()
