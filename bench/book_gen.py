"""Build an opening table over the platform's start positions with Stockfish.

The docs (2026-09-12) allow a shipped table that answers positions at move 20 or lower, whatever
produced it. From each root we search the position (MultiPV), record the best move, and expand the
replies that are within ``--window`` cp of the best, breadth-first, to ``--max-plies`` plies or
``--cap`` searched nodes per root. Every searched node is a table entry, so the engine answers from
the table as either colour for as long as the opponent stays inside the tree.

    uv run python -m bench.book_gen --start 0 --count 1 --out part.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import chess
import chess.engine


def key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, default=Path("bench/book_roots.json"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", default="tools/engines/stockfish")
    parser.add_argument("--nodes", type=int, default=2_500_000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--window", type=int, default=40, help="reply window in cp")
    parser.add_argument("--max-plies", type=int, default=6)
    parser.add_argument("--cap", type=int, default=160, help="searched nodes per root")
    parser.add_argument("--hash", type=int, default=256)
    args = parser.parse_args()

    with open(args.roots) as handle:
        roots = json.load(handle)[args.start : args.start + args.count]
    engine = chess.engine.SimpleEngine.popen_uci(args.engine, timeout=120.0)  # slow NFS nodes
    engine.configure({"Threads": 1, "Hash": args.hash})
    book: dict[str, list] = {}
    started = time.time()
    for root in roots:
        queue: deque[tuple[chess.Board, int]] = deque([(chess.Board(root["fen"]), 0)])
        searched = 0
        while queue and searched < args.cap:
            board, ply = queue.popleft()
            k = key(board.fen())
            if k in book or board.is_game_over():
                continue
            limit = chess.engine.Limit(nodes=args.nodes)
            infos = engine.analyse(board, limit, multipv=args.multipv)
            lines = []
            for info in infos:
                if "pv" not in info or not info["pv"]:
                    continue
                score = info["score"].relative.score(mate_score=10000)
                lines.append((info["pv"][0], int(score), int(info.get("depth", 0))))
            if not lines:
                continue
            best_move, best_cp, depth = lines[0]
            book[k] = [best_move.uci(), best_cp, depth, ply]
            searched += 1
            if ply + 1 < args.max_plies:
                for move, cp, _ in lines:
                    if best_cp - cp <= args.window:
                        child = board.copy(stack=False)
                        child.push(move)
                        queue.append((child, ply + 1))
        print(f"{root['name']}: {searched} nodes, {time.time() - started:.0f} s so far", flush=True)
    engine.quit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "nodes": args.nodes, "multipv": args.multipv, "window": args.window,
        "max_plies": args.max_plies, "cap": args.cap, "roots": [r["name"] for r in roots],
    }
    with open(args.out, "w") as handle:
        json.dump({"meta": meta, "book": book}, handle)
    print(f"wrote {len(book)} entries to {args.out}")


if __name__ == "__main__":
    main()
