"""Depth and nodes per second on fixed positions, for one agent directory.

    uv run python -m bench.speed                      # 2s per position, what depth do we reach
    uv run python -m bench.speed --depth 5            # fixed depth: node counts are reproducible
    uv run python -m bench.speed --agent versions/v1  # compare a snapshot

Fixed-depth node counts are the honest measure of a search change (better ordering or pruning
means fewer nodes to the same depth). Timed depth is the honest measure of a speed change.
Both need the agent to expose analyse(); see bench/load.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess

from bench.load import Agent, fmt

POSITIONS: list[tuple[str, str]] = [
    ("start", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("open middlegame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9"),
    (
        "closed middlegame",
        "r1bqr1k1/1p1n1pbp/p2p2p1/2pPp3/2P1P3/2N1BN1P/PP2BPP1/R2Q1RK1 w - - 0 14",
    ),
    ("queenless", "r4rk1/1b3ppp/p1n1p3/1pbp4/8/2P1PN2/PP1B1PPP/R4RK1 w - - 0 15"),
    ("rook endgame", "8/5pk1/6p1/8/2R5/6P1/5PK1/3r4 w - - 0 40"),
    ("pawn endgame", "8/pp3k2/2p5/3p4/3P1K2/2P3P1/PP6/8 w - - 0 45"),
    ("KQ vs K", "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", type=Path, default=Path("engine"))
    parser.add_argument("--ms", type=float, default=2000.0, help="time per position")
    parser.add_argument("--depth", type=int, default=None, help="fixed depth instead of time")
    args = parser.parse_args()

    agent = Agent(args.agent)
    print(f"{agent.directory.name}: {'introspective' if agent.introspective else 'get_move only'}")
    print(
        f"{'position':20} {'move':6} {'depth':>5} {'score':>7} {'nodes':>9} {'ms':>7} {'knps':>6}"
    )
    nodes = 0
    ms = 0.0
    depths: list[int] = []
    for name, fen in POSITIONS:
        result = agent.analyse(fen, args.ms, args.depth)
        nodes += result.nodes or 0
        ms += result.elapsed_ms
        if result.depth is not None:
            depths.append(result.depth)
        print(
            f"{name:20} {result.move:6} {fmt(result.depth, 5)} {fmt(result.score, 7)} "
            f"{fmt(result.nodes, 9)} {result.elapsed_ms:7.0f} {fmt(result.knps, 6, 1)}"
        )
    summary = f"\ntotal {ms:.0f}ms"
    if agent.introspective:
        summary += f", {nodes} nodes = {nodes / max(ms, 1e-3):.1f} knps"
        summary += f", mean depth {sum(depths) / len(depths):.1f}"
    print(summary)


if __name__ == "__main__":
    main()
