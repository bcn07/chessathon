"""Tactical test suite: does the agent find the move, and how fast.

    uv run python -m bench.tactics                # 3s per position
    uv run python -m bench.tactics --ms 1000      # tighter
    uv run python -m bench.tactics --agent versions/v1

Positions come with the accepted moves in UCI. Mates are checked at load by a brute-force
mate finder, so a wrong entry is reported rather than trusted; the remaining positions are
regression tests on the engine's own judgement. Solved count and time-to-solve both matter:
a change that solves the same set faster is a real gain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess

from bench.load import Agent, fmt

# name, fen, accepted moves, mate distance (0 when the position is not a forced mate)
SUITE: list[tuple[str, str, tuple[str, ...], int]] = [
    ("mate1 smothered", "6rk/6pp/7P/6N1/8/8/8/7K w - - 0 1", ("g5f7",), 1),
    ("mate1 back rank", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ("a1a8",), 1),
    (
        "mate1 scholar",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        ("h5f7",),
        1,
    ),
    ("mate1 arabian", "7k/R7/5N2/8/8/8/8/7K w - - 0 1", ("a7h7",), 1),
    ("promote", "8/1P4k1/8/8/8/8/6K1/8 w - - 0 1", ("b7b8q",), 0),
    ("WAC.001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", ("g3g6",), 0),
    ("WAC.002", "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1", ("b3b2",), 0),
    ("WAC.003", "5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1", ("e3g3",), 0),
    ("WAC.006", "7k/p7/1R5K/6r1/6p1/6P1/8/8 w - - 0 1", ("b6b7",), 0),
    ("WAC.007", "rnbqkb1r/pppp1ppp/8/4P3/6n1/7P/PPPNPPP1/R1BQKBNR b KQkq - 0 1", ("g4e3",), 0),
    ("WAC.008", "r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - - 0 1", ("e7f7",), 0),
    # Rxh2 is the book answer; Bxh2+ Kh1 Bg3+ Kg1 Rh1+ Kxh1 Qh4+ Kg1 Qh2# also mates
    ("WAC.010", "3q1rk1/p4pp1/2pb3p/3p4/6Pr/1PNQ4/P1PB1PP1/4RRK1 b - - 0 1", ("h4h2", "d6h2"), 0),
    ("WAC.012", "4k1r1/2p3r1/1pR1p3/3pP2p/3P2qP/P4N2/1PQ4P/5R1K b - - 0 1", ("g4f3",), 0),
    (
        "WAC.014",
        "r2rb1k1/pp1q1p1p/2n1p1p1/2bp4/5P2/PP1BPR1Q/1BPN2PP/R5K1 w - - 0 1",
        ("h3h7",),
        0,
    ),
    ("WAC.016", "r4rk1/ppp2ppp/2n5/2bqp3/8/P2PB3/1PP1NPPP/R2Q1RK1 w - - 0 1", ("e2c3",), 0),
    ("WAC.018", "R7/P4k2/8/8/8/8/r7/6K1 w - - 0 1", ("a8h8",), 0),
    ("knight fork", "2q3k1/5ppp/8/3N4/8/8/5PPP/6K1 w - - 0 1", ("d5e7",), 0),
    ("skewer", "4q3/4k3/8/8/8/8/8/R5K1 w - - 0 1", ("a1e1",), 0),
]


def mating_moves(board: chess.Board, plies: int) -> set[str]:
    """Every move that forces mate within `plies` half-moves, by brute force."""

    def forced(b: chess.Board, remaining: int) -> bool:
        # side to move is the defender; true if every reply still loses to a mate in time
        if b.is_checkmate():
            return True
        if remaining <= 0 or b.is_stalemate() or b.is_insufficient_material():
            return False
        for reply in b.legal_moves:
            b.push(reply)
            try:
                if not any_mates(b, remaining - 1):
                    return False
            finally:
                b.pop()
        return True

    def any_mates(b: chess.Board, remaining: int) -> bool:
        if remaining <= 0:
            return False
        for move in b.legal_moves:
            b.push(move)
            try:
                if forced(b, remaining - 1):
                    return True
            finally:
                b.pop()
        return False

    found = set()
    for move in board.legal_moves:
        board.push(move)
        try:
            if forced(board, plies - 1):
                found.add(move.uci())
        finally:
            board.pop()
    return found


def validate() -> list[tuple[str, str, tuple[str, ...]]]:
    """Return the usable rows, printing anything that does not hold up."""
    usable = []
    for name, fen, accepted, mate_in in SUITE:
        try:
            board = chess.Board(fen)
        except ValueError as error:
            print(f"  dropped {name}: bad fen ({error})")
            continue
        if not board.is_valid():
            print(f"  dropped {name}: illegal position ({board.status()!r})")
            continue
        legal = {m.uci() for m in board.legal_moves}
        if not set(accepted) <= legal:
            print(f"  dropped {name}: accepted moves {set(accepted) - legal} are not legal")
            continue
        if mate_in:
            mates = mating_moves(board, 2 * mate_in - 1)
            if set(accepted) != mates:
                print(f"  fixed {name}: forced mates are {sorted(mates)}, not {list(accepted)}")
                if not mates:
                    continue
                accepted = tuple(sorted(mates))
        usable.append((name, fen, accepted))
    return usable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", type=Path, default=Path("engine"))
    parser.add_argument("--ms", type=float, default=3000.0, help="time per position")
    args = parser.parse_args()

    agent = Agent(args.agent)
    print(f"{agent.directory.name}: {'introspective' if agent.introspective else 'get_move only'}")
    rows = validate()
    solved = 0
    total_ms = 0.0
    print(f"\n{'position':18} {'want':14} {'got':6} {'depth':>5} {'ms':>6}  result")
    for name, fen, accepted in rows:
        result = agent.analyse(fen, args.ms)
        ok = result.move in accepted
        solved += ok
        total_ms += result.elapsed_ms
        print(
            f"{name:18} {'/'.join(accepted):14} {result.move:6} {fmt(result.depth, 5)} "
            f"{result.elapsed_ms:6.0f}  {'ok' if ok else 'MISSED'}"
        )
    print(f"\nsolved {solved}/{len(rows)} in {total_ms / 1000:.1f}s at {args.ms:.0f}ms each")


if __name__ == "__main__":
    main()
