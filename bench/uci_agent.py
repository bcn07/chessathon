"""Wrap a UCI engine as a harness agent directory, for local sparring only.

Nothing here ships: the ban on third-party engines covers the submission, and these engines
exist so we can measure against the house bots the ladder actually fields, plus Stockfish at
a chosen Elo. An opponent directory is two lines:

    from bench.uci_agent import make_get_move
    get_move = make_get_move("tools/engines/rustic", {"Hash": 64})

The engine gets the same clock we do (the referee's time_left and the event increment), one
thread, no pondering, and runs for the whole game in one process like every other agent.
"""

from __future__ import annotations

import atexit
import os
from collections.abc import Callable
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
# The gauntlet exports the increment it plays with; the event's is the default.
INCREMENT_S = int(os.environ.get("HARNESS_INCREMENT_MS", "500")) / 1000.0
SAFETY = 0.9  # engines plan to the edge of their clock; the referee does not forgive


def make_get_move(
    binary: str, options: dict[str, object] | None = None
) -> Callable[[str, int], str]:
    path = ROOT / binary
    engine: chess.engine.SimpleEngine | None = None

    def start() -> chess.engine.SimpleEngine:
        nonlocal engine
        if engine is None:
            if not path.exists():
                raise FileNotFoundError(f"{path} — build it, see tools/engines/README.md")
            engine = chess.engine.SimpleEngine.popen_uci(str(path))
            wanted = {"Threads": 1, **(options or {})}  # Ponder is managed by python-chess
            engine.configure({k: v for k, v in wanted.items() if k in engine.options})
            atexit.register(engine.quit)
        return engine

    def get_move(fen: str, time_left_ms: int) -> str:
        board = chess.Board(fen)
        clock = time_left_ms / 1000.0 * SAFETY
        limit = chess.engine.Limit(
            white_clock=clock, black_clock=clock, white_inc=INCREMENT_S, black_inc=INCREMENT_S
        )
        result = start().play(board, limit, ponder=False)
        if result.move is None:
            raise RuntimeError("engine returned no move")
        return result.move.uci()

    return get_move
