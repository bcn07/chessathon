"""Load an agent directory for in-process benchmarking, whatever it exposes.

Every agent has get_move(fen, time_left_ms). An agent may also expose
analyse(fen, ms, max_depth) -> object with .move .score .depth .nodes .elapsed_ms, which lets a
benchmark control the time spent and see how deep it got. Without it, the benchmark hands
get_move a clock sized so a typical budget spends about the requested time, and reports only
what it can observe: the move and the wall time.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Benchmarks measure the search, not the opponent's clock: never ponder under bench tooling.
os.environ.setdefault("CHESSATHON_PONDER", "0")
from types import ModuleType

import chess

CLOCK_PER_MS = 30  # a get_move budget is usually clock / 30-ish; used only without analyse()


@dataclass(frozen=True)
class Result:
    move: str
    elapsed_ms: float
    score: int | None = None
    depth: int | None = None
    nodes: int | None = None

    @property
    def knps(self) -> float | None:
        if self.nodes is None:
            return None
        return self.nodes / max(self.elapsed_ms, 1e-3)


class Agent:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.module = _import(self.directory)
        self.introspective = hasattr(self.module, "analyse")

    def analyse(self, fen: str, ms: float, max_depth: int | None = None) -> Result:
        board = chess.Board(fen)
        if self.introspective:
            if max_depth is None:
                found = self.module.analyse(fen, ms)
            else:
                found = self.module.analyse(fen, 10**9, max_depth)
            move = found.move.uci() if isinstance(found.move, chess.Move) else str(found.move)
            return Result(move, found.elapsed_ms, found.score, found.depth, found.nodes)
        if max_depth is not None:
            raise SystemExit(
                f"{self.directory.name} has no analyse(); fixed-depth runs need it"
            )
        started = time.perf_counter()
        move = self.module.get_move(fen, int(ms * CLOCK_PER_MS))
        elapsed = (time.perf_counter() - started) * 1000.0
        if chess.Move.from_uci(move) not in board.legal_moves:
            raise SystemExit(f"{self.directory.name} played illegal {move} in {fen}")
        return Result(move, elapsed)


def _import(directory: Path) -> ModuleType:
    for name in [m for m in sys.modules if m == "agent" or m.startswith("agent.")]:
        del sys.modules[name]
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module("agent")
    finally:
        sys.path.remove(str(directory))


def fmt(value: int | float | None, width: int, decimals: int = 0) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{decimals}f}"
