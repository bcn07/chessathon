"""AI Chessathon entry: the driver that chooses between two engines and manages the game.

Two engines ship in this zip. ``pyengine`` (python-chess, ~20 knps) is ready the moment it is
imported. ``nativesearch`` (numba on ``fastboard``, ~2 M nps here) is a hundred times faster but
needs 20-40 s of compilation. The platform allows 90 s before the ready line, so the compile is
waited for at start-up (capped at 72 s; the platform needs 54-58 s); should it still be
running, the python engine answers the first moves and the driver switches over the instant
compilation finishes. If the native module ever fails to load, the python engine plays the game.

Between moves the active engine ponders: it keeps searching the position the opponent is looking
at, so the transposition table already holds what the next call needs. The next request stops
that search within a millisecond or two (the native search runs without the GIL and polls an
abort flag at every node). The game is rebuilt move by move from the FENs we are handed, so both
engines see the repetitions the referee would claim.
"""

from __future__ import annotations

import os

# Must precede any numba import (fastboard/nativesearch). Full optimisation costs about one
# extra second of compile, which the background thread absorbs, for ~5% more nodes.
os.environ.setdefault("NUMBA_OPT", "3")

import gc
import random
import sys
import threading
import time
import traceback
from types import ModuleType
from typing import Any

import chess

import pyengine

# Neither engine allocates reference cycles; the collector's sweeps over a large transposition
# table would only stall the clock.
gc.disable()

# The platform suspends the process while the opponent thinks (validation log), so pondering
# gains nothing there and only adds a thread hand-off per move; off unless asked for.
PONDER = bool(os.environ.get("CHESSATHON_PONDER"))

# ------------------------------------------------------------------------------------------
# Native engine, compiled in the background
# ------------------------------------------------------------------------------------------

_native: ModuleType | None = None
_native_error: str | None = None
_started_at = time.perf_counter()


def _load_native() -> None:
    global _native, _native_error
    try:
        import nativesearch

        _native = nativesearch
        elapsed = time.perf_counter() - _started_at
        print(f"native engine ready after {elapsed:.1f}s", file=sys.stderr)
    except Exception:
        _native_error = traceback.format_exc()
        print(f"native engine unavailable, playing pyengine:\n{_native_error}", file=sys.stderr)


_compile_thread = threading.Thread(target=_load_native, name="compile", daemon=True)
_compile_thread.start()

# The platform's validation log (2026-09-05) settled two things the docs left open: "your process
# is suspended while your opponent moves", so a background compile only progresses during our own
# moves and shares the core with the python engine (in the smoke games the native engine never
# came up), and the init budget is 90 s ("ready in 0.6 s of the 90 s init budget"). So the compile
# is waited for *before* the ready line, capped so that a 60 s budget would still be met; if the
# cap is hit the hybrid start-up below carries on exactly as before. The platform's core needs
# 54-58 s (its smoke logs), so the cap sits at 72 s: ready by ~73 s of the 90 s budget.
INIT_COMPILE_WAIT_S = float(os.environ.get("CHESSATHON_INIT_COMPILE_WAIT", "72"))
if os.environ.get("CHESSATHON_NATIVE_ONLY"):
    # Benchmarking switch: compile however long it takes, so fast-clock games measure the native
    # engine alone.
    _compile_thread.join()
else:
    _compile_thread.join(INIT_COMPILE_WAIT_S)
print(
    f"init: native engine {'ready' if _native is not None else 'still compiling'} "
    f"after {time.perf_counter() - _started_at:.1f}s",
    file=sys.stderr,
)


def native_ready() -> bool:
    return _native is not None


def wait_native(timeout: float | None = None) -> bool:
    """Block until the native engine has compiled (or failed). Benchmarks use this."""
    _compile_thread.join(timeout)
    return _native is not None


def _engine() -> ModuleType:
    return _native if _native is not None else pyengine


# ------------------------------------------------------------------------------------------
# Game history, rebuilt from the FENs we are handed
# ------------------------------------------------------------------------------------------

_game = chess.Board()
_history: list[str] = []  # FEN of every earlier position in this game, in order
_key_cache: dict[str, list[Any]] = {"pyengine": [], "nativesearch": []}


def _same_position(a: chess.Board, b: chess.Board) -> bool:
    return (
        a._transposition_key() == b._transposition_key()
        and a.halfmove_clock == b.halfmove_clock
        and a.fullmove_number == b.fullmove_number
    )


def _sync(fen: str) -> chess.Board:
    """Advance the remembered game to fen, keeping history when the opponent made one move."""
    global _game, _history
    target = chess.Board(fen)
    before = _game.fen()
    for move in list(_game.legal_moves):
        _game.push(move)
        if _same_position(_game, target):
            _history.append(before)
            return _game
        _game.pop()
    _game = target
    _history = []
    for cached in _key_cache.values():
        cached.clear()
    return _game


def _record(move: chess.Move) -> None:
    _history.append(_game.fen())
    _game.push(move)


def _reset() -> None:
    global _game, _history
    _stop_pondering()
    _game = chess.Board()
    _history = []
    for cached in _key_cache.values():
        cached.clear()


def _repetition_keys(engine: ModuleType) -> list[Any]:
    """The engine's own repetition key for every earlier position, extended incrementally."""
    cached = _key_cache[engine.__name__]
    if len(cached) > len(_history):
        cached.clear()
    for fen in _history[len(cached) :]:
        board = chess.Board(fen)
        if engine is pyengine:
            cached.append(board._transposition_key())
        else:
            cached.append(engine._native_repetition_key(board))
    return list(cached)


# ------------------------------------------------------------------------------------------
# Pondering
# ------------------------------------------------------------------------------------------

_ponder_thread: threading.Thread | None = None
_ponder_engine: ModuleType | None = None


def _ponder(engine: ModuleType, board: chess.Board, keys: list[Any]) -> None:
    try:
        if board.is_game_over():
            return
        if engine is pyengine:
            engine.SEARCHER.think(board, float("inf"), float("inf"), keys)
        else:
            engine.SEARCHER.ponder(board, keys)
    except Exception:
        traceback.print_exc()


def _start_pondering(engine: ModuleType) -> None:
    global _ponder_thread, _ponder_engine
    # While the native module is still compiling, a pondering python engine would only fight it
    # for the interpreter; the compile is worth more than a few nodes of lookahead.
    if not PONDER or _compile_thread.is_alive():
        return
    if engine is pyengine:
        engine.SEARCHER.abort = False
    else:
        engine.SEARCHER.clear_abort()
    thread = threading.Thread(
        target=_ponder,
        args=(engine, _game.copy(), _repetition_keys(engine)),
        name="ponder",
        daemon=True,
    )
    _ponder_engine = engine
    _ponder_thread = thread
    thread.start()


def _stop_pondering() -> None:
    global _ponder_thread, _ponder_engine
    thread, engine = _ponder_thread, _ponder_engine
    if thread is None or engine is None:
        return
    # Re-assert the abort until the thread is gone: a ponder search that had not yet started its
    # first iteration when we first asked would otherwise run on with the flag it reset.
    while thread.is_alive():
        if engine is pyengine:
            engine.SEARCHER.abort = True
        else:
            engine.SEARCHER.abort()
        thread.join(0.02)
    _ponder_thread = None
    _ponder_engine = None
    if engine is pyengine:
        engine.SEARCHER.abort = False
    else:
        engine.SEARCHER.clear_abort()


# ------------------------------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------------------------------


def budget(time_left_ms: int) -> tuple[float, float]:
    soft, hard = _engine().budget(time_left_ms)
    return float(soft), float(hard)


def evaluate(board: chess.Board) -> int:
    return int(_engine().evaluate(board))


def analyse(fen: str, ms: float, max_depth: int = 40) -> Any:
    """Search one position with the best engine available; benchmarks wait for the native one."""
    wait_native()
    return _engine().analyse(fen, ms, max_depth)


def get_move(fen: str, time_left_ms: int) -> str:
    try:
        _stop_pondering()
        board = _sync(fen)
        engine = _engine()
        if engine is pyengine:
            soft, hard = engine.budget(time_left_ms)
        else:
            soft, hard = engine.budget(time_left_ms, board.fullmove_number)
        keys = _repetition_keys(engine)
        result = engine.SEARCHER.think(board, soft, hard, keys)
        move: chess.Move = result.move
        if move not in board.legal_moves:
            raise ValueError(f"search returned illegal move {move}")
        _record(move)
        print(
            f"{engine.__name__} {move.uci()} depth {result.depth} score {result.score} "
            f"nodes {result.nodes} {result.elapsed_ms:.0f}ms clock {time_left_ms}",
            file=sys.stderr,
        )
        _start_pondering(engine)
        return move.uci()
    except Exception:
        # Whatever went wrong, a legal move beats a crash. History is lost, the game is not.
        traceback.print_exc()
        _reset()
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        captures = [move for move in moves if board.is_capture(move)]
        return random.choice(captures or moves).uci()
