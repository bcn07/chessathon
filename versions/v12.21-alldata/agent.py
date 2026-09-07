"""AI Chessathon entry: the driver that chooses between two engines and manages the game.

Two engines ship in this zip. ``pyengine`` (python-chess, ~20 knps) is ready the moment it is
imported. ``nativesearch`` (numba on ``fastboard``, ~2 M nps here) is a hundred times faster but
needs 20-40 s of compilation. The platform allows 90 s before the ready line, so the compile is
waited for at start-up (capped at 80 s; the platform has needed 52-78 s); should it still be
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
# On by default. The rules say the process keeps its core after get_move returns and call this
# "probably the largest single Elo lever available here"; a platform validation log says the
# process is suspended instead. If the log is right this thread is suspended too and costs
# nothing, so the bet is free either way -- and the between-moves probe below reports which it
# is. Our own benchmarks disable it (CHESSATHON_PONDER=0), because on the pool both engines
# share one core and a pondering side would steal the opponent's time.
PONDER = os.environ.get("CHESSATHON_PONDER", "1").lower() not in ("0", "", "false", "no")

# Does the process actually run between our moves? The docs say we keep our core and call
# pondering "probably the largest single Elo lever available here"; a platform validation log
# says the opposite ("your process is suspended while your opponent moves"). These two clocks
# settle it: perf_counter is wall time and advances even while suspended, process_time is our
# own CPU and does not. A gap with wall >> cpu means suspended and pondering is worthless; wall
# close to cpu means we are alive on the opponent's clock and pondering is worth turning on.
_last_move_exit: tuple[float, float] | None = None

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
# 42-78 s (its logs, host-dependent), and round 46 (2026-09-07) showed ~44 s of platform overhead
# outside this import on top of a normal 45 s compile: "Ready in 88.5 s" of a 90 s budget. So
# the import waits at most 40 s; a compile still running then finishes during the first move
# (get_move waits up to LATE_COMPILE_WAIT_S, charged to our clock) before the Python fallback.
INIT_COMPILE_WAIT_S = float(os.environ.get("CHESSATHON_INIT_COMPILE_WAIT", "40"))
LATE_COMPILE_WAIT_S = float(os.environ.get("CHESSATHON_LATE_COMPILE_WAIT", "30"))
INIT_BUDGET_S = 90.0  # what the platform's own logs report ("Budget 90.0 s")
INIT_SAFETY_S = 10.0  # left for the ready line, the referee's handshake and any slack


def _container_age_s() -> float | None:
    """Seconds since PID 1 started, or None when that cannot be trusted.

    Inside the platform's container PID 1 is the entrypoint, so this is the time already spent
    on the init budget before our process existed -- the part we cannot see from perf_counter.
    Round 46 (2026-09-07) lost 44 s of the budget that way while round 48 lost none, and we have
    no other way to tell those hosts apart. Outside a container (the Condor pool, a laptop) PID 1
    is the machine's init and the age is days, which the sanity check below rejects.
    """
    try:
        with open("/proc/uptime") as handle:
            uptime = float(handle.read().split()[0])
        with open("/proc/1/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        age = uptime - started
    except (OSError, ValueError, IndexError):
        return None
    return age if 0.0 <= age < INIT_BUDGET_S else None


def _init_wait_s() -> float:
    """How long the import may wait for the compile without risking the init budget."""
    if os.environ.get("CHESSATHON_INIT_COMPILE_WAIT"):
        return INIT_COMPILE_WAIT_S  # explicit setting (benchmarks) always wins
    age = _container_age_s()
    if age is None:
        return INIT_COMPILE_WAIT_S
    # Spend what is actually left rather than a fixed 40 s: on a host that wasted none of the
    # budget this waits ~75 s and the compile finishes inside init, instead of costing 30 s of
    # match clock on move one; on a host that has already burned 44 s it waits 36 s and still
    # reports ready by ~80 s.
    # The floor is 2 s, not a comfortable minimum: if the platform has already spent most of the
    # budget, returning at once is the only way to get the ready line out before it expires. The
    # compile then finishes on the match clock, which costs time but does not forfeit the game.
    return min(75.0, max(2.0, INIT_BUDGET_S - age - INIT_SAFETY_S))
if os.environ.get("CHESSATHON_NATIVE_ONLY"):
    # Benchmarking switch: compile however long it takes, so fast-clock games measure the native
    # engine alone.
    _compile_thread.join()
else:
    _compile_thread.join(_init_wait_s())
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
    global _last_move_exit
    if _last_move_exit is not None:
        wall = time.perf_counter() - _last_move_exit[0]
        cpu = time.process_time() - _last_move_exit[1]
        print(
            f"between-moves wall {wall:.2f}s cpu {cpu:.2f}s "
            f"({'ALIVE' if cpu > wall * 0.5 else 'SUSPENDED'})",
            file=sys.stderr,
        )
    try:
        _stop_pondering()
        board = _sync(fen)
        if _native is None and _compile_thread.is_alive():
            # init returned before the compile finished: spend a bounded slice of our clock on it
            # rather than open the game with the fallback engine
            waited_from = time.perf_counter()
            _compile_thread.join(min(LATE_COMPILE_WAIT_S, time_left_ms / 4000.0))
            waited_ms = int((time.perf_counter() - waited_from) * 1000)
            time_left_ms = max(time_left_ms - waited_ms, 1000)
            print(
                f"compile wait {waited_ms} ms in get_move: native engine "
                f"{'ready' if _native is not None else 'still compiling'}",
                file=sys.stderr,
            )
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
        _last_move_exit = (time.perf_counter(), time.process_time())
        return move.uci()
    except Exception:
        # Whatever went wrong, a legal move beats a crash. History is lost, the game is not.
        traceback.print_exc()
        _reset()
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        captures = [move for move in moves if board.is_capture(move)]
        return random.choice(captures or moves).uci()
