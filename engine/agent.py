"""AI Chessathon entry: the driver that answers get_move(fen, time_left_ms) and manages the game.

Two engines ship in this zip. ``pyengine`` (python-chess, ~20 knps) is ready the moment it is
imported. ``nativesearch`` (numba on ``fastboard``, ~1 M nps on the platform's core) is far
stronger but needs 20-27 s of compilation there, against a 30 s init budget at the final. The
compile runs in a background thread; import waits for it only as long as the budget allows
(``INIT_BUDGET_S`` minus the container's age minus ``INIT_SAFETY_S``) and then prints the ready
line. A compile still running finishes during the first moves: the opening table answers those
instantly and lends the compile a bounded slice of clock, otherwise the first searched move waits
up to ``LATE_COMPILE_WAIT_S``. Only if the native module fails to load does the python engine play.

Move selection, in order: the opening table (``weights/book.json``, positions at move 20 or
earlier, skipped once a position has repeated), the Syzygy root probe (``tablebase.py``, proven
wins only), then the search, whose move the tablebase may veto if it drops a win to a draw.

The platform suspends the process while the opponent thinks, so nothing runs between moves
(pondering is off unless ``CHESSATHON_PONDER`` is set). The game is rebuilt move by move from the
FENs we are handed, so the search sees the repetitions the referee would claim.
"""

from __future__ import annotations

import os

# Must precede any numba import (fastboard/nativesearch). Full optimisation costs about one
# extra second of compile, which the background thread absorbs, for ~5% more nodes.
os.environ.setdefault("NUMBA_OPT", "3")

import gc
import json
import random
import sys
import threading
import time
import traceback
from types import ModuleType
from typing import Any

import chess

import pyengine
import tablebase

# Neither engine allocates reference cycles; the collector's sweeps over a large transposition
# table would only stall the clock.
gc.disable()

# The platform suspends the process while the opponent thinks (validation log), so pondering
# gains nothing there and only adds a thread hand-off per move; off unless asked for.
# Measured, not assumed: round 59 (2026-09-07, 122 moves) reported gaps of 1.4-3.7 s of wall
# time between our moves against a flat 0.04-0.05 s of our own CPU -- the platform really does
# suspend the process while the opponent thinks, exactly as its validation log says and contrary
# to the written rules, which call pondering "probably the largest single Elo lever available
# here". A ponder thread gets no CPU there and only costs the ~45 ms per move it takes to spawn
# it and tear it down. Off unless CHESSATHON_PONDER says otherwise.
PONDER = os.environ.get("CHESSATHON_PONDER", "0").lower() not in ("0", "", "false", "no")


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

# Opening table (finals day, 2026-09-12). The docs allow a shipped table that answers positions at
# move 20 or lower. weights/book.json maps the first four FEN fields of a position reachable from
# the platform's start positions to [best move, eval cp, depth, ply] from a Stockfish tree
# (bench/book_gen.py). A table move is answered at once, which banks clock for the phase where we
# used to run dry, and a compile that missed the init budget gets a bounded slice of that time.
BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "book.json")
BOOK_MAX_FULLMOVE = 20
BOOK_COMPILE_SLICE_S = 5.0
_book: dict[str, list[Any]] = {}


def _load_book() -> None:
    global _book
    try:
        with open(BOOK_PATH) as handle:
            loaded = json.load(handle)
        _book = loaded["book"] if isinstance(loaded, dict) and "book" in loaded else loaded
        if not isinstance(_book, dict):
            raise TypeError(f"table is {type(_book).__name__}, not dict")
        print(f"opening table: {len(_book)} positions", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _book = {}
        print(f"opening table unavailable: {exc}", file=sys.stderr)


def _book_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def _book_move(board: chess.Board) -> tuple[chess.Move, int] | None:
    if not _book or board.fullmove_number > BOOK_MAX_FULLMOVE:
        return None
    entry = _book.get(_book_key(board))
    if not entry:
        return None
    try:
        move = chess.Move.from_uci(str(entry[0]))
        cp = int(entry[1]) if len(entry) > 1 else 0
    except (ValueError, TypeError, KeyError, IndexError):
        return None
    if move not in board.legal_moves:
        return None
    return move, cp


_load_book()

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
# 18 = the 30 s budget minus 10 s safety minus ~2 s of process age at import
INIT_COMPILE_WAIT_S = float(os.environ.get("CHESSATHON_INIT_COMPILE_WAIT", "18"))
LATE_COMPILE_WAIT_S = float(os.environ.get("CHESSATHON_LATE_COMPILE_WAIT", "30"))
INIT_BUDGET_S = 30.0  # the London final cut the init budget from 90 s to 30 s (finals day)
INIT_SAFETY_S = 10.0  # ready line never past ~20 s; 8 s platform stalls were seen after import


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
    try:
        _stop_pondering()
        board = _sync(fen)
        booked = _book_move(board)
        if booked is not None and _book_key(board) in {" ".join(f.split()[:4]) for f in _history}:
            booked = None  # position seen before this game: let the search weigh the repetition
        if booked is not None:
            table_move, table_cp = booked
            if _native is None and _compile_thread.is_alive():
                # the process is suspended between our moves, so an unfinished compile only
                # progresses while we "think": give it a slice of the clock the table just saved
                _compile_thread.join(min(BOOK_COMPILE_SLICE_S, time_left_ms / 20000.0))
            _record(table_move)
            print(f"book {table_move.uci()} cp {table_cp} clock {time_left_ms}", file=sys.stderr)
            return table_move.uci()
        tb = tablebase.best_move(board, time_left_ms)
        if tb is not None:
            _record(tb)
            print(f"tablebase {tb.uci()} clock {time_left_ms}", file=sys.stderr)
            return tb.uci()
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
        fixed = tablebase.correct(board, move)
        if fixed != move and fixed in board.legal_moves:
            print(f"tablebase veto {move.uci()} -> {fixed.uci()}", file=sys.stderr)
            move = fixed
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
        return random.choice(captures or moves).uci() if moves else "0000"
