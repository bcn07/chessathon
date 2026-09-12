"""Syzygy tablebases, probed at the root only: perfect play once few pieces remain.

``weights/syzygy/`` holds every 3- and 4-man ending (WDL and DTZ) and the two commonest 5-man
ones, KRPvKR (WDL only) and KPPvKP (WDL and DTZ). Nothing is read at import: the directory is
listed on the first probe and python-chess maps each file the first time it is touched.

``best_move`` returns a move when the tables settle the choice: a proven win, played along the
shortest distance to the next capture or pawn move, so the 50-move counter never catches up and
no position can repeat. Otherwise it returns None and the search plays; ``correct`` then swaps
the search's move for a tablebase move only when it would throw away a win or a draw. Without
DTZ the tables know the class of every root move but not how to make progress, and among moves
of one class the search chooses better than a table would.
"""

from __future__ import annotations

import os
import time

import chess
import chess.syzygy

SYZYGY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "syzygy")

# Move classes, from our side. LIKELY_WIN is a WDL win whose distance to zero is unknown (no DTZ
# table): never worse than a draw, not proven within the 50-move rule from this clock.
WIN, LIKELY_WIN, DRAW, LOSS = 2, 1, 0, -1

# A win counts only if the next capture or pawn move comes before the halfmove clock reaches
# 100; python-chess's DTZ can be one ply short (rounding), hence one ply of slack.
CLOCK_LIMIT = 99

# Probing must never flag us: below MIN_CLOCK_MS the search plays, and otherwise the probe gets
# BUDGET_SHARE of the clock, after which the distance-to-zero ranking is abandoned (the WDL
# classes stay, so the veto still holds). The platform's 0.5 s increment covers a normal probe.
MIN_CLOCK_MS = 300
BUDGET_SHARE = 0.1

Rank = tuple[int, int]
Classes = dict[chess.Move, tuple[int, Rank]]

_tables: chess.syzygy.Tablebase | None = None
_max_men = 0
_opened = False
_last: tuple[str, Classes] | None = None


def _open() -> chess.syzygy.Tablebase | None:
    global _tables, _max_men, _opened
    if not _opened:
        _opened = True
        try:
            names = [name for name in os.listdir(SYZYGY_DIR) if name.endswith(".rtbw")]
            if names:
                _max_men = max(len(name[:-5].replace("v", "")) for name in names)
                _tables = chess.syzygy.open_tablebase(SYZYGY_DIR)
        except Exception:
            _tables = None
    return _tables


def max_men() -> int:
    _open()
    return _max_men


def classify(board: chess.Board, deadline: float | None = None) -> Classes | None:
    """Class and rank of every legal move, or None when the tables cannot settle this position.

    Ranks order moves within a class, lower first. For WIN: plies until the clock resets (a mate
    or a capture/pawn move now is 1), then the opponent's distance to zero after that. For DRAW:
    cursed wins before dead draws before blessed losses. Past the deadline (perf_counter) no
    more DTZ is probed and only mates and zeroing moves stay WIN.
    """
    tables = _open()
    if tables is None or board.castling_rights or chess.popcount(board.occupied) > _max_men:
        return None
    if chess.syzygy.calc_key(board) not in tables.wdl or board.status() != chess.STATUS_VALID:
        return None
    clock = board.halfmove_clock
    classes: Classes = {}
    ranked = True
    work = board.copy(stack=False)
    for move in board.legal_moves:
        work.push(move)
        try:
            if work.is_checkmate():
                classes[move] = (WIN, (0, 0))
                continue
            if work.is_stalemate() or work.is_insufficient_material():
                classes[move] = (DRAW, (0, 0))
                continue
            try:
                wdl = -tables.probe_wdl(work)
            except chess.syzygy.MissingTableError:
                if move.promotion:
                    continue  # unknown class: never chosen, never vetoed
                raise
            if wdl < 2:
                classes[move] = (LOSS, (0, 0)) if wdl == -2 else (DRAW, (-wdl, 0))
                continue
            if ranked and deadline is not None and time.perf_counter() > deadline:
                ranked = False
            distance = None
            if ranked:
                dtz = tables.get_dtz(work)  # None when a sub-table (a promotion) is missing
                distance = -dtz if dtz is not None and dtz < 0 else None
            if work.halfmove_clock == 0:
                classes[move] = (WIN, (1, distance if distance is not None else 1000))
            elif distance is not None and clock + 1 + distance <= CLOCK_LIMIT:
                classes[move] = (WIN, (1 + distance, 0))
            elif distance is None:
                classes[move] = (LIKELY_WIN, (0, 0))
            else:
                classes[move] = (DRAW, (-1, 0))  # too slow for the 50-move rule: a cursed win
        finally:
            work.pop()
    if not ranked:
        # a partial ranking could pick a slow win that walks into the 50-move rule or repeats
        for move, (cls, rank) in classes.items():
            if cls == WIN and rank[0] > 1:
                classes[move] = (LIKELY_WIN, (0, 0))
    return classes


def _cached(board: chess.Board) -> Classes | None:
    if _last is not None and _last[0] == board.fen():
        return _last[1]
    return None


def best_move(board: chess.Board, time_left_ms: int | None = None) -> chess.Move | None:
    """A proven winning move along the shortest distance to zero, else None (the search plays)."""
    global _last
    try:
        deadline = None
        if time_left_ms is not None:
            if time_left_ms < MIN_CLOCK_MS:
                return None
            deadline = time.perf_counter() + time_left_ms * BUDGET_SHARE / 1000.0
        classes = classify(board, deadline)
        if not classes:
            return None
        _last = (board.fen(), classes)
        wins = [move for move, (cls, _) in classes.items() if cls == WIN]
        if not wins:
            return None
        return min(wins, key=lambda move: classes[move][1])
    except Exception:
        return None


def correct(board: chess.Board, move: chess.Move) -> chess.Move:
    """The search's move, unless the tables show a better class exists; then one of that class.

    Uses what best_move found for this position; it probes nothing itself.
    """
    try:
        classes = _cached(board)
        if classes is None or move not in classes:
            return move
        best = max(cls for cls, _ in classes.values())
        if classes[move][0] >= best:
            return move
        candidates = [m for m, (cls, _) in classes.items() if cls == best]
        return min(candidates, key=lambda m: classes[m][1])
    except Exception:
        return move
