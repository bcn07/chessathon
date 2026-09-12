"""Syzygy probing at the root (tablebase.py next to agent.py): won endings are won, draws are held.

Skipped unless the agent directory under test ships weights/syzygy/*.rtbw.
"""

import importlib
import os
import random
import sys
import time
from pathlib import Path

import chess
import chess.syzygy
import pytest

ROOT = Path(__file__).resolve().parents[1]
# the agent directory under test (CHESSATHON_AGENT_DIR, else the repo root, else the build dir)
_env_dir = os.environ.get("CHESSATHON_AGENT_DIR")
_candidates = ([Path(_env_dir)] if _env_dir else []) + [ROOT / "engine", ROOT]
_found = [d.resolve() for d in _candidates if (d / "tablebase.py").is_file()]
TB_DIR = _found[0] if _found else ROOT / "work" / "v13-tb"
SYZYGY = TB_DIR / "weights" / "syzygy"

if not SYZYGY.is_dir() or not any(SYZYGY.glob("*.rtbw")):
    pytest.skip(f"no Syzygy files in {SYZYGY}", allow_module_level=True)

if str(TB_DIR) not in sys.path:
    sys.path.insert(0, str(TB_DIR))
tablebase = importlib.import_module("tablebase")
assert Path(tablebase.__file__).resolve().parent == TB_DIR.resolve()

# an independent handle for checking what best_move returns
TABLES = chess.syzygy.open_tablebase(str(SYZYGY))

WINS = {
    "KQvK": "8/8/8/8/8/2k5/8/KQ6 w - - 0 1",
    "KRvK": "8/8/8/3k4/8/8/8/R3K3 w - - 0 1",
    "KBNvK": "8/8/8/8/8/2k5/8/KBN5 w - - 0 1",
    "KPvK": "8/8/8/8/8/3K4/3P4/6k1 w - - 0 1",
    "KPPvKP": "8/8/8/3k4/5p2/2K5/1P1P4/8 w - - 0 1",
}
DRAWS = {
    "KRPvKR Philidor": "8/8/8/4k3/4P3/5K2/r7/4R3 b - - 0 1",
    "KBvKP": "8/8/8/8/8/5k2/4p3/4K1B1 w - - 0 1",
    "KPvK opposition": "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
}


def our_wdl_after(board: chess.Board, move: chess.Move) -> int:
    """WDL from the mover's side once move is played (the tables answer for the side to move)."""
    child = board.copy(stack=False)
    child.push(move)
    if child.is_checkmate():
        return 2
    if child.is_stalemate() or child.is_insufficient_material():
        return 0
    return -TABLES.probe_wdl(child)


@pytest.fixture(autouse=True)
def forget_last_position():
    tablebase._last = None
    yield


@pytest.mark.parametrize("fen", list(WINS.values()), ids=list(WINS))
def test_won_position_returns_a_winning_move(fen: str) -> None:
    board = chess.Board(fen)
    assert TABLES.probe_wdl(board) == 2, "test position must be a tablebase win"
    move = tablebase.best_move(board)
    assert move is not None
    assert move in board.legal_moves
    assert our_wdl_after(board, move) == 2


def test_kbnvk_mates_a_random_defender_within_dtz() -> None:
    board = chess.Board(WINS["KBNvK"])
    bound = TABLES.probe_dtz(board)
    assert 0 < bound <= 100
    rng = random.Random(1)
    plies = 0
    while not board.is_checkmate() and plies < 120:
        if board.turn == chess.WHITE:
            move = tablebase.best_move(board)
            assert move is not None, f"no tablebase move at {board.fen()}"
        else:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
        plies += 1
        assert not board.is_fifty_moves()
        assert not board.is_repetition(3)
    assert board.is_checkmate(), f"no mate after {plies} plies: {board.fen()}"
    # python-chess's DTZ may be one ply short, and a lazy defender can only shorten the line
    assert plies <= bound + 1, f"mated in {plies} plies, DTZ bound {bound}"


def test_lost_position_leaves_the_move_to_the_search() -> None:
    board = chess.Board("8/8/8/8/8/2k5/8/KQ6 b - - 0 1")
    assert TABLES.probe_wdl(board) == -2
    assert tablebase.best_move(board) is None
    # and correct() keeps whatever the search chose: all moves lose alike
    move = next(iter(board.legal_moves))
    assert tablebase.correct(board, move) == move


@pytest.mark.parametrize("fen", list(DRAWS.values()), ids=list(DRAWS))
def test_drawn_position_never_yields_a_losing_move(fen: str) -> None:
    board = chess.Board(fen)
    assert TABLES.probe_wdl(board) == 0, "test position must be a tablebase draw"
    move = tablebase.best_move(board)
    if move is not None:
        assert move in board.legal_moves
        assert our_wdl_after(board, move) >= 0
    # the veto: a losing move from the search is replaced by a drawing one
    losing = [m for m in board.legal_moves if our_wdl_after(board, m) == -2]
    for bad in losing:
        fixed = tablebase.correct(board, bad)
        assert fixed != bad
        assert our_wdl_after(board, fixed) >= 0
    # and a drawing move from the search is left alone
    good = next(m for m in board.legal_moves if our_wdl_after(board, m) >= 0)
    assert tablebase.correct(board, good) == good


def test_low_clock_bounds_the_probe() -> None:
    board = chess.Board("8/8/4k3/8/8/3Q4/6r1/K7 w - - 0 1")  # 22 winning moves to rank by DTZ
    tablebase.best_move(board)  # first touch maps the files
    tablebase._last = None
    assert tablebase.best_move(board, 100) is None  # below MIN_CLOCK_MS: nothing is probed
    started = time.perf_counter()
    move = tablebase.best_move(board, 400)  # a 40 ms budget: WDL classes only
    elapsed = time.perf_counter() - started
    assert elapsed < 0.1
    # whatever came back must still be a proven win, and the veto still knows the classes
    if move is not None:
        assert our_wdl_after(board, move) == 2
    losing = next(m for m in board.legal_moves if our_wdl_after(board, m) < 2)
    assert our_wdl_after(board, tablebase.correct(board, losing)) == 2


def test_unsupported_material_returns_none() -> None:
    assert tablebase.best_move(chess.Board("8/8/8/3k4/3p4/8/2r5/K1Q5 w - - 0 1")) is None  # KQvKRP
    assert tablebase.best_move(chess.Board("8/8/8/3k4/3p4/8/2r5/K1QN4 w - - 0 1")) is None  # 6 men
    assert tablebase.best_move(chess.Board()) is None


@pytest.mark.parametrize(
    "fen",
    [
        "8/8/4k3/8/8/3Q4/6r1/K7 w - - 0 1",  # KQvKR, 26 moves, most of them wins needing DTZ
        "8/7r/8/8/1k1P4/8/R7/3K4 w - - 0 1",  # KRPvKR, 20 moves, WDL only
        "8/8/8/3k4/5p2/2K5/1P1P4/8 w - - 0 1",  # KPPvKP with DTZ
    ],
)
def test_probe_latency_under_200ms(fen: str) -> None:
    board = chess.Board(fen)
    tablebase.best_move(board)  # first touch maps the files
    tablebase._last = None
    started = time.perf_counter()
    tablebase.best_move(board)
    elapsed = time.perf_counter() - started
    print(f"{board.legal_moves.count()} moves probed in {elapsed * 1000:.0f} ms")
    assert elapsed < 0.2
