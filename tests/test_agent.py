"""Things that lose games for free: illegal moves, crashes on edge cases, flagging, protocol.

Everything here goes through get_move(fen, time_left_ms), so it holds for any agent.py.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import chess
import pytest

import agent

ROOT = Path(__file__).resolve().parents[1]


def legal(fen: str, uci: str) -> bool:
    return chess.Move.from_uci(uci) in chess.Board(fen).legal_moves


@pytest.fixture(autouse=True)
def fresh_game():
    if hasattr(agent, "_reset"):
        agent._reset()
    yield


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant available
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # castling
        "8/8/8/8/8/6k1/4q3/7K w - - 0 1",  # in check, few moves
        "7k/8/8/8/8/8/8/K6R w - - 0 1",  # bare kings and a rook
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # perft position 3, pins and ep tricks
    ],
)
def test_returns_legal_move(fen: str) -> None:
    assert legal(fen, agent.get_move(fen, 2000))


def test_promotion_carries_piece_suffix() -> None:
    assert agent.get_move("8/1P4k1/8/8/8/8/6K1/8 w - - 0 1", 2000) == "b7b8q"


def test_finds_mate_in_one() -> None:
    assert agent.get_move("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 2000) == "a1a8"


def test_single_legal_move_is_instant() -> None:
    board = chess.Board("7k/8/5Q1K/8/8/8/8/8 b - - 0 1")
    moves = list(board.legal_moves)
    assert len(moves) == 1
    started = time.perf_counter()
    assert agent.get_move(board.fen(), 60_000) == moves[0].uci()
    assert time.perf_counter() - started < 0.25


@pytest.mark.parametrize("clock", [15, 60, 250])
def test_low_clock_still_answers_in_time(clock: int) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    started = time.perf_counter()
    move = agent.get_move(fen, clock)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert legal(fen, move)
    assert elapsed_ms < clock * 0.8 + 20, f"spent {elapsed_ms:.0f}ms of a {clock}ms clock"


def test_spends_a_sane_share_of_a_full_clock() -> None:
    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9"
    started = time.perf_counter()
    agent.get_move(fen, 120_000)
    elapsed = time.perf_counter() - started
    assert 0.2 < elapsed < 12.0


def test_converts_a_won_ending_without_repeating() -> None:
    """Queen and king against a bare king. The referee claims threefold repetition for the
    defender, so the agent has to make progress from the FEN alone, move after move."""
    board = chess.Board("8/8/8/4k3/8/8/8/3QK3 w - - 0 1")
    for _ in range(60):
        if board.is_game_over(claim_draw=True):
            break
        move = agent.get_move(board.fen(), 20_000)
        assert legal(board.fen(), move)
        board.push_uci(move)
        if board.is_game_over(claim_draw=True):
            break
        # the defender runs to the corner and shuffles, the most repetition-prone defence
        replies = list(board.legal_moves)
        replies.sort(key=lambda m: chess.square_distance(m.to_square, chess.H8))
        board.push(replies[0])
    outcome = board.outcome(claim_draw=True)
    assert outcome is not None and outcome.winner == chess.WHITE, (
        f"did not win: {outcome} after {board.fullmove_number} moves\n{board}"
    )


def test_runner_protocol_end_to_end() -> None:
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "harness" / "runner.py"), str(ROOT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        assert json.loads(process.stdout.readline()) == {"ready": True}
        request = {"fen": chess.STARTING_FEN, "time_left_ms": 1000}
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        assert len(line.encode()) < 4096
        assert legal(chess.STARTING_FEN, json.loads(line)["move"])
    finally:
        process.kill()
        process.wait()


def test_ponders_between_moves_and_stops_promptly() -> None:
    if not getattr(agent, "PONDER", False):
        pytest.skip("agent does not ponder")
    if hasattr(agent, "wait_native"):
        agent.wait_native(120)  # pondering is deliberately off while the native engine compiles
    board = chess.Board()
    first = agent.get_move(board.fen(), 20_000)
    board.push_uci(first)
    time.sleep(0.3)
    thread = agent._ponder_thread
    assert thread is not None and thread.is_alive(), "no ponder search running"
    board.push(next(iter(board.legal_moves)))
    started = time.perf_counter()
    agent._stop_pondering()
    assert (time.perf_counter() - started) < 0.1, "ponder search did not stop within 100 ms"
    second = agent.get_move(board.fen(), 20_000)
    assert legal(board.fen(), second)
    agent._stop_pondering()


def test_ponder_stop_is_never_lost() -> None:
    """A stop that arrives before the ponder search has begun must still end it promptly
    (a fast opponent reply used to hang the engine here and lose on time)."""
    if not getattr(agent, "PONDER", False):
        pytest.skip("agent does not ponder")
    if hasattr(agent, "wait_native"):
        agent.wait_native(120)
    agent.get_move(chess.STARTING_FEN, 20_000)
    for _ in range(200):
        agent._start_pondering(agent._engine())
        started = time.perf_counter()
        agent._stop_pondering()
        assert time.perf_counter() - started < 0.5, "ponder stop hung"
