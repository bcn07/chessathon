"""Red-team probes for the finals build: legality, wall time, table and tablebase paths, bad input.

    CHESSATHON_AGENT_DIR=work/v13-redteam uv run pytest -q -p no:cacheprovider -s
tests/test_redteam.py

Every probe goes through get_move(fen, time_left_ms) in ONE process (the numba compile is paid
once). Each records input, stderr, wall time; at session end a table per probe class is printed
and, with REDTEAM_OUT=path.json, every probe is dumped. Skip-safe: table probes skip when the
agent ships no opening table, tablebase probes when it ships no Syzygy files.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chess
import pytest

import agent

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(agent.__file__).resolve().parent
HAS_BOOK = bool(getattr(agent, "_book", None))
SYZYGY = AGENT_DIR / "weights" / "syzygy"
HAS_TB = SYZYGY.is_dir() and any(SYZYGY.glob("*.rtbw"))
NATIVE = agent.wait_native() if hasattr(agent, "wait_native") else False

RESULTS: list[dict] = []


def probe(cls: str, fen: str, clock: int, reset: bool = False, note: str = "") -> dict:
    """One get_move call: legality, stderr, wall time. Never raises; the caller asserts."""
    if reset and hasattr(agent, "_reset"):
        agent._reset()
    err = io.StringIO()
    started = time.perf_counter()
    exc = None
    uci = None
    try:
        with contextlib.redirect_stderr(err):
            uci = agent.get_move(fen, clock)
    except Exception as e:
        exc = repr(e)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    board = chess.Board(fen)
    legal = False
    if uci is not None:
        try:
            legal = chess.Move.from_uci(uci) in board.legal_moves
        except ValueError:
            legal = False
    stderr = err.getvalue()
    rec = {
        "cls": cls,
        "fen": fen,
        "clock": clock,
        "move": uci,
        "legal": legal,
        "ms": round(elapsed_ms, 1),
        "stderr": stderr.strip()[-600:],
        "traceback": "Traceback" in stderr,
        "exc": exc,
        "note": note,
    }
    RESULTS.append(rec)
    return rec


def ok(rec: dict) -> bool:
    return rec["legal"] and not rec["traceback"] and rec["exc"] is None


def fail_msg(rec: dict) -> str:
    return f"{rec['fen']} clock={rec['clock']} -> {rec['move']} {rec['ms']}ms exc={rec['exc']}\n{rec['stderr']}"  # noqa: E501


# A probe may run to the engine's own hard budget for that clock and move number (15 s on the
# first move of a 120 s game, up to 30 s later) plus slack for the Python around the search; a
# mate score is deliberately deepened until the depth covers the mate distance, which alone can
# pass 8 s. A flat 8 s bound sat on the soft target at fullmove 21 and failed under suite load.
LIMIT_SLACK_MS = 1500


def limit_ms(fen: str, clock: int) -> float:
    engine = agent._engine()
    if engine is agent.pyengine:
        _, hard = engine.budget(clock)
    else:
        _, hard = engine.budget(clock, chess.Board(fen).fullmove_number)
    return float(hard) + LIMIT_SLACK_MS


def slow(rec: dict) -> bool:
    return bool(rec["ms"] > limit_ms(rec["fen"], rec["clock"]))


@pytest.fixture(scope="session", autouse=True)
def _summary():
    yield
    classes: dict[str, list[dict]] = {}
    for rec in RESULTS:
        classes.setdefault(rec["cls"], []).append(rec)
    lines = ["", f"{'probe class':34} {'count':>5} {'fail':>4} {'worst ms':>9}"]
    for cls, recs in classes.items():
        fails = sum(1 for r in recs if not ok(r) or r.get("failed"))
        worst = max(r["ms"] for r in recs)
        lines.append(f"{cls:34} {len(recs):5d} {fails:4d} {worst:9.0f}")
    print("\n".join(lines))
    out = os.environ.get("REDTEAM_OUT")
    if out:
        Path(out).write_text(json.dumps(RESULTS, indent=1))


def _load_roots() -> list[dict]:
    roots = []
    for name in ("platform_book.json", "book_roots.json"):
        path = ROOT / "bench" / name
        if path.is_file():
            for entry in json.loads(path.read_text()):
                roots.append({"fen": entry["fen"], "src": name})
    return roots


ROOTS = _load_roots()


# ------------------------------------------------------------------------------------------
# 1. every start position, one after another in one process (new game each time)
# ------------------------------------------------------------------------------------------


@pytest.mark.skipif(not ROOTS, reason="no bench/platform_book.json / book_roots.json")
def test_start_positions_full_clock() -> None:
    bad = []
    for root in ROOTS:
        rec = probe(f"start positions ({root['src']})", root["fen"], 120_000)
        if not ok(rec) or slow(rec):
            rec["failed"] = True
            bad.append(fail_msg(rec))
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------
# 2. call patterns the platform never produces: same fen twice, then normal, then a new game
# ------------------------------------------------------------------------------------------


@pytest.mark.skipif(not ROOTS, reason="no start positions")
def test_same_fen_twice_then_two_plies_then_new_game() -> None:
    rng = random.Random(7)
    sample = rng.sample(ROOTS, 10)
    bad = []
    for i, root in enumerate(sample):
        fen = root["fen"]
        first = probe("same fen twice", fen, 120_000, reset=True)
        second = probe("same fen twice", fen, 120_000)
        for rec in (first, second):
            if not ok(rec):
                rec["failed"] = True
                bad.append(fail_msg(rec))
        if not ok(second):
            continue
        board = chess.Board(fen)
        board.push_uci(second["move"])
        if board.is_game_over():
            continue
        # opponent reply: a table move when the table knows one, else random
        reply = None
        entry = agent._book.get(agent._book_key(board)) if HAS_BOOK else None
        if entry:
            try:
                candidate = chess.Move.from_uci(str(entry[0]))
                reply = candidate if candidate in board.legal_moves else None
            except ValueError:
                reply = None
        board.push(reply or rng.choice(list(board.legal_moves)))
        ahead = probe("two plies ahead", board.fen(), 119_500)
        if not ok(ahead) or slow(ahead):
            ahead["failed"] = True
            bad.append(fail_msg(ahead))
        elif not getattr(agent, "_history", [True]):
            ahead["failed"] = True
            bad.append("history lost on a normal two-ply advance: " + fail_msg(ahead))
        # a completely different game in the same process
        other = sample[(i + 1) % len(sample)]["fen"]
        rec = probe("new game same process", other, 120_000)
        if not ok(rec) or slow(rec):
            rec["failed"] = True
            bad.append(fail_msg(rec))
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------
# 3. the move-20 boundary of the opening table
# ------------------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_BOOK, reason="agent ships no opening table")
def test_table_answers_up_to_fullmove_20_only() -> None:
    rng = random.Random(3)
    keys = rng.sample(sorted(agent._book), 4)
    bad = []
    for key in keys:
        for fullmove in (19, 20, 21):
            fen = f"{key} 0 {fullmove}"
            if chess.Board(fen).status() != chess.STATUS_VALID:
                continue
            rec = probe(f"table boundary fullmove {fullmove}", fen, 120_000, reset=True)
            booked = rec["stderr"].startswith("book ")
            want = fullmove <= 20
            if not ok(rec) or booked != want or slow(rec):
                rec["failed"] = True
                bad.append(f"fullmove {fullmove}: booked={booked} want={want}\n" + fail_msg(rec))
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------
# 4. endgames at every clock, tablebase paths and their gaps
# ------------------------------------------------------------------------------------------

ENDGAMES = {
    "KQvK": "8/8/8/8/8/2k5/8/KQ6 w - - 0 1",
    "KRvK": "8/8/8/3k4/8/8/8/R3K3 w - - 0 1",
    "KBNvK": "8/8/8/8/8/2k5/8/KBN5 w - - 0 1",
    "KPvK": "8/8/8/8/8/3K4/3P4/6k1 w - - 0 1",
    "KPPvKP": "8/8/8/3k4/5p2/2K5/1P1P4/8 w - - 0 1",
    "KRPvKR (WDL only)": "1r6/8/8/8/8/2k5/2P5/1KR5 w - - 0 1",
}
CLOCKS = (120_000, 5000, 1000, 300, 250, 100, 1)


@pytest.mark.skipif(not HAS_TB, reason="agent ships no Syzygy files")
def test_endgames_every_clock() -> None:
    bad = []
    worst = {}
    for name, fen in ENDGAMES.items():
        for clock in CLOCKS:
            rec = probe(f"endgame {name}", fen, clock, reset=True)
            worst[name] = max(worst.get(name, 0), rec["ms"])
            over = clock >= 250 and rec["ms"] > clock * 0.5
            if not ok(rec) or over:
                rec["failed"] = True
                bad.append(f"{name} clock {clock}: " + fail_msg(rec))
    print("endgame worst ms per ending:", worst)
    assert not bad, "\n".join(bad)


@pytest.mark.skipif(not HAS_TB, reason="agent ships no Syzygy files")
def test_krpvkr_pawn_push_vs_piece_moves() -> None:
    import chess.syzygy

    tables = chess.syzygy.open_tablebase(str(SYZYGY))
    # Lucena-like: the winning plan needs the pawn; every move is WDL-classified only
    fens = [
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "8/8/8/8/3k4/8/2KP4/r6R w - - 0 1",
        "8/1k6/8/8/8/1K6/1PR5/7r w - - 0 1",
    ]
    bad = []
    for fen in fens:
        board = chess.Board(fen)
        rec = probe("KRPvKR win kept", fen, 20_000, reset=True)
        if not ok(rec):
            rec["failed"] = True
            bad.append(fail_msg(rec))
            continue
        before = tables.probe_wdl(board)
        child = board.copy()
        child.push_uci(rec["move"])
        after = 0 if child.is_stalemate() else -tables.probe_wdl(child)
        rec["note"] = f"wdl before {before} after {after}"
        if before == 2 and after < 2:
            rec["failed"] = True
            bad.append("threw away a tablebase win: " + fail_msg(rec))
    assert not bad, "\n".join(bad)


NO_PROBE = {
    "6-man KRPPvKR": "8/8/8/3k4/8/2K5/1P1P4/R6r w - - 0 1",
    "6-man KQRvKRB": "8/8/3k4/8/8/2K5/2Q5/R3r2b w - - 0 1",
    "KRvK with castling right": "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
    "KRvK opp castling right": "r3k3/8/8/8/8/8/8/4K3 w q - 0 1",
    "KRvK black castles": "r3k3/8/8/8/8/8/8/4K3 b q - 0 1",
    "KvK": "8/8/3k4/8/8/3K4/8/8 w - - 0 1",
    "KNvK": "8/8/3k4/8/8/3K4/8/6N1 w - - 0 1",
    "KBvKB": "8/8/3k4/8/8/3K4/8/b5B1 w - - 0 1",
    "KPvK halfmove 99": "8/8/8/8/8/3K4/3P4/6k1 w - - 99 60",
}


def test_no_probe_material_and_castling_rights() -> None:
    bad = []
    for name, fen in NO_PROBE.items():
        for clock in (120_000, 300, 1):
            rec = probe(f"no-probe {name}", fen, clock, reset=True)
            if not ok(rec) or slow(rec):
                rec["failed"] = True
                bad.append(f"{name}: " + fail_msg(rec))
    assert not bad, "\n".join(bad)


STALEMATE_TRAPS = [
    "7k/8/8/8/8/8/5Q2/6K1 w - - 0 1",  # Qf7?? stalemates
    "k7/8/1K6/8/8/8/8/6Q1 w - - 0 1",
    "8/8/8/8/8/1k6/1P6/1K6 b - - 0 1",  # black to move, drawn; must not crash
    "7k/5K2/8/8/8/8/8/6R1 w - - 0 1",  # Rg7?? stalemates; Rg8# wins... check
    "8/8/8/8/8/5k2/6p1/6K1 b - - 0 1",  # KPvK black wins? actually g1 blocks; drawn
    "8/8/8/8/8/6k1/5p2/6K1 b - - 0 1",  # ...Kf3? stalemate trap for black
]


@pytest.mark.skipif(not HAS_TB, reason="agent ships no Syzygy files")
def test_winning_side_never_stalemates() -> None:
    import chess.syzygy

    tables = chess.syzygy.open_tablebase(str(SYZYGY))
    bad = []
    for fen in STALEMATE_TRAPS:
        board = chess.Board(fen)
        try:
            before = tables.probe_wdl(board)
        except Exception:
            before = None
        for clock in (120_000, 200):
            rec = probe("stalemate traps", fen, clock, reset=True)
            if not ok(rec):
                rec["failed"] = True
                bad.append(fail_msg(rec))
                continue
            child = board.copy()
            child.push_uci(rec["move"])
            rec["note"] = f"wdl before {before}"
            if before == 2 and child.is_stalemate():
                rec["failed"] = True
                bad.append("stalemated from a won position: " + fail_msg(rec))
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------
# 5. clock edge cases in a middlegame at move 30
# ------------------------------------------------------------------------------------------

MIDDLEGAME_30 = "r2q1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQR1K1 w - - 4 30"


def test_clock_edge_cases_move_30() -> None:
    bad = []
    for clock in (2000, 700, 300, 120, 50, 0, -100):
        rec = probe("clock edge move 30", MIDDLEGAME_30, clock, reset=True)
        over = clock > 0 and rec["ms"] >= clock
        if not ok(rec) or over:
            rec["failed"] = True
            bad.append(f"clock {clock}: " + fail_msg(rec))
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------
# 6. corrupt opening table: the import-time loader and the per-move lookup
# ------------------------------------------------------------------------------------------

CORRUPT_FEN = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9"


def _corrupt_variants(tmp: Path) -> dict[str, Path]:
    key = " ".join(CORRUPT_FEN.split()[:4])
    variants = {
        "missing": tmp / "does-not-exist.json",
        "invalid json": tmp / "invalid.json",
        "json list": tmp / "list.json",
        "non-list value": tmp / "nonlist.json",
        "illegal move": tmp / "illegal.json",
        "wrong side move": tmp / "wrongside.json",
        "null value": tmp / "null.json",
        "empty list value": tmp / "emptylist.json",
        "nested list move": tmp / "nested.json",
        "book wrapper list": tmp / "wrapper.json",
    }
    variants["invalid json"].write_text('{"a": [1, 2')
    variants["json list"].write_text('[["e2e4", 0, 1, 0]]')
    variants["non-list value"].write_text(json.dumps({key: "e2e4", "x": 5, "y": {"a": 1}}))
    variants["illegal move"].write_text(json.dumps({key: ["e1g1", 0, 1, 0]}))
    variants["wrong side move"].write_text(json.dumps({key: ["e7e5", 0, 1, 0]}))
    variants["null value"].write_text(json.dumps({key: None}))
    variants["empty list value"].write_text(json.dumps({key: []}))
    variants["nested list move"].write_text(json.dumps({key: [["d4d5"], 0]}))
    variants["book wrapper list"].write_text(json.dumps({"book": [1, 2]}))
    return variants


@pytest.mark.skipif(not hasattr(agent, "_load_book"), reason="agent has no opening table loader")
def test_corrupt_table_in_process() -> None:
    saved_path, saved_book = agent.BOOK_PATH, agent._book
    bad = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for name, path in _corrupt_variants(Path(tmp)).items():
                agent.BOOK_PATH = str(path)
                err = io.StringIO()
                try:
                    with contextlib.redirect_stderr(err):
                        agent._load_book()
                except Exception as e:
                    bad.append(f"{name}: _load_book raised {e!r}")
                    continue
                rec = probe(f"corrupt table: {name}", CORRUPT_FEN, 6000, reset=True, note=err.getvalue().strip())  # noqa: E501
                searched = "nativesearch" in rec["stderr"] or "pyengine" in rec["stderr"]
                if not ok(rec) or not searched:
                    rec["failed"] = True
                    bad.append(f"{name}: not answered by the search\n" + fail_msg(rec))
    finally:
        agent.BOOK_PATH, agent._book = saved_path, saved_book
    assert not bad, "\n".join(bad)


@pytest.mark.skipif(not hasattr(agent, "_load_book"), reason="agent has no opening table loader")
def test_corrupt_table_fresh_import() -> None:
    """A fresh process whose weights/book.json is invalid JSON: import must succeed, move searched."""  # noqa: E501
    with tempfile.TemporaryDirectory() as tmp:
        shadow = Path(tmp) / "agent"
        shadow.mkdir()
        for item in AGENT_DIR.iterdir():
            if item.name == "weights":
                continue
            if item.suffix == ".py":
                (shadow / item.name).symlink_to(item)
        weights = shadow / "weights"
        weights.mkdir()
        for item in (AGENT_DIR / "weights").iterdir():
            if item.name != "book.json":
                (weights / item.name).symlink_to(item)
        (weights / "book.json").write_text("{not json")
        code = (
            "import sys, time; sys.path.insert(0, sys.argv[1]); t=time.perf_counter()\n"
            "import agent; agent.wait_native()\n"
            f"print(agent.get_move({CORRUPT_FEN!r}, 6000))\n"
        )
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", code, str(shadow)],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "CHESSATHON_INIT_COMPILE_WAIT": "80"},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
    move = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
    legal = move is not None and chess.Move.from_uci(move) in chess.Board(CORRUPT_FEN).legal_moves
    rec = {
        "cls": "corrupt table: fresh import", "fen": CORRUPT_FEN, "clock": 6000, "move": move,
        "legal": legal, "ms": round(elapsed_ms, 1), "stderr": proc.stderr[-800:],
        "traceback": "Traceback" in proc.stderr, "exc": None if proc.returncode == 0 else f"exit {proc.returncode}",  # noqa: E501
        "note": "fresh process, invalid book.json",
    }
    RESULTS.append(rec)
    searched = "nativesearch" in proc.stderr or "pyengine" in proc.stderr
    assert ok(rec) and searched and "opening table unavailable" in proc.stderr, fail_msg(rec)


# ------------------------------------------------------------------------------------------
# 7. fuzz: random legal positions from playouts
# ------------------------------------------------------------------------------------------


def _playout(rng: random.Random, start: str, plies: int) -> str | None:
    board = chess.Board(start)
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    if board.is_game_over():
        return None
    return board.fen()


def test_fuzz_random_positions() -> None:
    rng = random.Random(2026)
    starts = [chess.STARTING_FEN] * 100 + [r["fen"] for r in rng.sample(ROOTS, min(100, len(ROOTS)))]  # noqa: E501
    if len(starts) < 200:
        starts += [chess.STARTING_FEN] * (200 - len(starts))
    bad = []
    count = 0
    while count < 200:
        fen = _playout(rng, starts[count % len(starts)], rng.randint(5, 80))
        if fen is None:
            continue
        count += 1
        # every fifth: hand the fen with an X-FEN style ep square when a pawn just moved two
        board = chess.Board(fen)
        if count % 5 == 0:
            fen = board.fen(en_passant="fen")
        rec = probe("fuzz playouts 3000 ms", fen, 3000)
        if not ok(rec) or rec["ms"] > 3500:
            rec["failed"] = True
            bad.append(fail_msg(rec))
    assert not bad, "\n".join(bad)
