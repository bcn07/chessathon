"""Does a change alter the engine's decisions, and when it does, are the new moves better?

The 2026-09-09 fortress work passed every evaluation check it was given -- drawn suite halved,
won suite unmoved, middlegame node-identical -- and then two cheap measurements settled it in
minutes: the safe version changed 0.5% of moves on real positions (nothing to gain), the widened
version changed 2.8% and Stockfish scored the changed moves 2 better, 4 worse, one of them a
5,000 cp blunder. 796 games could not see either. This is that test, made repeatable, so every
evaluation or search change is gated on *decisions* before anyone asks for games.

    uv run python -m bench.agreement --build                            # positions from real games
    uv run python -m bench.agreement --agent . --depth 9 --json root.json
    uv run python -m bench.agreement --agent engine --depth 9 --json x.json
    uv run python -m bench.agreement --diff root.json x.json --adjudicate --sf-depth 22

Positions come from real games played at the competition clock (`--build` reads the sampled
games under `codex/draw_fixes/` and the ladder PGNs under `reference/games/`; `--pgn` adds more)
and are searched *with their game history*, so repetition handling sees what it would see in a
game. `--persistent` keeps one searcher per game so the transposition table carries between
the sampled moves as well, which is the only way to observe a cached-bound defect. Each build
runs in its own process because the engine modules share names; the diff step reads the two
JSON files.

Adjudication hands Stockfish the position *after* each build's chosen move and compares the
two from the mover's point of view. A change is judged on the moves it changed, not on
average accuracy, because average accuracy has stopped predicting Elo here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = ROOT / "tools" / "engines" / "stockfish"
POSITIONS = ROOT / "bench" / "decisions.json"
LADDER_PGNS = ROOT / "reference" / "games"
MATE_SCORE = 100_000
DECISIVE = 1000  # a swing this large is a lost game either way; larger values are horizon noise
DEAD_DRAW = 20  # Stockfish inside this of zero after both moves: the choice did not matter
ENGINE_MODULES = ("agent", "nativesearch", "fastboard", "nnue_eval", "pyengine")

# Phase buckets by men on the board: the fortress classifier could only fire at <= 14, and a
# search change that moves decisions only in one phase should show up as such. The sample is
# stratified across them, because real-clock games at this level spend most of their plies in
# the endgame and an even sample would be four-fifths endgame.
PHASES: tuple[tuple[str, int, int], ...] = (
    ("endgame <=14", 0, 14),
    ("late 15-22", 15, 22),
    ("middlegame 23+", 23, 32),
)


# ------------------------------------------------------------------------------------------
# Positions: sampled from real games, replayed with history
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    game: int
    ply: int
    fen: str
    men: int
    halfmove: int


def games_from_pgn(paths: Iterable[Path]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                board = game.board()
                moves = [m.uci() for m in game.mainline_moves()]
                if moves:
                    games.append(
                        {
                            "source": str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT)
                            else str(path),
                            "start_fen": board.fen(),
                            "moves": moves,
                        }
                    )
    return games


def candidates(
    games: list[dict[str, Any]],
    min_ply: int,
    min_men: int,
    max_men: int,
    min_halfmove: int,
) -> list[Position]:
    """Every distinct position a build could be asked to decide, in game order."""
    seen: set[str] = set()
    out: list[Position] = []
    for index, game in enumerate(games):
        board = chess.Board(game["start_fen"])
        for ply, uci in enumerate(game["moves"]):
            if ply >= min_ply and not board.is_game_over(claim_draw=True):
                men = chess.popcount(board.occupied)
                fen = board.fen()
                wanted = min_men <= men <= max_men and board.halfmove_clock >= min_halfmove
                if wanted and fen not in seen:
                    seen.add(fen)
                    out.append(Position(index, ply, fen, men, board.halfmove_clock))
            board.push_uci(uci)
    return out


def allocate(n: int, sizes: list[int]) -> list[int]:
    """Equal shares of n across buckets; a bucket too small to take its share hands the rest on."""
    quota = [0] * len(sizes)
    remaining = n
    open_buckets = [i for i, size in enumerate(sizes) if size > 0]
    while remaining > 0 and open_buckets:
        share = max(1, remaining // len(open_buckets))
        for i in list(open_buckets):
            take = min(share, sizes[i] - quota[i], remaining)
            quota[i] += take
            remaining -= take
            if quota[i] == sizes[i]:
                open_buckets.remove(i)
            if remaining == 0:
                break
    return quota


def spaced(pool: list[Position], k: int) -> list[Position]:
    """k positions evenly spaced through a game-ordered list: coverage proportional to game
    length, every phase of every game represented, and deterministic."""
    if len(pool) <= k:
        return list(pool)
    return [pool[(i * len(pool)) // k] for i in range(k)]


def sample(pool: list[Position], n: int) -> list[Position]:
    """n positions, split as evenly as the pool allows across the phase buckets."""
    if len(pool) <= n:
        return list(pool)
    buckets = [[p for p in pool if lo <= p.men <= hi] for _, lo, hi in PHASES]
    chosen: list[Position] = []
    for bucket, k in zip(buckets, allocate(n, [len(b) for b in buckets]), strict=True):
        chosen += spaced(bucket, k)
    return sorted(chosen, key=lambda p: (p.game, p.ply))


def build(args: argparse.Namespace) -> None:
    games: list[dict[str, Any]] = []
    if LADDER_PGNS.is_dir():
        games += games_from_pgn(sorted(LADDER_PGNS.glob("*.pgn")))
    games += games_from_pgn(Path(p) for p in args.pgn)
    if not games:
        raise SystemExit("no games found; pass --pgn")
    pool = candidates(games, args.min_ply, args.min_men, args.max_men, args.min_halfmove)
    chosen = sample(pool, args.n)
    payload = {
        "games": games,
        "filters": {
            "min_ply": args.min_ply,
            "min_men": args.min_men,
            "max_men": args.max_men,
            "min_halfmove": args.min_halfmove,
        },
        "pool": len(pool),
        "positions": [asdict(p) for p in chosen],
    }
    args.positions.write_text(json.dumps(payload, indent=1) + "\n")
    by_phase = {name: sum(1 for p in chosen if lo <= p.men <= hi) for name, lo, hi in PHASES}
    print(
        f"{args.positions.relative_to(ROOT)}: {len(chosen)} positions from {len(games)} games "
        f"(pool {len(pool)}); by phase {by_phase}"
    )


# ------------------------------------------------------------------------------------------
# Running one build
# ------------------------------------------------------------------------------------------


def load_engine(directory: Path) -> tuple[Any, Any, float]:
    """Import a build's agent and its native module from one directory, compile included.

    Both `pytest` and `bench/tactics.py` pass on the Python fallback, so a run that does not
    assert `native_ready()` measures nothing. This one refuses to continue without it.
    """
    os.environ["CHESSATHON_NATIVE_ONLY"] = "1"
    os.environ.setdefault("CHESSATHON_PONDER", "0")
    resolved = directory.resolve()
    if not (resolved / "agent.py").is_file():
        raise SystemExit(f"{resolved}: no agent.py")
    for name in ENGINE_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(resolved))
    started = time.perf_counter()
    try:
        agent = importlib.import_module("agent")
        agent.wait_native()
        native = importlib.import_module("nativesearch")
    finally:
        sys.path.remove(str(resolved))
    compile_s = time.perf_counter() - started
    if not agent.native_ready():
        raise SystemExit(f"{resolved}: native engine did not load; nothing to measure")
    if Path(native.__file__).resolve().parent != resolved:
        raise SystemExit(f"imported {native.__file__}, not the copy in {resolved}")
    return agent, native, compile_s


def run(args: argparse.Namespace) -> None:
    data = json.loads(args.positions.read_text())
    games = data["games"]
    wanted: dict[int, set[int]] = {}
    for p in data["positions"][: args.limit] if args.limit else data["positions"]:
        wanted.setdefault(p["game"], set()).add(p["ply"])

    _agent, native, compile_s = load_engine(args.agent)
    print(f"{args.agent}: native ready, compile {compile_s:.1f}s", file=sys.stderr)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, plies in sorted(wanted.items()):
        game = games[index]
        board = chess.Board(game["start_fen"])
        keys: list[int] = []
        searcher = native.Searcher() if args.persistent else None
        for ply, uci in enumerate(game["moves"]):
            if ply in plies and not board.is_game_over(claim_draw=True):
                engine = searcher if searcher is not None else native.Searcher()
                t0 = time.perf_counter()
                found = engine.think(board, 10**9, 10**9, keys, max_depth=args.depth)
                if found.move not in board.legal_moves:
                    raise SystemExit(f"{args.agent} played illegal {found.move} in {board.fen()}")
                results.append(
                    {
                        "game": index,
                        "ply": ply,
                        "fen": board.fen(),
                        "men": chess.popcount(board.occupied),
                        "halfmove": board.halfmove_clock,
                        "move": found.move.uci(),
                        "score": int(found.score),
                        "depth": int(found.depth),
                        "nodes": int(found.nodes),
                        "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    }
                )
                if args.verbose:
                    r = results[-1]
                    print(
                        f"g{index:02} p{ply:03} {r['move']:6} {r['score']:+6} d{r['depth']:2} "
                        f"{r['nodes']:>9} {r['ms']:7.0f}ms",
                        file=sys.stderr,
                    )
            keys.append(native._native_repetition_key(board))
            board.push_uci(uci)
    elapsed = time.perf_counter() - started
    payload = {
        "agent": str(args.agent),
        "depth": args.depth,
        "mode": "persistent" if args.persistent else "fresh",
        "positions_file": str(args.positions),
        "compile_s": round(compile_s, 2),
        "seconds": round(elapsed, 1),
        "positions": results,
    }
    args.json.write_text(json.dumps(payload, indent=1) + "\n")
    nodes = sum(r["nodes"] for r in results)
    print(
        f"{args.agent}: {len(results)} positions at depth {args.depth} "
        f"({payload['mode']}) in {elapsed:.0f}s, {nodes} nodes -> {args.json}"
    )


# ------------------------------------------------------------------------------------------
# Comparing two runs
# ------------------------------------------------------------------------------------------


@dataclass
class Difference:
    game: int
    ply: int
    fen: str
    men: int
    halfmove: int
    move_a: str
    move_b: str
    score_a: int
    score_b: int
    sf_a: int | None = None  # Stockfish, mover's view, after move_a
    sf_b: int | None = None
    verdict: str | None = None  # better / worse / neutral for B

    @property
    def gap(self) -> int:
        return abs(self.score_a - self.score_b)

    @property
    def delta(self) -> int | None:
        """B's move against A's in Stockfish centipawns, clamped so that a mate score found
        behind one move and not the other (a horizon effect in a lost ending, as often as a real
        blunder) counts as decisive rather than swamping every other decision in the total."""
        if self.sf_a is None or self.sf_b is None:
            return None
        return max(-DECISIVE, min(DECISIVE, self.sf_b - self.sf_a))

    @property
    def dead_draw(self) -> bool:
        """Both moves lead to a position Stockfish calls level: the builds picked different moves
        from a set of equivalent ones, which is what most disagreements in shuffle endings are."""
        return (
            self.sf_a is not None
            and self.sf_b is not None
            and abs(self.sf_a) < DEAD_DRAW
            and abs(self.sf_b) < DEAD_DRAW
        )


def compare(a: dict[str, Any], b: dict[str, Any], threshold: int) -> dict[str, Any]:
    """Line the two runs up by position and count what changed."""
    by_key_b = {(p["game"], p["ply"]): p for p in b["positions"]}
    shared = 0
    score_moved = 0
    differences: list[Difference] = []
    phase_n = dict.fromkeys((name for name, _, _ in PHASES), 0)
    phase_diff = dict.fromkeys((name for name, _, _ in PHASES), 0)
    for pa in a["positions"]:
        pb = by_key_b.get((pa["game"], pa["ply"]))
        if pb is None or pb["fen"] != pa["fen"]:
            continue
        shared += 1
        phase = next(name for name, lo, hi in PHASES if lo <= pa["men"] <= hi)
        phase_n[phase] += 1
        if abs(pa["score"] - pb["score"]) >= threshold:
            score_moved += 1
        if pa["move"] != pb["move"]:
            phase_diff[phase] += 1
            differences.append(
                Difference(
                    pa["game"],
                    pa["ply"],
                    pa["fen"],
                    pa["men"],
                    pa["halfmove"],
                    pa["move"],
                    pb["move"],
                    pa["score"],
                    pb["score"],
                )
            )
    return {
        "shared": shared,
        "score_moved": score_moved,
        "differences": differences,
        "phase_n": phase_n,
        "phase_diff": phase_diff,
    }


def judge(delta: int, margin: int) -> str:
    """B's move against A's, from the mover's point of view, in Stockfish centipawns."""
    if delta >= margin:
        return "better"
    if delta <= -margin:
        return "worse"
    return "neutral"


def adjudicate(
    differences: list[Difference],
    evaluate: Callable[[chess.Board, chess.Color], int],
    margin: int,
) -> None:
    """Score the position after each build's move for the side that made it."""
    for d in differences:
        board = chess.Board(d.fen)
        mover = board.turn
        after_a = board.copy()
        after_a.push_uci(d.move_a)
        after_b = board.copy()
        after_b.push_uci(d.move_b)
        d.sf_a = evaluate(after_a, mover)
        d.sf_b = evaluate(after_b, mover)
        d.verdict = judge(d.delta or 0, margin)


Evaluator = Callable[[chess.Board, chess.Color], int]


def stockfish_evaluator(depth: int, hash_mb: int) -> tuple[Evaluator, Callable[[], None]]:
    if not STOCKFISH.exists():
        raise SystemExit(f"{STOCKFISH} missing (tools/engines/README.md)")
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    engine.configure({"Threads": 1, "Hash": hash_mb})

    def evaluate(board: chess.Board, pov: chess.Color) -> int:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0
            return MATE_SCORE if outcome.winner == pov else -MATE_SCORE
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        return int(info["score"].pov(pov).score(mate_score=MATE_SCORE) or 0)

    return evaluate, engine.quit


def diff(args: argparse.Namespace) -> None:
    a = json.loads(args.diff[0].read_text())
    b = json.loads(args.diff[1].read_text())
    name_a, name_b = a["agent"], b["agent"]
    if a["depth"] != b["depth"] or a["mode"] != b["mode"]:
        print(
            f"warning: runs differ in depth/mode ({a['depth']}/{a['mode']} vs "
            f"{b['depth']}/{b['mode']}); the comparison is not like for like",
            file=sys.stderr,
        )
    result = compare(a, b, args.threshold)
    differences: list[Difference] = result["differences"]
    shared = result["shared"]
    if not shared:
        raise SystemExit("no positions in common between the two runs")

    print(f"A = {name_a}\nB = {name_b}\ndepth {a['depth']}, {a['mode']} searcher")
    print(
        f"{shared} shared positions: {len(differences)} different moves "
        f"({100.0 * len(differences) / shared:.1f}%), score differs >= {args.threshold} cp "
        f"in {result['score_moved']} ({100.0 * result['score_moved'] / shared:.1f}%)"
    )
    for name, _, _ in PHASES:
        n = result["phase_n"][name]
        if n:
            k = result["phase_diff"][name]
            print(f"  {name:16} {k:3}/{n:<4} different ({100.0 * k / n:.1f}%)")
    if differences:
        gaps = [d.gap for d in differences]
        print(
            f"engine score gap where the move changed: median {statistics.median(gaps):.0f} cp, "
            f"max {max(gaps)} cp"
        )

    if args.adjudicate and differences:
        evaluate, quit_engine = stockfish_evaluator(args.sf_depth, args.sf_hash)
        try:
            started = time.perf_counter()
            adjudicate(differences, evaluate, args.margin)
            seconds = time.perf_counter() - started
        finally:
            quit_engine()
        tally = {"better": 0, "worse": 0, "neutral": 0}
        for d in differences:
            tally[d.verdict or "neutral"] += 1
        deltas = [d.delta or 0 for d in differences]
        dead = sum(1 for d in differences if d.dead_draw)
        print(
            f"\nStockfish depth {args.sf_depth} on the {len(differences)} changed decisions "
            f"({seconds:.0f}s): B is better {tally['better']}, worse {tally['worse']}, "
            f"neutral {tally['neutral']} (margin {args.margin} cp); "
            f"net {sum(deltas):+d} cp with swings clamped to +/-{DECISIVE}, "
            f"worst {min(deltas):+d}, best {max(deltas):+d}"
        )
        print(
            f"{dead} of the {len(differences)} were dead draws either way (both moves inside "
            f"+/-{DEAD_DRAW} cp), so {len(differences) - dead} decisions actually differed: "
            f"{100.0 * (len(differences) - dead) / shared:.1f}% of {shared}"
        )
        blunders = [d for d in differences if d.delta is not None and abs(d.delta) >= args.blunder]
        if blunders:
            print(f"{len(blunders)} decision(s) swing >= {args.blunder} cp; listed with '!'")

    if args.out:
        # written before the table so that a closed pipe (`| head`) cannot lose the evidence
        args.out.write_text(
            json.dumps(
                {
                    "a": name_a,
                    "b": name_b,
                    "depth": a["depth"],
                    "mode": a["mode"],
                    "shared": shared,
                    "score_moved": result["score_moved"],
                    "threshold": args.threshold,
                    "sf_depth": args.sf_depth if args.adjudicate else None,
                    "margin": args.margin,
                    "differences": [
                        asdict(d) | {"delta": d.delta, "dead_draw": d.dead_draw}
                        for d in differences
                    ],
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwritten {args.out}")

    print()
    header = f"{'':1} {'g/ply':7} {'men':>3} {'hm':>3} {'A move':7} {'B move':7} {'A':>6} {'B':>6}"
    if args.adjudicate:
        header += f" {'sf A':>7} {'sf B':>7} {'delta':>6} verdict"
    header += "  fen"
    print(header)
    for d in sorted(differences, key=lambda d: -(abs(d.delta) if d.delta is not None else d.gap)):
        flag = "!" if d.delta is not None and abs(d.delta) >= args.blunder else " "
        line = (
            f"{flag} {d.game:02}/{d.ply:<4} {d.men:>3} {d.halfmove:>3} {d.move_a:7} {d.move_b:7} "
            f"{d.score_a:+6} {d.score_b:+6}"
        )
        if args.adjudicate and d.delta is not None:
            line += f" {d.sf_a:+7} {d.sf_b:+7} {d.delta:+6} {d.verdict:7}"
        print(f"{line}  {d.fen}")


# ------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--positions", type=Path, default=POSITIONS, help="positions file")
    mode = parser.add_argument_group("build a positions file")
    mode.add_argument("--build", action="store_true", help="sample positions from real games")
    mode.add_argument("--pgn", nargs="*", default=[], help="extra PGN files to sample from")
    mode.add_argument("--n", type=int, default=400, help="positions to keep")
    mode.add_argument("--min-ply", type=int, default=16, help="skip the opening")
    mode.add_argument("--min-men", type=int, default=0)
    mode.add_argument("--max-men", type=int, default=32)
    mode.add_argument("--min-halfmove", type=int, default=0, help="e.g. 60 for near-limit only")
    mode = parser.add_argument_group("run one build")
    mode.add_argument("--agent", type=Path, help="directory holding agent.py")
    mode.add_argument("--depth", type=int, default=9, help="fixed depth (reproducible)")
    mode.add_argument("--json", type=Path, help="where to write this build's decisions")
    mode.add_argument(
        "--persistent", action="store_true", help="one searcher per game (TT carries)"
    )
    mode.add_argument("--limit", type=int, default=0, help="first N positions only (smoke test)")
    mode.add_argument("--verbose", action="store_true")
    mode = parser.add_argument_group("compare two runs")
    mode.add_argument("--diff", nargs=2, type=Path, metavar=("A.json", "B.json"))
    mode.add_argument("--threshold", type=int, default=25, help="cp for 'score differs'")
    mode.add_argument("--adjudicate", action="store_true", help="ask Stockfish about changed moves")
    mode.add_argument("--sf-depth", type=int, default=22)
    mode.add_argument("--sf-hash", type=int, default=256, help="Stockfish hash in MB")
    mode.add_argument("--margin", type=int, default=30, help="cp below which a change is neutral")
    mode.add_argument("--blunder", type=int, default=300, help="cp swing flagged in the table")
    mode.add_argument("--out", type=Path, help="write the comparison as JSON")
    args = parser.parse_args()

    if args.build:
        build(args)
    elif args.diff:
        diff(args)
    elif args.agent is not None:
        if args.json is None:
            raise SystemExit("--agent needs --json OUT")
        if not args.positions.exists():
            raise SystemExit(f"{args.positions} missing: run --build first")
        run(args)
    else:
        parser.error("one of --build, --agent, --diff")


if __name__ == "__main__":
    main()
