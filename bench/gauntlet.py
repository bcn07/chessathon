"""Score one agent against another over many games, in parallel, from varied openings.

    uv run python -m bench.gauntlet --opponent baselines/minimax --games 40
    uv run python -m bench.gauntlet --opponent versions/v1 --games 200 --base-ms 5000

Each opening is played twice, once with each colour, so colour and opening bias cancel. The
result is a score, an Elo difference with a 95% interval, the termination mix, and a PGN of
every game. A game the agent lost by crashing, flagging or moving illegally fails the run.

Games run concurrently in threads; each game is two single-threaded processes, of which one
thinks at a time, so --workers games need about --workers free cores to keep the clocks honest.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.pgn

import harness.referee
from bench.openings import OPENINGS
from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def summarise(results: list[float]) -> str:
    """Score with an Elo difference and a 95% interval from the per-game variance."""
    n = len(results)
    if n == 0:
        return "no games"
    mean = sum(results) / n
    variance = sum((r - mean) ** 2 for r in results) / max(n - 1, 1)
    se = math.sqrt(variance / n)
    low, high = mean - 1.96 * se, mean + 1.96 * se
    elo = elo_from_score(mean)
    if low > 0 and high < 1:
        spread = (elo_from_score(high) - elo_from_score(low)) / 2
        interval = f"{elo:+.0f} ± {spread:.0f} Elo"
    else:
        interval = f"{elo:+.0f} Elo (interval unbounded at this sample size)"
    return f"score {mean:.1%}, {interval}"


def sprt_llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio of H1 (elo1) against H0 (elo0), fishtest's GSPRT approximation."""
    n = wins + draws + losses
    if n < 2:
        return 0.0
    # Half a pseudo-game in each bin keeps the variance finite when one side wins everything,
    # so a lopsided result reaches a verdict instead of stalling at zero.
    total = n + 1.5
    w, d = (wins + 0.5) / total, (draws + 0.5) / total
    mean = w + d / 2
    variance = (w + d / 4) - mean**2
    if variance <= 0:
        return 0.0

    def expected(elo: float) -> float:
        return 1 / (1 + 10 ** (-elo / 400))

    s0, s1 = expected(elo0), expected(elo1)
    return (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance / total)


def play_one(
    game: int, agent: Path, opponent: Path, base_ms: int, increment_ms: int, ply_cap: int
) -> tuple[int, str, bool, Outcome, str, str]:
    name, fen = OPENINGS[(game // 2) % len(OPENINGS)]
    plays_white = game % 2 == 0
    white, black = (agent, opponent) if plays_white else (opponent, agent)
    white_agent, black_agent = local(white), local(black)
    outcome = play_match(
        white_agent, black_agent, base_ms, increment_ms, ply_cap=ply_cap, start_fen=fen
    )
    ours = white_agent if plays_white else black_agent
    theirs = black_agent if plays_white else white_agent
    return game, name, plays_white, outcome, ours.stderr_tail, theirs.stderr_tail


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/minimax"))
    parser.add_argument("--games", type=int, default=40, help="rounded up to an even number")
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--pgn", type=Path, default=None, help="write every game here")
    parser.add_argument(
        "--start", type=int, default=0, help="first game index, so cluster jobs cover the openings"
    )
    parser.add_argument("--json", type=Path, default=None, help="machine-readable results")
    parser.add_argument(
        "--sprt",
        nargs=2,
        type=float,
        metavar=("ELO0", "ELO1"),
        help="stop early once the sequential test decides between these Elo hypotheses "
        "(alpha = beta = 0.05); --games becomes the cap",
    )
    args = parser.parse_args()

    agent = args.agent.resolve()
    opponent = args.opponent.resolve()
    games = args.games + args.games % 2
    os.environ["HARNESS_INCREMENT_MS"] = str(args.increment_ms)  # for UCI-wrapped opponents
    # CHESSATHON_NATIVE_ONLY makes every engine compile before its ready line, and a parallel
    # gauntlet starts 2 x workers compiles at once, so the platform's 60 s init budget is not
    # meaningful here (the platform never compiles at init: the hybrid startup answers in 0.1 s).
    # Locally, give init the time it needs; the games themselves are unaffected.
    init_budget = os.environ.get("HARNESS_INIT_BUDGET_S")
    if init_budget is None and os.environ.get("CHESSATHON_NATIVE_ONLY"):
        init_budget = "300"
    if init_budget is not None:
        harness.referee.INIT_BUDGET_S = float(init_budget)
    started = time.monotonic()

    results: list[float] = []
    by_colour: dict[bool, list[float]] = {True: [], False: []}
    terminations: dict[str, int] = {}
    plies: list[int] = []
    failures: list[str] = []
    pgns: list[str] = []

    tracebacks = {"agent": 0, "opponent": 0}
    tails_dir = args.pgn.with_suffix(".stderr") if args.pgn else None

    def _record(played: tuple[int, str, bool, Outcome, str, str]) -> None:
        game, opening, plays_white, outcome, our_tail, their_tail = played
        # An exception inside get_move is caught by the agent and answered with a random move,
        # so it never shows as a failure; count the tracebacks to make that visible.
        tracebacks["agent"] += our_tail.count("Traceback (most recent call last)")
        tracebacks["opponent"] += their_tail.count("Traceback (most recent call last)")
        if tails_dir is not None and (our_tail or their_tail):
            tails_dir.mkdir(exist_ok=True)
            (tails_dir / f"game_{game + 1}.txt").write_text(
                f"--- {agent.name}\n{our_tail}\n--- {opponent.name}\n{their_tail}\n"
            )
        if outcome.result in ("draw", "void"):
            points = 0.5
        else:
            points = 1.0 if (outcome.result == "white") == plays_white else 0.0
        results.append(points)
        by_colour[plays_white].append(points)
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        pgn_game = chess.pgn.read_game(_lines(outcome.pgn))
        if pgn_game is not None:
            pgn_game.headers["Event"] = "gauntlet"
            pgn_game.headers["Round"] = str(game + 1)
            pgn_game.headers["White"] = agent.name if plays_white else opponent.name
            pgn_game.headers["Black"] = opponent.name if plays_white else agent.name
            pgn_game.headers["Opening"] = opening
            plies.append(len(list(pgn_game.mainline_moves())))
            pgns.append(str(pgn_game))
        if outcome.termination in FAILED_TERMINATIONS and points == 0.0:
            failures.append(f"game {game + 1} ({opening}): {outcome.termination}")
        colour = "white" if plays_white else "black"
        print(
            f"[{done:3}/{games}] game {game + 1:3} {opening:26} as {colour:5} "
            f"{'win ' if points == 1 else 'draw' if points == 0.5 else 'loss'} "
            f"by {outcome.termination}",
            flush=True,
        )

    bound = math.log(19)  # alpha = beta = 0.05
    verdict = None
    schedule = list(range(args.start, args.start + games))
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # games go out in waves so a decided SPRT does not keep the pool busy for nothing
        for wave_start in range(0, len(schedule), args.workers * 2):
            if verdict:
                break
            wave = schedule[wave_start : wave_start + args.workers * 2]
            futures = [
                pool.submit(
                    play_one, game, agent, opponent, args.base_ms, args.increment_ms, args.ply_cap
                )
                for game in wave
            ]
            for future in as_completed(futures):
                done += 1
                _record(future.result())
            if args.sprt:
                wins = sum(1 for r in results if r == 1.0)
                draws = sum(1 for r in results if r == 0.5)
                llr = sprt_llr(wins, draws, len(results) - wins - draws, *args.sprt)
                print(f"    sprt llr {llr:+.2f} (bounds ±{bound:.2f})", flush=True)
                if llr >= bound:
                    verdict = f"H1 accepted: at least {args.sprt[1]:+g} Elo"
                elif llr <= -bound:
                    verdict = f"H0 accepted: no better than {args.sprt[0]:+g} Elo"
    games = len(results)
    if verdict:
        print(f"\n{verdict} after {games} games")

    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.5)
    losses = len(results) - wins - draws
    print(f"\n{agent.name} vs {opponent.name}: {games} games in {time.monotonic() - started:.0f}s")
    print(f"+{wins} ={draws} -{losses}   {summarise(results)}")
    white = by_colour[True]
    black = by_colour[False]
    print(
        f"as white {sum(white) / max(len(white), 1):.1%}, "
        f"as black {sum(black) / max(len(black), 1):.1%}"
    )
    if plies:
        print(f"average game length {sum(plies) / len(plies):.0f} plies")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    print(
        f"tracebacks (crash → fallback move): agent {tracebacks['agent']}, "
        f"opponent {tracebacks['opponent']}"
    )
    if args.pgn:
        args.pgn.write_text("\n\n".join(pgns) + "\n")
        print(f"pgn written to {args.pgn}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "agent": str(agent),
                    "opponent": str(opponent),
                    "base_ms": args.base_ms,
                    "increment_ms": args.increment_ms,
                    "results": results,
                    "white": white,
                    "black": black,
                    "terminations": terminations,
                    "plies": plies,
                    "failures": failures,
                    "tracebacks": tracebacks,
                }
            )
        )
    if failures:
        raise SystemExit("the agent failed to finish a game:\n  " + "\n  ".join(failures))


def _lines(text: str) -> io.StringIO:
    return io.StringIO(text)


if __name__ == "__main__":
    main()
