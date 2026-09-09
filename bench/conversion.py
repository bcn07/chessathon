"""Conversion suite: does the engine finish positions it has already won?

Self-play SPRTs cannot see a conversion defect, because both sides carry it and the games
cancel: the mate-loop fix measured -10 +/- 15 and +15 +/- 15 in self-play while it converted
36 more won endings out of 110. This suite is the instrument that can see it. Every position
is already winning for the side to move; the defender is Stockfish at a fixed depth, so it
never hands the win back. What is measured is our own technique:

    conversion rate         mate delivered inside the ply cap
    mean moves to win       over the converted positions only (a shorter win is a better win)
    fifty / threefold       the two ways a won ending gets thrown away
    max halfmove clock      how close to the fifty-move cliff we let the game drift
    resets                  clock-resetting moves we played (captures and pawn moves)

    uv run python -m bench.conversion
    uv run python -m bench.conversion --agent work/v13-rule50 --json out.json
    uv run python -m bench.conversion --tier textbook --base 60000
    uv run python -m bench.conversion --positions bench/endings.json

The clock is real: the agent is called through get_move(fen, time_left_ms) exactly as the
platform calls it, so its own time manager, game history and transposition table all stay
live across the playout, and running out of time is a loss like any other.

Building the real-game tier (positions taken from won endings in our own pool games):

    uv run python -m bench.conversion --build-endings bench/endings.json \\
        --from 'results/condor/lmr-vs-v12.27/games_*.pgn'
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

from bench.load import Agent

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = ROOT / "tools" / "engines" / "stockfish"
SCORE_RE = re.compile(r"score (-?\d+)")

# tier, name, fen, expected: "mate" = a forced win by chess theory or a tablebase,
# "win" = winning by a decisive margin (Stockfish depth 24 agrees; see --verify),
# "draw" = a control that is NOT won and must not be lost.
SUITE: list[tuple[str, str, str, str]] = [
    # ---- textbook: bare-king mates. Every one is a tablebase win; failing one is a bug. ----
    ("textbook", "KQ v K", "8/8/1k6/3K4/8/2Q5/8/8 w - - 0 1", "mate"),
    ("textbook", "KQ v K corner", "8/8/8/8/8/k7/8/1K1Q4 w - - 0 1", "mate"),
    ("textbook", "KR v K", "8/8/1k6/3K4/8/2R5/8/8 w - - 0 1", "mate"),
    ("textbook", "KR v K far", "8/8/8/8/8/k7/8/1K1R4 w - - 0 1", "mate"),
    ("textbook", "KRR v K", "8/8/1k6/3K4/8/2R5/6R1/8 w - - 0 1", "mate"),
    ("textbook", "KBB v K", "8/8/1k6/3K4/8/2B5/4B3/8 w - - 0 1", "mate"),
    ("textbook", "KBN v K", "8/8/1k6/3K4/8/2B5/4N3/8 w - - 0 1", "mate"),
    ("textbook", "KQ v KR", "8/8/1k6/3K4/8/2Q5/5r2/8 w - - 0 1", "mate"),
    ("textbook", "KRP v K", "8/8/1k6/3K4/1P6/2R5/8/8 w - - 0 1", "mate"),
    ("textbook", "KQPP v K r388", "7Q/2k5/8/5K2/1P6/8/5P2/8 w - - 1 66", "mate"),
    ("textbook", "KQ v K r205", "8/8/8/8/4q3/6K1/2k5/8 b - - 8 71", "mate"),
    ("textbook", "KQ v KP", "8/8/8/8/2k5/2p5/8/K1Q5 w - - 0 1", "mate"),
    # ---- clock: the same wins with the fifty-move counter already running. This is the tier
    # that a halfmove-clock damping term is supposed to move: the win is still forced, but the
    # engine has to make progress rather than shuffle, and it has fewer moves to do it in.
    ("clock", "KQ v K hm50", "8/8/1k6/3K4/8/2Q5/8/8 w - - 50 60", "mate"),
    ("clock", "KQ v K hm80", "8/8/1k6/3K4/8/2Q5/8/8 w - - 80 75", "mate"),
    ("clock", "KR v K hm50", "8/8/1k6/3K4/8/2R5/8/8 w - - 50 60", "mate"),
    ("clock", "KR v K hm70", "8/8/1k6/3K4/8/2R5/8/8 w - - 70 70", "mate"),
    ("clock", "KBB v K hm40", "8/8/1k6/3K4/8/2B5/4B3/8 w - - 40 55", "mate"),
    ("clock", "KBN v K hm30", "8/8/1k6/3K4/8/2B5/4N3/8 w - - 30 45", "mate"),
    ("clock", "KQ v KR hm60", "8/8/1k6/3K4/8/2Q5/5r2/8 w - - 60 65", "mate"),
    ("clock", "KRP v K hm60", "8/8/1k6/3K4/1P6/2R5/8/8 w - - 60 65", "mate"),
    # ---- technical: rook endings, pawn endings and opposite-coloured bishops, all winning,
    # all needing a plan rather than a tactic. Verified with --verify.
    ("technical", "R+2P v R", "8/8/4k3/8/2P1P3/8/4K3/2r1R3 w - - 0 1", "win"),
    ("technical", "R+2P v R hm60", "8/8/4k3/8/2P1P3/8/4K3/2r1R3 w - - 60 70", "win"),
    ("technical", "R+P v R Lucena", "1K6/1P1k4/8/8/8/8/r7/2R5 w - - 0 1", "win"),
    ("technical", "K+2P v K", "8/8/4k3/8/8/3PKP2/8/8 w - - 0 1", "win"),
    ("technical", "K+2P v K hm70", "8/8/4k3/8/8/3PKP2/8/8 w - - 70 80", "win"),
    ("technical", "outside passer", "8/5k2/8/p7/P7/4K3/4P3/8 w - - 0 1", "win"),
    ("technical", "OCB two pawns", "8/5k2/8/3B4/1P3P2/8/5b2/6K1 w - - 0 1", "win"),
    ("technical", "OCB two pawns hm60", "8/5k2/8/3B4/1P3P2/8/5b2/6K1 w - - 60 70", "win"),
    ("technical", "Q v R endgame", "8/5k2/8/8/8/2Q5/5r2/6K1 w - - 0 1", "win"),
    ("technical", "R v N", "8/5k2/6n1/8/8/2R5/8/6K1 w - - 20 40", "win"),
    ("technical", "N+B+P v N", "8/5k2/6n1/8/2P5/2NB4/8/6K1 w - - 30 45", "win"),
    # ---- material: middlegames a rook or a queen up. Conversion here is where our ladder
    # record is strong (65.5% under 110 moves), so a regression shows up in this tier first.
    (
        "material",
        "rook up middlegame",
        "r2q1rk1/pp2ppbp/2n2np1/3p4/3P4/2N1BN2/PP2BPPP/3Q1RK1 w - - 0 12",
        "win",
    ),
    (
        "material",
        "queen for rook",
        "r4rk1/pp2ppbp/2n2np1/3p4/3P4/2N1BN2/PP2BPPP/3QR1K1 w - - 0 14",
        "win",
    ),
    (
        "material",
        "two pieces up",
        "r2q1rk1/pp2ppbp/5np1/3p4/3P4/2N1BN2/PP2BPPP/R2Q1RK1 w - - 0 12",
        "win",
    ),
    (
        "material",
        "rook up simplified",
        "6k1/pp3ppp/8/3p4/3P4/5N2/PP3PPP/3R2K1 w - - 0 26",
        "win",
    ),
    # ---- controls: NOT won. A damping or drawishness term that over-corrects loses these.
    ("control", "wrong bishop r34", "8/5k2/8/8/2b5/p7/2K5/8 b - - 0 40", "draw"),
    ("control", "OCB dead r89", "2k5/8/8/4B1K1/8/7b/8/8 b - - 0 60", "draw"),
    ("control", "R+P v R book r382", "R7/P5k1/8/r7/8/3K4/8/8 b - - 0 55", "draw"),
    ("control", "KNN v K", "8/8/1k6/3K4/8/2N5/4N3/8 w - - 0 1", "draw"),
]

CAP_PLIES = 200


@dataclass
class Outcome:
    tier: str
    name: str
    fen: str
    expected: str
    result: str  # MATE, MATED, FIFTY, THREEFOLD, STALEMATE, INSUFFICIENT, FLAG, CAP
    plies: int
    max_halfmove: int
    resets: int
    final_score: int | None
    seconds: float
    depths: list[int] = field(default_factory=list)

    @property
    def converted(self) -> bool:
        return self.result == "MATE"

    @property
    def held(self) -> bool:
        """A control is held when we did not lose it."""
        return self.result != "MATED" and self.result != "FLAG"


def defender_engine(depth: int, elo: int | None) -> chess.engine.SimpleEngine:
    if not STOCKFISH.exists():
        raise SystemExit(f"{STOCKFISH} missing — the defender needs it (tools/engines/README.md)")
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    options: dict[str, Any] = {"Threads": 1, "Hash": 64}
    if elo is not None:
        options["UCI_LimitStrength"] = True
        options["UCI_Elo"] = elo
    engine.configure(options)
    del depth
    return engine


def play_out(
    module: Any,
    entry: tuple[str, str, str, str],
    defender: chess.engine.SimpleEngine,
    depth: int,
    base_ms: int,
    inc_ms: int,
    cap: int,
) -> Outcome:
    tier, name, fen, expected = entry
    board = chess.Board(fen)
    ours = board.turn
    clock = float(base_ms)
    plies = 0
    max_halfmove = board.halfmove_clock
    resets = 0
    scores: list[int] = []
    depths: list[int] = []
    result = "CAP"
    started = time.perf_counter()
    limit = chess.engine.Limit(depth=depth)
    while plies < cap:
        if board.is_checkmate():
            result = "MATE" if board.turn != ours else "MATED"
            break
        if board.is_stalemate():
            result = "STALEMATE"
            break
        if board.is_insufficient_material():
            result = "INSUFFICIENT"
            break
        if board.halfmove_clock >= 100:
            result = "FIFTY"
            break
        if board.is_repetition(3):
            result = "THREEFOLD"
            break
        if board.turn == ours:
            if clock <= 0:
                result = "FLAG"
                break
            captured = io.StringIO()
            move_started = time.perf_counter()
            with contextlib.redirect_stderr(captured):
                uci = module.get_move(board.fen(), int(clock))
            spent = (time.perf_counter() - move_started) * 1000.0
            clock = clock - spent + inc_ms
            text = captured.getvalue()
            found = SCORE_RE.findall(text)
            if found:
                scores.append(int(found[-1]))
            depth_found = re.findall(r"depth (\d+)", text)
            if depth_found:
                depths.append(int(depth_found[-1]))
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise SystemExit(f"{name}: illegal move {uci} in {board.fen()}")
            if board.is_capture(move) or board.piece_type_at(move.from_square) == chess.PAWN:
                resets += 1
        else:
            move = defender.play(board, limit).move
            if move is None:
                raise SystemExit(f"{name}: defender returned no move in {board.fen()}")
        board.push(move)
        plies += 1
        max_halfmove = max(max_halfmove, board.halfmove_clock)
    return Outcome(
        tier=tier,
        name=name,
        fen=fen,
        expected=expected,
        result=result,
        plies=plies,
        max_halfmove=max_halfmove,
        resets=resets,
        final_score=scores[-1] if scores else None,
        seconds=time.perf_counter() - started,
        depths=depths,
    )


def build_endings(pgn_glob: str, out: Path, per_signature: int = 3, back: int = 60) -> None:
    """Won endings from real games: the position `back` plies before a checkmate finish,
    few men, the winner to move, deduplicated by material signature."""
    kept: list[dict[str, Any]] = []
    seen: Counter[tuple[int, str]] = Counter()
    for path in sorted(glob.glob(pgn_glob)):
        with open(path) as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                if game.headers.get("Termination") != "checkmate":
                    continue
                board = game.board()
                fens = []
                for move in game.mainline_moves():
                    board.push(move)
                    fens.append(board.fen())
                if not board.is_checkmate() or len(fens) < 12:
                    continue
                winner = not board.turn
                for index in range(max(0, len(fens) - back), len(fens) - 8):
                    probe = chess.Board(fens[index])
                    men = chess.popcount(probe.occupied)
                    if men > 7 or probe.turn != winner or probe.is_game_over():
                        continue
                    signature = (men, str(sorted(probe.piece_map().values(), key=str)))
                    if seen[signature] >= per_signature:
                        break
                    seen[signature] += 1
                    kept.append(
                        {
                            "tier": "real",
                            "name": f"r{game.headers.get('Round', '?')} {men}men",
                            "fen": fens[index],
                            "expected": "win",
                        }
                    )
                    break
    out.write_text(json.dumps(kept, indent=1))
    print(f"{len(kept)} positions -> {out}")


def verify(entries: list[tuple[str, str, str, str]], seconds: float) -> None:
    """Ask Stockfish whether each entry really is what the table claims."""
    engine = defender_engine(24, None)
    try:
        for tier, name, fen, expected in entries:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(time=seconds))
            score = info["score"].pov(board.turn)
            mate = score.mate()
            value = score.score(mate_score=100000) or 0
            verdict = "mate" if mate is not None else ("win" if value >= 400 else "draw")
            flag = " " if verdict == expected or (expected == "win" and verdict == "mate") else "!"
            print(f"{flag} {tier:9} {name:22} claims {expected:5} sf says {verdict:5} {value:+7}")
    finally:
        engine.quit()


def report(outcomes: list[Outcome], label: str) -> dict[str, Any]:
    print(f"\n{label}")
    print(
        f"{'tier':10} {'position':22} {'result':11} {'moves':>6} {'hm':>4} "
        f"{'rst':>4} {'score':>7} {'d':>4} {'s':>6}"
    )
    for out in outcomes:
        mean_depth = sum(out.depths) / len(out.depths) if out.depths else 0.0
        print(
            f"{out.tier:10} {out.name:22} {out.result:11} {out.plies / 2:6.1f} "
            f"{out.max_halfmove:4} {out.resets:4} "
            f"{'-' if out.final_score is None else out.final_score:>7} "
            f"{mean_depth:4.1f} {out.seconds:6.1f}"
        )
    summary: dict[str, Any] = {"tiers": {}}
    wins = [o for o in outcomes if o.expected in ("mate", "win")]
    controls = [o for o in outcomes if o.expected == "draw"]
    print(f"\n{'tier':12} {'conv':>9}  {'rate':>6}  {'mean moves':>10}  {'charged':>8}  outcomes")
    for tier in sorted({o.tier for o in wins}):
        group = [o for o in wins if o.tier == tier]
        done = [o for o in group if o.converted]
        mean_moves = sum(o.plies for o in done) / (2 * len(done)) if done else float("nan")
        charged = sum(o.plies if o.converted else CAP_PLIES for o in group) / (2 * len(group))
        mix = ", ".join(f"{k} {v}" for k, v in Counter(o.result for o in group).most_common())
        print(
            f"{tier:12} {len(done):4}/{len(group):<4} {100 * len(done) / len(group):5.1f}%  "
            f"{mean_moves:10.1f}  {charged:8.1f}  {mix}"
        )
        summary["tiers"][tier] = {
            "converted": len(done),
            "total": len(group),
            "mean_moves": mean_moves,
            "charged_moves": charged,
            "outcomes": dict(Counter(o.result for o in group)),
        }
    done = [o for o in wins if o.converted]
    mean_moves = sum(o.plies for o in done) / (2 * len(done)) if done else float("nan")
    charged = sum(o.plies if o.converted else CAP_PLIES for o in wins) / (2 * len(wins))
    fifty = sum(1 for o in wins if o.result == "FIFTY")
    threefold = sum(1 for o in wins if o.result == "THREEFOLD")
    print(
        f"\nCONVERTED {len(done)}/{len(wins)} = {100 * len(done) / len(wins):.1f}%   "
        f"mean moves to win {mean_moves:.1f}   charged {charged:.1f}   "
        f"fifty {fifty}   threefold {threefold}"
    )
    if controls:
        held = sum(1 for o in controls if o.held)
        print(f"controls held {held}/{len(controls)}: " + ", ".join(
            f"{o.name}={o.result}" for o in controls
        ))
        summary["controls"] = {"held": held, "total": len(controls)}
    print(
        f"max halfmove clock reached: mean {sum(o.max_halfmove for o in wins) / len(wins):.1f}, "
        f"at 99+ in {sum(1 for o in wins if o.max_halfmove >= 99)} of {len(wins)}; "
        f"clock resets played: {sum(o.resets for o in wins)}"
    )
    summary["converted"] = len(done)
    summary["total"] = len(wins)
    summary["rate"] = len(done) / len(wins)
    summary["mean_moves"] = mean_moves
    summary["charged_moves"] = charged
    summary["fifty"] = fifty
    summary["threefold"] = threefold
    summary["max_halfmove_mean"] = sum(o.max_halfmove for o in wins) / len(wins)
    summary["resets"] = sum(o.resets for o in wins)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default=".", help="agent directory (default: the root)")
    parser.add_argument("--tier", action="append", help="only these tiers (repeatable)")
    parser.add_argument("--positions", help="extra positions as JSON (see --build-endings)")
    parser.add_argument("--base", type=int, default=15000, help="clock in ms (default 15000)")
    parser.add_argument("--inc", type=int, default=500, help="increment in ms (default 500)")
    parser.add_argument("--sf-depth", type=int, default=12, help="defender depth (default 12)")
    parser.add_argument("--sf-elo", type=int, help="limit the defender's strength")
    parser.add_argument("--cap", type=int, default=CAP_PLIES, help="ply cap per position")
    parser.add_argument("--json", help="write the summary here")
    parser.add_argument("--verify", action="store_true", help="check the table against Stockfish")
    parser.add_argument("--build-endings", help="write real won endings to this JSON file")
    parser.add_argument("--from", dest="pgn_glob", help="PGN glob for --build-endings")
    args = parser.parse_args()

    entries = list(SUITE)
    if args.positions:
        loaded = json.loads(Path(args.positions).read_text())
        entries += [(e["tier"], e["name"], e["fen"], e["expected"]) for e in loaded]
    if args.tier:
        entries = [e for e in entries if e[0] in args.tier]

    if args.build_endings:
        if not args.pgn_glob:
            raise SystemExit("--build-endings needs --from '<pgn glob>'")
        build_endings(args.pgn_glob, Path(args.build_endings))
        return
    if args.verify:
        verify(entries, 3.0)
        return

    directory = Path(args.agent)
    agent = Agent(directory)
    module = agent.module
    compile_started = time.perf_counter()
    module.wait_native()
    compile_s = time.perf_counter() - compile_started
    if not module.native_ready():
        raise SystemExit(
            f"{directory}: agent.native_ready() is False — this suite measures the native "
            "engine, and the Python fallback would pass it while proving nothing"
        )
    print(f"{directory.resolve().name}: native_ready True, compile {compile_s:.1f} s")
    print(
        f"{len(entries)} positions, clock {args.base / 1000:.0f}s + {args.inc / 1000:.1f}s, "
        f"defender Stockfish depth {args.sf_depth}"
        + (f" limited to {args.sf_elo}" if args.sf_elo else "")
    )
    defender = defender_engine(args.sf_depth, args.sf_elo)
    outcomes: list[Outcome] = []
    try:
        for entry in entries:
            outcome = play_out(
                module, entry, defender, args.sf_depth, args.base, args.inc, args.cap
            )
            outcomes.append(outcome)
            print(
                f"  {outcome.tier:10} {outcome.name:22} {outcome.result:11} "
                f"{outcome.plies / 2:5.1f} moves  hm{outcome.max_halfmove:3}  "
                f"{outcome.seconds:5.1f}s",
                flush=True,
            )
    finally:
        defender.quit()
    summary = report(outcomes, f"{directory.resolve().name} — conversion suite")
    if args.json:
        summary["agent"] = str(directory.resolve())
        summary["positions"] = [
            {
                "tier": o.tier,
                "name": o.name,
                "result": o.result,
                "plies": o.plies,
                "max_halfmove": o.max_halfmove,
                "resets": o.resets,
                "final_score": o.final_score,
            }
            for o in outcomes
        ]
        Path(args.json).write_text(json.dumps(summary, indent=1))
        print(f"summary -> {args.json}")


if __name__ == "__main__":
    main()
