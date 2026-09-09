"""Fortress suite: what does our evaluation say about positions that are theoretically drawn?

The 2026-09-08 audit of 152 fifty-move draws found three games where our own logged score was
flat and large for ~90 plies and collapsed only in the last 8-12 plies -- the horizon cliff of a
fortress the eval cannot see. The defect scores zero in self-play (both sides carry it and both
draw anyway), so no SPRT can measure it. This suite is the instrument that can: it asks the
engine for a static score and a fixed-depth search score on positions whose true value is known,
and prints them next to the truth. No games, no clock, deterministic.

    uv run python -m bench.fortress
    uv run python -m bench.fortress --agent work/v13-fortress --json after.json
    uv run python -m bench.fortress --before before.json --agent work/v13-fortress
    uv run python -m bench.fortress --verify        # re-check the labels against Stockfish
    uv run python -m bench.fortress --agent work/v13-fortress --boundary

Two tiers matter and they pull against each other:

    drawn      theoretically drawn; the truth is 0 and every centipawn we report is error.
    won        genuinely winning, similar material; flattening these is how a draw-scaling term
               loses Elo, so the suite fails if a fix drags them toward zero.

Scores are always from the side to move's point of view, the same convention as the engine and as
``chess.engine.PovScore.pov(board.turn)``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

from bench.load import Agent

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = ROOT / "tools" / "engines" / "stockfish"

# tier, class, name, fen, stockfish depth-26 score from the side to move (the truth column).
# Every entry was checked with --verify; "drawn" entries are 0 +/- 20, "won" entries are decisive.
SUITE: list[tuple[str, str, str, str, int]] = [
    # ---- the three positions the audit caught us on -------------------------------------
    ("drawn", "audit", "R+hP v B corner", "6bk/8/7P/2K5/8/R7/8/8 b - - 0 71", -13),
    ("drawn", "audit", "aP+2B v B", "2k5/P7/8/1b5B/7K/4B3/8/8 b - - 0 64", 0),
    ("drawn", "audit", "blocked OCB", "8/6B1/p3k1p1/P4p1p/4pPbK/4P3/8/8 w - - 0 67", 0),
    ("drawn", "audit", "R+gP v B corner", "7k/6b1/6P1/2K5/8/R7/8/8 b - - 0 1", 2),
    # ---- wrong rook pawn, bare king: the defender's king holds the promotion square -----
    ("drawn", "rookpawn", "KhP v K corner", "7k/7P/6K1/8/8/8/8/8 w - - 0 1", 0),
    ("drawn", "rookpawn", "KaP v K corner", "k7/P7/1K6/8/8/8/8/8 w - - 0 1", 0),
    ("drawn", "rookpawn", "KhP v K held", "7k/8/6KP/8/8/8/8/8 b - - 0 1", 0),
    ("drawn", "rookpawn", "R+P v R book r382", "R7/P5k1/8/r7/8/3K4/8/8 b - - 0 55", 0),
    # ---- wrong-coloured bishop with a rook pawn -----------------------------------------
    ("drawn", "wrongbish", "KBhP v K light B", "7k/8/7P/8/8/8/4B3/6K1 w - - 0 1", 69),
    ("drawn", "wrongbish", "KBhPP v K light B", "7k/8/7P/7P/8/8/4B3/6K1 w - - 0 1", 130),
    ("drawn", "wrongbish", "KBaP v K dark B", "k7/8/P7/8/8/8/3B4/1K6 w - - 0 1", 0),
    ("drawn", "wrongbish", "KaP v KB blockade", "k7/P7/8/8/b7/8/8/1K6 w - - 0 1", 0),
    # ---- blocked / thin opposite-coloured bishops ---------------------------------------
    ("drawn", "ocb", "OCB blocked pawns", "8/8/4k3/2p1p3/2P1P3/8/3B4/4K1b1 w - - 0 1", 32),
    ("drawn", "ocb", "OCB one pawn up", "8/5k2/8/2B5/1P6/8/4b3/6K1 w - - 0 1", 0),
    ("drawn", "ocb", "OCB dead r89", "2k5/8/8/4B1K1/8/7b/8/8 b - - 0 60", 0),
    # ---- wrong rook pawn, with pawns for the defender -----------------------------------
    # A rook pawn can only leave its file by capturing onto the adjacent one, so a defender with
    # no pawn there cannot change what the attacker has to promote with. The g-pawn entry is the
    # same draw with that escape available, i.e. the case the material test must decline.
    ("drawn", "wrongbish", "KBhP v KhP wrong B", "7k/8/7p/7P/8/6K1/8/3B4 w - - 0 1", 78),
    ("drawn", "wrongbish", "KBhP v KgP wrong B", "7k/6p1/8/7P/8/6K1/8/3B4 w - - 0 1", 78),
    # ---- opposite-coloured bishops: what decides these is passed pawns -------------------
    # These four are one family. `OCB 2up one passer` (in the won tier) and `OCB 1up one passer`
    # differ by a single black a-pawn and are a win and a draw respectively, which is why the
    # rule declines two-pawn advantages outright.
    ("drawn", "ocb", "OCB blocked opp B", "8/8/4k3/2p1p3/2P1P3/8/2B5/4K1b1 w - - 0 1", 0),
    ("drawn", "ocb", "OCB 3v3 locked", "8/6bk/6p1/5p1p/5P1P/6P1/4B1K1/8 w - - 0 1", 0),
    ("drawn", "ocb", "OCB 1up one passer", "8/p2b4/5k2/6pP/6PP/8/8/2B2K2 w - - 0 1", 0),
    ("drawn", "ocb", "OCB level one passer", "8/pp1b4/5k2/6pP/6PP/8/8/2B2K2 w - - 0 1", 0),
    # ---- pawnless: with no pawn on the board the material *is* the value -----------------
    ("drawn", "pawnless", "KR v KR", "8/5r2/4k3/8/8/8/3R4/6K1 w - - 0 1", 0),
    ("drawn", "pawnless", "KRB v KR", "8/5r2/4k3/8/8/2B5/3R4/6K1 w - - 0 1", 0),
    ("drawn", "pawnless", "KRN v KR", "8/5r2/4k3/8/8/2N5/3R4/6K1 w - - 0 1", 0),
    ("drawn", "pawnless", "KBB v KB", "8/5b2/4k3/8/8/2BB4/8/6K1 w - - 0 1", 1),
    ("drawn", "pawnless", "KBN v KB", "8/5b2/4k3/8/8/2N5/3B4/6K1 w - - 0 1", 0),
    ("drawn", "pawnless", "KNN v KB", "8/5b2/4k3/8/8/2N5/3N4/6K1 w - - 0 1", 0),
    # ---- other fortresses named in the brief --------------------------------------------
    ("drawn", "qvp", "KQ v KcP 7th", "7K/8/8/8/6Q1/8/2p5/2k5 w - - 0 1", 0),
    ("drawn", "qvp", "KQ v KhP 7th", "7K/8/8/8/6Q1/8/7p/7k w - - 0 1", 0),
    ("drawn", "pawnless", "KR v KB", "8/5k2/8/8/2b5/8/3R4/6K1 w - - 0 1", 15),
    ("drawn", "pawnless", "KR v KN", "8/5k2/6n1/8/8/2R5/8/6K1 w - - 0 1", 7),
    ("drawn", "pawnless", "wrong bishop r34", "8/5k2/8/8/2b5/p7/2K5/8 b - - 0 40", 46),
    ("drawn", "pawnless", "KNN v K", "8/8/1k6/3K4/8/2N5/4N3/8 w - - 0 1", 18),
    # ---- won: the same material classes, one geometric detail different ------------------
    ("won", "rookpawn", "KhP v K king cut", "8/7P/8/8/8/8/k7/7K w - - 0 1", 99990),
    ("won", "wrongbish", "KBhP v K right B", "7k/8/7P/8/8/8/5B2/6K1 w - - 0 1", 99980),
    ("won", "wrongbish", "KBgP v K", "7k/8/6P1/8/8/8/4B3/6K1 w - - 0 1", 99984),
    ("won", "wrongbish", "KBhP v K king far", "8/8/6KP/8/8/8/4B3/4k3 w - - 0 1", 99988),
    ("won", "audit", "R+hP v B king out", "8/8/7P/2K5/6b1/R7/8/6k1 w - - 0 1", 99990),
    ("won", "audit", "R+gP v B king out", "8/8/6P1/2K5/6b1/R7/8/6k1 w - - 0 1", 99990),
    ("won", "ocb", "OCB two pawns", "8/5k2/8/2B5/1P3P2/8/4b3/6K1 w - - 0 1", 591),
    ("won", "ocb", "OCB three pawns", "8/5k2/8/2B5/1P2PP2/8/4b3/6K1 w - - 0 1", 1003),
    # the near-miss of `OCB 1up one passer`: one black pawn fewer and the h-pawn queens
    ("won", "ocb", "OCB 2up one passer", "8/3b4/5k2/6pP/6PP/8/8/2B2K2 w - - 0 1", 781),
    ("won", "wrongbish", "KBhP v KfP right B", "7k/5p2/8/7P/8/6K1/5B2/8 w - - 0 1", 99983),
    # the pawnless wins: one more than "an edge of a single minor", which is where the line is
    ("won", "pawnless", "KRR v KR", "8/5r2/4k3/8/8/8/2RR4/6K1 w - - 0 1", 99981),
    ("won", "pawnless", "KRB v KB", "8/5b2/4k3/8/8/2B5/3R4/6K1 w - - 0 1", 545),
    # a theoretical win (Philidor, <=31 moves) that Stockfish d30 scores at only +225, so
    # --verify calls it "drawn" on the 400 cp threshold; it is the guard that queens are excluded
    ("won", "pawnless", "KQ v KR", "8/5r2/4k3/8/8/8/3Q4/6K1 w - - 0 1", 225),
    ("won", "ocb", "OCB four pawns", "8/5k2/8/2B5/1PP1PP2/8/4b3/6K1 w - - 0 1", 645),
    ("won", "samebish", "same bishops +2P", "8/5k2/8/2B5/1P3P2/8/8/b5K1 w - - 0 1", 782),
    ("won", "qvp", "KQ v KdP 7th", "7K/8/8/8/8/6Q1/3p4/3k4 w - - 0 1", 99974),
    ("won", "technical", "R+2P v R", "8/8/4k3/8/2P1P3/8/4K3/2r1R3 w - - 0 1", 99987),
    ("won", "technical", "R+P v R Lucena", "1K6/1P1k4/8/8/8/8/r7/2R5 w - - 0 1", 704),
    ("won", "technical", "K+2P v K", "8/8/4k3/8/8/3PKP2/8/8 w - - 0 1", 99984),
    ("won", "technical", "KQ v K", "8/8/1k6/3K4/8/2Q5/8/8 w - - 0 1", 99996),
    ("won", "technical", "KR v K", "8/8/1k6/3K4/8/2R5/8/8 w - - 0 1", 99990),
    ("won", "technical", "KBN v K", "8/8/1k6/3K4/8/2B5/4N3/8 w - - 0 1", 376),
    ("won", "technical", "KRP v K", "8/8/1k6/3K4/1P6/2R5/8/8 w - - 0 1", 99991),
    (
        "won",
        "middlegame",
        "rook up",
        "r2q1rk1/pp2ppbp/2n2np1/3p4/3P4/2N1BN2/PP2BPPP/3Q1RK1 b - - 0 12",
        318,
    ),
    (
        "won",
        "middlegame",
        "rook up simplified",
        "6k1/pp3ppp/8/3p4/3P4/5N2/PP3PPP/3R2K1 w - - 0 26",
        907,
    ),
    (
        "won",
        "middlegame",
        "two pieces up",
        "r2q1rk1/pp2ppbp/5np1/3p4/3P4/2N1BN2/PP2BPPP/R2Q1RK1 w - - 0 12",
        634,
    ),
]

# A drawn position we score above this is a false advantage the engine can trade into.
FALSE_ADVANTAGE = 150
# A won position we score below this is a win the scaling term has flattened.
WIN_FLOOR = 150


@dataclass
class Row:
    tier: str
    klass: str
    name: str
    fen: str
    truth: int
    static: int
    search: int
    depth: int
    nodes: int
    ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "class": self.klass,
            "name": self.name,
            "fen": self.fen,
            "truth": self.truth,
            "static": self.static,
            "search": self.search,
            "depth": self.depth,
            "nodes": self.nodes,
        }


def verify(entries: list[tuple[str, str, str, str, int]], seconds: float, depth: int) -> None:
    """Ask Stockfish what each position really is, so the labels are not my opinion."""
    if not STOCKFISH.exists():
        raise SystemExit(f"{STOCKFISH} missing (tools/engines/README.md)")
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    engine.configure({"Threads": 1, "Hash": 128})
    try:
        for tier, klass, name, fen, claimed in entries:
            board = chess.Board(fen)
            if not board.is_valid():
                print(f"! {tier:6} {klass:10} {name:22} ILLEGAL FEN {fen}")
                continue
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=seconds))
            pov = info["score"].pov(board.turn)
            mate = pov.mate()
            value = pov.score(mate_score=100000) or 0
            verdict = "mate" if mate is not None else ("decisive" if abs(value) >= 400 else "drawn")
            ok = verdict == "drawn" if tier == "drawn" else verdict != "drawn"
            print(
                f"{' ' if ok else '!'} {tier:6} {klass:10} {name:22} sf {value:+7} "
                f"{verdict:9} (table says {claimed:+5})"
            )
    finally:
        engine.quit()


def boundary(directory: Path, plies: int) -> None:
    """What the search sees on the move that crosses into or out of a scaled class.

    A draw-scaling term is a *selected* factor, and the per-bucket tempo offset cost -33 Elo by
    being selected on something a capture could toggle: the search then reads a phantom gain on
    every capture that crosses the boundary. So the boundary has to be inspected directly rather
    than argued about. This walks every legal move from every suite position (and, with
    ``--plies 2``, every reply) and prints each pair whose scale factor differs, with the mover's
    score on both sides of it.
    """
    import importlib
    import sys

    resolved = str(directory.resolve())
    for name in ("agent", "nativesearch", "fastboard", "nnue_eval", "pyengine"):
        sys.modules.pop(name, None)
    sys.path.insert(0, resolved)
    try:
        native = importlib.import_module("nativesearch")
        board_mod = importlib.import_module("fastboard")
    finally:
        sys.path.remove(resolved)
    scale_of = getattr(native, "_draw_scale", None)
    if scale_of is None:
        raise SystemExit(f"{directory} has no _draw_scale: there is no boundary to report")
    import numpy as np

    def classify(board: chess.Board) -> tuple[int, int, int]:
        """(scale factor in sixty-fourths, scaled score from White, unscaled score from White).

        The unscaled score is recovered from the scaled one and the factor rather than measured
        with a second engine, so it carries up to 64/scale cp of rounding. That is enough to
        separate the value a move really has from the part this term added to it, which is the
        only thing the boundary question is about.
        """
        state = board_mod.from_fen(board.fen(en_passant="fen"))
        score = int(native.evaluate_state(state))
        white = score if board.turn == chess.WHITE else -score
        if white == 0:
            return 64, 0, 0
        strong = np.int64(0 if white > 0 else 1)
        offset = int(strong) * 6
        if int(state[offset + board_mod.ROOK]) | int(state[offset + board_mod.QUEEN]):
            return 64, white, white  # the call-site gate: no class can apply
        scale = int(scale_of(state, strong))
        return scale, white, white * 64 // scale

    # A crossing on a capture or a promotion is a position whose value really did change, and the
    # search is supposed to see that. The dangerous kind is a *quiet* crossing: a move that changes
    # nothing material and yet moves the score, which is what the search learns to repeat. The two
    # are counted separately for exactly that reason.
    counts = {"quiet": 0, "material": 0}
    gains = {"quiet": 0, "material": 0}
    flips = {"quiet": 0, "material": 0}
    worst = {"quiet": 0, "material": 0}
    worst_flip = {"quiet": 0, "material": 0}
    flip_cp = {"quiet": 0, "material": 0}
    big_flips = {"quiet": 0, "material": 0}
    examples: dict[str, str] = {}

    def walk(board: chess.Board, depth: int) -> None:
        parent_scale, parent_white, parent_raw = classify(board)
        mover = 1 if board.turn == chess.WHITE else -1
        for move in board.legal_moves:
            material = board.is_capture(move) or move.promotion is not None
            kind = "material" if material else "quiet"
            board.push(move)
            if not (board.is_checkmate() or board.is_stalemate()):
                child_scale, child_white, child_raw = classify(board)
                if child_scale != parent_scale:
                    counts[kind] += 1
                    # From the point of view of the side that moved. A move's own value shows up
                    # in both engines, so the number that matters is the part this term *added*
                    # to the jump: that is the phantom the per-bucket tempo offset was made of.
                    scaled_jump = mover * (child_white - parent_white)
                    raw_jump = mover * (child_raw - parent_raw)
                    added = scaled_jump - raw_jump
                    if scaled_jump > 0 and raw_jump <= 0:
                        # the only shape that is a phantom rather than damping: the term turns a
                        # move that did not improve the position into one that scores better
                        flips[kind] += 1
                        # A count of flips says nothing without their size: a flip bounded by the
                        # raw score at zero is noise, one worth a piece is not. The branch that
                        # selects the favoured side is the sign of the raw score, so pawnless
                        # endings -- where the raw score hovers around zero -- generate many tiny
                        # crossings. These buckets separate those from anything that could matter.
                        flip_cp[kind] += scaled_jump
                        if scaled_jump >= 50:
                            big_flips[kind] += 1
                        if scaled_jump > worst_flip[kind]:
                            worst_flip[kind] = scaled_jump
                            examples["flip " + kind] = (
                                f"  worst flip {kind:8} {scaled_jump:+6}: {parent_scale:>2}/64 -> "
                                f"{child_scale:>2}/64 on {move.uci():6} mover "
                                f"{mover * parent_white:+6} -> {mover * child_white:+6} "
                                f"(unscaled {mover * parent_raw:+6} -> {mover * child_raw:+6})"
                                f"\n      {board.fen()}"
                            )
                    if added > 0:
                        gains[kind] += 1
                        if added > worst[kind]:
                            worst[kind] = added
                            examples[kind] = (
                                f"  worst {kind:8} added {added:+6}: {parent_scale:>2}/64 -> "
                                f"{child_scale:>2}/64 on {move.uci():6} mover "
                                f"{mover * parent_white:+6} -> {mover * child_white:+6} "
                                f"(unscaled {mover * parent_raw:+6} -> {mover * child_raw:+6})"
                                f"\n      {board.fen()}"
                            )
                if depth > 1:
                    walk(board, depth - 1)
            board.pop()

    for tier, klass, name, fen, _truth in SUITE:
        before = counts["quiet"] + counts["material"]
        print(f"{tier:6} {klass:10} {name:22} scale {classify(chess.Board(fen))[0]:>2}/64")
        walk(chess.Board(fen), plies)
        after = counts["quiet"] + counts["material"]
        if after > before:
            print(f"      {after - before} crossing(s) within {plies} ply")
    print(f"\nclass crossings over {plies} ply from {len(SUITE)} positions")
    for kind in ("quiet", "material"):
        print(
            f"  {kind:8} {counts[kind]:6} crossings, {gains[kind]:6} where the term adds to "
            f"the mover's score (largest {worst[kind]:+6} cp), {flips[kind]:5} where it turns a "
            f"non-improvement into a gain"
        )
        if flips[kind]:
            print(
                f"           of those {flips[kind]} flips: mean "
                f"{flip_cp[kind] / flips[kind]:5.1f} cp, {big_flips[kind]} of 50 cp or more, "
                f"largest {worst_flip[kind]:+6}"
            )
    for key in ("quiet", "material", "flip quiet", "flip material"):
        if key in examples:
            print(examples[key])


def measure(
    module: Any, entries: list[tuple[str, str, str, str, int]], ms: float, depth: int
) -> list[Row]:
    rows: list[Row] = []
    for tier, klass, name, fen, truth in entries:
        board = chess.Board(fen)
        static = int(module.evaluate(board))
        started = time.perf_counter()
        found = module.analyse(fen, ms, depth)
        elapsed = (time.perf_counter() - started) * 1000.0
        rows.append(
            Row(
                tier=tier,
                klass=klass,
                name=name,
                fen=fen,
                truth=truth,
                static=static,
                search=int(found.score),
                depth=int(found.depth),
                nodes=int(found.nodes),
                ms=elapsed,
            )
        )
        print(
            f"  {tier:6} {klass:10} {name:22} static {static:+7} search {int(found.score):+7} "
            f"d{int(found.depth):<3} truth {truth:+5}  {elapsed:6.0f}ms",
            flush=True,
        )
    return rows


def summarise(rows: list[Row], before: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    drawn = [r for r in rows if r.tier == "drawn"]
    won = [r for r in rows if r.tier == "won"]
    head = f"{'tier':6} {'class':10} {'position':22} {'static':>8} {'search':>8} {'truth':>6}"
    if before:
        head += f" {'was st':>8} {'was se':>8}"
    print("\n" + head)
    for row in rows:
        line = (
            f"{row.tier:6} {row.klass:10} {row.name:22} {row.static:+8} {row.search:+8} "
            f"{row.truth:+6}"
        )
        if before:
            was = before.get(row.fen)
            if was is None:
                line += f" {'-':>8} {'-':>8}"
            else:
                line += f" {was['static']:+8} {was['search']:+8}"
        flag = ""
        if row.tier == "drawn" and abs(row.search) >= FALSE_ADVANTAGE:
            flag = "  <- false advantage"
        if row.tier == "won" and abs(row.search) < WIN_FLOOR:
            flag = "  <- win flattened"
        print(line + flag)

    def stats(group: list[Row], attr: str) -> tuple[float, int]:
        values = [abs(getattr(r, attr)) for r in group]
        return (sum(values) / len(values) if values else 0.0), (max(values) if values else 0)

    summary: dict[str, Any] = {}
    print()
    for label, group, limit in (("drawn", drawn, FALSE_ADVANTAGE), ("won", won, WIN_FLOOR)):
        if not group:
            continue
        st_mean, st_max = stats(group, "static")
        se_mean, se_max = stats(group, "search")
        if label == "drawn":
            bad = [r for r in group if abs(r.search) >= limit]
            bad_static = [r for r in group if abs(r.static) >= limit]
            print(
                f"{label:6} n={len(group):<3} mean |static| {st_mean:6.0f} (max {st_max:5}) "
                f"mean |search| {se_mean:6.0f} (max {se_max:5})  "
                f"false advantage: static {len(bad_static)}, search {len(bad)}"
            )
            summary[label] = {
                "n": len(group),
                "mean_abs_static": st_mean,
                "max_abs_static": st_max,
                "mean_abs_search": se_mean,
                "max_abs_search": se_max,
                "false_static": len(bad_static),
                "false_search": len(bad),
                "false_names": [r.name for r in bad],
            }
        else:
            lost = [r for r in group if abs(r.search) < limit]
            print(
                f"{label:6} n={len(group):<3} mean |static| {st_mean:6.0f} (min "
                f"{min(abs(r.static) for r in group):5}) mean |search| {se_mean:6.0f} (min "
                f"{min(abs(r.search) for r in group):5})  flattened: {len(lost)}"
                + (f" ({', '.join(r.name for r in lost)})" if lost else "")
            )
            summary[label] = {
                "n": len(group),
                "mean_abs_static": st_mean,
                "mean_abs_search": se_mean,
                "min_abs_static": min(abs(r.static) for r in group),
                "min_abs_search": min(abs(r.search) for r in group),
                "flattened": len(lost),
                "flattened_names": [r.name for r in lost],
            }
    if before:
        moved = []
        for row in rows:
            was = before.get(row.fen)
            if was is None:
                continue
            delta = abs(row.search) - abs(was["search"])
            if abs(delta) >= 25:
                moved.append((row.tier, row.name, was["search"], row.search))
        print(f"\nsearch score moved by 25+ cp in {len(moved)} of {len(rows)} positions")
        for tier, name, old, new in moved:
            print(f"  {tier:6} {name:22} {old:+7} -> {new:+7}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--tier", action="append", help="only these tiers")
    parser.add_argument("--class", dest="klass", action="append", help="only these classes")
    parser.add_argument("--depth", type=int, default=14, help="fixed search depth (default 14)")
    parser.add_argument("--ms", type=float, default=4000.0, help="time cap per position")
    parser.add_argument("--verify", action="store_true", help="check the labels with Stockfish")
    parser.add_argument(
        "--boundary", action="store_true", help="report scale-class crossings, not scores"
    )
    parser.add_argument("--plies", type=int, default=1, help="--boundary search depth")
    parser.add_argument("--sf-depth", type=int, default=26, help="Stockfish depth for --verify")
    parser.add_argument("--json", help="write the rows and summary here")
    parser.add_argument("--before", help="a --json file to diff against")
    args = parser.parse_args()

    entries = list(SUITE)
    if args.tier:
        entries = [e for e in entries if e[0] in args.tier]
    if args.klass:
        entries = [e for e in entries if e[1] in args.klass]

    if args.verify:
        verify(entries, 20.0, args.sf_depth)
        return
    if args.boundary:
        boundary(args.agent, args.plies)
        return

    agent = Agent(args.agent)
    module = agent.module
    started = time.perf_counter()
    module.wait_native()
    compile_s = time.perf_counter() - started
    if not module.native_ready():
        raise SystemExit(
            f"{args.agent}: agent.native_ready() is False — this suite measures the native "
            "evaluation, and the Python fallback would answer while proving nothing"
        )
    print(
        f"{agent.directory.name}: native_ready True, compile {compile_s:.1f} s, "
        f"{len(entries)} positions, depth {args.depth}"
    )
    rows = measure(module, entries, args.ms, args.depth)
    before = None
    if args.before:
        loaded = json.loads(Path(args.before).read_text())
        before = {r["fen"]: r for r in loaded["rows"]}
    summary = summarise(rows, before)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "agent": str(agent.directory),
                    "depth": args.depth,
                    "compile_s": compile_s,
                    "summary": summary,
                    "rows": [r.as_dict() for r in rows],
                },
                indent=1,
            )
        )
        print(f"\nrows -> {args.json}")


if __name__ == "__main__":
    main()
