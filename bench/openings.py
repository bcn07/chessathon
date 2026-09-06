"""Balanced start positions for benchmarking.

Rated games start from curated near-level openings rather than the initial position, so local
games should too. Each entry is a name and a move sequence; the FENs are derived, which
guarantees they are legal. Every opening is played once with each colour by the gauntlet.
"""

import json
from pathlib import Path

import chess

_LINES: list[tuple[str, str]] = [
    ("Italian", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6"),
    ("Ruy Lopez", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O"),
    ("Sicilian Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7"),
    ("Sicilian Sveshnikov", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5"),
    ("French Tarrasch", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4 cxd4 f6"),
    (
        "Caro-Kann Classical",
        "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6",
    ),
    ("Scandinavian", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6"),
    ("Pirc", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4 Nbd7"),
    ("Petroff", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nb4"),
    (
        "Scotch",
        "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bc4 Ne5 Be2 Qg6",
    ),
    (
        "Vienna",
        "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d4 O-O Bd3 f5 exf6 Bxf6",
    ),
    (
        "Queen's Gambit Declined",
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6 Be2 Bb7 Bxf6 Bxf6 cxd5 exd5",
    ),
    ("Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7"),
    ("Semi-Slav Meran", "d4 d5 c4 c6 Nc3 Nf6 Nf3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 Bb7"),
    (
        "Nimzo-Indian",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7",
    ),
    (
        "Queen's Indian",
        "d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4+ Bd2 Be7 Bg2 c6 Bc3 d5 Ne5 Nfd7",
    ),
    (
        "King's Indian Classical",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7",
    ),
    (
        "Grunfeld Exchange",
        "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Rb1 O-O Be2 cxd4 cxd4 Qa5+",
    ),
    (
        "Catalan",
        "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7",
    ),
    ("London", "d4 d5 Bf4 Nf6 e3 c5 c3 Nc6 Nd2 e6 Ngf3 Bd6 Bg3 O-O Bd3 b6"),
    (
        "English Symmetrical",
        "c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 Nf6 O-O O-O d4 cxd4 Nxd4 Nxd4 Qxd4 d6",
    ),
    (
        "English Reversed Sicilian",
        "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 d3 O-O",
    ),
    (
        "Reti",
        "Nf3 d5 g3 Nf6 Bg2 c6 O-O Bg4 d3 Nbd7 Nbd2 e5 e4 dxe4 dxe4 Bc5",
    ),
    ("Dutch Leningrad", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 Qe8 d5 Na6"),
]


def _fen(moves: str) -> str:
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    return board.fen()


OPENINGS: list[tuple[str, str]] = [(name, _fen(moves)) for name, moves in _LINES]

BOOK_FILE = Path(__file__).with_name("book.json")


def load_book(name: str = "big") -> list[tuple[str, str]]:
    """``classic``: the 24 curated lines above. ``big``: ``bench/book.json`` — 1,000 balanced
    positions taken from our own real-clock games (``bench.build_book``), so a 400-game run
    plays 200 distinct openings instead of 24 repeated eight times. Falls back to classic."""
    if name == "classic" or (name == "big" and not BOOK_FILE.exists()):
        return OPENINGS
    if name == "platform":  # the ladder's own start positions, crawled from public games
        entries = json.loads(BOOK_FILE.with_name("platform_book.json").read_text())
    elif name == "big":
        entries = json.loads(BOOK_FILE.read_text())
    else:
        raise ValueError(f"unknown book {name!r}")
    return [(entry["name"], entry["fen"]) for entry in entries]


if __name__ == "__main__":
    for name, fen in OPENINGS:
        print(f"{name:28} {fen}")
