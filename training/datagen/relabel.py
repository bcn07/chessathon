"""Relabel self-play chunks with Stockfish.

    uv run python training/datagen/relabel.py --input results/datagen/gen3/chunk_0.bin \
        --output results/datagen/gen3sf/chunk_0.bin --engine tools/engines/stockfish --nodes 20000

Reads selfplay.py's 27-byte records (occ, 16 nibbles, mover score, mover result), rebuilds the
canonical position (side to move as White; castling and en-passant rights are not stored and are
left empty), asks Stockfish for a fixed-node search and writes the same record with Stockfish's
score for the side to move (clipped to +/-2000, mates at the clip). Positions python-chess deems
invalid keep their old score. Same layout in and out, so build_set.py reads the result as a gen.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nnue"))
from encoding import SYMBOL_FROM_PIECE

CHUNK = np.dtype([("occ", "<u8"), ("nib", "u1", (16,)), ("score", "<i2"), ("result", "i1")])
CLAMP = 2000


def board_from_record(occ: int, nib: np.ndarray) -> chess.Board:
    board = chess.Board(None)
    count = 0
    bits = int(occ)
    while bits:
        square = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        byte = int(nib[count >> 1])
        code = (byte >> 4) if (count & 1) else (byte & 15)
        board.set_piece_at(square, chess.Piece.from_symbol(SYMBOL_FROM_PIECE[code]))
        count += 1
    board.turn = chess.WHITE
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", default="tools/engines/stockfish")
    parser.add_argument("--nodes", type=int, default=20_000)
    parser.add_argument("--limit", type=int, default=0, help="only the first N records (tests)")
    args = parser.parse_args()

    records = np.fromfile(args.input, dtype=CHUNK)
    if args.limit:
        records = records[: args.limit]
    out = records.copy()
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 16})
    limit = chess.engine.Limit(nodes=args.nodes)
    started = time.perf_counter()
    invalid = 0
    for index, record in enumerate(records):
        board = board_from_record(int(record["occ"]), record["nib"])
        if not board.is_valid():
            invalid += 1
            continue
        info = engine.analyse(board, limit)
        score = info["score"].pov(chess.WHITE).score(mate_score=CLAMP)
        out["score"][index] = max(-CLAMP, min(CLAMP, int(score)))
        if (index + 1) % 5000 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"{index + 1}/{len(records)} positions, {rate:.1f}/s", flush=True)
    engine.quit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(args.output)
    old = records["score"].astype(np.float64)
    new = out["score"].astype(np.float64)
    corr = float(np.corrcoef(old, new)[0, 1]) if len(records) > 1 else float("nan")
    elapsed = time.perf_counter() - started
    print(
        f"wrote {args.output}: {len(records):,} records in {elapsed:.0f}s "
        f"({len(records) / max(elapsed, 1e-9):.1f}/s), {invalid} invalid kept; "
        f"corr(old, new) {corr:.3f}, MAE {np.abs(old - new).mean():.1f} cp, "
        f"mean old {old.mean():+.1f} new {new.mean():+.1f}"
    )


if __name__ == "__main__":
    main()
