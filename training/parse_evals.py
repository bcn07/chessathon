"""Stream ``lichess_db_eval.jsonl.zst`` into a packed position/score file.

Offline tooling only: it needs ``zstandard`` (``uv run --with zstandard python ...``), which the
competition platform does not have.  The reader stops as soon as enough positions are kept, so a
byte-range prefix of the 21.7 GB archive is all that has to be downloaded.

Per line we take the deepest evaluation's first principal variation, map ``mate`` to +/-2000 cp,
clamp ``cp`` to +/-2000, flip white-relative scores to the side to move, and drop positions where
the side to move is in check (a static evaluation cannot be trained on those).  JSON is scanned
with byte searches rather than ``json.loads``: the ``line`` strings are 95% of the file and are
never needed.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numba
from encoding import RECORD_DTYPE, SCORE_CLAMP, pack_placement

import fastboard as fb

_PVS = b'"pvs":[{'
_DEPTH = b'"depth":'
_FEN = b'{"fen":"'
_CP = b'"cp":'
_MATE = b'"mate":'
_LINE = b'"line":"'
_FILES = b"abcdefgh"


def _is_capture(placement: bytes, move: bytes, black_to_move: bool) -> bool:
    """True when the move takes something, or promotes.

    The destination square's occupant decides it. A piece of the mover's own colour there means
    castling, which Stockfish's principal variations write as e1h1 -- not a capture. En passant
    (an empty destination) is missed, which is 0.3% of moves and not worth a board.

    Positions whose best move wins material are why the corpus's side to move averages +103 cp;
    training on them teaches the net a 150 cp tempo bonus that wrecks the search (see REPORT.md).
    """
    if len(move) < 4:
        return True
    if len(move) > 4 and move[4:5] not in (b"", b" ", b'"'):
        return True  # promotion
    file_index = _FILES.find(move[2:3])
    rank_digit = move[3:4]
    if file_index < 0 or not rank_digit.isdigit():
        return True
    ranks = placement.split(b"/")
    if len(ranks) != 8:
        return True
    column = 0
    for symbol in ranks[8 - int(rank_digit)]:
        if 49 <= symbol <= 56:  # "1".."8"
            column += symbol - 48
        else:
            if column == file_index:
                return (symbol >= 97) != black_to_move  # lowercase == black
            column += 1
        if column > file_index:
            return False
    return False


def parse_chunk(
    task: tuple[bytes, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """Parse newline-delimited JSON into (occ, nib, score, seen, no_score, bad, noisy)."""
    chunk, quiet_only = task
    lines = chunk.split(b"\n")
    occ_out = np.empty(len(lines), dtype=np.uint64)
    nib_out = np.empty((len(lines), 16), dtype=np.uint8)
    score_out = np.empty(len(lines), dtype=np.int16)
    kept = 0
    seen = 0
    no_score = 0
    bad = 0
    noisy = 0
    for line in lines:
        if not line:
            continue
        seen += 1
        if not line.startswith(_FEN):
            bad += 1
            continue
        end = line.index(b'"', 8)
        fen = line[8:end]
        space = fen.index(b" ")
        placement = fen[:space]
        black_to_move = fen[space + 1 : space + 2] == b"b"

        best_depth = -1
        best_score = 0
        best_move = b""
        pos = end
        while True:
            pos = line.find(_PVS, pos)
            if pos < 0:
                break
            head = pos + len(_PVS)
            depth_at = line.find(_DEPTH, head)
            if depth_at < 0:
                break
            depth_end = line.find(b"}", depth_at)
            depth = int(line[depth_at + len(_DEPTH) : depth_end])
            if depth > best_depth:
                if line.startswith(_CP, head):
                    value = int(line[head + len(_CP) : line.index(b",", head)])
                    value = max(-SCORE_CLAMP, min(SCORE_CLAMP, value))
                elif line.startswith(_MATE, head):
                    mate = int(line[head + len(_MATE) : line.index(b",", head)])
                    value = SCORE_CLAMP if mate > 0 else -SCORE_CLAMP
                else:
                    pos = head
                    continue
                best_depth = depth
                best_score = value
                line_at = line.find(_LINE, head)
                best_move = line[line_at + len(_LINE) : line_at + len(_LINE) + 5]
            pos = depth_end
        if best_depth < 0:
            no_score += 1
            continue
        if quiet_only and _is_capture(placement, best_move, black_to_move):
            noisy += 1
            continue

        try:
            occ, nib = pack_placement(placement.decode(), black_to_move)
        except (KeyError, ValueError):
            bad += 1
            continue
        occ_out[kept] = occ
        nib_out[kept] = np.frombuffer(nib, dtype=np.uint8)
        score_out[kept] = -best_score if black_to_move else best_score
        kept += 1
    return occ_out[:kept], nib_out[:kept], score_out[:kept], seen, no_score, bad, noisy


@numba.njit(cache=True)
def in_check_mask(occ: np.ndarray, nib: np.ndarray) -> np.ndarray:
    """True where the (canonical, i.e. white) side to move is in check."""
    out = np.zeros(occ.shape[0], dtype=np.bool_)
    state = np.zeros(fb.STATE_SIZE, dtype=np.uint64)
    one = np.uint64(1)
    for i in range(occ.shape[0]):
        for piece in range(12):
            state[piece] = np.uint64(0)
        bits = occ[i]
        count = 0
        while bits:
            square = fb.lsb_square(bits)
            bits &= bits - one
            byte = np.int64(nib[i, count >> 1])
            shift = 4 if count & 1 else 0
            code = (byte >> shift) & 15
            state[code] |= one << np.uint64(square)
            count += 1
        state[fb.ALL_OCC] = occ[i]
        out[i] = fb.is_in_check(state, fb.WHITE)
    return out


def iter_chunks(path: Path, chunk_bytes: int):
    import zstandard as zstd

    with path.open("rb") as handle:
        reader = zstd.ZstdDecompressor().stream_reader(handle)
        remainder = b""
        while True:
            try:
                block = reader.read(chunk_bytes)
            except zstd.ZstdError:
                # A byte-range prefix ends mid-frame; everything before the cut is still valid.
                block = b""
            if not block:
                if remainder:
                    yield remainder
                return
            block = remainder + block
            cut = block.rfind(b"\n")
            if cut < 0:
                remainder = block
                continue
            yield block[:cut]
            remainder = block[cut + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=20_000_000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument(
        "--quiet", action="store_true",
        help="drop positions whose best move is a capture or a promotion"
    )
    args = parser.parse_args()

    limit = args.max_positions
    records = np.empty(limit, dtype=RECORD_DTYPE)
    total = 0
    seen = 0
    no_score = 0
    bad = 0
    noisy = 0
    checks = 0
    start = time.perf_counter()

    in_check_mask(np.zeros(1, np.uint64), np.zeros((1, 16), np.uint8))  # compile once
    chunks = iter_chunks(args.input, args.chunk_mb * 1024 * 1024)
    with mp.Pool(args.workers) as pool:
        tasks = ((chunk, args.quiet) for chunk in chunks)
        for occ, nib, score, chunk_seen, chunk_none, chunk_bad, chunk_noisy in pool.imap(
            parse_chunk, tasks, chunksize=1
        ):
            seen += chunk_seen
            no_score += chunk_none
            bad += chunk_bad
            noisy += chunk_noisy
            if occ.size:
                keep = ~in_check_mask(occ, nib)
                checks += int(occ.size - keep.sum())
                occ, nib, score = occ[keep], nib[keep], score[keep]
            room = min(occ.size, limit - total)
            if room > 0:
                records["occ"][total : total + room] = occ[:room]
                records["nib"][total : total + room] = nib[:room]
                records["score"][total : total + room] = score[:room]
                total += room
            if total >= limit:
                pool.terminate()
                break
            if seen % 2_000_000 < 20_000:
                rate = total / max(1e-9, time.perf_counter() - start)
                print(f"  {total:,} kept / {seen:,} seen  ({rate:,.0f}/s)", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, records[:total])
    elapsed = time.perf_counter() - start
    print(
        f"lines seen        {seen:,}\n"
        f"no evaluation     {no_score:,}\n"
        f"unparsable        {bad:,}\n"
        f"best move a capture (dropped)   {noisy:,}\n"
        f"side to move in check (dropped)  {checks:,}\n"
        f"positions written {total:,} -> {args.output} "
        f"({args.output.stat().st_size / 1e6:.0f} MB) in {elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
