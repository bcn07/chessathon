"""Label-quality gate for the self-play data.

    uv run python training/datagen/gate.py --count 20000 --nodes 20000

Takes a random sample of quiet Lichess positions with Stockfish scores (training/data/
positions_quiet.npy, mover-relative cp), labels each with the same node-budget search that
training/datagen/selfplay.py uses, and reports the correlation with Stockfish next to PeSTO's static
evaluation on the same positions. The bar is a clear margin over PeSTO's 0.68 (target
≥ 0.85) before anything is trained on the self-play labels. Castling rights and en passant are
not in the records, so both evaluations see the position without them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_OPT", "3")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "training"))

import numpy as np  # noqa: E402

import fastboard as fb  # noqa: E402
import nativesearch as ns  # noqa: E402
from encoding import RECORD_DTYPE  # noqa: E402

SYMBOLS = "PNBRQKpnbrqk"
NO_LIMIT = 1 << 62
QUIET_MARGIN = 60


def record_fen(occ: int, nib: np.ndarray) -> str:
    """The canonical record is always 'white to move' (the packer flipped black-to-move boards)."""
    board = [""] * 64
    count = 0
    bits = int(occ)
    while bits:
        square = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        byte = int(nib[count >> 1])
        code = (byte >> 4) if count & 1 else (byte & 15)
        board[square] = SYMBOLS[code]
        count += 1
    ranks = []
    for rank in range(7, -1, -1):
        text = ""
        empty = 0
        for file in range(8):
            symbol = board[rank * 8 + file]
            if symbol:
                if empty:
                    text += str(empty)
                    empty = 0
                text += symbol
            else:
                empty += 1
        if empty:
            text += str(empty)
        ranks.append(text)
    return "/".join(ranks) + " w - - 0 1"


def search_score(searcher: ns.Searcher, state: np.ndarray, nodes: int) -> int:
    searcher.tt_depth.fill(-1)
    searcher.history.fill(0)
    searcher.killers.fill(0)
    searcher.stats[:] = 0
    best = 0
    for depth in range(1, ns.MAX_DEPTH + 1):
        searcher.stats[1] = 0
        limit = nodes if depth > 1 else NO_LIMIT
        score, _, aborted = searcher._iteration(state, depth, -ns.INF, ns.INF, 0, limit)
        if aborted:
            break
        best = int(score)
        if abs(best) >= ns.MATE_BOUND or searcher.stats[0] >= nodes:
            break
    return best


def quiescence(searcher: ns.Searcher, state: np.ndarray) -> int:
    searcher.stats[:] = 0
    return int(
        ns._quiesce(
            state,
            -ns.INF,
            ns.INF,
            0,
            searcher.rep_keys,
            0,
            searcher.tt_keys,
            searcher.tt_depth,
            searcher.tt_scores,
            searcher.tt_flags,
            searcher.tt_moves,
            searcher.tt_mask,
            searcher.killers,
            searcher.history,
            searcher.move_buffers,
            searcher.score_buffers,
            searcher.stats,
            NO_LIMIT,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "training/data/positions_quiet.npy")
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--nodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    records = np.load(args.data, mmap_mode="r")
    assert records.dtype == RECORD_DTYPE
    rng = np.random.default_rng(args.seed)
    order = np.sort(rng.choice(len(records), args.count, replace=False))
    occ = np.ascontiguousarray(records["occ"][order])
    nib = np.ascontiguousarray(records["nib"][order])
    target = np.ascontiguousarray(records["score"][order]).astype(np.float64)

    searcher = ns.Searcher()
    static = np.empty(args.count)
    qsearch = np.empty(args.count)
    label = np.empty(args.count)
    started = time.perf_counter()
    for i in range(args.count):
        state = fb.from_fen(record_fen(int(occ[i]), nib[i]))
        if fb.is_in_check(state, fb.WHITE):
            static[i] = qsearch[i] = label[i] = np.nan
            continue
        static[i] = ns.evaluate_state(state)
        qsearch[i] = quiescence(searcher, state)
        label[i] = search_score(searcher, state, args.nodes)
        if i % 2000 == 1999:
            rate = (i + 1) / (time.perf_counter() - started)
            print(f"{i + 1}/{args.count} positions, {rate:.1f}/s", flush=True)
    ok = ~np.isnan(label)
    clip = lambda x: np.clip(x, -2000, 2000)  # noqa: E731

    def report(name: str, mask: np.ndarray) -> None:
        n = int(mask.sum())
        c_static = np.corrcoef(clip(static[mask]), target[mask])[0, 1]
        c_label = np.corrcoef(clip(label[mask]), target[mask])[0, 1]
        mae_static = np.abs(clip(static[mask]) - target[mask]).mean()
        mae_label = np.abs(clip(label[mask]) - target[mask]).mean()
        print(
            f"{name:34} n={n:6d}  PeSTO corr {c_static:.4f} MAE {mae_static:5.1f}   "
            f"search({args.nodes} nodes) corr {c_label:.4f} MAE {mae_label:5.1f}"
        )

    report("all sampled (not in check)", ok)
    quiet = ok & (np.abs(qsearch - static) <= QUIET_MARGIN)
    report(f"our quiet filter (|qs-static|<={QUIET_MARGIN})", quiet)
    report("quiet and |SF| < 400", quiet & (np.abs(target) < 400))
    print(f"{time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
