"""Concatenate packed position sets into one training set with a result column.

    uv run python work/nnue/mix_sets.py --out work/nnue/data/positions_mix.npy \
        work/nnue/data/positions_lichess.npy work/nnue/data/positions_gen34567r.npy

Sets without a ``result`` field get result = -2 ("no game result"), which train.py treats as
eval-only rows when ``--wdl`` is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RECORD_WDL = np.dtype(
    [("occ", "<u8"), ("nib", "u1", (16,)), ("score", "<i2"), ("result", "i1")]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keep-clamped", action="store_true", help="keep rows at the +/-2000 mate clamp"
    )
    args = parser.parse_args()
    parts = []
    for path in args.sets:
        src = np.load(path, mmap_mode="r")
        if not args.keep_clamped:
            keep = np.abs(src["score"].astype(np.int32)) < 2000
            print(f"{path.name}: dropping {int((~keep).sum()):,} clamped rows")
            src = src[keep]
        out = np.empty(len(src), dtype=RECORD_WDL)
        for field in ("occ", "nib", "score"):
            out[field] = src[field]
        out["result"] = src["result"] if "result" in (src.dtype.names or ()) else -2
        print(f"{path.name}: {len(src):,} positions, result column: "
              f"{'present' if 'result' in (src.dtype.names or ()) else 'none (-2)'}")
        parts.append(out)
    combined = np.concatenate(parts)
    np.save(args.out, combined)
    print(f"wrote {args.out} ({len(combined):,} positions, {args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
