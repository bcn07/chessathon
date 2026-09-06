"""Assemble a training set for work/nnue/train.py from self-play chunks.

    uv run python work/datagen/build_set.py --gens gen3 gen4 \
        --out work/nnue/data/positions_gen34.npy

Reads every ``chunk_*.bin`` under ``results/datagen/<gen>/`` (27-byte records written by
selfplay.py: occ, 16 nibbles, mover score, mover result), drops the result byte and the records
whose score hit the +/-2000 mate clamp (a static evaluation cannot learn those; 13% of gen1, 0.2%
of gen3/gen4), and saves the (occ, nib, score) records the trainer expects. Chunks are read in
numeric order. This reproduces the record set of ``positions_gen34.npy`` and friends (same
count and content; the earlier ad-hoc builds used another chunk order, which only changes
which rows train.py's seeded permutation holds out).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNK = np.dtype([("occ", "<u8"), ("nib", "u1", (16,)), ("score", "<i2"), ("result", "i1")])
RECORD = np.dtype([("occ", "<u8"), ("nib", "u1", (16,)), ("score", "<i2")])
MATE_CLAMP = 2000  # selfplay.py clamps mate scores to +/-2000; those records are dropped


def chunk_index(path: Path) -> int:
    match = re.search(r"chunk_(\d+)\.bin$", path.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gens", nargs="+", required=True, help="results/datagen/<gen> names")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datagen", type=Path, default=ROOT / "results/datagen")
    args = parser.parse_args()

    parts: list[np.ndarray] = []
    for gen in args.gens:
        chunks = sorted((args.datagen / gen).glob("chunk_*.bin"), key=chunk_index)
        if not chunks:
            raise SystemExit(f"no chunks under {args.datagen / gen}")
        count = 0
        for chunk in chunks:
            raw = np.fromfile(chunk, dtype=CHUNK)
            raw = raw[np.abs(raw["score"].astype(np.int32)) < MATE_CLAMP]
            records = np.empty(len(raw), dtype=RECORD)
            for field in RECORD.names:
                records[field] = raw[field]
            parts.append(records)
            count += len(raw)
        print(f"{gen}: {len(chunks)} chunks, {count:,} positions after the mate-clamp filter")
    combined = np.concatenate(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, combined)
    print(f"wrote {args.out} ({len(combined):,} positions, {args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
