"""Merge bench.book_gen parts into weights/book.json ({fen4: [move, cp, depth, ply]}).

    uv run python -m bench.book_merge results/book/book1 --out work/v13-book/weights/book.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    book: dict[str, list] = {}
    files = sorted(path for folder in args.parts for path in folder.glob("part_*.json"))
    for path in files:
        with open(path) as handle:
            part = json.load(handle)["book"]
        for fen, entry in part.items():
            # on a transposition keep the entry searched closer to its root (shallower ply)
            if fen not in book or entry[3] < book[fen][3]:
                book[fen] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(book, handle, separators=(",", ":"))
    by_ply = collections.Counter(entry[3] for entry in book.values())
    size = args.out.stat().st_size
    print(f"{len(files)} parts, {len(book)} positions, {size:,} bytes -> {args.out}")
    print("entries by ply from the root:", dict(sorted(by_ply.items())))
    cps = sorted(abs(entry[1]) for entry in book.values())
    print(f"|cp| median {cps[len(cps)//2]}, p90 {cps[int(len(cps)*0.9)]}, max {cps[-1]}")


if __name__ == "__main__":
    main()
