"""Validate a submission zip the way the platform does: unpack to an empty directory, start the
runner, time the ready line, play a few moves, show the agent's stderr.

    uv run python -m bench.validate_zip submission.zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = (
    ("rnbqkb1r/ppp2ppp/8/3p4/3Pn3/P4N2/1PP2PPP/RNBQKB1R w KQkq - 0 7", 120000),
    ("r1bqk1nr/pp3pbp/2np2p1/2p1p3/4P3/P1NP2PP/1PP2PB1/R1BQK1NR b KQkq - 0 7", 118000),
    ("8/5pk1/6p1/8/2R5/6P1/5PK1/3r4 w - - 0 40", 60000),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, nargs="?", default=ROOT / "submission.zip")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(directory)
            names = archive.namelist()
        unpacked = sum((Path(directory) / n).stat().st_size for n in names if not n.endswith("/"))
        print(f"{args.zip}: {len(names)} files, {unpacked:,} bytes unpacked: {', '.join(names)}")
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "harness" / "runner.py"), directory],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin and process.stdout and process.stderr
        started = time.perf_counter()
        ready = process.stdout.readline().strip()
        print(f"ready: {ready} after {time.perf_counter() - started:.1f}s")
        for fen, clock in POSITIONS:
            process.stdin.write(json.dumps({"fen": fen, "time_left_ms": clock}) + "\n")
            process.stdin.flush()
            started = time.perf_counter()
            line = process.stdout.readline().strip()
            print(f"move: {line} in {time.perf_counter() - started:.1f}s")
        process.kill()
        tail = process.stderr.read().strip().splitlines()
        print("stderr:\n  " + "\n  ".join(tail[-6:]))


if __name__ == "__main__":
    main()
