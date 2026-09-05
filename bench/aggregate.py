"""Combine the --json outputs of many gauntlet runs (e.g. one per Condor job) into one result.

    uv run python -m bench.aggregate results/condor/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.gauntlet import summarise


def main(paths: list[str]) -> None:
    results: list[float] = []
    white: list[float] = []
    black: list[float] = []
    terminations: dict[str, int] = {}
    plies: list[int] = []
    failures: list[str] = []
    tracebacks = {"agent": 0, "opponent": 0}
    header = None
    for path in sorted(paths):
        data = json.loads(Path(path).read_text())
        header = header or (data["agent"], data["opponent"], data["base_ms"], data["increment_ms"])
        results += data["results"]
        white += data["white"]
        black += data["black"]
        plies += data["plies"]
        failures += [f"{Path(path).name}: {f}" for f in data["failures"]]
        for side, count in data.get("tracebacks", {}).items():
            tracebacks[side] += count
        for name, count in data["terminations"].items():
            terminations[name] = terminations.get(name, 0) + count
    if header is None:
        raise SystemExit("no result files")
    agent, opponent, base_ms, increment_ms = header
    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.5)
    losses = len(results) - wins - draws
    print(
        f"{Path(agent).name} vs {Path(opponent).name}: {len(results)} games from {len(paths)} runs "
        f"at {base_ms / 1000:g}s + {increment_ms / 1000:g}s"
    )
    print(f"+{wins} ={draws} -{losses}   {summarise(results)}")
    print(
        f"as white {sum(white) / max(len(white), 1):.1%}, "
        f"as black {sum(black) / max(len(black), 1):.1%}"
    )
    if plies:
        print(f"average game length {sum(plies) / len(plies):.0f} plies")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    print(
        f"tracebacks (crash → fallback move): agent {tracebacks['agent']}, "
        f"opponent {tracebacks['opponent']}"
    )
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main(sys.argv[1:])
