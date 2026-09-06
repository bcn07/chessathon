"""Combine the --json outputs of many gauntlet runs (e.g. one per Condor job) into one result.

    uv run python -m bench.aggregate results/condor/<tag>/results_*.json [--sprt ELO0 ELO1]

Scores are summarised by opening pairs (pentanomial); with ``--sprt`` the GSPRT log-likelihood
ratio and its verdict are printed so a sequential pool run can decide whether to submit another
wave. Files written before the ``order`` field existed are paired in file order, which is exact
for pool jobs (one worker, games in schedule order).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.stats import SPRT_BOUND, llr, summarise, verdict


def collect(paths: list[str]) -> dict:
    results: list[float] = []
    order: list[int] = []
    white: list[float] = []
    black: list[float] = []
    terminations: dict[str, int] = {}
    plies: list[int] = []
    failures: list[str] = []
    tracebacks = {"agent": 0, "opponent": 0}
    header = None
    next_index = 0
    for path in sorted(paths):
        data = json.loads(Path(path).read_text())
        header = header or (data["agent"], data["opponent"], data["base_ms"], data["increment_ms"])
        games = data["results"]
        results += games
        if "order" in data and len(data["order"]) == len(games):
            order += data["order"]
        else:  # legacy file: consecutive games are the two halves of a pair
            order += list(range(next_index, next_index + len(games)))
        next_index = max(order, default=-1) + 1
        next_index += next_index % 2
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
    return {
        "header": header, "results": results, "order": order, "white": white, "black": black,
        "terminations": terminations, "plies": plies, "failures": failures,
        "tracebacks": tracebacks, "files": len(paths),
    }


def report(data: dict, sprt: tuple[float, float] | None = None) -> str | None:
    """Print the summary; return the SPRT verdict (None when undecided or not requested)."""
    agent, opponent, base_ms, increment_ms = data["header"]
    results, order = data["results"], data["order"]
    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.5)
    losses = len(results) - wins - draws
    print(
        f"{Path(agent).name} vs {Path(opponent).name}: {len(results)} games from {data['files']} "
        f"runs at {base_ms / 1000:g}s + {increment_ms / 1000:g}s"
    )
    print(f"+{wins} ={draws} -{losses}   {summarise(results, order)}")
    decided = None
    if sprt is not None:
        value = llr(results, order, *sprt)
        decided = verdict(value, *sprt)
        print(
            f"sprt H0 {sprt[0]:+g} / H1 {sprt[1]:+g}: llr {value:+.2f} (bounds ±{SPRT_BOUND:.2f}) "
            f"→ {decided or 'undecided'}"
        )
    white, black = data["white"], data["black"]
    print(
        f"as white {sum(white) / max(len(white), 1):.1%}, "
        f"as black {sum(black) / max(len(black), 1):.1%}"
    )
    if data["plies"]:
        print(f"average game length {sum(data['plies']) / len(data['plies']):.0f} plies")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(data["terminations"].items())))
    tracebacks = data["tracebacks"]
    print(
        f"tracebacks (crash → fallback move): agent {tracebacks['agent']}, "
        f"opponent {tracebacks['opponent']}"
    )
    if data["failures"]:
        print("FAILURES:\n  " + "\n  ".join(data["failures"]))
    return decided


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"))
    args = parser.parse_args()
    report(collect(args.paths), tuple(args.sprt) if args.sprt else None)


if __name__ == "__main__":
    main()
