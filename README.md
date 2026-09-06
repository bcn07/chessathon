# chessathon

An engine for [AI Chessathon](https://aichessathon.com), plus the tooling to know whether a
change made it stronger. Rules digest: [RULES.md](RULES.md). Two AI coding agents work here;
who owns what is in [CLAUDE.md](CLAUDE.md) and the top of [AGENTS.md](AGENTS.md).

```
agent.py          Claude's engine — the submission candidate at the root
codex/agent.py    Codex's engine — played as an opponent directory
opponents/        house bots and Stockfish-at-Elo as agent directories (binaries in tools/engines, local only)
condor/           job script, submit file and notes for the Imperial DoC Condor pool
versions/         frozen snapshots of agent.py, for "better than my last one" comparisons
bench/            gauntlet, speed, tactics; work on any directory holding an agent.py
tests/            contract tests: legal moves, edge cases, clock, protocol, repetition
harness/          the platform's runner, referee, clock and packaging (from the starter; do not edit)
baselines/        random, greedy, minimax, numba — the ladder from the starter repo
docs/             starter README and IDEAS.md, kept for reference
reference/        verbatim extracts of the official docs and rules pages
```

## Setup

```
make setup          # uv sync — Python 3.12 and the platform's five pinned packages
make test           # contract tests against agent.py
```

## Measuring

| Command | What it answers |
|---|---|
| `make gauntlet OPP=codex GAMES=40` | Who wins. Parallel games from `bench/book.json` (1,000 balanced positions taken from our own real-clock games; `--book classic` for the 24 curated lines), both colours per opening, Elo ± 95% CI scored by opening pairs, PGN to `gauntlet.pgn`. Fails if our agent crashed, flagged or moved illegally. |
| `make gauntlet OPP=versions/v1-nullmove GAMES=200 ` | Did this change help, against the last snapshot. A 3% change needs hundreds of games. |
| `make speed` | How deep in 2 s, and knps, on eight fixed positions. `--depth 5` for reproducible node counts. |
| `make tactics` | Tactical suite at 3 s per position: solved count and time to solve. Mates are checked by brute force at load. |
| `make snapshot NAME=v2` | Freeze `agent.py` as `versions/v2` before the next change. |
| `make play OPP=codex` | One game at the real 120 s + 0.5 s, printing what both engines wrote to stderr. |
| `make zip` | `submission.zip` with `agent.py` at the root. |

Any bench takes `AGENT=<dir>` to point at another engine: `make tactics AGENT=codex`.

### Calibrated opponents

The ladder's house bots are open-source UCI engines with published CCRL ratings, and Stockfish
can be pinned to an Elo. `make engines` builds them into `tools/engines/` (local only, never
shipped — the engine ban covers the submission, not sparring), and `opponents/<name>/` wraps
each as an agent directory:

| opponent | what it is | CCRL |
|---|---|---|
| `opponents/shallow-blue` | house bot, Shallow Blue 2.0 | 1576 |
| `opponents/rustic` | house bot, Rustic Alpha 3 (needs `cargo`) | 1792 |
| `opponents/zagreus` | house bot, Zagreus 5.0 | 2168 |
| `opponents/loki` | house bot, Loki 3.0 | 2428 |
| `opponents/stockfish-1400` … `-2600` | Stockfish with `UCI_LimitStrength` | as named |

`make gauntlet OPP=opponents/shallow-blue GAMES=50` is the closest thing to a ladder result we
can get locally. Real-clock games against these on the cluster are the reference measurement.

### Deciding A/B tests

Four tiers, cheapest first; a candidate that fails one does not go on.

1. **Laptop, minutes.** `ruff`, `mypy`, `make test`; `bench.speed --depth 8` node counts (identical
   counts prove a change did not alter the search); `make tactics`. Laptop timings are noise on a
   loaded machine — use node counts and the pool for anything timed.
2. **Screen, ~20 min on the pool.** 200 games at 10 s + 0.1 s vs the root, native-only
   (`env="CHESSATHON_NATIVE_ONLY=1 CHESSATHON_INIT_COMPILE_WAIT=58"`). Drops crashes and anything
   below about −10 Elo. A fast-clock result is never a verdict (the accumulator engine was +58 fast
   and −16 at the real clock).
3. **Decision, 40 min per wave.** `make condor-sprt AGENT=work/<cand> OPP=. TAG=<cand>-vs-root`:
   waves of 100 jobs × 4 games at 120 s + 0.5 s with pondering off (the platform suspends us between
   moves), openings continuing through the book, scored by opening pairs (pentanomial), GSPRT with
   H0 = −5 / H1 = +15 Elo, cap 1,600 games. Simulated: a real +20 is accepted 90% of the time in
   about 1,000 games, a neutral change 8%, a −20 regression never (rejected in ~700). Ship on
   "H1 accepted"; undecided at the cap keeps the incumbent unless the estimate is ≥ +10 and the
   candidate costs nothing extra. `bench.aggregate <files> --sprt -5 15` scores any finished run.
4. **Per shipped version.** `bench.validate_zip` from a clean unpack, then
   `make condor-calibrate TAG=v12.x`: 400 real-clock games each vs Loki, Zagreus and Stockfish at
   UCI_Elo 2600. The strength estimate comes from these, not from chaining self-play deltas, which
   overstate transfer to other engines.

`bench.build_book` regenerates the book from `results/condor/*/games_*.pgn` (one position per game
at ply 8–16, |score| ≤ 40 cp at 200 ms, duplicates dropped).

### Cluster

`condor/README.md` covers running hundreds of real-clock games on Imperial DoC's Condor pool:
`make condor-sync`, `make condor-gauntlet …`, `make condor-fetch TAG=…`.

The gauntlet's default clock is 10 s + 0.1 s so results arrive quickly; that favours speed over
depth compared with the real control, so confirm any close call with `--base-ms 120000
--increment-ms 500`. Local cores are also faster than the platform's one core, so depth reached
here is an upper bound.

## The engine (agent.py)

Alpha-beta over python-chess: iterative deepening, fail-soft negamax, transposition table that
survives between moves, null-move pruning, late move reductions, check extension, quiescence on
captures and queen promotions. Evaluation is material plus tapered piece-square tables, bishop
pair and tempo. Time is a soft/hard budget from the clock handed in; the game is rebuilt move by
move from the FENs so the search sees repetitions the referee would claim. A fallback returns a
legal move if anything throws.
