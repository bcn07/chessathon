# Competition rules — digest

A paraphrase of the AI Chessathon rules as they stood on finals day (12 September 2026). It is here so the
engine's constraints can be understood without leaving the repository; the organisers' pages are the
authority: [docs](https://aichessathon.com/docs) · [competition rules](https://aichessathon.com/terms).

## Interface

- Submission: a zip, at most 50 MB unzipped, with `agent.py` at the archive root (not inside a folder).
- `agent.py` exposes `get_move(fen: str, time_left_ms: int) -> str` and returns a legal UCI move.
- Fixed environment: Python 3.12, the standard library, and five pinned packages (torch 2.13.0+cpu,
  numpy 2.5.2, python-chess 1.11.2, onnxruntime 1.29.0, numba 0.67.0). Nothing else installs;
  a `requirements.txt` is ignored. Native binaries and compiled extensions in the zip are rejected.
- The zip is first on `sys.path`, so a file named after a module you import shadows it.

## Process model

- One fresh process per game. The init budget covers importing the agent: 90 s in the qualifier,
  30 s at the final. Anything deferred to the first `get_move` comes off the match clock.
- The process is **suspended while the opponent thinks**, so there is no pondering. One thread is fastest;
  extra threads share the single core.
- The referee claims threefold and fifty-move draws automatically. Games end on an illegal move, a
  crash, out-of-memory, a missed init budget, or the clock; a game still running at 600 plies is a draw.
- The runner talks JSON over stdin/stdout and redirects fd 1 to stderr before importing the agent, so
  `print` is safe. Up to 8 KB of your output (first 4 KB and last 4 KB) is kept in the per-game log.

## Match conditions

| | |
|---|---|
| Time control | 120 s + 0.5 s per move, per side, wall time |
| Hardware | one core of an AMD EPYC 9V74 (2.60 GHz), 2 GB RAM, no network, no GPU |
| Filesystem | read-only apart from 256 MB at `/tmp`, wiped between games |
| Processes | at most 128 processes and threads |
| Openings | every game starts from a curated, near-level opening position (set unpublished) |

## What may be shipped

- No third-party engines: no Stockfish, Lc0, Maia, wrappers, ports or translations. An engine you wrote
  yourself before the event is fine. AI assistance in writing the code is allowed if you can explain it.
- Any network shipped must be trained by the team. Starting from, fine-tuning or re-exporting a published
  chess network counts as shipping it. Training *data* is unrestricted, including engine-annotated positions.
- A shipped table may answer the **opening** (move number 20 or lower) or the **endgame** (at most 7 pieces
  including kings). A table that answers a middlegame position is a stored search and counts as an engine.
  Books and tablebases count against the 50 MB cap; `chess.polyglot` and `chess.syzygy` are available.
- Everything that runs must be readable Python source; obfuscated agents are disqualified. Finalists walk
  through how the agent was built and how any network was trained. Disqualification can be retroactive.

## Format

- Qualifier ladder 4–11 September with rated rounds every hour; up to 30 uploads per team per day, each
  validated by a build plus two smoke games. The ladder only *seeds* the final Swiss.
- Uploads closed 11 September 11:00; a 13-round Swiss over locked builds decided qualification by points
  (tie-breaks: Buchholz, head-to-head, earlier submission). Teams need a UK university student to enter it.
- London final 12 September at Encode Club: uploads reopened 10:30–14:00 with a 30 s init budget, practice
  rounds every ten minutes, then a knockout seeded from the Swiss (two positions per tie, played once
  with each colour; three in the semis, four in the final). Top seeds took a first-round bye.
- Prizes: £1,000 / £500 / £250, third being the losing semi-finalist with the better Swiss standing.

## Data and IP

Participants keep ownership of their agent code; entry grants the organisers only what is needed to run,
test, review and record the event. Match results, pairings and factual records may be published.
