# Training the evaluation network

The engine's evaluation is an NNUE trained from scratch here. Nothing in it derives from a published
network; the labels come from engines (which the competition rules allow for training data), the
positions come from public game databases and our own play.

## The final recipe (v12.25 network, shipped through v12.31)

| | |
|---|---|
| architecture | 768 piece-square inputs × 8 mirrored king buckets → 768 SCReLU units → 8 output buckets by piece count → 1 |
| quantisation | int16 accumulator, `QA = QB = 1024`, weight clip 1.0 so the accumulator bound stays under 32,767 for every legal placement (`train.py` refuses to export otherwise) |
| target | Stockfish evaluation blended with the game result, λ = 0.3, on rows that carry a result |
| data | 574 M positions: 269 M from the Lichess evaluation database (depth-20+ Stockfish labels, quiet positions only), 34 M of our own games relabelled by Stockfish at 20 k nodes, and Stockfish self-play kept to about 1:1 with the human rows |
| schedule | 6 epochs, batch 16,384, cosine schedule; about two minutes per epoch per 28 M rows on one A30 |
| learned tempo | measured on 60 k played-out positions and absorbed as a flat `MOVER_BIAS` (33 cp for the shipped net); a value much above 50 cp means self-play is over-represented and the net will not gain |

## Pipeline

| step | script |
|---|---|
| Lichess evaluation dump → packed positions | `parse_evals.py` |
| self-play generation, relabelling with Stockfish, set building and mixing | `datagen/selfplay.py`, `datagen/sfgen.py`, `datagen/relabel.py`, `datagen/build_set.py`, `mix_sets.py` |
| training and int16 export | `train.py` (single perspective, king buckets, output buckets, SCReLU); `train_dual.py`, `train2.py`, `train_kb32.py` are the dual-perspective, two-layer and 32-bucket variants, all measured and not shipped |
| put a net into an engine copy, measure tempo, run the benches | `candidate.py`, `tempo_check.py`, `bench_nnue.py` |
| feature encoding shared with `nnue_eval.py` | `encoding.py`, `verify_encoding.py` |

Data files live under `training/data/` and are not tracked. What the data and architecture decisions
measured in games, in short: label quality beat label volume every time, more of our own 20 k-node games
added nothing, width paid only once the data was there, and held-out fit stopped predicting strength at
the end.
