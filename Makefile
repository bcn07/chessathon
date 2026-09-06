SHELL := /bin/bash

.PHONY: setup test play arena gauntlet speed tactics snapshot zip gate engines

AGENT ?= .
OPP   ?= baselines/minimax
GAMES ?= 40
MS    ?= 2000

setup:
	uv sync

test:
	uv run pytest -q tests

play:
	uv run python -m harness.play --white . --black $(OPP) $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent $(OPP) --games $(GAMES)

# parallel games from varied openings, with an Elo estimate; OPP=versions/v1 to compare snapshots
gauntlet:
	uv run python -m bench.gauntlet --agent $(AGENT) --opponent $(OPP) --games $(GAMES) --pgn gauntlet.pgn

speed:
	uv run python -m bench.speed --agent $(AGENT) --ms $(MS)

tactics:
	uv run python -m bench.tactics --agent $(AGENT) --ms $(MS)

# freeze the current agent as an opponent: make snapshot NAME=v1
snapshot:
	@test -n "$(NAME)" || (echo "usage: make snapshot NAME=v1" && exit 1)
	mkdir -p versions/$(NAME) && cp agent.py versions/$(NAME)/agent.py
	@echo "versions/$(NAME)/agent.py frozen"

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run pytest -q tests
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

# --- Imperial DoC Condor pool; see condor/README.md --------------------------------------
CLUSTER      ?= shell3
CLUSTER_ROOT ?= /vol/bitbucket/USER/chessathon
CONDOR_ENV    = export PATH="$$PATH:$$HOME/.local/bin:/vol/condor/pool/doc/release/Ubuntu-24.04/X86_64.LINUX/bin"
JOBS ?= 10
BASE ?= 10000
INC  ?= 100
TAG  ?= run

.PHONY: condor-sync condor-setup condor-gauntlet condor-status condor-fetch condor-datagen

condor-sync:
	rsync -az --delete --exclude .venv --exclude .git --exclude __pycache__ --exclude '*.pgn' \
	  --exclude results --exclude .mypy_cache --exclude .ruff_cache --exclude .pytest_cache \
	  --exclude work/nnue/data --exclude 'work/*/codex.log' --exclude tools \
	  ./ $(CLUSTER):$(CLUSTER_ROOT)/

condor-setup:
	ssh $(CLUSTER) 'cd $(CLUSTER_ROOT) && $(CONDOR_ENV) && \
	  export UV_CACHE_DIR=/vol/bitbucket/USER/uv-cache UV_PYTHON_INSTALL_DIR=/vol/bitbucket/USER/uv-python && \
	  (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) && \
	  uv sync --frozen --no-dev && .venv/bin/python -c "import chess, numba, torch; print(\"cluster env ok\")"'

condor-gauntlet:
	ssh $(CLUSTER) 'cd $(CLUSTER_ROOT) && $(CONDOR_ENV) && mkdir -p results/condor/$(TAG) && \
	  condor_submit agent=$(AGENT) opp=$(OPP) games=$(GAMES) jobs=$(JOBS) base=$(BASE) inc=$(INC) tag=$(TAG) condor/gauntlet.submit'

# self-play data for Texel/NNUE: make condor-datagen GAMES=3000 JOBS=100 TAG=gen1
NODES ?= 20000
SEED_BASE ?= 1000
ENGINE ?= .
condor-datagen:
	ssh $(CLUSTER) 'cd $(CLUSTER_ROOT) && $(CONDOR_ENV) && mkdir -p results/datagen/$(TAG) && \
	  condor_submit games=$(GAMES) jobs=$(JOBS) nodes=$(NODES) seed_base=$(SEED_BASE) tag=$(TAG) engine=$(ENGINE) condor/datagen.submit'

# sequential real-clock A/B on the pool (waves of 400 games, GSPRT -5/+15, cap 1,600):
#   make condor-sprt AGENT=work/v12-x OPP=. TAG=x-vs-root   (run in the background; ~40 min per wave)
ELO0 ?= -5
ELO1 ?= 15
WAVES ?= 4
condor-sprt:
	uv run python -m bench.sprt_pool --agent $(AGENT) --opponent $(OPP) --tag $(TAG) \
	  --elo0 $(ELO0) --elo1 $(ELO1) --wave-jobs 100 --games 4 --max-waves $(WAVES) --base 120000 --inc 500

# external gauntlet for a shipped version: 400 real-clock games each vs Loki, Zagreus and
# Stockfish at UCI_Elo 2600 (make condor-calibrate TAG=v12.3); fetch with condor-fetch TAG=<tag>-<bot>
condor-calibrate:
	@test -n "$(TAG)" || (echo "usage: make condor-calibrate TAG=v12.3 [AGENT=.]" && exit 1)
	ssh $(CLUSTER) 'cd $(CLUSTER_ROOT) && $(CONDOR_ENV) && for o in loki zagreus stockfish-2600; do \
	  mkdir -p results/condor/$(TAG)-$$o && \
	  condor_submit agent=$(AGENT) opp=opponents/$$o games=4 jobs=100 base=120000 inc=500 tag=$(TAG)-$$o \
	    env="CHESSATHON_INIT_COMPILE_WAIT=58" condor/gauntlet.submit | tail -n 1; done'

condor-status:
	ssh $(CLUSTER) '$(CONDOR_ENV) && condor_q; condor_status -total | tail -3'

condor-fetch:
	mkdir -p results/condor
	rsync -az $(CLUSTER):$(CLUSTER_ROOT)/results/condor/$(TAG)/ results/condor/$(TAG)/
	uv run python -m bench.aggregate results/condor/$(TAG)/results_*.json

# reference engines for local sparring (house bots + Stockfish); never shipped
engines:
	tools/build_engines.sh
