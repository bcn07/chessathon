SHELL := /bin/bash

.PHONY: setup test play arena gauntlet speed tactics snapshot zip gate engines games games-login games-watch

AGENT ?= engine
OPP   ?= baselines/minimax
GAMES ?= 40
MS    ?= 2000

setup:
	uv sync

test:
	CHESSATHON_AGENT_DIR=$(AGENT) uv run pytest -q tests

play:
	uv run python -m harness.play --white $(AGENT) --black $(OPP) $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --agent $(AGENT) --opponent $(OPP) --games $(GAMES)

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
	mkdir -p versions/$(NAME) && cp -R engine/*.py engine/weights versions/$(NAME)/
	@echo "versions/$(NAME) frozen"

zip:
	uv run python -c 'from pathlib import Path; from harness.package import build; print(build(Path("engine"), Path("submission.zip"), ("weights",)))'

gate:
	uv run ruff check .
	uv run mypy
	CHESSATHON_AGENT_DIR=engine uv run pytest -q tests
	uv run python -m bench.gauntlet --agent engine --opponent baselines/random --games 2 --base-ms 5000 --increment-ms 100

# reference engines for local sparring (house bots + Stockfish); never shipped
engines:
	tools/build_engines.sh

# platform games + agent logs -> reference/games, reference/platform-logs (tools/platform_sync.py)
games-login:
	python3 tools/platform_sync.py login

games:
	python3 tools/platform_sync.py sync

# background launchd job, every EVERY (default 10m); `make games-watch EVERY=off` removes it
EVERY ?= 10m
games-watch:
	$(if $(filter off,$(EVERY)),python3 tools/platform_sync.py uninstall-agent,python3 tools/platform_sync.py install-agent --every $(EVERY))
