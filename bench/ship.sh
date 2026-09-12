#!/usr/bin/env bash
# Promote a candidate engine directory to engine/, snapshot it under versions/, run every gate,
# package and validate the zip.  Usage:  bench/ship.sh <candidate-dir> <version-name>
set -euo pipefail
cd "$(dirname "$0")/.."
CAND="${1:?usage: bench/ship.sh <candidate-dir> <version-name>}"
VER="${2:?usage: bench/ship.sh <candidate-dir> <version-name>}"
[ -f "$CAND/agent.py" ] || { echo "no agent.py in $CAND" >&2; exit 1; }
[ ! -e "versions/$VER" ] || { echo "versions/$VER already exists" >&2; exit 1; }

cp "$CAND"/*.py engine/
cp -R "$CAND/weights/." engine/weights/
mkdir -p "versions/$VER/weights"
cp "$CAND"/*.py "versions/$VER/"
cp -R engine/weights/. "versions/$VER/weights/"

uv run ruff check --fix --select I engine/agent.py >/dev/null || true
uv run ruff check engine bench tests harness training
uv run mypy
CHESSATHON_AGENT_DIR=engine uv run pytest -q -p no:cacheprovider tests
uv run python -c "import sys; sys.path.insert(0, 'engine'); import agent; assert agent.native_ready(), 'native engine not live'; print('native_ready OK')"
make zip
uv run python -m bench.validate_zip submission.zip | grep -E "ready|move:|native|Traceback|fallback|pyengine"
ls -la submission.zip
md5 -q submission.zip 2>/dev/null || md5sum submission.zip
