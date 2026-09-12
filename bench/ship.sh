#!/bin/zsh
# Ship a candidate directory into the root: usage  ship.sh work/v13-tm30 v12.29-tm30
set -e
cd .
CAND="$1"; VER="$2"
[ -f "$CAND/agent.py" ] || { echo "no agent.py in $CAND"; exit 1; }
[ ! -e "versions/$VER" ] || { echo "versions/$VER exists"; exit 1; }
for f in "$CAND"/*.py; do cp "$f" "engine/$(basename "$f")"; done
cp -R "$CAND/weights/." engine/weights/
mkdir -p "versions/$VER/weights"
cp "$CAND"/*.py "versions/$VER/"
cp -R engine/weights/. "versions/$VER/weights/"
uv run ruff check --fix --select I engine/agent.py >/dev/null || true
uv run ruff check engine bench tests harness training || exit 1   # versions/ holds historical snapshots
uv run mypy || exit 1
uv run pytest -q -p no:cacheprovider tests 2>&1 | tail -1
CHESSATHON_AGENT_DIR=engine uv run python -c "import sys; sys.path.insert(0, 'engine'); import agent; assert agent.native_ready(), 'NATIVE ENGINE NOT LIVE'; print('native_ready OK')"
make zip
uv run python -m bench.validate_zip submission.zip 2>&1 | grep -E "ready|move:|native|Traceback|fallback|pyengine"
ls -la submission.zip; md5 -q submission.zip
