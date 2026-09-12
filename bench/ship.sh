#!/bin/zsh
# Ship a candidate directory into the root: usage  ship.sh work/v13-tm30 v12.29-tm30
set -e
cd .
CAND="$1"; VER="$2"
[ -f "$CAND/agent.py" ] || { echo "no agent.py in $CAND"; exit 1; }
[ ! -e "versions/$VER" ] || { echo "versions/$VER exists"; exit 1; }
for f in "$CAND"/*.py; do cp "$f" "$(basename "$f")"; done
cp -R "$CAND/weights/." weights/
mkdir -p "versions/$VER/weights"
cp "$CAND"/*.py "versions/$VER/"
cp -R weights/. "versions/$VER/weights/"
uv run ruff check --fix --select I agent.py >/dev/null || true   # tablebase becomes first-party at the root
uv run ruff check ./*.py bench tests harness || exit 1   # work/ and versions/ hold historical snapshots
uv run mypy || exit 1
uv run pytest -q -p no:cacheprovider tests 2>&1 | tail -1
uv run python -c "import agent; assert agent.native_ready(), 'NATIVE ENGINE NOT LIVE'; print('native_ready OK')"
make zip
uv run python -m bench.validate_zip submission.zip 2>&1 | grep -E "ready|move:|native|Traceback|fallback|pyengine"
ls -la submission.zip; md5 -q submission.zip
