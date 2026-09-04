#!/usr/bin/env bash
# Build the reference engines into tools/engines/. Local sparring only; nothing here ships.
# Each step is independent; a failure is reported and the rest continue.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/tools/engines"
SRC="${ENGINE_SRC:-$ROOT/tools/src}"
mkdir -p "$OUT" "$SRC"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

step() { echo; echo "=== $1"; }

step "stockfish (homebrew)"
if command -v stockfish >/dev/null; then ln -sf "$(command -v stockfish)" "$OUT/stockfish"
elif command -v brew >/dev/null; then brew install stockfish && ln -sf "$(command -v stockfish)" "$OUT/stockfish"
else echo "no brew; install stockfish manually"; fi

step "shallow-blue 2.0.0 (house bot, CCRL 1576)"
if [ ! -d "$SRC/shallow-blue" ]; then git clone -q --depth 1 --branch v2.0.0 https://github.com/GunshipPenguin/shallow-blue "$SRC/shallow-blue"; fi
( cd "$SRC/shallow-blue" && make -j"$JOBS" >/dev/null && cp shallowblue "$OUT/shallow-blue" && echo built ) || echo "shallow-blue FAILED"

step "zagreus 5.0 (house bot, CCRL 2168)"
if [ ! -d "$SRC/zagreus" ]; then git clone -q --depth 1 --branch v5.0 https://github.com/Dannyj1/Zagreus "$SRC/zagreus"; fi
( cd "$SRC/zagreus" && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MARCH=OFF -DENABLE_MTUNE=OFF >/dev/null && cmake --build build -j"$JOBS" >/dev/null \
  && cp "$(find build -maxdepth 2 -type f -perm +111 -name 'Zagreus*' | head -1)" "$OUT/zagreus" && echo built ) || echo "zagreus FAILED"

step "loki 3.0.0 (house bot, CCRL 2428)"
if [ ! -d "$SRC/loki" ]; then git clone -q --depth 1 --branch v3.0.0 https://github.com/BimmerBass/Loki "$SRC/loki"; fi
( cd "$SRC/loki" && ls && ( [ -f CMakeLists.txt ] && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j"$JOBS" >/dev/null \
  && cp "$(find build -maxdepth 3 -type f -perm +111 -iname 'loki*' | head -1)" "$OUT/loki" || ( cd src 2>/dev/null || true; make -j"$JOBS" >/dev/null && cp "$(find . -maxdepth 2 -type f -perm +111 -iname 'loki*' | head -1)" "$OUT/loki" ) ) && echo built ) || echo "loki FAILED"

step "rustic alpha 3 (house bot, CCRL 1792) — needs cargo"
if ! command -v cargo >/dev/null; then echo "no cargo; skipping rustic (install rust via 'brew install rust' or rustup, then rerun)";
else
  if [ ! -d "$SRC/rustic" ]; then git clone -q --depth 1 --branch alpha-3.0.0 https://github.com/mvanthoor/rustic "$SRC/rustic" || git clone -q --depth 1 https://github.com/mvanthoor/rustic "$SRC/rustic"; fi
  ( cd "$SRC/rustic" && cargo build --release -q && cp "$(find target/release -maxdepth 1 -type f -perm +111 -name 'rustic*' | head -1)" "$OUT/rustic" && echo built ) || echo "rustic FAILED"
fi

step "summary"
ls -la "$OUT"
for e in "$OUT"/*; do [ -x "$e" ] && [ ! -d "$e" ] && printf '%-14s ' "$(basename "$e")" && (printf 'uci\nquit\n' | "$e" 2>/dev/null | grep -m1 '^id name' || echo "no uci id"); done
