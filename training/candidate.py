"""Turn a trained net into an engine candidate directory, the way v12.1/v12.2 were built.

    uv run python training/candidate.py --net training/data/nnue_gen34_h384.npz \
        --name v12-g34 [--bench]

Copies ``--base`` (default the v12.1 build) to ``work/<name>``, swaps ``weights/nnue.npz``, runs
``training/tempo_check.py`` inside the copy, sets ``MOVER_BIAS = round(net tempo - 10)`` in its
``nnue_eval.py`` (so the search sees PeSTO's 10 cp tempo), lints, and with ``--bench`` runs the
depth-8 speed bench and the 3 s tactics suite. Ends by printing the fast-SPRT submit line.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPO_CHECK = ROOT / "training/tempo_check.py"


def run(command: list[str], cwd: Path | None = None) -> str:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise SystemExit(f"failed ({completed.returncode}): {' '.join(command)}")
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, required=True, help="trained .npz from train.py")
    parser.add_argument("--name", required=True, help="directory name under work/")
    parser.add_argument("--base", type=Path, default=ROOT / "work/v12-h384")
    parser.add_argument(
        "--bench", action="store_true", help="also run bench.speed and bench.tactics"
    )
    args = parser.parse_args()

    dest = ROOT / "work" / args.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    shutil.copytree(args.base, dest, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(args.net, dest / "weights" / "nnue.npz")

    check = dest / "tempo_check.py"
    shutil.copy(TEMPO_CHECK, check)
    try:
        output = run([sys.executable, str(check)], cwd=dest)
    finally:
        check.unlink()
    match = re.search(r"net tempo\s+mean\s+(-?[\d.]+)", output)
    if not match:
        raise SystemExit("tempo_check.py printed no 'net tempo mean' line")
    tempo = float(match.group(1))
    bias = round(tempo - 10)

    eval_path = dest / "nnue_eval.py"
    source = eval_path.read_text()
    source, count = re.subn(
        r"^MOVER_BIAS = -?\d+\s+#.*$",
        f"MOVER_BIAS = {bias}  # measured {tempo:.1f} cp learned tempo, PeSTO has 10",
        source,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit(f"expected one MOVER_BIAS line in {eval_path}, found {count}")
    eval_path.write_text(source)
    print(f"{dest.relative_to(ROOT)}: net tempo {tempo:.1f} cp -> MOVER_BIAS = {bias}")

    run(["uv", "run", "ruff", "check", str(dest)])
    if args.bench:
        run(["uv", "run", "python", "-m", "bench.speed", "--agent", str(dest), "--depth", "8"])
        run(["uv", "run", "python", "-m", "bench.tactics", "--agent", str(dest), "--ms", "3000"])

    rel = dest.relative_to(ROOT)
    print(f"\nnext: a real-clock gauntlet, e.g. bench.gauntlet --agent {rel} --opponent engine")


if __name__ == "__main__":
    main()
