"""Import side effect: put training/ and engine/ on sys.path.

The trainers share ``encoding.py`` (here) with the shipped engine modules in ``engine/``
(``fastboard``, ``nnue_eval``, ``nativesearch``). Import this module before those imports.
"""

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
ROOT = TRAINING.parent
ENGINE = ROOT / "engine"

for _dir in (ENGINE, TRAINING):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
