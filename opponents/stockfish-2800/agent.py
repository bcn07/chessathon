"""Sparring partner: stockfish-2800. Local measurement only, never shipped."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.uci_agent import make_get_move

get_move = make_get_move("tools/engines/stockfish", {"UCI_LimitStrength": True, "UCI_Elo": 2800})
