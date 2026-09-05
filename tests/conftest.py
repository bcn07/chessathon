"""The contract tests exercise the repo-root engine by default. To test a candidate copy instead:

    CHESSATHON_AGENT_DIR=work/see uv run pytest -q -p no:cacheprovider tests

pytest puts the repo root first on sys.path (tests/ is a package), so PYTHONPATH alone cannot
redirect `import agent`; this hook loads the engine modules from the requested directory before
any test module imports them, and fails loudly if the wrong copy was picked up.
"""

import importlib
import os
import sys
from pathlib import Path

_agent_dir = os.environ.get("CHESSATHON_AGENT_DIR")
if _agent_dir:
    _path = Path(_agent_dir).resolve()
    if not (_path / "agent.py").is_file():
        raise RuntimeError(f"CHESSATHON_AGENT_DIR={_agent_dir}: no agent.py in {_path}")
    sys.path.insert(0, str(_path))
    for _name in ("agent", "nativesearch", "fastboard", "pyengine"):
        sys.modules.pop(_name, None)
    _agent = importlib.import_module("agent")
    if Path(_agent.__file__ or "").resolve().parent != _path:
        raise RuntimeError(f"imported {_agent.__file__}, not the copy in {_path}")
