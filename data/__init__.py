"""Episode data backends."""
import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.append(str(_AGENT_DIR))
