"""Run checkout entry points without depending on the caller's working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "src", ROOT / "pace/src", ROOT / "packages/openpi-client/src"):
    sys.path.insert(0, str(source))
