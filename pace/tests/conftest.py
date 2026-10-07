from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "src", ROOT / "pace/src", ROOT / "packages/openpi-client/src", ROOT / "pace/scripts"):
    sys.path.insert(0, str(source))
