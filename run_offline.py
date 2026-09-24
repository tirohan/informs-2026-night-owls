"""Backward-compatible entry point. Implementation: night_owls.offline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from night_owls.offline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
