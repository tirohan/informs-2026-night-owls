"""Backward-compatible entry point. Implementation: night_owls.submission."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from night_owls.submission import validate  # noqa: E402
from night_owls.validate_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
