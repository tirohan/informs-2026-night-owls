"""Backward-compatible entry point. Implementation: night_owls.sequence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from night_owls.config import SEQ_CONFIG  # noqa: E402
from night_owls.sequence import (  # noqa: E402
    BiMamba, build_sequences, chunked_scan, fit_predict, main, with_prior_channel,
)

if __name__ == "__main__":
    main()
