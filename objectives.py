"""Backward-compatible entry point. Implementation: night_owls.metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from night_owls.metrics import (  # noqa: E402
    FUTURE_HOURS, HORIZONS, LEAD_BUCKETS, apply_bucket_blend, fit_bucket_blend,
    fit_simplex_blend, horizon_scores, horizon_weights, self_test,
)

if __name__ == "__main__":
    self_test()
    print("Objective self-test passed.")
