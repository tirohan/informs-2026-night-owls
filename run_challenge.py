"""Backward-compatible entry point. Implementation: night_owls.pipeline.

    python run_challenge.py --data-dir data --output-dir results
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from night_owls.config import (  # noqa: E402
    DIRECT_PARAMS, KINETICS_PARAMS, LEAD, LGB_FIXED, OBSERVED, ONSET_PARAMS, WEATHER,
)
from night_owls.cv import make_folds  # noqa: E402
from night_owls.features import Inputs, build_inputs, leakage_test, weather_transform  # noqa: E402
from night_owls.io import CUTOFF_TS as CUTOFF  # noqa: E402
from night_owls.io import read_frame, validate_frame  # noqa: E402
from night_owls.metrics import county_bootstrap, paired_bootstrap  # noqa: E402
from night_owls.pipeline import main  # noqa: E402
from night_owls.prior import estimate_half_life, restoration_prior  # noqa: E402
from night_owls.submission import write_submission  # noqa: E402

if __name__ == "__main__":
    main()
