"""Read and validate the organiser train and test frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from night_owls.config import CUTOFF, HORIZONS, OBSERVED, START, WEATHER

CUTOFF_TS = pd.Timestamp(CUTOFF)
START_TS = pd.Timestamp(START)


def read_frame(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep the supplied naive Eastern-time labels. Do not localise them.
    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"], format="mixed", errors="raise")
    return df.sort_values(["fipsCode", "timestamp_et"]).reset_index(drop=True)


def validate_frame(df: pd.DataFrame, is_test: bool) -> None:
    if df.duplicated(["fipsCode", "timestamp_et"]).any():
        raise ValueError("Duplicate county/hour keys.")
    grid = pd.date_range(START_TS, periods=216, freq="h").to_numpy()
    required = ["fipsCode", "timestamp_et", "customersTracked"] + OBSERVED + WEATHER
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for _, g in df.groupby("fipsCode", sort=True):
        if len(g) != 216 or not np.array_equal(g["timestamp_et"].to_numpy(), grid):
            raise ValueError("County must have the exact chronological 216-hour grid.")
    if not np.isfinite(df[WEATHER + ["customersTracked"]].to_numpy()).all():
        raise ValueError("This release should have complete weather and exposure values.")
    if (df["customersTracked"] <= 0).any():
        raise ValueError("Nonpositive customer denominator.")
    observed = df["timestamp_et"] <= CUTOFF_TS
    if df.loc[observed, OBSERVED].isna().any().any():
        raise ValueError("Missing observed-window outage input.")
    if is_test and df.loc[~observed, OBSERVED].notna().any().any():
        raise ValueError("Unexpected future test outage data: do not use it.")
    if not is_test:
        for h in HORIZONS:
            shifted = df.groupby("fipsCode")["osi"].shift(-h)
            if not np.allclose(shifted, df[f"osi_target_t{h:02d}h"], equal_nan=True, atol=1e-10):
                raise ValueError(f"Stored target does not equal shifted OSI for {h}h.")
