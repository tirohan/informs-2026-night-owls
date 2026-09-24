"""Write and check the organiser submission template."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from night_owls.config import FUTURE_START, HORIZONS

FUTURE_TS = pd.Timestamp(FUTURE_START)
TARGET_COLS = [f"osi_target_t{h:02d}h" for h in HORIZONS]


def write_submission(template: pd.DataFrame, test_x, curves: np.ndarray, path: Path | str) -> None:
    result = template.copy()
    positions = {int(f): i for i, f in enumerate(test_x.fips)}
    county_idx = np.array([positions[int(f)] for f in template.fipsCode])
    ts = pd.to_datetime(template.timestamp_et, format="mixed", errors="raise")
    origin = ((ts - FUTURE_TS) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    if not np.all((origin >= 0) & (origin < 144)):
        raise ValueError("Invalid submission origin timestamps.")
    for h, col in zip(HORIZONS, TARGET_COLS):
        target = origin + h
        valid = target < 144
        if not np.array_equal(result[col].notna().to_numpy(), valid):
            raise ValueError(f"Template tail mask mismatch for {col}.")
        result[col] = np.nan
        result.loc[valid, col] = curves[county_idx[valid], target[valid]]
        vals = result.loc[valid, col].to_numpy()
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError("Invalid predicted OSI values.")
    id_cols = [c for c in template.columns if c not in TARGET_COLS]
    pd.testing.assert_frame_equal(template[id_cols], result[id_cols])
    if len(result) != 9072:
        raise ValueError("Submission must have 9072 rows.")
    result.to_csv(path, index=False, float_format="%.10f")
    reloaded = pd.read_csv(path)
    if reloaded[TARGET_COLS].isna().sum().tolist() != [63, 378, 1512, 3024]:
        raise ValueError("NaN tails do not match the unscoreable horizon ends.")
    long = []
    for h, col in zip(HORIZONS, TARGET_COLS):
        valid = origin + h < 144
        long.append(pd.DataFrame({
            "county": template.loc[valid, "fipsCode"],
            "target": (origin + h)[valid],
            "pred": reloaded.loc[valid, col],
        }))
    if pd.concat(long).groupby(["county", "target"]).pred.nunique().max() != 1:
        raise ValueError("Cross-horizon inconsistency.")


def validate(template_path, prediction_path) -> dict:
    t = pd.read_csv(template_path)
    p = pd.read_csv(prediction_path)
    ids = [c for c in t.columns if c not in TARGET_COLS]
    if list(p.columns) != list(t.columns):
        raise ValueError("Column names or order differ from the template.")
    if len(p) != len(t):
        raise ValueError("Row count differs from the template.")
    if not p[ids].equals(t[ids]):
        raise ValueError("Identifier columns or row order were altered.")
    origin = ((pd.to_datetime(t.timestamp_et, format="mixed") - FUTURE_TS) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    long = []
    for h, c in zip(HORIZONS, TARGET_COLS):
        valid = origin + h < 144
        if not np.array_equal(p[c].notna().to_numpy(), valid):
            raise ValueError(f"{c}: NaN pattern differs from the unscoreable tail.")
        vals = p.loc[valid, c].to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError(f"{c}: non-finite or negative predictions.")
        long.append(pd.DataFrame({"county": t.loc[valid, "fipsCode"], "target": (origin + h)[valid], "pred": vals}))
    long = pd.concat(long)
    if long.groupby(["county", "target"]).pred.nunique().max() != 1:
        raise ValueError("Cross-horizon inconsistency: the same county-hour has different predicted values.")
    return {
        "passed": True, "rows": int(len(p)),
        "scoreable_cells": [int(v) for v in p[TARGET_COLS].notna().sum()],
        "unique_scored_county_hours": int(long.groupby(["county", "target"]).ngroups),
        "prediction_mean_by_horizon": [float(v) for v in p[TARGET_COLS].mean()],
        "prediction_max_by_horizon": [float(v) for v in p[TARGET_COLS].max()],
    }
