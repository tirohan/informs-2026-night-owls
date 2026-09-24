"""Fixed challenge constants, model hyperparameters, and the seed schedule.

Every stochastic step uses BASE_SEED plus the offsets below. Do not retune these
inside a run; change them here and refit.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BASE_SEED", "START", "CUTOFF", "FUTURE_START", "HORIZONS", "FUTURE_HOURS", "LEAD",
    "LEAD_BUCKETS", "WEATHER", "OBSERVED", "MEMBERS", "N_FEATURES",
    "KINETICS_PARAMS", "DIRECT_PARAMS", "ONSET_PARAMS", "LGB_FIXED",
    "COUNTY_WEIGHT_ALPHA", "COUNTY_WEIGHT_BETA", "ONSET_OSI_MIN", "ONSET_MIN_COUNTIES",
    "DROPPED_FEATURE_BLOCKS",
    "ICING_T_MIN_C", "ICING_T_MAX_C", "ICING_RH_MIN", "GUST_EXCESS_MPH",
    "SEQ_CONFIG",
]

# Seed schedule (documented offsets, all deterministic):
#   outer county folds                         BASE_SEED
#   outer boosting fit, fold k                 BASE_SEED + k
#   inner county folds, outer fold k           BASE_SEED + 100 + k
#   inner boosting fit, outer k, inner i       BASE_SEED + 1000 + 10k + i
#   full-training boosting fit                 BASE_SEED
#   sequence outer fold k, seed s              BASE_SEED + 100k + s
#   sequence final fit, seed s                 BASE_SEED + 1000 + s
BASE_SEED = 42

START = "2026-03-11 00:00:00"
CUTOFF = "2026-03-13 23:00:00"
FUTURE_START = "2026-03-14 00:00:00"

HORIZONS = (1, 6, 24, 48)
FUTURE_HOURS = 144
LEAD = np.arange(1, 145, dtype=float)
# Trajectory index j (hours after March 14 00:00). Index 0 is never scored.
LEAD_BUCKETS = ((1, 13), (13, 49), (49, 144))

WEATHER = [
    "gust", "wind_speed_10m", "wind_dir_10m", "t2m", "d2m", "sp", "mslma", "blh", "tp", "rain",
    "csnow", "sdwe", "tcc", "lcc", "mcc", "hcc", "sdswrf", "direct_rad", "diffuse_rad", "r2",
    "vpd", "et0", "soil_moist",
]
OBSERVED = ["outageCount", "outage_pct", "P_t", "N_t", "D_t", "R_t"]

# Submitted tree stack. The onset member is an ablation only.
MEMBERS = ["kinetics_gbm", "direct_gbm"]
# 206 base features plus the three cutoff-neighbour features. Icing, slope/unrepaired, and the
# gust-excess ratio each raised t+24h and t+48h RMSE when added one block at a time.
N_FEATURES = 209
DROPPED_FEATURE_BLOCKS = ("icing", "slope_unrepaired", "gust_ratio")

KINETICS_PARAMS = dict(
    n_estimators=350, learning_rate=0.035, num_leaves=15, max_depth=5, min_child_samples=100,
    colsample_bytree=0.85, reg_lambda=10,
)
DIRECT_PARAMS = dict(
    n_estimators=800, learning_rate=0.02, num_leaves=31, max_depth=5, min_child_samples=60,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
)
# Same capacity as the kinetics member: the specialist sees fewer counties.
ONSET_PARAMS = dict(KINETICS_PARAMS)
LGB_FIXED = dict(objective="regression", verbosity=-1, deterministic=True, force_col_wise=True, max_bin=127)

# County reweighting moved the loss off the official pooled RMSE (about 1–2% worse at every
# horizon on the two-tree stack). Both coefficients stay at zero: every county has weight 1.
COUNTY_WEIGHT_ALPHA = 0.0
COUNTY_WEIGHT_BETA = 0.0
ONSET_OSI_MIN = 0.01
ONSET_MIN_COUNTIES = 25

ICING_T_MIN_C = -2.0
ICING_T_MAX_C = 1.0
ICING_RH_MIN = 85.0
GUST_EXCESS_MPH = 30.0

SEQ_CONFIG = dict(
    d_model=24, n_layers=2, d_state=8, d_conv=4, expand=2, dropout=0.1, lr=3e-3, weight_decay=1e-2,
    batch_size=64, max_epochs=120, patience=25, es_fraction=0.15, seeds=4,
)
