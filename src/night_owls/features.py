"""Causal feature panels. Outage inputs are sliced to the cutoff before any summary.

Weather at any hour is used. Neighbor OSI is the cutoff value only (March 13 23:00),
never a post-cutoff outage. severity_tier, peak_pct, peak_customers, and
time_to_restore_h are not features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from night_owls.config import (
    DROPPED_FEATURE_BLOCKS, GUST_EXCESS_MPH, ICING_RH_MIN, ICING_T_MAX_C, ICING_T_MIN_C,
    LEAD, N_FEATURES, OBSERVED, WEATHER,
)
from night_owls.io import CUTOFF_TS

CENTROIDS = Path(__file__).resolve().parent / "data" / "county_centroids.csv"


@dataclass
class Inputs:
    fips: np.ndarray
    panel: np.ndarray            # (n, 144, n_features) float32
    feature_names: list
    last_osi: np.ndarray
    net_flow: np.ndarray | None = None


def weather_transform(g: pd.DataFrame) -> pd.DataFrame:
    w = g[WEATHER].reset_index(drop=True).copy()
    direction = np.deg2rad(w.pop("wind_dir_10m"))
    w["wind_sin"], w["wind_cos"] = np.sin(direction), np.cos(direction)
    w["t2m"] -= 273.15
    w["d2m"] -= 273.15
    w["dewpoint_spread"] = w["t2m"] - w["d2m"]
    return w


def reconstruct_osi(frame: pd.DataFrame) -> np.ndarray:
    """Official OSI: the formula rounded to 4 decimals, clipped at 0."""
    raw = 0.40 * frame.P_t + 0.35 * frame.N_t + 0.25 * frame.D_t - 0.10 * frame.R_t
    return np.round(np.maximum(np.asarray(raw, dtype=float), 0), 4)


def load_centroids(path: Path | None = None) -> pd.DataFrame:
    table = pd.read_csv(path or CENTROIDS)
    table["fips"] = table["fips"].astype(int)
    return table.set_index("fips")[["latitude", "longitude"]]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - lat1)
    dlmb = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cutoff_state(df: pd.DataFrame) -> pd.DataFrame:
    """One row per county: OSI and net flow at the last observed hour. Ignores later rows."""
    ordered = df.sort_values(["fipsCode", "timestamp_et"])
    last = ordered.loc[ordered["timestamp_et"] <= CUTOFF_TS].groupby("fipsCode", sort=True).tail(1)
    return pd.DataFrame({
        "cutoff_osi": reconstruct_osi(last),
        "net_flow": (last.N_t - last.R_t).to_numpy(dtype=float),
    }, index=last.fipsCode.astype(int).to_numpy())


def _spatial_lookup(fips_codes, context: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    state = cutoff_state(context).join(load_centroids(), how="left")
    if state[["latitude", "longitude"]].isna().any().any():
        missing = state.index[state.latitude.isna()].tolist()
        raise ValueError(f"Missing centroids for {missing[:5]}")
    lat = state.latitude.to_numpy()
    lon = state.longitude.to_numpy()
    osi = state.cutoff_osi.to_numpy()
    index = {int(f): i for i, f in enumerate(state.index)}
    out = {}
    for code in fips_codes:
        i = index[int(code)]
        dist = haversine_km(lat[i], lon[i], lat, lon)
        dist[i] = np.inf
        weights = 1.0 / np.maximum(dist, 1.0)
        weights[~np.isfinite(dist)] = 0.0
        nearest = int(np.argmin(dist))
        out[int(code)] = (
            float(np.sum(weights * osi) / np.sum(weights)),
            float(osi[nearest]),
            float(dist[nearest]),
        )
    return out


def build_inputs(df: pd.DataFrame, context: pd.DataFrame | None = None, drop_blocks: tuple[str, ...] = ()) -> Inputs:
    """Per-county panel of 144 forecast hours by N_FEATURES.

    `context`, when given, supplies neighbor cutoff OSI (train and test observed
    windows). Own history always comes from `df` after the cutoff slice.
    """
    spatial = _spatial_lookup(df["fipsCode"].unique(), context if context is not None else df)
    extra = set(drop_blocks)
    drop = set(DROPPED_FEATURE_BLOCKS) | extra
    panels, lasts, flows, fips = [], [], [], []
    names = None
    for county, g0 in df.groupby("fipsCode", sort=True):
        g = g0.reset_index(drop=True)
        o = g.loc[g["timestamp_et"] <= CUTOFF_TS, OBSERVED].copy().reset_index(drop=True)
        if len(o) != 72:
            raise ValueError(f"County {county} does not have 72 observed hours.")
        osi = pd.Series(reconstruct_osi(o))
        o["observed_osi"] = osi
        f: dict[str, np.ndarray] = {}

        def static(key, value):
            f[key] = np.full(144, float(value), dtype=float)

        for col in ["P_t", "observed_osi"]:
            v = o[col].to_numpy()
            for length in [3, 6, 12, 24, 48, 72]:
                seg = v[-length:]
                for fun in ["mean", "max", "std"]:
                    static(f"hist_{col}_{fun}_{length}", getattr(np, fun)(seg))
                static(f"hist_{col}_change_{length}", v[-1] - v[-length])
            for lag in [0, 1, 3, 6, 24, 48]:
                static(f"hist_{col}_cutoff_lag_{lag}", v[-1 - lag])
            static(f"hist_{col}_zero_fraction", np.mean(v == 0))
            static(f"hist_{col}_hours_since_peak", 71 - int(np.argmax(v)))
        for col in ["N_t", "R_t", "D_t"]:
            v = o[col].to_numpy()
            for length in [6, 24, 72]:
                static(f"hist_{col}_mean_{length}", v[-length:].mean())
            static(f"hist_{col}_last", v[-1])
        customer = np.log1p(g.loc[:71, "customersTracked"].median())
        static("log_customers_observed_median", customer)
        net = float(o.N_t.iloc[-1] - o.R_t.iloc[-1])
        static("hist_net_flow_last", net)
        osi_v = osi.to_numpy(dtype=float)
        peak_obs = float(osi_v.max())
        unrepaired = float(osi_v[-1] / (peak_obs + 1e-4))
        if "slope_unrepaired" not in drop:
            static("hist_osi_slope_3h", (osi_v[-1] - osi_v[-4]) / 3.0)
            static("hist_osi_slope_6h", (osi_v[-1] - osi_v[-7]) / 6.0)
            static("hist_observed_osi_peak", peak_obs)
            static("hist_unrepaired_fraction", unrepaired)
            static("hist_rising_at_cutoff", float(net > 0 or osi_v[-1] > osi_v[-7]))
        f["lead_since_cutoff_h"] = LEAD
        f["log_lead_since_cutoff"] = np.log1p(LEAD)
        for half_life in [6, 12, 24, 48]:
            f[f"past_osi_decay_hl{half_life}"] = osi_v[-1] * np.exp2(-LEAD / half_life)
        w = weather_transform(g)
        for col in w:
            f[f"wx_{col}"] = w[col].to_numpy()[72:]
        for col in ["gust", "wind_speed_10m", "tp", "rain", "soil_moist", "mslma"]:
            for length in [3, 6, 12, 24, 48]:
                f[f"wx_{col}_past_mean_{length}"] = w[col].rolling(length, min_periods=1).mean().to_numpy()[72:]
        for col in ["gust", "wind_speed_10m"]:
            for length in [6, 12, 24, 48]:
                f[f"wx_{col}_past_max_{length}"] = w[col].rolling(length, min_periods=1).max().to_numpy()[72:]
            for lag in [1, 3, 6, 24, 48]:
                f[f"wx_{col}_lag_{lag}"] = w[col].shift(lag).to_numpy()[72:]
        for length in [1, 3, 6, 24]:
            f[f"wx_gust_forward_max_{length}"] = np.array(
                [w.gust.iloc[t:min(t + length + 1, 216)].max() for t in range(72, 216)]
            )
            f[f"wx_forward_coverage_{length}"] = np.minimum(length, 215 - np.arange(72, 216))
        for threshold in [30, 40, 50]:
            exposure = np.maximum(w.gust.to_numpy() - threshold, 0)
            for length in [6, 24, 48]:
                f[f"wx_gust_excess_{threshold}_sum_{length}"] = (
                    pd.Series(exposure).rolling(length, min_periods=1).sum().to_numpy()[72:]
                )
            f[f"wx_gust_excess_{threshold}_since_cutoff"] = np.cumsum(exposure[72:])
        f["wx_gust_x_soil"] = w.gust.to_numpy()[72:] * w.soil_moist.to_numpy()[72:]
        f["wx_gust_x_observed_P"] = w.gust.to_numpy()[72:] * float(o.P_t.iloc[-1])
        observed_gust_max = float(w.gust.iloc[:72].max())
        static("wx_observed_gust_max", observed_gust_max)
        static("wx_future_gust_max", w.gust.iloc[72:].max())
        f["wx_gust_ratio_observed_peak"] = w.gust.to_numpy()[72:] / max(observed_gust_max, 1.0)
        for col in ["gust", "wind_speed_10m", "tp", "soil_moist", "mslma", "t2m"]:
            for segname, seg in [("observed", w[col].iloc[:72]), ("future", w[col].iloc[72:])]:
                static(f"wx_{col}_{segname}_mean", seg.mean())
                static(f"wx_{col}_{segname}_max", seg.max())
        gust = w.gust.to_numpy(dtype=float)
        if "gust_ratio" not in drop:
            observed_excess = float(np.maximum(gust[:72] - GUST_EXCESS_MPH, 0).sum())
            static("wx_observed_gust_excess_30", observed_excess)
            f["wx_gust_excess_ratio_30"] = np.cumsum(np.maximum(gust[72:] - GUST_EXCESS_MPH, 0)) / (observed_excess + 1.0)
        if "slope_unrepaired" not in drop:
            f["wx_gust_x_unrepaired"] = gust[72:] * unrepaired
            f["wx_gust_x_unrepaired_x_soil"] = gust[72:] * unrepaired * w.soil_moist.to_numpy()[72:]
        if "icing" not in drop:
            icing = (
                (w.t2m.to_numpy() >= ICING_T_MIN_C) & (w.t2m.to_numpy() <= ICING_T_MAX_C)
                & (w.r2.to_numpy() >= ICING_RH_MIN) & (w.rain.to_numpy() > 0)
            ).astype(float)
            static("wx_icing_hours_observed", icing[:72].sum())
            static("wx_icing_hours_future", icing[72:].sum())
            f["wx_icing_hour"] = icing[72:]
            f["wx_icing_forward_6h"] = np.array([icing[t:t + 6].sum() for t in range(72, 216)])
        if "spatial" not in drop:
            idw, nearest_osi, nearest_km = spatial[int(county)]
            static("spatial_idw_cutoff_osi", idw)
            static("spatial_nearest_cutoff_osi", nearest_osi)
            static("spatial_nearest_km", nearest_km)
        current_names = list(f)
        if names is None:
            names = current_names
        if names != current_names:
            raise ValueError(f"Feature columns drifted at county {county}.")
        panel = np.column_stack(list(f.values()))
        if not np.isfinite(panel).all():
            raise ValueError(f"Non-finite engineered feature for county {county}.")
        fips.append(int(county))
        panels.append(panel)
        lasts.append(float(osi_v[-1]))
        flows.append(net)
    if names is None or (not extra and len(names) != N_FEATURES):
        raise ValueError(f"Expected {N_FEATURES} features, built {0 if names is None else len(names)}.")
    return Inputs(
        np.asarray(fips), np.asarray(panels, dtype=np.float32), names,
        np.asarray(lasts, dtype=float), np.asarray(flows, dtype=float),
    )


def _poison_frame(df: pd.DataFrame) -> pd.DataFrame:
    poisoned = df.copy()
    future = poisoned.timestamp_et > CUTOFF_TS
    forbidden = [
        c for c in poisoned
        if c in list(OBSERVED) + ["osi"] or str(c).startswith(("osi_target_", "osi_delta_", "osi_lag", "outage_pct_lag"))
    ]
    poisoned.loc[future, forbidden] = 987654
    for c in ["peak_pct", "peak_customers", "time_to_restore_h", "event_duration_h", "severity_tier"]:
        if c in poisoned:
            poisoned[c] = 987654.321
    return poisoned


def leakage_test(df: pd.DataFrame, original: Inputs, context: pd.DataFrame | None = None,
                 drop_blocks: tuple[str, ...] = ()) -> dict:
    """Scramble post-cutoff outages and whole-event columns on the county frame and on context.

    Neighbour features are built from context, so context is poisoned too. The feature tensor,
    cutoff OSI, and cutoff net flow must be bitwise unchanged.
    """
    poisoned_context = _poison_frame(context) if context is not None else None
    rebuilt = build_inputs(_poison_frame(df), context=poisoned_context, drop_blocks=drop_blocks)
    for field in ["panel", "last_osi", "net_flow"]:
        if not np.array_equal(getattr(original, field), getattr(rebuilt, field)):
            raise AssertionError(f"Forbidden future data changed feature field {field}.")
    return {
        "passed": True,
        "test": "Scrambling future outage fields and whole-event columns on the county frame and on the neighbour context leaves all features bitwise unchanged.",
    }
