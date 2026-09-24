#!/usr/bin/env python3
"""Data-shape diagnostics that fixed the modelling decisions (training counties only).

    python shape_diagnostics.py --data-dir data --results-dir results

Writes results/shape_diagnostics.json and prints the table used in the report. Every statistic is computed
from DM_Train.csv; the mean-variance slope uses out-of-fold residuals from results/development_oof.npz if present.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from run_challenge import estimate_half_life, make_folds, WEATHER

CUTOFF_INDEX = 72  # March 14 00:00 is row 72 of each county's 216-hour record
LEAD = np.arange(1, 145)


def reconstructed_osi(df):
    return np.maximum(.40 * df.P_t + .35 * df.N_t + .25 * df.D_t - .10 * df.R_t, 0).round(4)


def one_way_anova(values, groups):
    """Unequal-group, one-way random-effects ANOVA components.

    Counts weight both the grand mean and the between-group sum of squares.
    n_eff = (N - sum(n_i**2)/N)/(k-1), rather than the balanced-design N/k.
    ICC = sigma_between / (sigma_between + MS_within); a negative estimate
    is retained. eta_squared = SS_between / SS_total is reported separately.
    Formula cross-check: ICC package, ICCest.R, ANOVA variance components.
    https://github.com/matthewwolak/ICC/blob/master/R/ICCest.R
    """
    v = np.asarray(values, dtype=float)
    labels = np.asarray(groups)
    if v.ndim != 1 or labels.ndim != 1 or len(v) != len(labels):
        raise ValueError("Values and groups must be same-length one-dimensional arrays.")
    if not np.isfinite(v).all() or pd.isna(labels).any():
        raise ValueError("ANOVA inputs must not contain missing or non-finite values.")
    codes, levels = pd.factorize(labels, sort=True)
    N, k = len(v), len(levels)
    if k < 2 or N <= k:
        raise ValueError("At least two groups and positive within-group degrees of freedom are required.")
    counts = np.bincount(codes).astype(float)
    means = np.bincount(codes, weights=v) / counts
    grand = float(v.mean())
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(np.sum((v - means[codes]) ** 2))
    ss_total = ss_between + ss_within
    if ss_total <= 0:
        raise ValueError("ICC and eta-squared are undefined for a constant outcome.")
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    n_eff = float((N - np.sum(counts ** 2) / N) / (k - 1))
    var_between = (ms_between - ms_within) / n_eff
    denominator = var_between + ms_within
    if denominator == 0:
        raise ValueError("The estimated ICC denominator is zero.")
    return {"n_observations": N, "n_groups": k,
            "group_counts": {str(g): int(n) for g, n in zip(levels, counts)},
            "ss_between": ss_between, "ss_within": ss_within,
            "ms_between": ms_between, "ms_within": ms_within,
            "effective_group_size": n_eff,
            "variance_between": float(var_between), "variance_within": float(ms_within),
            "icc": float(var_between / denominator),
            "eta_squared": float(ss_between / ss_total)}


def icc_one_way(values, groups):
    """One-way ANOVA ICC, accounting for unequal group sizes."""
    return one_way_anova(values, groups)["icc"]


def cv_half_life_diagnostics(y, last, strata, manifest):
    """Reconstruct every fitted prior half-life using the actual training folds.

    This diagnostic never changes fitted models or their predictions. The same
    estimator and fold seeds as run_challenge.py are used, including inner folds.
    """
    seed = int(manifest["seed"])
    outer = make_folds(strata, int(manifest["folds"]), seed)
    records = []
    for k, (tr, _) in enumerate(outer):
        records.append({"fit": "outer", "outer_fold": k, "inner_fold": None,
                        "n_training_counties": len(tr),
                        "half_life_h": estimate_half_life(y[tr], last[tr])})
        if manifest.get("nested", True):
            for i, (itr, _) in enumerate(make_folds(strata[tr], int(manifest["inner_folds"]), seed + 100 + k)):
                selected = tr[itr]
                records.append({"fit": "inner", "outer_fold": k, "inner_fold": i,
                                "n_training_counties": len(selected),
                                "half_life_h": estimate_half_life(y[selected], last[selected])})
    summary = {}
    for kind in ("outer", "inner"):
        h = [r["half_life_h"] for r in records if r["fit"] == kind]
        if h:
            summary[kind] = {"n_fits": len(h), "minimum_h": float(min(h)), "maximum_h": float(max(h))}
    return {"base_seed": seed, "summary": summary, "records": records,
            "full_training_half_life_h": estimate_half_life(y, last)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    a = ap.parse_args()
    tr = pd.read_csv(a.data_dir / "DM_Train.csv", parse_dates=["timestamp_et"]).sort_values(["fipsCode", "timestamp_et"])
    te = pd.read_csv(a.data_dir / "DM_Test.csv", parse_dates=["timestamp_et"])
    n_c = tr.fipsCode.nunique()
    osi = tr.osi.to_numpy().reshape(n_c, 216)
    y = osi[:, CUTOFF_INDEX:]                       # (239, 144) post-cutoff trajectories
    obs = tr[tr.timestamp_et < "2026-03-14"]
    last = reconstructed_osi(obs.groupby("fipsCode").tail(1)).to_numpy()   # OSI at the cutoff hour
    state = tr.drop_duplicates("fipsCode").stateAbbr.to_numpy()
    out = {}

    # 1. availability structure in the test file
    te_out = te[["outageCount", "P_t", "N_t", "D_t", "R_t"]].notna().groupby(te.timestamp_et >= "2026-03-14").mean().mean(axis=1)
    out["availability"] = {"test_outage_share_present_before_cutoff": float(te_out.get(False, np.nan)),
                           "test_outage_share_present_after_cutoff": float(te_out.get(True, np.nan)),
                           "test_weather_missing_cells": int(te[WEATHER].isna().sum().sum())}

    # 2. marginal shape of the scored target (t+1h window: leads 1..143)
    Y = y[:, 1:].ravel(); pos = Y[Y > 0]
    out["marginal"] = {"zero_fraction": float(np.mean(Y == 0)), "scored_median": float(np.median(Y)), "positive_median": float(np.median(pos)),
                       "positive_p90": float(np.quantile(pos, .9)), "positive_p99": float(np.quantile(pos, .99)),
                       "max": float(Y.max()), "skewness": float(stats.skew(Y)), "excess_kurtosis": float(stats.kurtosis(Y)),
                       "share_of_sum_y2_in_top_1pct_cells": float(np.sort(Y ** 2)[::-1][: len(Y) // 100].sum() / np.sum(Y ** 2)),
                       "log10_positive_sd_decades": float(np.log10(pos).std())}

    # 3. what generates the zeros
    affected = last > .005          # "materially affected at the cutoff"; the March 11-12 record carries lingering pre-event outages
    zero_cells = y[:, 1:] == 0
    stay = []
    for i in np.where(affected)[0]:
        z = np.where(y[i, 1:] == 0)[0]
        if len(z):
            stay.append(np.mean(y[i, 1 + z[0]:] == 0))
    out["zeros"] = {"counties_with_cutoff_osi_above_0.005": int(affected.sum()),
                    "counties_with_cutoff_osi_exactly_zero": int((last == 0).sum()),
                    "share_of_zero_cells_from_counties_with_cutoff_osi_at_most_0.005": float(1 - zero_cells[affected].sum() / zero_cells.sum()),
                    "share_of_zero_cells_from_counties_with_cutoff_osi_exactly_zero": float(1 - zero_cells[last > 0].sum() / zero_cells.sum()),
                    "median_share_zero_after_first_zero_affected": float(np.median(stay))}

    # 4. mean-variance relation of out-of-fold residuals (if available)
    oof = a.results_dir / "development_oof.npz"
    if oof.exists():
        d = np.load(oof); pred = d["stack_bucket_nested"]; model_name = "stack_bucket_nested"
        seq = a.results_dir / "sequence_oof.npz"
        if seq.exists():                 # submitted model = equal-weight average of the stack and the sequence model (combine.py)
            s = np.load(seq)
            if np.array_equal(s["fips"], d["fips"]):
                pred = 0.5 * pred + 0.5 * s["oof"]; model_name = "final_equal_average"
        def mv_slope(pr):
            p = pr[:, 1:].ravel(); r = d["truth"][:, 1:].ravel() - p
            bins = np.quantile(p[p > 0], np.linspace(0, 1, 9)); idx = np.digitize(p, bins) - 1
            mv = np.array([(p[idx == k].mean(), r[idx == k].var()) for k in range(8) if (idx == k).sum() > 100])
            return float(np.polyfit(np.log(mv[:, 0]), np.log(mv[:, 1]), 1)[0])
        out["mean_variance"] = {"loglog_slope_residual_var_vs_pred_mean": mv_slope(pred), "model": model_name,
                                "loglog_slope_tree_stack": mv_slope(d["stack_bucket_nested"])}

    # 5. temporal dependence and the predictive reach of the cutoff state
    out["temporal"] = {f"acf_lag_{lag}h": float(np.corrcoef(osi[:, :-lag].ravel(), osi[:, lag:].ravel())[0, 1]) for lag in (1, 3, 6, 24, 48)}
    # index j of the trajectory is j+1 hours after the cutoff, so lead L hours is column L-1
    out["cutoff_state_r2_by_lead"] = {f"lead_{L}h": float(np.corrcoef(last, y[:, L - 1])[0, 1] ** 2) for L in (1, 6, 12, 24, 48, 96)}
    big = last > .02; hl, r2 = [], []
    for yy, l in zip(y[big], last[big]):
        res = minimize_scalar(lambda h: np.sum((yy[:48] - l * np.exp2(-LEAD[:48] / h)) ** 2), bounds=(1, 400), method="bounded")
        hl.append(res.x); r2.append(1 - res.fun / np.sum((yy[:48] - yy[:48].mean()) ** 2))
    norm24 = y[big, 23] / last[big]        # normalised OSI 24 hours after the cutoff (index 23)
    out["restoration_half_life_h"] = {"n_counties": int(big.sum()), "median": float(np.median(hl)),
                                      "q25": float(np.quantile(hl, .25)), "q75": float(np.quantile(hl, .75)),
                                      "median_r2_of_exponential_fit": float(np.median(r2)),
                                      "normalised_osi_at_24h_mean": float(norm24.mean()), "normalised_osi_at_24h_median": float(np.median(norm24))}

    # 6. shape of the response to the main driver
    g = tr.gust.to_numpy(); n = tr.N_t.to_numpy()
    out["gust_response_new_outage_rate_x1e3"] = {f"{lo}-{hi}mph": float(n[(g >= lo) & (g < hi)].mean() * 1e3) for lo, hi in
                                                 [(0, 20), (20, 30), (30, 40), (40, 50), (50, 80)]}

    # 6b. day of each county's OSI maximum (first occurrence), 9. counties still rising after the cutoff, 13. wave correlations
    peak_day = osi.argmax(1) // 24
    out["osi_peak_day_counts"] = {f"Mar{11 + k}": int((peak_day == k).sum()) for k in range(9)}
    cut_rows = obs.groupby("fipsCode").tail(1).set_index("fipsCode").loc[tr.fipsCode.unique()]
    net = (cut_rows.N_t - cut_rows.R_t).to_numpy()
    big = last > .02
    rising = big & (y[:, :12].max(1) > 1.2 * last)
    out["rising_after_cutoff"] = {"definition": "cutoff OSI > 0.02 and max OSI in the first 12 leads > 1.2 x cutoff OSI",
                                  "n_rising": int(rising.sum()), "n_rising_with_positive_net_inflow": int((net[rising] > 0).sum()),
                                  "share_positive_net_inflow_other_affected": float((net[big & ~rising] > 0).mean()),
                                  "mean_net_inflow_rising": float(net[rising].mean()), "mean_net_inflow_other_affected": float(net[big & ~rising].mean())}
    g = tr.gust.to_numpy().reshape(n_c, 216)
    w1, w2 = slice(48, 120), slice(120, 216)   # wave 1: Mar 13 00:00-Mar 15 23:00; wave 2: Mar 16 00:00-Mar 19 23:00
    p1, p2, g1, g2 = osi[:, w1].max(1), osi[:, w2].max(1), g[:, w1].max(1), g[:, w2].max(1)
    out["wave_peak_correlations"] = {"windows": "wave1 = Mar 13-15, wave2 = Mar 16-19 (same window for gust and OSI)",
                                     "corr_peak_gust_peak_osi_wave1": float(np.corrcoef(g1, p1)[0, 1]),
                                     "corr_peak_gust_peak_osi_wave2": float(np.corrcoef(g2, p2)[0, 1]),
                                     "corr_peak_osi_wave1_wave2": float(np.corrcoef(p1, p2)[0, 1]),
                                     "mean_peak_gust_mph_wave1": float(g1.mean()), "mean_peak_gust_mph_wave2": float(g2.mean())}

    # 7. heterogeneity across states, 8. effective sample for the severe tail
    peak_anova = one_way_anova(y.max(1), state)
    cutoff_anova = one_way_anova(last, state)
    out["heterogeneity"] = {"icc_state_post_cutoff_peak": peak_anova["icc"],
                             "icc_state_cutoff_osi": cutoff_anova["icc"],
                             "eta_squared_state_post_cutoff_peak": peak_anova["eta_squared"],
                             "method": "Unequal-group one-way ANOVA variance-component ICC; eta-squared is separate.",
                             "post_cutoff_peak_anova": peak_anova, "cutoff_osi_anova": cutoff_anova}
    manifest_path = a.results_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        county = tr.drop_duplicates("fipsCode")
        strata = (county.stateAbbr.astype(str) + "_" + county.severity_tier.astype(str)).to_numpy()
        out["half_life_cv"] = cv_half_life_diagnostics(y, last, strata, manifest)
    out["tail_sample"] = {"counties_with_post_cutoff_peak_over_0.1": int((y.max(1) > .1).sum()),
                          "top10_county_share_of_sum_y2": float(np.sort((y[:, 1:] ** 2).sum(1))[::-1][:10].sum() / (y[:, 1:] ** 2).sum())}

    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "shape_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
