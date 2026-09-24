#!/usr/bin/env python3
"""Development experiments reported in the "What did not help" paragraph, made reproducible.

    python experiments.py --data-dir data --output-dir results        (~10 min on 2 cores)

Same county folds (seed 42) and same feature panel as run_challenge.py. Writes
results/experiments_cv_metrics.csv, results/experiments_paired_bootstrap.json and results/experiments_nested_stacks.json.
Variants: hurdle (classifier x positive-part regressor), Tweedie and log1p targets, five-seed averaging (all with the
direct member's configuration); a multi-output ExtraTrees trajectory model and a PCA nearest-analogue model, each also
evaluated as a third stacking member in the same nested procedure as the submitted two-member stack.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from objectives import HORIZONS, horizon_weights, horizon_scores, fit_bucket_blend, apply_bucket_blend
from run_challenge import (read_frame, validate_frame, build_inputs, make_folds, estimate_half_life, restoration_prior,
                           paired_bootstrap, KINETICS_PARAMS, DIRECT_PARAMS, LGB_FIXED, OBSERVED, CUTOFF)


def rows(panel, idx):
    return panel[idx, 1:].reshape(-1, panel.shape[-1])


def fit_lgb(x, y, w, params, seed, threads, **extra):
    p = dict(LGB_FIXED, **params, random_state=seed, n_jobs=threads); p.update(extra)
    return lgb.LGBMRegressor(**p).fit(x, y, sample_weight=w)


def signatures(df, x):
    """County signature for the trajectory-level models: static panel features + log history + 6-hourly weather means."""
    static_idx = [i for i, k in enumerate(x.feature_names) if k.startswith("hist_") or k.startswith("wx_") and ("_observed_" in k or "_future_" in k) or k == "log_customers_observed_median"]
    hist, wx = [], []
    for _, g in df.groupby("fipsCode", sort=True):
        g = g.reset_index(drop=True)
        o = g.loc[g.timestamp_et <= CUTOFF, OBSERVED]
        hist.append(np.log1p(100 * o[["P_t", "N_t", "D_t", "R_t"]].to_numpy()).ravel())
        w = g[["gust", "wind_speed_10m", "tp", "soil_moist", "mslma", "t2m"]].to_numpy().reshape(36, 6, -1).mean(axis=1)
        wx.append(w.ravel())
    hist, wx = np.asarray(hist), np.asarray(wx)
    return np.c_[x.panel[:, 0, static_idx], hist, wx], hist, wx


class Analog:
    """Fold-local PCA of history and weather signatures; inverse-distance average of the 8 nearest training trajectories."""
    def __init__(self, k=8, seed=42):
        self.k, self.seed = k, seed

    def fit(self, h, w, y):
        self.hs, self.ws = StandardScaler().fit(h), StandardScaler().fit(w)
        self.hp = PCA(n_components=min(10, len(h) - 1), random_state=self.seed).fit(self.hs.transform(h))
        self.wp = PCA(n_components=min(20, len(w) - 1), random_state=self.seed).fit(self.ws.transform(w))
        hz, wz = self.hp.transform(self.hs.transform(h)), self.wp.transform(self.ws.transform(w))
        self.hn, self.wn = np.sqrt(np.mean(np.sum(hz ** 2, 1))), np.sqrt(np.mean(np.sum(wz ** 2, 1)))
        self.z, self.y = np.c_[hz / self.hn, wz / self.wn], y.copy()
        return self

    def predict(self, h, w):
        z = np.c_[self.hp.transform(self.hs.transform(h)) / self.hn, self.wp.transform(self.ws.transform(w)) / self.wn]
        d = cdist(z, self.z, metric="sqeuclidean"); idx = np.argsort(d, 1)[:, :self.k]
        wts = 1 / np.maximum(np.take_along_axis(d, idx, 1), 1e-8); wts /= wts.sum(1, keepdims=True)
        return np.einsum("nk,nkt->nt", wts, self.y[idx])


def members(x, y, sig, hist, wx, tr, va, seed, threads, which):
    """Fit the requested members on tr and predict va. Returns dict name -> (len(va), 144)."""
    xt, xq = rows(x.panel, tr), x.panel[va].reshape(-1, x.panel.shape[-1])
    w = np.tile(horizon_weights(normalize=True)[1:], len(tr))
    yt = y[tr, 1:].ravel()
    out = {}
    if "kinetics_gbm" in which or "direct_gbm" in which:
        H = estimate_half_life(y[tr], x.last_osi[tr])
        if "kinetics_gbm" in which:
            pr = restoration_prior(x.last_osi[tr], H)
            m = fit_lgb(xt, (y[tr, 1:] - pr[:, 1:]).ravel(), w, KINETICS_PARAMS, seed, threads)
            out["kinetics_gbm"] = np.maximum(restoration_prior(x.last_osi[va], H) + m.predict(xq).reshape(-1, 144), 0)
        if "direct_gbm" in which:
            out["direct_gbm"] = np.maximum(fit_lgb(xt, yt, w, DIRECT_PARAMS, seed, threads).predict(xq).reshape(-1, 144), 0)
    if "hurdle" in which:
        clf = lgb.LGBMClassifier(**dict(LGB_FIXED, **DIRECT_PARAMS, objective="binary", random_state=seed, n_jobs=threads)).fit(xt, (yt > 0).astype(int), sample_weight=w)
        pos = yt > 0
        reg = fit_lgb(xt[pos], yt[pos], w[pos], DIRECT_PARAMS, seed, threads)
        out["hurdle"] = np.maximum(clf.predict_proba(xq)[:, 1] * reg.predict(xq), 0).reshape(-1, 144)
    if "tweedie" in which:
        out["tweedie"] = np.maximum(fit_lgb(xt, yt, w, DIRECT_PARAMS, seed, threads, objective="tweedie", tweedie_variance_power=1.5).predict(xq).reshape(-1, 144), 0)
    if "log1p" in which:
        m = fit_lgb(xt, np.log1p(100 * yt), w, DIRECT_PARAMS, seed, threads)
        out["log1p"] = np.maximum(np.expm1(m.predict(xq)) / 100, 0).reshape(-1, 144)
    if "seeds5" in which:
        out["seeds5"] = np.mean([np.maximum(fit_lgb(xt, yt, w, DIRECT_PARAMS, seed + 17 * i, threads).predict(xq).reshape(-1, 144), 0) for i in range(5)], axis=0)
    if "extra_trees" in which:
        scale = np.sqrt(horizon_weights(normalize=True)[1:])
        et = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.7, n_jobs=threads, random_state=seed).fit(sig[tr], y[tr, 1:] * scale)
        pp = et.predict(sig[va]) / scale
        out["extra_trees"] = np.maximum(np.c_[pp[:, 0], pp], 0)
    if "analog" in which:
        out["analog"] = np.maximum(Analog(seed=seed).fit(hist[tr], wx[tr], y[tr]).predict(hist[va], wx[va]), 0)
    out["mean_curve"] = np.tile(y[tr].mean(0), (len(va), 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True); ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = read_frame(a.data_dir / "DM_Train.csv"); validate_frame(train, False)
    x = build_inputs(train); y = train["osi"].to_numpy().reshape(-1, 216)[:, 72:]
    sig, hist, wx = signatures(train, x)
    county = train.drop_duplicates("fipsCode").set_index("fipsCode").loc[x.fips].reset_index()
    strata = (county.stateAbbr.astype(str) + "_" + county.severity_tier.astype(str)).to_numpy()
    folds = make_folds(strata, 5, a.seed)
    ALL = ["kinetics_gbm", "direct_gbm", "hurdle", "tweedie", "log1p", "seeds5", "extra_trees", "analog"]
    oof = {m: np.full_like(y, np.nan) for m in ALL + ["mean_curve"]}
    stacks = {"kinetics+direct": ["kinetics_gbm", "direct_gbm"], "+extra_trees": ["kinetics_gbm", "direct_gbm", "extra_trees"], "+analog": ["kinetics_gbm", "direct_gbm", "analog"]}
    stack_oof = {s: np.full_like(y, np.nan) for s in stacks}; weights_log = []
    with threadpool_limits(limits=a.threads):
        for k, (tr, va) in enumerate(folds):
            print(f"outer fold {k + 1}/5", flush=True)
            outer = members(x, y, sig, hist, wx, tr, va, a.seed + k, a.threads, ALL)
            for m, p in outer.items():
                oof[m][va] = p
            inner_members = ["kinetics_gbm", "direct_gbm", "extra_trees", "analog"]
            inner = {m: np.full((len(tr), 144), np.nan) for m in inner_members + ["mean_curve"]}
            for i, (itr, iva) in enumerate(make_folds(strata[tr], 3, a.seed + 100 + k)):
                ip = members(x, y, sig, hist, wx, tr[itr], tr[iva], a.seed + 1000 + k * 10 + i, a.threads, inner_members)
                for m in inner:
                    inner[m][iva] = ip[m]
            rec = {"outer_fold": k}
            for s, mem in stacks.items():
                bw = fit_bucket_blend(y[tr], [inner[m] for m in mem], inner["mean_curve"])
                stack_oof[s][va] = apply_bucket_blend(bw, [outer[m] for m in mem])
                rec[s] = [dict(zip(mem, np.round(b, 3).tolist())) for b in bw]
            weights_log.append(rec)
            print(f"  elapsed {time.time() - t0:.0f}s", flush=True)
    rows_ = []
    for name, p in list(oof.items()) + [(f"stack:{s}", p) for s, p in stack_oof.items()]:
        rows_.append({"model": name, **horizon_scores(y, p)})
    metrics = pd.DataFrame(rows_); metrics.to_csv(a.output_dir / "experiments_cv_metrics.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    comparisons = {"hurdle_vs_direct_gbm": paired_bootstrap(y, oof["hurdle"], oof["direct_gbm"]),
                   "tweedie_vs_direct_gbm": paired_bootstrap(y, oof["tweedie"], oof["direct_gbm"]),
                   "log1p_vs_direct_gbm": paired_bootstrap(y, oof["log1p"], oof["direct_gbm"]),
                   "seeds5_vs_direct_gbm": paired_bootstrap(y, oof["seeds5"], oof["direct_gbm"]),
                   "extra_trees_vs_direct_gbm": paired_bootstrap(y, oof["extra_trees"], oof["direct_gbm"]),
                   "analog_vs_direct_gbm": paired_bootstrap(y, oof["analog"], oof["direct_gbm"]),
                   "stack+extra_trees_vs_stack": paired_bootstrap(y, stack_oof["+extra_trees"], stack_oof["kinetics+direct"]),
                   "stack+analog_vs_stack": paired_bootstrap(y, stack_oof["+analog"], stack_oof["kinetics+direct"])}
    (a.output_dir / "experiments_paired_bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (a.output_dir / "experiments_nested_stacks.json").write_text(json.dumps(weights_log, indent=2))
    np.savez_compressed(a.output_dir / "experiments_oof.npz", truth=y, **oof, **{f"stack_{s}": p for s, p in stack_oof.items()})
    print(json.dumps({k: {h: round(v["prob_improvement"], 2) for h, v in c.items()} for k, c in comparisons.items()}, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
