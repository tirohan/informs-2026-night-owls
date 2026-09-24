"""Two LightGBM members on one causal panel: decay residual and direct level.

The onset member is fit only when ablations=True.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from night_owls.config import DIRECT_PARAMS, KINETICS_PARAMS, LGB_FIXED, ONSET_PARAMS
from night_owls.metrics import horizon_weights
from night_owls.prior import estimate_half_life, restoration_prior
from night_owls.weights import county_multipliers, lead_row_weights, onset_training_index


def _fit_lgb(x_rows, y_rows, weights, params, seed, threads):
    model = lgb.LGBMRegressor(**LGB_FIXED, **params, random_state=seed, n_jobs=threads)
    return model.fit(x_rows, y_rows, sample_weight=weights)


def fit_predict_members(x, y, train_idx, query, query_idx, seed, threads, ablations: bool = False):
    """Fit the submitted members on train_idx and predict query_idx.

    Lead 0 is excluded from training (it is never scored) and is still predicted so the
    trajectory is complete. County multipliers are 1. The onset member is an ablation.
    """
    train_idx = np.asarray(train_idx)
    query_idx = np.asarray(query_idx)
    lead_w = horizon_weights(normalize=True)[1:]
    multipliers = county_multipliers(x.last_osi[train_idx], x.net_flow[train_idx])
    weights = lead_row_weights(multipliers, lead_w)
    xt = x.panel[train_idx, 1:].reshape(-1, x.panel.shape[-1])
    xq = query.panel[query_idx].reshape(-1, query.panel.shape[-1])
    half_life = estimate_half_life(y[train_idx], x.last_osi[train_idx])
    prior_train = restoration_prior(x.last_osi[train_idx], half_life)
    prior_query = restoration_prior(query.last_osi[query_idx], half_life)
    out = {"half_life_h": half_life}

    kin = _fit_lgb(xt, (y[train_idx, 1:] - prior_train[:, 1:]).ravel(), weights, KINETICS_PARAMS, seed, threads)
    out["kinetics_gbm"] = np.maximum(prior_query + kin.predict(xq).reshape(-1, 144), 0)
    direct = _fit_lgb(xt, y[train_idx, 1:].ravel(), weights, DIRECT_PARAMS, seed, threads)
    out["direct_gbm"] = np.maximum(direct.predict(xq).reshape(-1, 144), 0)
    out["prior_only"] = np.maximum(prior_query, 0)
    out["mean_curve"] = np.tile(y[train_idx].mean(axis=0), (len(query_idx), 1))

    if ablations:
        onset_idx = onset_training_index(x.last_osi, x.net_flow, train_idx)
        onset_mask = np.isin(train_idx, onset_idx)
        xt_onset = x.panel[onset_idx, 1:].reshape(-1, x.panel.shape[-1])
        prior_onset = restoration_prior(x.last_osi[onset_idx], half_life)
        onset = _fit_lgb(
            xt_onset,
            (y[onset_idx, 1:] - prior_onset[:, 1:]).ravel(),
            lead_row_weights(multipliers[onset_mask], lead_w),
            ONSET_PARAMS, seed, threads,
        )
        out["onset_gbm"] = np.maximum(prior_query + onset.predict(xq).reshape(-1, 144), 0)
        out["n_onset_counties"] = int(len(onset_idx))
        prior24 = restoration_prior(x.last_osi[train_idx], 24.0)
        kin24 = _fit_lgb(xt, (y[train_idx, 1:] - prior24[:, 1:]).ravel(), weights, KINETICS_PARAMS, seed, threads)
        out["kinetics_gbm_hl24"] = np.maximum(
            restoration_prior(query.last_osi[query_idx], 24.0) + kin24.predict(xq).reshape(-1, 144), 0,
        )
        small = _fit_lgb(xt, y[train_idx, 1:].ravel(), weights, KINETICS_PARAMS, seed, threads)
        out["direct_gbm_small"] = np.maximum(small.predict(xq).reshape(-1, 144), 0)
        uniform = _fit_lgb(
            xt, y[train_idx, 1:].ravel(), np.repeat(multipliers, len(lead_w)), DIRECT_PARAMS, seed, threads,
        )
        out["direct_gbm_uniform_loss"] = np.maximum(uniform.predict(xq).reshape(-1, 144), 0)
        keep = np.array([
            (name.startswith("wx_") or name in (
                "lead_since_cutoff_h", "log_lead_since_cutoff", "log_customers_observed_median",
            )) and "unrepaired" not in name and not name.startswith("spatial_")
            for name in x.feature_names
        ]) & np.array([name != "wx_gust_x_observed_P" for name in x.feature_names])
        wx = _fit_lgb(xt[:, keep], y[train_idx, 1:].ravel(), weights, DIRECT_PARAMS, seed, threads)
        out["weather_only_gbm"] = np.maximum(wx.predict(xq[:, keep]).reshape(-1, 144), 0)
        out["direct_gbm_feature_importance"] = dict(zip(
            x.feature_names, direct.booster_.feature_importance("gain").tolist(),
        ))
        out["kinetics_gbm_feature_importance"] = dict(zip(
            x.feature_names, kin.booster_.feature_importance("gain").tolist(),
        ))
        out["onset_gbm_feature_importance"] = dict(zip(
            x.feature_names, onset.booster_.feature_importance("gain").tolist(),
        ))
    return out
