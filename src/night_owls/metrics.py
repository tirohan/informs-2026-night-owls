"""Official horizon RMSE, the horizon-balanced training weights, and convex stacking."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from night_owls.config import FUTURE_HOURS, HORIZONS, LEAD_BUCKETS

__all__ = [
    "HORIZONS", "FUTURE_HOURS", "LEAD_BUCKETS", "horizon_weights", "horizon_scores",
    "fit_simplex_blend", "fit_bucket_blend", "apply_bucket_blend", "county_bootstrap",
    "paired_bootstrap", "self_test",
]


def horizon_weights(n_hours: int = FUTURE_HOURS, normalize: bool = False) -> np.ndarray:
    """Per-lead weights such that sum_j w_j * e_j^2 equals the mean of the four horizon MSEs.

    Lead j is scored by horizon h iff j >= h. w_0 = 0. Un-normalized weights sum to one.
    """
    if n_hours <= max(HORIZONS):
        raise ValueError("n_hours must exceed the largest horizon.")
    j = np.arange(n_hours)
    w = np.mean([(j >= h) / (n_hours - h) for h in HORIZONS], axis=0)
    if normalize:
        w = w / w[w > 0].mean()
    return w


def _check_arrays(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != FUTURE_HOURS:
        raise ValueError("Expected matching (n_counties, 144) arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Scores require complete finite trajectory arrays.")
    return y, p


def horizon_scores(y, p) -> dict:
    """Official per-horizon pooled RMSE (cells j >= h) plus two team-level summaries."""
    y, p = _check_arrays(y, p)
    e2 = (p - y) ** 2
    result = {f"rmse_t{h:02d}h": float(np.sqrt(e2[:, h:].mean())) for h in HORIZONS}
    result["mean_horizon_rmse"] = float(np.mean([result[f"rmse_t{h:02d}h"] for h in HORIZONS]))
    result["horizon_balanced_mse"] = float(np.mean(e2 @ horizon_weights()))
    return result


def fit_simplex_blend(y, predictions, reference, shrinkage: float = 0.01, mask=None):
    """Non-negative, sum-to-one stacking weights. Pass out-of-fold predictions only."""
    if not predictions or shrinkage < 0:
        raise ValueError("Need at least one model and nonnegative shrinkage.")
    for p in predictions:
        _check_arrays(y, p)
    _check_arrays(y, reference)
    stack = np.stack(predictions, axis=-1)
    m = stack.shape[-1]
    a0 = np.full(m, 1 / m)
    sel = np.ones_like(y, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    b = np.array([np.sqrt(np.mean(((reference[:, h:] - y[:, h:]) * sel[:, h:]) ** 2)) for h in HORIZONS])
    b = np.maximum(b, 1e-12)

    def value_grad(a):
        pred = stack @ a
        value, gradient = 0.0, np.zeros(m)
        for k, h in enumerate(HORIZONS):
            residual = (pred[:, h:] - y[:, h:]) * sel[:, h:]
            rmse = np.sqrt(np.mean(residual ** 2) + 1e-24)
            value += rmse / (len(HORIZONS) * b[k])
            gradient += np.mean(residual[..., None] * stack[:, h:, :], axis=(0, 1)) / (len(HORIZONS) * b[k] * rmse)
        value += shrinkage * np.sum((a - a0) ** 2)
        gradient += 2 * shrinkage * (a - a0)
        return float(value), gradient

    result = minimize(
        value_grad, a0, jac=True, method="SLSQP", bounds=[(0, 1)] * m,
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1, "jac": lambda a: np.ones_like(a)},
        options={"maxiter": 300, "ftol": 1e-11},
    )
    if not result.success:
        raise RuntimeError(f"Blend optimisation failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    weights /= weights.sum()
    return weights, {
        "converged": bool(result.success), "objective": float(result.fun),
        "iterations": int(result.nit), "shrinkage": float(shrinkage), "reference_rmse": b.tolist(),
    }


def fit_bucket_blend(y, predictions, reference, buckets=LEAD_BUCKETS, shrinkage: float = 0.01):
    """One simplex weight vector per lead bucket."""
    weights = []
    for lo, hi in buckets:
        mask = np.zeros_like(y, dtype=bool)
        mask[:, lo:hi] = True
        w, _ = fit_simplex_blend(y, predictions, reference, shrinkage=shrinkage, mask=mask)
        weights.append(w)
    return np.asarray(weights)


def apply_bucket_blend(weights, predictions, buckets=LEAD_BUCKETS):
    out = np.zeros_like(predictions[0], dtype=float)
    for (lo, hi), w in zip(buckets, weights):
        out[:, lo:hi] = sum(wk * p[:, lo:hi] for wk, p in zip(w, predictions))
    out[:, 0] = out[:, 1]
    return out


def county_bootstrap(y, p, seed: int = 42, reps: int = 2000):
    """County-level bootstrap of pooled RMSE per horizon, conditional on fixed predictions."""
    rng = np.random.default_rng(seed)
    county_mse = np.column_stack([np.mean((p[:, h:] - y[:, h:]) ** 2, axis=1) for h in HORIZONS])
    draws = np.sqrt(np.mean(county_mse[rng.integers(0, len(y), size=(reps, len(y)))], axis=1))
    return {
        f"t{h:02d}h": {
            "lower_95": float(np.quantile(draws[:, k], 0.025)),
            "upper_95": float(np.quantile(draws[:, k], 0.975)),
        }
        for k, h in enumerate(HORIZONS)
    }


def paired_bootstrap(y, p_new, p_base, seed: int = 42, reps: int = 3000):
    """County-level paired bootstrap of RMSE(new) - RMSE(base) per horizon."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {}
    for h in HORIZONS:
        a = ((p_new[:, h:] - y[:, h:]) ** 2).mean(1)
        b = ((p_base[:, h:] - y[:, h:]) ** 2).mean(1)
        idx = rng.integers(0, n, (reps, n))
        diff = np.sqrt(a[idx].mean(1)) - np.sqrt(b[idx].mean(1))
        out[f"t{h:02d}h"] = {
            "mean_diff": float(diff.mean()),
            "lower_95": float(np.quantile(diff, 0.025)),
            "upper_95": float(np.quantile(diff, 0.975)),
            "prob_improvement": float((diff < 0).mean()),
        }
    return out


def self_test() -> None:
    rng = np.random.default_rng(42)
    y = rng.random((5, 144))
    p = rng.random((5, 144))
    expanded = np.mean([np.mean((p[:, h:] - y[:, h:]) ** 2) for h in HORIZONS])
    compact = np.mean((p - y) ** 2 @ horizon_weights())
    assert np.isclose(expanded, compact, rtol=1e-13)
    assert horizon_weights()[0] == 0 and np.isclose(horizon_weights().sum(), 1)
    w, _ = fit_simplex_blend(y, [p, np.zeros_like(p)], np.ones_like(p) * 0.5)
    assert np.all(w >= 0) and np.isclose(w.sum(), 1)
    bw = fit_bucket_blend(y, [p, y], np.ones_like(p) * 0.5)
    assert bw.shape == (len(LEAD_BUCKETS), 2) and np.allclose(bw.sum(axis=1), 1)
    assert bw[:, 1].min() > 0.9
    assert apply_bucket_blend(bw, [p, y]).shape == y.shape
