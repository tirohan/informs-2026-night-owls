"""Restoration half-life and the first-order decay prior."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from night_owls.config import LEAD


def estimate_half_life(y_train, last_train, min_cutoff: float = 0.02, window: int = 48) -> float:
    """Median best-fit half-life (hours) of OSI_cut * 2^(-lead/H) over the first `window` leads.

    Uses training counties whose cutoff OSI exceeds `min_cutoff`. Falls back to 12 h.
    """
    hs = []
    leads = LEAD[:window]
    for yy, last in zip(y_train, last_train):
        if last <= min_cutoff:
            continue
        target = yy[:window]
        res = minimize_scalar(
            lambda h: np.sum((target - last * np.exp2(-leads / h)) ** 2),
            bounds=(1, 400), method="bounded",
        )
        hs.append(res.x)
    return float(np.median(hs)) if len(hs) >= 5 else 12.0


def restoration_prior(last, half_life: float) -> np.ndarray:
    """Shape (n_counties, 144). Column k is the prior k+1 hours after the cutoff."""
    last = np.asarray(last, dtype=float)
    return last[:, None] * np.exp2(-LEAD[None, :] / half_life)
