"""County sample weights and the delayed-onset training mask.

These depend only on the cutoff state of the counties inside the current fit.
"""
from __future__ import annotations

import numpy as np

from night_owls.config import COUNTY_WEIGHT_ALPHA, COUNTY_WEIGHT_BETA, ONSET_MIN_COUNTIES, ONSET_OSI_MIN


def county_multipliers(last_osi, net_flow, reference_last=None) -> np.ndarray:
    """w_c = 1 + alpha * 1[N-R > 0] + beta * OSI_cut / median(positive reference OSI).

    `reference_last` is the training-fold cutoff OSI used for the median. When omitted,
    the median is taken from `last_osi` itself.
    """
    last_osi = np.asarray(last_osi, dtype=float)
    net_flow = np.asarray(net_flow, dtype=float)
    ref = last_osi if reference_last is None else np.asarray(reference_last, dtype=float)
    positive = ref[ref > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    median = max(median, 1e-6)
    return (
        1.0
        + COUNTY_WEIGHT_ALPHA * (net_flow > 0).astype(float)
        + COUNTY_WEIGHT_BETA * last_osi / median
    )


def lead_row_weights(multipliers, lead_weights) -> np.ndarray:
    """Repeat each county multiplier across the scored leads and multiply by the lead weights."""
    multipliers = np.asarray(multipliers, dtype=float)
    lead_weights = np.asarray(lead_weights, dtype=float)
    return (multipliers[:, None] * lead_weights[None, :]).reshape(-1)


def onset_training_index(last_osi, net_flow, train_idx, osi_min: float = ONSET_OSI_MIN,
                         minimum: int = ONSET_MIN_COUNTIES) -> np.ndarray:
    """Counties that are still materially out, or still gaining outages, at the cutoff.

    Falls back to every training county when the specialist subset is too small to fit.
    """
    train_idx = np.asarray(train_idx)
    last = np.asarray(last_osi, dtype=float)
    net = np.asarray(net_flow, dtype=float)
    chosen = train_idx[(last[train_idx] > osi_min) | (net[train_idx] > 0)]
    if len(chosen) < minimum:
        return train_idx
    return chosen
