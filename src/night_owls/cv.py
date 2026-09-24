"""County-grouped folds. severity_tier is a stratification key, never a feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_folds(strata, n_splits: int, seed: int):
    if pd.Series(strata).value_counts().min() < n_splits:
        raise ValueError(f"Too few counties per stratum for {n_splits} folds.")
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(strata)), strata))
