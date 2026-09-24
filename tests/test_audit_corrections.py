"""Regression coverage for the final audit corrections; does not alter any model."""
from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_diagnostics import one_way_anova, icc_one_way

DATA = Path(os.environ.get("NIGHT_OWLS_DATA_DIR", str(ROOT / "data"))).resolve()


class AuditCorrectionTests(unittest.TestCase):
    def test_unbalanced_icc_matches_hand_calculation(self):
        result = one_way_anova([1, 2, 4, 6, 8], ["a", "a", "b", "b", "b"])
        self.assertAlmostEqual(result["icc"], 161 / 212, places=13)
        self.assertAlmostEqual(result["eta_squared"], 243 / 328, places=13)
        self.assertAlmostEqual(result["effective_group_size"], 2.4, places=13)
        self.assertEqual(result["group_counts"], {"a": 2, "b": 3})

    def test_balanced_icc_matches_hand_calculation(self):
        self.assertAlmostEqual(icc_one_way([1, 2, 4, 5], [0, 0, 1, 1]), 17 / 19, places=13)

    def test_icc_is_invariant_to_joint_permutation(self):
        v = np.array([1., 2., 4., 6., 8.]); g = np.array([0, 0, 1, 1, 1])
        order = np.array([4, 0, 2, 1, 3])
        self.assertAlmostEqual(icc_one_way(v, g), icc_one_way(v[order], g[order]), places=13)

    def test_negative_icc_is_not_silently_clipped(self):
        self.assertLess(icc_one_way([1, 4, 2, 3], [0, 0, 1, 1]), 0)

    def test_icc_rejects_invalid_groups(self):
        for v, g in [([1, 2], [0, 0]), ([1, 2], [0, 1]), ([1, 1, 1], [0, 0, 1])]:
            with self.assertRaises(ValueError):
                one_way_anova(v, g)

    def test_icc_rejects_missing_or_mismatched_inputs(self):
        for v, g in [([1, np.nan, 3], [0, 0, 1]), ([1, 2, 3], [0, None, 1]), ([1, 2], [0, 1, 1])]:
            with self.assertRaises(ValueError):
                one_way_anova(v, g)

    def test_training_state_icc_regression(self):
        tr = pd.read_csv(DATA / "DM_Train.csv").sort_values(["fipsCode", "timestamp_et"])
        y = tr.osi.to_numpy().reshape(-1, 216)[:, 72:]
        state = tr.drop_duplicates("fipsCode").stateAbbr.to_numpy()
        result = one_way_anova(y.max(1), state)
        self.assertEqual(result["group_counts"], {"IN": 72, "OH": 70, "PA": 52, "WV": 45})
        self.assertAlmostEqual(result["icc"], 0.082346336878, places=10)
        self.assertNotAlmostEqual(result["icc"], result["eta_squared"], places=3)

    def test_figure_cutoff_row_is_last_observed_hour(self):
        text = (ROOT / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn("CUTOFF_ROW = 71", text)
        self.assertNotIn("ax.axvline(72,", text)
        start = pd.Timestamp("2026-03-11 00:00")
        self.assertEqual(start + pd.Timedelta(hours=71), pd.Timestamp("2026-03-13 23:00"))
        self.assertEqual(start + pd.Timedelta(hours=72), pd.Timestamp("2026-03-14 00:00"))

    def test_log_target_retains_experiment_scale(self):
        text = (ROOT / "experiments.py").read_text(encoding="utf-8")
        self.assertIn("np.log1p(100 * yt)", text)
        self.assertIn("np.expm1(m.predict(xq)) / 100", text)

    def test_core_config_and_seed_offsets_preserved(self):
        from run_challenge import KINETICS_PARAMS, DIRECT_PARAMS
        self.assertEqual(KINETICS_PARAMS["n_estimators"], 350)
        self.assertEqual(DIRECT_PARAMS["n_estimators"], 800)
        text = (ROOT / "src" / "night_owls" / "pipeline.py").read_text(encoding="utf-8")
        for expression in ("args.seed + k", "args.seed + 100 + k", "args.seed + 1000 + k * 10 + i"):
            self.assertIn(expression, text)


if __name__ == "__main__":
    unittest.main()
