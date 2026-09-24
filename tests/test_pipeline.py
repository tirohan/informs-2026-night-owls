"""Unit and regression tests: objective identities, causal data boundaries, the leakage mutation test,
the restoration prior, stacking, and submission-file checks.  Run:  python -m unittest discover -s tests -v"""
from __future__ import annotations
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from objectives import (HORIZONS, LEAD_BUCKETS, horizon_weights, horizon_scores, fit_simplex_blend,  # noqa: E402
                        fit_bucket_blend, apply_bucket_blend)
from run_challenge import (read_frame, validate_frame, build_inputs, leakage_test, make_folds, weather_transform,  # noqa: E402
                           estimate_half_life, restoration_prior, write_submission, Inputs, CUTOFF, LEAD)
from validate_submission import validate  # noqa: E402

DATA = Path(os.environ.get("NIGHT_OWLS_DATA_DIR", str(ROOT / "data"))).resolve()


class ObjectiveTests(unittest.TestCase):
    def test_weights_reproduce_equal_horizon_mse(self):
        r = np.random.default_rng(17); y = r.random((7, 144)); p = r.random((7, 144))
        expected = np.mean([np.mean((p[:, h:] - y[:, h:]) ** 2) for h in HORIZONS])
        self.assertAlmostEqual(np.mean((p - y) ** 2 @ horizon_weights()), expected, places=14)
        self.assertEqual(horizon_weights()[0], 0); self.assertAlmostEqual(horizon_weights().sum(), 1.0)

    def test_scores_ignore_lead_zero(self):
        y = np.zeros((2, 144)); p = y.copy(); p[:, 0] = 1000
        for h in HORIZONS:
            self.assertEqual(horizon_scores(y, p)[f"rmse_t{h:02d}h"], 0)

    def test_scoreable_counts(self):
        self.assertEqual([63 * (144 - h) for h in HORIZONS], [9009, 8694, 7560, 6048])

    def test_simplex_blend_is_convex(self):
        y = np.ones((4, 144)) * .2
        w, d = fit_simplex_blend(y, [y + .1, y - .1], np.zeros_like(y))
        self.assertTrue(d["converged"]); self.assertTrue((w >= 0).all()); self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue(np.allclose(w, [.5, .5], atol=1e-6))

    def test_bucket_blend_prefers_the_better_member_per_bucket(self):
        r = np.random.default_rng(3); y = r.random((30, 144))
        a = y + r.normal(0, .01, y.shape); b = y + r.normal(0, .01, y.shape)
        a[:, 49:] += r.normal(0, .3, (30, 95)); b[:, :49] += r.normal(0, .3, (30, 49))  # a good early, b good late
        bw = fit_bucket_blend(y, [a, b], np.zeros_like(y))
        self.assertEqual(bw.shape, (len(LEAD_BUCKETS), 2))
        self.assertGreater(bw[0, 0], .9); self.assertGreater(bw[-1, 1], .9)
        out = apply_bucket_blend(bw, [a, b])
        self.assertLess(horizon_scores(y, out)["rmse_t01h"], min(horizon_scores(y, a)["rmse_t01h"], horizon_scores(y, b)["rmse_t01h"]))


class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_frame(DATA / "DM_Train.csv"); cls.test = read_frame(DATA / "DM_Test.csv")
        counties = cls.train.fipsCode.unique()[:3]
        cls.small = cls.train[cls.train.fipsCode.isin(counties)].copy()
        cls.features = build_inputs(cls.small)

    def test_grids_targets_and_disjoint_counties(self):
        validate_frame(self.train, False); validate_frame(self.test, True)
        self.assertEqual(self.train.fipsCode.nunique(), 239); self.assertEqual(self.test.fipsCode.nunique(), 63)
        self.assertFalse(set(self.train.fipsCode) & set(self.test.fipsCode))

    def test_future_test_outage_values_rejected(self):
        bad = self.test.copy(); bad.loc[bad.timestamp_et > CUTOFF, "P_t"] = 0
        with self.assertRaises(ValueError):
            validate_frame(bad, True)

    def test_corrupted_target_rejected(self):
        bad = self.train.copy(); bad.loc[100, "osi_target_t06h"] = 123
        with self.assertRaises(ValueError):
            validate_frame(bad, False)

    def test_leakage_mutation(self):
        self.assertTrue(leakage_test(self.small, self.features)["passed"])

    def test_neighbour_context_is_covered_by_the_mutation_test(self):
        context = pd.concat([self.train, self.test], ignore_index=True, sort=False)
        built = build_inputs(self.small, context=context)
        self.assertIn("spatial_idw_cutoff_osi", built.feature_names)
        self.assertTrue(leakage_test(self.small, built, context=context)["passed"])

    def test_future_customer_counts_not_used(self):
        bad = self.small.copy(); bad.loc[bad.timestamp_et > CUTOFF, "customersTracked"] = 123456789
        np.testing.assert_array_equal(build_inputs(bad).panel, self.features.panel)

    def test_supplied_cutoff_components_are_used(self):
        m = self.small.copy(); m.loc[m.timestamp_et == CUTOFF, "N_t"] += .05
        self.assertFalse(np.array_equal(build_inputs(m).panel, self.features.panel))

    def test_feature_allowlist_and_shape(self):
        from night_owls.config import N_FEATURES
        self.assertEqual(self.features.panel.shape, (3, 144, N_FEATURES))
        forbidden = {"fipsCode", "timestamp_et", "stateAbbr", "severity_tier", "peak_pct", "peak_customers", "time_to_restore_h"}
        self.assertFalse(forbidden & set(self.features.feature_names))
        for name in ("spatial_idw_cutoff_osi", "spatial_nearest_cutoff_osi", "spatial_nearest_km"):
            self.assertIn(name, self.features.feature_names)
        for name in ("hist_unrepaired_fraction", "hist_osi_slope_3h", "wx_gust_excess_ratio_30",
                     "wx_gust_x_unrepaired", "wx_icing_hour"):
            self.assertNotIn(name, self.features.feature_names)
        self.assertTrue(np.isfinite(self.features.panel).all())

    def test_reconstructed_osi_matches_supplied(self):
        g = self.small[self.small.timestamp_et <= CUTOFF]
        rebuilt = np.maximum(.4 * g.P_t + .35 * g.N_t + .25 * g.D_t - .1 * g.R_t, 0)
        self.assertLess(np.abs(rebuilt - g.osi).max(), 1e-4)

    def test_circular_wind_transform(self):
        g = self.small.iloc[:216].copy(); g.loc[g.index[0], "wind_dir_10m"] = 0; g.loc[g.index[1], "wind_dir_10m"] = 360
        w = weather_transform(g)
        self.assertNotIn("wind_dir_10m", w.columns)
        np.testing.assert_allclose(w.loc[0, ["wind_sin", "wind_cos"]], w.loc[1, ["wind_sin", "wind_cos"]], atol=1e-12)

    def test_fold_county_separation(self):
        c = self.train.drop_duplicates("fipsCode"); strata = (c.stateAbbr + "_" + c.severity_tier.astype(str)).to_numpy()
        seen = []
        for tr, va in make_folds(strata, 5, 42):
            self.assertFalse(set(tr) & set(va)); seen.extend(va)
        self.assertEqual(sorted(seen), list(range(239)))


class PriorTests(unittest.TestCase):
    def test_half_life_recovered_from_synthetic_decay(self):
        last = np.array([.1, .2, .3, .05, .4, .25])
        y = last[:, None] * np.exp2(-LEAD[None, :] / 9.0)
        self.assertAlmostEqual(estimate_half_life(y, last), 9.0, places=2)

    def test_prior_shape_and_monotonicity(self):
        p = restoration_prior(np.array([.3, 0.]), 12.0)
        self.assertEqual(p.shape, (2, 144)); self.assertTrue((np.diff(p[0]) < 0).all()); self.assertTrue((p[1] == 0).all())
        self.assertAlmostEqual(p[0, 11], .15, places=6)  # lead 12 h = one half-life


class SubmissionTests(unittest.TestCase):
    def test_write_and_validate_roundtrip_and_tampering(self):
        if not (DATA / "sample_submission.csv").exists():
            self.skipTest("sample_submission.csv is not in data/")
        template = pd.read_csv(DATA / "sample_submission.csv")
        fips = np.sort(template.fipsCode.unique())
        rng = np.random.default_rng(0)
        curves = rng.random((len(fips), 144)) * .1
        dummy = Inputs(fips, np.zeros((len(fips), 144, 1), dtype=np.float32), ["x"], np.zeros(len(fips)))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pred.csv"
            write_submission(template, dummy, curves, out)
            r = validate(DATA / "sample_submission.csv", out)
            self.assertTrue(r["passed"]); self.assertEqual(r["unique_scored_county_hours"], 9009)
            self.assertEqual(r["scoreable_cells"], [9009, 8694, 7560, 6048])
            p = pd.read_csv(out); p.loc[5, "osi_target_t01h"] += .1; bad = Path(d) / "bad.csv"; p.to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                validate(DATA / "sample_submission.csv", bad)  # origin 5 + 1 h must equal origin 0 + 6 h
            p = pd.read_csv(out); p.loc[p.osi_target_t48h.isna(), "osi_target_t48h"] = 0; p.to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                validate(DATA / "sample_submission.csv", bad)
            p = pd.read_csv(out); p.loc[0, "countyName"] = "changed"; p.to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                validate(DATA / "sample_submission.csv", bad)


class RegimeTests(unittest.TestCase):
    def test_county_multiplier_and_onset_mask(self):
        from night_owls.weights import county_multipliers, onset_training_index
        weights = county_multipliers(np.array([0.06, 0.0]), np.array([0.01, -0.1]), np.array([0.02, 0.04]))
        self.assertTrue(np.allclose(weights, 1.0))
        chosen = onset_training_index(np.array([0.001, 0.02, 0.0]), np.array([0.0, -1.0, 0.1]),
                                      np.array([0, 1, 2]), minimum=1)
        np.testing.assert_array_equal(chosen, [1, 2])
        fallback = onset_training_index(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0, 1]), minimum=25)
        np.testing.assert_array_equal(fallback, [0, 1])


if __name__ == "__main__":
    unittest.main()
