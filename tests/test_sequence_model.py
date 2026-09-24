"""Tests for the sequence-model view: exact scan, causal masking, determinism, and output shape."""
from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sequence_model import chunked_scan, BiMamba, build_sequences, with_prior_channel, fit_predict, SEQ_CONFIG  # noqa: E402
from run_challenge import read_frame, CUTOFF  # noqa: E402

DATA = Path(os.environ.get("NIGHT_OWLS_DATA_DIR", str(ROOT / "data"))).resolve()


class ScanAndModelTests(unittest.TestCase):
    def test_chunked_scan_matches_sequential_recurrence(self):
        torch.manual_seed(0)
        dA = torch.exp(-torch.rand(3, 50, 6, 4) * 3); dBx = torch.randn(3, 50, 6, 4)
        h = torch.zeros(3, 6, 4); ref = []
        for t in range(50):
            h = dA[:, t] * h + dBx[:, t]; ref.append(h)
        self.assertLess(float((chunked_scan(dA, dBx, 4) - torch.stack(ref, 1)).abs().max()), 1e-5)

    def test_output_shape_and_parameter_count(self):
        m = BiMamba(36, **SEQ_CONFIG)
        self.assertEqual(tuple(m(torch.randn(5, 216, 36)).shape), (5, 144))
        self.assertLess(sum(p.numel() for p in m.parameters()), 50_000)


class SequenceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_frame(DATA / "DM_Train.csv")
        cls.small = cls.train[cls.train.fipsCode.isin(cls.train.fipsCode.unique()[:4])].copy()
        cls.X, cls.y, cls.last, cls.fips = build_sequences(cls.small)

    def test_outage_channels_are_zero_after_the_cutoff(self):
        outage = self.X[:, :, 25:30]                      # P_t, N_t, D_t, R_t, reconstructed OSI
        self.assertTrue(np.all(outage[:, 72:] == 0)); self.assertTrue(np.any(outage[:, :72] != 0))
        self.assertTrue(np.all(self.X[:, :72, 30] == 1) and np.all(self.X[:, 72:, 30] == 0))   # observed flag

    def test_future_outage_values_do_not_change_sequences(self):
        bad = self.small.copy(); future = bad.timestamp_et > CUTOFF
        bad.loc[future, ["P_t", "N_t", "D_t", "R_t", "outageCount", "outage_pct"]] = 987654.0
        X2, _, last2, _ = build_sequences(bad)
        np.testing.assert_array_equal(self.X, X2); np.testing.assert_array_equal(self.last, last2)

    def test_future_customer_counts_not_used(self):
        bad = self.small.copy(); bad.loc[bad.timestamp_et > CUTOFF, "customersTracked"] = 123456789
        np.testing.assert_array_equal(build_sequences(bad)[0], self.X)

    def test_training_is_deterministic(self):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        cfg = dict(SEQ_CONFIG, max_epochs=2, patience=5)
        prior = self.last[:, None] * np.exp2(-np.arange(1, 145) / 13.0)
        Xp = with_prior_channel(self.X, prior)
        Xtr = np.concatenate([Xp] * 6); ytr = np.concatenate([self.y] * 6)     # 24 sequences so the split has >= 20 + training rows
        a, _ = fit_predict(Xtr, ytr, Xp, seed=7, cfg=cfg); b, _ = fit_predict(Xtr, ytr, Xp, seed=7, cfg=cfg)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (4, 144)); self.assertTrue(np.all(a >= 0))


if __name__ == "__main__":
    unittest.main()
