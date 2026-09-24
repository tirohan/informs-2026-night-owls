"""Final submission.

Equal half of the nested two-tree stack and the sequence model, unless a cross-fitted
lead-bucket mix beats that half on a paired county bootstrap at every horizon
(95% interval of the RMSE difference entirely below zero). The onset member is not in the blend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from night_owls.config import HORIZONS, LEAD_BUCKETS
from night_owls.features import build_inputs
from night_owls.io import read_frame
from night_owls.metrics import (
    apply_bucket_blend, county_bootstrap, fit_bucket_blend, horizon_scores, paired_bootstrap,
)
from night_owls.submission import write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    res = args.output_dir
    tree = np.load(res / "test_trajectories.npz")
    seq = np.load(res / "sequence_test.npz")
    dev = np.load(res / "development_oof.npz")
    seq_oof = np.load(res / "sequence_oof.npz")
    if not np.array_equal(tree["fips"], seq["fips"]):
        raise ValueError("County order differs between the tree stack and the sequence model.")
    if not (np.array_equal(dev["fips"], seq_oof["fips"]) and np.allclose(dev["truth"], seq_oof["truth"])):
        raise ValueError("Out-of-fold arrays are not aligned.")
    y = dev["truth"]
    fold = dev["fold"]
    stack = dev["stack_bucket_nested"]
    sequence = seq_oof["oof"]
    equal = 0.5 * stack + 0.5 * sequence
    cross = np.zeros_like(y)
    cross_weights = []
    for k in range(int(fold.max()) + 1):
        train = fold != k
        bw = fit_bucket_blend(y[train], [stack[train], sequence[train]], dev["mean_curve"][train])
        cross_weights.append(bw)
        blended = apply_bucket_blend(bw, [stack, sequence])
        cross[fold == k] = blended[fold == k]
    comparison = paired_bootstrap(y, cross, equal)
    bucket_wins = all(comparison[f"t{h:02d}h"]["upper_95"] < 0 for h in HORIZONS)
    if bucket_wins:
        mix = fit_bucket_blend(y, [stack, sequence], dev["mean_curve"])
        curves = apply_bucket_blend(mix, [tree["blend"], seq["pred"]])
        choice = "lead_bucket"
    else:
        mix = np.array([[0.5, 0.5]] * len(LEAD_BUCKETS))
        curves = 0.5 * tree["blend"] + 0.5 * seq["pred"]
        choice = "equal_half"
    if not np.isfinite(curves).all() or (curves < 0).any():
        raise ValueError("Invalid blended trajectories.")
    test_x = build_inputs(read_frame(args.data_dir / "DM_Test.csv"), context=_context(args.data_dir))
    template = pd.read_csv(args.data_dir / "sample_submission.csv")
    write_submission(template, test_x, tree["blend"], res / "predictions_tree_stack.csv")
    write_submission(template, test_x, curves, res / "predictions.csv")
    reported = cross if choice == "lead_bucket" else equal
    np.savez_compressed(
        res / "final_trajectories.npz", fips=tree["fips"], final=curves, tree_stack=tree["blend"], sequence=seq["pred"],
    )
    np.savez_compressed(res / "submitted_oof.npz", fips=dev["fips"], pred=reported, truth=y, fold=fold)

    reference = horizon_scores(y, dev["mean_curve"])
    rows = []
    named = [
        ("mean_curve", dev["mean_curve"]), ("kinetics_gbm", dev["kinetics_gbm"]),
        ("direct_gbm", dev["direct_gbm"]), ("stack_bucket_nested", stack),
        ("sequence_model", sequence), ("equal_half", equal),
        ("lead_bucket_crossfit", cross), ("submitted", reported),
    ]
    if "onset_gbm" in dev.files:
        named.insert(3, ("onset_gbm", dev["onset_gbm"]))
    for name, pred in named:
        row = {"model": name, **horizon_scores(y, pred)}
        row["mean_relative_rmse"] = float(np.mean([
            row[f"rmse_t{h:02d}h"] / reference[f"rmse_t{h:02d}h"] for h in HORIZONS
        ]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(res / "blend_metrics.csv", index=False)
    out = {
        "submitted": choice,
        "bucket_wins_paired_bootstrap": bucket_wins,
        "cross_fitted_weights": [
            {f"lead_{lo}-{hi - 1}h": {"tree_stack": float(w[0]), "sequence_model": float(w[1])}
             for (lo, hi), w in zip(LEAD_BUCKETS, bw)}
            for bw in cross_weights
        ],
        "applied_weights": [
            {"tree_stack": float(w[0]), "sequence_model": float(w[1])} for w in mix
        ],
        "lead_bucket_crossfit_vs_equal_half": comparison,
        "final_vs_stack": paired_bootstrap(y, reported, stack),
        "final_vs_sequence": paired_bootstrap(y, reported, sequence),
        "sequence_vs_stack": paired_bootstrap(y, sequence, stack),
        "error_correlation_stack_sequence": float(np.corrcoef((stack - y)[:, 1:].ravel(), (sequence - y)[:, 1:].ravel())[0, 1]),
        "per_fold_pct_change_submitted_vs_stack": {
            str(k): {
                f"t{h:02d}h_pct_change": float(100 * (
                    horizon_scores(y[fold == k], reported[fold == k])[f"rmse_t{h:02d}h"]
                    / horizon_scores(y[fold == k], stack[fold == k])[f"rmse_t{h:02d}h"] - 1
                ))
                for h in HORIZONS
            }
            for k in range(int(fold.max()) + 1)
        },
        "final_county_bootstrap_ci": county_bootstrap(y, reported),
    }
    (res / "blend_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("Submitted blend:", choice)


def _context(data_dir: Path):
    train = read_frame(data_dir / "DM_Train.csv")
    test = read_frame(data_dir / "DM_Test.csv")
    return pd.concat([train, test], ignore_index=True, sort=False)


if __name__ == "__main__":
    main()
