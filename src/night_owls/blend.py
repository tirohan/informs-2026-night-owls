"""Final submission: equal-weight average of the tree stack and the sequence model.

The weight is fixed at 1/2. No extra blend parameter is fit for the submitted file.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from night_owls.config import HORIZONS
from night_owls.features import build_inputs
from night_owls.io import read_frame
from night_owls.metrics import county_bootstrap, fit_simplex_blend, horizon_scores, paired_bootstrap
from night_owls.submission import write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    res = args.output_dir
    tree = np.load(res / "test_trajectories.npz")
    seq = np.load(res / "sequence_test.npz")
    if not np.array_equal(tree["fips"], seq["fips"]):
        raise ValueError("County order differs between the tree stack and the sequence model.")
    curves = 0.5 * tree["blend"] + 0.5 * seq["pred"]
    if not np.isfinite(curves).all() or (curves < 0).any():
        raise ValueError("Invalid blended trajectories.")
    if (res / "predictions.csv").exists() and not (res / "predictions_tree_stack.csv").exists():
        shutil.copy2(res / "predictions.csv", res / "predictions_tree_stack.csv")
    test_x = build_inputs(read_frame(args.data_dir / "DM_Test.csv"), context=_context(args.data_dir))
    template = pd.read_csv(args.data_dir / "sample_submission.csv")
    write_submission(template, test_x, curves, res / "predictions.csv")
    np.savez_compressed(
        res / "final_trajectories.npz", fips=tree["fips"], final=curves,
        tree_stack=tree["blend"], sequence=seq["pred"],
    )

    dev = np.load(res / "development_oof.npz")
    seq_oof = np.load(res / "sequence_oof.npz")
    if not (np.array_equal(dev["fips"], seq_oof["fips"]) and np.allclose(dev["truth"], seq_oof["truth"])):
        raise ValueError("Out-of-fold arrays are not aligned.")
    y, stack, sequence, fold = dev["truth"], dev["stack_bucket_nested"], seq_oof["oof"], dev["fold"]
    blend = 0.5 * stack + 0.5 * sequence
    reference = horizon_scores(y, dev["mean_curve"])
    rows = []
    for name, pred in [
        ("mean_curve", dev["mean_curve"]), ("kinetics_gbm", dev["kinetics_gbm"]),
        ("direct_gbm", dev["direct_gbm"]), ("onset_gbm", dev["onset_gbm"]),
        ("stack_bucket_nested", stack), ("sequence_model", sequence), ("final_equal_average", blend),
    ]:
        row = {"model": name, **horizon_scores(y, pred)}
        row["mean_relative_rmse"] = float(np.mean([
            row[f"rmse_t{h:02d}h"] / reference[f"rmse_t{h:02d}h"] for h in HORIZONS
        ]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(res / "blend_metrics.csv", index=False)
    per_fold = {
        f"fold_{k}": {
            f"t{h:02d}h_pct_change": float(100 * (
                np.sqrt(((blend[fold == k] - y[fold == k])[:, h:] ** 2).mean())
                / np.sqrt(((stack[fold == k] - y[fold == k])[:, h:] ** 2).mean()) - 1
            ))
            for h in HORIZONS
        }
        for k in range(int(fold.max()) + 1)
    }
    cross = {}
    for k in range(int(fold.max()) + 1):
        w, _ = fit_simplex_blend(
            y, [stack, sequence], dev["mean_curve"], shrinkage=0.0,
            mask=np.broadcast_to((fold != k)[:, None], y.shape),
        )
        cross[f"fitted_without_fold_{k}"] = float(w[1])
    w_all, _ = fit_simplex_blend(y, [stack, sequence], dev["mean_curve"], shrinkage=0.0)
    out = {
        "final_vs_stack": paired_bootstrap(y, blend, stack),
        "final_vs_sequence": paired_bootstrap(y, blend, sequence),
        "sequence_vs_stack": paired_bootstrap(y, sequence, stack),
        "final_county_bootstrap_ci": county_bootstrap(y, blend),
        "per_fold_pct_change_final_vs_stack": per_fold,
        "error_correlation_stack_sequence": float(np.corrcoef(
            (stack - y)[:, 1:].ravel(), (sequence - y)[:, 1:].ravel(),
        )[0, 1]),
        "sequence_weight_diagnostic": {
            "submitted_fixed_weight": 0.5, "cross_fitted": cross,
            "cross_fitted_range": [min(cross.values()), max(cross.values())],
            "fitted_on_all_folds": float(w_all[1]),
        },
    }
    (res / "blend_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("Submission written: results/predictions.csv (equal-weight average).")


def _context(data_dir: Path):
    train = read_frame(data_dir / "DM_Train.csv")
    test = read_frame(data_dir / "DM_Test.csv")
    return pd.concat([train, test], ignore_index=True, sort=False)


if __name__ == "__main__":
    main()
