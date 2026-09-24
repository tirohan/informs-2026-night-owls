"""Final submission: lead-bucket blend of the two-tree stack and the sequence model.

The onset member is left out. Kinetics and direct are stacked by lead, then that stack is
mixed with the sequence model by the same lead buckets. Weights are fit on out-of-fold
predictions only.
"""
from __future__ import annotations

import argparse
import json
import shutil
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
    tree_weights = fit_bucket_blend(y, [dev["kinetics_gbm"], dev["direct_gbm"]], dev["mean_curve"])
    stack2 = apply_bucket_blend(tree_weights, [dev["kinetics_gbm"], dev["direct_gbm"]])
    mix_weights = fit_bucket_blend(y, [stack2, seq_oof["oof"]], dev["mean_curve"])
    stack2_test = apply_bucket_blend(tree_weights, [tree["kinetics_gbm"], tree["direct_gbm"]])
    curves = apply_bucket_blend(mix_weights, [stack2_test, seq["pred"]])
    if not np.isfinite(curves).all() or (curves < 0).any():
        raise ValueError("Invalid blended trajectories.")
    if (res / "predictions.csv").exists() and not (res / "predictions_equal_average.csv").exists():
        shutil.copy2(res / "predictions.csv", res / "predictions_equal_average.csv")
    if (res / "predictions.csv").exists() and not (res / "predictions_tree_stack.csv").exists():
        shutil.copy2(res / "predictions.csv", res / "predictions_tree_stack.csv")
    test_x = build_inputs(read_frame(args.data_dir / "DM_Test.csv"), context=_context(args.data_dir))
    template = pd.read_csv(args.data_dir / "sample_submission.csv")
    write_submission(template, test_x, curves, res / "predictions.csv")
    np.savez_compressed(
        res / "final_trajectories.npz", fips=tree["fips"], final=curves,
        two_tree_stack=stack2_test, sequence=seq["pred"], three_member_stack=tree["blend"],
    )

    sequence = seq_oof["oof"]
    stack3 = dev["stack_bucket_nested"]
    equal = 0.5 * stack3 + 0.5 * sequence
    submitted = apply_bucket_blend(mix_weights, [stack2, sequence])
    reference = horizon_scores(y, dev["mean_curve"])
    rows = []
    for name, pred in [
        ("mean_curve", dev["mean_curve"]), ("kinetics_gbm", dev["kinetics_gbm"]),
        ("direct_gbm", dev["direct_gbm"]), ("onset_gbm", dev["onset_gbm"]),
        ("stack_bucket_nested", stack3), ("two_tree_stack", stack2),
        ("sequence_model", sequence), ("equal_average", equal),
        ("submitted_lead_blend", submitted),
    ]:
        row = {"model": name, **horizon_scores(y, pred)}
        row["mean_relative_rmse"] = float(np.mean([
            row[f"rmse_t{h:02d}h"] / reference[f"rmse_t{h:02d}h"] for h in HORIZONS
        ]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(res / "blend_metrics.csv", index=False)
    weight_rows = {
        f"lead_{lo}-{hi - 1}h": {
            "kinetics_gbm": float(tw[0]), "direct_gbm": float(tw[1]),
            "two_tree_stack": float(mw[0]), "sequence_model": float(mw[1]),
        }
        for (lo, hi), tw, mw in zip(LEAD_BUCKETS, tree_weights, mix_weights)
    }
    out = {
        "submitted": "lead-bucket blend of the kinetics+direct stack and the sequence model; onset excluded",
        "lead_weights": weight_rows,
        "final_vs_sequence": paired_bootstrap(y, submitted, sequence),
        "final_vs_equal_average": paired_bootstrap(y, submitted, equal),
        "final_vs_three_member_stack": paired_bootstrap(y, submitted, stack3),
        "final_county_bootstrap_ci": county_bootstrap(y, submitted),
    }
    (res / "blend_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("Lead weights:", json.dumps(weight_rows, indent=2))
    print("Submission written: results/predictions.csv")


def _context(data_dir: Path):
    train = read_frame(data_dir / "DM_Train.csv")
    test = read_frame(data_dir / "DM_Test.csv")
    return pd.concat([train, test], ignore_index=True, sort=False)


if __name__ == "__main__":
    main()
