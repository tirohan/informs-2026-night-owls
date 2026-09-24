"""Tree-stack training: three members, nested lead-bucket weights, test trajectories.

Seed offsets, kept identical to the documented schedule:
  args.seed + k
  args.seed + 100 + k
  args.seed + 1000 + k * 10 + i
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from night_owls import __version__
from night_owls.boosting import fit_predict_members
from night_owls.config import (
    BASE_SEED, COUNTY_WEIGHT_ALPHA, COUNTY_WEIGHT_BETA, CUTOFF, DIRECT_PARAMS, KINETICS_PARAMS,
    LEAD_BUCKETS, MEMBERS, ONSET_PARAMS,
)
from night_owls.cv import make_folds
from night_owls.features import CENTROIDS, build_inputs, leakage_test
from night_owls.io import read_frame, validate_frame
from night_owls.metrics import (
    HORIZONS, apply_bucket_blend, county_bootstrap, fit_bucket_blend, fit_simplex_blend,
    horizon_scores, paired_bootstrap, self_test,
)
from night_owls.prior import restoration_prior
from night_owls.submission import write_submission


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-nested", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--no-final", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_test()
    started = time.monotonic()

    train = read_frame(args.data_dir / "DM_Train.csv")
    test = read_frame(args.data_dir / "DM_Test.csv")
    template_path = args.data_dir / "sample_submission.csv"
    template = pd.read_csv(template_path) if template_path.exists() else None
    validate_frame(train, False)
    validate_frame(test, True)
    if set(train.fipsCode) & set(test.fipsCode):
        raise ValueError("Train/test county overlap.")
    context = pd.concat([train, test], ignore_index=True, sort=False)
    log("Building causal feature panels (cutoff outages, full-window weather, neighbor cutoff OSI).")
    x = build_inputs(train, context=context)
    test_x = build_inputs(test, context=context)
    y = train["osi"].to_numpy().reshape(-1, 216)[:, 72:]
    if len(y) != 239 or len(test_x.fips) != 63 or x.feature_names != test_x.feature_names:
        raise ValueError("Unexpected county counts or feature columns.")
    audit = leakage_test(train, x, context=context)
    log(f"Leakage mutation test passed. Panel {x.panel.shape}.")

    county = train.drop_duplicates("fipsCode").set_index("fipsCode").loc[x.fips].reset_index()
    strata = (county.stateAbbr.astype(str) + "_" + county.severity_tier.astype(str)).to_numpy()
    folds = make_folds(strata, args.folds, args.seed)
    assignments = np.zeros(len(y), dtype=int)
    oof: dict[str, np.ndarray] = {}
    nested_bucket = np.full_like(y, np.nan)
    nested_global = np.full_like(y, np.nan)
    fold_records, weight_records, half_lives = [], [], []

    with threadpool_limits(limits=args.threads):
        for k, (tr, va) in enumerate(folds):
            assignments[va] = k
            log(f"Outer fold {k + 1}/{args.folds}: {len(tr)} training / {len(va)} validation counties.")
            pred = fit_predict_members(x, y, tr, x, va, args.seed + k, args.threads, args.ablations)
            half_lives.append(pred.pop("half_life_h"))
            n_onset = pred.pop("n_onset_counties")
            importances = {n: pred.pop(n) for n in list(pred) if n.endswith("feature_importance")}
            for name, p in pred.items():
                oof.setdefault(name, np.full_like(y, np.nan))
                oof[name][va] = p
                fold_records.append({"fold": k, "model": name, "n_onset_counties": n_onset, **horizon_scores(y[va], p)})
            if importances:
                (args.output_dir / f"feature_importance_fold{k}.json").write_text(json.dumps(importances, indent=1))
            if not args.no_nested:
                inner_pred = {m: np.full((len(tr), 144), np.nan) for m in MEMBERS + ["mean_curve"]}
                for i, (itr, iva) in enumerate(make_folds(strata[tr], args.inner_folds, args.seed + 100 + k)):
                    ip = fit_predict_members(x, y, tr[itr], x, tr[iva], args.seed + 1000 + k * 10 + i, args.threads)
                    ip.pop("half_life_h")
                    ip.pop("n_onset_counties")
                    for m in inner_pred:
                        inner_pred[m][iva] = ip[m]
                members_inner = [inner_pred[m] for m in MEMBERS]
                bw = fit_bucket_blend(y[tr], members_inner, inner_pred["mean_curve"])
                gw, _ = fit_simplex_blend(y[tr], members_inner, inner_pred["mean_curve"])
                nested_bucket[va] = apply_bucket_blend(bw, [pred[m] for m in MEMBERS])
                nested_global[va] = sum(w * pred[m] for w, m in zip(gw, MEMBERS))
                weight_records.append({
                    "outer_fold": k, "half_life_h": half_lives[-1], "n_onset_counties": n_onset,
                    "bucket_weights": {
                        f"lead_{lo}-{hi - 1}h": dict(zip(MEMBERS, np.round(w, 4).tolist()))
                        for (lo, hi), w in zip(LEAD_BUCKETS, bw)
                    },
                    "global_weights": dict(zip(MEMBERS, np.round(gw, 4).tolist())),
                })
                log(
                    f"  half-life {half_lives[-1]:.1f} h; onset counties {n_onset}; "
                    f"outer RMSE {[round(v, 5) for v in list(horizon_scores(y[va], nested_bucket[va]).values())[:4]]}"
                )
            log(f"  elapsed {time.monotonic() - started:.0f}s")

        oof["zero"] = np.zeros_like(y)
        oof["persistence"] = np.repeat(x.last_osi[:, None], 144, axis=1)
        oof["decay_24h"] = restoration_prior(x.last_osi, 24.0)
        oof["equal_blend"] = sum(oof[m] for m in MEMBERS) / len(MEMBERS)
        if not args.no_nested:
            oof["stack_global_nested"] = nested_global
            oof["stack_bucket_nested"] = nested_bucket
        rows = []
        reference = horizon_scores(y, oof["mean_curve"])
        for name, p in oof.items():
            if not np.isfinite(p).all():
                raise ValueError(f"Incomplete OOF output: {name}")
            row = {"model": name, **horizon_scores(y, p)}
            row["mean_relative_rmse"] = float(np.mean([
                row[f"rmse_t{h:02d}h"] / reference[f"rmse_t{h:02d}h"] for h in HORIZONS
            ]))
            rows.append(row)
        metrics = pd.DataFrame(rows).sort_values("mean_relative_rmse")
        metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
        pd.DataFrame(fold_records).to_csv(args.output_dir / "fold_metrics.csv", index=False)
        print("\n" + metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}") + "\n", flush=True)
        pd.DataFrame({"fipsCode": x.fips, "outer_fold": assignments, "stratum": strata}).to_csv(
            args.output_dir / "county_folds.csv", index=False,
        )
        (args.output_dir / "nested_blend_weights.json").write_text(json.dumps(weight_records, indent=2))
        np.savez_compressed(
            args.output_dir / "development_oof.npz", truth=y, fips=x.fips, fold=assignments, last_osi=x.last_osi, **oof,
        )
        final_model = "stack_bucket_nested" if not args.no_nested else "equal_blend"
        (args.output_dir / "bootstrap.json").write_text(json.dumps({
            "final_model": final_model,
            "county_bootstrap_ci": county_bootstrap(y, oof[final_model]),
            "paired_vs_direct_gbm": paired_bootstrap(y, oof[final_model], oof["direct_gbm"]),
            "paired_vs_kinetics_gbm": paired_bootstrap(y, oof[final_model], oof["kinetics_gbm"]),
            "paired_vs_onset_gbm": paired_bootstrap(y, oof[final_model], oof["onset_gbm"]),
            "paired_vs_mean_curve": paired_bootstrap(y, oof[final_model], oof["mean_curve"]),
        }, indent=2))

        final_bw = fit_bucket_blend(y, [oof[m] for m in MEMBERS], oof["mean_curve"])
        config = {
            "package_version": __version__, "seed": args.seed, "folds": args.folds,
            "inner_folds": args.inner_folds, "nested": not args.no_nested, "threads": args.threads,
            "members": MEMBERS, "kinetics_params": KINETICS_PARAMS, "direct_params": DIRECT_PARAMS,
            "onset_params": ONSET_PARAMS,
            "county_weight_alpha": COUNTY_WEIGHT_ALPHA, "county_weight_beta": COUNTY_WEIGHT_BETA,
            "lead_buckets": [list(b) for b in LEAD_BUCKETS],
            "final_bucket_weights": {
                f"lead_{lo}-{hi - 1}h": dict(zip(MEMBERS, np.round(w, 4).tolist()))
                for (lo, hi), w in zip(LEAD_BUCKETS, final_bw)
            },
            "half_life_by_outer_fold_h": half_lives, "feature_count": len(x.feature_names),
            "leakage_audit": audit,
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__,
            },
            "file_sha256": {
                n: hashlib.sha256((args.data_dir / n).read_bytes()).hexdigest()
                for n in ["DM_Train.csv", "DM_Test.csv", "sample_submission.csv"]
                if (args.data_dir / n).exists()
            },
            "centroids_sha256": hashlib.sha256(CENTROIDS.read_bytes()).hexdigest(),
            "cutoff": CUTOFF,
            "note": "Scores are county-held-out development estimates. Hidden test labels are unavailable.",
        }
        (args.output_dir / "feature_names.json").write_text(json.dumps(x.feature_names, indent=1))
        if not args.no_final:
            if template is None:
                raise FileNotFoundError(f"Required competition file not found: {template_path}")
            log("Refitting all three members on 239 counties and predicting 63 test counties.")
            test_pred = fit_predict_members(
                x, y, np.arange(len(y)), test_x, np.arange(len(test_x.fips)), args.seed, args.threads,
            )
            config["final_half_life_h"] = test_pred.pop("half_life_h")
            config["final_onset_counties"] = test_pred.pop("n_onset_counties")
            curves = apply_bucket_blend(final_bw, [test_pred[m] for m in MEMBERS])
            write_submission(template, test_x, curves, args.output_dir / "predictions.csv")
            np.savez_compressed(
                args.output_dir / "test_trajectories.npz", fips=test_x.fips, blend=curves,
                **{m: test_pred[m] for m in MEMBERS + ["prior_only"]},
            )
            log("Tree-stack submission written and checked.")
        config["runtime_seconds"] = round(time.monotonic() - started, 1)
        (args.output_dir / "run_manifest.json").write_text(json.dumps(config, indent=2))
    log(f"Done in {time.monotonic() - started:.0f}s. Outputs in {args.output_dir}.")


if __name__ == "__main__":
    main()
