#!/usr/bin/env python3
"""Check Table 2 and the numerical claims in report.tex against the result files.

Every displayed RMSE is compared to ``format(value, '.5f')`` (relative RMSE to '.3f'),
so a hand-rounded neighbour such as 0.01300 for 0.01299 fails.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Row order of Table 2. The last three rows come from blend_metrics.csv.
CV_MODELS = [
    "zero", "mean_curve", "persistence", "decay_24h", "prior_only", "weather_only_gbm",
    "direct_gbm_small", "direct_gbm_uniform_loss", "kinetics_gbm_hl24", "onset_gbm",
    "kinetics_gbm", "direct_gbm", "stack_bucket_nested",
]
BLEND_MODELS = ["sequence_model", "lead_bucket_crossfit", "equal_half"]
HORIZONS = (1, 6, 24, 48)
COLUMNS = [f"rmse_t{h:02d}h" for h in HORIZONS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--report-source", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    res = a.results_dir.resolve()
    tex = a.report_source.read_text(encoding="utf-8")
    metrics = pd.read_csv(res / "cv_metrics.csv").set_index("model")
    blend = pd.read_csv(res / "blend_metrics.csv").set_index("model")
    for name in BLEND_MODELS:
        metrics.loc[name] = blend.loc[name]
    ex = pd.read_csv(res / "experiments_cv_metrics.csv").set_index("model")
    diag = json.loads((res / "shape_diagnostics.json").read_text(encoding="utf-8"))
    fig = json.loads((res / "figure_text_numbers.json").read_text(encoding="utf-8"))
    boot = json.loads((res / "blend_bootstrap.json").read_text(encoding="utf-8"))
    seq = json.loads((res / "sequence_manifest.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, passed, evidence):
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    expected_models = CV_MODELS + BLEND_MODELS
    table_lines = [line for line in tex.splitlines() if re.search(r"&.*&.*0\.\d{5}", line)]
    check("Table 2 row count", len(table_lines) == len(expected_models),
          {"actual": len(table_lines), "expected": len(expected_models)})
    for model, line in zip(expected_models, table_lines):
        numbers = re.findall(r"\d+\.\d+", line)
        expected = [f"{metrics.loc[model, col]:.5f}" for col in COLUMNS]
        expected.append(f"{metrics.loc[model, 'mean_relative_rmse']:.3f}")
        check("Table 2: " + model, numbers == expected, {"printed": numbers, "from_results": expected})

    pct = 100 * (ex.loc["analog", COLUMNS].to_numpy(float) / ex.loc["direct_gbm", COLUMNS].to_numpy(float) - 1)
    analog_string = "/".join(f"{v:.1f}" for v in pct)
    check("Analogue per-horizon percentages", analog_string in tex,
          {"percentage_increases": pct.tolist(), "printed_sequence": analog_string})
    five_seed = 100 * np.abs(ex.loc["seeds5", COLUMNS].to_numpy(float) / ex.loc["direct_gbm", COLUMNS].to_numpy(float) - 1)
    check("Five-seed bound", five_seed.max() <= 0.61 and r"at most 0.61\%" in tex,
          {"absolute_changes_percent": five_seed.tolist()})

    icc = diag["heterogeneity"]["icc_state_post_cutoff_peak"]
    check("Unbalanced ICC value and label", f"= {icc:.3f}" in tex and "unequal-group one-way ANOVA" in tex, {"icc": icc})
    for kind, label in (("outer", "five outer fits"), ("inner", "15 inner fits")):
        item = diag["half_life_cv"]["summary"][kind]
        formatted = f"{item['minimum_h']:.1f}--{item['maximum_h']:.1f}"
        check(kind.title() + " half-life range", formatted in tex and label in tex, item)
    check("Logarithmic target scale", r"\log(1+100y)" in tex and r"\log(1+y)" not in tex, {})
    check("Small-model comparison", "same small configuration" in tex, {})
    check("Seed description", "deterministic fold-specific offsets" in tex and "Seed 42 everywhere" not in tex, {})
    check("Figure 1 cutoff", fig["figure1_cutoff"]["full_record_index"] == 71, fig["figure1_cutoff"])
    check("Author placeholders removed", "Team member 2" not in tex and "Team member 3" not in tex, {})
    check("SARIMAX characterization", "SARIMAX can use future weather as exogenous inputs" in tex, {})
    check("No equivalence claim from nonsignificance", "statistically tied" not in tex, {})
    check("Stale reported row removed", "0.01071" not in tex and "222" not in tex, {})

    submitted = boot["submitted"]
    check("Submitted blend is the equal half", submitted == "equal_half", {"submitted": submitted})
    st = metrics.loc["stack_bucket_nested", COLUMNS].to_numpy(float)
    fi = metrics.loc["equal_half", COLUMNS].to_numpy(float)
    sq = metrics.loc["sequence_model", COLUMNS].to_numpy(float)
    cl = metrics.loc["mean_curve", COLUMNS].to_numpy(float)
    gain = "/".join(f"{v:.1f}" for v in 100 * (1 - fi / st))
    gain_seq = "/".join(f"{v:.1f}" for v in 100 * (1 - fi / sq))
    clim = "/".join(f"{v:.0f}" for v in 100 * (1 - fi / cl))
    check("Final vs stack percentages", f"by {gain}\\%" in tex, {"printed": gain})
    check("Final vs sequence percentages", f"by {gain_seq}\\%" in tex, {"printed": gain_seq})
    check("Final vs climatology percentages", f"{clim}\\%" in tex, {"printed": clim})
    fractions = "/".join(f"{boot['final_vs_stack'][f't{h:02d}h']['prob_improvement']:.2f}" for h in HORIZONS)
    check("Final vs stack improvement fractions", f"fractions {fractions}" in tex, {"printed": fractions})
    fr_seq = "/".join(f"{boot['final_vs_sequence'][f't{h:02d}h']['prob_improvement']:.2f}" for h in HORIZONS)
    check("Final vs sequence improvement fractions", fr_seq in tex, {"printed": fr_seq})
    bucket = "/".join(f"{boot['lead_bucket_crossfit_vs_equal_half'][f't{h:02d}h']['prob_improvement']:.2f}" for h in HORIZONS)
    check("Lead-bucket vs half improvement fractions", bucket in tex, {"printed": bucket})
    corr = boot["error_correlation_stack_sequence"]
    check("Error correlation", f"at {corr:.2f}" in tex, {"value": corr})
    early = [row["lead_1-12h"]["sequence_model"] for row in boot["cross_fitted_weights"]]
    mid = [row["lead_13-48h"]["tree_stack"] for row in boot["cross_fitted_weights"]]
    span = f"{min(early):.2f}--{max(early):.2f}"
    mid_span = f"{min(mid):.2f}--{max(mid):.2f}"
    check("Cross-fitted weight spans", span in tex and mid_span in tex, {"sequence_1_12": span, "tree_13_48": mid_span})
    worse = sum(v[f"t{h:02d}h_pct_change"] > 0 for v in boot["per_fold_pct_change_submitted_vs_stack"].values() for h in HORIZONS)
    check("Per-fold consistency", f"{worse} of 20" in tex, {"fold_horizon_cells_worse": worse})
    check("Sequence model parameter count", f"{seq['n_parameters']:,}".replace(",", "{,}") in tex, {"n_parameters": seq["n_parameters"]})
    check("Top-1 county error share", f"holds {fig['top1_share_pct']:.0f}\\%" in tex, {"value": fig["top1_share_pct"]})
    check("Top-10 county error share", f"account for {fig['top10_county_share_sse_pct']:.0f}\\%" in tex, {"value": fig["top10_county_share_sse_pct"]})
    check("Severe-cell shrinkage ratio", f"predicts {100 * fig['severe_cells_ratio']:.0f}\\%" in tex, {"value": fig["severe_cells_ratio"]})
    check("Wave-2 peak correlation", f"correlate at {fig['wave2_peak_corr']:.2f}" in tex, {"value": fig["wave2_peak_corr"]})
    check("March 16-17 skill",
          f"RMSE is {fig['mar16_17_skill_vs_clim_pct']:.0f}\\% below climatology" in tex
          and f"alone: {fig['tree_stack']['mar16_17_skill_vs_clim_pct']:.0f}\\%" in tex,
          {"final": fig["mar16_17_skill_vs_clim_pct"], "stack": fig["tree_stack"]["mar16_17_skill_vs_clim_pct"]})
    check("Top-1 county maximum prediction", f"peaks at {fig['top1_county_max_prediction']:.2f}" in tex,
          {"value": fig["top1_county_max_prediction"]})
    mv = diag["mean_variance"]
    check("Mean-variance slopes",
          f"$={mv['loglog_slope_tree_stack']:.2f}$ for the stack, {mv['loglog_slope_residual_var_vs_pred_mean']:.2f} for the submitted average" in tex,
          mv)
    uniform = 100 * (metrics.loc["direct_gbm_uniform_loss", COLUMNS].to_numpy(float) / metrics.loc["direct_gbm", COLUMNS].to_numpy(float) - 1)
    uniform_string = "/".join(f"{v:+.1f}" for v in uniform)
    check("Lead-weight ablation", uniform_string in tex, {"percent": uniform.tolist()})
    check("Final fit protocol", "early-stopping holdout" in tex and "72 epochs" not in tex, {"final_fit": seq.get("final_fit")})
    check("Centroid sentence", "Census county internal points" in tex and "optional" in tex, {})

    out = {"all_passed": all(item["passed"] for item in checks), "checks": checks}
    destination = a.output or res / "report_number_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2), encoding="utf-8")
    failed = [item for item in checks if not item["passed"]]
    print(json.dumps({"all_passed": out["all_passed"], "failed": failed}, indent=2))
    if not out["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
