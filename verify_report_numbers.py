#!/usr/bin/env python3
"""Check Table 2 and the audited numerical claims against regenerated CSV/JSON files.

This is an explicit regression checklist, not an automated proof of every prose claim.
Requires the main run, experiments.py, shape_diagnostics.py and make_figures.py outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

MODELS = ["zero", "mean_curve", "persistence", "decay_24h", "prior_only", "weather_only_gbm",
          "direct_gbm_small", "direct_gbm_uniform_loss", "kinetics_gbm_hl24", "kinetics_gbm",
          "direct_gbm", "equal_blend", "stack_global_nested", "stack_bucket_nested"]
HORIZONS = (1, 6, 24, 48)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--report-source", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    res = a.results_dir.resolve()
    tex = a.report_source.read_text(encoding="utf-8")
    metrics = pd.read_csv(res / "cv_metrics.csv").set_index("model")
    ex = pd.read_csv(res / "experiments_cv_metrics.csv").set_index("model")
    diag = json.loads((res / "shape_diagnostics.json").read_text(encoding="utf-8"))
    fig = json.loads((res / "figure_text_numbers.json").read_text(encoding="utf-8"))
    checks = []
    def check(name, passed, evidence):
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    table_lines = [l for l in tex.splitlines() if re.search(r"&.*&.*0\.\d{5}", l)]
    blend_path = res / "blend_metrics.csv"
    if blend_path.exists():   # the two final-view rows come from combine.py
        bl = pd.read_csv(blend_path).set_index("model")
        for m in ["sequence_model", "final_equal_average"]:
            metrics.loc[m] = bl.loc[m]
        expected_models = MODELS + ["sequence_model", "final_equal_average"]
    else:
        expected_models = MODELS
    check("Table 2 row count", len(table_lines) == len(expected_models), {"actual": len(table_lines), "expected": len(expected_models)})
    for model, line in zip(expected_models, table_lines):
        numbers = re.findall(r"\d+\.\d+", line)
        expected = [f"{metrics.loc[model, f'rmse_t{h:02d}h']:.5f}" for h in HORIZONS]
        expected.append(f"{metrics.loc[model, 'mean_relative_rmse']:.3f}")
        check("Table 2: " + model, numbers == expected, {"printed": numbers, "from_results": expected})
    columns = [f"rmse_t{h:02d}h" for h in HORIZONS]
    pct = 100 * (ex.loc["analog", columns].to_numpy(float) / ex.loc["direct_gbm", columns].to_numpy(float) - 1)
    analog_string = "/".join(f"{v:.1f}" for v in pct)
    check("Analogue per-horizon percentages", analog_string in tex,
          {"percentage_increases": pct.tolist(), "printed_sequence": analog_string})
    five_seed = 100 * np.abs(ex.loc["seeds5", columns].to_numpy(float) / ex.loc["direct_gbm", columns].to_numpy(float) - 1)
    check("Five-seed 0.61 percent bound", five_seed.max() <= .61 and r"at most 0.61\%" in tex,
          {"absolute_changes_percent": five_seed.tolist()})
    icc = diag["heterogeneity"]["icc_state_post_cutoff_peak"]
    check("Unbalanced ICC value and label", f"= {icc:.3f}" in tex and "unequal-group one-way ANOVA" in tex,
          diag["heterogeneity"]["post_cutoff_peak_anova"])
    for kind, label in (("outer", "five outer fits"), ("inner", "15 inner fits")):
        item = diag["half_life_cv"]["summary"][kind]
        formatted = f"{item['minimum_h']:.1f}--{item['maximum_h']:.1f}"
        check(kind.title() + " half-life range", formatted in tex and label in tex, item)
    check("Logarithmic target scale", r"\log(1+100y)" in tex and r"\log(1+y)" not in tex,
          {"implemented_transformation": "log1p(100 * y)", "inverse": "expm1(prediction) / 100"})
    check("Small-model comparison", "same small configuration" in tex and "with identical features and a smaller model" not in tex,
          {"comparison": "kinetics_gbm vs direct_gbm_small, both using KINETICS_PARAMS"})
    check("Seed description", "deterministic fold-specific offsets" in tex and "Seed 42 everywhere" not in tex,
          {"base": 42, "outer_model_seeds": list(range(42, 47)), "inner_model_expression": "42 + 1000 + k*10 + i"})
    check("Figure 1 cutoff", fig["figure1_cutoff"]["full_record_index"] == 71, fig["figure1_cutoff"])
    check("Author placeholders removed", "Team member 2" not in tex and "Team member 3" not in tex,
          {"credit": "Team Night Owls; team lead Pritam Kumar Deb; Kennesaw State University"})
    check("SARIMAX characterization", "SARIMAX can use future weather as exogenous inputs" in tex
          and "have nothing to condition on after the cutoff" not in tex,
          {"basis": "Existing report reference [1], arXiv:2511.01017, explicitly uses weather exogenous inputs."})
    check("No equivalence claim from nonsignificance", "statistically tied" not in tex,
          {"wording": "Paired difference intervals include zero; equivalence is not established."})
    if blend_path.exists():   # claims about the sequence model and the submitted average
        bb = json.loads((res / "blend_bootstrap.json").read_text(encoding="utf-8"))
        seq_manifest = json.loads((res / "sequence_manifest.json").read_text(encoding="utf-8"))
        st, fi, sq, cl = (bl.loc[m, columns].to_numpy(float) for m in ("stack_bucket_nested", "final_equal_average", "sequence_model", "mean_curve"))
        gain = "/".join(f"{v:.1f}" for v in 100 * (1 - fi / st)); gain_seq = "/".join(f"{v:.1f}" for v in 100 * (1 - fi / sq))
        clim = "/".join(f"{v:.1f}" for v in 100 * (1 - fi / cl))
        check("Final vs stack percentages", f"by {gain}\\%" in tex, {"printed": gain})
        check("Final vs sequence percentages", f"by {gain_seq}\\%" in tex, {"printed": gain_seq})
        check("Final vs climatology percentages", f"{clim}\\%" in tex, {"printed": clim})
        fractions = "/".join(f"{bb['final_vs_stack'][f't{h:02d}h']['prob_improvement']:.2f}" for h in HORIZONS)
        check("Final vs stack improvement fractions", f"fractions {fractions}" in tex, {"printed": fractions})
        fr_seq = "/".join(f"{bb['sequence_vs_stack'][f't{h:02d}h']['prob_improvement']:.2f}" for h in HORIZONS)
        check("Sequence vs stack improvement fractions", f"fractions of {fr_seq}" in tex, {"printed": fr_seq})
        corr = bb["error_correlation_stack_sequence"]
        check("Error correlation", f"at {corr:.2f}" in tex, {"value": corr})
        lo, hi = bb["sequence_weight_diagnostic"]["cross_fitted_range"]
        check("Cross-fitted weight range", f"{lo:.2f}--{hi:.2f}" in tex, {"range": [lo, hi]})
        per_fold = bb["per_fold_pct_change_final_vs_stack"]
        worse = sum(v[f"t{h:02d}h_pct_change"] > 0 for v in per_fold.values() for h in HORIZONS)
        check("Per-fold consistency statement", worse == 1 and "in four of five at 6\\,h" in tex, {"fold_horizon_cells_worse": worse})
        check("Sequence model parameter count", f"{seq_manifest['n_parameters']:,}".replace(",", "{,}") in tex, {"n_parameters": seq_manifest["n_parameters"]})
        check("Top-1 county error share", f"alone for {fig['top1_share_pct']:.0f}\\%" in tex and f"holds {fig['top1_share_pct']:.0f}\\%" in tex, {"value": fig["top1_share_pct"]})
        check("Top-10 county error share", f"account for {fig['top10_county_share_sse_pct']:.0f}\\%" in tex, {"value": fig["top10_county_share_sse_pct"]})
        check("Severe-cell shrinkage ratio", f"predicts {100 * fig['severe_cells_ratio']:.0f}\\%" in tex, {"value": fig["severe_cells_ratio"]})
        check("Wave-2 peak correlation", f"correlate at {fig['wave2_peak_corr']:.2f}" in tex, {"value": fig["wave2_peak_corr"]})
        check("March 16-17 skill", f"RMSE is {fig['mar16_17_skill_vs_clim_pct']:.0f}\\% below climatology" in tex
              and f"alone: {fig['tree_stack']['mar16_17_skill_vs_clim_pct']:.0f}\\%" in tex, {"final": fig["mar16_17_skill_vs_clim_pct"], "stack": fig["tree_stack"]["mar16_17_skill_vs_clim_pct"]})
        check("Top-1 county maximum prediction", f"more than {fig['top1_county_max_prediction']:.2f}" in tex, {"value": fig["top1_county_max_prediction"]})
        mv = diag["mean_variance"]
        check("Mean-variance slopes", f"$={mv['loglog_slope_tree_stack']:.2f}$ for the stack, {mv['loglog_slope_residual_var_vs_pred_mean']:.2f} for the submitted average" in tex, mv)
    out = {"all_passed": all(x["passed"] for x in checks), "checks": checks,
           "scope": "Table 2 plus the explicitly enumerated corrected numerical and textual claims; not all possible prose assertions."}
    destination = a.output or res / "report_number_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not out["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
