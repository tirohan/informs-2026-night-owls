# Team Night Owls — reproducible package

INFORMS 2026 Data Mining Society Data Challenge, Kennesaw State University.

Installable package `night_owls` (`src/` layout). One 144-hour OSI trajectory per county, issued from the
March 13 23:00 information set. Two LightGBM members share that panel: a restoration-kinetics residual and a
direct level model. County weights are off. A delayed-onset member is an ablation only and is not in the stack.
A bidirectional selective state-space model reads the same causal record; its final fit uses the same stratified
early-stopping holdout as cross-validation, four seeds. The submitted file follows `blend.py`: an equal half of
the nested two-tree stack and the sequence model, unless a cross-fitted lead-bucket mix beats that half on a
paired county bootstrap at every horizon.

Outage features use only March 11–13. Weather may come from any hour. Neighbor features use cutoff OSI only,
with Census internal points shipped in `src/night_owls/data/county_centroids.csv`. `severity_tier`, `peak_pct`,
`peak_customers`, and `time_to_restore_h` are never features. A mutation test checks that on every run.

## Environment and data

Verified on this machine, not inherited from an earlier package. The code archive
`results/cleanroom/code.zip` (SHA-256 `6fde83f2a595f3cd13d1582fb7790a0582b59c681a7f2481d31da75189309f88`)
was unpacked and `requirements-report.txt` was installed into a fresh prefix with CPython 3.13.11
on macOS 26.6.2 arm64 (torch 2.14.0, LightGBM 4.6.0, NumPy 2.3.5). That run reproduced
`predictions.csv` (SHA-256 `ca2bb95f1db077b85eeeccfda4d44c5aa47d1c2ab6488eafb70344edfe031c21`) and
`predictions_tree_stack.csv` (SHA-256 `5aca8db430fa69a1e035de098f32f922d754aac3246f6800053e6ef043a54ad4`)
byte for byte. The prefix is not the development `.venv`. PyTorch can still differ in the last decimals
on another CPU or operating system. Windows has not been executed.

The challenge files are not in this repository. After you clone, copy the three organiser files into `data/` (the folder is already there; only `data/README.txt` is tracked):

```text
data/DM_Train.csv
data/DM_Test.csv
data/sample_submission.csv
```

Those files are covered by the competition NDA. Do not commit them. `data/*.csv` and `results/` are gitignored. You can instead keep the three files in any folder and pass that folder with `--data-dir`.

## Installation

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[report]"    # Windows: .venv\Scripts\python.exe -m pip install -e ".[report]"
```

`pip install -e .` is enough to train. The `report` extra adds matplotlib. Equivalent pins are in
`requirements.txt` and `pyproject.toml`.

`requirements.txt` suffices for training, tests and diagnostics; `requirements-report.txt` adds matplotlib for the
figures. LaTeX (`pdflatex`) is needed only to rebuild the report PDF. The pinned `torch==2.14.0` wheel on PyPI
bundles CUDA libraries (about 1.2 GB) although only the CPU is used; the CPU-only build of the same version can be
installed first with `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu` and the remaining
requirements afterwards.

## Run everything

Creating the virtualenv does not switch the shell onto it. `python` here is still Conda, which cannot see the package. Call the virtualenv interpreter directly:

```bash
.venv/bin/python -m night_owls offline --data-dir data --output-dir results --threads 2
```

runs the unit tests, the five-outer / three-inner county cross-validation of the two-member boosting stack with the
ablations of Table 2 and its final fit (about 3 minutes on two CPU cores), the sequence model's five-fold
out-of-fold evaluation and final fit with the same early-stopping holdout (about an hour; PyTorch is pinned to one thread for determinism), the
blend decision in `blend.py`, the shape diagnostics, and the submission validator. Output: **`results/predictions.csv`**
(the submitted file) and `results/predictions_tree_stack.csv` (the nested two-tree stack). Add `--with-experiments`
for the development experiments of the report (about 5 more minutes), `--with-figures` for the figures, and
`--with-report` to compile the report with an installed `pdflatex` (`results/report.pdf`). Logs go to
`results/logs/`.

## Individual commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m night_owls train --data-dir data --output-dir results --ablations --threads 2
.venv/bin/python -m night_owls sequence --data-dir data --output-dir results   # add --no-cv to skip the OOF evaluation
.venv/bin/python -m night_owls combine --data-dir data --output-dir results
.venv/bin/python shape_diagnostics.py --data-dir data --results-dir results
.venv/bin/python experiments.py --data-dir data --output-dir results --threads 2
.venv/bin/python -m night_owls validate --template data/sample_submission.csv --predictions results/predictions.csv
.venv/bin/python make_figures.py --data-dir data --results-dir results --output-dir report_source/figures
.venv/bin/python verify_report_numbers.py --results-dir results --report-source report_source/report.tex
.venv/bin/python build_report.py --source-dir report_source --output results/report.pdf
```

`verify_report_numbers.py` checks all 80 displayed Table 2 values and the enumerated numerical claims of the report
against the regenerated result files (it needs the outputs of the calculation scripts). To run the unit tests with
the data elsewhere, set `NIGHT_OWLS_DATA_DIR` to that directory.

## What the pipeline does

1. **Validate** both files: unique county-hour keys, complete 216-hour grids, disjoint train/test counties, stored
   training targets equal to shifted OSI, no outage data after March 13 23:00 in the test file.
2. **Build causal features** — one panel per county of shape (144 forecast hours × 209 features) using only the first
   72 observed hours (sliced *before* any summary is computed) and weather at any hour. OSI is reconstructed from
   P_t, N_t, D_t, R_t and rounded to 4 decimals. The three neighbour features are inverse-distance and nearest
   cutoff OSI, plus distance in kilometres, from Census internal points in `src/night_owls/data/county_centroids.csv`.
   Icing, OSI slope / unrepaired fraction, and the gust-excess ratio were dropped after one-block ablations raised
   the 24 h and 48 h error. County identifiers, names, timestamps, `severity_tier`, `peak_pct`, `peak_customers`
   and `time_to_restore_h` are never features; state and severity tier only stratify folds. A **mutation test**
   overwrites every post-cutoff outage field and every whole-event column on the county frame and on the
   train+test neighbour context, and asserts the feature tensor is bitwise unchanged; it runs on every execution.
3. **Trajectory index.** `j = 0` is March 14 00:00, one hour after the cutoff; elapsed lead is `ell = j + 1`.
   Submission column `t+h` at origin `j` receives trajectory value `j + h` when that index is below 144.
4. **Two LightGBM members** on the pooled county × forecast-hour rows. Per-lead weights reproduce the mean of the
   four horizon MSEs. County weights are 1.
   `kinetics_gbm` learns the residual from `OSI_cut * 2**(-ell/H)`; `direct_gbm` learns the OSI level.
   `onset_gbm` is fit only under `--ablations`, on counties with cutoff OSI > 0.01 or positive net flow, and is not stacked.
5. **Stack** with non-negative, sum-to-one weights per bucket of `j` (1–12, 13–48, 49–143), fitted on inner
   out-of-fold predictions only, with shrinkage 0.01 toward equal weights. That nested stack is `predictions_tree_stack.csv`.
6. **Sequence model** (`python -m night_owls sequence`). Each county is one 216-step sequence. Outage channels are
   zero after the cutoff and OSI is rounded to 4 decimals. Early stopping is a severity-stratified 15% county holdout,
   both inside cross-validation and in the final fit. Four seeds are averaged. The mutation test applies to the sequences as well.
   `--no-cv` skips the out-of-fold evaluation and still uses that holdout for the final fit.
7. **Combine** (`python -m night_owls combine`): refit the lead-bucket mix of the two-tree stack and the sequence model
   on four outer folds and score the fifth. Submit that mix only if its paired county-bootstrap interval beats an equal
   half at every horizon; otherwise submit the half. The reported scores are the fold-held-out array written to
   `blend_metrics.csv`, not an in-sample refit.
8. **Refit** on all 239 counties, predict one trajectory per test county, and write the four columns as shifted
   views of it into the organiser template; identifiers, row order, NaN tails, non-negativity and cross-horizon
   consistency are re-checked on the written file.

## Random-seed schedule (all deterministic)

| Operation | Seed |
|---|---|
| Outer splitting (both models) | 42 |
| Outer model fitting (boosting) | 42 + k |
| Inner splitting | 42 + 100 + k |
| Inner model fitting | 42 + 1000 + 10k + i |
| Full-training final fit (boosting) | 42 |
| Sequence model, outer fold k, seed s | 42 + 100k + s (four seeds, s = 0..3) |
| Sequence model, final fit, seed s | 42 + 1000 + s, same stratified early-stopping holdout as cross-validation |
| Five-seed experiment | fit seed + 17i, i = 0..4 |

Bootstrap helpers default to seed 42 (2,000 resamples for intervals, 3,000 for paired comparisons). LightGBM runs in
deterministic mode; PyTorch runs single-threaded with `torch.use_deterministic_algorithms(True)`. Development scores
carry the usual model-selection optimism; hidden-test labels are unavailable.

## Diagnostics

`shape_diagnostics.py` computes the data-shape statistics of report Table 1 (marginal shape, origin of zeros,
mean–variance slope of the submitted model's out-of-fold residuals, autocorrelation, cutoff-state R² by lead,
restoration half-lives with the exponential-fit R², gust response, unequal-group one-way ANOVA ICC by state with
eta-squared reported separately, the per-fold half-life estimates, peak-day counts, rising-county counts and wave
correlations with explicit windows). Nothing in it enters a model feature.

## Files

`run_challenge.py` (boosting pipeline), `objectives.py` (metric, horizon weights, convex stacking),
`sequence_model.py` (selective state-space sequence model), `combine.py` (blend decision and final file),
`shape_diagnostics.py`, `experiments.py`, `make_figures.py`, `validate_submission.py`, `verify_report_numbers.py`,
`build_report.py`, `run_offline.py`, `tests/`, `requirements*.txt`, `report_source/` (LaTeX source and figures).

Outputs in `results/`: `predictions.csv` (submitted), `predictions_tree_stack.csv`, `cv_metrics.csv`,
`fold_metrics.csv`, `bootstrap.json`, `nested_blend_weights.json`, `run_manifest.json` (configuration, final
weights, half-life, library versions, SHA-256 of the input files, run time), `sequence_manifest.json`,
`blend_metrics.csv`, `blend_bootstrap.json`, `feature_names.json`, `feature_importance_fold*.json`,
`development_oof.npz`, `sequence_oof.npz`, `submitted_oof.npz`, `test_trajectories.npz`, `sequence_test.npz`, `final_trajectories.npz`,
`shape_diagnostics.json`, `experiments_*.csv/json/npz`, `figure_text_numbers.json`.
