# Team Night Owls — reproducible package

INFORMS 2026 Data Mining Society Data Challenge, Kennesaw State University.

Installable package `night_owls` (`src/` layout). One 144-hour OSI trajectory per county, issued from the
March 13 23:00 information set. Three LightGBM members share that panel: a restoration-kinetics residual, a
direct level model, and a delayed-onset specialist for counties that are still out or still rising at the
cutoff. County weights up-weight positive net flow and high cutoff OSI. A bidirectional selective state-space
model reads the same causal record and is fit on all 239 training counties for the median early-stopping epoch.
The submitted file is the fixed one-half average of the stack and the sequence model.

Outage features use only March 11–13. Weather may come from any hour. Neighbor features use cutoff OSI only,
with Census internal points shipped in `src/night_owls/data/county_centroids.csv`. `severity_tier`, `peak_pct`,
`peak_customers`, and `time_to_restore_h` are never features. A mutation test checks that on every run.

## Environment and data

Verified environment: a fresh virtual environment installed from the pinned `requirements.txt` on Linux / CPython
3.11 (x86-64, CPU only) reproduced the submitted `predictions.csv` byte-for-byte. The tree stack alone was also
reproduced byte-for-byte on CPython 3.13.5. PyTorch results can differ in the last decimals on other CPUs,
operating systems or BLAS builds, so a rerun elsewhere may change the final file at the 1e-6 level; the tree-only
file (`results/predictions_tree_stack.csv`) is platform-stable. Windows has not been executed.

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

runs the unit tests, the five-outer / three-inner county cross-validation of the three-member boosting stack with the
ablations of Table 2 and its final fit (about 3 minutes on two CPU cores), the sequence model's five-fold
out-of-fold evaluation and final fit (60–75 minutes; PyTorch is pinned to one thread for determinism), the
equal-weight combination, the shape diagnostics, and the submission validator. Output: **`results/predictions.csv`**
(the submitted file) and `results/predictions_tree_stack.csv` (the boosting stack alone). Add `--with-experiments`
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
2. **Build causal features** — one panel per county of shape (144 forecast hours × 222 features) using only the first
   72 observed hours (sliced *before* any summary is computed) and weather at any hour. OSI is reconstructed from
   P_t, N_t, D_t, R_t and rounded to 4 decimals. Added blocks: OSI slope and unrepaired fraction, gust-excess ratio,
   gust × unrepaired × soil, an icing indicator, and inverse-distance neighbor cutoff OSI. County identifiers, names,
   timestamps, `severity_tier`, `peak_pct`, `peak_customers` and `time_to_restore_h` are never features; state and
   severity tier only stratify folds. A **mutation test** overwrites every post-cutoff outage field and every
   whole-event column and asserts the feature tensor is bitwise unchanged; it runs on every execution.
3. **Trajectory index.** `j = 0` is March 14 00:00, one hour after the cutoff; elapsed lead is `ell = j + 1`.
   Submission column `t+h` at origin `j` receives trajectory value `j + h` when that index is below 144.
4. **Three LightGBM members** on the pooled county × forecast-hour rows. Per-lead weights reproduce the mean of the
   four horizon MSEs, multiplied by a county weight
   `1 + 5·1[N−R>0] + 2·OSI_cut/median(positive OSI_cut)` computed inside the training fold:
   `kinetics_gbm` learns the residual from `OSI_cut * 2**(-ell/H)`; `direct_gbm` learns the OSI level;
   `onset_gbm` uses the kinetics target and is fit only on counties with cutoff OSI > 0.01 or positive net flow.
5. **Stack** with non-negative, sum-to-one weights per bucket of `j` (1–12, 13–48, 49–143), fitted on inner
   out-of-fold predictions only.
6. **Sequence model** (`python -m night_owls sequence`). Each county is one 216-step sequence. Outage channels are
   zero after the cutoff and OSI is rounded to 4 decimals. Early stopping inside cross-validation is stratified by
   state × severity. The submitted network is then trained on all 239 counties for the median of those epochs,
   four seeds averaged. The mutation test applies to the sequences as well.
7. **Combine** (`python -m night_owls combine`): the submitted trajectory is the fixed equal-weight average of the
   stack and the sequence model. Cross-fitted convex weights are reported and not used for the file.
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
| Sequence model, final fit, seed s | 42 + 1000 + s, fixed epoch = median of the cross-validation runs |
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
`sequence_model.py` (selective state-space sequence model), `combine.py` (equal-weight average and final file),
`shape_diagnostics.py`, `experiments.py`, `make_figures.py`, `validate_submission.py`, `verify_report_numbers.py`,
`build_report.py`, `run_offline.py`, `tests/`, `requirements*.txt`, `report_source/` (LaTeX source and figures).

Outputs in `results/`: `predictions.csv` (submitted), `predictions_tree_stack.csv`, `cv_metrics.csv`,
`fold_metrics.csv`, `bootstrap.json`, `nested_blend_weights.json`, `run_manifest.json` (configuration, final
weights, half-life, library versions, SHA-256 of the input files, run time), `sequence_manifest.json`,
`blend_metrics.csv`, `blend_bootstrap.json`, `feature_names.json`, `feature_importance_fold*.json`,
`development_oof.npz`, `sequence_oof.npz`, `test_trajectories.npz`, `sequence_test.npz`, `final_trajectories.npz`,
`shape_diagnostics.json`, `experiments_*.csv/json/npz`, `figure_text_numbers.json`.
