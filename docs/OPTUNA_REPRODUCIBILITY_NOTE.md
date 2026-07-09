# Optuna tuning — reproducibility note

**Status (2026-07-09).** The original Optuna tuning script is archived as
`scripts/ml/hyperparameter_tuning.py` and is verified as authentic: its
pipeline (train 2017–2022, three expanding-window folds with validation
years 2020–2022, hold-out 2023–2024) reproduces the published model
comparison (`results/tables/table4_16_ml_performance.csv`) to four decimals
for CatBoost, LightGBM, and the decision tree, and all archived best
hyperparameters lie inside the archived search spaces
(`results/tables/table_A3_search_space.csv`).

**What is not reproducible.** The Optuna *study object* (trial-by-trial
history) was not archived. A seed-42 re-run under catboost 1.2.10 yields a
different best configuration (iterations 790, learning_rate 0.036, depth 10,
l2_leaf_reg 1.043) than the archived one (677 / 0.1088 / 8 / 2.152). This is
expected: TPE proposals depend on the observed objective values of earlier
trials, so any library-version change that perturbs training propagates
through the entire search trajectory.

**Consequences for the manuscript.**
- Appendix Table A3 (search spaces + selected values) is retained: both
  columns are extracted directly from the archived script and JSON.
- No optimization-history figure is included in the paper: a re-run history
  would depict a different optimization than the one that produced the
  published model, creating an internal inconsistency with Table A3.
- Any re-run history figures/CSVs committed to `results/` are reproduction
  artefacts of the current environment and are labelled as such; they are
  not sources for manuscript values.

**If a reviewer asks.** State that the tuning script, search spaces, seeds,
and selected values are archived and verified, but the study database was
not retained; exact trial-history reproduction is version-dependent.
