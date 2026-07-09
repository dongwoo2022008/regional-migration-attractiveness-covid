# Determinants of Regional Migration Attractiveness Before and After COVID-19

Interpretable machine-learning analysis of the determinants of net in-migration
across 229 South Korean municipalities (Track C, 2017–2024), and how their
relative importance was restructured across the COVID-19 pandemic.

> This repository is split from the integrated study
> (github.com/dongwoo2022008/korea-migration-network) and contains only the
> RQ5 (ML determinants), RQ6 (temporal change), and RAI materials.
> Network-structure (RQ1) and age-group (RQ4) analyses live in separate repos.

## Research questions

- **RQ1.** What is the nonlinear determinant structure of regional in-migration attractiveness, and where does network position rank?
- **RQ2 (headline).** How was this determinant structure restructured across COVID-19?
- **RQ3.** Does a SHAP-weighted Regional Attraction Index (RAI) validly explain net migration?

## Repository structure

```
regional-migration-attractiveness-covid/
├── data/
│   ├── analysis_dataset_FINAL_v4.csv   # main panel (net_rate, pagerank, pagerank_lag1, SOC, ...)
│   ├── track_C_2017_2024.csv           # ML dataset (Track C, 2017–2024)
│   └── 분석데이터셋_최종_v2.xlsx         # variable dictionary / metadata
├── scripts/
│   ├── ml/         # model training, SHAP, RAI, tuning
│   ├── temporal/   # period-split FE, yearly OLS (RQ6)
│   └── rai/        # RAI construction & validation (if separated)
├── results/
│   ├── tables/     # table4_16~20, table4_A1, lm_test, table_R5_yearly_coefs
│   └── figures/    # fig4_14~4_19
├── outputs_reference/   # confirmed_values.json, feature_cols.json, rai_domain_weights.json,
│                        # best_model_name.txt, table_A3_best_hyperparams.json
├── docs/
│   └── _reference_legacy/   # 03_방법론, 04_연구결과 (reference only — new paper written fresh)
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

- `data/analysis_dataset_FINAL_v4.csv` — main panel (net_rate, pagerank/pagerank_lag1, SOC).
- `data/track_C_2017_2024.csv` — ML dataset (Track C, 2017–2024, 229 municipalities × 8 years).

## Environment

Python 3.10; see `requirements.txt` (numpy==1.26.4 / scipy==1.13.1 pinned).

```bash
conda create -n rq56 python=3.10 -y
conda activate rq56
pip install -r requirements.txt
```

## Status

Materials migrated from the integrated repo (`korea-migration-network`); standalone analysis & paper in progress.

## Source

Migrated from: https://github.com/dongwoo2022008/korea-migration-network  
Original repo contains RQ1–RQ4 (network structure, spatial econometrics, age-group) — not included here.

## Reproduction pipeline (spatial-heterogeneity paper)

Order matters; later steps consume earlier outputs. All seeds fixed (42).

| Step | Script | Output |
|---|---|---|
| 1. Hyperparameter tuning | `scripts/ml/hyperparameter_tuning.py` | `outputs_reference/table_A3_best_hyperparams.json`, `results/tables/table_A3_search_space.csv` (Optuna TPE, 80 trials/algorithm; objective = mean validation R2 over 3 expanding-window folds: 2017-2019>2020, 2017-2020>2021, 2017-2021>2022) |
| 2. Model comparison (Table 2) | recovered in `scripts/ml/bootstrap_performance_ci.py` (fit section) | `results/tables/table4_16_ml_performance.csv` (train 2017-2022, hold-out 2023-2024, n = 458) |
| 3. Bootstrap performance CIs (Table A7) | `scripts/ml/bootstrap_performance_ci.py` | `results/tables/tableA7_bootstrap.csv` |
| 4. Typology & diagnostics | `scripts/ml/lifecycle_typology.py`, `scripts/ml/cluster_selection.py` | `results/tables/t1_region_clusters.csv`, `cluster_selection_metrics.csv` |
| 5. Stage-stratified SHAP | `scripts/ml/lifecycle_verification.py` | stage ranks (verified: PageRank 4/18/13, closeness 10/1/8); `results/tables/stage_shap_ranks.csv` |
| 6. Main-text figures | `scripts/ml/make_paper_figures.py` | `results/figures/paper/fig2,3,5,6,7` (LOWESS frac = 0.5) |
| 7. Maps & appendix figures | `scripts/ml/make_appendix_figures.py` | migration/cluster choropleths, stage-rank heatmap, Figures A2-A6 (requires southkorea-maps geojson for maps) |
| 8. Optuna history figure | `scripts/ml/plot_optuna_history.py` | re-runs tuning with seed 42 to rebuild trial history |

Preprocessing conventions (used consistently in steps 2-3, 5-7):
`pagerank_lag1` = within-municipality one-year lag of `pagerank`; missing
predictor values imputed with year-specific medians for model training
(`hyperparameter_tuning.py` expects a pre-imputed panel - its `dropna` is a
no-op on imputed inputs); stage-stratified analyses use listwise deletion
(N = 1,480: growth 588 / middle 528 / mature 364). The linear baseline is a
fixed ridge penalty (alpha = 1.0), not tuned.

Known limitation: the archived `results/tables/table4_16_ml_performance.csv`
values for LR/RF/GB/XGB deviate from current-library re-runs by up to 0.05 in
R2 (CatBoost, LightGBM, and DT reproduce exactly); see the note to Table A7.
