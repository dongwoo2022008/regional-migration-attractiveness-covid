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
