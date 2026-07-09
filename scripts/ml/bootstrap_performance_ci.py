# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals for hold-out model comparison (Appendix Table A7).

Recovered Table 2 pipeline (verified to reproduce results/tables/
table4_16_ml_performance.csv for CatBoost/LightGBM/DT to 4 decimals):
  data   : data/analysis_dataset_FINAL_v4.csv
  lag    : pagerank_lag1 = within-municipality shift(1) of pagerank
  missing: year-specific median imputation per predictor
  split  : train 2017-2022, hold-out 2023-2024 (n = 458)
  params : outputs_reference/table_A3_best_hyperparams.json (Optuna, 80 trials)

Bootstrap: 2,000 replications, (i) observation-level resampling and
(ii) province-block resampling (17 provinces redrawn with replacement)
to preserve within-province dependence of prediction errors.
Outputs: results/tables/tableA7_bootstrap.csv
"""
import json
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED, B = 42, 2000
FEATS = json.load(open("outputs_reference/feature_cols.json"))
HP = json.load(open("outputs_reference/table_A3_best_hyperparams.json"))

df = pd.read_csv("data/analysis_dataset_FINAL_v4.csv").sort_values(["region_code", "year"]).reset_index(drop=True)
df["pagerank_lag1"] = df.groupby("region_code")["pagerank"].shift(1)
d = df[(df.year >= 2017) & (df.year <= 2024)].dropna(subset=["net_rate"]).copy()
for c in FEATS:
    d[c] = d.groupby("year")[c].transform(lambda s: s.fillna(s.median()))
    d[c] = d[c].fillna(d[c].median())
tr, te = d[d.year <= 2022], d[d.year >= 2023]

models = {"LR": LinearRegression(**HP["LR"]),
          "DT": DecisionTreeRegressor(random_state=SEED, **HP["DT"]),
          "RF": RandomForestRegressor(random_state=SEED, n_jobs=-1, **HP["RF"]),
          "GB": GradientBoostingRegressor(random_state=SEED, **HP["GB"]),
          "XGB": xgb.XGBRegressor(random_state=SEED, verbosity=0, **HP["XGB"]),
          "LGBM": lgb.LGBMRegressor(random_state=SEED, verbose=-1, **HP["LGBM"]),
          "CatBoost": CatBoostRegressor(random_state=SEED, verbose=0, **HP["CatBoost"])}

y = te["net_rate"].values
prov = (te["region_code"] // 1000).values
P = {}
for n, m in models.items():
    m.fit(tr[FEATS].values, tr["net_rate"].values)
    P[n] = m.predict(te[FEATS].values)


def r2(yt, pt):
    return 1 - np.sum((yt - pt) ** 2) / np.sum((yt - yt.mean()) ** 2)


rng = np.random.default_rng(SEED)
names = list(models)
N = len(y)
res = {n: [] for n in names}; dif = {n: [] for n in names if n != "CatBoost"}
resB = {n: [] for n in names}; difB = {n: [] for n in dif}
uprov = np.unique(prov)
pidx = {p: np.where(prov == p)[0] for p in uprov}
for b in range(B):
    ii = rng.integers(0, N, N)
    r = {n: r2(y[ii], P[n][ii]) for n in names}
    for n in names: res[n].append(r[n])
    for n in dif: dif[n].append(r["CatBoost"] - r[n])
    ps = rng.choice(uprov, len(uprov), replace=True)
    jj = np.concatenate([pidx[p] for p in ps])
    rB = {n: r2(y[jj], P[n][jj]) for n in names}
    for n in names: resB[n].append(rB[n])
    for n in difB: difB[n].append(rB["CatBoost"] - rB[n])


def ci(a):
    return f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"


rows = []
for n in names:
    rows.append([n, round(r2(y, P[n]), 3), ci(res[n]), ci(resB[n]),
                 "—" if n == "CatBoost" else ci(dif[n]),
                 "—" if n == "CatBoost" else ci(difB[n])])
out = pd.DataFrame(rows, columns=["model", "R2", "obs_CI", "block_CI",
                                  "dR2_obs_CI", "dR2_block_CI"])
out.to_csv("results/tables/tableA7_bootstrap.csv", index=False)
print(out.to_string(index=False))
