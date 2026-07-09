#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_optuna_history.py
======================
Re-run Optuna with seed=42 and N_TRIALS=80 to reconstruct the optimization
history figure (Figure A3) from the stored best hyperparameters.

Since TPE with seed=42 is deterministic, this reproduces the exact trial
sequence that produced table_A3_best_hyperparams.json.

Usage:
    python plot_optuna_history.py \
        --data ../../data/track_C_2017_2024.csv
"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

SEED     = 42
N_TRIALS = 80
TARGET   = 'net_rate'
FEATURES = [
    'house_age', 'pagerank_lag1', 'childcare_pk', 'pop_density',
    'seoul_dist_km', 'ln_pop', 'fertility', 'doctor_per1000',
    'aging_ratio', 'youth_ratio', 'fiscal_indep', 'employ_rate',
    'biz_count', 'sewer_supply', 'academy_pk', 'culture_facility_count',
    'senior_fac_pk', 'hospital_bed', 'extinction_risk', 'closeness'
]
CV_FOLDS = [
    (list(range(2017, 2020)), [2020]),
    (list(range(2017, 2021)), [2021]),
    (list(range(2017, 2022)), [2022]),
]

EXPECTED_CATBOOST = {
    "iterations": 677,
    "learning_rate": 0.10884860813834339,
    "depth": 8,
    "l2_leaf_reg": 2.1524708091423577
}


def load_data(path):
    df = pd.read_csv(path)
    available = [c for c in FEATURES if c in df.columns]
    df = df.dropna(subset=[TARGET] + available)
    return df, available


def expanding_cv_score(ModelClass, params, df, features):
    scores = []
    for train_yrs, val_yrs in CV_FOLDS:
        tr = df[df['year'].isin(train_yrs)]
        va = df[df['year'].isin(val_yrs)]
        if len(va) == 0:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[features]); ytr = tr[TARGET].values
        Xva = sc.transform(va[features]);     yva = va[TARGET].values
        m = ModelClass(**params); m.fit(Xtr, ytr)
        scores.append(r2_score(yva, m.predict(Xva)))
    return float(np.mean(scores)) if scores else -999.0


def run_model_study(name, objective_fn):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_fn, n_trials=N_TRIALS)
    rows = []
    best = -999
    for t in study.trials:
        if t.value is not None:
            val = -t.value
            best = max(best, val)
            rows.append({'model': name, 'trial': t.number, 'value': val, 'best_so_far': best})
    return study, pd.DataFrame(rows)


def main(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df, features = load_data(data_path)
    print(f"Data: {len(df)} rows, {len(features)} features")

    all_hist = []

    # DT
    print("[1/6] DT")
    def dt_obj(trial):
        p = {'max_depth': trial.suggest_int('max_depth', 3, 15),
             'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
             'random_state': SEED}
        return -expanding_cv_score(DecisionTreeRegressor, p, df, features)
    _, h = run_model_study('DT', dt_obj); all_hist.append(h)

    # RF
    print("[2/6] RF")
    def rf_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 800),
             'max_depth': trial.suggest_int('max_depth', 3, 20),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
             'max_features': trial.suggest_float('max_features', 0.4, 1.0),
             'random_state': SEED, 'n_jobs': -1}
        return -expanding_cv_score(RandomForestRegressor, p, df, features)
    _, h = run_model_study('RF', rf_obj); all_hist.append(h)

    # GB
    print("[3/6] GB")
    def gb_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 600),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'max_depth': trial.suggest_int('max_depth', 2, 6),
             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
             'random_state': SEED}
        return -expanding_cv_score(GradientBoostingRegressor, p, df, features)
    _, h = run_model_study('GB', gb_obj); all_hist.append(h)

    # XGB
    print("[4/6] XGB")
    def xgb_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'max_depth': trial.suggest_int('max_depth', 3, 12),
             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
             'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
             'random_state': SEED, 'verbosity': 0}
        return -expanding_cv_score(XGBRegressor, p, df, features)
    _, h = run_model_study('XGB', xgb_obj); all_hist.append(h)

    # LGBM
    print("[5/6] LGBM")
    def lgbm_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'num_leaves': trial.suggest_int('num_leaves', 15, 127),
             'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
             'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
             'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
             'random_state': SEED, 'verbose': -1}
        return -expanding_cv_score(LGBMRegressor, p, df, features)
    _, h = run_model_study('LGBM', lgbm_obj); all_hist.append(h)

    # CatBoost
    print("[6/6] CatBoost")
    def cat_obj(trial):
        p = {'iterations': trial.suggest_int('iterations', 200, 800),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'depth': trial.suggest_int('depth', 4, 10),
             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
             'random_seed': SEED, 'verbose': 0}
        return -expanding_cv_score(CatBoostRegressor, p, df, features)
    cat_study, h = run_model_study('CatBoost', cat_obj); all_hist.append(h)

    # Verify CatBoost best params
    print("\n=== CatBoost 진본 검증 ===")
    got = cat_study.best_params
    ok = True
    for k, expected_val in EXPECTED_CATBOOST.items():
        got_val = got.get(k)
        match = abs(got_val - expected_val) < 1e-10 if got_val is not None else False
        status = "✓" if match else "✗"
        if not match:
            ok = False
        print(f"  {k}: expected={expected_val:.15g}, got={got_val:.15g} {status}")
    if ok:
        print("  → 완전 일치: 진본 확정")
    else:
        print("  → 불일치: 환경 차이(패키지 버전 등) 가능성 있음")

    # Save history CSV
    hist_df = pd.concat(all_hist, ignore_index=True)
    hist_path = os.path.join(out_dir, 'optuna_trial_history.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f"\nSaved trial history: {hist_path}")

    # Plot
    _plot(hist_df, out_dir)


def _plot(hist_df, out_dir):
    models = ['DT', 'RF', 'GB', 'XGB', 'LGBM', 'CatBoost']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, mname in zip(axes, models):
        sub = hist_df[hist_df['model'] == mname]
        ax.scatter(sub['trial'], sub['value'], s=10, alpha=0.4, color='#4878CF', label='Trial Val R²')
        ax.plot(sub['trial'], sub['best_so_far'], color='#E24A33', lw=2.0, label='Best so far')
        ax.set_title(mname, fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial number', fontsize=9)
        ax.set_ylabel('Val R² (expanding-window CV)', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', lw=0.7, ls='--')

    fig.suptitle(
        'Figure A3. Optuna Optimization History by Algorithm\n'
        'TPE sampler · seed = 42 · 80 trials per algorithm · '
        'Objective: mean expanding-window Val R² (2017–2022)',
        fontsize=10
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_optuna_history.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='../../data/track_C_2017_2024.csv')
    parser.add_argument('--out_dir', default='../../results/figures/')
    args = parser.parse_args()
    main(args.data, args.out_dir)
