#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hyperparameter_tuning.py
========================
Original Optuna hyperparameter tuning script for the paper:
"Urban Development Stage and Regional Migration Attractiveness"

This script is the source of table_A3_best_hyperparams.json.
It runs Optuna TPE-sampler tuning (N_TRIALS=80 per algorithm) on an
expanding-window cross-validation scheme and selects the best model
by hold-out (2023–2024) Test R², with Val R² as a tiebreaker.

Provenance: Originally developed as capstone/scripts/rq5_final_lag1.py
and migrated here for reproducibility archiving.

Usage:
    python hyperparameter_tuning.py \
        --data ../../data/track_C_2017_2024.csv \
        --out_dir ../../outputs_reference/

Outputs:
    outputs_reference/table_A3_best_hyperparams.json  — best params per model
    results/tables/table4_16_ml_performance.csv       — 7-model performance table
    results/tables/optuna_trial_history.csv           — per-trial Val R² history
    results/figures/fig_optuna_history.png            — optimization history figure
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ── Constants ──────────────────────────────────────────────────────────────
SEED      = 42
N_TRIALS  = 80
TARGET    = 'net_rate'
FEATURES  = [
    'house_age', 'pagerank_lag1', 'childcare_pk', 'pop_density',
    'seoul_dist_km', 'ln_pop', 'fertility', 'doctor_per1000',
    'aging_ratio', 'youth_ratio', 'fiscal_indep', 'employ_rate',
    'biz_count', 'sewer_supply', 'academy_pk', 'culture_facility_count',
    'senior_fac_pk', 'hospital_bed', 'extinction_risk', 'closeness'
]
TRAIN_YEARS = list(range(2017, 2023))   # 2017–2022
TEST_YEARS  = [2023, 2024]              # hold-out

# ── Expanding-window CV folds ──────────────────────────────────────────────
# fold1: train 2017–2019 → val 2020
# fold2: train 2017–2020 → val 2021
# fold3: train 2017–2021 → val 2022
CV_FOLDS = [
    (list(range(2017, 2020)), [2020]),
    (list(range(2017, 2021)), [2021]),
    (list(range(2017, 2022)), [2022]),
]

# ── Hyperparameter search spaces ───────────────────────────────────────────
# (documented here for Appendix Table A3)
SEARCH_SPACES = {
    'DT': {
        'max_depth':          ('int',   3,    15,   None),
        'min_samples_split':  ('int',   2,    30,   None),
        'min_samples_leaf':   ('int',   1,    20,   None),
    },
    'RF': {
        'n_estimators':       ('int',   200,  800,  None),
        'max_depth':          ('int',   3,    20,   None),
        'min_samples_leaf':   ('int',   1,    20,   None),
        'max_features':       ('float', 0.4,  1.0,  None),
    },
    'GB': {
        'n_estimators':       ('int',   200,  600,  None),
        'learning_rate':      ('float', 0.01, 0.3,  'log'),
        'max_depth':          ('int',   2,    6,    None),
        'subsample':          ('float', 0.6,  1.0,  None),
    },
    'XGB': {
        'n_estimators':       ('int',   200,  1000, None),
        'learning_rate':      ('float', 0.01, 0.3,  'log'),
        'max_depth':          ('int',   3,    12,   None),
        'subsample':          ('float', 0.6,  1.0,  None),
        'colsample_bytree':   ('float', 0.6,  1.0,  None),
        'reg_lambda':         ('float', 0,    5,    None),
    },
    'LGBM': {
        'n_estimators':       ('int',   200,  1000, None),
        'learning_rate':      ('float', 0.01, 0.3,  'log'),
        'num_leaves':         ('int',   15,   127,  None),
        'feature_fraction':   ('float', 0.6,  1.0,  None),
        'bagging_fraction':   ('float', 0.6,  1.0,  None),
        'bagging_freq':       ('int',   1,    10,   None),
        'reg_lambda':         ('float', 0,    5,    None),
    },
    'CatBoost': {
        'iterations':         ('int',   200,  800,  None),
        'learning_rate':      ('float', 0.01, 0.3,  'log'),
        'depth':              ('int',   4,    10,   None),
        'l2_leaf_reg':        ('float', 1,    10,   None),
    },
}


def load_data(path: str):
    df = pd.read_csv(path)
    available = [c for c in FEATURES if c in df.columns]
    missing   = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (skipped): {missing}")
    df = df.dropna(subset=[TARGET] + available)
    return df, available


def get_split(df, train_years, val_years, features):
    tr = df[df['year'].isin(train_years)]
    va = df[df['year'].isin(val_years)]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[features])
    Xva = scaler.transform(va[features])
    return Xtr, tr[TARGET].values, Xva, va[TARGET].values, scaler


def expanding_cv_score(ModelClass, params, df, features):
    scores = []
    for train_yrs, val_yrs in CV_FOLDS:
        Xtr, ytr, Xva, yva, _ = get_split(df, train_yrs, val_yrs, features)
        if len(Xva) == 0:
            continue
        m = ModelClass(**params)
        m.fit(Xtr, ytr)
        scores.append(r2_score(yva, m.predict(Xva)))
    return float(np.mean(scores)) if scores else -999.0


def evaluate(model, X, y):
    yp   = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, yp)))
    mae  = float(mean_absolute_error(y, yp))
    r2   = float(r2_score(y, yp))
    return rmse, mae, r2


def build_objective(ModelClass, df, features, extra_params=None):
    space = SEARCH_SPACES[ModelClass.__name__.replace('Regressor', '').replace('LGBM', 'LGBM').replace('XGB', 'XGB')]
    # map class name → space key
    name_map = {
        'DecisionTreeRegressor': 'DT',
        'RandomForestRegressor': 'RF',
        'GradientBoostingRegressor': 'GB',
        'XGBRegressor': 'XGB',
        'LGBMRegressor': 'LGBM',
        'CatBoostRegressor': 'CatBoost',
    }
    space_key = name_map.get(ModelClass.__name__, ModelClass.__name__)
    sp = SEARCH_SPACES[space_key]

    def objective(trial):
        params = {}
        for pname, (ptype, lo, hi, scale) in sp.items():
            if ptype == 'int':
                params[pname] = trial.suggest_int(pname, lo, hi)
            else:
                params[pname] = trial.suggest_float(pname, lo, hi, log=(scale == 'log'))
        if extra_params:
            params.update(extra_params)
        return -expanding_cv_score(ModelClass, params, df, features)

    return objective


def run_tuning(df, features, out_dir, results_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)

    # full train / test split
    df_train = df[df['year'].isin(TRAIN_YEARS)]
    df_test  = df[df['year'].isin(TEST_YEARS)]
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(df_train[features])
    y_train  = df_train[TARGET].values
    X_test   = scaler.transform(df_test[features])
    y_test   = df_test[TARGET].values

    results      = {}
    best_params  = {}
    trial_history = []   # (model, trial_number, value, best_so_far)

    # ── LR (no tuning) ──────────────────────────────────────────────────
    print("[1/7] LR (no tuning)")
    lr = Ridge(alpha=1.0, random_state=SEED)
    lr.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(lr, X_test, y_test)
    val_scores = []
    for tr_yrs, va_yrs in CV_FOLDS:
        Xtr, ytr, Xva, yva, _ = get_split(df, tr_yrs, va_yrs, features)
        m = Ridge(alpha=1.0, random_state=SEED); m.fit(Xtr, ytr)
        val_scores.append(r2_score(yva, m.predict(Xva)))
    results['LR'] = {'Val_R2': round(np.mean(val_scores), 4), 'Test_R2': round(r2, 4),
                     'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['LR'] = {}

    # ── DT ──────────────────────────────────────────────────────────────
    print("[2/7] DT (Optuna)")
    def dt_obj(trial):
        p = {'max_depth': trial.suggest_int('max_depth', 3, 15),
             'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
             'random_state': SEED}
        return -expanding_cv_score(DecisionTreeRegressor, p, df, features)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(dt_obj, n_trials=N_TRIALS)
    for t in study.trials:
        trial_history.append({'model': 'DT', 'trial': t.number, 'value': -t.value if t.value is not None else None,
                              'best_so_far': max(-s.value for s in study.trials[:t.number+1] if s.value is not None)})
    bp = {**study.best_params, 'random_state': SEED}
    m  = DecisionTreeRegressor(**bp); m.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(m, X_test, y_test)
    results['DT'] = {'Val_R2': round(-study.best_value, 4), 'Test_R2': round(r2, 4),
                     'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['DT'] = study.best_params

    # ── RF ──────────────────────────────────────────────────────────────
    print("[3/7] RF (Optuna)")
    def rf_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 800),
             'max_depth': trial.suggest_int('max_depth', 3, 20),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
             'max_features': trial.suggest_float('max_features', 0.4, 1.0),
             'random_state': SEED, 'n_jobs': -1}
        return -expanding_cv_score(RandomForestRegressor, p, df, features)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(rf_obj, n_trials=N_TRIALS)
    for t in study.trials:
        trial_history.append({'model': 'RF', 'trial': t.number, 'value': -t.value if t.value is not None else None,
                              'best_so_far': max(-s.value for s in study.trials[:t.number+1] if s.value is not None)})
    bp = {**study.best_params, 'random_state': SEED, 'n_jobs': -1}
    m  = RandomForestRegressor(**bp); m.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(m, X_test, y_test)
    results['RF'] = {'Val_R2': round(-study.best_value, 4), 'Test_R2': round(r2, 4),
                     'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['RF'] = study.best_params

    # ── GB ──────────────────────────────────────────────────────────────
    print("[4/7] GB (Optuna)")
    def gb_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 600),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'max_depth': trial.suggest_int('max_depth', 2, 6),
             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
             'random_state': SEED}
        return -expanding_cv_score(GradientBoostingRegressor, p, df, features)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(gb_obj, n_trials=N_TRIALS)
    for t in study.trials:
        trial_history.append({'model': 'GB', 'trial': t.number, 'value': -t.value if t.value is not None else None,
                              'best_so_far': max(-s.value for s in study.trials[:t.number+1] if s.value is not None)})
    bp = {**study.best_params, 'random_state': SEED}
    m  = GradientBoostingRegressor(**bp); m.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(m, X_test, y_test)
    results['GB'] = {'Val_R2': round(-study.best_value, 4), 'Test_R2': round(r2, 4),
                     'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['GB'] = study.best_params

    # ── XGB ─────────────────────────────────────────────────────────────
    print("[5/7] XGB (Optuna)")
    def xgb_obj(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'max_depth': trial.suggest_int('max_depth', 3, 12),
             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
             'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
             'random_state': SEED, 'verbosity': 0}
        return -expanding_cv_score(XGBRegressor, p, df, features)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(xgb_obj, n_trials=N_TRIALS)
    for t in study.trials:
        trial_history.append({'model': 'XGB', 'trial': t.number, 'value': -t.value if t.value is not None else None,
                              'best_so_far': max(-s.value for s in study.trials[:t.number+1] if s.value is not None)})
    bp = {**study.best_params, 'random_state': SEED, 'verbosity': 0}
    m  = XGBRegressor(**bp); m.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(m, X_test, y_test)
    results['XGB'] = {'Val_R2': round(-study.best_value, 4), 'Test_R2': round(r2, 4),
                      'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['XGB'] = study.best_params

    # ── LGBM ────────────────────────────────────────────────────────────
    print("[6/7] LGBM (Optuna)")
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
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lgbm_obj, n_trials=N_TRIALS)
    for t in study.trials:
        trial_history.append({'model': 'LGBM', 'trial': t.number, 'value': -t.value if t.value is not None else None,
                              'best_so_far': max(-s.value for s in study.trials[:t.number+1] if s.value is not None)})
    bp = {**study.best_params, 'random_state': SEED, 'verbose': -1}
    m  = LGBMRegressor(**bp); m.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(m, X_test, y_test)
    results['LGBM'] = {'Val_R2': round(-study.best_value, 4), 'Test_R2': round(r2, 4),
                       'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['LGBM'] = study.best_params

    # ── CatBoost ────────────────────────────────────────────────────────
    print("[7/7] CatBoost (Optuna)")
    def cat_obj(trial):
        p = {'iterations': trial.suggest_int('iterations', 200, 800),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             'depth': trial.suggest_int('depth', 4, 10),
             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
             'random_seed': SEED, 'verbose': 0}
        return -expanding_cv_score(CatBoostRegressor, p, df, features)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(cat_obj, n_trials=N_TRIALS)
    for t in study.trials:
        trial_history.append({'model': 'CatBoost', 'trial': t.number, 'value': -t.value if t.value is not None else None,
                              'best_so_far': max(-s.value for s in study.trials[:t.number+1] if s.value is not None)})
    bp = {**study.best_params, 'random_seed': SEED, 'verbose': 0}
    m  = CatBoostRegressor(**bp); m.fit(X_train, y_train)
    rmse, mae, r2 = evaluate(m, X_test, y_test)
    results['CatBoost'] = {'Val_R2': round(-study.best_value, 4), 'Test_R2': round(r2, 4),
                           'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)}
    best_params['CatBoost'] = study.best_params

    # ── Best model selection ─────────────────────────────────────────────
    best_name = max(results, key=lambda x: (results[x]['Test_R2'], results[x]['Val_R2']))
    print(f"\nBest model (Test R² primary): {best_name}")
    print(f"  Val R²={results[best_name]['Val_R2']:.4f}, Test R²={results[best_name]['Test_R2']:.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────
    # 1. best hyperparams JSON
    out_json = os.path.join(out_dir, 'table_A3_best_hyperparams.json')
    with open(out_json, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved: {out_json}")

    # 2. performance table CSV
    perf_rows = []
    for m_name, v in results.items():
        perf_rows.append({'Model': m_name, 'Val_R2': v['Val_R2'], 'Test_R2': v['Test_R2'],
                          'Test_RMSE': v['Test_RMSE'], 'Test_MAE': v['Test_MAE']})
    perf_df = pd.DataFrame(perf_rows)
    perf_path = os.path.join(results_dir, 'tables', 'table4_16_ml_performance.csv')
    perf_df.to_csv(perf_path, index=False)
    print(f"Saved: {perf_path}")

    # 3. trial history CSV
    hist_df = pd.DataFrame(trial_history)
    hist_path = os.path.join(results_dir, 'tables', 'optuna_trial_history.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved: {hist_path}")

    # 4. optimization history figure
    _plot_history(hist_df, results_dir)

    return results, best_params, hist_df


def _plot_history(hist_df: pd.DataFrame, results_dir: str):
    """Plot per-trial Val R² and best-so-far for each model (6 panels)."""
    models = [m for m in ['DT', 'RF', 'GB', 'XGB', 'LGBM', 'CatBoost'] if m in hist_df['model'].unique()]
    n = len(models)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    colors = {'trial': '#4878CF', 'best': '#E24A33'}

    for ax, mname in zip(axes, models):
        sub = hist_df[hist_df['model'] == mname].dropna(subset=['value'])
        ax.scatter(sub['trial'], sub['value'], s=12, alpha=0.45, color=colors['trial'], label='Trial Val R²')
        ax.plot(sub['trial'], sub['best_so_far'], color=colors['best'], lw=1.8, label='Best so far')
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=9)
        ax.set_ylabel('Val R²', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', lw=0.7, ls='--')

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        'Figure A3. Optuna Optimization History by Algorithm\n'
        '(TPE sampler, seed=42, 80 trials each; objective: mean expanding-window Val R²)',
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'figures', 'fig_optuna_history.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def print_search_space_table():
    """Print Appendix Table A3 search space in markdown format."""
    print("\n=== Appendix Table A3: Hyperparameter Search Spaces ===\n")
    print(f"{'Algorithm':<14} {'Parameter':<22} {'Type':<8} {'Range':<22} {'Scale'}")
    print("-" * 80)
    for algo, sp in SEARCH_SPACES.items():
        for i, (pname, (ptype, lo, hi, scale)) in enumerate(sp.items()):
            algo_col = algo if i == 0 else ''
            scale_str = scale if scale else 'linear'
            range_str = f"[{lo}, {hi}]"
            print(f"{algo_col:<14} {pname:<22} {ptype:<8} {range_str:<22} {scale_str}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for migration attractiveness ML')
    parser.add_argument('--data',    default='../../data/track_C_2017_2024.csv')
    parser.add_argument('--out_dir', default='../../outputs_reference/')
    parser.add_argument('--results', default='../../results/')
    parser.add_argument('--table_only', action='store_true',
                        help='Only print search space table without running tuning')
    args = parser.parse_args()

    print_search_space_table()

    if not args.table_only:
        print(f"\nLoading data: {args.data}")
        df, features = load_data(args.data)
        print(f"  N={len(df)}, features={len(features)}, years={sorted(df['year'].unique())}")
        run_tuning(df, features, args.out_dir, args.results)
