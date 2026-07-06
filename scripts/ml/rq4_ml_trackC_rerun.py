"""
RQ4 ML 재현 — Track C (2017~2024)
프로토콜:
  - 2017~2023: Expanding Window 5-Fold CV
    fold1: train=2017, test=2018
    fold2: train=2017-2018, test=2019
    fold3: train=2017-2019, test=2020
    fold4: train=2017-2020, test=2021
    fold5: train=2017-2021, test=2022+2023 (마지막 fold는 2022·2023 합산)
  - 2024: Hold-out Test (완전 독립)
  - 7개 모형: LR / DT / RF / GB / XGB / LGBM / CAT
  - 지표: RMSE, MAE, R² (CV mean±SD + Test)
  - 최고성능 모형(LGBM 예상)으로 SHAP 재계산
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap

# ─── 경로 ────────────────────────────────────────────────────────────────
DATA   = '/home/ubuntu/korea-migration-network/data/track_C_2017_2024.csv'
FIGDIR = '/home/ubuntu/korea-migration-network/results/figures'
TABDIR = '/home/ubuntu/korea-migration-network/results/tables'
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

# ─── 데이터 로드 ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Track C 원본: {df.shape}")

# ─── 특성 변수 목록 (방법론 3.6.1 기준, Track C 확장변수 포함) ──────────
feature_cols = [
    # 네트워크 (핵심)
    'pagerank_lag1',
    # 인구구조
    'youth_ratio', 'aging_ratio', 'pop_density', 'ln_population',
    # 경제·산업 (Track C 확장)
    'employ_rate', 'unemp_rate', 'ln_biz_count', 'ln_worker_count', 'grdp_pc',
    # 주거
    'house_age', 'apt_price',
    # 의료·교육·생활SOC
    'doctor_per1000', 'univ_count', 'culture_facility_count', 'childcare_pk',
    # 재정
    'fiscal_indep',
    # 접근성
    'seoul_dist_km',
    # 지역 더미
    'metro_dummy', 'large_city_dummy',
    # COVID 통제
    'covid_dummy',
]

available = [c for c in feature_cols if c in df.columns]
print(f"사용 특성: {len(available)}개 → {available}")

# ─── 결측 처리 ───────────────────────────────────────────────────────────
df_ml = df[['region_code', 'sgg_name', 'year', 'net_rate'] + available].copy()
df_ml = df_ml.dropna(subset=['net_rate'])
for col in available:
    df_ml[col] = df_ml[col].fillna(df_ml[col].median())

print(f"결측 처리 후: {df_ml.shape}")

# ─── Train/Test 분리 ─────────────────────────────────────────────────────
train_df = df_ml[df_ml['year'] <= 2023].copy()
test_df  = df_ml[df_ml['year'] == 2024].copy()
print(f"Train (2017~2023): {train_df.shape}, Test (2024): {test_df.shape}")

X_test = test_df[available].values
y_test = test_df['net_rate'].values

# ─── Expanding Window CV 설정 ────────────────────────────────────────────
# fold1~5: 점진적으로 훈련 기간 확장
folds = [
    (list(range(2017, 2018)), [2018]),
    (list(range(2017, 2019)), [2019]),
    (list(range(2017, 2020)), [2020]),
    (list(range(2017, 2021)), [2021]),
    (list(range(2017, 2022)), [2022, 2023]),
]

# ─── 모형 정의 ───────────────────────────────────────────────────────────
models = {
    'LR':   LinearRegression(),
    'DT':   DecisionTreeRegressor(max_depth=6, random_state=42),
    'RF':   RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'GB':   GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'XGB':  xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0),
    'LGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, verbose=-1),
    'CAT':  CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                               random_seed=42, verbose=0),
}

# ─── Expanding Window CV 실행 ────────────────────────────────────────────
print("\n=== Expanding Window 5-Fold CV (2017~2023) ===")
cv_results = {}

for name, model in models.items():
    rmse_list, mae_list, r2_list = [], [], []
    for train_years, val_years in folds:
        tr = train_df[train_df['year'].isin(train_years)]
        va = train_df[train_df['year'].isin(val_years)]
        X_tr = tr[available].values
        y_tr = tr['net_rate'].values
        X_va = va[available].values
        y_va = va['net_rate'].values

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)

        rmse_list.append(np.sqrt(mean_squared_error(y_va, y_pred)))
        mae_list.append(mean_absolute_error(y_va, y_pred))
        r2_list.append(r2_score(y_va, y_pred))

    cv_results[name] = {
        'CV_RMSE_mean': round(np.mean(rmse_list), 4),
        'CV_RMSE_std':  round(np.std(rmse_list), 4),
        'CV_MAE_mean':  round(np.mean(mae_list), 4),
        'CV_R2_mean':   round(np.mean(r2_list), 4),
        'CV_R2_std':    round(np.std(r2_list), 4),
    }
    print(f"  {name:6s}: CV R²={cv_results[name]['CV_R2_mean']:.4f}±{cv_results[name]['CV_R2_std']:.4f}")

# ─── Hold-out Test (2024) ────────────────────────────────────────────────
print("\n=== Hold-out Test (2024) ===")
test_results = {}

for name, model in models.items():
    # 전체 2017~2023으로 재학습
    X_full = train_df[available].values
    y_full = train_df['net_rate'].values
    model.fit(X_full, y_full)
    y_pred_test = model.predict(X_test)

    test_results[name] = {
        'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
        'Test_MAE':  round(mean_absolute_error(y_test, y_pred_test), 4),
        'Test_R2':   round(r2_score(y_test, y_pred_test), 4),
    }
    print(f"  {name:6s}: Test R²={test_results[name]['Test_R2']:.4f}, "
          f"RMSE={test_results[name]['Test_RMSE']:.4f}")

# ─── 결과 통합 ───────────────────────────────────────────────────────────
rows = []
for name in models.keys():
    row = {'Model': name}
    row.update(cv_results[name])
    row.update(test_results[name])
    rows.append(row)

result_df = pd.DataFrame(rows)
result_df.to_csv(f'{TABDIR}/table4_16_ml_performance_trackC.csv', index=False)
print("\n=== 최종 성능 표 ===")
print(result_df[['Model','CV_R2_mean','CV_R2_std','Test_R2','Test_RMSE','Test_MAE']].to_string(index=False))

# ─── SHAP 분석 (LGBM, 전체 Train 데이터 기준) ───────────────────────────
print("\n=== SHAP 분석 (LightGBM) ===")

# LGBM 전체 재학습
lgbm_final = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 random_state=42, verbose=-1)
X_all = df_ml[available].values
y_all = df_ml['net_rate'].values
lgbm_final.fit(X_all, y_all)

explainer   = shap.TreeExplainer(lgbm_final)
shap_values = explainer.shap_values(X_all)

# 변수 중요도 (Mean |SHAP|)
mean_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature':    available,
    'mean_shap':  np.round(mean_shap, 4),
}).sort_values('mean_shap', ascending=False).reset_index(drop=True)
shap_df['rank'] = range(1, len(shap_df)+1)

print("\nTop 10 SHAP 변수 중요도:")
print(shap_df.head(10).to_string(index=False))
shap_df.to_csv(f'{TABDIR}/table4_17_shap_importance_trackC.csv', index=False)

# ─── Figure 4-14: SHAP Summary Plot (beeswarm) ───────────────────────────
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      10,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(
    shap_values, X_all,
    feature_names=available,
    max_display=15,
    show=False,
    plot_size=None,
    color_bar_label='Feature Value',
)
plt.title('Figure 4-14. SHAP Summary Plot (LightGBM, Track C)',
          fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig4_14_shap_summary.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"\nFigure 4-14 저장: {FIGDIR}/fig4_14_shap_summary.png")

# ─── Figure 4-15: SHAP Dependence Plot (PageRank) ────────────────────────
top1_feat = shap_df.iloc[0]['feature']  # 1위 변수 (pagerank_lag1 예상)
top1_idx  = available.index(top1_feat)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 패널 (a): PageRank_lag1 Dependence
ax = axes[0]
shap.dependence_plot(
    top1_idx, shap_values, X_all,
    feature_names=available,
    ax=ax, show=False,
    dot_size=8, alpha=0.5,
)
ax.set_title(f'(a) {top1_feat}', fontsize=10.5, fontweight='bold')
ax.set_xlabel(top1_feat, fontsize=9.5)
ax.set_ylabel('SHAP Value', fontsize=9.5)

# 패널 (b): 2위 변수 Dependence
top2_feat = shap_df.iloc[1]['feature']
top2_idx  = available.index(top2_feat)
ax2 = axes[1]
shap.dependence_plot(
    top2_idx, shap_values, X_all,
    feature_names=available,
    ax=ax2, show=False,
    dot_size=8, alpha=0.5,
)
ax2.set_title(f'(b) {top2_feat}', fontsize=10.5, fontweight='bold')
ax2.set_xlabel(top2_feat, fontsize=9.5)
ax2.set_ylabel('SHAP Value', fontsize=9.5)

fig.suptitle('Figure 4-15. SHAP Dependence Plot (Top 2 Variables)',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig4_15_shap_dependence.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"Figure 4-15 저장: {FIGDIR}/fig4_15_shap_dependence.png")

# ─── N 제외지역 확인 ─────────────────────────────────────────────────────
print("\n=== N 제외지역 확인 ===")
all_regions = set(df['region_code'].unique())
ml_regions  = set(df_ml['region_code'].unique())
excluded    = all_regions - ml_regions
print(f"원본 지역 수: {len(all_regions)}, ML 사용: {len(ml_regions)}, 제외: {len(excluded)}")
if excluded:
    exc_names = df[df['region_code'].isin(excluded)][['region_code','sgg_name']].drop_duplicates()
    print(exc_names.to_string(index=False))

print("\n=== 완료 ===")
print(f"CV R² 최고: {result_df.loc[result_df['CV_R2_mean'].idxmax(), 'Model']} "
      f"({result_df['CV_R2_mean'].max():.4f})")
print(f"Test R² 최고: {result_df.loc[result_df['Test_R2'].idxmax(), 'Model']} "
      f"({result_df['Test_R2'].max():.4f})")
print(f"SHAP 1위: {shap_df.iloc[0]['feature']} (mean|SHAP|={shap_df.iloc[0]['mean_shap']:.4f})")
