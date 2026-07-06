"""
Step 3 보완 + Step 4: 통합 분석 스크립트
- SDM (Spatial Lag Model) 추정 및 직접/간접/총효과 분해
- 강건성 검증 (다중 중심성 지표)
- Step 4: 머신러닝 기반 지역 흡인력 분석 (RF, XGB, LGBM)
- SHAP 분석
- RAI (지역 흡인력 지수) 구축
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
from libpysal.weights import Queen, KNN
import esda
from esda.moran import Moran
import spreg
from spreg import OLS, ML_Lag, ML_Error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
font_list = [f.name for f in fm.fontManager.ttflist 
             if any(k in f.name for k in ['Noto', 'Gothic', 'Malgun'])]
kr_font = next((f for f in font_list if 'CJK' in f or 'KR' in f or 'Gothic' in f), 'DejaVu Sans')
plt.rcParams['font.family'] = kr_font
plt.rcParams['axes.unicode_minus'] = False

BASE = '/home/ubuntu/capstone'
FIGS = f'{BASE}/results/figures'
TABS = f'{BASE}/results/tables'

print("=" * 70)
print("Step 3 보완 + Step 4: SDM + 머신러닝 통합 분석")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# 데이터 로드 및 공간 매핑
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{BASE}/data/track_B_2009_2024.csv')
gdf_raw = gpd.read_file(f'{BASE}/data/shp/sgg_municipalities_2018.geojson')
code_map = pd.read_csv(f'{BASE}/data/code_mapping.csv')

print(f"패널 데이터: {df.shape}")
print(f"코드 매핑: {len(code_map)}개 지역")

# 패널에 GDF 코드 추가
df = df.merge(code_map[['region_code','gdf_code']], on='region_code', how='left')
df_mapped = df[df['gdf_code'].notna()].copy()
# float -> int -> str 변환 (11010.0 -> '11010')
df_mapped['gdf_code'] = df_mapped['gdf_code'].astype(float).astype(int).astype(str)

# GDF 정렬
gdf_raw['code'] = gdf_raw['code'].astype(str)
valid_codes = df_mapped['gdf_code'].unique()
gdf = gdf_raw[gdf_raw['code'].isin(valid_codes)].copy()
gdf = gdf.sort_values('code').reset_index(drop=True)

print(f"매핑된 패널: {df_mapped.shape}, 공간 지역: {len(gdf)}")

# ─────────────────────────────────────────────────────────────────────────
# STEP 3 보완: SDM 분석
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("STEP 3: SDM 추정 및 공간효과 분해")
print("─" * 50)

# 공간가중행렬
W_queen = Queen.from_dataframe(gdf, use_index=False)
W_queen.transform = 'r'
print(f"Queen W: {W_queen.n}개 지역, 평균 이웃 {W_queen.mean_neighbors:.2f}개")

X_vars = ['pagerank', 'youth_ratio', 'aging_ratio', 'fiscal_indep',
          'employ_rate', 'ln_pop_density', 'doctor_per1000']

sdm_summary = []
direct_indirect_all = []

for yr in [2012, 2015, 2019, 2022, 2023]:
    df_yr = df_mapped[df_mapped['year'] == yr].copy()
    df_yr = df_yr.set_index('gdf_code').reindex(gdf['code'])
    
    y = df_yr['net_rate'].copy()
    y_median = df_mapped['net_rate'].median()
    y = y.fillna(y_median).values.astype(float)
    
    X_data = pd.DataFrame(index=gdf['code'])
    for col in X_vars:
        col_vals = df_yr[col].fillna(df_mapped[col].median())
        X_data[col] = col_vals.values
    X = X_data.values.astype(float)
    
    # 잔여 NaN 처리
    nan_mask = np.isnan(y) | np.any(np.isnan(X), axis=1)
    if nan_mask.sum() > 0:
        y[nan_mask] = y_median
        for j in range(X.shape[1]):
            col_median = np.nanmedian(X[:, j])
            X[nan_mask, j] = col_median
    
    try:
        # OLS
        ols = OLS(y.reshape(-1,1), X, w=W_queen,
                  name_y='net_rate', name_x=X_vars, spat_diag=True)
        
        # Spatial Lag Model (SLM)
        slm = ML_Lag(y.reshape(-1,1), X, w=W_queen,
                     name_y='net_rate', name_x=X_vars)
        
        # Spatial Error Model (SEM)
        sem = ML_Error(y.reshape(-1,1), X, w=W_queen,
                       name_y='net_rate', name_x=X_vars)
        
        # 직접/간접/총효과 분해 (SLM 기반)
        rho = slm.rho
        # betas: [const, x1, ..., xk, rho] → x 계수는 인덱스 1~k
        betas_x = slm.betas[1:len(X_vars)+1].flatten()
        
        n = len(y)
        W_dense = W_queen.full()[0]
        I_n = np.eye(n)
        
        # S = (I - rho*W)^{-1}
        try:
            S = np.linalg.inv(I_n - rho * W_dense)
            for i, var in enumerate(X_vars):
                S_r = betas_x[i] * S
                direct_eff = float(np.mean(np.diag(S_r)))
                total_eff = float(np.mean(S_r.sum(axis=1)))
                indirect_eff = total_eff - direct_eff
                direct_indirect_all.append({
                    'year': yr, 'variable': var,
                    'direct': direct_eff,
                    'indirect': indirect_eff,
                    'total': total_eff,
                    'indirect_ratio': indirect_eff / total_eff if total_eff != 0 else 0
                })
        except np.linalg.LinAlgError:
            print(f"  {yr}: 행렬 역산 실패")
        
        # LM 검정 결과 추출
        lm_lag = getattr(ols, 'lm_lag', [None, None])
        lm_err = getattr(ols, 'lm_error', [None, None])
        
        sdm_summary.append({
            'year': yr,
            'OLS_R2': round(ols.r2, 4),
            'SLM_rho': round(rho, 4),
            'SLM_PseudoR2': round(slm.pr2, 4),
            'SLM_AIC': round(slm.aic, 2),
            'SEM_lambda': round(sem.lam, 4),
            'SEM_PseudoR2': round(sem.pr2, 4),
            'SEM_AIC': round(sem.aic, 2),
            'LM_lag_stat': round(lm_lag[0], 4) if lm_lag[0] else None,
            'LM_lag_pval': round(lm_lag[1], 4) if lm_lag[1] else None,
            'LM_err_stat': round(lm_err[0], 4) if lm_err[0] else None,
            'LM_err_pval': round(lm_err[1], 4) if lm_err[1] else None,
        })
        
        print(f"\n[{yr}년] n={len(y)}")
        print(f"  OLS R²={ols.r2:.4f}")
        print(f"  SLM ρ={rho:.4f}, Pseudo-R²={slm.pr2:.4f}, AIC={slm.aic:.1f}")
        print(f"  SEM λ={sem.lam:.4f}, Pseudo-R²={sem.pr2:.4f}, AIC={sem.aic:.1f}")
        
    except Exception as e:
        print(f"  {yr} 오류: {e}")

# 결과 저장
sdm_df = pd.DataFrame(sdm_summary)
sdm_df.to_csv(f'{TABS}/table_sdm_comparison.csv', index=False)

di_df = pd.DataFrame(direct_indirect_all)
if not di_df.empty:
    di_df.to_csv(f'{TABS}/table_sdm_effects.csv', index=False)
    print(f"\n직접/간접효과 저장: {len(di_df)}행")

# ─────────────────────────────────────────────────────────────────────────
# 강건성 검증
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("강건성 검증: 다중 중심성 지표")
print("─" * 50)

robust_rows = []
cent_vars = ['pagerank', 'in_deg_cent', 'betweenness', 'in_strength']
ctrl_vars = ['youth_ratio', 'aging_ratio', 'fiscal_indep', 'employ_rate', 'ln_pop_density']

for yr in [2019, 2023]:
    df_yr = df_mapped[df_mapped['year'] == yr].copy()
    df_yr = df_yr.set_index('gdf_code').reindex(gdf['code'])
    y = df_yr['net_rate'].fillna(df_mapped['net_rate'].median()).values.astype(float)
    
    for cent in cent_vars:
        try:
            X_ctrl = pd.DataFrame(index=gdf['code'])
            for col in ctrl_vars:
                X_ctrl[col] = df_yr[col].fillna(df_mapped[col].median()).values
            cent_col = df_yr[cent].fillna(df_mapped[cent].median()).values.reshape(-1,1)
            X_full = np.hstack([cent_col, X_ctrl.values]).astype(float)
            
            # NaN 처리
            nan_mask = np.isnan(y) | np.any(np.isnan(X_full), axis=1)
            if nan_mask.sum() > 0:
                y_tmp = y.copy()
                X_tmp = X_full.copy()
                y_tmp[nan_mask] = np.nanmedian(y)
                for j in range(X_tmp.shape[1]):
                    X_tmp[nan_mask, j] = np.nanmedian(X_tmp[:, j])
            else:
                y_tmp, X_tmp = y, X_full
            
            slm = ML_Lag(y_tmp.reshape(-1,1), X_tmp, w=W_queen,
                         name_y='net_rate', name_x=[cent]+ctrl_vars)
            
            robust_rows.append({
                'year': yr, 'centrality': cent,
                'rho': round(slm.rho, 4),
                'beta_cent': round(float(slm.betas[1]), 4),
                'pseudo_r2': round(slm.pr2, 4),
                'aic': round(slm.aic, 2)
            })
            print(f"  {yr} {cent}: ρ={slm.rho:.4f}, β={float(slm.betas[1]):.4f}, R²={slm.pr2:.4f}")
        except Exception as e:
            print(f"  {yr} {cent} 오류: {e}")

robust_df = pd.DataFrame(robust_rows)
if not robust_df.empty:
    robust_df.to_csv(f'{TABS}/table_robustness_centrality.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# STEP 4: 머신러닝 기반 지역 흡인력 분석
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("STEP 4: 머신러닝 기반 지역 흡인력 분석")
print("─" * 50)

# 특성 변수 목록
feature_cols = [
    # 네트워크 지표
    'pagerank', 'in_deg_cent', 'betweenness', 'closeness',
    # 인구 구조
    'youth_ratio', 'aging_ratio', 'pop_density', 'ln_population',
    # 경제
    'fiscal_indep', 'employ_rate', 'ln_biz_count', 'ln_worker_count', 'grdp_pc',
    # 교육/의료/생활SOC
    'doctor_per1000', 'univ_count', 'culture_facility_count',
    # 교통/접근성
    'seoul_dist_km',
    # 지역 특성
    'metro_dummy', 'large_city_dummy'
]

# 실제 존재하는 컬럼만 사용
available_features = [c for c in feature_cols if c in df_mapped.columns]
print(f"사용 특성 변수: {len(available_features)}개")

# ML 데이터셋 준비
df_ml = df_mapped[['region_code', 'sgg_name', 'year', 'net_rate'] + available_features].copy()
df_ml = df_ml.dropna(subset=['net_rate'])

# 결측값 처리 (중앙값 대체)
for col in available_features:
    df_ml[col] = df_ml[col].fillna(df_ml[col].median())

X_ml = df_ml[available_features].values
y_ml = df_ml['net_rate'].values
groups_ml = df_ml['region_code'].values  # GroupKFold용: 지역별 그룹 (시간 누수 방지)

print(f"ML 데이터셋: {X_ml.shape}")

# 모델 정의
models = {
    'LR': LinearRegression(),
    'DT': DecisionTreeRegressor(max_depth=6, random_state=42),
    'RF': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'GB': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'XGB': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, 
                             random_state=42, verbosity=0),
    'LGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, verbose=-1),
}

# 5-Fold Group CV 성능 평가 (지역 누수 방지: 같은 지역이 train/test에 동시 등장하지 않도록)
# GroupKFold: 지역(region_code)을 그룹으로 사용 → 시간 정보 누출(data leakage) 방지
gkf = GroupKFold(n_splits=5)
ml_results = []

print("\n5-Fold Group CV 성능 평가 (지역 누수 방지):")
for name, model in models.items():
    rmse_list, mae_list, r2_list = [], [], []
    
    for train_idx, test_idx in gkf.split(X_ml, y_ml, groups=groups_ml):
        X_tr, X_te = X_ml[train_idx], X_ml[test_idx]
        y_tr, y_te = y_ml[train_idx], y_ml[test_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        rmse_list.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae_list.append(mean_absolute_error(y_te, y_pred))
        r2_list.append(r2_score(y_te, y_pred))
    
    result = {
        'Model': name,
        'RMSE_mean': round(np.mean(rmse_list), 4),
        'RMSE_std': round(np.std(rmse_list), 4),
        'MAE_mean': round(np.mean(mae_list), 4),
        'MAE_std': round(np.std(mae_list), 4),
        'R2_mean': round(np.mean(r2_list), 4),
        'R2_std': round(np.std(r2_list), 4),
    }
    ml_results.append(result)
    print(f"  {name:6s}: RMSE={result['RMSE_mean']:.4f}±{result['RMSE_std']:.4f}, "
          f"MAE={result['MAE_mean']:.4f}, R²={result['R2_mean']:.4f}±{result['R2_std']:.4f}")

ml_df = pd.DataFrame(ml_results)
ml_df.to_csv(f'{TABS}/table_ml_performance.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# SHAP 분석 (최고 성능 모델: LGBM)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("SHAP 분석 (LightGBM)")
print("─" * 50)

best_model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                random_state=42, verbose=-1)
best_model.fit(X_ml, y_ml)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_ml)

# SHAP 중요도 저장
shap_importance = pd.DataFrame({
    'feature': available_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
shap_importance.to_csv(f'{TABS}/table_shap_importance.csv', index=False)
print("\nSHAP 변수 중요도 Top 10:")
print(shap_importance.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
# STEP 5: 지역 흡인력 지수 (RAI) 구축
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("STEP 5: 지역 흡인력 지수 (RAI) 구축")
print("─" * 50)

# SHAP 기반 RAI: 각 관측치의 예측값에서 기준값(평균) 대비 기여도 합산
# RAI = 예측 순이동률 (LGBM 예측값 기반)
df_ml['rai_raw'] = best_model.predict(X_ml)

# 연도별 표준화 (Z-score)
df_ml['rai_zscore'] = df_ml.groupby('year')['rai_raw'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# 5년 평균 RAI (2019-2023)
rai_recent = df_ml[df_ml['year'].between(2019, 2023)].groupby(
    ['region_code', 'sgg_name']
)['rai_zscore'].mean().reset_index()
rai_recent.columns = ['region_code', 'sgg_name', 'rai_5yr_avg']

# 지역 유형 분류 (4분위)
rai_recent['rai_quartile'] = pd.qcut(rai_recent['rai_5yr_avg'], q=4, 
                                      labels=['Q1_저흡인', 'Q2_중저흡인', 'Q3_중고흡인', 'Q4_고흡인'])

# 추가 분류: 수도권/비수도권
metro_codes = df_mapped[df_mapped['metro_dummy'] == 1]['region_code'].unique()
rai_recent['region_type'] = rai_recent['region_code'].apply(
    lambda x: '수도권' if x in metro_codes else '비수도권'
)

rai_recent.to_csv(f'{TABS}/table_rai_scores.csv', index=False)
print(f"\nRAI 구축 완료: {len(rai_recent)}개 지역")
print("\nRAI 분위별 분포:")
print(rai_recent['rai_quartile'].value_counts().sort_index())
print("\nTop 20 고흡인 지역 (2019-2023 평균):")
print(rai_recent.nlargest(20, 'rai_5yr_avg')[['sgg_name', 'rai_5yr_avg', 'rai_quartile', 'region_type']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────
print("\n시각화 생성 중...")

# ① SDM 모형 비교 그래프
if not sdm_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spatial Model Comparison: OLS vs. SLM vs. SEM", 
                 fontsize=13, fontweight='bold')
    
    years_plot = sdm_df['year'].tolist()
    x = np.arange(len(years_plot))
    w = 0.25
    
    ax = axes[0]
    ax.bar(x - w, sdm_df['OLS_R2'], w, label='OLS R²', color='#607D8B', alpha=0.85)
    ax.bar(x, sdm_df['SLM_PseudoR2'], w, label='SLM Pseudo-R²', color='#2196F3', alpha=0.85)
    ax.bar(x + w, sdm_df['SEM_PseudoR2'], w, label='SEM Pseudo-R²', color='#FF5722', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years_plot])
    ax.set_ylabel('R² / Pseudo-R²')
    ax.set_title('모형 적합도 비교')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    ax = axes[1]
    ax.plot(years_plot, sdm_df['SLM_rho'], 'o-', color='#2196F3', 
            linewidth=2.5, markersize=9, label='SLM ρ (공간지체)')
    ax.plot(years_plot, sdm_df['SEM_lambda'], 's--', color='#FF5722', 
            linewidth=2.5, markersize=9, label='SEM λ (공간오차)')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
    ax.fill_between(years_plot, sdm_df['SLM_rho'], alpha=0.1, color='#2196F3')
    ax.set_xlabel('연도')
    ax.set_ylabel('공간 모수 추정값')
    ax.set_title('공간 의존성 모수 추이 (ρ, λ)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGS}/fig_sdm_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig_sdm_model_comparison.png 저장")

# ② 직접/간접/총효과 그래프
if not di_df.empty:
    yr_latest = di_df['year'].max()
    di_latest = di_df[di_df['year'] == yr_latest].copy()
    
    var_labels = {
        'pagerank': 'PageRank', 'youth_ratio': '청년 비율',
        'aging_ratio': '고령화율', 'fiscal_indep': '재정자립도',
        'employ_rate': '고용률', 'ln_pop_density': 'ln(인구밀도)',
        'doctor_per1000': '의사수/1000명'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    vars_list = di_latest['variable'].tolist()
    x = np.arange(len(vars_list))
    w = 0.28
    
    ax.bar(x - w, di_latest['direct'], w, label='직접효과', color='#1976D2', alpha=0.85)
    ax.bar(x, di_latest['indirect'], w, label='간접효과 (Spillover)', color='#FF9800', alpha=0.85)
    ax.bar(x + w, di_latest['total'], w, label='총효과', color='#388E3C', alpha=0.85)
    
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([var_labels.get(v, v) for v in vars_list], rotation=30, ha='right')
    ax.set_ylabel('효과 크기')
    ax.set_title(f'SLM 직접/간접/총효과 분해 ({yr_latest}년)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGS}/fig_sdm_direct_indirect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  fig_sdm_direct_indirect.png 저장")

# ③ ML 성능 비교 그래프
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Machine Learning Model Performance Comparison (5-Fold CV)", 
             fontsize=13, fontweight='bold')

metrics = [('RMSE_mean', 'RMSE_std', 'RMSE', '#E53935'),
           ('MAE_mean', 'MAE_std', 'MAE', '#FB8C00'),
           ('R2_mean', 'R2_std', 'R²', '#43A047')]

colors_ml = ['#607D8B', '#795548', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4']

for ax_idx, (mean_col, std_col, label, color) in enumerate(metrics):
    ax = axes[ax_idx]
    models_list = ml_df['Model'].tolist()
    values = ml_df[mean_col].tolist()
    errors = ml_df[std_col].tolist()
    
    bars = ax.bar(models_list, values, color=colors_ml[:len(models_list)], alpha=0.85,
                  yerr=errors, capsize=5, error_kw={'linewidth': 1.5})
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.1,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title(f'{label} (낮을수록 좋음)' if label != 'R²' else f'{label} (높을수록 좋음)')
    ax.set_ylabel(label)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f'{FIGS}/fig_ml_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_ml_performance.png 저장")

# ④ SHAP Summary Plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_ml, feature_names=available_features, 
                  show=False, max_display=15)
plt.title("SHAP Summary Plot: 지역 흡인력 결정요인 (LightGBM)", 
          fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{FIGS}/fig_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_shap_summary.png 저장")

# ⑤ SHAP 변수 중요도 막대 그래프
fig, ax = plt.subplots(figsize=(10, 7))
top_n = min(15, len(shap_importance))
top_feats = shap_importance.head(top_n)

colors_shap = ['#E53935' if 'pagerank' in f or 'cent' in f or 'strength' in f 
               else '#1E88E5' for f in top_feats['feature']]

bars = ax.barh(range(top_n), top_feats['mean_abs_shap'].values, color=colors_shap, alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats['feature'].tolist(), fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('지역 흡인력 결정요인 중요도 (SHAP)\n빨간색: 네트워크 지표, 파란색: 지역 특성 변수', 
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E53935', alpha=0.85, label='네트워크 지표'),
                   Patch(facecolor='#1E88E5', alpha=0.85, label='지역 특성 변수')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIGS}/fig_shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_shap_importance.png 저장")

# ⑥ RAI 분포 및 지역 유형 그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("지역 흡인력 지수 (RAI) 분포 및 지역 유형 분류 (2019-2023 평균)", 
             fontsize=13, fontweight='bold')

# RAI 분포 히스토그램
ax = axes[0]
ax.hist(rai_recent['rai_5yr_avg'], bins=30, color='#1976D2', alpha=0.75, edgecolor='white')
ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--', label='평균')
ax.axvline(x=rai_recent['rai_5yr_avg'].quantile(0.75), color='orange', 
           linewidth=1.5, linestyle=':', label='Q3 (고흡인 경계)')
ax.set_xlabel('RAI (표준화 점수)')
ax.set_ylabel('지역 수')
ax.set_title('RAI 분포')
ax.legend()
ax.grid(alpha=0.3)

# 수도권/비수도권별 RAI 분포
ax = axes[1]
metro_rai = rai_recent[rai_recent['region_type'] == '수도권']['rai_5yr_avg']
non_metro_rai = rai_recent[rai_recent['region_type'] == '비수도권']['rai_5yr_avg']
ax.boxplot([metro_rai, non_metro_rai], labels=['수도권', '비수도권'],
           patch_artist=True,
           boxprops=dict(facecolor='#BBDEFB', color='#1976D2'),
           medianprops=dict(color='#E53935', linewidth=2))
ax.set_ylabel('RAI (표준화 점수)')
ax.set_title('수도권 vs. 비수도권 RAI 분포')
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGS}/fig_rai_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig_rai_distribution.png 저장")

print("\n" + "=" * 70)
print("전체 분석 완료!")
print("=" * 70)
print(f"\n생성된 테이블:")
for f in ['table_sdm_comparison.csv', 'table_sdm_effects.csv', 
          'table_robustness_centrality.csv', 'table_ml_performance.csv',
          'table_shap_importance.csv', 'table_rai_scores.csv']:
    import os
    path = f'{TABS}/{f}'
    if os.path.exists(path):
        print(f"  ✓ {f}")

print(f"\n생성된 그래프:")
for f in ['fig_sdm_model_comparison.png', 'fig_sdm_direct_indirect.png',
          'fig_ml_performance.png', 'fig_shap_summary.png',
          'fig_shap_importance.png', 'fig_rai_distribution.png']:
    path = f'{FIGS}/{f}'
    if os.path.exists(path):
        print(f"  ✓ {f}")
