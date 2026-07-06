"""
Figure 4-14 (ML 성능 비교) + Figure 4-15 (SHAP Feature Importance) 재생성
- Figure 4-14: CatBoost 빨간색 강조 유지 (최적 모형 표시), 파일명 통일
- Figure 4-15: pagerank_lag1 단독 빨간색 제거 → 모든 막대 단일 색상(#1f77b4) 통일
"""

import pandas as pd
import numpy as np
import shap
import json
import os
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ── 폰트 설정 ─────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

FIGDIR = '/home/ubuntu/korea-migration-network/results/figures'
TABDIR = '/home/ubuntu/korea-migration-network/results/tables'

# ── 1. 데이터 및 모형 로드 ────────────────────────────────────
df = pd.read_csv('/home/ubuntu/korea-migration-network/data/analysis_dataset_FINAL_v4.csv')
df = df.sort_values(['region_code', 'year']).reset_index(drop=True)
df['pagerank_lag1'] = df.groupby('region_code')['pagerank'].shift(1)

with open(f'{TABDIR}/feature_cols.json') as f:
    FEATS = json.load(f)
with open(f'{TABDIR}/best_model_name.txt') as f:
    best_model_name = f.read().strip()

model = joblib.load(f'{TABDIR}/best_model.joblib')
print(f"최적 모형: {best_model_name}")
print(f"피처 수: {len(FEATS)}")

# ── 2. 분석 데이터 준비 ───────────────────────────────────────
d = df[(df['year'] >= 2017) & (df['year'] <= 2024)].dropna(subset=['net_rate']).copy()
for c in FEATS:
    d[c] = d.groupby('year')[c].transform(lambda s: s.fillna(s.median()))
    d[c] = d[c].fillna(d[c].median())

X_all = d[FEATS].values

# ── 3. SHAP 계산 ──────────────────────────────────────────────
print("SHAP 계산 중...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_all)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': FEATS,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print("\nTop 10 SHAP 중요도:")
print(shap_df.head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# Figure 4-15: SHAP Feature Importance (단일 색상 통일)
# ═══════════════════════════════════════════════════════════════
print("\nFigure 4-15 재생성 중 (pagerank 하이라이트 제거)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# (a) Bar plot — 모든 막대 단일 색상 (#1f77b4)
top10 = shap_df.head(10)
SINGLE_BLUE = '#1f77b4'
colors_uniform = [SINGLE_BLUE] * 10

axes[0].barh(range(10), top10['mean_abs_shap'].values[::-1],
             color=colors_uniform, edgecolor='white', linewidth=0.5)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(top10['feature'].values[::-1], fontsize=11)
axes[0].set_xlabel('Mean |SHAP value|', fontsize=12)
axes[0].set_title(f'(a) Feature Importance — {best_model_name} (lag1)',
                  fontsize=13, fontweight='bold')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(axis='x', alpha=0.3)

# 값 레이블 추가
for i, val in enumerate(top10['mean_abs_shap'].values[::-1]):
    axes[0].text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

# (b) Beeswarm scatter (Top 5)
top5_feats = shap_df.head(5)['feature'].tolist()
feat_idx = [FEATS.index(f) for f in top5_feats]

for i, (fi, fname) in enumerate(zip(feat_idx, top5_feats)):
    feat_vals = X_all[:, fi]
    shap_vals = shap_values[:, fi]
    vmin, vmax = np.percentile(feat_vals, [5, 95])
    norm_vals = np.clip((feat_vals - vmin) / (vmax - vmin + 1e-8), 0, 1)
    sc = axes[1].scatter(shap_vals, [i] * len(shap_vals),
                         c=norm_vals, cmap='coolwarm', alpha=0.4, s=10)

axes[1].set_yticks(range(5))
axes[1].set_yticklabels(top5_feats, fontsize=11)
axes[1].set_xlabel('SHAP value (impact on model output)', fontsize=12)
axes[1].set_title('(b) SHAP Beeswarm — Top 5 Features',
                  fontsize=13, fontweight='bold')
axes[1].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.colorbar(sc, ax=axes[1], label='Feature value (normalized)')

plt.suptitle(f'Figure 4-15. SHAP Feature Importance & Beeswarm — {best_model_name} (Track C, lag1)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig4_15_shap_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figure 4-15 저장: {FIGDIR}/fig4_15_shap_summary.png")

# ═══════════════════════════════════════════════════════════════
# Figure 4-14: ML 성능 비교 (CatBoost 빨간색 유지, 파일명 통일)
# ═══════════════════════════════════════════════════════════════
print("\nFigure 4-14 재생성 중...")

perf_path = f'{TABDIR}/table4_16_ml_performance.csv'
if not os.path.exists(perf_path):
    # 대체 경로 탐색
    for alt in [f'{TABDIR}/table4_16_ml_performance_trackC.csv',
                f'{TABDIR}/table_ml_performance.csv']:
        if os.path.exists(alt):
            perf_path = alt
            break

perf_df = pd.read_csv(perf_path)
print(f"성능 데이터 로드: {perf_path}")
print(perf_df.to_string(index=False))

# 컬럼명 정규화
col_map = {}
for c in perf_df.columns:
    cl = c.lower()
    if 'model' in cl:
        col_map[c] = 'model'
    elif 'test_r2' in cl or 'test r2' in cl:
        col_map[c] = 'Test_R2'
    elif 'test_rmse' in cl or 'test rmse' in cl:
        col_map[c] = 'Test_RMSE'
    elif 'test_mae' in cl or 'test mae' in cl:
        col_map[c] = 'Test_MAE'
perf_df = perf_df.rename(columns=col_map)

# 필요 컬럼 확인
required = ['model', 'Test_R2', 'Test_RMSE', 'Test_MAE']
for r in required:
    if r not in perf_df.columns:
        print(f"WARNING: column '{r}' not found. Available: {perf_df.columns.tolist()}")

colors_bar = ['#d62728' if str(m) == best_model_name else '#1f77b4'
              for m in perf_df['model']]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [('Test_R2', 'Test R²'), ('Test_RMSE', 'Test RMSE'), ('Test_MAE', 'Test MAE')]

for ax, (col, label) in zip(axes, metrics):
    if col not in perf_df.columns:
        ax.set_title(f'{label} (data unavailable)')
        continue
    vals = perf_df[col].values
    bars = ax.bar(perf_df['model'], vals, color=colors_bar,
                  edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        offset = 0.005 * max(abs(vals.max()), abs(vals.min()), 0.01)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    if col == 'Test_R2':
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

plt.suptitle(
    f'Figure 4-14. ML Model Performance Comparison (lag1, Hold-out Test 2023–2024)\n'
    f'Red bar = {best_model_name} (best model)',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()

# 두 파일명 모두 저장 (문서 링크 호환)
fig.savefig(f'{FIGDIR}/fig4_14_ml_performance.png',
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{FIGDIR}/fig4_14_ml_performance_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figure 4-14 저장 완료 (두 파일명 모두 저장)")

print("\n=== 재생성 완료 ===")
