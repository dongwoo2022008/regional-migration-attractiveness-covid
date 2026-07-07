"""
Generate Figure 1 (Research Framework) and Figure 6 (Cluster Map proxy)
for the manuscript.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = '/home/ubuntu/regional-migration-attractiveness-covid/results/figures'
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# Figure 1. Research Framework (horizontal flow)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

def box(ax, x, y, w, h, text, fontsize=9, color='#f0f0f0', edgecolor='#333333', bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontfamily='DejaVu Sans', fontweight=weight,
            wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

# --- Row 1: Input → Model → SHAP ---
box(ax, 0.3, 3.5, 2.8, 1.8,
    "Municipal\nCharacteristics\n(20 variables,\n222 units, 2017–2024)",
    fontsize=8.5, color='#dce8f5', bold=False)

arrow(ax, 3.1, 4.4, 3.7, 4.4)

box(ax, 3.7, 3.5, 2.2, 1.8,
    "CatBoost\n(Optuna-tuned)\nExpanding-window CV\nTest R²=0.29",
    fontsize=8.5, color='#dce8f5', bold=False)

arrow(ax, 5.9, 4.4, 6.5, 4.4)

box(ax, 6.5, 3.5, 2.2, 1.8,
    "SHAP\nImportance\n+\nDirection",
    fontsize=8.5, color='#dce8f5', bold=False)

# --- Row 2: Three RQ boxes ---
arrow(ax, 7.6, 3.5, 7.6, 3.0)
arrow(ax, 7.6, 3.0, 4.5, 3.0)
arrow(ax, 7.6, 3.0, 7.6, 3.0)
arrow(ax, 7.6, 3.0, 10.8, 3.0)

# RQ1
arrow(ax, 4.5, 3.0, 4.5, 2.5)
box(ax, 3.0, 0.4, 3.0, 2.0,
    "RQ1\nSpatial Structure\n\nCluster Typology\nLOWESS\nDual Decline",
    fontsize=8.5, color='#e8f5e9', bold=True)

# RQ2
arrow(ax, 7.6, 3.0, 7.6, 2.5)
box(ax, 6.1, 0.4, 3.0, 2.0,
    "RQ2\nConditional\nNetwork Effect\n\nPageRank rank:\nGrowth=4, Mature=13",
    fontsize=8.5, color='#fff8e1', bold=True)

# RQ3
arrow(ax, 10.8, 3.0, 10.8, 2.5)
box(ax, 9.3, 0.4, 3.0, 2.0,
    "RQ3\nMethodological\nCaution\n\nImportance ≠ Direction\n≠ Policy Effect",
    fontsize=8.5, color='#fce4ec', bold=True)

# --- Conclusion box ---
arrow(ax, 10.8, 0.4, 10.8, 0.2)
arrow(ax, 4.5, 0.4, 4.5, 0.2)
ax.annotate('', xy=(12.5, 0.2), xytext=(3.0, 0.2),
            arrowprops=dict(arrowstyle='-', color='#333333', lw=1.0))
arrow(ax, 7.75, 0.2, 7.75, 0.05)

box(ax, 5.5, -0.05, 4.5, 0.45,
    "Spatial Polarization: Capital concentration + Suburbanization + Dual Decline",
    fontsize=8.5, color='#ede7f6', bold=True)

ax.set_title("Figure 1. Research Framework: From Municipal Characteristics to Spatial Polarization",
             fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_research_framework.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ─────────────────────────────────────────────
# Figure 6. Cluster Characteristics Bar Chart
# (proxy for cluster map using available data)
# ─────────────────────────────────────────────
import pandas as pd

cluster_data = {
    'Cluster': ['C2\nCapital-proximate\nGrowth', 'C0\nMixed', 'C1\nRural\nDecline', 'C3\nNon-capital\nMetro Decline'],
    'Net Rate (‰)': [3.0, -2.1, -2.8, -3.0],
    'Aging Ratio (%)': [14.7, 23.4, 34.0, 17.0],
    'House Age (yrs)': [17.3, 28.5, 44.9, 22.7],
    'Seoul Dist (km)': [35, 157, 229, 295],
    'n': [65, 56, 63, 40]
}
df = pd.DataFrame(cluster_data)

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
colors = ['#4CAF50', '#9E9E9E', '#FF9800', '#F44336']
labels = df['Cluster'].tolist()

metrics = [
    ('Net Rate (‰)', 'Net Migration Rate (‰)', True),
    ('Aging Ratio (%)', 'Aging Ratio (%)', False),
    ('House Age (yrs)', 'Mean Housing Age (years)', False),
    ('Seoul Dist (km)', 'Distance to Seoul (km)', False),
]

for ax, (col, ylabel, is_netrate) in zip(axes, metrics):
    vals = df[col].values
    bar_colors = colors if not is_netrate else ['#4CAF50' if v > 0 else '#F44336' for v in vals]
    bars = ax.bar(range(len(labels)), vals, color=bar_colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if val >= 0 else -2),
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    "Figure 6. Four Municipal Clusters Identifying Dual Decline and Capital-Proximate Growth\n"
    "(C1: Rural Decline; C3: Non-capital Metropolitan Decline; C2: Capital-proximate Growth)",
    fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_cluster_characteristics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ─────────────────────────────────────────────
# Figure 7. Stage-specific SHAP Importance (PageRank focus)
# ─────────────────────────────────────────────
stage_data = {
    'Feature': ['pagerank_lag1', 'childcare_pk', 'house_age', 'fertility', 'aging_ratio', 'hospital_bed'],
    'Growth': [0.869, 0.458, 1.294, 0.890, 0.733, 0.353],
    'Middle': [0.444, 3.170, 1.748, 0.889, 0.795, 1.169],
    'Mature': [0.546, 1.480, 1.357, 1.112, 0.489, 0.701],
}
df7 = pd.DataFrame(stage_data)

x = np.arange(len(df7['Feature']))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width, df7['Growth'], width, label='Growth Stage', color='#4CAF50', edgecolor='black', linewidth=0.8)
b2 = ax.bar(x, df7['Middle'], width, label='Middle Stage', color='#FFC107', edgecolor='black', linewidth=0.8)
b3 = ax.bar(x + width, df7['Mature'], width, label='Mature Stage', color='#F44336', edgecolor='black', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(['PageRank\n(lag 1)', 'Childcare\nDensity', 'Housing\nAge', 'Fertility\nRate', 'Aging\nRatio', 'Hospital\nBeds'], fontsize=9)
ax.set_ylabel('Mean |SHAP| Value', fontsize=10)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate PageRank ranks
ax.annotate('Rank 4', xy=(x[0] - width, df7['Growth'][0] + 0.03), ha='center', fontsize=8, color='darkgreen', fontweight='bold')
ax.annotate('Rank 18', xy=(x[0], df7['Middle'][0] + 0.03), ha='center', fontsize=8, color='darkorange', fontweight='bold')
ax.annotate('Rank 13', xy=(x[0] + width, df7['Mature'][0] + 0.03), ha='center', fontsize=8, color='darkred', fontweight='bold')

ax.set_title(
    "Figure 7. Network Centrality (PageRank) Matters Primarily for Growing Municipalities:\n"
    "Stage-Conditional SHAP Importance Across Urban Development Stages",
    fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_stage_shap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

print("\nAll new figures saved to:", OUT)
