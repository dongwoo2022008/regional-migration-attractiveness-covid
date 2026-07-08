# -*- coding: utf-8 -*-
"""Cluster number selection for place-type typology (Task 3).

Evaluates k = 2–8 using three criteria:
  1. Elbow method (inertia)
  2. Silhouette coefficient
  3. Calinski–Harabasz index

Input:  data/track_C_2017_2024.csv
Output: results/tables/cluster_selection_metrics.csv
        results/figures/cluster_selection.png

Run:
  python scripts/ml/cluster_selection.py --data data/track_C_2017_2024.csv
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

warnings.filterwarnings("ignore")

# Variables used in lifecycle_typology.py (TYPO list)
TYPO = [
    "pop_density", "house_age", "aging_ratio", "seoul_dist_km",
    "childcare_pk", "ln_pop", "youth_ratio", "fiscal_indep", "extinction_risk"
]
TAB = "results/tables"
FIG = "results/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/track_C_2017_2024.csv")
    a = ap.parse_args()

    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    # ── Load and prepare data ────────────────────────────────────────────
    df = pd.read_csv(a.data)
    keep = [c for c in TYPO if c in df.columns]
    missing = [c for c in TYPO if c not in df.columns]
    if missing:
        print(f"Warning: variables not found in data: {missing}")

    # Region-level averages (same as lifecycle_typology.py)
    reg = df.groupby("region_code")[keep].mean().reset_index()
    reg = reg.dropna(subset=keep)
    print(f"Regions after dropna: {len(reg)}")

    # Standardize
    Z = StandardScaler().fit_transform(reg[keep])

    # ── Compute metrics for k = 2–8 ─────────────────────────────────────
    k_range = range(2, 9)
    inertias, silhouettes, calinski_scores = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Z)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Z, labels))
        calinski_scores.append(calinski_harabasz_score(Z, labels))
        print(f"  k={k}: inertia={km.inertia_:.1f}, "
              f"silhouette={silhouettes[-1]:.4f}, "
              f"calinski={calinski_scores[-1]:.2f}")

    # ── Save metrics table ───────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "k": list(k_range),
        "inertia": [round(v, 2) for v in inertias],
        "silhouette": [round(v, 4) for v in silhouettes],
        "calinski_harabasz": [round(v, 2) for v in calinski_scores],
    })
    metrics_df.to_csv(f"{TAB}/cluster_selection_metrics.csv", index=False)
    print(f"\nMetrics saved to {TAB}/cluster_selection_metrics.csv")
    print(metrics_df.to_string(index=False))

    # ── Identify optimal k per criterion ────────────────────────────────
    best_sil_k = list(k_range)[np.argmax(silhouettes)]
    best_cal_k = list(k_range)[np.argmax(calinski_scores)]

    # Elbow: largest second difference in inertia
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_k = list(k_range)[np.argmax(np.abs(second_diffs)) + 1]

    print(f"\nOptimal k per criterion:")
    print(f"  Elbow (largest curvature): k = {elbow_k}")
    print(f"  Silhouette (max):          k = {best_sil_k} ({max(silhouettes):.4f})")
    print(f"  Calinski–Harabasz (max):   k = {best_cal_k} ({max(calinski_scores):.2f})")

    supports_k4 = (elbow_k == 4 and best_sil_k == 4 and best_cal_k == 4)
    print(f"\nDo all three criteria support k=4? {'YES' if supports_k4 else 'NO'}")
    if not supports_k4:
        print("  → Discrepancy: manuscript states k=4 but criteria suggest different values.")
        print(f"    Elbow={elbow_k}, Silhouette={best_sil_k}, Calinski={best_cal_k}")

    # ── Plot: 3-panel figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = list(k_range)

    # Panel 1: Elbow
    axes[0].plot(ks, inertias, "o-", color="#2c3e50", linewidth=2, markersize=7)
    axes[0].axvline(elbow_k, color="#e74c3c", linestyle="--", alpha=0.7,
                    label=f"Elbow k={elbow_k}")
    axes[0].axvline(4, color="#3498db", linestyle=":", alpha=0.7, label="k=4 (used)")
    axes[0].set_xlabel("Number of clusters (k)", fontsize=11)
    axes[0].set_ylabel("Inertia (within-cluster SSE)", fontsize=11)
    axes[0].set_title("(A) Elbow Method", fontsize=12, fontweight="bold")
    axes[0].set_xticks(ks)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel 2: Silhouette
    axes[1].plot(ks, silhouettes, "s-", color="#27ae60", linewidth=2, markersize=7)
    axes[1].axvline(best_sil_k, color="#e74c3c", linestyle="--", alpha=0.7,
                    label=f"Max k={best_sil_k}")
    axes[1].axvline(4, color="#3498db", linestyle=":", alpha=0.7, label="k=4 (used)")
    axes[1].set_xlabel("Number of clusters (k)", fontsize=11)
    axes[1].set_ylabel("Silhouette Coefficient", fontsize=11)
    axes[1].set_title("(B) Silhouette Coefficient", fontsize=12, fontweight="bold")
    axes[1].set_xticks(ks)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel 3: Calinski–Harabasz
    axes[2].plot(ks, calinski_scores, "^-", color="#8e44ad", linewidth=2, markersize=7)
    axes[2].axvline(best_cal_k, color="#e74c3c", linestyle="--", alpha=0.7,
                    label=f"Max k={best_cal_k}")
    axes[2].axvline(4, color="#3498db", linestyle=":", alpha=0.7, label="k=4 (used)")
    axes[2].set_xlabel("Number of clusters (k)", fontsize=11)
    axes[2].set_ylabel("Calinski–Harabasz Index", fontsize=11)
    axes[2].set_title("(C) Calinski–Harabasz Index", fontsize=12, fontweight="bold")
    axes[2].set_xticks(ks)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.suptitle(
        "Figure A4. Cluster Number Selection Metrics (k = 2–8)\n"
        "Variables: pop_density, house_age, aging_ratio, seoul_dist_km, childcare_pk, "
        "ln_pop, youth_ratio, fiscal_indep, extinction_risk",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"{FIG}/cluster_selection.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\nFigure saved to {FIG}/cluster_selection.png")

    # ── Appendix Table A4 (English) ──────────────────────────────────────
    appendix = metrics_df.copy()
    appendix.columns = ["k", "Inertia (SSE)", "Silhouette Coefficient",
                        "Calinski–Harabasz Index"]
    appendix["Note"] = ""
    appendix.loc[appendix["k"] == elbow_k, "Note"] += "Elbow "
    appendix.loc[appendix["k"] == best_sil_k, "Note"] += "Silhouette "
    appendix.loc[appendix["k"] == best_cal_k, "Note"] += "Calinski "
    appendix.loc[appendix["k"] == 4, "Note"] += "(used)"
    appendix["Note"] = appendix["Note"].str.strip()
    appendix.to_csv(f"{TAB}/table_A4_cluster_selection.csv", index=False)
    print(f"\nAppendix Table A4 saved to {TAB}/table_A4_cluster_selection.csv")
    print(appendix.to_string(index=False))

    return metrics_df, elbow_k, best_sil_k, best_cal_k


if __name__ == "__main__":
    main()
