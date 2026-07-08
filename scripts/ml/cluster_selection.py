# -*- coding: utf-8 -*-
"""Cluster-number selection diagnostics for the place-type typology (Step 1, RQ1).

Reproduces the preprocessing of scripts/ml/lifecycle_typology.py (municipality-level
means of nine standardized indicators) and evaluates k = 2..8 with three criteria:
  - Elbow method (within-cluster sum of squares / inertia)
  - Silhouette coefficient
  - Calinski-Harabasz index

Outputs:
  results/tables/cluster_selection_metrics.csv
  results/figures/cluster_selection.png

Run:  python scripts/ml/cluster_selection.py --data data/track_C_2017_2024.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

TARGET = "net_rate"
TAB = "results/tables"
FIG = "results/figures"
# identical to the TYPO feature list in lifecycle_typology.py
TYPO = ["pop_density", "house_age", "aging_ratio", "seoul_dist_km", "childcare_pk",
        "ln_pop", "youth_ratio", "fiscal_indep", "extinction_risk"]
K_RANGE = range(2, 9)
SEED = 42


def load_region_means(path):
    """Same preprocessing as lifecycle_typology.load(): drop incomplete rows,
    then municipality-level means over the study period."""
    df = pd.read_csv(path)
    keep = [c for c in set(TYPO + [TARGET, "region_code", "year"]) if c in df.columns]
    df = df.dropna(subset=[c for c in keep if c != "year"]).copy()
    reg = df.groupby("region_code").mean(numeric_only=True).reset_index()
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/track_C_2017_2024.csv")
    a = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    reg = load_region_means(a.data)
    feats = [f for f in TYPO if f in reg.columns]
    Z = StandardScaler().fit_transform(reg[feats])
    print(f"regions={len(reg)}, features={len(feats)}")

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        rows.append({
            "k": k,
            "inertia_wcss": round(float(km.inertia_), 3),
            "silhouette": round(float(silhouette_score(Z, km.labels_)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(Z, km.labels_)), 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/cluster_selection_metrics.csv", index=False)
    print(out.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ks = out["k"]
    for ax, col, label in zip(
        axes,
        ["inertia_wcss", "silhouette", "calinski_harabasz"],
        ["Within-cluster SS (elbow)", "Silhouette coefficient", "Calinski-Harabasz index"],
    ):
        ax.plot(ks, out[col], "o-", color="#333333")
        ax.axvline(4, color="#999999", linestyle="--", linewidth=1)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG}/cluster_selection.png", dpi=200)
    print(f"saved: {TAB}/cluster_selection_metrics.csv, {FIG}/cluster_selection.png")


if __name__ == "__main__":
    main()
