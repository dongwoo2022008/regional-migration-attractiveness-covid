# -*- coding: utf-8 -*-
"""Journal-ready figures for the spatial-heterogeneity paper (no embedded titles).

Outputs (results/figures/paper/):
  fig2.png  LOWESS density-migration curve (frac=0.5, municipality means)
  fig3.png  k-means typology on first two PCs (k=4, seed 42)
  fig5.png  stage-wise SHAP importance of PageRank(t-1) and closeness (rank labels)
  fig6.png  SHAP contributions vs density by development stage (4 predictors)
  fig7.png  childcare SHAP dependence coloured by distance to Seoul

Run:  python scripts/ml/make_paper_figures.py --data data/track_C_2017_2024.csv
"""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor, Pool

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8})

FEATS = ["pagerank_lag1", "closeness", "house_age", "childcare_pk", "pop_density",
         "seoul_dist_km", "ln_pop", "fertility", "doctor_per1000", "aging_ratio",
         "youth_ratio", "fiscal_indep", "employ_rate", "biz_count", "sewer_supply",
         "academy_pk", "culture_facility_count", "senior_fac_pk", "hospital_bed",
         "extinction_risk"]
TYPO = ["pop_density", "house_age", "aging_ratio", "seoul_dist_km", "childcare_pk",
        "ln_pop", "youth_ratio", "fiscal_indep", "extinction_risk"]
CB = dict(iterations=677, learning_rate=0.10884860813834339, depth=8,
          l2_leaf_reg=2.1524708091423577, random_state=42, verbose=0)
TARGET = "net_rate"
OUT = "results/figures/paper"


def fit(X, y):
    m = CatBoostRegressor(**CB); m.fit(X, y); return m


def shap_vals(m, X):
    return m.get_feature_importance(type="ShapValues", data=Pool(X))[:, :-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/track_C_2017_2024.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(a.data)

    # ---- Fig 2: LOWESS (municipality means, same preprocessing as lifecycle_typology) ----
    keep = [c for c in set(TYPO + [TARGET, "region_code", "year"]) if c in df.columns]
    d = df.dropna(subset=[c for c in keep if c != "year"]).copy()
    reg = d.groupby("region_code").mean(numeric_only=True).reset_index()
    x, y = reg["pop_density"].values, reg[TARGET].values
    lo = lowess(y, x, frac=0.5, return_sorted=True)
    plt.figure(figsize=(7.5, 5))
    plt.scatter(x, y, s=20, alpha=0.55, color="#7f8c8d", edgecolor="none")
    plt.plot(lo[:, 0], lo[:, 1], color="#c0392b", lw=2.5, label="LOWESS (frac = 0.5)")
    plt.axhline(0, color="#999999", ls="--", lw=0.9)
    plt.xscale("log")
    plt.xlabel("Population density (persons/km², log scale)")
    plt.ylabel("Net migration rate (‰)")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2.png", dpi=300); plt.close()

    # ---- Fig 3: typology PCA ----
    feats = [f for f in TYPO if f in reg.columns]
    Z = StandardScaler().fit_transform(reg[feats])
    km = KMeans(4, n_init=10, random_state=42).fit(Z)
    reg["cluster"] = km.labels_
    P = PCA(2).fit(Z); PP = P.transform(Z)
    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(PP[:, 0], PP[:, 1], c=reg[TARGET], cmap="RdBu", vmin=-20, vmax=20,
                     s=34, edgecolor="k", linewidth=0.3)
    for c in sorted(reg.cluster.unique()):
        cx, cy = PP[reg.cluster == c].mean(0)
        plt.text(cx, cy, f"C{c}", fontsize=14, fontweight="bold", ha="center", va="center")
    plt.colorbar(sc, label="Net migration rate (‰)")
    plt.xlabel(f"PC1 ({P.explained_variance_ratio_[0]*100:.0f}% of variance)")
    plt.ylabel(f"PC2 ({P.explained_variance_ratio_[1]*100:.0f}% of variance)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3.png", dpi=300); plt.close()

    # ---- stage assignment (lifecycle_verification logic) ----
    region_density = df.groupby("region_code")["pop_density"].mean()
    tert = pd.qcut(region_density, 3, labels=["growth", "middle", "mature"])
    df["stage"] = df["region_code"].map(tert.to_dict())
    d2 = df.dropna(subset=FEATS + [TARGET, "stage"])

    res = {}
    for st in ["growth", "middle", "mature"]:
        sub = d2[d2.stage == st]; tr = sub[sub.year <= 2022]
        m = fit(tr[FEATS], tr[TARGET]); sv = shap_vals(m, sub[FEATS])
        imp = np.abs(sv).mean(0); rank = (-imp).argsort().argsort() + 1
        res[st] = (imp, rank)
    pr, cl = FEATS.index("pagerank_lag1"), FEATS.index("closeness")

    # ---- Fig 5: stage-wise importance, two panels ----
    stages = ["Growth", "Middle", "Mature"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (idx, name) in zip(axes, [(pr, "PageRank (t−1)"), (cl, "Closeness centrality")]):
        vals = [res[s][0][idx] for s in ["growth", "middle", "mature"]]
        rks = [int(res[s][1][idx]) for s in ["growth", "middle", "mature"]]
        bars = ax.bar(stages, vals, color=["#4d4d4d", "#8c8c8c", "#bfbfbf"],
                      edgecolor="black", linewidth=0.6)
        for b, v, r in zip(bars, vals, rks):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"rank {r}",
                    ha="center", fontsize=10)
        ax.set_ylabel("Mean |SHAP|"); ax.set_xlabel("Development stage")
        ax.set_title(name, fontsize=12); ax.set_ylim(0, max(vals) * 1.18)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig5.png", dpi=300); plt.close()

    # ---- full model for Figs 6-7 ----
    tr3 = d2[d2.year <= 2022]
    mf = fit(tr3[FEATS], tr3[TARGET]); sv = shap_vals(mf, d2[FEATS])

    cols = {"growth": "#2166ac", "middle": "#f4a582", "mature": "#b2182b"}
    sel = [("pagerank_lag1", "PageRank (t−1)"), ("closeness", "Closeness centrality"),
           ("childcare_pk", "Childcare facilities"), ("house_age", "Housing age")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (f, label) in zip(axes.ravel(), sel):
        j = FEATS.index(f)
        for st in ["growth", "middle", "mature"]:
            g = d2[d2.stage == st]
            ax.scatter(g["pop_density"], sv[d2.stage.values == st, j], s=9, alpha=0.35,
                       color=cols[st], label=st.capitalize())
        ax.set_xscale("log"); ax.axhline(0, color="#555555", lw=0.8, ls="--")
        ax.set_xlabel("Population density (log scale)"); ax.set_ylabel(f"SHAP: {label}")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:3], l[:3], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(); plt.savefig(f"{OUT}/fig6.png", dpi=300, bbox_inches="tight"); plt.close()

    j = FEATS.index("childcare_pk")
    plt.figure(figsize=(7.5, 5))
    sc = plt.scatter(d2["childcare_pk"], sv[:, j], c=d2["seoul_dist_km"], cmap="coolwarm_r",
                     s=14, alpha=0.8)
    plt.colorbar(sc, label="Distance from Seoul (km)")
    plt.axhline(0, color="#555555", lw=0.8, ls="--")
    plt.xlabel("Childcare facilities (per 1,000 residents)")
    plt.ylabel("SHAP contribution of childcare facilities")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig7.png", dpi=300); plt.close()
    print("saved figures to", OUT)


if __name__ == "__main__":
    main()
