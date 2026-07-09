# -*- coding: utf-8 -*-
"""Journal-ready figures for the spatial-heterogeneity paper (no embedded titles).

Style: SciencePlots ['science','no-latex'] (falls back to serif defaults).
Line/bar figures are exported as vector PDF + 400dpi PNG; dense scatter
figures are raster-only (vector files would bloat with thousands of points).

Outputs (results/figures/paper/):
  fig2.pdf/.png  LOWESS density-migration curve (frac=0.5, municipality means)
  fig3.png       k-means typology on first two PCs (k=4, seed 42)
  fig5.pdf/.png  stage-wise SHAP importance of PageRank(t-1) and closeness
  fig6.png       SHAP contributions vs density by development stage (4 predictors)
  fig7.png       childcare SHAP dependence coloured by distance to Seoul

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

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    plt.rcParams.update({"font.family": "serif",
                         "xtick.direction": "in", "ytick.direction": "in"})
plt.rcParams.update({"font.size": 9})

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


def save(fig, name, vector=False):
    if vector:
        fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(x, y, s=10, alpha=0.5, color="#666666", edgecolor="none")
    ax.plot(lo[:, 0], lo[:, 1], lw=1.8, color="#b2182b", label="LOWESS (frac = 0.5)")
    ax.axhline(0, color="#999999", ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Population density (persons/km$^2$, log scale)")
    ax.set_ylabel("Net migration rate (‰)")
    ax.legend(frameon=False)
    save(fig, "fig2", vector=True)

    # ---- Fig 3: typology PCA ----
    feats = [f for f in TYPO if f in reg.columns]
    Z = StandardScaler().fit_transform(reg[feats])
    km = KMeans(4, n_init=10, random_state=42).fit(Z)
    reg["cluster"] = km.labels_
    P = PCA(2).fit(Z); PP = P.transform(Z)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(PP[:, 0], PP[:, 1], c=reg[TARGET], cmap="RdBu", vmin=-20, vmax=20,
                    s=26, edgecolor="k", linewidth=0.3)
    for c in sorted(reg.cluster.unique()):
        cx, cy = PP[reg.cluster == c].mean(0)
        ax.text(cx, cy, f"C{c}", fontsize=11, fontweight="bold", ha="center", va="center")
    fig.colorbar(sc, label="Net migration rate (‰)")
    ax.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]*100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]*100:.0f}% of variance)")
    save(fig, "fig3")

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
    assert [int(res[s][1][pr]) for s in ["growth", "middle", "mature"]] == [4, 18, 13]
    assert [int(res[s][1][cl]) for s in ["growth", "middle", "mature"]] == [10, 1, 8]

    # ---- Fig 5: stage-wise importance, two panels ----
    stages = ["Growth", "Middle", "Mature"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (idx, name) in zip(axes, [(pr, "PageRank (t$-$1)"), (cl, "Closeness centrality")]):
        vals = [res[s][0][idx] for s in ["growth", "middle", "mature"]]
        rks = [int(res[s][1][idx]) for s in ["growth", "middle", "mature"]]
        bars = ax.bar(stages, vals, color=["#4d4d4d", "#8c8c8c", "#c9c9c9"],
                      edgecolor="black", linewidth=0.5)
        for b, v, r in zip(bars, vals, rks):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, f"rank {r}",
                    ha="center", fontsize=8)
        ax.set_ylabel("Mean $|$SHAP$|$"); ax.set_xlabel("Development stage")
        ax.set_title(name, fontsize=9); ax.set_ylim(0, max(vals) * 1.2)
    save(fig, "fig5", vector=True)

    # ---- full model for Figs 6-7 ----
    tr3 = d2[d2.year <= 2022]
    mf = fit(tr3[FEATS], tr3[TARGET]); sv = shap_vals(mf, d2[FEATS])

    cols = {"growth": "#2166ac", "middle": "#f4a582", "mature": "#b2182b"}
    sel = [("pagerank_lag1", "PageRank (t$-$1)"), ("closeness", "Closeness centrality"),
           ("childcare_pk", "Childcare facilities"), ("house_age", "Housing age")]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, (f, label) in zip(axes.ravel(), sel):
        j = FEATS.index(f)
        for st in ["growth", "middle", "mature"]:
            g = d2[d2.stage == st]
            ax.scatter(g["pop_density"], sv[d2.stage.values == st, j], s=6, alpha=0.35,
                       color=cols[st], label=st.capitalize())
        ax.set_xscale("log"); ax.axhline(0, color="#555555", lw=0.6, ls="--")
        ax.set_xlabel("Population density (log scale)"); ax.set_ylabel(f"SHAP: {label}")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:3], l[:3], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    save(fig, "fig6")

    j = FEATS.index("childcare_pk")
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    sc = ax.scatter(d2["childcare_pk"], sv[:, j], c=d2["seoul_dist_km"], cmap="coolwarm_r",
                    s=9, alpha=0.8)
    fig.colorbar(sc, label="Distance from Seoul (km)")
    ax.axhline(0, color="#555555", lw=0.6, ls="--")
    ax.set_xlabel("Childcare facilities (per 1,000 residents)")
    ax.set_ylabel("SHAP contribution of childcare facilities")
    save(fig, "fig7")
    print("saved figures to", OUT)


if __name__ == "__main__":
    main()
