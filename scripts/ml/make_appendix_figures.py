# -*- coding: utf-8 -*-
"""Appendix figures and manuscript maps (SciencePlots style).

Outputs (results/figures/paper/):
  fig2_model_perf   candidate-model performance bars (Val/Test R2, RMSE)
  fig4_migration_map  choropleth of mean municipal net migration rate 2017-2024
  fig6_cluster_map    choropleth of the four k-means clusters
  fig8_stage_heatmap  20-predictor x 3-stage SHAP-importance rank heatmap
  figA2 bootstrap SHAP-rank CIs   figA3 LOWESS span sensitivity
  figA4 cluster-number diagnostics  figA5 predictor Spearman correlation
  figA6 cross-model importance agreement

Boundary data: southkorea-maps (kostat 2013 municipal geojson),
  https://github.com/southkorea/southkorea-maps  -> clone next to this repo
  or pass --geojson. 2013 boundaries are matched to 2024 administrative
  units (Cheongju merger, Michuhol rename, Gunwi transfer, Yeoju upgrade).

Run: python scripts/ml/make_appendix_figures.py --data data/track_C_2017_2024.csv
"""
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPoly
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np, pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from catboost import CatBoostRegressor, Pool

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    plt.rcParams.update({"font.family": "serif"})
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
LAB = {"pagerank_lag1": "PageRank ($t-1$)", "closeness": "Closeness",
       "house_age": "Housing age", "childcare_pk": "Childcare facilities",
       "pop_density": "Population density", "seoul_dist_km": "Distance to Seoul",
       "ln_pop": "ln(Population)", "fertility": "Fertility rate",
       "doctor_per1000": "Physicians per 1,000", "aging_ratio": "Ageing ratio",
       "youth_ratio": "Youth ratio", "fiscal_indep": "Fiscal independence",
       "employ_rate": "Employment rate", "biz_count": "Business establishments",
       "sewer_supply": "Sewerage coverage", "academy_pk": "Private academies",
       "culture_facility_count": "Cultural facilities", "senior_fac_pk": "Senior facilities",
       "hospital_bed": "Hospital beds", "extinction_risk": "Extinction-risk index"}
OUT = "results/figures/paper"

KOSIS_PROV = {11: "S", 26: "B", 27: "D", 28: "I", 29: "G", 30: "J", 31: "U", 36: "SJ",
              41: "GG", 42: "GW", 51: "GW", 43: "CB", 44: "CN", 45: "JB", 52: "JB",
              46: "JN", 47: "GB", 48: "GN", 50: "JJ"}
KOSTAT_PROV = {11: "S", 21: "B", 22: "D", 23: "I", 24: "G", 25: "J", 26: "U", 29: "SJ",
               31: "GG", 32: "GW", 33: "CB", 34: "CN", 35: "JB", 36: "JN", 37: "GB",
               38: "GN", 39: "JJ"}
# 2013-boundary -> 2024-unit manual matches (Korean names as in the sources)
MANUAL = {("CB", "청원군"): ("CB", "청주시"),
          ("GB", "군위군"): ("D", "군위군"),
          ("GG", "여주군"): ("GG", "여주시"),
          ("I", "남구"): ("I", "미추홀구"),
          ("SJ", "세종시"): ("SJ", "세종특별자치시")}


def build_polys(geojson_path, reg):
    g = json.load(open(geojson_path, encoding="utf-8"))
    reg = reg.copy()
    reg["sgg"] = reg["sgg"].fillna(reg["name"].str.split().str[-1])
    reg["prov"] = (reg["region_code"] // 1000).map(KOSIS_PROV)
    lookup = {(p, str(s)): rc for rc, p, s in zip(reg.region_code, reg.prov, reg.sgg)}

    def match(feat):
        code, nm = feat["properties"]["code"], str(feat["properties"]["name"])
        prov = KOSTAT_PROV.get(int(code[:2]))
        key = MANUAL.get((prov, nm), (prov, nm))
        if key in lookup:
            return lookup[key]
        for (p, s), rc in lookup.items():          # ward -> parent city
            if p == key[0] and nm.startswith(s):
                return rc
        hits = [rc for (p, s), rc in lookup.items() if s == key[1]]
        return hits[0] if len(hits) == 1 else None

    polys = []
    for f in g["features"]:
        rc = match(f)
        if rc is None:
            continue
        geom = f["geometry"]
        rings = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in rings:
            polys.append((rc, poly[0]))
    return polys


def draw_map(polys, values, fname, cbar_label=None, cmap=None, norm=None,
             categorical=None, legend=None):
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    patches, cols = [], []
    for rc, ring in polys:
        if rc not in values:
            continue
        patches.append(MPoly(np.asarray(ring)))
        cols.append(values[rc])
    pc = PatchCollection(patches, edgecolor="#666666", linewidth=0.25)
    if categorical:
        pc.set_facecolor([categorical[v] for v in cols])
    else:
        pc.set_array(np.asarray(cols)); pc.set_cmap(cmap); pc.set_norm(norm)
    ax.add_collection(pc); ax.autoscale(); ax.set_aspect(1.15); ax.axis("off")
    if not categorical:
        fig.colorbar(pc, ax=ax, shrink=0.55, label=cbar_label)
    if legend:
        ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/{fname}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{OUT}/{fname}.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/track_C_2017_2024.csv")
    ap.add_argument("--geojson", default="../southkorea-maps/kostat/2013/json/"
                                          "skorea_municipalities_geo_simple.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(a.data)

    # ---- maps ----
    reg = df.groupby("region_code").agg(net=("net_rate", "mean"),
                                        sgg=("sgg_name", "first"),
                                        name=("name", "first")).reset_index()
    if os.path.exists(a.geojson):
        polys = build_polys(a.geojson, reg)
        vals = dict(zip(reg.region_code, reg.net))
        lim = np.nanpercentile(list(vals.values()), [2, 98])
        m = max(abs(lim[0]), abs(lim[1]))
        draw_map(polys, vals, "fig4_migration_map",
                 "Mean net migration rate, 2017–2024 (‰)", "RdBu",
                 TwoSlopeNorm(vcenter=0, vmin=-m, vmax=m))
        cl = pd.read_csv("results/tables/t1_region_clusters.csv")
        colors = {0: "#bdbdbd", 1: "#d7301f", 2: "#2b8cbe", 3: "#fdcc8a"}
        leg = [mpatches.Patch(color=colors[k], label=f"Cluster {k}") for k in sorted(colors)]
        draw_map(polys, dict(zip(cl.region_code, cl.cluster)), "fig6_cluster_map",
                 categorical=colors, legend=leg)
    else:
        print("[WARN] geojson not found; skipping maps:", a.geojson)

    # ---- model performance bars ----
    perf = pd.read_csv("results/tables/table4_16_ml_performance.csv")
    names = {"LR": "Linear regression", "DT": "Decision tree", "RF": "Random forest",
             "GB": "Gradient boosting", "XGB": "XGBoost", "LGBM": "LightGBM",
             "CatBoost": "CatBoost"}
    perf["label"] = perf["model"].map(names)
    y = np.arange(len(perf))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]; h = 0.38
    ax.barh(y + h / 2, perf["Val_R2"], height=h, color="#9ecae1",
            edgecolor="#333333", lw=0.5, label="Validation (CV)")
    ax.barh(y - h / 2, perf["Test_R2"], height=h, color="#2171b5",
            edgecolor="#333333", lw=0.5, label="Test (2023–2024 hold-out)")
    ax.set_yticks(y); ax.set_yticklabels(perf["label"], fontsize=8)
    ax.axvline(0, color="#333333", lw=0.8); ax.set_xlabel("$R^2$")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("(a) Coefficient of determination", fontsize=9)
    ax = axes[1]
    cols = ["#c6dbef" if m_ != "CatBoost" else "#2171b5" for m_ in perf["model"]]
    ax.barh(y, perf["Test_RMSE"], height=0.55, color=cols, edgecolor="#333333", lw=0.5)
    for yi, v in zip(y, perf["Test_RMSE"]):
        ax.text(v + 0.15, yi, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 18.5)
    ax.set_xlabel("Test RMSE (‰)"); ax.set_title("(b) Hold-out prediction error", fontsize=9)
    fig.savefig(f"{OUT}/fig2_model_perf.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig2_model_perf.pdf"); plt.close(fig)

    # ---- stage-rank heatmap (verified against lifecycle_verification ranks) ----
    rd = df.groupby("region_code")["pop_density"].mean()
    df["stage"] = df["region_code"].map(pd.qcut(rd, 3, labels=["growth", "middle", "mature"]).to_dict())
    d2 = df.dropna(subset=FEATS + ["net_rate", "stage"])
    res = {}
    for st in ["growth", "middle", "mature"]:
        sub = d2[d2.stage == st]; tr = sub[sub.year <= 2022]
        m = CatBoostRegressor(**CB); m.fit(tr[FEATS], tr["net_rate"])
        sv = m.get_feature_importance(type="ShapValues", data=Pool(sub[FEATS]))[:, :-1]
        res[st] = (-np.abs(sv).mean(0)).argsort().argsort() + 1
    R = pd.DataFrame({st: res[st] for st in ["growth", "middle", "mature"]}, index=FEATS)
    assert R.loc["pagerank_lag1"].tolist() == [4, 18, 13]
    assert R.loc["closeness"].tolist() == [10, 1, 8]
    R.to_csv("results/tables/stage_shap_ranks.csv")
    order = R.min(axis=1).sort_values().index
    Rp = R.loc[order]
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    im = ax.imshow(Rp.values, cmap="viridis_r", vmin=1, vmax=20, aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Growth", "Middle", "Mature"], fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[f] for f in order], fontsize=8)
    for i in range(len(order)):
        for j in range(3):
            v = int(Rp.values[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=7.5,
                    color="white" if v <= 10 else "black")
    fig.colorbar(im, label="SHAP-importance rank (1 = most important)", shrink=0.75)
    fig.savefig(f"{OUT}/fig8_stage_heatmap.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig8_stage_heatmap.pdf"); plt.close(fig)

    # ---- A3: LOWESS span sensitivity ----
    keep = [c for c in set(TYPO + ["net_rate", "region_code", "year"]) if c in df.columns]
    dd = df.dropna(subset=[c for c in keep if c != "year"])
    regm = dd.groupby("region_code").mean(numeric_only=True).reset_index()
    x, yv = regm["pop_density"].values, regm["net_rate"].values
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(x, yv, s=8, alpha=0.35, color="#999999", edgecolor="none")
    for frac, c, ls in [(0.3, "#2166ac", "--"), (0.5, "#b2182b", "-"), (0.7, "#4d4d4d", ":")]:
        lo = lowess(yv, x, frac=frac, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], lw=1.6, color=c, ls=ls, label=f"frac = {frac}")
    ax.axhline(0, color="#999999", ls="--", lw=0.6); ax.set_xscale("log")
    ax.set_xlabel("Population density (persons/km$^2$, log scale)")
    ax.set_ylabel("Net migration rate (‰)"); ax.legend(frameon=False)
    fig.savefig(f"{OUT}/figA3_lowess_sensitivity.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    # ---- A4: cluster diagnostics ----
    feats = [f for f in TYPO if f in regm.columns]
    Z = StandardScaler().fit_transform(regm[feats])
    rows = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=42).fit(Z)
        rows.append((k, km.inertia_, silhouette_score(Z, km.labels_),
                     calinski_harabasz_score(Z, km.labels_)))
    mtr = pd.DataFrame(rows, columns=["k", "wcss", "sil", "ch"])
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for ax, col, lab in zip(axes, ["wcss", "sil", "ch"],
                            ["Within-cluster SS", "Silhouette", "Calinski–Harabasz"]):
        ax.plot(mtr["k"], mtr[col], "o-", color="#333333", ms=3.5, lw=1.2)
        ax.axvline(4, color="#aaaaaa", ls="--", lw=0.8)
        ax.set_xlabel("Number of clusters ($k$)"); ax.set_title(lab, fontsize=9)
    fig.savefig(f"{OUT}/figA4_cluster_diagnostics.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    # ---- A5: predictor correlation matrix ----
    corr = df[FEATS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=7)
    fig.colorbar(im, label="Spearman correlation", shrink=0.8)
    fig.savefig(f"{OUT}/figA5_corr_matrix.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    # ---- A2: bootstrap SHAP-rank CIs / A6: cross-model agreement ----
    ci = pd.read_csv("results/tables/a2_shap_rank_ci.csv").head(10)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ypos = np.arange(len(ci))[::-1]
    ax.errorbar(ci["rank_med"], ypos,
                xerr=[ci["rank_med"] - ci["rank_lo"], ci["rank_hi"] - ci["rank_med"]],
                fmt="o", color="#333333", ms=4, capsize=2.5, lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(ci["feature"], fontsize=8)
    ax.set_xlabel("Bootstrap SHAP-importance rank (median and 95% CI)")
    fig.savefig(f"{OUT}/figA2_rank_ci.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    sp = pd.read_csv("results/tables/b2_model_rank_spearman.csv", index_col=0)
    orderm = ["CatBoost", "XGBoost", "LightGBM", "RF_gain", "Permutation"]
    labm = ["CatBoost", "XGBoost", "LightGBM", "RF (gain)", "Permutation"]
    S = sp.loc[orderm, orderm]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(S.values, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(labm, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(labm, fontsize=8)
    for i in range(5):
        for j in range(5):
            v = S.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black")
    fig.colorbar(im, label="Spearman rank correlation", shrink=0.8)
    fig.savefig(f"{OUT}/figA6_agreement.png", dpi=400, bbox_inches="tight"); plt.close(fig)
    print("saved appendix figures to", OUT)


if __name__ == "__main__":
    main()
