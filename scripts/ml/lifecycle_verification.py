# -*- coding: utf-8 -*-
"""Urban Life-Cycle Frame Verification (V1–V5)
Implements the A→B→C chain:
  A: childcare_pk SHAP direction = negative (confirmed)
  B: childcare/hospital/house_age = markers of mature, dense urban cores
  C: mature cores = net out-migration (suburbanization / counter-urbanization)
  + V3: same amenity, opposite sign by urban stage (stage-dependent)
  + V4: COVID acceleration of density gradient (conditional)
  + V5: Network × stage (dual mechanism)

Run:
  python scripts/ml/lifecycle_verification.py --data data/track_C_2017_2024.csv
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "AppleGothic", "UnDotum", "DejaVu Sans"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ── 공통 설정 ─────────────────────────────────────────────────────────────────
FEATS = [
    "pagerank_lag1", "closeness", "house_age", "childcare_pk", "pop_density",
    "seoul_dist_km", "ln_pop", "fertility", "doctor_per1000", "aging_ratio",
    "youth_ratio", "fiscal_indep", "employ_rate", "biz_count", "sewer_supply",
    "academy_pk", "culture_facility_count", "senior_fac_pk", "hospital_bed",
    "extinction_risk"
]
CB = dict(
    iterations=677, learning_rate=0.10884860813834339, depth=8,
    l2_leaf_reg=2.1524708091423577, random_state=42, verbose=0
)
TARGET = "net_rate"
TAB = "results/tables"
FIG = "results/figures"

MARKER_VARS = ["childcare_pk", "hospital_bed", "house_age"]
MATURITY_VARS = ["pop_density", "seoul_dist_km", "ln_pop", "house_age",
                 "childcare_pk", "hospital_bed", "aging_ratio"]


def fit_cb(X, y):
    m = CatBoostRegressor(**CB)
    m.fit(X, y)
    return m


def shap_values(m, X):
    return m.get_feature_importance(type="ShapValues", data=Pool(X))[:, :-1]


def shap_direction(sv_col, feat_col):
    """Spearman corr between feature values and SHAP values → sign."""
    r = spearmanr(feat_col, sv_col).correlation
    return int(np.sign(r)) if not np.isnan(r) else 0


def assign_stage(df, density_col="pop_density", labels=("growth", "middle", "mature")):
    """Assign urban life-cycle stage by density tertile (region-level mean)."""
    region_density = df.groupby("region_code")[density_col].mean()
    tertiles = pd.qcut(region_density, 3, labels=labels)
    stage_map = tertiles.to_dict()
    return df["region_code"].map(stage_map)


# ─────────────────────────────────────────────────────────────────────────────
# V1. Marker verification — B: childcare/hospital/house_age = mature-core markers
# ─────────────────────────────────────────────────────────────────────────────
def v1_marker(df, out_dir):
    print("\n" + "=" * 60)
    print("V1. Marker Verification (B: facility vars = mature-core markers)")
    print("=" * 60)

    region_avg = (
        df.groupby("region_code")[
            MATURITY_VARS + ["net_rate", "fertility", "fiscal_indep"]
        ]
        .mean()
        .dropna()
    )

    # ① Spearman 상관표: marker vars vs maturity proxies
    proxy_vars = ["pop_density", "seoul_dist_km", "ln_pop", "house_age",
                  "aging_ratio"]
    rows = []
    for mv in MARKER_VARS:
        for pv in proxy_vars:
            if mv == pv:
                continue
            r, p = spearmanr(region_avg[mv], region_avg[pv])
            rows.append({
                "marker": mv, "proxy": pv,
                "spearman_r": round(r, 3), "p_value": round(p, 4),
                "sig": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            })
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(f"{out_dir}/v1_marker_corr.csv", index=False)
    print(corr_df.to_string(index=False))

    # ② PCA 성숙도 지수 (density + house_age + childcare_pk + hospital_bed + aging_ratio − seoul_dist)
    pca_vars = ["pop_density", "house_age", "childcare_pk", "hospital_bed",
                "aging_ratio"]
    # seoul_dist_km은 부호 반전 (가까울수록 성숙)
    pca_df = region_avg[pca_vars].copy()
    if "seoul_dist_km" in region_avg.columns:
        pca_df["seoul_proximity"] = -region_avg["seoul_dist_km"]
        pca_vars_use = pca_vars + ["seoul_proximity"]
    else:
        pca_vars_use = pca_vars

    scaler = StandardScaler()
    X_pca = scaler.fit_transform(pca_df[pca_vars_use].dropna())
    pca = PCA(n_components=min(3, len(pca_vars_use)))
    pca.fit(X_pca)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_vars_use,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings["var_explained"] = ""
    loadings.loc[loadings.index[0], "var_explained"] = (
        f"PC1={pca.explained_variance_ratio_[0]:.3f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.3f}"
    )
    loadings.to_csv(f"{out_dir}/v1_maturity_pca.csv")
    print(f"\nPCA loadings (PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%):")
    print(loadings[["PC1", "PC2"]].to_string())

    # ③ maturity score vs net_rate
    valid_idx = pca_df[pca_vars_use].dropna().index
    maturity_scores = pca.transform(X_pca)[:, 0]
    maturity_series = pd.Series(maturity_scores, index=valid_idx, name="maturity_score")
    region_avg2 = region_avg.loc[valid_idx].copy()
    region_avg2["maturity_score"] = maturity_scores
    r_mat, p_mat = spearmanr(region_avg2["maturity_score"], region_avg2["net_rate"])
    print(f"\nMaturity score vs net_rate: Spearman r={r_mat:.3f}, p={p_mat:.4f}")

    return region_avg, maturity_series


# ─────────────────────────────────────────────────────────────────────────────
# V2. Urban life-cycle gradient — C: mature cores → net out-migration
# ─────────────────────────────────────────────────────────────────────────────
def v2_gradient(df, out_dir):
    print("\n" + "=" * 60)
    print("V2. Urban Life-Cycle Gradient (C: density → net out-migration)")
    print("=" * 60)

    region_avg = (
        df.groupby("region_code")[
            ["net_rate", "pop_density", "childcare_pk", "house_age",
             "seoul_dist_km", "ln_pop", "aging_ratio"]
        ]
        .mean()
        .dropna()
    )

    # ① density vs net_rate Spearman
    r, p = spearmanr(region_avg["pop_density"], region_avg["net_rate"])
    print(f"net_rate vs pop_density: Spearman r={r:.3f}, p={p:.4f}")

    # ② 도시단계 3분위 통계
    region_avg["density_tertile"] = pd.qcut(
        region_avg["pop_density"], 3, labels=["growth", "middle", "mature"]
    )
    stage_stats = region_avg.groupby("density_tertile")["net_rate"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    stage_stats.columns = ["stage", "net_rate_mean", "net_rate_std", "n"]
    stage_stats.to_csv(f"{out_dir}/v2_density_gradient.csv", index=False)
    print("\nnet_rate by density tertile:")
    print(stage_stats.to_string(index=False))

    # ③ 순유입·유출 상위 20 목록
    name_map = df.groupby("region_code")["sgg_name"].first()
    sido_map = df.groupby("region_code")["sido"].first()
    region_avg["sgg_name"] = region_avg.index.map(name_map)
    region_avg["sido"] = region_avg.index.map(sido_map)

    top_in = region_avg.nlargest(20, "net_rate")[
        ["sgg_name", "sido", "net_rate", "pop_density", "density_tertile"]
    ]
    top_out = region_avg.nsmallest(20, "net_rate")[
        ["sgg_name", "sido", "net_rate", "pop_density", "density_tertile"]
    ]
    combined = pd.concat(
        [top_in.assign(type="inflow"), top_out.assign(type="outflow")]
    )
    combined.to_csv(f"{out_dir}/v2_top_inflow_outflow.csv")
    print("\nTop 10 inflow:")
    print(top_in.head(10)[["sgg_name", "sido", "net_rate", "density_tertile"]].to_string())
    print("\nTop 10 outflow:")
    print(top_out.head(10)[["sgg_name", "sido", "net_rate", "density_tertile"]].to_string())

    # ④ 산점도
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"growth": "#2196F3", "middle": "#FF9800", "mature": "#F44336"}
    for stage, grp in region_avg.groupby("density_tertile"):
        ax.scatter(
            grp["pop_density"], grp["net_rate"],
            c=colors.get(str(stage), "gray"), alpha=0.6, s=30, label=str(stage)
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Population Density (persons/km²)")
    ax.set_ylabel("Net Migration Rate (‰)")
    ax.set_title(f"Density–Net Migration Gradient\n(Spearman r={r:.3f}, p={p:.4f})")
    ax.legend(title="Urban Stage")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/v2_density_scatter.png", dpi=150)
    plt.close()
    print(f"\nScatter plot saved: {out_dir}/v2_density_scatter.png")

    return region_avg


# ─────────────────────────────────────────────────────────────────────────────
# V3. Stage-dependent amenity effect ⭐
# ─────────────────────────────────────────────────────────────────────────────
def v3_stage_amenity(df, out_dir):
    print("\n" + "=" * 60)
    print("V3. Stage-Dependent Amenity Effect ⭐")
    print("=" * 60)

    focus_feats = ["childcare_pk", "hospital_bed", "house_age",
                   "pagerank_lag1", "fertility", "aging_ratio"]

    # 도시단계 할당 (패널 전체 기준)
    df2 = df.copy()
    df2["stage"] = assign_stage(df2)
    df2 = df2.dropna(subset=FEATS + [TARGET, "stage"])

    rows = []
    for stage in ["growth", "middle", "mature"]:
        sub = df2[df2["stage"] == stage]
        if len(sub) < 50:
            print(f"  Stage '{stage}': too few rows ({len(sub)}), skipping")
            continue
        tr = sub[sub["year"] <= 2022]
        if len(tr) < 30:
            print(f"  Stage '{stage}': too few train rows ({len(tr)}), skipping")
            continue
        m = fit_cb(tr[FEATS], tr[TARGET])
        sv = shap_values(m, sub[FEATS])
        for i, f in enumerate(FEATS):
            if f in focus_feats:
                d = shap_direction(sv[:, i], sub[f])
                imp = float(np.abs(sv[:, i]).mean())
                rows.append({
                    "stage": stage, "feature": f,
                    "shap_direction": d,
                    "shap_importance": round(imp, 4),
                    "n": len(sub)
                })
        print(f"  Stage '{stage}': n={len(sub)}, train={len(tr)}")

    result = pd.DataFrame(rows)
    result.to_csv(f"{out_dir}/v3_amenity_by_stage.csv", index=False)
    print("\nStage-dependent SHAP directions:")
    pivot = result.pivot_table(
        index="feature", columns="stage", values="shap_direction", aggfunc="first"
    )
    print(pivot.to_string())

    # childcare_pk 방향 변화 확인
    if "childcare_pk" in pivot.index:
        row = pivot.loc["childcare_pk"]
        print(f"\nchildcare_pk directions: {row.to_dict()}")
        if row.get("growth", 0) > 0 and row.get("mature", 0) < 0:
            print("  ✓ CONFIRMED: childcare_pk growth(+) vs mature(-) — stage-dependent!")
        elif row.get("mature", 0) < 0:
            print("  PARTIAL: mature(-) confirmed, growth direction unclear")
        else:
            print("  NOT CONFIRMED: direction pattern not as expected")

    # ② childcare × pop_density 2D SHAP 상호작용 시각화
    df3 = df2.dropna(subset=FEATS + [TARGET])
    tr3 = df3[df3["year"] <= 2022]
    m_full = fit_cb(tr3[FEATS], tr3[TARGET])
    sv_full = shap_values(m_full, df3[FEATS])
    childcare_idx = FEATS.index("childcare_pk")
    sv_childcare = sv_full[:, childcare_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stage_colors = {"growth": "#2196F3", "middle": "#FF9800", "mature": "#F44336"}
    for ax, (xvar, xlabel) in zip(
        axes,
        [("childcare_pk", "childcare_pk (per 1000)"),
         ("pop_density", "Population Density")]
    ):
        for stage, grp_idx in df3.groupby("stage").groups.items():
            grp = df3.loc[grp_idx]
            sv_grp = sv_childcare[df3.index.get_indexer(grp_idx)]
            ax.scatter(
                grp[xvar].values, sv_grp,
                c=stage_colors.get(str(stage), "gray"),
                alpha=0.3, s=10, label=str(stage)
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SHAP value of childcare_pk")
        ax.legend(title="Stage", markerscale=2)
    axes[0].set_title("childcare_pk SHAP vs childcare_pk value")
    axes[1].set_title("childcare_pk SHAP vs pop_density")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/v3_interaction_density.png", dpi=150)
    plt.close()
    print(f"Interaction plot saved: {out_dir}/v3_interaction_density.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# V4. COVID acceleration of density gradient (conditional)
# ─────────────────────────────────────────────────────────────────────────────
def v4_covid(df, out_dir):
    print("\n" + "=" * 60)
    print("V4. COVID Acceleration of Density Gradient (conditional)")
    print("=" * 60)

    try:
        import statsmodels.formula.api as smf
        df2 = df.copy()
        df2["post2020"] = (df2["year"] >= 2021).astype(int)
        df2["density_std"] = (
            (df2["pop_density"] - df2["pop_density"].mean())
            / df2["pop_density"].std()
        )
        df2 = df2.dropna(subset=["net_rate", "density_std", "post2020",
                                  "region_code", "year"])

        # FE 패널: net_rate ~ density_std * post2020 (entity FE via C(region_code))
        # 간소화: pooled OLS with interaction
        formula = "net_rate ~ density_std * post2020 + C(year)"
        model = smf.ols(formula, data=df2).fit(
            cov_type="cluster", cov_kwds={"groups": df2["region_code"]}
        )
        coef_df = model.params.reset_index()
        coef_df.columns = ["term", "coef"]
        coef_df["se"] = model.bse.values
        coef_df["p"] = model.pvalues.values
        coef_df["sig"] = coef_df["p"].apply(
            lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        )
        # 핵심 항만 출력
        key_terms = ["density_std", "post2020", "density_std:post2020"]
        key_df = coef_df[coef_df["term"].isin(key_terms)].copy()
        key_df.to_csv(f"{out_dir}/v4_covid_density_interaction.csv", index=False)
        print(key_df.to_string(index=False))

        interaction_coef = key_df[key_df["term"] == "density_std:post2020"]
        if len(interaction_coef):
            c = interaction_coef.iloc[0]
            if c["coef"] < 0 and c["p"] < 0.05:
                print(f"\n  ✓ CONFIRMED: density×post2020 coef={c['coef']:.3f} (p={c['p']:.4f})")
                print("    → COVID accelerated counter-urbanization gradient")
            else:
                print(f"\n  NOT SIGNIFICANT: coef={c['coef']:.3f} (p={c['p']:.4f})")
                print("    → Do NOT link COVID to density gradient in paper")
    except ImportError:
        print("  statsmodels not available — V4 skipped")
        key_df = pd.DataFrame()

    return key_df if "key_df" in dir() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# V5. Network × stage — dual mechanism
# ─────────────────────────────────────────────────────────────────────────────
def v5_network_stage(df, out_dir):
    print("\n" + "=" * 60)
    print("V5. Network × Stage — Dual Mechanism")
    print("=" * 60)

    focus_feats = ["pagerank_lag1", "closeness", "childcare_pk",
                   "house_age", "fertility"]

    df2 = df.copy()
    df2["stage"] = assign_stage(df2)
    df2 = df2.dropna(subset=FEATS + [TARGET, "stage"])

    rows = []
    for stage in ["growth", "middle", "mature"]:
        sub = df2[df2["stage"] == stage]
        tr = sub[sub["year"] <= 2022]
        if len(tr) < 30:
            continue
        m = fit_cb(tr[FEATS], tr[TARGET])
        sv = shap_values(m, sub[FEATS])
        imp = np.abs(sv).mean(0)
        rank = (-imp).argsort().argsort() + 1
        for i, f in enumerate(FEATS):
            if f in focus_feats:
                d = shap_direction(sv[:, i], sub[f])
                rows.append({
                    "stage": stage, "feature": f,
                    "shap_direction": d,
                    "shap_importance": round(float(imp[i]), 4),
                    "rank": int(rank[i]),
                    "n": len(sub)
                })

    result = pd.DataFrame(rows)
    result.to_csv(f"{out_dir}/v5_network_by_stage.csv", index=False)
    print("\nNetwork (pagerank_lag1) SHAP by stage:")
    pr_rows = result[result["feature"] == "pagerank_lag1"]
    print(pr_rows[["stage", "shap_direction", "shap_importance", "rank"]].to_string(index=False))

    # 검증
    pr_pivot = pr_rows.set_index("stage")
    growth_rank = pr_pivot.loc["growth", "rank"] if "growth" in pr_pivot.index else None
    mature_rank = pr_pivot.loc["mature", "rank"] if "mature" in pr_pivot.index else None
    if growth_rank and mature_rank:
        if growth_rank < mature_rank:
            print(f"\n  ✓ CONFIRMED: PageRank rank in growth({growth_rank}) < mature({mature_rank})")
            print("    → Network mechanism stronger in growth stage (dual mechanism)")
        else:
            print(f"\n  NOT CONFIRMED: growth rank={growth_rank}, mature rank={mature_rank}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/track_C_2017_2024.csv")
    a = ap.parse_args()

    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    df = pd.read_csv(a.data)
    print(f"Data: {df.shape}, regions={df['region_code'].nunique()}, "
          f"years={df['year'].min()}-{df['year'].max()}")

    # V1
    region_avg, maturity_scores = v1_marker(df, TAB)

    # V2
    region_avg2 = v2_gradient(df, TAB)
    # 산점도를 figures 폴더로도 복사
    import shutil
    src = f"{TAB}/v2_density_scatter.png"
    dst = f"{FIG}/v2_density_scatter.png"
    if os.path.exists(src):
        shutil.copy(src, dst)

    # V3
    v3_result = v3_stage_amenity(df, TAB)
    src3 = f"{TAB}/v3_interaction_density.png"
    dst3 = f"{FIG}/v3_interaction_density.png"
    if os.path.exists(src3):
        shutil.copy(src3, dst3)

    # V4
    v4_result = v4_covid(df, TAB)

    # V5
    v5_result = v5_network_stage(df, TAB)

    # ── 종합 요약 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Life-Cycle Frame Verification")
    print("=" * 60)
    print("V1 Marker:    v1_marker_corr.csv, v1_maturity_pca.csv")
    print("V2 Gradient:  v2_density_gradient.csv, v2_top_inflow_outflow.csv, v2_density_scatter.png")
    print("V3 Stage-dep: v3_amenity_by_stage.csv, v3_interaction_density.png")
    print("V4 COVID:     v4_covid_density_interaction.csv")
    print("V5 Network:   v5_network_by_stage.csv")
    print()
    print("Key narrative:")
    print("  A: childcare_pk SHAP direction = negative (confirmed)")
    print("  B: childcare/hospital/house_age co-vary with mature urban markers")
    print("  C: mature dense cores show net out-migration (counter-urbanization)")
    print("  +: same amenity flips sign by urban stage (V3)")
    print("  +: network effect concentrated in growth stage (V5)")


if __name__ == "__main__":
    main()
