# -*- coding: utf-8 -*-
"""Absolute-count re-analysis: test whether per-capita facility variables' NEGATIVE
SHAP direction is a per-capita confounding artifact (flips positive with absolute counts).
Run:  python scripts/ml/abs_count_reanalysis.py --data data/track_C_2017_2024.csv

ABS_MAP 설정 근거 (2026-07-06 확인):
- track_C_2017_2024.csv에 절대량 컬럼(childcare_count 등) 없음
- ln_doctor = log(doctor_per1000) 확인 (r=1.000) → 절대량 아님
- _pk 변수(childcare_pk, senior_fac_pk, academy_pk): 값 범위 0.1~32.8 → 인구 1000명당 단위
- doctor_per1000: 명칭 그대로 인구 1000명당 의사 수 → DERIVE_SCALE=1000
- hospital_bed: mean=15.1, max=70.1 → 인구 1000명당 병상 수 → DERIVE_SCALE=1000
- sewer_supply(보급률%), house_age(노후비율%), pop_density(면적당) → 교체 대상 아님
"""
import os, argparse, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

FEATS = ["pagerank_lag1", "closeness", "house_age", "childcare_pk", "pop_density",
         "seoul_dist_km", "ln_pop", "fertility", "doctor_per1000", "aging_ratio",
         "youth_ratio", "fiscal_indep", "employ_rate", "biz_count", "sewer_supply",
         "academy_pk", "culture_facility_count", "senior_fac_pk", "hospital_bed",
         "extinction_risk"]

CB = dict(iterations=677, learning_rate=0.10884860813834339, depth=8,
          l2_leaf_reg=2.1524708091423577, random_state=42, verbose=0)

TARGET = "net_rate"
TAB = "results/tables"

# ABS_MAP: per-capita 변수 → 절대량 컬럼명 매핑
# 절대량 컬럼이 데이터에 없으므로 POP_COL + DERIVE_SCALE로 유도(derive)
# 실제 절대량 컬럼이 있으면 해당 컬럼명으로 교체 (유도보다 우선)
ABS_MAP = {
    "childcare_pk":   "childcare_count",   # 없음 → derive: childcare_pk * pop / 1000
    "doctor_per1000": "doctor_count",       # 없음 → derive: doctor_per1000 * pop / 1000
    "senior_fac_pk":  "senior_fac_count",  # 없음 → derive: senior_fac_pk * pop / 1000
    "academy_pk":     "academy_count",     # 없음 → derive: academy_pk * pop / 1000
    "hospital_bed":   "hospital_bed_count" # 없음 → derive: hospital_bed * pop / 1000
}

POP_COL = "population"          # 인구 컬럼 (절대량 유도용)
DERIVE_SCALE = {                # per-capita 분모 (기본 1.0 = 인구당)
    "childcare_pk":   1000.0,   # 인구 1000명당
    "doctor_per1000": 1000.0,   # 인구 1000명당
    "senior_fac_pk":  1000.0,   # 인구 1000명당
    "academy_pk":     1000.0,   # 인구 1000명당
    "hospital_bed":   1000.0,   # 인구 1000명당
}


def fit(X, y):
    m = CatBoostRegressor(**CB)
    m.fit(X, y)
    return m


def shap_cb(m, X):
    return m.get_feature_importance(type="ShapValues", data=Pool(X))[:, :-1]


def dirs_ranks(m, X, feats):
    sv = shap_cb(m, X[feats])
    imp = np.abs(sv).mean(0)
    rank = (-imp).argsort().argsort() + 1
    d = {}
    for i, f in enumerate(feats):
        corr = spearmanr(X[f], sv[:, i]).correlation
        if np.isnan(corr):
            corr = 0.0
        d[f] = int(np.sign(corr))
    return d, dict(zip(feats, rank))


def build_abs(df):
    """per-capita 변수를 절대량으로 교체한 데이터프레임과 피처 목록 반환."""
    d = df.copy()
    feats_abs = []
    for f in FEATS:
        if f in ABS_MAP:
            col = ABS_MAP[f]
            if col in d.columns:
                # 실제 절대량 컬럼 존재 → 그대로 사용
                feats_abs.append(col)
            elif POP_COL in d.columns:
                # 유도(derive): per_capita × population / scale
                nf = f + "_abs"
                scale = DERIVE_SCALE.get(f, 1.0)
                d[nf] = d[f] * d[POP_COL] / scale
                feats_abs.append(nf)
            else:
                # population 컬럼도 없음 → per-capita 그대로 유지
                feats_abs.append(f)
        else:
            feats_abs.append(f)
    return d, feats_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/track_C_2017_2024.csv")
    a = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)

    df = pd.read_csv(a.data)
    df = df.dropna(subset=[TARGET] + FEATS + ["region_code", "year"]).copy()

    print(f"Data loaded: {len(df)} rows, {df['region_code'].nunique()} regions, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # ── per-capita 모형 ──────────────────────────────────────────────────────
    tr = df[df.year <= 2022]
    te = df[df.year == 2024]
    print(f"\nTrain (≤2022): {len(tr)}, Test (2024): {len(te)}")

    m_pc = fit(tr[FEATS], tr[TARGET])
    r2_pc = r2_score(te[TARGET], m_pc.predict(te[FEATS])) if len(te) else np.nan
    dir_pc, rank_pc = dirs_ranks(m_pc, df, FEATS)

    # ── 절대량 모형 ──────────────────────────────────────────────────────────
    d2, fa = build_abs(df)
    d2 = d2.dropna(subset=fa).copy()
    tr2 = d2[d2.year <= 2022]
    te2 = d2[d2.year == 2024]

    print(f"\nAbsolute-count features: {fa}")
    print(f"Swapped: {[(f, fa[FEATS.index(f)]) for f in ABS_MAP if f in FEATS]}")

    m_ab = fit(tr2[fa], tr2[TARGET])
    r2_ab = r2_score(te2[TARGET], m_ab.predict(te2[fa])) if len(te2) else np.nan
    dir_ab, rank_ab = dirs_ranks(m_ab, d2, fa)

    # ── 비교표 ───────────────────────────────────────────────────────────────
    rows = []
    for f in ABS_MAP:
        if f not in FEATS:
            continue
        af = fa[FEATS.index(f)]
        rows.append({
            "per_capita":      f,
            "absolute":        af,
            "dir_pc":          dir_pc.get(f),
            "dir_abs":         dir_ab.get(af),
            "rank_pc":         rank_pc.get(f),
            "rank_abs":        rank_ab.get(af),
            "flip_to_positive": int((dir_pc.get(f, 0) < 0) and (dir_ab.get(af, 0) > 0))
        })

    r = pd.DataFrame(rows)
    r.to_csv(f"{TAB}/abs_reanalysis_comparison.csv", index=False)

    print(f"\nTest R²:  per-capita={r2_pc:.3f}   absolute={r2_ab:.3f}")
    print()
    print(r.to_string(index=False))

    fl = r[r.flip_to_positive == 1]["per_capita"].tolist()
    if fl:
        print(f"\nVERDICT: per-capita confounding CONFIRMED (dir flipped to +): {', '.join(fl)}")
    else:
        print("\nVERDICT: NO flip -> negative direction persists even with absolute counts")


if __name__ == "__main__":
    main()
