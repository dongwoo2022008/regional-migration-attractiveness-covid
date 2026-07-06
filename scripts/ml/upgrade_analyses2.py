# -*- coding: utf-8 -*-
"""Tier-2 upgrade analyses (differentiation) for the RQ5-6 paper. Run from repo root.
Adds: A) region-transfer generalization, B) consensus SHAP across models + permutation,
C) ALE (+PDP overlay), D) SHAP direction stability pre/post COVID (RQ6 headline),
E) counterfactual scenario sensitivity (NOT causal), F) SHAP clustering + local top-driver.
CatBoost native SHAP; XGB/LGBM contributions if available (skipped gracefully otherwise).
Run:  python scripts/ml/upgrade_analyses2.py --data data/track_C_2017_2024.csv
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

FEATS=["pagerank_lag1","closeness","house_age","childcare_pk","pop_density","seoul_dist_km",
"ln_pop","fertility","doctor_per1000","aging_ratio","youth_ratio","fiscal_indep","employ_rate",
"biz_count","sewer_supply","academy_pk","culture_facility_count","senior_fac_pk","hospital_bed","extinction_risk"]
CB=dict(iterations=677,learning_rate=0.10884860813834339,depth=8,l2_leaf_reg=2.1524708091423577,random_state=42,verbose=0)
TARGET="net_rate"; TAB="results/tables"; FIG="results/figures"
POLICY=["childcare_pk","doctor_per1000","house_age","sewer_supply","pagerank_lag1"]

def fit(X,y): m=CatBoostRegressor(**CB); m.fit(X,y); return m
def shap_cb(m,X): return m.get_feature_importance(type="ShapValues",data=Pool(X))[:,:-1]
def load(path):
    df=pd.read_csv(path); assert TARGET in df
    df=df.dropna(subset=[TARGET]+FEATS+["region_code","year"]).copy()
    df["sido"]=(df["region_code"]//1000).astype(int)
    df["capital"]=df["sido"].isin([11,28,41]).astype(int); return df

def A_transfer(df):
    cap=df[df.capital==1]; groups={s:df[df.sido==s] for s in sorted(df[df.capital==0].sido.unique())}
    m=fit(cap[FEATS],cap[TARGET]); rows=[]
    for s,g in groups.items():
        if len(g)<20: continue
        rows.append({"train":"capital","test_sido":s,"n":len(g),"r2":r2_score(g[TARGET],m.predict(g[FEATS]))})
    r=pd.DataFrame(rows); r.to_csv(f"{TAB}/b1_region_transfer.csv",index=False)
    print(f"[A] capital->region transfer: mean R2={r.r2.mean():.3f} (min {r.r2.min():.3f})"); return r

def _imp_xgb(X,y):
    try:
        import xgboost as xgb
        d=xgb.DMatrix(X,label=y); bst=xgb.train({"max_depth":6,"eta":0.05},d,120)
        c=bst.predict(d,pred_contribs=True)[:,:-1]; return np.abs(c).mean(0)
    except Exception: return None
def _imp_lgbm(X,y):
    try:
        import lightgbm as lgb
        m=lgb.LGBMRegressor(n_estimators=300,verbose=-1).fit(X,y)
        c=m.predict(X,pred_contrib=True)[:,:-1]; return np.abs(c).mean(0)
    except Exception: return None
def B_consensus(df):
    d=df[df.year<=2023]; X,y=d[FEATS],d[TARGET]
    cb=fit(X,y); imp={"CatBoost":np.abs(shap_cb(cb,X)).mean(0)}
    rf=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1).fit(X,y); imp["RF_gain"]=rf.feature_importances_
    x=_imp_xgb(X,y);  imp["XGBoost"]=x if x is not None else np.full(len(FEATS),np.nan)
    l=_imp_lgbm(X,y); imp["LightGBM"]=l if l is not None else np.full(len(FEATS),np.nan)
    pi=permutation_importance(cb,X,y,n_repeats=10,random_state=42,n_jobs=-1); imp["Permutation"]=pi.importances_mean
    M=pd.DataFrame(imp,index=FEATS).dropna(axis=1,how="all")
    R=M.rank(ascending=False)
    R["consensus_rank"]=R.mean(1); M["consensus_rank"]=R["consensus_rank"]
    M.sort_values("consensus_rank").to_csv(f"{TAB}/b2_consensus_importance.csv")
    corr=R.drop(columns="consensus_rank").corr(method="spearman"); corr.to_csv(f"{TAB}/b2_model_rank_spearman.csv")
    print("[B] consensus rank (top5):",list(M.sort_values("consensus_rank").index[:5]))
    tri=corr.values[np.triu_indices(len(corr),1)]
    print("    mean pairwise Spearman across models = %.3f"%np.nanmean(tri))
    return M

def _ale1d(m,X,f,bins=12):
    q=np.quantile(X[f],np.linspace(0,1,bins+1)); q=np.unique(q); eff=[]
    for i in range(len(q)-1):
        idx=(X[f]>=q[i])&(X[f]<=q[i+1])
        if idx.sum()==0: eff.append(0); continue
        a=X[idx].copy(); a[f]=q[i]; b=X[idx].copy(); b[f]=q[i+1]
        eff.append((m.predict(b)-m.predict(a)).mean())
    ale=np.cumsum(eff); ale=ale-ale.mean(); cen=(q[:-1]+q[1:])/2
    return cen,ale
def C_ale(df,m):
    d=df[df.year<=2023]
    fig,ax=plt.subplots(1,4,figsize=(15,3.4))
    for i,f in enumerate(["childcare_pk","house_age","closeness","pagerank_lag1"]):
        cen,ale=_ale1d(m,d[FEATS],f); ax[i].plot(cen,ale,color="#c0392b",lw=2); ax[i].set_title(f"ALE: {f}"); ax[i].axhline(0,color="#999",ls=":")
    plt.tight_layout(); plt.savefig(f"{FIG}/c1_ale.png",dpi=200); plt.close()
    print("[C] ALE curves saved (robust to feature correlation; complements PDP)")

def D_direction(df):
    rows=[]
    for lab,sub in [("pre_2017_2020",df[df.year<=2020]),("post_2021_2024",df[df.year>=2021])]:
        m=fit(sub[FEATS],sub[TARGET]); sv=shap_cb(m,sub[FEATS])
        for j,f in enumerate(FEATS):
            dirn=spearmanr(sub[f],sv[:,j]).correlation
            rows.append({"period":lab,"feature":f,"mean_abs_shap":np.abs(sv[:,j]).mean(),"direction_sign":np.sign(dirn),"direction_corr":dirn})
    r=pd.DataFrame(rows)
    piv=r.pivot(index="feature",columns="period",values="direction_sign")
    piv["dir_flip"]=(piv["pre_2017_2020"]!=piv["post_2021_2024"]).astype(int)
    mag=r.pivot(index="feature",columns="period",values="mean_abs_shap")
    out=piv.join(mag,rsuffix="_mag"); out.to_csv(f"{TAB}/d1_shap_direction_stability.csv")
    r.to_csv(f"{TAB}/d1_shap_direction_long.csv",index=False)
    flips=out[out.dir_flip==1].index.tolist()
    print(f"[D] SHAP direction stability: {len(flips)} feature(s) flipped sign pre->post COVID: {flips if flips else 'NONE (all stable)'}")
    return out

def E_counterfactual(df,m):
    d=df[df.year>=2023].copy(); base=m.predict(d[FEATS]); sd=df[FEATS].std(); sv=shap_cb(m,d[FEATS]); rows=[]
    for f in POLICY:
        j=FEATS.index(f); direction=int(np.sign(spearmanr(d[f],sv[:,j]).correlation))
        for k in [1.0,-1.0]:
            x=d[FEATS].copy(); x[f]=(x[f]+k*sd[f]).clip(df[f].min(),df[f].max())
            rows.append({"feature":f,"shock":"+1SD" if k>0 else "-1SD","shap_direction":direction,
                         "mean_pred_change":round((m.predict(x)-base).mean(),3)})
    r=pd.DataFrame(rows); r.to_csv(f"{TAB}/e1_counterfactual_scenario.csv",index=False)
    print("[E] counterfactual (additive +/-1 SD, clipped; MODEL SENSITIVITY, NOT causal; sign may reflect per-capita confounding):")
    print(r.to_string(index=False))
    return r

def F_cluster(df,k=4):
    d=df[df.year<=2023]; m=fit(d[FEATS],d[TARGET]); sv=shap_cb(m,d[FEATS])
    S=pd.DataFrame(sv,columns=FEATS); S["region_code"]=d["region_code"].values
    prof=S.groupby("region_code")[FEATS].mean()
    Z=(prof-prof.mean())/prof.std()
    km=KMeans(k,n_init=10,random_state=42).fit(Z.fillna(0)); prof["cluster"]=km.labels_
    prof["top_driver"]=prof[FEATS].abs().idxmax(1)
    prof[["cluster","top_driver"]].to_csv(f"{TAB}/f1_region_cluster_topdriver.csv")
    cp=prof.groupby("cluster")[FEATS].mean(); cp["label_topfeatures"]=cp.abs().apply(lambda r:", ".join(r.sort_values(ascending=False).index[:3]),axis=1)
    cp.to_csv(f"{TAB}/f2_cluster_profiles.csv")
    print(f"[F] SHAP clusters (k={k}) — attractiveness regimes:")
    print(cp["label_topfeatures"].to_string())
    print("    (map rendering optional: join f1_region_cluster_topdriver.csv to sgg geojson)")
    return prof

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/track_C_2017_2024.csv"); a=ap.parse_args()
    os.makedirs(TAB,exist_ok=True); os.makedirs(FIG,exist_ok=True)
    df=load(a.data); print(f"loaded {len(df)} rows, {df.region_code.nunique()} regions")
    m=fit(df[df.year<=2023][FEATS],df[df.year<=2023][TARGET])
    A_transfer(df); B_consensus(df); C_ale(df,m); D_direction(df); E_counterfactual(df,m); F_cluster(df)
    print("\nDONE -> results/tables, results/figures")
if __name__=="__main__": main()
