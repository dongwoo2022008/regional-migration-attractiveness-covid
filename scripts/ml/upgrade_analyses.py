# -*- coding: utf-8 -*-

"""Quality-upgrade analyses for the RQ5-6 paper (run from repo root).

Reproduces the canonical CatBoost model and adds 5 reviewer-facing analyses:

  1) spatial block CV (leave-one-sido-out)

  2) SHAP-rank bootstrap CI + temporal-stability test

  3) RAI weighting robustness + external (next-period) validity

  4) PDP + SHAP interaction (dependence, colored by 2nd feature)

  5) subgroup determinant structure (capital vs non-capital; city-size tertiles)

Uses CatBoost's native SHAP (no shap lib needed).  Outputs -> results/tables, results/figures.

Run:  python scripts/ml/upgrade_analyses.py --data data/track_C_2017_2024.csv

DOMAIN mapping corrected to match repo Table 3-11 (03_방법론_Methodology.md):
  fertility, ln_pop, hospital_bed, senior_fac_pk excluded from RAI sub-indices
  (network/demographic-dynamics variables; used separately per paper).
"""

import os, json, argparse, warnings

import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from sklearn.decomposition import PCA

from scipy.stats import spearmanr

FEATS=["pagerank_lag1","closeness","house_age","childcare_pk","pop_density","seoul_dist_km",
"ln_pop","fertility","doctor_per1000","aging_ratio","youth_ratio","fiscal_indep","employ_rate",
"biz_count","sewer_supply","academy_pk","culture_facility_count","senior_fac_pk","hospital_bed","extinction_risk"]

CB=dict(iterations=677,learning_rate=0.10884860813834339,depth=8,l2_leaf_reg=2.1524708091423577,
        random_state=42,verbose=0)

TARGET="net_rate"; NETWORK={"pagerank_lag1","closeness"}

# RAI domain mapping — aligned with repo Table 3-11 (03_방법론_Methodology.md §3.7).
# NOTE: fertility, ln_pop, hospital_bed, senior_fac_pk are EXCLUDED from RAI sub-indices
#       (classified as network/demographic-dynamics; used in separate analyses per paper).
DOMAIN={
    # Econ
    "employ_rate":"Econ","biz_count":"Econ","fiscal_indep":"Econ",
    # Demo
    "youth_ratio":"Demo","aging_ratio":"Demo","pop_density":"Demo","extinction_risk":"Demo",
    # Infra
    "house_age":"Infra","sewer_supply":"Infra","seoul_dist_km":"Infra",
    # Serv
    "childcare_pk":"Serv","doctor_per1000":"Serv","academy_pk":"Serv","culture_facility_count":"Serv"
}

TAB="results/tables"; FIG="results/figures"

def fit(X,y):
    m=CatBoostRegressor(**CB); m.fit(X,y); return m

def mean_abs_shap(m,X):
    sv=m.get_feature_importance(type="ShapValues",data=__import__("catboost").Pool(X))
    return np.abs(sv[:,:-1]).mean(0)   # drop bias col

def load(path):
    df=pd.read_csv(path)
    assert TARGET in df, f"'{TARGET}' column missing"
    miss=[f for f in FEATS if f not in df]; assert not miss, f"missing features {miss}"
    df=df.dropna(subset=[TARGET]+FEATS+["region_code","year"]).copy()
    df["sido"]=(df["region_code"]//1000).astype(int)
    df["capital"]=df["sido"].isin([11,28,41]).astype(int)   # Seoul/Incheon/Gyeonggi
    return df

def baseline(df):
    tr=df[df.year<=2022]; va=df[df.year==2023]; te=df[df.year==2024]
    m=fit(tr[FEATS],tr[TARGET])
    r2=r2_score(te[TARGET],m.predict(te[FEATS])) if len(te) else np.nan
    print(f"[baseline] CatBoost Test(2024) R2 = {r2:.3f}  (canonical ~0.29)")
    return m,r2

def a1_spatial_cv(df):
    rows=[]
    for s in sorted(df.sido.unique()):
        tr=df[df.sido!=s]; te=df[df.sido==s]
        if len(te)<10: continue
        m=fit(tr[FEATS],tr[TARGET]); rows.append((s,len(te),r2_score(te[TARGET],m.predict(te[FEATS]))))
    r=pd.DataFrame(rows,columns=["sido","n","r2"]); r.to_csv(f"{TAB}/a1_spatial_cv.csv",index=False)
    print(f"[1] spatial block CV: mean R2={r.r2.mean():.3f} (sd {r.r2.std():.3f}); random-CV ~0.29 -> spatial drop = generalization gap")
    plt.figure(figsize=(7,3.5)); plt.bar(r.sido.astype(str),r.r2,color="#2c6fbb"); plt.axhline(r.r2.mean(),color="#c0392b",ls="--")
    plt.title("Leave-one-sido-out spatial CV R²"); plt.ylabel("R²"); plt.xticks(rotation=90,fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/a1_spatial_cv.png",dpi=200); plt.close()
    return r

def a2_bootstrap_temporal(df,B=50):
    tr=df[df.year<=2023]
    base=mean_abs_shap(fit(tr[FEATS],tr[TARGET]),tr[FEATS])
    boot=np.zeros((B,len(FEATS)))
    for b in range(B):
        s=tr.sample(len(tr),replace=True,random_state=b)
        boot[b]=mean_abs_shap(fit(s[FEATS],s[TARGET]),s[FEATS])
    ranks=(-boot).argsort(1).argsort(1)+1
    ci=pd.DataFrame({"feature":FEATS,"mean_abs_shap":base,
        "rank_med":np.median(ranks,0),"rank_lo":np.percentile(ranks,2.5,0),"rank_hi":np.percentile(ranks,97.5,0)})
    ci=ci.sort_values("mean_abs_shap",ascending=False); ci.to_csv(f"{TAB}/a2_shap_rank_ci.csv",index=False)
    pr=ci[ci.feature=="pagerank_lag1"].iloc[0]
    print(f"[2] SHAP bootstrap: PageRank rank median {pr.rank_med:.0f} (95% CI {pr.rank_lo:.0f}-{pr.rank_hi:.0f})")
    pre=mean_abs_shap(fit(df[df.year<=2020][FEATS],df[df.year<=2020][TARGET]),df[df.year<=2020][FEATS])
    post=mean_abs_shap(fit(df[df.year>=2021][FEATS],df[df.year>=2021][TARGET]),df[df.year>=2021][FEATS])
    rho=spearmanr(pre,post).correlation
    null=[]
    for b in range(10):  # permutation 10회 (p값 방향성 확인용)
        yr=df.year.sample(frac=1,random_state=b).values; d=df.assign(_p=(yr>=2021))
        p1=mean_abs_shap(fit(d[~d._p][FEATS],d[~d._p][TARGET]),d[~d._p][FEATS])
        p2=mean_abs_shap(fit(d[d._p][FEATS],d[d._p][TARGET]),d[d._p][FEATS])
        null.append(spearmanr(p1,p2).correlation)
    pval=float(np.mean([n>=rho for n in null]))
    pd.DataFrame([{"pre_post_rank_spearman":rho,"perm_p_ge":pval,"interpretation":"high rho = importance structure stable"}]).to_csv(f"{TAB}/a2_temporal_stability.csv",index=False)
    print(f"    temporal stability: pre/post Spearman rho={rho:.3f} (perm p={pval:.2f}) -> structure {'STABLE' if rho>0.7 else 'shifted'}")
    plt.figure(figsize=(8,4)); c=ci.head(10)
    plt.errorbar(c.mean_abs_shap,range(len(c))[::-1],xerr=None,fmt="o",color="#2c6fbb")
    for i,(_,r) in enumerate(c.iterrows()):
        plt.text(r.mean_abs_shap,len(c)-1-i,f"  #{int(r.rank_med)} [{int(r.rank_lo)}-{int(r.rank_hi)}]",va="center",fontsize=8)
    plt.yticks(range(len(c))[::-1],c.feature,fontsize=8); plt.xlabel("mean |SHAP|")
    plt.title("SHAP importance with bootstrap rank 95% CI"); plt.tight_layout(); plt.savefig(f"{FIG}/a2_rank_ci.png",dpi=200); plt.close()
    return ci

def a3_rai(df):
    d=df[df.year<=2022]; reg=d.groupby("region_code")
    rai_feats=[f for f in FEATS if f not in NETWORK and f in DOMAIN]
    Z=reg[rai_feats].mean().apply(lambda c:(c-c.mean())/c.std())
    doms=sorted(set(DOMAIN.values()))
    S={dm:Z[[f for f in rai_feats if DOMAIN.get(f)==dm]].mean(1) for dm in doms}
    Sdf=pd.DataFrame(S)
    try: w_shap=json.load(open("outputs_reference/rai_domain_weights.json"))
    except Exception: w_shap={"Econ":.1875,"Demo":.1771,"Infra":.2901,"Serv":.3453}
    w_shap={k:w_shap.get(k,0) for k in doms}
    schemes={"SHAP":w_shap,"Equal":{k:1/len(doms) for k in doms},
             "Entropy":dict(zip(doms,(lambda e:e/e.sum())(Sdf.apply(lambda c:c.std()).values)))}
    rai={n:(Sdf*pd.Series(w)).sum(1) for n,w in schemes.items()}
    pcs=PCA(1).fit(Sdf.fillna(0)); rai["PCA"]=pd.Series(pcs.transform(Sdf.fillna(0))[:,0],index=Sdf.index)
    R=pd.DataFrame(rai)
    corr=R.corr(method="spearman"); corr.to_csv(f"{TAB}/a3_rai_weight_robustness.csv")
    print("[3] RAI weight robustness (Spearman across schemes):\n",corr.round(3).to_string())
    fut=df[df.year>=2023].groupby("region_code")[TARGET].mean()
    ext=pd.concat([R["SHAP"],fut],axis=1,keys=["RAI","net_future"]).dropna()
    r=spearmanr(ext.RAI,ext.net_future).correlation
    pd.DataFrame([{"rai_vs_future_net_spearman":r,"n":len(ext)}]).to_csv(f"{TAB}/a3_rai_external_validity.csv",index=False)
    print(f"    external validity: RAI(<=2022) vs net migration 2023-24  Spearman r={r:.3f} (N={len(ext)})")
    return R

def a4_pdp_interaction(df,m):
    from sklearn.inspection import PartialDependenceDisplay
    d=df[df.year<=2023]
    fig,ax=plt.subplots(1,4,figsize=(15,3.4))
    for i,f in enumerate(["childcare_pk","house_age","closeness","pagerank_lag1"]):
        try: PartialDependenceDisplay.from_estimator(m,d[FEATS],[f],ax=ax[i]); ax[i].set_title(f)
        except Exception: ax[i].set_title(f+" (skip)")
    plt.tight_layout(); plt.savefig(f"{FIG}/a4_pdp.png",dpi=200); plt.close()
    sv=m.get_feature_importance(type="ShapValues",data=__import__("catboost").Pool(d[FEATS]))[:,:-1]
    j=FEATS.index("childcare_pk")
    plt.figure(figsize=(6,4)); sc=plt.scatter(d["childcare_pk"],sv[:,j],c=d["seoul_dist_km"],cmap="coolwarm_r",s=12)
    plt.colorbar(sc,label="seoul_dist_km"); plt.xlabel("childcare_pk"); plt.ylabel("SHAP(childcare)")
    plt.title("Childcare SHAP vs value, colored by distance-to-Seoul (cost proxy)")
    plt.tight_layout(); plt.savefig(f"{FIG}/a4_interaction_childcare.png",dpi=200); plt.close()
    print("[4] PDP + interaction figures saved")

def a5_subgroups(df):
    out=[]
    groups={"capital":df[df.capital==1],"non_capital":df[df.capital==0]}
    q=df.ln_pop.quantile([1/3,2/3]).values
    groups["size_small"]=df[df.ln_pop<=q[0]]; groups["size_mid"]=df[(df.ln_pop>q[0])&(df.ln_pop<=q[1])]; groups["size_large"]=df[df.ln_pop>q[1]]
    for name,g in groups.items():
        if len(g)<100: continue
        imp=mean_abs_shap(fit(g[FEATS],g[TARGET]),g[FEATS])
        top=pd.Series(imp,index=FEATS).sort_values(ascending=False)
        out.append({"group":name,"n":len(g),"top1":top.index[0],"top2":top.index[1],"top3":top.index[2],
                    "pagerank_rank":int((-imp).argsort().argsort()[FEATS.index("pagerank_lag1")]+1)})
    r=pd.DataFrame(out); r.to_csv(f"{TAB}/a5_subgroup_structure.csv",index=False)
    print("[5] subgroup determinant structure:\n",r.to_string(index=False))
    return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/track_C_2017_2024.csv"); a=ap.parse_args()
    os.makedirs(TAB,exist_ok=True); os.makedirs(FIG,exist_ok=True)
    df=load(a.data); print(f"loaded {len(df)} rows, {df.region_code.nunique()} regions, years {df.year.min()}-{df.year.max()}")
    m,_=baseline(df); a1_spatial_cv(df); a2_bootstrap_temporal(df); a3_rai(df); a4_pdp_interaction(df,m); a5_subgroups(df)
    print("\nDONE -> results/tables, results/figures")

if __name__=="__main__": main()
