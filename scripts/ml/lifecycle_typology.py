# -*- coding: utf-8 -*-
"""Life-cycle frame — two additional analyses (run from repo root):
  A) Place-type typology: cluster municipalities to test whether net-outflow regions
     split into distinct types (dense metropolitan cores vs. old-housing rural areas).
  B) Inverted-U: LOWESS of net migration vs. population density (suburbanization).
Run:  python scripts/ml/lifecycle_typology.py --data data/track_C_2017_2024.csv
"""
import os,argparse,warnings,numpy as np,pandas as pd; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess
TARGET="net_rate"; TAB="results/tables"; FIG="results/figures"
# place-type features for the typology (edit to what exists / is meaningful)
TYPO=["pop_density","house_age","aging_ratio","seoul_dist_km","childcare_pk","ln_pop","youth_ratio","fiscal_indep","extinction_risk"]

def load(path):
    df=pd.read_csv(path)
    keep=[c for c in set(TYPO+[TARGET,"region_code","year"]) if c in df.columns]
    return df.dropna(subset=[c for c in keep if c!="year"]).copy()

def A_typology(reg,k=4):
    feats=[f for f in TYPO if f in reg.columns]
    Z=StandardScaler().fit_transform(reg[feats])
    reg=reg.copy(); reg["cluster"]=KMeans(k,n_init=10,random_state=42).fit_predict(Z)
    prof=reg.groupby("cluster")[feats+[TARGET]].mean().round(3); prof["n"]=reg.groupby("cluster").size()
    prof.to_csv(f"{TAB}/t1_cluster_profiles.csv"); reg[["region_code","cluster",TARGET]].to_csv(f"{TAB}/t1_region_clusters.csv",index=False)
    print("[A] place-type typology (cluster means):"); print(prof.to_string())
    P=PCA(2).fit_transform(Z)
    plt.figure(figsize=(7,6))
    sc=plt.scatter(P[:,0],P[:,1],c=reg[TARGET],cmap="RdBu",vmin=-20,vmax=20,s=30,edgecolor="k",linewidth=0.3)
    for c in sorted(reg.cluster.unique()):
        cx,cy=P[reg.cluster==c].mean(0); plt.text(cx,cy,f"C{c}",fontsize=13,fontweight="bold")
    plt.colorbar(sc,label="net migration rate"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Place-type typology (PCA), colored by net migration"); plt.tight_layout()
    plt.savefig(f"{FIG}/t1_typology_pca.png",dpi=200); plt.close(); return reg

def B_invertedU(reg):
    x=reg["pop_density"].values; y=reg[TARGET].values
    lo=lowess(y,x,frac=0.5,return_sorted=True)
    plt.figure(figsize=(7.5,5)); plt.scatter(x,y,s=18,alpha=0.5,color="#7f8c8d")
    plt.plot(lo[:,0],lo[:,1],color="#c0392b",lw=2.5,label="LOWESS"); plt.axhline(0,color="#999",ls="--")
    plt.xscale("log"); plt.xlabel("population density (log)"); plt.ylabel("net migration rate")
    plt.title("Inverted-U: net migration vs. population density (suburbanization)")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/t2_invertedU_density.png",dpi=200); plt.close()
    reg=reg.copy(); reg["dens_tertile"]=pd.qcut(reg["pop_density"],3,labels=["low","mid","high"])
    t=reg.groupby("dens_tertile")[TARGET].agg(["mean","std","count"]); t.to_csv(f"{TAB}/t2_density_tertile.csv")
    print("[B] net_rate by density tertile (inverted-U):"); print(t.round(2).to_string())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/track_C_2017_2024.csv"); a=ap.parse_args()
    os.makedirs(TAB,exist_ok=True); os.makedirs(FIG,exist_ok=True)
    df=load(a.data); reg=df.groupby("region_code").mean(numeric_only=True).reset_index()
    reg=A_typology(reg); B_invertedU(reg); print("\nDONE -> results/tables, results/figures")
if __name__=="__main__": main()
