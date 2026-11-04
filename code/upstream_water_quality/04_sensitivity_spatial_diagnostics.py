# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"E:\桌面\博士课题\Data")
OUT = Path(__file__).resolve().parent.parent
WQROOT = OUT
PROC, RES = OUT / "01_INPUT与过程数据", OUT / "02_统计结果"
SCALES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
RNG = np.random.default_rng(20260720)


def perm_rho(x, y, nperm=4999):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx-rx.mean())/np.linalg.norm(rx-rx.mean()); ry = (ry-ry.mean())/np.linalg.norm(ry-ry.mean())
    rho = float(rx @ ry)
    rp = np.vstack([RNG.permutation(ry) for _ in range(nperm)]) @ rx
    return rho, float((np.sum(np.abs(rp) >= abs(rho)-1e-12)+1)/(nperm+1))


def bh(x):
    x = np.asarray(x, float); out = np.full(len(x), np.nan); ok = np.isfinite(x)
    if ok.any(): out[ok] = multipletests(x[ok], method="fdr_bh")[1]
    return out


def moran_knn(df, ycol, k=4, nperm=4999):
    d = df[["spatial_id", "longitude", "latitude", ycol]].dropna().copy()
    n = len(d)
    if n < 6:
        return {"response": ycol, "n": n, "moran_I": np.nan, "permutation_p": np.nan, "k": np.nan}
    kk = min(k, n-1)
    xy = d[["longitude", "latitude"]].to_numpy()
    # Local UTM-like scaling sufficient for neighbour identification.
    xy[:, 0] *= np.cos(np.deg2rad(xy[:, 1].mean()))
    nbr = NearestNeighbors(n_neighbors=kk+1).fit(xy).kneighbors(return_distance=False)[:, 1:]
    W = np.zeros((n, n), float)
    for i in range(n): W[i, nbr[i]] = 1.0
    W = np.maximum(W, W.T); W /= W.sum(axis=1, keepdims=True)
    y = d[ycol].to_numpy(float); z = y-y.mean()
    I = float(n/W.sum() * (z @ W @ z)/(z @ z))
    vals=[]
    for _ in range(nperm):
        zp=RNG.permutation(z); vals.append(n/W.sum()*(zp@W@zp)/(zp@zp))
    vals=np.asarray(vals)
    p=float((np.sum(np.abs(vals-vals.mean()) >= abs(I-vals.mean()))+1)/(nperm+1))
    return {"response": ycol, "n": n, "moran_I": I, "permutation_p": p, "k": kk}


site = pd.read_csv(PROC / "WQ_Site_Special_13Ranges.csv")
scores = pd.read_csv(RES / "环境模块梯度得分_13尺度.csv")

# One row per physical geometry for spatial diagnostics.  All public station
# units remain in WQ_Site_Special, but duplicated source/publication IDs must
# not be treated as independent neighbours.
site_geom = site.groupby(["spatial_id", "spatial_waterbody_type"], as_index=False).agg(
    longitude=("longitude", "first"), latitude=("latitude", "first"),
    wq_class_score_mean=("wq_class_score_mean", "mean"),
    pollution_index_mean=("pollution_index_mean", "mean"),
    n_station_units=("station_id", "nunique"),
)

# Cross-driver dependence at each scale.
cross=[]
for wt in ["Lake", "River"]:
    for sc in SCALES:
        p=scores[(scores.water_body_type==wt)&(scores.range_m==sc)].pivot(index="spatial_id", columns="driver", values="gradient_score")
        for a,b in [("城市化建设","地形地貌"),("城市化建设","人类直接活动"),("地形地貌","人类直接活动")]:
            d=p[[a,b]].dropna(); rho,pv=perm_rho(d[a],d[b],1999)
            cross.append({"water_body_type":wt,"range_m":sc,"driver_1":a,"driver_2":b,"n":len(d),"spearman_rho":rho,"permutation_p":pv})
cross=pd.DataFrame(cross); cross["fdr_q"]=cross.groupby(["water_body_type","driver_1","driver_2"])["permutation_p"].transform(bh)
cross.to_csv(RES / "三类环境梯度跨尺度相关性.csv",index=False,encoding="utf-8-sig")

# Spatial clustering of the response itself.
mrows=[]
for wt,y in [("Lake","wq_class_score_mean"),("River","wq_class_score_mean"),("River","pollution_index_mean")]:
    r=moran_knn(site_geom[site_geom.spatial_waterbody_type==wt],y); r["water_body_type"]=wt; mrows.append(r)
pd.DataFrame(mrows).to_csv(RES / "水质响应空间自相关_MoranI.csv",index=False,encoding="utf-8-sig")

# Sensitivity of upper-bound categories (≤II/≤III).  Prefer the archived
# standardized long table when present; otherwise reconstruct the required
# fields from the self-contained record-level INPUT retained in this output.
long_path=WQROOT/r"02_结构化数据\深圳及周边公共水质_标准长表.csv"
if long_path.exists():
    long=pd.read_csv(long_path,low_memory=False)
else:
    record=pd.read_csv(PROC/"WQ_Record_Special_13Ranges.csv",low_memory=False)
    long=record.loc[record["水质类别"].notna(),
                    ["station_id","水质类别","monitoring_date_end"]].copy()
    long=long.rename(columns={"水质类别":"water_quality_class"})
    long["include_primary"]=True
    long["indicator_name_std"]="水质类别"
reg=pd.read_csv(PROC/"WQ_Station_Spatial_Registry.csv")
reg["spatial_eligible"]=reg.spatial_eligible.astype(str).str.lower().isin(["true","1"])
long=long[long.include_primary.astype(str).str.lower().isin(["true","1"]) & (long.indicator_name_std=="水质类别")]
long=long.merge(reg[["station_id","geometry_id","spatial_waterbody_type","spatial_eligible"]],on="station_id",how="left")
# Sensitivity inference must use the same unique physical geometry key as
# the environmental gradient table.  Registry ``spatial_id`` identifies a
# public station/source unit and therefore cannot be used for this join.
long["spatial_id"] = long["geometry_id"]
long=long[long.spatial_eligible].copy(); long["month"]=pd.to_datetime(long.monitoring_date_end,errors="coerce").dt.to_period("M").astype(str)

def score_scenario(x,scenario):
    s="" if pd.isna(x) else str(x).strip().replace("类","")
    exact={"Ⅰ":1.,"Ⅱ":2.,"Ⅲ":3.,"Ⅳ":4.,"Ⅴ":5.,"劣Ⅴ":6.}
    if s in exact:return exact[s]
    if s=="≤Ⅱ":return {"conservative":2.,"midpoint":1.5,"optimistic":1.}[scenario]
    if s=="≤Ⅲ":return {"conservative":3.,"midpoint":2.,"optimistic":1.}[scenario]
    return np.nan

srows=[]
for scenario in ["conservative","midpoint","optimistic"]:
    x=long.copy(); x["score"]=[score_scenario(v,scenario) for v in x.water_quality_class]
    # Equal-month mean within station/source unit, followed by an equal mean
    # across source units sharing the same geometry.
    station_mon=(x.dropna(subset=["score"])
                 .groupby(["station_id","spatial_id","spatial_waterbody_type","month"],as_index=False)
                 .score.mean())
    mon=(station_mon.groupby(["spatial_id","spatial_waterbody_type","month"],as_index=False)
         .score.mean())
    sy=mon.groupby(["spatial_id","spatial_waterbody_type"],as_index=False).score.mean()
    for wt in ["Lake","River"]:
        y=sy[sy.spatial_waterbody_type==wt][["spatial_id","score"]]
        for driver in ["城市化建设","人类直接活动","地形地貌"]:
            for sc in SCALES:
                g=scores[(scores.water_body_type==wt)&(scores.driver==driver)&(scores.range_m==sc)][["spatial_id","gradient_score"]]
                d=y.merge(g,on="spatial_id"); rho,pv=perm_rho(d.gradient_score,d.score,1999)
                srows.append({"scenario":scenario,"water_body_type":wt,"driver":driver,"range_m":sc,"n":len(d),"rho":rho,"p":pv})
sens=pd.DataFrame(srows); sens["q"]=sens.groupby(["scenario","water_body_type","driver"])["p"].transform(bh)
sens.to_csv(RES/"水质类别上界赋值敏感性_13尺度.csv",index=False,encoding="utf-8-sig")
best=(sens.assign(abs_rho=sens.rho.abs()).sort_values("abs_rho",ascending=False).groupby(["scenario","water_body_type","driver"],as_index=False).head(1))
best.to_csv(RES/"水质类别上界赋值敏感性_候选最强尺度.csv",index=False,encoding="utf-8-sig")
print("cross",len(cross),"moran",len(mrows),"sensitivity",len(sens))
