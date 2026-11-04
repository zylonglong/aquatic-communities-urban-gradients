from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_CN = {"terrain": "地形地貌", "urbanization": "城市化建设", "human_activity": "人类直接活动"}
ALPHAS = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
SEEDS = [11, 29, 53]


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float); q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any(): return q
    pv = p[ok]; order = np.argsort(pv); ranked = pv[order]; m = len(ranked)
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    back = np.empty(m); back[order] = np.clip(adj, 0, 1); q[ok] = back
    return q


def tag(r): return f"{r:04d}m"


def raw_module_variables(df, module, r):
    t = tag(r)
    if module == "terrain":
        cols = [c for c in df.columns if c.startswith(f"terrain_{r}m_")]
        cols += [c for c in ["terrain_point_dem_m", "terrain_point_slope_pct"] if c in df]
        cols = [c for c in cols if "dem_var_m2" not in c]  # exact square of SD, redundant by construction
    elif module == "urbanization":
        cols = [c for c in df.columns if c.startswith(f"urbanization_{t}_") or c.startswith(f"landuse_{t}_")]
        p = f"worldpop_{t}_mean_density"
        if p in df: cols.append(p)
    else:
        cols = [c for c in df.columns if c.startswith(f"api_{t}_")]
    return list(dict.fromkeys(cols))


def response_transform(x, column, family):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    c = column.lower()
    rule = "原尺度"
    if "pcoa" in c or family in ["beta_diversity"]:
        return x, rule
    if any(k in c for k in ["proportion", "percentage", "_pct", "percent"]):
        p = x / 100 if x.max(skipna=True) > 1.01 else x
        p = p.clip(0, 1)
        return np.arcsin(np.sqrt(p)), "比例换算至0–1后arcsin(sqrt(p))"
    if x.min(skipna=True) >= 0 and any(k in c for k in ["density", "biomass", "weight", "length", "ratio"]):
        return np.log1p(x), "log1p"
    return x, rule


def control_frame(d):
    lon = pd.to_numeric(d.longitude, errors="coerce")
    lat = pd.to_numeric(d.latitude, errors="coerce")
    return pd.DataFrame({
        "longitude": lon, "latitude": lat,
        "longitude2": lon * lon, "latitude2": lat * lat,
        "lon_lat": lon * lat,
        "n_sample_records": pd.to_numeric(d.get("n_sample_records", 1), errors="coerce"),
    }, index=d.index)


def make_pipe(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def choose_alpha(X, y, groups):
    unique = np.unique(groups)
    losses = []
    for alpha in ALPHAS:
        fold_loss = []
        for g in unique:
            tr, va = groups != g, groups == g
            if tr.sum() < 12 or va.sum() < 3: continue
            try:
                m = make_pipe(alpha).fit(X.iloc[tr], y[tr])
                fold_loss.append(mean_absolute_error(y[va], m.predict(X.iloc[va])))
            except Exception:
                pass
        losses.append(np.mean(fold_loss) if fold_loss else np.inf)
    return ALPHAS[int(np.argmin(losses))]


def spatial_cv_one(d, y, xmodule, water, seed):
    valid = np.isfinite(y)
    dd = d.loc[valid].copy(); yy = np.asarray(y)[valid]
    X0 = control_frame(dd)
    Xm = pd.concat([X0, dd[xmodule].apply(pd.to_numeric, errors="coerce")], axis=1)
    k = 5 if water == "River" else 4
    coords = RobustScaler().fit_transform(dd[["longitude", "latitude"]])
    groups = KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(coords)
    pred0 = np.full(len(dd), np.nan); predm = np.full(len(dd), np.nan)
    chosen0, chosenm = [], []
    for fold in range(k):
        tr, te = groups != fold, groups == fold
        if tr.sum() < 18 or te.sum() < 3: continue
        # GCV is performed only inside the outer training fold. This retains
        # leakage control while avoiding an unstable second spatial split in
        # Lake, where an outer-training set contains only three spatial blocks.
        def fit_gcv(X):
            imp = SimpleImputer(strategy="median", add_indicator=True)
            scl = RobustScaler()
            ztr = scl.fit_transform(imp.fit_transform(X.iloc[tr]))
            zte = scl.transform(imp.transform(X.iloc[te]))
            mdl = RidgeCV(alphas=ALPHAS, cv=None).fit(ztr, yy[tr])
            return mdl.predict(zte), float(mdl.alpha_)
        pred0[te], a0 = fit_gcv(X0)
        predm[te], am = fit_gcv(Xm)
        chosen0.append(a0); chosenm.append(am)
    ok = np.isfinite(pred0) & np.isfinite(predm)
    if ok.sum() < max(15, int(0.7 * len(dd))): return None
    def safe_r2(a,b):
        try: return r2_score(a,b)
        except Exception: return np.nan
    rho0 = stats.spearmanr(yy[ok], pred0[ok]).statistic
    rhom = stats.spearmanr(yy[ok], predm[ok]).statistic
    return {
        "n_sites": int(ok.sum()), "outer_folds": k, "seed": seed,
        "baseline_r2": safe_r2(yy[ok], pred0[ok]), "module_r2": safe_r2(yy[ok], predm[ok]),
        "delta_cv_r2": safe_r2(yy[ok], predm[ok]) - safe_r2(yy[ok], pred0[ok]),
        "baseline_mae": mean_absolute_error(yy[ok], pred0[ok]), "module_mae": mean_absolute_error(yy[ok], predm[ok]),
        "delta_mae_baseline_minus_module": mean_absolute_error(yy[ok], pred0[ok]) - mean_absolute_error(yy[ok], predm[ok]),
        "baseline_rmse": math.sqrt(mean_squared_error(yy[ok], pred0[ok])), "module_rmse": math.sqrt(mean_squared_error(yy[ok], predm[ok])),
        "baseline_spearman": rho0, "module_spearman": rhom,
        "delta_prediction_spearman": rhom - rho0,
        "median_alpha_baseline": float(np.median(chosen0)), "median_alpha_module": float(np.median(chosenm)),
    }


def select_cv_targets(candidates, redundancy):
    c = candidates[candidates.screen_rank <= 3].copy()
    if not redundancy.empty:
        reps = redundancy[redundancy.is_omnibus_representative][["water_body_type", "response_family", "column_name"]].drop_duplicates()
        c = c.merge(reps.assign(is_rep=True), on=["water_body_type", "response_family", "column_name"], how="left")
        c["is_rep"] = c.is_rep.fillna(False)
    else: c["is_rep"] = True
    c["non_pcoa"] = ~c.column_name.str.contains("pcoa", case=False, na=False)
    c = c.sort_values(["screen_rank", "is_rep", "non_pcoa", "max_abs_rho"], ascending=[True, False, False, False])
    return c.groupby(["water_body_type", "environment_module", "response_family"], as_index=False).head(1)


def cluster_predictors(d, cols, threshold=.85):
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    good = [c for c in cols if X[c].notna().mean() >= .7 and X[c].nunique(dropna=True) > 1]
    if len(good) <= 1: return good, {c: c for c in good}
    corr = X[good].corr(method="spearman").fillna(0).abs().clip(0, 1)
    np.fill_diagonal(corr.values, 1)
    dist = 1 - corr.values
    Z = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(Z, t=1-threshold, criterion="distance")
    mapping, reps = {}, []
    for cl in np.unique(lab):
        members = [good[i] for i in np.where(lab == cl)[0]]
        rep = sorted(members, key=lambda c: (-X[c].notna().sum(), len(c), c))[0]
        reps.append(rep)
        mapping.update({m: rep for m in members})
    return reps, mapping


def residualize_matrix(y, X, controls):
    ok = np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    ok &= np.all(np.isfinite(X), axis=1)
    yy = y[ok]; xx = X[ok]; cc = controls[ok]
    C = np.column_stack([np.ones(ok.sum()), cc])
    yr = yy - C @ np.linalg.lstsq(C, yy, rcond=None)[0]
    xr = xx - C @ np.linalg.lstsq(C, xx, rcond=None)[0]
    return ok, yr, xr


def mechanism_stability(site, targets, env_names):
    single_rows, boot_rows, cluster_rows = [], [], []
    rng = np.random.default_rng(20260722)
    for _, t in targets.iterrows():
        water, module, col, fam = t.water_body_type, t.environment_module, t.column_name, t.response_family
        d = site[site.water_body_type.eq(water)].copy()
        y, yrule = response_transform(d[col], col, fam)
        controls = control_frame(d).to_numpy(dtype=float)
        for r in RANGES:
            cols = raw_module_variables(d, module, r)
            reps, cmap = cluster_predictors(d, cols)
            for raw, rep in cmap.items():
                cluster_rows.append({"water_body_type":water,"environment_module":module,"response_variable":col,"range_m":r,"raw_predictor":raw,"cluster_representative":rep,"is_representative":raw==rep})
            for xcol in reps:
                xv = pd.to_numeric(d[xcol], errors="coerce").to_numpy(float)
                valid = np.isfinite(y) & np.isfinite(xv) & np.all(np.isfinite(controls),axis=1)
                if valid.sum() < (25 if water=="River" else 20): continue
                # Partial Spearman via rank residuals.
                C = np.column_stack([np.ones(valid.sum()), controls[valid]])
                ry = stats.rankdata(np.asarray(y)[valid]); rx = stats.rankdata(xv[valid])
                ry -= C @ np.linalg.lstsq(C, ry, rcond=None)[0]
                rx -= C @ np.linalg.lstsq(C, rx, rcond=None)[0]
                rho = np.corrcoef(ry,rx)[0,1]
                n=valid.sum(); k=controls.shape[1]; df=n-k-2
                p=2*stats.t.sf(abs(rho)*math.sqrt(df/max(1e-12,1-rho*rho)),df) if abs(rho)<1 else 0
                key = re.sub(r"_(?:0?200|0?500|1000|1500|2000|2500|3000|3500|4000|4500|5000|5500|6000)m_", "_{range}_", xcol)
                single_rows.append({"water_body_type":water,"environment_module":module,"response_family":fam,"response_variable":col,"response_name_cn":t.variable_name_cn,"range_m":r,"predictor":xcol,"predictor_name_cn":env_names.get(xcol,xcol),"predictor_scale_invariant_key":key,"partial_spearman_rho":rho,"p_value":p,"n_sites":int(n),"response_transform":yrule})
        # Bootstrap conditional Ridge coefficients at the response's peak association scale.
        r = int(t.peak_range_m); cols = raw_module_variables(d,module,r); reps,cmap=cluster_predictors(d,cols)
        Xdf=d[reps].apply(pd.to_numeric,errors="coerce"); Xdf=Xdf.fillna(Xdf.median())
        yy=np.asarray(y,float); C=control_frame(d).to_numpy(float)
        ok, yr, xr=residualize_matrix(yy,Xdf.to_numpy(float),C)
        if len(yr) < 20 or xr.shape[1]==0: continue
        xs=RobustScaler().fit_transform(xr); ys=RobustScaler().fit_transform(yr.reshape(-1,1)).ravel()
        # Select alpha by generalized leave-one-out-like fixed grid using 5 random folds.
        fold=KMeans(n_clusters=min(5,max(3,len(ys)//10)),random_state=17,n_init=20).fit_predict(C[ok,:2])
        losses=[]
        for a in ALPHAS:
            ls=[]
            for g in np.unique(fold):
                tr,te=fold!=g,fold==g
                if tr.sum()>10 and te.sum()>2:
                    m=Ridge(alpha=a).fit(xs[tr],ys[tr]); ls.append(mean_absolute_error(ys[te],m.predict(xs[te])))
            losses.append(np.mean(ls) if ls else np.inf)
        alpha=ALPHAS[int(np.argmin(losses))]
        coefs=[]
        for b in range(300):
            ii=rng.integers(0,len(ys),len(ys))
            coefs.append(Ridge(alpha=alpha).fit(xs[ii],ys[ii]).coef_)
        coefs=np.asarray(coefs)
        base=Ridge(alpha=alpha).fit(xs,ys).coef_
        for j,xcol in enumerate(reps):
            boot_rows.append({"water_body_type":water,"environment_module":module,"response_family":fam,"response_variable":col,"response_name_cn":t.variable_name_cn,"range_m":r,"predictor":xcol,"predictor_name_cn":env_names.get(xcol,xcol),"ridge_beta":base[j],"bootstrap_beta_median":np.median(coefs[:,j]),"bootstrap_ci95_low":np.quantile(coefs[:,j],.025),"bootstrap_ci95_high":np.quantile(coefs[:,j],.975),"direction_stability":max(np.mean(coefs[:,j]>0),np.mean(coefs[:,j]<0)),"alpha":alpha,"n_sites":len(ys)})
    single=pd.DataFrame(single_rows)
    if not single.empty:
        single["q_within_target_module_all_scales"] = np.nan
        for _,ix in single.groupby(["water_body_type","environment_module","response_variable"]).groups.items():
            single.loc[ix,"q_within_target_module_all_scales"] = bh_fdr(single.loc[ix,"p_value"])
    return single,pd.DataFrame(boot_rows),pd.DataFrame(cluster_rows)


def river_lake_effect_differences(assoc):
    key=["environment_module","response_family","column_name","variable_name_cn","range_m"]
    r=assoc[assoc.water_body_type.eq("River")].set_index(key)
    l=assoc[assoc.water_body_type.eq("Lake")].set_index(key)
    common=r.index.intersection(l.index); rows=[]; k=6
    for ix in common:
        rr=r.loc[ix]; ll=l.loc[ix]
        if isinstance(rr,pd.DataFrame): rr=rr.iloc[0]
        if isinstance(ll,pd.DataFrame): ll=ll.iloc[0]
        if not np.isfinite(rr.partial_spearman_rho) or not np.isfinite(ll.partial_spearman_rho): continue
        zr=np.arctanh(np.clip(rr.partial_spearman_rho,-.999,.999)); zl=np.arctanh(np.clip(ll.partial_spearman_rho,-.999,.999))
        se=math.sqrt(1/max(1,rr.n_sites-k-3)+1/max(1,ll.n_sites-k-3)); z=(zr-zl)/se; p=2*stats.norm.sf(abs(z))
        row = dict(zip(key, ix))
        row.update({"river_rho":rr.partial_spearman_rho,"lake_rho":ll.partial_spearman_rho,"rho_difference_river_minus_lake":rr.partial_spearman_rho-ll.partial_spearman_rho,"fisher_z_test":z,"p_value":p,"river_n":rr.n_sites,"lake_n":ll.n_sites})
        rows.append(row)
    out=pd.DataFrame(rows)
    if not out.empty:
        out["q_within_module_all_common_responses_ranges"]=np.nan
        for _,ix in out.groupby("environment_module").groups.items(): out.loc[ix,"q_within_module_all_common_responses_ranges"]=bh_fdr(out.loc[ix,"p_value"])
    return out


def main():
    out=Path(__file__).resolve().parents[1]; root=out.parent/"ML_Input_原始量_13尺度_2026-07-04"
    site=pd.read_excel(root/"ML_Input_Raw_13Ranges_Sample_Site.xlsx",sheet_name="Site_Special")
    candidates=pd.read_csv(out/"05_Q4_具体生物响应"/"具体响应尺度类型与候选清单.csv")
    assoc=pd.read_csv(out/"05_Q4_具体生物响应"/"全部具体响应_偏Spearman_分层FDR.csv")
    redundancy=pd.read_csv(out/"01_数据审计"/"家族全局检验响应冗余控制映射.csv")
    targets=select_cv_targets(candidates,redundancy)
    targets.to_csv(out/"05_Q4_具体生物响应"/"空间诊断预设代表响应清单.csv",index=False,encoding="utf-8-sig")

    cv_path=out/"07_River_Lake时间空间敏感性"/"代表响应_重复空间阻塞CV_13尺度.csv"
    if cv_path.exists():
        cv=pd.read_csv(cv_path)
    else:
        cv_rows=[]
        for _,t in targets.iterrows():
            d=site[site.water_body_type.eq(t.water_body_type)].copy()
            y,rule=response_transform(d[t.column_name],t.column_name,t.response_family)
            for r in RANGES:
                cols=raw_module_variables(d,t.environment_module,r)
                for seed in SEEDS:
                    res=spatial_cv_one(d,np.asarray(y,float),cols,t.water_body_type,seed)
                    if res:
                        cv_rows.append({"water_body_type":t.water_body_type,"environment_module":t.environment_module,"environment_module_cn":MODULE_CN[t.environment_module],"response_family":t.response_family,"response_family_cn":t.response_family_cn,"response_variable":t.column_name,"response_name_cn":t.variable_name_cn,"range_m":r,"n_module_predictors":len(cols),"response_transform":rule,**res})
        cv=pd.DataFrame(cv_rows)
        cv.to_csv(cv_path,index=False,encoding="utf-8-sig")
    cv_summary=cv.groupby(["water_body_type","environment_module","environment_module_cn","response_family","response_family_cn","response_variable","response_name_cn","range_m"]).agg(n_repeats=("seed","nunique"),n_sites=("n_sites","median"),delta_cv_r2_median=("delta_cv_r2","median"),delta_cv_r2_min=("delta_cv_r2","min"),delta_cv_r2_max=("delta_cv_r2","max"),delta_mae_median=("delta_mae_baseline_minus_module","median"),delta_prediction_spearman_median=("delta_prediction_spearman","median"),positive_delta_r2_fraction=("delta_cv_r2",lambda x:np.mean(np.asarray(x)>0)),positive_delta_mae_fraction=("delta_mae_baseline_minus_module",lambda x:np.mean(np.asarray(x)>0))).reset_index()
    cv_summary.to_csv(out/"07_River_Lake时间空间敏感性"/"代表响应_空间CV汇总.csv",index=False,encoding="utf-8-sig")

    # Environmental Chinese names.
    env_names={}
    for fn in ["Feature_Process_DEM_Terrain_13Ranges.xlsx","Feature_Process_Urbanization_OSM_13Ranges.xlsx","Feature_Process_WorldCover_13Ranges.xlsx","Feature_Process_WorldPop_13Ranges.xlsx","Feature_Process_API_Raw_13Ranges.xlsx"]:
        dct=pd.read_excel(root/fn,sheet_name="Feature_Dictionary")
        env_names.update(dict(zip(dct.column_name,dct.variable_name_cn)))
    mdir=out/"05_Q4_具体生物响应"; mdir.mkdir(exist_ok=True)
    sp=mdir/"代表响应_模块内具体变量跨尺度偏相关.csv"; bp=mdir/"峰值尺度_共线性控制后Ridge_Bootstrap稳定性.csv"; cp=mdir/"模块内预测变量相关聚类映射.csv"
    if sp.exists() and bp.exists() and cp.exists():
        single=pd.read_csv(sp); boot=pd.read_csv(bp); clusters=pd.read_csv(cp)
    else:
        single,boot,clusters=mechanism_stability(site,targets,env_names)
        single.to_csv(sp,index=False,encoding="utf-8-sig")
        boot.to_csv(bp,index=False,encoding="utf-8-sig")
        clusters.to_csv(cp,index=False,encoding="utf-8-sig")

    diff=river_lake_effect_differences(assoc)
    ddir=out/"07_River_Lake时间空间敏感性"; ddir.mkdir(exist_ok=True)
    diff.to_csv(ddir/"River_Lake标准化效应差_FisherZ检验.csv",index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(out/"空间CV_机制变量_RiverLake差异汇总.xlsx",engine="openpyxl") as xw:
        targets.to_excel(xw,sheet_name="CV代表响应",index=False)
        cv_summary.to_excel(xw,sheet_name="空间CV汇总",index=False)
        single.to_excel(xw,sheet_name="具体变量跨尺度",index=False)
        boot.to_excel(xw,sheet_name="RidgeBootstrap",index=False)
        diff.to_excel(xw,sheet_name="RiverLake效应差",index=False)
    print(json.dumps({"targets":len(targets),"cv_rows":len(cv),"mechanism_single":len(single),"bootstrap_rows":len(boot),"river_lake_differences":len(diff)},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
