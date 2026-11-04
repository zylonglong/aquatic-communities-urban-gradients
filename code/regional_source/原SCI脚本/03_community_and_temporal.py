from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

RANGES=[200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
MODULES=["terrain","urbanization","human_activity"]
MODULE_CN={"terrain":"地形地貌","urbanization":"城市化建设","human_activity":"人类直接活动"}
GROUP_SHEETS={"benthos":"Benthos","fish":"Fish","macrophyte":"Macrophytes","phytoplankton":"Phytoplankton","periphyton":"Periphyton","zooplankton":"Zooplankton"}
GROUP_CN={"benthos":"底栖动物","fish":"鱼类","macrophyte":"水生植物","phytoplankton":"浮游植物","periphyton":"着生藻类","zooplankton":"浮游动物"}


def bh(p):
    p=np.asarray(p,float); q=np.full(p.shape,np.nan); ok=np.isfinite(p)
    if not ok.any(): return q
    x=p[ok]; o=np.argsort(x); s=x[o]; m=len(s); a=np.minimum.accumulate((s*m/np.arange(1,m+1))[::-1])[::-1]
    b=np.empty(m); b[o]=np.clip(a,0,1); q[ok]=b; return q


def robust_z(x):
    x=pd.to_numeric(x,errors="coerce"); med=x.median(); iqr=x.quantile(.75)-x.quantile(.25)
    if not np.isfinite(iqr) or iqr<=0: iqr=x.std(ddof=0) or 1
    return (x-med)/iqr


def controls(d):
    lon=robust_z(d.longitude).to_numpy(); lat=robust_z(d.latitude).to_numpy(); nr=robust_z(d.n_sample_records).fillna(0).to_numpy()
    return np.column_stack([lon,lat,lon*lon,lat*lat,lon*lat,nr])


def residualize(A,C):
    C=np.column_stack([np.ones(len(C)),C]); return A-C@np.linalg.lstsq(C,A,rcond=None)[0]


def rda_test(Y,x,C,rng,nperm=499):
    ok=np.isfinite(x)&np.all(np.isfinite(C),axis=1)&np.all(np.isfinite(Y),axis=1)
    Y=Y[ok]; x=x[ok]; C=C[ok]
    if len(x)<20 or Y.shape[1]<2: return None
    Yr=residualize(Y,C); xr=residualize(x[:,None],C).ravel(); xr=xr-xr.mean()
    ss_total=np.sum(Yr*Yr)
    if ss_total<=0 or np.sum(xr*xr)<=0: return None
    beta=(xr[:,None]*Yr).sum(axis=0)/np.sum(xr*xr); fit=xr[:,None]*beta
    ss=np.sum(fit*fit); r2=ss/ss_total; df2=max(1,len(x)-C.shape[1]-2); F=(r2/max(1e-12,1-r2))*df2
    null=[]
    for _ in range(nperm):
        xp=rng.permutation(xr); b=(xp[:,None]*Yr).sum(axis=0)/np.sum(xp*xp); s=np.sum((xp[:,None]*b)**2)/ss_total
        null.append((s/max(1e-12,1-s))*df2)
    p=(1+np.sum(np.asarray(null)>=F))/(nperm+1)
    return {"n_sites":len(x),"n_taxa":Y.shape[1],"partial_rda_r2":r2,"pseudo_F":F,"permutation_p":p}


def pcoa_coords(D):
    n=D.shape[0]; H=np.eye(n)-np.ones((n,n))/n; B=-.5*H@(D*D)@H
    val,vec=np.linalg.eigh(B); o=np.argsort(val)[::-1]; val=val[o]; vec=vec[:,o]; pos=val>1e-12
    return vec[:,pos]*np.sqrt(val[pos])


def dispersion_test(Y,score,rng,nperm=499):
    ok=np.isfinite(score)&np.all(np.isfinite(Y),axis=1); Y=Y[ok]; s=score[ok]
    if len(s)<20:return None
    q1,q2=np.quantile(s,[1/3,2/3]); keep=(s<=q1)|(s>=q2); Y=Y[keep]; s=s[keep]
    lab=np.where(s<=q1,0,1); D=squareform(pdist(Y,metric="braycurtis")); D=np.nan_to_num(D); Z=pcoa_coords(D)
    if Z.shape[1]==0:return None
    dist=np.zeros(len(lab))
    for g in [0,1]:
        cen=Z[lab==g].mean(axis=0); dist[lab==g]=np.sqrt(((Z[lab==g]-cen)**2).sum(axis=1))
    delta=dist[lab==1].mean()-dist[lab==0].mean(); null=[]
    for _ in range(nperm):
        lp=rng.permutation(lab); dd=np.zeros(len(lp))
        for g in [0,1]:
            cen=Z[lp==g].mean(axis=0); dd[lp==g]=np.sqrt(((Z[lp==g]-cen)**2).sum(axis=1))
        null.append(dd[lp==1].mean()-dd[lp==0].mean())
    p=(1+np.sum(np.abs(null)>=abs(delta)))/(nperm+1)
    return {"n_low":int(np.sum(lab==0)),"n_high":int(np.sum(lab==1)),"dispersion_low":dist[lab==0].mean(),"dispersion_high":dist[lab==1].mean(),"delta_high_minus_low":delta,"permutation_p_dispersion":p}


def taxon_label(df):
    cols=[c for c in ["Scientific name","Species","Genus","Family","Order"] if c in df]
    out=pd.Series(index=df.index,dtype=object)
    for c in cols: out=out.fillna(df[c].astype(str).replace({"nan":np.nan,"None":np.nan,"":np.nan}))
    return out.fillna("未定分类单元")


def abundance_value(df,group):
    if group=="fish": return pd.Series(1.0,index=df.index)
    for c in ["Density","Raw value","Biomass"]:
        if c in df and pd.to_numeric(df[c],errors="coerce").notna().any(): return pd.to_numeric(df[c],errors="coerce").fillna(0).clip(lower=0)
    return pd.Series(1.0,index=df.index)


def build_matrix(raw,group,water,site_names):
    d=raw.copy(); wcol="Water body type"; scol="Site"
    d=d[d[wcol].astype(str).str.lower().eq(water.lower())]
    d=d[d[scol].astype(str).isin(site_names)]; d["taxon"]=taxon_label(d); d["value"]=abundance_value(d,group)
    mat=d.pivot_table(index=scol,columns="taxon",values="value",aggfunc="sum",fill_value=0)
    mat=mat.reindex(site_names,fill_value=0); keep=(mat.gt(0).sum(axis=0)>=3)&(mat.sum(axis=0)>0); mat=mat.loc[:,keep]
    rel=mat.div(mat.sum(axis=1).replace(0,np.nan),axis=0).fillna(0); hell=np.sqrt(rel)
    return mat,hell


def partial_taxon_correlations(mat,score,C):
    rows=[]
    for taxon in mat.columns:
        y=np.sqrt(mat[taxon]/mat.sum(axis=1).replace(0,np.nan)).fillna(0).to_numpy(); x=np.asarray(score,float)
        ok=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(C),axis=1)
        if ok.sum()<20:continue
        cc=np.column_stack([np.ones(ok.sum()),C[ok]]); ry=stats.rankdata(y[ok]); rx=stats.rankdata(x[ok])
        ry-=cc@np.linalg.lstsq(cc,ry,rcond=None)[0]; rx-=cc@np.linalg.lstsq(cc,rx,rcond=None)[0]
        rho=np.corrcoef(ry,rx)[0,1]; df=ok.sum()-C.shape[1]-2; p=2*stats.t.sf(abs(rho)*math.sqrt(df/max(1e-12,1-rho*rho)),df)
        rows.append({"taxon":taxon,"partial_spearman_rho":rho,"p_value":p,"n_sites":ok.sum()})
    out=pd.DataFrame(rows)
    if not out.empty:out["q_value"]=bh(out.p_value)
    return out


def community_analysis(out,root,site,scores):
    raw_path=next((root.parents[1]/"Summary").glob("Aquatic_Biological_Database.xlsx"))
    rng=np.random.default_rng(20260722); rows=[]; matrices={}
    for group,sheet in GROUP_SHEETS.items():
        raw=pd.read_excel(raw_path,sheet_name=sheet)
        for water in ["River","Lake"]:
            if group=="periphyton" and water=="Lake":continue
            d=site[site.water_body_type.eq(water)].copy(); mat,hell=build_matrix(raw,group,water,d.site.astype(str).tolist())
            if hell.shape[1]<2:continue
            matrices[(group,water)]=(mat,hell,d)
            C=controls(d)
            for module in MODULES:
                for r in RANGES:
                    sc=scores[(scores.water_body_type==water)&(scores.environment_module==module)&(scores.range_m==r)].set_index("site").environment_score.reindex(d.site).to_numpy()
                    res=rda_test(hell.to_numpy(),sc,C,rng); disp=dispersion_test(mat.to_numpy(),sc,rng)
                    if res:
                        row={"water_body_type":water,"taxon_group":group,"taxon_group_cn":GROUP_CN[group],"environment_module":module,"environment_module_cn":MODULE_CN[module],"range_m":r,**res}
                        if disp:row.update(disp)
                        rows.append(row)
    result=pd.DataFrame(rows)
    result["q_13ranges_within_group_module_rda"]=np.nan; result["q_13ranges_within_group_module_dispersion"]=np.nan
    for _,ix in result.groupby(["water_body_type","taxon_group","environment_module"]).groups.items():
        result.loc[ix,"q_13ranges_within_group_module_rda"]=bh(result.loc[ix,"permutation_p"])
        result.loc[ix,"q_13ranges_within_group_module_dispersion"]=bh(result.loc[ix,"permutation_p_dispersion"])
    # Taxa explaining the strongest community relation for each water/group/module.
    taxa=[]; ordrows=[]
    for keys,g in result.groupby(["water_body_type","taxon_group","environment_module"]):
        water,group,module=keys; peak=g.sort_values(["permutation_p","partial_rda_r2"],ascending=[True,False]).iloc[0]; r=int(peak.range_m)
        mat,hell,d=matrices[(group,water)]; C=controls(d)
        sc=scores[(scores.water_body_type==water)&(scores.environment_module==module)&(scores.range_m==r)].set_index("site").environment_score.reindex(d.site).to_numpy()
        tx=partial_taxon_correlations(mat,sc,C)
        if not tx.empty:
            tx["water_body_type"]=water;tx["taxon_group"]=group;tx["taxon_group_cn"]=GROUP_CN[group];tx["environment_module"]=module;tx["range_m"]=r;taxa.append(tx)
        D=squareform(pdist(hell.to_numpy(),metric="euclidean")); Z=pcoa_coords(D)
        if Z.shape[1]>=2:
            for i,siteid in enumerate(d.site):ordrows.append({"water_body_type":water,"taxon_group":group,"taxon_group_cn":GROUP_CN[group],"environment_module":module,"range_m":r,"site":siteid,"axis1":Z[i,0],"axis2":Z[i,1],"environment_score":sc[i]})
    taxa=pd.concat(taxa,ignore_index=True) if taxa else pd.DataFrame(); ords=pd.DataFrame(ordrows)
    return result,taxa,ords


def transform_response(x,col,fam):
    x=pd.to_numeric(x,errors="coerce").astype(float); c=col.lower()
    if "pcoa" in c or fam=="beta_diversity":return x,"原尺度"
    if any(k in c for k in ["proportion","percentage","_pct","percent"]):
        p=x/100 if x.max(skipna=True)>1.01 else x;return np.arcsin(np.sqrt(p.clip(0,1))),"arcsin(sqrt(p))"
    if x.min(skipna=True)>=0 and any(k in c for k in ["density","biomass","weight","length","ratio"]):return np.log1p(x),"log1p"
    return x,"原尺度"


def temporal_analysis(out,root,site,sample,scores,targets):
    rows=[]
    # At most two non-PCoA/non-beta targets per water×module.
    t=targets[~targets.column_name.str.contains("pcoa",case=False,na=False)&~targets.response_family.eq("beta_diversity")]
    t=t.sort_values(["screen_rank","max_abs_rho"],ascending=[True,False]).groupby(["water_body_type","environment_module"]).head(2)
    site_scores=scores.merge(site[["water_body_type","site"]],left_on=["water_body_type","row_index"],right_on=["water_body_type",site.index.name or "site"] if site.index.name else None,how="left") if False else None
    # score row_index is the original Site_Special row index.
    score2=scores.copy(); score2["site"]=[site.loc[i,"site"] for i in score2.row_index]
    for _,z in t.iterrows():
        d=sample[sample.water_body_type.eq(z.water_body_type)].copy(); r=int(z.peak_range_m)
        sc=score2[(score2.water_body_type==z.water_body_type)&(score2.environment_module==z.environment_module)&(score2.range_m==r)][["site","environment_score"]]
        d=d.merge(sc,on="site",how="left"); y,rule=transform_response(d[z.column_name],z.column_name,z.response_family); d["y"]=y
        d=d.dropna(subset=["y","environment_score","sampling_year","flow_period","site"])
        if d.site.nunique()<20 or len(d)<35:continue
        formula="y ~ environment_score + C(sampling_year) + C(flow_period) + environment_score:C(flow_period)"
        method="MixedLM随机截距"
        try:
            fit=smf.mixedlm(formula,d,groups=d["site"]).fit(reml=False,method="lbfgs",maxiter=300,disp=False)
        except Exception:
            fit=smf.ols(formula,d).fit(cov_type="cluster",cov_kwds={"groups":d["site"]});method="OLS按site聚类稳健标准误"
        for term in [x for x in fit.params.index if x=="environment_score" or "environment_score:C(flow_period)" in x]:
            rows.append({"water_body_type":z.water_body_type,"environment_module":z.environment_module,"response_family":z.response_family,"response_variable":z.column_name,"response_name_cn":z.variable_name_cn,"range_m":r,"term":term,"estimate":fit.params[term],"se":fit.bse[term],"p_value":fit.pvalues[term],"n_samples":len(d),"n_sites":d.site.nunique(),"response_transform":rule,"model":method})
    outdf=pd.DataFrame(rows)
    if not outdf.empty:outdf["q_value"]=bh(outdf.p_value)
    return outdf


def main():
    out=Path(__file__).resolve().parents[1];root=out.parent/"ML_Input_原始量_13尺度_2026-07-04"
    ml=root/"ML_Input_Raw_13Ranges_Sample_Site.xlsx";site=pd.read_excel(ml,sheet_name="Site_Special");sample=pd.read_excel(ml,sheet_name="Sample_Special")
    scores=pd.read_csv(out/"09_SourceData"/"环境模块13尺度固定指数.csv")
    # Add site label to score table for community matching.
    scores["site"]=[site.loc[int(i),"site"] for i in scores.row_index]
    result,taxa,ords=community_analysis(out,root,site,scores)
    cdir=out/"06_群落组成与Beta过程";cdir.mkdir(exist_ok=True)
    result.to_csv(cdir/"原始分类单元矩阵_Hellinger偏RDA与群落离散度.csv",index=False,encoding="utf-8-sig")
    taxa.to_csv(cdir/"峰值尺度_分类单元偏相关与FDR.csv",index=False,encoding="utf-8-sig")
    ords.to_csv(out/"09_SourceData"/"群落组成排序坐标_峰值尺度.csv",index=False,encoding="utf-8-sig")
    targets=pd.read_csv(out/"05_Q4_具体生物响应"/"空间诊断预设代表响应清单.csv")
    temp=temporal_analysis(out,root,site,sample,scores,targets)
    tdir=out/"07_River_Lake时间空间敏感性";tdir.mkdir(exist_ok=True)
    temp.to_csv(tdir/"Sample层年份水文期混合模型稳健性.csv",index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(out/"群落组成与时间稳健性汇总.xlsx",engine="openpyxl") as xw:
        result.to_excel(xw,sheet_name="群落Hellinger偏RDA",index=False);taxa.to_excel(xw,sheet_name="峰值分类单元",index=False);temp.to_excel(xw,sheet_name="时间稳健性",index=False)
    print(json.dumps({"community_tests":len(result),"taxon_associations":len(taxa),"ordination_rows":len(ords),"temporal_terms":len(temp)},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
