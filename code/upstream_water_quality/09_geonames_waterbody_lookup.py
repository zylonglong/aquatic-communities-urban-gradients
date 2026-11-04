# -*- coding: utf-8 -*-
from pathlib import Path
import re
import pandas as pd

ROOT=Path(r"E:\桌面\博士课题\Data")
OUT=ROOT/r"data\processed\公共水质数据_深圳及周边_2026-07-19\WQ_13尺度环境特征与水质响应分析_2026-07-20"
PROC,QA=OUT/"01_INPUT与过程数据",OUT/"06_QA"
GN=ROOT/r"data\processed\公共水质数据_深圳及周边_2026-07-19\01_原始下载\GeoNames\CN.txt"
reg=pd.read_csv(PROC/"WQ_Station_Spatial_Registry.csv")

def clean(x):
    x=str(x or "").strip()
    x=re.sub(r"^东深供水-", "", x)
    x=re.sub(r"（.*?）|\(.*?\)", "", x)
    x=re.sub(r"饮用水源地?|饮用水|饮用$", "", x)
    return x.strip()

targets=sorted(set(reg.waterbody_name.dropna().map(clean)))
cols=["geonameid","name","asciiname","alternatenames","latitude","longitude","feature_class","feature_code","country_code","cc2","admin1_code","admin2_code","admin3_code","admin4_code","population","elevation","dem","timezone","modification_date"]
hits=[]
for chunk in pd.read_csv(GN,sep="\t",names=cols,header=None,quoting=3,dtype=str,chunksize=150000,encoding="utf-8"):
    lat=pd.to_numeric(chunk.latitude,errors="coerce"); lon=pd.to_numeric(chunk.longitude,errors="coerce")
    c=chunk[lon.between(112.7,115.7)&lat.between(22.1,23.5)].copy()
    if c.empty: continue
    c["all_names"]=(c.name.fillna("")+","+c.asciiname.fillna("")+","+c.alternatenames.fillna(""))
    for t in targets:
        m=c.all_names.str.contains(re.escape(t),case=False,regex=True,na=False)
        if m.any():
            z=c.loc[m].copy(); z["target_name"]=t; hits.append(z)
res=pd.concat(hits,ignore_index=True).drop_duplicates(["target_name","geonameid"]) if hits else pd.DataFrame()
if len(res):
    res["is_hydrographic"]=res.feature_class.eq("H")
    res=res.sort_values(["target_name","is_hydrographic"],ascending=[True,False])
res.to_csv(QA/"GeoNames水体候选坐标.csv",index=False,encoding="utf-8-sig")
print(res[["target_name","name","alternatenames","latitude","longitude","feature_class","feature_code","admin1_code","admin2_code"]].to_string(index=False) if len(res) else "NO HITS")
