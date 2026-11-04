# -*- coding: utf-8 -*-
from pathlib import Path
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT=Path(r"E:\桌面\博士课题\Data")
OUT=ROOT/r"data\processed\公共水质数据_深圳及周边_2026-07-19\WQ_13尺度环境特征与水质响应分析_2026-07-20"
PROC,QA=OUT/"01_INPUT与过程数据",OUT/"06_QA"
OSM=ROOT/r"data\raw\栅格数据集\guangdong-free-shp"
reg=pd.read_csv(PROC/"WQ_Station_Spatial_Registry.csv")
water=gpd.read_file(OSM/"gis_osm_water_a_free_1.shp").to_crs(32649)
river=gpd.read_file(OSM/"gis_osm_waterways_free_1.shp").to_crs(32649)
water=water[water.geometry.notna()&~water.geometry.is_empty].copy()
river=river[river.geometry.notna()&~river.geometry.is_empty].copy()

def norm(x):
    x=str(x or "").strip().lower()
    x=re.sub(r"[（）()·\-—_、，,\s]", "", x)
    x=re.sub(r"(饮用水源地?|饮用|水源地?|河流评价单元|评价单元|集中式|中型|小型)$", "", x)
    return x

water["name_norm"]=water.name.fillna("").map(norm)
river["name_norm"]=river.name.fillna("").map(norm)

def candidate_table(layer,target,pt=None,k=5):
    tn=norm(target); d=layer[layer.name.notna()].copy()
    if tn:
        d["name_score"]=d.name_norm.map(lambda x: SequenceMatcher(None,tn,x).ratio() if x else 0)
        exact=d[(d.name_norm==tn)|(d.name_norm.map(lambda x: bool(x) and (x in tn or tn in x)))]
    else:
        d["name_score"]=0.; exact=d.iloc[0:0]
    if pt is not None:
        # Restrict distance work to a 30-km envelope around the provisional point.
        near=d.iloc[list(d.sindex.query(pt.buffer(30000),predicate="intersects"))].copy()
        near["distance_m"]=near.geometry.distance(pt)
        pool=pd.concat([exact,near.nlargest(20,"name_score"),near.nsmallest(20,"distance_m")]).drop_duplicates("osm_id")
        pool["distance_m"]=pool.geometry.distance(pt)
        pool["combined"]=pool.name_score-0.15*np.minimum(pool.distance_m/5000,1)
    else:
        pool=pd.concat([exact,d.nlargest(50,"name_score")]).drop_duplicates("osm_id")
        pool["distance_m"]=np.nan; pool["combined"]=pool.name_score
    return pool.sort_values(["combined","name_score"],ascending=False).head(k)

rows=[]; cand_rows=[]
for _,r in reg.iterrows():
    lon=pd.to_numeric(pd.Series([r.get("spatial_longitude")]),errors="coerce").iloc[0]
    lat=pd.to_numeric(pd.Series([r.get("spatial_latitude")]),errors="coerce").iloc[0]
    if pd.isna(lon): lon=pd.to_numeric(pd.Series([r.get("longitude_wgs84")]),errors="coerce").iloc[0]
    if pd.isna(lat): lat=pd.to_numeric(pd.Series([r.get("latitude_wgs84")]),errors="coerce").iloc[0]
    pt=None if pd.isna(lon) or pd.isna(lat) else gpd.GeoSeries([Point(lon,lat)],crs=4326).to_crs(32649).iloc[0]
    is_lake=str(r.get("waterbody_type")) in ["水库","湖泊","Lake","Reservoir"] or str(r.get("spatial_waterbody_type"))=="Lake"
    layer=water if is_lake else river
    target=r.get("waterbody_name_clean") or r.get("waterbody_name") or r.get("station_name_canonical")
    cand=candidate_table(layer,target,pt,5)
    best=cand.iloc[0] if len(cand) else None
    on_geom=False; nearest=np.nan
    if pt is not None:
        if is_lake:
            ids=list(water.sindex.query(pt,predicate="intersects")); on_geom=len(ids)>0
            if ids: nearest=0.
            else:
                near_idx=water.sindex.nearest(pt,return_all=False)
                ids2=list(np.asarray(near_idx)[1]); nearest=float(water.iloc[ids2].geometry.distance(pt).min()) if ids2 else np.nan
        else:
            near_idx=river.sindex.nearest(pt,return_all=False)
            ids2=list(np.asarray(near_idx)[1]); nearest=float(river.iloc[ids2].geometry.distance(pt).min()) if ids2 else np.nan
            on_geom=bool(pd.notna(nearest) and nearest<=30)
    rows.append({
        "station_id":r.station_id,"station_name":r.station_name_canonical,"declared_waterbody":r.waterbody_name,
        "declared_type":r.waterbody_type,"current_lon":lon,"current_lat":lat,"current_spatial_eligible":r.spatial_eligible,
        "on_osm_target_geometry":on_geom,"nearest_osm_water_or_channel_m":nearest,
        "best_candidate_osm_id":None if best is None else best.osm_id,"best_candidate_name":None if best is None else best["name"],
        "best_name_score":None if best is None else best.name_score,"best_candidate_distance_m":None if best is None else best.distance_m,
        "needs_manual_review":pt is None or (is_lake and not on_geom) or ((not is_lake) and (pd.isna(nearest) or nearest>100))
    })
    for rank,(_,c) in enumerate(cand.iterrows(),1):
        cand_rows.append({"station_id":r.station_id,"rank":rank,"target_name":target,"candidate_osm_id":c.osm_id,
                          "candidate_name":c["name"],"fclass":c.fclass,"name_score":c.name_score,"distance_m":c.distance_m})

audit=pd.DataFrame(rows)
audit.to_csv(QA/"103站点OSM几何审计.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(cand_rows).to_csv(QA/"103站点OSM候选水体.csv",index=False,encoding="utf-8-sig")
print(audit.needs_manual_review.value_counts(dropna=False).to_string())
print(audit[audit.needs_manual_review][["station_id","station_name","declared_waterbody","declared_type","current_lon","current_lat","nearest_osm_water_or_channel_m","best_candidate_name","best_name_score","best_candidate_distance_m"]].to_string(index=False))
