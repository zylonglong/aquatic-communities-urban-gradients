# -*- coding: utf-8 -*-
"""Inspect the nearest OSM waterway candidates for the nine supplied sections."""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points, unary_union

ROOT = Path(r"E:\桌面\博士课题\Data")
OSM = ROOT / r"data\raw\栅格数据集\guangdong-free-shp"
OUT = ROOT / r"data\processed\WQ_13尺度环境特征与水质响应分析_2026-07-20"
QA = OUT / "06_QA"
CRS_METRIC = "EPSG:32649"

SECTIONS = {
    "SZ_WQ_RIV_001": ("凤塘河", "凤塘河河口", 114.0178, 22.5218, ["凤塘河"]),
    "SZ_WQ_RIV_003": ("大沙河", "大冲桥", 113.9554, 22.5633, ["大沙河"]),
    "SZ_WQ_RIV_004": ("布吉河", "人民桥", 114.1092, 22.5468, ["布吉河"]),
    "SZ_WQ_RIV_005": ("新洲河", "新洲河河口", 114.0379, 22.5099, ["新洲河"]),
    "SZ_WQ_RIV_006": ("深圳水库排洪河(沙湾河)", "沙湾河河口", 114.1370, 22.5513, ["沙湾河", "深圳水库排洪河"]),
    "SZ_WQ_RIV_009": ("王母河", "王母河河口／河口水闸", 114.4746, 22.5923, ["王母河"]),
    "SZ_WQ_RIV_010": ("盐田河", "盐港中学", 114.2625, 22.5907, ["盐田河"]),
    "SZ_WQ_RIV_011": ("福田河", "福田河河口", 114.0756, 22.5181, ["福田河"]),
    "SZ_WQ_RIV_016": ("西乡河", "新水闸／西乡河水闸", 113.8566, 22.5865, ["西乡河"]),
}


def main():
    rivers = gpd.read_file(
        OSM / "gis_osm_waterways_free_1.shp", bbox=(113.65, 22.35, 115.25, 22.95)
    )
    rivers = rivers[rivers.geometry.notna() & ~rivers.geometry.is_empty].copy().to_crs(CRS_METRIC)
    rivers["name_text"] = rivers.name.fillna("").astype(str).str.strip()
    rows = []
    summary = []
    for station_id, (waterbody, section, lon, lat, aliases) in SECTIONS.items():
        p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(CRS_METRIC).iloc[0]
        idx = list(rivers.sindex.query(p.buffer(3000), predicate="intersects"))
        local = rivers.iloc[idx].copy()
        local["distance_m"] = local.geometry.distance(p)
        local = local.sort_values("distance_m").head(20)
        for rank, (_, r) in enumerate(local.iterrows(), 1):
            q = nearest_points(p, r.geometry)[1]
            qw = gpd.GeoSeries([q], crs=CRS_METRIC).to_crs(4326).iloc[0]
            rows.append({
                "station_id": station_id, "waterbody": waterbody, "section_name": section,
                "seed_lon": lon, "seed_lat": lat, "candidate_rank": rank,
                "osm_id": str(r.get("osm_id", "")), "osm_name": r.get("name_text", ""),
                "fclass": r.get("fclass", ""), "distance_m": float(r.distance_m),
                "nearest_lon": float(qw.x), "nearest_lat": float(qw.y),
            })
        named = rivers[rivers.name_text.apply(lambda x: any(a in x for a in aliases))]
        if named.empty:
            summary.append({
                "station_id": station_id, "waterbody": waterbody, "section_name": section,
                "seed_lon": lon, "seed_lat": lat, "named_osm_features": 0,
                "named_distance_m": None, "named_nearest_lon": None, "named_nearest_lat": None,
                "named_osm_ids": None, "named_osm_names": None,
            })
        else:
            line = unary_union(named.geometry.tolist())
            q = nearest_points(p, line)[1]
            qw = gpd.GeoSeries([q], crs=CRS_METRIC).to_crs(4326).iloc[0]
            summary.append({
                "station_id": station_id, "waterbody": waterbody, "section_name": section,
                "seed_lon": lon, "seed_lat": lat, "named_osm_features": len(named),
                "named_distance_m": float(p.distance(q)),
                "named_nearest_lon": float(qw.x), "named_nearest_lat": float(qw.y),
                "named_osm_ids": ";".join(named.osm_id.astype(str)),
                "named_osm_names": ";".join(sorted(named.name_text.unique())),
            })
    QA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(QA / "9个River断面OSM候选水道诊断.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(summary)
    result.to_csv(QA / "9个River断面OSM同名河流最近点诊断.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
