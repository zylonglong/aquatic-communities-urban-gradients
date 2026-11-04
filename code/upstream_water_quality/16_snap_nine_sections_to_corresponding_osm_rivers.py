# -*- coding: utf-8 -*-
"""Snap nine supplied section seeds to the nearest point on the corresponding OSM river."""
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points, unary_union

ROOT = Path(r"E:\桌面\博士课题\Data")
OUT = ROOT / r"data\processed\WQ_13尺度环境特征与水质响应分析_2026-07-20"
PROC = OUT / "01_INPUT与过程数据"
QA = OUT / "06_QA"
OSM = ROOT / r"data\raw\栅格数据集\guangdong-free-shp"
CRS_METRIC = "EPSG:32649"

SECTIONS = {
    "SZ_WQ_RIV_001": {"waterbody": "凤塘河", "section": "凤塘河河口", "lon": 114.0178, "lat": 22.5218, "aliases": ["凤塘河"], "fallback_osm_id": "231410745"},
    "SZ_WQ_RIV_003": {"waterbody": "大沙河", "section": "大冲桥", "lon": 113.9554, "lat": 22.5633, "aliases": ["大沙河"]},
    "SZ_WQ_RIV_004": {"waterbody": "布吉河", "section": "人民桥", "lon": 114.1092, "lat": 22.5468, "aliases": ["布吉河"]},
    "SZ_WQ_RIV_005": {"waterbody": "新洲河", "section": "新洲河河口", "lon": 114.0379, "lat": 22.5099, "aliases": ["新洲河"]},
    "SZ_WQ_RIV_006": {"waterbody": "深圳水库排洪河(沙湾河)", "section": "沙湾河河口", "lon": 114.1370, "lat": 22.5513, "aliases": ["沙湾河", "深圳水库排洪河"]},
    "SZ_WQ_RIV_009": {"waterbody": "王母河", "section": "王母河河口／河口水闸", "lon": 114.4746, "lat": 22.5923, "aliases": ["王母河"], "fallback_osm_id": "314862312"},
    "SZ_WQ_RIV_010": {"waterbody": "盐田河", "section": "盐港中学", "lon": 114.2625, "lat": 22.5907, "aliases": ["盐田河"]},
    "SZ_WQ_RIV_011": {"waterbody": "福田河", "section": "福田河河口", "lon": 114.0756, "lat": 22.5181, "aliases": ["福田河"]},
    "SZ_WQ_RIV_016": {"waterbody": "西乡河", "section": "新水闸／西乡河水闸", "lon": 113.8566, "lat": 22.5865, "aliases": ["西乡河"]},
}


def main():
    reg_path = PROC / "WQ_Station_Spatial_Registry.csv"
    reg = pd.read_csv(reg_path, encoding="utf-8-sig")
    if len(reg) != 103 or reg.station_id.nunique() != 103:
        raise ValueError("The spatial registry must contain 103 unique station units")

    rivers = gpd.read_file(
        OSM / "gis_osm_waterways_free_1.shp", bbox=(113.65, 22.35, 115.25, 22.95)
    )
    rivers = rivers[rivers.geometry.notna() & ~rivers.geometry.is_empty].copy().to_crs(CRS_METRIC)
    rivers["name_text"] = rivers.name.fillna("").astype(str).str.strip()
    rivers["osm_id_text"] = rivers.osm_id.astype(str)

    audit = []
    for station_id, cfg in SECTIONS.items():
        idx = reg.index[reg.station_id.eq(station_id)]
        if len(idx) != 1:
            raise ValueError(f"Expected exactly one registry row for {station_id}")
        i = idx[0]
        p = gpd.GeoSeries([Point(cfg["lon"], cfg["lat"])], crs=4326).to_crs(CRS_METRIC).iloc[0]

        named = rivers[rivers.name_text.apply(lambda x: any(a in x for a in cfg["aliases"]))].copy()
        if not named.empty:
            candidates = named
            match_rule = "nearest_point_on_matching_named_osm_river"
        else:
            fallback = cfg.get("fallback_osm_id")
            candidates = rivers[rivers.osm_id_text.eq(str(fallback))].copy() if fallback else rivers.iloc[0:0].copy()
            if candidates.empty:
                local_idx = list(rivers.sindex.query(p.buffer(3000), predicate="intersects"))
                candidates = rivers.iloc[local_idx].copy()
                candidates = candidates[candidates.fclass.eq("river")]
                candidates["_d"] = candidates.geometry.distance(p)
                candidates = candidates.sort_values("_d").head(1)
            match_rule = "nearest_point_on_nearest_corresponding_unnamed_osm_river"
        if candidates.empty:
            raise ValueError(f"No corresponding OSM river candidate for {station_id}")

        line = unary_union(candidates.geometry.tolist())
        q = nearest_points(p, line)[1]
        displacement = float(p.distance(q))
        q_wgs = gpd.GeoSeries([q], crs=CRS_METRIC).to_crs(4326).iloc[0]
        candidates["_distance"] = candidates.geometry.distance(p)
        nearest_row = candidates.sort_values("_distance").iloc[0]
        geometry_id = f"RIV_GEOM_{round(float(q_wgs.x), 6)}_{round(float(q_wgs.y), 6)}"

        old_lon = float(reg.at[i, "spatial_longitude"])
        old_lat = float(reg.at[i, "spatial_latitude"])
        reg.at[i, "station_name_canonical"] = cfg["section"]
        reg.at[i, "waterbody_name_spatial"] = cfg["waterbody"]
        reg.at[i, "spatial_longitude"] = float(q_wgs.x)
        reg.at[i, "spatial_latitude"] = float(q_wgs.y)
        reg.at[i, "geometry_id"] = geometry_id
        reg.at[i, "spatial_id"] = station_id
        reg.at[i, "osm_id"] = str(nearest_row.osm_id)
        reg.at[i, "osm_name"] = str(nearest_row.name_text or cfg["waterbody"])
        reg.at[i, "osm_match_rule"] = match_rule
        reg.at[i, "geometry_rule"] = "named_section_snapped_to_corresponding_osm_river"
        reg.at[i, "river_snap_distance_m"] = displacement
        reg.at[i, "provided_section_name"] = cfg["section"]
        reg.at[i, "provided_section_longitude"] = cfg["lon"]
        reg.at[i, "provided_section_latitude"] = cfg["lat"]
        reg.at[i, "spatial_coordinate_method"] = "具名断面种子坐标投影至对应OSM河流的最近点"
        reg.at[i, "spatial_coordinate_source"] = "具名监测断面坐标 + Geofabrik OSM对应河流"
        reg.at[i, "spatial_coordinate_source_url"] = "https://download.geofabrik.de/asia/china.html"
        reg.at[i, "spatial_quality"] = "高（具名断面已投影至对应OSM河流）" if displacement <= 500 else "中（具名断面已投影至对应OSM河流；原坐标与OSM偏移较大）"
        reg.at[i, "coordinate_uncertainty_m_revised"] = max(50.0, displacement)
        reg.at[i, "spatial_notes"] = "最终分析坐标位于对应OSM河流线上；原具名断面坐标保留于provided_section字段。"
        reg.at[i, "spatial_eligible"] = True
        audit.append({
            "station_id": station_id, "waterbody": cfg["waterbody"], "section_name": cfg["section"],
            "seed_lon": cfg["lon"], "seed_lat": cfg["lat"],
            "previous_lon": old_lon, "previous_lat": old_lat,
            "final_lon": float(q_wgs.x), "final_lat": float(q_wgs.y),
            "snap_displacement_m": displacement, "osm_id": str(nearest_row.osm_id),
            "osm_name": str(nearest_row.name_text), "osm_fclass": str(nearest_row.fclass),
            "match_rule": match_rule, "geometry_id": geometry_id,
        })

    if reg.geometry_id.nunique() != 56:
        raise ValueError(f"Expected 56 independent geometries, got {reg.geometry_id.nunique()}")
    reg.to_csv(reg_path, index=False, encoding="utf-8-sig")

    points = gpd.GeoDataFrame(
        reg.copy(),
        geometry=gpd.points_from_xy(reg.spatial_longitude, reg.spatial_latitude),
        crs=4326,
    )
    gpkg = PROC / "WQ_103_Station_Repaired_Spatial_Registry.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    points.to_file(gpkg, layer="station_registry", driver="GPKG")

    qa_path = QA / "103站点坐标与水体边界修缮QA.csv"
    qa = pd.read_csv(qa_path, encoding="utf-8-sig")
    for a in audit:
        mask = qa.station_id.eq(a["station_id"])
        qa.loc[mask, "station_name"] = a["section_name"]
        qa.loc[mask, "waterbody_name_spatial"] = a["waterbody"]
        qa.loc[mask, "longitude_wgs84"] = a["final_lon"]
        qa.loc[mask, "latitude_wgs84"] = a["final_lat"]
        qa.loc[mask, "on_assigned_osm_water_geometry"] = True
        qa.loc[mask, "distance_to_assigned_geometry_m"] = 0.0
        qa.loc[mask, "spatial_quality"] = reg.loc[reg.station_id.eq(a["station_id"]), "spatial_quality"].iloc[0]
        qa.loc[mask, "geometry_id"] = a["geometry_id"]
    qa.to_csv(qa_path, index=False, encoding="utf-8-sig")

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(QA / "9个River断面对应OSM河流投影QA.csv", index=False, encoding="utf-8-sig")
    summary = {
        "updated_sections": len(audit_df),
        "all_final_points_on_assigned_osm_river": True,
        "min_snap_displacement_m": float(audit_df.snap_displacement_m.min()),
        "max_snap_displacement_m": float(audit_df.snap_displacement_m.max()),
        "independent_geometries": int(reg.geometry_id.nunique()),
    }
    (QA / "9个River断面OSM河流投影摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_df.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
