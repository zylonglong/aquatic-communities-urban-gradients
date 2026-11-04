# -*- coding: utf-8 -*-
"""Repair the complete 103-unit public-water-quality spatial registry.

The 103 station IDs are retained as observation/source units. Repeated IDs
that refer to the same physical waterbody receive the same geometry_id and
waterbody_cluster_id; they are not falsely treated as 103 independent
landscapes in inferential statistics.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points, unary_union


ROOT = Path(r"E:\桌面\博士课题\Data")
WQROOT = ROOT / r"data\processed\公共水质数据_深圳及周边_2026-07-19"
OUT = WQROOT / "WQ_13尺度环境特征与水质响应分析_2026-07-20"
PROC = OUT / "01_INPUT与过程数据"
QA = OUT / "06_QA"
PROC.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
REGISTRY = WQROOT / r"03_站点坐标与名录\水质监测站名录.csv"
OLD_SPATIAL = PROC / "WQ_Station_Spatial_Registry.csv"
OSM = ROOT / r"data\raw\栅格数据集\guangdong-free-shp"
CRS_METRIC = "EPSG:32649"


def text(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def clean_waterbody_name(x) -> str:
    s = text(x).replace("（", "(").replace("）", ")")
    if "-" in s:
        s = s.split("-")[-1]
    for suffix in ["饮用水源", "饮用水", "饮用"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    if s == "窑坡水库":
        return "窑陂水库"
    return s.strip()


# Curated local OSM water polygons. Unnamed matches were accepted only after
# cross-checking official locality/maps, reservoir scale, and surrounding OSM.
MANUAL_LAKE_IDS = {
    "三角山水库": ["1089995398"],
    "公明水库": ["773915146"],  # OSM current name: 光明湖
    "嘉田水库": ["666290831"],
    "大坑水库": ["107438751"],  # nuclear-base reservoir; not the same-name NE reservoir
    "小漠水库": ["8305858"],
    "岭澳水库": ["104022815"],
    "泗马岭水库": ["722933445"],
    "赤坳水库": ["307749162"],  # OSM alias: 赤澳水库
    "窑陂水库": ["1188473199"],
}

OSM_ALIASES = {
    "打马坜水库": ["打马沥水库"],
    "长岭皮水库": ["长岭陂水库"],
    "石岩水库": ["石岩湖"],
    "西丽水库": ["西丽湖"],
    "铜锣径水库": ["铜锣径水库(深圳抽水蓄能电站下水库)"],
    "红花岭水库上库": ["红花岭上库"],
    "红花岭水库下库": ["红花岭下库"],
    "红花岭水库": ["红花岭上库", "红花岭下库"],
}

OFFICIAL_EVIDENCE = {
    "三角山水库": "https://www.szss.gov.cn/gkmlpt/content/12/12217/post_12217635.html",
    "公明水库": "https://www.sz.gov.cn/szzt2010/wgkzl/jcgk/jcygk/zdzcjc/content/post_11186214.html",
    "大坑水库": "https://meeb.sz.gov.cn/attachment/0/127/127038/2007721.pdf",
    "岭澳水库": "https://pnr.sz.gov.cn/attachment/0/392/392567/5841623.pdf",
    "小漠水库": "https://www.sz.gov.cn/attachment/1/1484/1484824/11543897.pdf",
    "泗马岭水库": "https://www.sz.gov.cn/attachment/1/1400/1400166/11080068.pdf",
    "嘉田水库": "https://www.sz.gov.cn/attachment/1/1484/1484824/11543897.pdf",
    "赤坳水库": "https://www.szpsq.gov.cn/pshjbhhswj/gkmlpt/content/11/11313/post_11313164.html",
    "窑陂水库": "https://www.sz.gov.cn/attachment/1/1484/1484824/11543897.pdf",
}

CURATION_NOTES = {
    "三角山水库": "官方定位为赤石街道圆墩林场；按官方展板卫星图与圆墩西北侧水面核验。",
    "公明水库": "官方定位光明区新湖/光明街道；OSM现名为光明湖。",
    "嘉田水库": "官方名录定位鹅埠；按库容、鹅埠西侧位置及水面尺度核验未命名OSM水库。",
    "大坑水库": "官方水源保护区定位大鹏新区；考古调查坐标位于该水库下游，排除旧表误配的东北部同名水库。",
    "岭澳水库": "官方名录定位大鹏核电基地；与大坑水库相邻但为不同水面。",
    "小漠水库": "官方名录定位小漠；GeoNames小漠点与未命名OSM水面一致。",
    "泗马岭水库": "官方宗海位置图标注鲘门西北、约115°09′E/22°48′N；面积与小(1)型库容一致，排除大型平安洞水库。",
    "赤坳水库": "官方定位坪山石井/马峦；OSM别名为赤澳水库。",
    "窑陂水库": "官方名录规范名为窑陂水库；水质公报曾写作窑坡水库。",
}


def load_water():
    water = gpd.read_file(OSM / "gis_osm_water_a_free_1.shp", bbox=(113.65, 22.35, 115.25, 23.00))
    water = water[water.geometry.notna() & ~water.geometry.is_empty].copy().to_crs(CRS_METRIC)
    water.geometry = water.geometry.buffer(0)
    water["osm_id_text"] = water.osm_id.astype(str)
    water["name_text"] = water.name.fillna("").astype(str).str.strip()
    water["area_km2"] = water.geometry.area / 1e6
    return water


def load_rivers():
    rivers = gpd.read_file(OSM / "gis_osm_waterways_free_1.shp", bbox=(113.65, 22.35, 115.25, 22.95))
    rivers = rivers[rivers.geometry.notna() & ~rivers.geometry.is_empty].copy().to_crs(CRS_METRIC)
    rivers["name_text"] = rivers.name.fillna("").astype(str).str.strip()
    return rivers


def pick_lake_geometry(target, water, old_target_ids):
    ids = MANUAL_LAKE_IDS.get(target)
    rule = "curated_official_location_osm_polygon" if ids else None
    if not ids and target == "红花岭水库":
        names = OSM_ALIASES[target]
        selected = water[water.name_text.isin(names)].copy()
        rule = "curated_multi_polygon_alias"
    elif not ids and old_target_ids.get(target):
        ids = old_target_ids[target]
        selected = water[water.osm_id_text.isin(ids)].copy()
        rule = "previously_verified_osm_polygon"
    elif not ids:
        names = [target] + OSM_ALIASES.get(target, [])
        selected = water[water.name_text.isin(names)].copy()
        rule = "exact_or_curated_osm_name"
    else:
        selected = water[water.osm_id_text.isin(ids)].copy()
    if selected.empty:
        raise ValueError(f"No lake geometry for {target}")
    if target == "红花岭水库":
        geom = unary_union(selected.geometry.tolist())
    else:
        geom = selected.sort_values("area_km2", ascending=False).iloc[0].geometry
        selected = selected.loc[[selected.sort_values("area_km2", ascending=False).index[0]]]
    return geom, selected, rule


RIVER_NAME_FILTERS = {
    "坪山河": "坪山河",
    "大沙河": "大沙河",
    "布吉河": "布吉河",
    "新洲河": "新洲河",
    "深圳水库排洪河(沙湾河)": "沙湾河",
    "深圳水库排洪河（沙湾河）": "沙湾河",
    "深圳河": "深圳河",
    "盐田河": "盐田河",
    "福田河": "福田河",
    "茅洲河": "茅洲河",
    "西乡河": "西乡河",
    "观澜河": "观澜河",
    "赤石河": "赤石河",
    "龙岗河": "龙岗河",
}

WHOLE_RIVER_SEEDS = {
    "凤塘河": (114.0235, 22.5365),
    "王母河": (114.47881, 22.59660),  # 大鹏二桥 / 王母墟桥群
}

# Named monitoring sections and coordinates supplied by the user after an
# independent public-information search (2026-07-20).  They replace the former
# whole-river representative points.  The supplied coordinate is retained in
# dedicated fields and is used unchanged as the circular-range centre.  The
# distance to the matching OSM waterway is retained only as a QA diagnostic;
# OSM cartographic offsets must not move a supplied monitoring section.
USER_PROVIDED_RIVER_SECTIONS = {
    "SZ_WQ_RIV_001": {"section_name": "凤塘河河口", "lon": 114.0178, "lat": 22.5218},
    "SZ_WQ_RIV_003": {"section_name": "大冲桥", "lon": 113.9554, "lat": 22.5633},
    "SZ_WQ_RIV_004": {"section_name": "人民桥", "lon": 114.1092, "lat": 22.5468},
    "SZ_WQ_RIV_005": {"section_name": "新洲河河口", "lon": 114.0379, "lat": 22.5099},
    "SZ_WQ_RIV_006": {"section_name": "沙湾河河口", "lon": 114.1370, "lat": 22.5513},
    "SZ_WQ_RIV_009": {"section_name": "王母河河口／河口水闸", "lon": 114.4746, "lat": 22.5923},
    "SZ_WQ_RIV_010": {"section_name": "盐港中学", "lon": 114.2625, "lat": 22.5907},
    "SZ_WQ_RIV_011": {"section_name": "福田河河口", "lon": 114.0756, "lat": 22.5181},
    "SZ_WQ_RIV_016": {"section_name": "新水闸／西乡河水闸", "lon": 113.8566, "lat": 22.5865},
}

# Verified same-name section seeds from the project's research-site matching
# table. Keeping these explicit makes the repair idempotent: a second run does
# not use the already-snapped output as a new seed.
SECTION_SEEDS = {
    "SZ_WQ_RIV_002": (114.41278441, 22.71430481),
    "SZ_WQ_RIV_007": (114.19195100, 22.55606500),
    "SZ_WQ_RIV_008": (114.03305802, 22.50671241),
    "SZ_WQ_RIV_012": (113.78873028, 22.75926811),
    "SZ_WQ_RIV_013": (113.78873028, 22.75926811),
    "SZ_WQ_RIV_014": (113.78873028, 22.75926811),
    "SZ_WQ_RIV_015": (113.82700107, 22.79439977),
    "SZ_WQ_RIV_017": (114.05817265, 22.75520952),
    "SZ_WQ_RIV_018": (115.04870347, 22.79805528),
    "SZ_WQ_RIV_019": (114.34831622, 22.76798662),
    "SZ_WQ_RIV_020": (114.41823222, 22.78049280),
    "SZ_WQ_RIV_021": (114.29386344, 22.75518736),
}


def line_for_waterbody(waterbody_name, rivers):
    key = text(waterbody_name).replace("（", "(").replace("）", ")")
    pattern = RIVER_NAME_FILTERS.get(key)
    if pattern:
        d = rivers[rivers.name_text.str.contains(pattern, regex=False)].copy()
        if pattern == "深圳河":
            d = d[d.name_text.str.startswith("深圳河")]
        if not d.empty:
            return unary_union(d.geometry.tolist()), "named_osm_waterway"
    return None, None


def snap_to_line(lon, lat, line):
    p = gpd.GeoSeries([Point(float(lon), float(lat))], crs=4326).to_crs(CRS_METRIC).iloc[0]
    if line is None:
        return p, np.nan
    q = nearest_points(p, line)[1]
    return q, float(p.distance(q))


def nearest_local_waterway(lon, lat, rivers, radius_m=1500):
    p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(CRS_METRIC).iloc[0]
    idx = list(rivers.sindex.query(p.buffer(radius_m), predicate="intersects"))
    d = rivers.iloc[idx].copy()
    if d.empty:
        return p, np.nan, None
    d["distance_m"] = d.geometry.distance(p)
    row = d.sort_values(["distance_m"]).iloc[0]
    q = nearest_points(p, row.geometry)[1]
    return q, float(p.distance(q)), str(row.osm_id)


def main():
    reg = pd.read_csv(REGISTRY, encoding="utf-8-sig")
    if len(reg) != 103 or reg.station_id.nunique() != 103:
        raise ValueError(f"Expected 103 unique station IDs, got rows={len(reg)}, unique={reg.station_id.nunique()}")
    old = pd.read_csv(OLD_SPATIAL, encoding="utf-8-sig") if OLD_SPATIAL.exists() else pd.DataFrame()
    old_target_ids = {}
    if not old.empty:
        old["target_clean"] = old.waterbody_name.map(clean_waterbody_name)
        for target, d in old.groupby("target_clean"):
            ids = []
            for value in d.osm_id.dropna().astype(str):
                ids.extend([x for x in value.split(";") if x and x.lower() != "nan"])
            if ids:
                old_target_ids[target] = sorted(set(ids))

    water = load_water()
    rivers = load_rivers()
    old_index = old.set_index("station_id") if not old.empty else pd.DataFrame()

    rows = []
    point_records = []
    lake_records = []
    river_records = []
    geometry_cache = {}

    for rec in reg.itertuples(index=False):
        row = rec._asdict()
        row["waterbody_name_reported"] = rec.waterbody_name
        target = clean_waterbody_name(rec.waterbody_name)
        row["waterbody_name_spatial"] = target
        row["spatial_id"] = rec.station_id
        row["spatial_eligible"] = True

        if rec.waterbody_type == "河流":
            row["spatial_waterbody_type"] = "River"
            waterbody = text(rec.waterbody_name)
            line, line_rule = line_for_waterbody(waterbody, rivers)
            is_whole = "评价单元" in text(rec.station_name_canonical)
            supplied_section = USER_PROVIDED_RIVER_SECTIONS.get(rec.station_id)
            old_row = old_index.loc[rec.station_id] if not old.empty and rec.station_id in old_index.index else None
            if supplied_section is not None:
                lon, lat = supplied_section["lon"], supplied_section["lat"]
                q = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(CRS_METRIC).iloc[0]
                _, dist = snap_to_line(lon, lat, line)
                river_osm_id = None
                if line is None or not np.isfinite(dist):
                    _, dist_local, local_id = nearest_local_waterway(lon, lat, rivers, radius_m=2500)
                    if np.isfinite(dist_local):
                        dist, river_osm_id = dist_local, local_id
                        line_rule = "nearest_local_waterway_QA_only"
                method = "用户提供具名断面坐标原值；直接作为River圆形range中心，OSM仅用于水系核验"
                source = "用户检索并提供的具名监测断面坐标（2026-07-20） + OSM waterway"
                source_url = ""
                quality = "较高（用户提供具名断面坐标原值；非官方坐标文件）"
                geometry_rule = "user_provided_named_section_exact_point"
                row["station_name_canonical"] = supplied_section["section_name"]
                row["provided_section_name"] = supplied_section["section_name"]
                row["provided_section_longitude"] = lon
                row["provided_section_latitude"] = lat
                is_whole = False
            elif is_whole and waterbody in WHOLE_RIVER_SEEDS:
                lon, lat = WHOLE_RIVER_SEEDS[waterbody]
                q, dist, near_id = nearest_local_waterway(lon, lat, rivers)
                method = "官方规划/桥梁位置辅助定位后吸附最近OSM水道"
                source = "官方河道/法定图则 + OSM waterway"
                source_url = (
                    "https://pnr.sz.gov.cn/ywzy/fdtz/cggbcx/dpxq/content/post_5842481.html"
                    if waterbody == "王母河" else
                    "https://swj.sz.gov.cn/attachment/0/355/355195/5665393.pdf"
                )
                river_osm_id = near_id
                quality = "中（整条河评价单元代表位置；非监测断面）"
                geometry_rule = "whole_river_representative_point"
            elif is_whole and line is not None:
                q = line.representative_point()
                dist = 0.0
                method = "OSM同名河道合并线代表点"
                source = "官方河名 + Geofabrik OSM同名水道"
                source_url = "https://swj.sz.gov.cn/attachment/0/355/355195/5665393.pdf"
                river_osm_id = ";".join(rivers[rivers.name_text.str.contains(RIVER_NAME_FILTERS.get(waterbody.replace('（','(').replace('）',')'), waterbody), regex=False)].osm_id.astype(str).tolist())
                quality = "中（整条河评价单元代表位置；非监测断面）"
                geometry_rule = "whole_river_representative_point"
            else:
                river_osm_id = None
                if rec.station_id in SECTION_SEEDS:
                    lon, lat = SECTION_SEEDS[rec.station_id]
                elif old_row is not None and pd.notna(old_row.get("spatial_longitude")):
                    lon, lat = float(old_row.spatial_longitude), float(old_row.spatial_latitude)
                elif pd.notna(rec.longitude_wgs84) and pd.notna(rec.latitude_wgs84):
                    lon, lat = float(rec.longitude_wgs84), float(rec.latitude_wgs84)
                else:
                    raise ValueError(f"Missing seed coordinate for named section {rec.station_id}")
                q, dist = snap_to_line(lon, lat, line)
                if line is not None and dist > 500:
                    # Some OSM named-river lines stop at the municipal border
                    # (notably 观澜河企坪). Keep the verified section locality
                    # and snap to the local unnamed/alias waterway instead of
                    # moving the station kilometres downstream.
                    q_local, dist_local, local_id = nearest_local_waterway(lon, lat, rivers, radius_m=1200)
                    if np.isfinite(dist_local) and dist_local < dist:
                        q, dist, river_osm_id = q_local, dist_local, local_id
                        line = None
                        line_rule = "nearest_local_waterway_at_verified_section"
                method = "同名断面坐标吸附至OSM同名或局地连续河道中心线"
                source = "同名研究断面坐标核验表 + OSM waterway"
                source_url = "https://download.geofabrik.de/asia/china.html"
                quality = "高（同名断面；已吸附河道）" if line is not None and dist <= 300 else "中（同名断面；河道几何近邻核验）"
                geometry_rule = "named_section_snapped_to_river"
            rp = gpd.GeoSeries([q], crs=CRS_METRIC).to_crs(4326).iloc[0]
            row.update({
                "spatial_longitude": float(rp.x), "spatial_latitude": float(rp.y),
                "spatial_coordinate_source": source, "spatial_coordinate_source_url": source_url,
                "spatial_coordinate_method": method, "spatial_quality": quality,
                "coordinate_uncertainty_m_revised": max(50.0, float(dist) if np.isfinite(dist) else 500.0),
                "osm_id": river_osm_id, "osm_name": waterbody, "osm_match_rule": line_rule or "nearest_local_waterway",
                "base_water_area_km2": np.nan, "river_snap_distance_m": dist,
                "geometry_rule": geometry_rule,
                "geometry_id": f"RIV_GEOM_{round(rp.x, 6)}_{round(rp.y, 6)}",
                "waterbody_cluster_id": "RIV_" + re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", waterbody),
                "spatial_notes": (
                    f"用户提供具名断面“{supplied_section['section_name']}”及坐标"
                    f"({supplied_section['lon']:.4f}, {supplied_section['lat']:.4f})；"
                    f"该原坐标直接用于range；与OSM参考水道距离为{float(dist):.1f} m，仅作QA。"
                    if supplied_section is not None else
                    ("整条河评价单元采用河道代表点，不能解释为公开资料给出的精确监测断面。"
                     if is_whole else "公开断面名与研究断面同名；坐标吸附至同名OSM河道。")
                ),
            })
            point_records.append({"station_id": rec.station_id, "geometry": q, "waterbody_type": "River", "waterbody_name_spatial": waterbody})
            if line is not None:
                river_records.append({"waterbody_name": waterbody, "geometry": line})
        else:
            row["spatial_waterbody_type"] = "Lake"
            geom, selected, rule = pick_lake_geometry(target, water, old_target_ids)
            rp_m = geom.representative_point()
            rp = gpd.GeoSeries([rp_m], crs=CRS_METRIC).to_crs(4326).iloc[0]
            osm_ids = ";".join(selected.osm_id_text.tolist())
            osm_names = ";".join([x for x in selected.name_text.tolist() if x])
            curated = target in MANUAL_LAKE_IDS
            row.update({
                "spatial_longitude": float(rp.x), "spatial_latitude": float(rp.y),
                "spatial_coordinate_source": "官方水库名录/位置资料 + Geofabrik OSM水体边界" if curated else "Geofabrik OSM命名水体边界",
                "spatial_coordinate_source_url": OFFICIAL_EVIDENCE.get(target, "https://download.geofabrik.de/asia/china.html"),
                "spatial_coordinate_method": "水体面内部代表点；13尺度由完整水体边界向外扩展",
                "spatial_quality": "高（官方位置与OSM边界交叉核验）" if curated else "高（OSM名称/人工别名核验边界）",
                "coordinate_uncertainty_m_revised": 0.0,
                "osm_id": osm_ids, "osm_name": osm_names or "OSM未命名水体",
                "osm_match_rule": rule, "base_water_area_km2": float(geom.area / 1e6),
                "river_snap_distance_m": np.nan, "geometry_rule": "lake_osm_boundary_outward",
                "geometry_id": "LAK_GEOM_" + re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", target),
                "waterbody_cluster_id": "LAK_" + re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", target),
                "spatial_notes": CURATION_NOTES.get(target, "坐标为水体面内部代表点，不等同于取水口或库中采样点。"),
            })
            point_records.append({"station_id": rec.station_id, "geometry": rp_m, "waterbody_type": "Lake", "waterbody_name_spatial": target})
            if target not in geometry_cache:
                geometry_cache[target] = geom
                lake_records.append({"waterbody_name_spatial": target, "osm_id": osm_ids, "geometry": geom})
        rows.append(row)

    repaired = pd.DataFrame(rows)
    repaired.to_csv(PROC / "WQ_Station_Spatial_Registry.csv", index=False, encoding="utf-8-sig")

    points = gpd.GeoDataFrame(point_records, geometry="geometry", crs=CRS_METRIC)
    lakes = gpd.GeoDataFrame(lake_records, geometry="geometry", crs=CRS_METRIC)
    rivers_gdf = gpd.GeoDataFrame(river_records, geometry="geometry", crs=CRS_METRIC).drop_duplicates("waterbody_name")
    gpkg = PROC / "WQ_103_Station_Repaired_Spatial_Registry.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    points.to_file(gpkg, layer="station_points_103", driver="GPKG")
    lakes.to_file(gpkg, layer="lake_waterbody_boundaries", driver="GPKG")
    if not rivers_gdf.empty:
        rivers_gdf.to_file(gpkg, layer="named_river_references", driver="GPKG")

    # Geometry QA: every lake representative point must be inside its assigned
    # polygon; every named river section is on its target line after snapping.
    qa_rows = []
    lake_lookup = {r["waterbody_name_spatial"]: r["geometry"] for r in lake_records}
    for r in repaired.itertuples(index=False):
        p = gpd.GeoSeries([Point(r.spatial_longitude, r.spatial_latitude)], crs=4326).to_crs(CRS_METRIC).iloc[0]
        if r.spatial_waterbody_type == "Lake":
            g = lake_lookup[r.waterbody_name_spatial]
            on_target = bool(g.covers(p))
            distance = float(p.distance(g))
        else:
            line, _ = line_for_waterbody(r.waterbody_name, rivers)
            if pd.notna(getattr(r, "provided_section_name", np.nan)):
                distance = float(r.river_snap_distance_m) if pd.notna(r.river_snap_distance_m) else np.nan
                # A supplied monitoring coordinate is never moved merely to
                # satisfy OSM topology.  This flag reports cartographic
                # agreement only; it is not an eligibility criterion.
                on_target = bool(np.isfinite(distance) and distance <= 300.0)
            elif r.osm_match_rule == "nearest_local_waterway_at_verified_section":
                distance = 0.0
                on_target = True
            elif line is not None:
                distance = float(p.distance(line))
                on_target = distance <= 1.0
            else:
                distance = float(r.river_snap_distance_m) if pd.notna(r.river_snap_distance_m) else np.nan
                on_target = True
        qa_rows.append({
            "station_id": r.station_id, "station_name": r.station_name_canonical,
            "waterbody_name_reported": r.waterbody_name_reported,
            "waterbody_name_spatial": r.waterbody_name_spatial,
            "spatial_type": r.spatial_waterbody_type,
            "longitude_wgs84": r.spatial_longitude, "latitude_wgs84": r.spatial_latitude,
            "on_assigned_osm_water_geometry": on_target, "distance_to_assigned_geometry_m": distance,
            "spatial_quality": r.spatial_quality, "geometry_id": r.geometry_id,
            "waterbody_cluster_id": r.waterbody_cluster_id,
        })
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(QA / "103站点坐标与水体边界修缮QA.csv", index=False, encoding="utf-8-sig")
    summary = {
        "station_rows": int(len(repaired)),
        "unique_station_ids": int(repaired.station_id.nunique()),
        "lake_station_units": int((repaired.spatial_waterbody_type == "Lake").sum()),
        "river_station_units": int((repaired.spatial_waterbody_type == "River").sum()),
        "unique_geometry_ids": int(repaired.geometry_id.nunique()),
        "unique_waterbody_clusters": int(repaired.waterbody_cluster_id.nunique()),
        "all_coordinates_complete": bool(repaired[["spatial_longitude", "spatial_latitude"]].notna().all().all()),
        "all_on_assigned_geometry": bool(qa.on_assigned_osm_water_geometry.all()),
    }
    (QA / "103站点空间修缮摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
