# -*- coding: utf-8 -*-
from pathlib import Path
import re
from importlib.util import module_from_spec, spec_from_file_location

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

ROOT = Path(r"E:\桌面\博士课题\Data")
OUT = Path(__file__).resolve().parent.parent
PROC, QA = OUT / "01_INPUT与过程数据", OUT / "06_QA"
OSM = ROOT / r"data\raw\栅格数据集\guangdong-free-shp"

urban_path = PROC / "Feature_Process_Urbanization_Long.csv"
static_path = PROC / "WQ_Environmental_Features_Static_Wide.csv"
urban = pd.read_csv(urban_path)
static = pd.read_csv(static_path)
targets = urban[urban.natural_land_pct.isna()][["spatial_id", "range_m", "analysis_land_area_m2"]].copy()

if len(targets):
    geoms = gpd.read_file(PROC / "WQ_13Range_Analysis_Geometries.gpkg", layer="analysis_land_ranges")
    geoms = geoms.merge(targets[["spatial_id", "range_m"]], on=["spatial_id", "range_m"], how="inner")
    bbox = geoms.to_crs("EPSG:4326").total_bounds
    lu = gpd.read_file(OSM / "gis_osm_landuse_a_free_1.shp", bbox=tuple(bbox))
    na = gpd.read_file(OSM / "gis_osm_natural_a_free_1.shp", bbox=tuple(bbox))
    lu = lu[lu.fclass.isin(["forest", "grass", "heath", "meadow", "scrub"])].to_crs(geoms.crs)
    na = na[na.fclass.isin(["beach", "cliff"])].to_crs(geoms.crs)
    repair_rows = []
    for rec in geoms.itertuples(index=False):
        land = rec.geometry.buffer(0)
        pieces = []
        for layer in [lu, na]:
            try:
                idx = layer.sindex.query(land, predicate="intersects")
                sub = layer.iloc[np.asarray(idx, dtype=int)]
            except Exception:
                sub = layer[layer.geometry.intersects(land)]
            for geom in sub.geometry:
                try:
                    piece = geom.buffer(0).intersection(land)
                    if not piece.is_empty:
                        pieces.append(piece)
                except Exception:
                    pass
        natural = unary_union(pieces).area if pieces else 0.0
        land_area = land.area
        pct = natural / land_area * 100 if land_area > 0 else np.nan
        mask = urban.spatial_id.eq(rec.spatial_id) & urban.range_m.eq(rec.range_m)
        urban.loc[mask, "natural_area_m2"] = natural
        urban.loc[mask, "natural_land_pct"] = pct
        col = f"urbanization_{int(rec.range_m):04d}m_natural_land_pct"
        static.loc[static.spatial_id.eq(rec.spatial_id), col] = pct
        repair_rows.append({"spatial_id": rec.spatial_id, "range_m": rec.range_m,
                            "natural_area_m2_recomputed": natural, "natural_land_pct_recomputed": pct})
    pd.DataFrame(repair_rows).to_csv(QA / "自然用地比例定点几何修复.csv", index=False, encoding="utf-8-sig")
    urban.to_csv(urban_path, index=False, encoding="utf-8-sig")
    static.to_csv(static_path, index=False, encoding="utf-8-sig")

# Rare GEOS topology failures can make a single road/impervious union NA.
# Recompute only affected geometries from the original OSM layers rather than
# interpolating across scales.
urban = pd.read_csv(urban_path)
static = pd.read_csv(static_path)
impervious_targets = urban[urban.impervious_pct.isna()][["spatial_id", "range_m"]].copy()
if len(impervious_targets):
    spec = spec_from_file_location("river_osm_repair", ROOT / r"tests\OSM数据处理\River_OSM缓冲区分析最终版.py")
    river_mod = module_from_spec(spec)
    spec.loader.exec_module(river_mod)
    geoms_imp = gpd.read_file(PROC / "WQ_13Range_Analysis_Geometries.gpkg", layer="analysis_land_ranges")
    geoms_imp = geoms_imp.merge(impervious_targets, on=["spatial_id", "range_m"], how="inner")
    bbox = tuple(geoms_imp.to_crs("EPSG:4326").total_bounds)
    buildings = gpd.read_file(OSM / "gis_osm_buildings_a_free_1.shp", bbox=bbox).to_crs(geoms_imp.crs)
    roads = gpd.read_file(OSM / "gis_osm_roads_free_1.shp", bbox=bbox).to_crs(geoms_imp.crs)
    traffic = gpd.read_file(OSM / "gis_osm_traffic_a_free_1.shp", bbox=bbox).to_crs(geoms_imp.crs)
    transport = gpd.read_file(OSM / "gis_osm_transport_a_free_1.shp", bbox=bbox).to_crs(geoms_imp.crs)
    roads = roads[roads.fclass.isin(river_mod.ROAD_IMPERVIOUS_FCLASS)].copy()
    traffic = traffic[traffic.fclass.isin(river_mod.TRAFFIC_IMPERVIOUS_FCLASS)].copy()
    transport = transport[transport.fclass.isin(river_mod.TRANSPORT_IMPERVIOUS_FCLASS)].copy()
    repair_imp = []
    for rec in geoms_imp.itertuples(index=False):
        land = rec.geometry.buffer(0)
        pieces = []
        for layer, is_road in [(buildings, False), (roads, True), (traffic, False), (transport, False)]:
            try:
                idx = layer.sindex.query(land, predicate="intersects")
                sub = layer.iloc[np.asarray(idx, dtype=int)].copy()
            except Exception:
                sub = layer[layer.geometry.intersects(land)].copy()
            if is_road and len(sub):
                sub = river_mod.roads_to_polygons(sub, land, geoms_imp.crs)
            for geom in sub.geometry:
                try:
                    x = geom.buffer(0).intersection(land)
                    if not x.is_empty:
                        pieces.append(x)
                except Exception:
                    continue
        impervious_area = float(unary_union(pieces).area) if pieces else 0.0
        land_area = float(land.area)
        pct = impervious_area / land_area * 100 if land_area > 0 else np.nan
        mask = urban.spatial_id.eq(rec.spatial_id) & urban.range_m.eq(rec.range_m)
        urban.loc[mask, "impervious_area_m2"] = impervious_area
        urban.loc[mask, "impervious_pct"] = pct
        col = f"urbanization_{int(rec.range_m):04d}m_impervious_pct"
        static.loc[static.spatial_id.eq(rec.spatial_id), col] = pct
        repair_imp.append({"spatial_id": rec.spatial_id, "range_m": rec.range_m,
                           "impervious_area_m2_recomputed": impervious_area,
                           "impervious_pct_recomputed": pct})
    pd.DataFrame(repair_imp).to_csv(QA / "不透水面比例定点几何修复.csv", index=False, encoding="utf-8-sig")
    urban.to_csv(urban_path, index=False, encoding="utf-8-sig")
    static.to_csv(static_path, index=False, encoding="utf-8-sig")

# Correct percentage-domain QA: percent slope is a rise/run measure and can
# exceed 100; only area/coverage shares are constrained to [0,100].
share_cols = [c for c in static.columns if (
    c.startswith("urbanization_") or c.startswith("landuse_") or
    (c.startswith("api_") and c.endswith("polygon_area_pct")) or
    (c.startswith("terrain_") and ("slope_gt10pct_area_pct" in c or "slope_gt15pct_area_pct" in c))
)]
long = static.melt(id_vars=["spatial_id", "water_body_type"], value_vars=share_cols,
                   var_name="feature", value_name="value")
bad = long[(long.value < -1e-8) | (long.value > 100 + 1e-8)].copy()
bad.to_csv(QA / "面积比例0至100范围异常.csv", index=False, encoding="utf-8-sig")

missing = static.melt(id_vars=["spatial_id", "water_body_type"], var_name="feature", value_name="value")
missing = missing[missing.value.isna()]
missing.to_csv(QA / "静态特征缺失单元格_修复后.csv", index=False, encoding="utf-8-sig")
print({"repaired": len(targets), "remaining_missing": len(missing), "share_out_of_range": len(bad)})
