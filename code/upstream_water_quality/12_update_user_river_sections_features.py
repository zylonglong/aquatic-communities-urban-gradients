# -*- coding: utf-8 -*-
"""Selectively rebuild 13-range features for nine OSM-snapped river sections.

The unchanged 47 geometries are retained byte-for-byte in the long feature
tables.  Only the nine former whole-river representative geometries are
recomputed, which avoids rerunning the full 56-geometry GIS workflow.
"""
from pathlib import Path
import importlib.util
import json

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROC = OUT / "01_INPUT与过程数据"
QA = OUT / "06_QA"
SEL = OUT / "intermediate" / "river_sections_osm_snapped_20260721"
SEL_PROC = SEL / "process"
SEL_TMP = SEL / "tmp"
SEL_PROC.mkdir(parents=True, exist_ok=True)
SEL_TMP.mkdir(parents=True, exist_ok=True)

STATION_IDS = [
    "SZ_WQ_RIV_001", "SZ_WQ_RIV_003", "SZ_WQ_RIV_004",
    "SZ_WQ_RIV_005", "SZ_WQ_RIV_006", "SZ_WQ_RIV_009",
    "SZ_WQ_RIV_010", "SZ_WQ_RIV_011", "SZ_WQ_RIV_016",
]


def load_builder():
    path = HERE / "01_build_wq_13range_features.py"
    spec = importlib.util.spec_from_file_location("wq_feature_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_long(filename, new_df, old_ids, expected_rows):
    path = PROC / filename
    old = pd.read_csv(path, low_memory=False)
    kept = old[~old.spatial_id.isin(old_ids)].copy()
    merged = pd.concat([kept, new_df], ignore_index=True, sort=False)
    if len(merged) != expected_rows:
        raise ValueError(f"{filename}: expected {expected_rows}, got {len(merged)}")
    if {"spatial_id", "range_m"}.issubset(merged.columns):
        extra = ["pressure_group"] if "pressure_group" in merged.columns else []
        extra += ["year"] if "year" in merged.columns else []
        if merged.duplicated(["spatial_id", "range_m", *extra]).any():
            raise ValueError(f"{filename}: duplicate keys after selective merge")
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def main():
    old_site = pd.read_csv(PROC / "WQ_Site_Special_13Ranges.csv", low_memory=False)
    old_ids = set(old_site.loc[old_site.station_id.isin(STATION_IDS), "spatial_id"].dropna())
    if len(old_ids) != 9:
        raise ValueError(f"Expected 9 old geometry ids, found {len(old_ids)}")

    m = load_builder()
    m.OUT, m.PROC, m.QA, m.TMP = OUT, PROC, QA, SEL_TMP
    reg, geoms, water_m = m.make_station_spatial_registry()
    new_ids = set(reg.loc[reg.station_id.isin(STATION_IDS), "geometry_id"])
    if len(new_ids) != 9:
        raise ValueError(f"Expected 9 new geometry ids, found {len(new_ids)}")
    subset_geoms = {sid: geoms[sid] for sid in sorted(new_ids)}

    # Redirect intermediate outputs so the full production tables are not
    # overwritten until the selective run has completed successfully.
    original_proc, original_tmp = m.PROC, m.TMP
    m.PROC, m.TMP = SEL_PROC, SEL_TMP
    try:
        urban_n, terrain_n, cover_n, api_n, pop_n, point_n, buffers_n = m.build_all_features(subset_geoms, water_m)
    finally:
        m.PROC, m.TMP = original_proc, original_tmp

    urban = merge_long("Feature_Process_Urbanization_Long.csv", urban_n, old_ids, 56 * 13)
    terrain = merge_long("Feature_Process_Terrain_Long.csv", terrain_n, old_ids, 56 * 13)
    cover = merge_long("Feature_Process_WorldCover_Long.csv", cover_n, old_ids, 56 * 13)
    api = merge_long("Feature_Process_API_Long.csv", api_n, old_ids, 56 * 13 * 8)
    pop = merge_long("Feature_Process_WorldPop_Long.csv", pop_n, old_ids, 56 * 13 * 5)
    point = merge_long("Feature_Process_Terrain_Point.csv", point_n, old_ids, 56)

    # Rebuild the geometry audit package with all current geometries.
    gpkg = PROC / "WQ_13Range_Analysis_Geometries.gpkg"
    old_buffers = gpd.read_file(gpkg, layer="analysis_land_ranges")
    buffers = pd.concat([old_buffers[~old_buffers.spatial_id.isin(old_ids)], buffers_n], ignore_index=True)
    buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=old_buffers.crs)
    if len(buffers) != 56 * 13 or buffers.duplicated(["spatial_id", "range_m"]).any():
        raise ValueError("Merged range geometries are incomplete or duplicated")
    points = m.geodata_from_spatial(geoms)
    if gpkg.exists():
        gpkg.unlink()
    points.to_file(gpkg, layer="spatial_units", driver="GPKG")
    buffers.to_file(gpkg, layer="analysis_land_ranges", driver="GPKG")

    # Recreate wide matrices from the complete long tables.
    m.PROC = PROC
    wide, pop_wide = m.make_wide(urban, terrain, cover, api, pop, point)
    summary = {
        "updated_station_ids": STATION_IDS,
        "old_geometry_ids": sorted(old_ids),
        "new_geometry_ids": sorted(new_ids),
        "static_rows": len(wide),
        "static_feature_columns": wide.shape[1] - 2,
        "population_rows": len(pop_wide),
        "range_geometry_rows": len(buffers),
        "static_missing_cells_before_targeted_repair": int(wide.drop(columns=["spatial_id", "water_body_type"]).isna().sum().sum()),
    }
    (QA / "9个River对应河流投影点选择性重算摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
