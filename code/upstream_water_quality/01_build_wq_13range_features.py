# -*- coding: utf-8 -*-
"""Build 13-range environmental features for public water-quality units.

Rules reproduce the project's raw 13-range ML-input workflow:
River = point-centred cumulative circles; Lake/Reservoir = outward buffers
from a verified OSM open-water polygon, with open water excluded from the
analysis land denominator. No global transformation or standardisation is
applied in this construction stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import math
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(r"E:\桌面\博士课题\Data")
OUT = Path(__file__).resolve().parent.parent
PROC = OUT / "01_INPUT与过程数据"
QA = OUT / "06_QA"
TMP = OUT / "intermediate"
for p in [PROC, QA, TMP]:
    p.mkdir(parents=True, exist_ok=True)

SCALES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
CRS_METRIC = "EPSG:32649"
OSM = ROOT / r"data\raw\栅格数据集\guangdong-free-shp"
WATER_SHP = OSM / "gis_osm_water_a_free_1.shp"
WORLD_COVER = ROOT / r"data\raw\栅格数据集\WorldCover\landuse2021.tif"
WORLDPOP_CLIP = ROOT / r"data\raw\栅格数据集\WorldPop\WorldPop100m数据分析"
WORLDPOP_FULL = ROOT / r"data\raw\栅格数据集\WorldPop\100m人口密度原始数据"
RAW_WQ = ROOT / r"data\raw\深圳研究区水质监测站公开数据_2021-2025.xlsx"


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


river_mod = load_module("river_osm_wq", ROOT / r"tests\OSM数据处理\River_OSM缓冲区分析最终版.py")
poi_mod = load_module("poi_osm_wq", ROOT / r"tests\OSM数据处理\poi_p1_p8_all_sites_batch.py")
dem_mod = load_module("dem_wq", ROOT / r"src\features\build_dem_terrain_features.py")
for mod in [river_mod, poi_mod, dem_mod]:
    if hasattr(mod, "BUFFER_DISTANCES"):
        mod.BUFFER_DISTANCES = SCALES
    if hasattr(mod, "SCALES"):
        mod.SCALES = SCALES
    if hasattr(mod, "MAX_BUFFER_M"):
        mod.MAX_BUFFER_M = max(SCALES)
# The original terrain workflow uses UTM 49N; force the OSM modules to the
# same metric CRS so every analysis geometry is passed to the rasters without
# a hidden projection mismatch.
river_mod.choose_metric_crs = lambda _gdf: CRS_METRIC
poi_mod.choose_metric_crs = lambda _gdf: CRS_METRIC


def norm_text(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def clean_waterbody_name(x) -> str:
    s = norm_text(x).replace("（", "(").replace("）", ")")
    if "-" in s:
        s = s.split("-")[-1]
    for suffix in ["饮用水源", "饮用水", "饮用"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s.strip()


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


def make_station_spatial_registry():
    """Load the independently repaired 103-row registry.

    Feature extraction is performed once per unique geometry_id (56 after
    curation), then linked back to all 103 station/source units. This avoids
    both memory blow-up and artificial recomputation of identical lake
    landscapes.
    """
    repaired_path = PROC / "WQ_Station_Spatial_Registry.csv"
    reg = pd.read_csv(repaired_path, encoding="utf-8-sig")
    required = {"station_id", "geometry_id", "spatial_longitude", "spatial_latitude", "osm_id", "spatial_waterbody_type"}
    if len(reg) != 103 or not required.issubset(reg.columns):
        raise ValueError("Run 11_repair_103_station_spatial_registry.py before feature construction")
    water = gpd.read_file(WATER_SHP, bbox=(113.35, 22.25, 115.25, 23.15))
    water = water[water.geometry.notna() & ~water.geometry.is_empty].copy()
    water["name_text"] = water["name"].fillna("").astype(str).str.strip()
    water = water[water.fclass.isin(["water", "reservoir", "riverbank", "dock"])].copy()
    water_m = water.to_crs(CRS_METRIC)
    water_m.geometry = water_m.geometry.buffer(0)
    water_m["area_km2"] = water_m.geometry.area / 1e6

    spatial_geoms = {}
    for geometry_id, d in reg.groupby("geometry_id", sort=True):
        rec = d.iloc[0]
        lon, lat = float(rec.spatial_longitude), float(rec.spatial_latitude)
        if rec.spatial_waterbody_type == "Lake":
            ids = [x for x in norm_text(rec.osm_id).split(";") if x]
            selected = water_m[water_m.osm_id.astype(str).isin(ids)].copy()
            if selected.empty:
                raise ValueError(f"Assigned OSM lake geometry missing: {geometry_id} ids={ids}")
            geom = unary_union(selected.geometry.tolist())
            spatial_geoms[geometry_id] = {
                "type": "Lake", "geometry": Point(lon, lat), "base": geom,
                "name": rec.waterbody_name_spatial,
            }
        else:
            spatial_geoms[geometry_id] = {
                "type": "River", "geometry": Point(lon, lat), "base": None,
                "name": rec.station_name_canonical,
            }

    qa = reg.groupby(["waterbody_type", "spatial_eligible", "spatial_quality"], dropna=False).size().reset_index(name="n_station_units")
    qa.to_csv(QA / "空间适用性汇总.csv", index=False, encoding="utf-8-sig")
    return reg, spatial_geoms, water_m


def geodata_from_spatial(spatial_geoms):
    rows = []
    for sid, d in spatial_geoms.items():
        geom = d["geometry"]
        if d["type"] == "Lake":
            geom = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(CRS_METRIC).iloc[0]
        else:
            geom = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(CRS_METRIC).iloc[0]
        rows.append({"Site_ID": sid, "Water body type": d["type"], "name": d["name"], "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_METRIC)


def subset(gdf, geom, fclasses=None):
    if gdf is None or len(gdf) == 0:
        return gdf.iloc[0:0].copy()
    src = gdf
    if fclasses is not None and "fclass" in src:
        src = src[src.fclass.isin(fclasses)]
    if len(src) == 0:
        return src.copy()
    try:
        idx = src.sindex.query(geom, predicate="intersects")
        return src.iloc[np.asarray(idx, dtype=int)].copy()
    except Exception:
        return src[src.geometry.intersects(geom)].copy()


def clipped_union(gdf, geom, fclasses=None):
    src = subset(gdf, geom, fclasses)
    if src is None or len(src) == 0:
        return None
    geoms = []
    for g in src.geometry:
        try:
            x = g.intersection(geom)
            if x is not None and not x.is_empty:
                geoms.append(x)
        except Exception:
            continue
    return None if not geoms else unary_union(geoms)


def area_in(g, geom):
    if g is None:
        return 0.0
    try:
        return float(g.intersection(geom).area)
    except Exception:
        return np.nan


def raster_values(ds, geom_m, all_touched=True, invalid_to_zero=False):
    to_ds = Transformer.from_crs(CRS_METRIC, ds.crs, always_xy=True).transform
    geom_ds = transform(to_ds, geom_m)
    try:
        arr = mask(ds, [mapping(geom_ds)], crop=True, all_touched=all_touched, filled=False)[0][0]
    except ValueError:
        return np.array([], dtype=float)
    vals = arr.compressed().astype(float)
    if ds.nodata is not None:
        vals = vals[~np.isclose(vals, ds.nodata)]
    vals = vals[np.isfinite(vals)]
    if invalid_to_zero and len(vals):
        vals[~np.isfinite(vals)] = 0
    return vals


def prepare_dem(points_m, water_m):
    dem_dir = TMP / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    dem_mod.DERIVED_DIR = dem_dir
    dem_mod.WORK_DIR = dem_dir
    dem_mod.DEM_UTM = dem_dir / "WQ_DEM_ASTER_UTM49N_30m.tif"
    dem_mod.SLOPE_DEG = dem_dir / "WQ_Slope_Degrees_UTM49N_30m.tif"
    dem_mod.SLOPE_PCT = dem_dir / "WQ_Slope_Percent_UTM49N_30m.tif"
    dem_mod.OSM_WATER_UTM_10 = dem_dir / "WQ_OSM_Water_UTM49N_10m.tif"
    dem_mod.OSM_LAND_FRACTION_30 = dem_dir / "WQ_OSM_LandFraction_UTM49N_30m.tif"
    site_xy = pd.DataFrame({"x": points_m.geometry.x, "y": points_m.geometry.y})
    dem_mod.prepare_derived_rasters(site_xy, water_m)
    return dem_mod


def build_all_features(spatial_geoms, water_m):
    points_m = geodata_from_spatial(spatial_geoms)
    points_ll = points_m.to_crs("EPSG:4326").copy()
    points_ll["longitude"] = points_ll.geometry.x
    points_ll["latitude"] = points_ll.geometry.y
    points_ll["group_name"] = "WQ"

    river_mod.MAX_BUFFER_M = max(SCALES)
    layers = river_mod.load_and_local_clip_layers(OSM, points_ll)
    metric_crs = layers["metric_crs"]
    if str(metric_crs) != CRS_METRIC:
        points_m = points_ll.to_crs(metric_crs)
    else:
        points_m = points_m.to_crs(metric_crs)
    water_layer = layers["water_a"]

    poi_mod.MAX_BUFFER_M = max(SCALES)
    group_map = poi_mod.load_group_mapping(poi_mod.GROUP_XLSX)
    poi_poly, poi_points = poi_mod.load_poi_layers(points_ll, metric_crs, group_map)
    group_map.to_csv(PROC / "OSM_API_P1_P8_Group_Mapping.csv", index=False, encoding="utf-8-sig")

    dem = prepare_dem(points_m.to_crs(CRS_METRIC), water_m.to_crs(CRS_METRIC))
    dem_handles = [rasterio.open(dem.DEM_UTM), rasterio.open(dem.SLOPE_DEG), rasterio.open(dem.SLOPE_PCT), rasterio.open(dem.OSM_LAND_FRACTION_30)]

    wc_ds = rasterio.open(WORLD_COVER)
    pop_ds = {}
    for year in range(2021, 2026):
        clip_path = WORLDPOP_CLIP / f"chn_pop_{year}_CN_100m_R2025A_v1_clip.tif"
        full_path = WORLDPOP_FULL / f"chn_pop_{year}_CN_100m_R2025A_v1.tif"
        pop_ds[year] = {
            "clip": rasterio.open(clip_path) if clip_path.exists() else None,
            "full": rasterio.open(full_path) if full_path.exists() else None,
        }

    terrain_rows, urban_rows, cover_rows, api_rows, pop_rows, buffer_rows = [], [], [], [], [], []
    point_rows, lake_qa = [], []
    groups = [f"P{i}" for i in range(1, 9)]
    wc_groups = {"urban": [50], "agriculture": [40], "green": [10, 20, 30], "water_wetland": [80, 90, 95]}

    point_lookup = points_m.set_index("Site_ID")
    for i, (sid, d) in enumerate(sorted(spatial_geoms.items()), 1):
        print(f"FEATURE {i}/{len(spatial_geoms)} {sid}", flush=True)
        p = point_lookup.loc[sid].geometry
        base = None
        if d["type"] == "Lake":
            base = d["base"]
            if gpd.GeoSeries([base], crs=CRS_METRIC).crs != metric_crs:
                base = gpd.GeoSeries([base], crs=CRS_METRIC).to_crs(metric_crs).iloc[0]
        max_buffer = p.buffer(max(SCALES)) if base is None else base.buffer(max(SCALES))
        water_union = clipped_union(water_layer, max_buffer)
        max_land = max_buffer if water_union is None else max_buffer.difference(water_union)

        building_u = clipped_union(layers["buildings_a"], max_land)
        roads_src = subset(layers["roads_free"], max_buffer, river_mod.ROAD_IMPERVIOUS_FCLASS)
        roads_poly = river_mod.roads_to_polygons(roads_src, max_buffer, metric_crs)
        roads_u = clipped_union(roads_poly, max_land)
        traffic_u = clipped_union(layers["traffic_a"], max_land, river_mod.TRAFFIC_IMPERVIOUS_FCLASS)
        transport_u = clipped_union(layers["transport_a"], max_land, river_mod.TRANSPORT_IMPERVIOUS_FCLASS)
        impervious_parts = [g for g in [building_u, roads_u, traffic_u, transport_u] if g is not None]
        impervious_u = unary_union(impervious_parts) if impervious_parts else None
        natural_lu = clipped_union(layers["landuse_a"], max_land, river_mod.NATURAL_LANDUSE_FCLASS)
        natural_a = clipped_union(layers["natural_a"], max_land, river_mod.NATURAL_A_FCLASS)
        natural_parts = [g for g in [natural_lu, natural_a] if g is not None]
        natural_u = unary_union(natural_parts) if natural_parts else None

        # POI candidates are restricted once at maximum scale.
        poi_pt_sub = subset(poi_points, max_land)
        poi_poly_sub = subset(poi_poly, max_land)

        # Terrain point/reference variables.
        dm, sd, sp, lf = dem_handles
        x, y = p.x, p.y
        point_dem = dem.sample_nearest(dm, x, y)
        if base is not None:
            bnd = dem.boundary_reference(base, dm, sd, sp, lf)
            point_slope = bnd["boundary_mean_slope_pct"]
            point_rule = "OSM shoreline outer-land band mean"
            lake_qa.append({"spatial_id": sid, "base_water_area_km2": base.area / 1e6,
                            "boundary_mean_dem_m": bnd["boundary_mean_dem_m"], **bnd})
        else:
            _, point_slope, dist = dem.nearest_land_slope(x, y, sd, sp, lf)
            point_rule = f"nearest non-water land pixel; distance={dist:.1f}m"
        point_rows.append({"spatial_id": sid, "water_body_type": d["type"], "point_dem_m": point_dem,
                           "point_slope_pct": point_slope, "point_slope_rule": point_rule})

        for scale in SCALES:
            buffer_geom = p.buffer(scale, resolution=48) if base is None else base.buffer(scale)
            water_u = clipped_union(water_layer, buffer_geom)
            land = buffer_geom if water_u is None else buffer_geom.difference(water_u)
            land_area = float(land.area)
            buffer_rows.append({"spatial_id": sid, "water_body_type": d["type"], "range_m": scale,
                                "geometry_rule": "river_point_circle" if base is None else "lake_osm_shoreline_outward",
                                "buffer_area_m2": buffer_geom.area, "analysis_land_area_m2": land_area,
                                "excluded_water_area_m2": buffer_geom.area - land_area, "geometry": land})

            urban_rows.append({
                "spatial_id": sid, "water_body_type": d["type"], "range_m": scale,
                "analysis_land_area_m2": land_area,
                "impervious_area_m2": area_in(impervious_u, land),
                "building_area_m2": area_in(building_u, land),
                "road_area_m2": area_in(roads_u, land),
                "traffic_area_m2": area_in(traffic_u, land),
                "transport_area_m2": area_in(transport_u, land),
                "natural_area_m2": area_in(natural_u, land),
            })
            u = urban_rows[-1]
            for a, n in [("impervious_area_m2", "impervious_pct"), ("building_area_m2", "building_pct"), ("natural_area_m2", "natural_land_pct")]:
                u[n] = u[a] / land_area * 100 if land_area > 0 else np.nan

            vals = raster_values(wc_ds, land)
            vals = vals[vals > 0]
            rec = {"spatial_id": sid, "water_body_type": d["type"], "range_m": scale, "valid_pixel_count": len(vals)}
            for name, codes in wc_groups.items():
                cnt = int(np.isin(vals.astype(np.int16), codes).sum())
                rec[f"{name}_pixel_count"] = cnt
                rec[f"{name}_pct"] = cnt / len(vals) * 100 if len(vals) else np.nan
            cover_rows.append(rec)

            stats = dem.summarize_geometry(land, dm, sd, sp, lf)
            stats["delta_dem_buffer_minus_point_m"] = stats["mean_dem_m"] - point_dem
            terrain_rows.append({"spatial_id": sid, "water_body_type": d["type"], "range_m": scale,
                                 "point_dem_m": point_dem, "point_slope_pct": point_slope, **stats})

            # Direct human-activity pressure groups.
            for g in groups:
                pts_g = poi_pt_sub[poi_pt_sub.pressure_group.eq(g)]
                pcount = int(pts_g.geometry.intersects(land).sum()) if len(pts_g) else 0
                polys_g = poi_poly_sub[poi_poly_sub.pressure_group.eq(g)]
                parea = 0.0
                if len(polys_g):
                    for geom in polys_g.geometry:
                        try:
                            parea += geom.intersection(land).area
                        except Exception:
                            pass
                api_rows.append({"spatial_id": sid, "water_body_type": d["type"], "range_m": scale,
                                 "pressure_group": g, "analysis_land_area_m2": land_area,
                                 "point_count": pcount, "polygon_area_m2": parea,
                                 "point_density_per_km2": pcount / (land_area / 1e6) if land_area > 0 else np.nan,
                                 "polygon_area_pct": parea / land_area * 100 if land_area > 0 else np.nan})

            for year, handles in pop_ds.items():
                pv = raster_values(handles["clip"], land) if handles["clip"] is not None else np.array([])
                source_extent = "study_clip"
                if len(pv) == 0 and handles["full"] is not None:
                    pv = raster_values(handles["full"], land)
                    source_extent = "national_fallback"
                pop_rows.append({"spatial_id": sid, "water_body_type": d["type"], "year": year,
                                 "range_m": scale, "valid_pixel_count": len(pv),
                                 "mean_density": float(np.mean(pv)) if len(pv) else np.nan,
                                 "raster_extent_used": source_extent})

        # Resume-friendly stage snapshots.
        pd.DataFrame(urban_rows).to_csv(TMP / "urban_long_checkpoint.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(terrain_rows).to_csv(TMP / "terrain_long_checkpoint.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(api_rows).to_csv(TMP / "api_long_checkpoint.csv", index=False, encoding="utf-8-sig")

    wc_ds.close()
    for handles in pop_ds.values():
        for ds in handles.values():
            if ds is not None:
                ds.close()
    for ds in dem_handles:
        ds.close()

    urban = pd.DataFrame(urban_rows)
    terrain = pd.DataFrame(terrain_rows)
    cover = pd.DataFrame(cover_rows)
    api = pd.DataFrame(api_rows)
    pop = pd.DataFrame(pop_rows)
    point = pd.DataFrame(point_rows)
    lake = pd.DataFrame(lake_qa)
    for name, df in [("Feature_Process_Urbanization_Long", urban), ("Feature_Process_Terrain_Long", terrain),
                     ("Feature_Process_WorldCover_Long", cover), ("Feature_Process_API_Long", api),
                     ("Feature_Process_WorldPop_Long", pop), ("Feature_Process_Terrain_Point", point),
                     ("Lake_Geometry_QA", lake)]:
        df.to_csv(PROC / f"{name}.csv", index=False, encoding="utf-8-sig")

    # Save geometries for reproducible spatial audit.
    bg = gpd.GeoDataFrame(buffer_rows, geometry="geometry", crs=metric_crs)
    gpkg = PROC / "WQ_13Range_Analysis_Geometries.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    points_m.to_file(gpkg, layer="spatial_units", driver="GPKG")
    bg.to_file(gpkg, layer="analysis_land_ranges", driver="GPKG")

    return urban, terrain, cover, api, pop, point, bg


def make_wide(urban, terrain, cover, api, pop, point):
    meta = point[["spatial_id", "water_body_type"]].drop_duplicates()
    wide = meta.copy()
    wide = wide.merge(point[["spatial_id", "point_dem_m", "point_slope_pct"]].rename(
        columns={"point_dem_m": "terrain_point_dem_m", "point_slope_pct": "terrain_point_slope_pct"}), on="spatial_id", how="left")

    def add_range(df, metrics, prefix):
        nonlocal wide
        for scale in SCALES:
            part = df[df.range_m.eq(scale)][["spatial_id", *metrics]].copy()
            part = part.rename(columns={c: f"{prefix}_{scale:04d}m_{c}" for c in metrics})
            wide = wide.merge(part, on="spatial_id", how="left", validate="1:1")

    add_range(urban, ["impervious_pct", "building_pct", "natural_land_pct"], "urbanization")
    add_range(cover, ["urban_pct", "agriculture_pct", "green_pct", "water_wetland_pct"], "landuse")
    terrain_metrics = ["mean_dem_m", "dem_range_m", "dem_std_m", "dem_var_m2", "delta_dem_buffer_minus_point_m",
                       "mean_slope_pct", "max_slope_pct", "slope_gt10pct_area_pct", "slope_gt15pct_area_pct"]
    add_range(terrain, terrain_metrics, "terrain")

    api_piv = api.pivot_table(index=["spatial_id", "range_m"], columns="pressure_group",
                              values=["point_density_per_km2", "polygon_area_pct"], aggfunc="first").reset_index()
    api_piv.columns = ["spatial_id", "range_m"] + [f"{g}_{metric}" for metric, g in api_piv.columns[2:]]
    api_metrics = [c for c in api_piv.columns if c not in ["spatial_id", "range_m"]]
    add_range(api_piv, api_metrics, "api")

    # Population wide remains year-specific for record-level matching.
    pop_wide = pop.pivot(index=["spatial_id", "water_body_type", "year"], columns="range_m", values="mean_density").reset_index()
    pop_wide.columns = ["spatial_id", "water_body_type", "year"] + [f"worldpop_{int(c):04d}m_mean_density" for c in pop_wide.columns[3:]]
    static_path = PROC / "WQ_Environmental_Features_Static_Wide.csv"
    wide.to_csv(static_path, index=False, encoding="utf-8-sig")
    pop_wide.to_csv(PROC / "WQ_WorldPop_Annual_Wide.csv", index=False, encoding="utf-8-sig")
    return wide, pop_wide


def main():
    reg, geoms, water_m = make_station_spatial_registry()
    print(json.dumps({"station_units": len(reg), "eligible_station_units": int(reg.spatial_eligible.sum()),
                      "independent_spatial_units": len(geoms)}, ensure_ascii=False), flush=True)
    urban, terrain, cover, api, pop, point, buffers = build_all_features(geoms, water_m)
    wide, pop_wide = make_wide(urban, terrain, cover, api, pop, point)
    qa = {
        "ranges": SCALES,
        "station_units": len(reg),
        "eligible_station_units": int(reg.spatial_eligible.sum()),
        "independent_spatial_units": len(geoms),
        "static_feature_rows": len(wide),
        "static_feature_columns": wide.shape[1] - 2,
        "population_rows": len(pop_wide),
        "buffer_rows": len(buffers),
        "expected_buffer_rows": len(geoms) * len(SCALES),
        "duplicate_spatial_range": int(buffers.duplicated(["spatial_id", "range_m"]).sum()),
        "static_missing_cells": int(wide.drop(columns=["spatial_id", "water_body_type"]).isna().sum().sum()),
    }
    (QA / "特征构建QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
