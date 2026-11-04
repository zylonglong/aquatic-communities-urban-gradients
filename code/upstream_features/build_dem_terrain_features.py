from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import Point, mapping


ROOT = Path(r"E:\桌面\博士课题\Data")
COORD_XLSX = ROOT / "data" / "processed" / "DEM数据处理" / "coordinates" / "深圳KML点位经纬度汇总_converted.xlsx"
COORD_SOURCE = ROOT / "data" / "raw" / "深圳KML点位经纬度汇总.xls"
DEM_SOURCE = ROOT / "data" / "raw" / "shenzhen.dem" / "shenzhen_未裁剪DEM.tif"
OSM_WATER_SOURCE = ROOT / "data" / "raw" / "栅格数据集" / "guangdong-free-shp" / "gis_osm_water_a_free_1.shp"
SAMPLE_MATRIX = ROOT / "data" / "processed" / "训练数据汇总" / "Sample_Special" / "ML_Input_Sample_Special_Feature_Matrix_Expanded_Biological_Response.xlsx"
SITE_MATRIX = ROOT / "data" / "processed" / "训练数据汇总" / "Site_Special" / "ML_Input_Site_Special_Feature_Matrix_Expanded_Biological_Response.xlsx"

PROCESS_ROOT = ROOT / "data" / "processed" / "DEM数据处理"
DERIVED_DIR = PROCESS_ROOT / "derived_rasters"
WORK_DIR = ROOT / "work_dem" / "terrain_intermediate"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEM_UTM = DERIVED_DIR / "DEM_ASTER_UTM49N_30m.tif"
SLOPE_DEG = DERIVED_DIR / "Slope_Degrees_UTM49N_30m.tif"
SLOPE_PCT = DERIVED_DIR / "Slope_Percent_UTM49N_30m.tif"
OSM_WATER_UTM_10 = DERIVED_DIR / "OSM_Water_2026-04-15_UTM49N_10m.tif"
OSM_LAND_FRACTION_30 = DERIVED_DIR / "OSM_LandFraction_2026-04-15_UTM49N_30m.tif"

TARGET_CRS = "EPSG:32649"
SCALES = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000]
OSM_OPEN_WATER_CLASSES = {"water", "reservoir", "riverbank", "dock"}
NODATA_FLOAT = -9999.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_grid(bounds: tuple[float, float, float, float], resolution: float):
    left, bottom, right, top = bounds
    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    return Affine(resolution, 0, left, 0, -resolution, top), width, height


def write_raster(path: Path, array: np.ndarray, transform, crs, nodata, dtype):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype, copy=False), 1)


def prepare_derived_rasters(site_xy: pd.DataFrame, osm_water: gpd.GeoDataFrame):
    margin = max(SCALES) + 1500
    bounds = (
        float(site_xy.x.min() - margin),
        float(site_xy.y.min() - margin),
        float(site_xy.x.max() + margin),
        float(site_xy.y.max() + margin),
    )
    transform30, width30, height30 = aligned_grid(bounds, 30.0)
    transform10, width10, height10 = aligned_grid(bounds, 10.0)
    if DEM_UTM.exists():
        with rasterio.open(DEM_UTM) as existing_dem:
            transform30 = existing_dem.transform
            width30, height30 = existing_dem.width, existing_dem.height
            transform10, width10, height10 = aligned_grid(existing_dem.bounds, 10.0)

    if not all(path.exists() for path in [DEM_UTM, SLOPE_DEG, SLOPE_PCT]):
        with rasterio.open(DEM_SOURCE) as src:
            dem = np.full((height30, width30), NODATA_FLOAT, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dem,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform30,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA_FLOAT,
                resampling=Resampling.bilinear,
            )
        write_raster(DEM_UTM, dem, transform30, TARGET_CRS, NODATA_FLOAT, "float32")

        valid = dem != NODATA_FLOAT
        work = dem.astype(np.float64)
        if not valid.all():
            nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            work[~valid] = work[tuple(nearest[:, ~valid])]
        dz_dy, dz_dx = np.gradient(work, 30.0, 30.0)
        rise_run = np.hypot(dz_dx, dz_dy)
        slope_pct = (rise_run * 100.0).astype(np.float32)
        slope_deg = np.degrees(np.arctan(rise_run)).astype(np.float32)
        slope_pct[~valid] = NODATA_FLOAT
        slope_deg[~valid] = NODATA_FLOAT
        write_raster(SLOPE_DEG, slope_deg, transform30, TARGET_CRS, NODATA_FLOAT, "float32")
        write_raster(SLOPE_PCT, slope_pct, transform30, TARGET_CRS, NODATA_FLOAT, "float32")

    if OSM_WATER_UTM_10.exists() and OSM_LAND_FRACTION_30.exists():
        return

    water10 = features.rasterize(
        [(geom, 1) for geom in osm_water.geometry if geom is not None and not geom.is_empty],
        out_shape=(height10, width10),
        transform=transform10,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    write_raster(OSM_WATER_UTM_10, water10, transform10, TARGET_CRS, 255, "uint8")
    land10 = (1 - water10).astype(np.float32)
    land30 = np.zeros((height30, width30), dtype=np.float32)
    reproject(
        source=land10,
        destination=land30,
        src_transform=transform10,
        src_crs=TARGET_CRS,
        dst_transform=transform30,
        dst_crs=TARGET_CRS,
        dst_nodata=0,
        resampling=Resampling.average,
    )
    write_raster(OSM_LAND_FRACTION_30, land30, transform30, TARGET_CRS, 0, "float32")


def clamp_window(dataset, window: Window) -> Window:
    full = Window(0, 0, dataset.width, dataset.height)
    return window.round_offsets().round_lengths().intersection(full)


def window_for_bounds(dataset, bounds) -> Window:
    window = window_from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
    padded = Window(window.col_off - 1, window.row_off - 1, window.width + 2, window.height + 2)
    return clamp_window(dataset, padded)


def identify_lake_geometry(osm_water: gpd.GeoDataFrame, x: float, y: float):
    point = Point(x, y)
    covering = osm_water[osm_water.geometry.covers(point)].copy()
    if len(covering):
        covering["_area"] = covering.geometry.area
        chosen = covering.sort_values("_area", ascending=False).iloc[0]
        nearest_distance = 0.0
        selection_rule = "sampling point inside OSM open-water polygon; largest covering polygon selected"
    else:
        distances = osm_water.geometry.distance(point)
        idx = distances.idxmin()
        nearest_distance = float(distances.loc[idx])
        if nearest_distance > 100.0:
            raise RuntimeError(f"No OSM open-water polygon within 100 m of Lake point at {x:.2f}, {y:.2f}")
        chosen = osm_water.loc[idx]
        selection_rule = "nearest OSM open-water polygon within 100 m"
    lake = chosen.geometry.buffer(0).simplify(1.0, preserve_topology=True)
    return lake, {
        "osm_id": str(chosen.osm_id),
        "osm_fclass": str(chosen.fclass),
        "osm_name": None if pd.isna(chosen["name"]) else str(chosen["name"]),
        "nearest_water_distance_m": nearest_distance,
        "component_area_km2": float(lake.area / 1_000_000.0),
        "component_perimeter_km": float(lake.length / 1000.0),
        "point_inside_component": bool(lake.covers(point)),
        "covering_polygon_count": int(len(covering)),
        "selection_rule": selection_rule,
        "match_quality": "pass",
    }


def read_geom_arrays(geom, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds):
    window = window_for_bounds(dem_ds, geom.bounds)
    dem = dem_ds.read(1, window=window)
    slope_deg = slope_deg_ds.read(1, window=window)
    slope_pct = slope_pct_ds.read(1, window=window)
    land_fraction = land_fraction_ds.read(1, window=window)
    transform = rasterio.windows.transform(window, dem_ds.transform)
    inside = features.geometry_mask([mapping(geom)], out_shape=dem.shape, transform=transform, invert=True, all_touched=True)
    valid_dem = dem != dem_ds.nodata
    valid_slope = slope_deg != slope_deg_ds.nodata
    valid = inside & valid_dem & valid_slope & (land_fraction > 0)
    return dem, slope_deg, slope_pct, land_fraction, inside, valid


def summarize_geometry(geom, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds):
    dem, slope_deg, slope_pct, land_fraction, inside, valid = read_geom_arrays(geom, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds)
    if not valid.any():
        raise RuntimeError("No valid non-water DEM pixels in analysis geometry")
    z = dem[valid].astype(float)
    sd = slope_deg[valid].astype(float)
    sp = slope_pct[valid].astype(float)
    weights = land_fraction[valid].astype(float)
    weight_sum = float(weights.sum())
    mean_z = float(np.average(z, weights=weights))
    mean_sd = float(np.average(sd, weights=weights))
    mean_sp = float(np.average(sp, weights=weights))
    n_geom = int(inside.sum())
    n_land = int(valid.sum())
    return {
        "mean_dem_m": mean_z,
        "min_dem_m": float(np.min(z)),
        "max_dem_m": float(np.max(z)),
        "dem_range_m": float(np.max(z) - np.min(z)),
        "dem_std_m": float(np.sqrt(np.average((z - mean_z) ** 2, weights=weights))),
        "dem_var_m2": float(np.average((z - mean_z) ** 2, weights=weights)),
        "mean_slope_deg": mean_sd,
        "max_slope_deg": float(np.max(sd)),
        "mean_slope_pct": mean_sp,
        "max_slope_pct": float(np.max(sp)),
        "slope_gt10pct_area_pct": float(np.average(sp > 10.0, weights=weights) * 100.0),
        "slope_gt15pct_area_pct": float(np.average(sp > 15.0, weights=weights) * 100.0),
        "n_geometry_pixels": n_geom,
        "n_land_valid_pixels": n_land,
        "n_water_excluded_pixels": int((inside & (land_fraction == 0)).sum()),
        "n_partial_water_pixels": int((inside & (land_fraction > 0) & (land_fraction < 1)).sum()),
        "valid_land_fraction": float(weight_sum / n_geom if n_geom else np.nan),
        "land_area_km2": float(weight_sum * 30.0 * 30.0 / 1_000_000.0),
    }


def sample_nearest(dataset, x: float, y: float) -> float:
    value = next(dataset.sample([(x, y)]))[0]
    return float(value) if value != dataset.nodata else np.nan


def nearest_land_slope(x, y, slope_deg_ds, slope_pct_ds, land_fraction_ds, radius=600):
    window = window_for_bounds(land_fraction_ds, (x - radius, y - radius, x + radius, y + radius))
    sd = slope_deg_ds.read(1, window=window)
    sp = slope_pct_ds.read(1, window=window)
    land_fraction = land_fraction_ds.read(1, window=window)
    rows, cols = np.where((land_fraction > 0) & (sd != slope_deg_ds.nodata))
    if len(rows) == 0:
        raise RuntimeError(f"No land slope pixel within {radius} m")
    transform = rasterio.windows.transform(window, land_fraction_ds.transform)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    dist2 = (np.asarray(xs) - x) ** 2 + (np.asarray(ys) - y) ** 2
    idx = int(np.argmin(dist2))
    return float(sd[rows[idx], cols[idx]]), float(sp[rows[idx], cols[idx]]), float(math.sqrt(dist2[idx]))


def boundary_reference(lake_geom, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds):
    for width in [30.0, 60.0, 90.0]:
        band = lake_geom.buffer(width).difference(lake_geom)
        dem, sd, sp, land_fraction, _, valid = read_geom_arrays(band, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds)
        if int(valid.sum()) >= 5:
            weights = land_fraction[valid].astype(float)
            return {
                "boundary_band_width_m": width,
                "boundary_land_pixels": int(valid.sum()),
                "boundary_mean_dem_m": float(np.average(dem[valid], weights=weights)),
                "boundary_mean_slope_deg": float(np.average(sd[valid], weights=weights)),
                "boundary_mean_slope_pct": float(np.average(sp[valid], weights=weights)),
            }
    raise RuntimeError("No valid land pixels along Lake boundary")


def feature_names():
    names = ["terrain_point_dem_m", "terrain_point_slope_pct"]
    suffixes = [
        "mean_dem_m",
        "dem_range_m",
        "dem_std_m",
        "dem_var_m2",
        "delta_dem_buffer_minus_point_m",
        "mean_slope_pct",
        "max_slope_pct",
        "slope_gt10pct_area_pct",
        "slope_gt15pct_area_pct",
    ]
    for scale in SCALES:
        names.extend([f"terrain_{scale}m_{suffix}" for suffix in suffixes])
    return names


def build_feature_dictionary(names):
    rows = []
    for name in names:
        if name == "terrain_point_dem_m":
            description = "ASTER DEM value at sampling coordinate"
        elif name == "terrain_point_slope_pct":
            description = "Lake: OSM shoreline mean land-side percent slope; River: nearest OSM-defined land pixel percent slope"
        else:
            description = "Non-water terrain statistic in cumulative River point buffer or Lake shoreline outward ring"
        rows.append({
            "column_name": name,
            "role": "predictor",
            "taxon_group": None,
            "feature_category": "terrain_dem_slope",
            "model_include": True,
            "source_basis": f"{description}; ASTER ASTGTMV003 + Geofabrik OpenStreetMap Guangdong water polygons dated 2026-04-15",
            "sample_special_rule": "static site feature replicated to every sample record at the same site",
            "site_special_rule": "one value per independent sampling site",
            "missing_rule": "no imputation; any missing value is a processing failure requiring review",
        })
    return rows


def load_osm_water(coords: pd.DataFrame) -> gpd.GeoDataFrame:
    bbox = (
        float(coords.longitude.min() - 0.12),
        float(coords.latitude.min() - 0.12),
        float(coords.longitude.max() + 0.12),
        float(coords.latitude.max() + 0.12),
    )
    water = gpd.read_file(OSM_WATER_SOURCE, bbox=bbox)
    water = water[water.fclass.isin(OSM_OPEN_WATER_CLASSES)].copy()
    water = water[water.geometry.notna() & ~water.geometry.is_empty].copy()
    water = water.to_crs(TARGET_CRS)
    water.geometry = water.geometry.buffer(0)
    water = water[water.geometry.notna() & ~water.geometry.is_empty]
    if len(water) == 0:
        raise RuntimeError("No OSM open-water polygons intersect the study extent")
    return water


def main():
    coords = pd.read_excel(COORD_XLSX, sheet_name="Location").iloc[:, :4].copy()
    coords.columns = ["site", "longitude", "latitude", "water_body_type"]
    coords["site"] = coords["site"].astype(str).str.strip()
    coords["water_body_type"] = coords["water_body_type"].astype(str).str.strip()
    if len(coords) != 130 or coords["site"].duplicated().any():
        raise RuntimeError("Coordinate table must contain 130 unique sites")

    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    coords[["x", "y"]] = np.array([transformer.transform(lon, lat) for lon, lat in zip(coords.longitude, coords.latitude)])

    site_matrix = pd.read_excel(SITE_MATRIX, sheet_name="Model_Input_Matrix")
    sample_matrix = pd.read_excel(SAMPLE_MATRIX, sheet_name="Model_Input_Matrix")
    if len(site_matrix) != 130 or len(sample_matrix) != 577:
        raise RuntimeError("Unexpected current matrix row count")
    coord_sites = set(coords.site)
    site_sites = set(site_matrix.site.astype(str))
    sample_sites = set(sample_matrix.site.astype(str))
    if coord_sites != site_sites or coord_sites != sample_sites:
        raise RuntimeError(f"Site mismatch: coord-only={sorted(coord_sites-site_sites)}, matrix-only={sorted(site_sites-coord_sites)}")

    comparison = site_matrix[["site", "longitude", "latitude", "water_body_type"]].merge(coords, on="site", suffixes=("_matrix", "_raw"), validate="1:1")
    comparison["coordinate_difference_m"] = [
        math.hypot(*(
            np.subtract(
                transformer.transform(r.longitude_matrix, r.latitude_matrix),
                transformer.transform(r.longitude_raw, r.latitude_raw),
            )
        )) for r in comparison.itertuples()
    ]
    if comparison.coordinate_difference_m.max() > 1.0:
        raise RuntimeError("Raw coordinates and current matrix coordinates differ by more than 1 m")
    if not (comparison.water_body_type_matrix == comparison.water_body_type_raw).all():
        raise RuntimeError("Water-body type mismatch between coordinate table and current matrix")

    osm_water = load_osm_water(coords)
    prepare_derived_rasters(coords, osm_water)

    point_rows = []
    buffer_rows = []
    lake_rows = []
    wide_rows = []
    with rasterio.open(DEM_UTM) as dem_ds, rasterio.open(SLOPE_DEG) as slope_deg_ds, rasterio.open(SLOPE_PCT) as slope_pct_ds, rasterio.open(OSM_LAND_FRACTION_30) as land_fraction_ds:
        for record in coords.sort_values(["water_body_type", "site"]).itertuples(index=False):
            x, y = float(record.x), float(record.y)
            point_dem = sample_nearest(dem_ds, x, y)
            lake_geom = None
            lake_qa = None
            if record.water_body_type == "Lake":
                lake_geom, lake_qa = identify_lake_geometry(osm_water, x, y)
                lake_geometry_usable = True
                boundary = boundary_reference(lake_geom, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds)
                point_slope_deg = boundary["boundary_mean_slope_deg"]
                point_slope_pct = boundary["boundary_mean_slope_pct"]
                reference_distance = 0.0
                point_rule = "OSM lake shoreline outer land band mean percent slope"
                lake_mode = "osm_shoreline_ring"
                lake_rows.append({"site": record.site, "longitude": record.longitude, "latitude": record.latitude, "analysis_geometry_mode": lake_mode, **lake_qa, **boundary})
            else:
                lake_geometry_usable = False
                point_slope_deg, point_slope_pct, reference_distance = nearest_land_slope(x, y, slope_deg_ds, slope_pct_ds, land_fraction_ds)
                boundary = {}
                point_rule = "nearest land pixel after excluding OSM open-water polygons"
                lake_mode = None

            site_special_id = f"{record.water_body_type}_{record.site}"
            point_row = {
                "site_special_id": site_special_id,
                "water_body_type": record.water_body_type,
                "site": record.site,
                "longitude": float(record.longitude),
                "latitude": float(record.latitude),
                "utm_x_m": x,
                "utm_y_m": y,
                "point_dem_m": point_dem,
                "point_slope_deg": point_slope_deg,
                "point_slope_pct": point_slope_pct,
                "point_slope_reference_distance_m": reference_distance,
                "point_slope_rule": point_rule,
                "lake_boundary_mean_dem_m": boundary.get("boundary_mean_dem_m"),
                "lake_water_component_area_km2": None if lake_qa is None else lake_qa["component_area_km2"],
                "analysis_geometry_mode": "river_point_circle" if record.water_body_type == "River" else lake_mode,
            }
            point_rows.append(point_row)
            wide = {
                "site_special_id": site_special_id,
                "water_body_type": record.water_body_type,
                "site": record.site,
                "longitude": float(record.longitude),
                "latitude": float(record.latitude),
                "terrain_point_dem_m": point_dem,
                "terrain_point_slope_pct": point_slope_pct,
            }

            previous_n_land = -1
            for scale in SCALES:
                if record.water_body_type == "Lake" and lake_geometry_usable:
                    geom = lake_geom.buffer(scale).difference(lake_geom)
                    geometry_rule = "OSM open-water polygon shoreline outward ring; lake body excluded"
                else:
                    geom = Point(x, y).buffer(scale, resolution=48)
                    geometry_rule = "point-centred cumulative circle"
                try:
                    stats = summarize_geometry(geom, dem_ds, slope_deg_ds, slope_pct_ds, land_fraction_ds)
                except RuntimeError as exc:
                    raise RuntimeError(f"{record.water_body_type} {record.site} at {scale} m: {exc}") from exc
                stats["delta_dem_buffer_minus_point_m"] = stats["mean_dem_m"] - point_dem
                monotonic = stats["n_land_valid_pixels"] >= previous_n_land
                previous_n_land = stats["n_land_valid_pixels"]
                row = {
                    "site_special_id": site_special_id,
                    "water_body_type": record.water_body_type,
                    "site": record.site,
                    "longitude": float(record.longitude),
                    "latitude": float(record.latitude),
                    "scale_m": scale,
                    "geometry_rule": geometry_rule,
                    "point_dem_m": point_dem,
                    "point_slope_deg": point_slope_deg,
                    "point_slope_pct": point_slope_pct,
                    "point_slope_reference_distance_m": reference_distance,
                    "cumulative_land_pixel_count_monotonic": monotonic,
                    **stats,
                }
                buffer_rows.append(row)
                ml_map = {
                    "mean_dem_m": stats["mean_dem_m"],
                    "dem_range_m": stats["dem_range_m"],
                    "dem_std_m": stats["dem_std_m"],
                    "dem_var_m2": stats["dem_var_m2"],
                    "delta_dem_buffer_minus_point_m": stats["delta_dem_buffer_minus_point_m"],
                    "mean_slope_pct": stats["mean_slope_pct"],
                    "max_slope_pct": stats["max_slope_pct"],
                    "slope_gt10pct_area_pct": stats["slope_gt10pct_area_pct"],
                    "slope_gt15pct_area_pct": stats["slope_gt15pct_area_pct"],
                }
                for suffix, value in ml_map.items():
                    wide[f"terrain_{scale}m_{suffix}"] = value
            wide_rows.append(wide)

    point_df = pd.DataFrame(point_rows)
    buffer_df = pd.DataFrame(buffer_rows)
    lake_df = pd.DataFrame(lake_rows)
    site_wide = pd.DataFrame(wide_rows)
    expected_features = feature_names()
    if list(site_wide.columns[5:]) != expected_features:
        raise RuntimeError("Terrain feature ordering mismatch")
    if site_wide[expected_features].isna().any().any():
        missing = int(site_wide[expected_features].isna().sum().sum())
        raise RuntimeError(f"Terrain site database contains {missing} missing cells")

    sample_wide = sample_matrix[["sample_id", "water_body_type", "site", "longitude", "latitude"]].merge(
        site_wide[["site", *expected_features]], on="site", how="left", validate="m:1"
    )
    if sample_wide[expected_features].isna().any().any():
        raise RuntimeError("Terrain sample database contains missing cells")

    monotonic_violations = int((~buffer_df.cumulative_land_pixel_count_monotonic).sum())
    low_land = buffer_df[buffer_df.valid_land_fraction < 0.10]
    qa = {
        "created_date": "2026-06-29",
        "coordinate_rows": int(len(coords)),
        "coordinate_unique_sites": int(coords.site.nunique()),
        "site_matrix_rows": int(len(site_matrix)),
        "sample_matrix_rows": int(len(sample_matrix)),
        "terrain_feature_columns": int(len(expected_features)),
        "site_database_rows": int(len(site_wide)),
        "sample_database_rows": int(len(sample_wide)),
        "site_missing_feature_cells": int(site_wide[expected_features].isna().sum().sum()),
        "sample_missing_feature_cells": int(sample_wide[expected_features].isna().sum().sum()),
        "maximum_coordinate_difference_m": float(comparison.coordinate_difference_m.max()),
        "lake_sites": int((coords.water_body_type == "Lake").sum()),
        "river_sites": int((coords.water_body_type == "River").sum()),
        "lake_max_nearest_water_distance_m": float(lake_df.nearest_water_distance_m.max()),
        "lake_points_inside_osm_water_polygon": int(lake_df.point_inside_component.sum()),
        "lake_osm_match_review_sites": int((lake_df.match_quality == "review").sum()),
        "lake_osm_match_review_site_ids": ";".join(lake_df.loc[lake_df.match_quality == "review", "site"].astype(str)),
        "osm_open_water_polygons_in_processing_extent": int(len(osm_water)),
        "buffer_rows": int(len(buffer_df)),
        "land_pixel_monotonic_violations": monotonic_violations,
        "buffer_rows_land_fraction_below_0_10": int(len(low_land)),
        "sites_with_land_fraction_below_0_10": ";".join(sorted(low_land.site.astype(str).unique())),
        "coordinate_source_sha256": sha256(COORD_SOURCE),
        "dem_source_sha256": sha256(DEM_SOURCE),
        "osm_water_shp_sha256": sha256(OSM_WATER_SOURCE),
        "osm_water_dbf_sha256": sha256(OSM_WATER_SOURCE.with_suffix(".dbf")),
    }
    if monotonic_violations:
        raise RuntimeError(f"Found {monotonic_violations} cumulative pixel-count violations")

    metadata = []
    for path, role in [
        (DEM_SOURCE, "source_dem"),
        (DEM_UTM, "derived_dem"),
        (SLOPE_DEG, "derived_slope_degrees"),
        (SLOPE_PCT, "derived_slope_percent"),
        (OSM_WATER_UTM_10, "derived_osm_water_10m"),
        (OSM_LAND_FRACTION_30, "derived_osm_land_fraction_dem_aligned"),
    ]:
        with rasterio.open(path) as ds:
            metadata.append({
                "role": role,
                "path": str(path),
                "driver": ds.driver,
                "crs": str(ds.crs),
                "width": ds.width,
                "height": ds.height,
                "resolution_x": float(ds.res[0]),
                "resolution_y": float(ds.res[1]),
                "nodata": ds.nodata,
                "bounds_left": float(ds.bounds.left),
                "bounds_bottom": float(ds.bounds.bottom),
                "bounds_right": float(ds.bounds.right),
                "bounds_top": float(ds.bounds.top),
            })
    metadata.append({
        "role": "source_osm_water_polygon",
        "path": str(OSM_WATER_SOURCE),
        "driver": "ESRI Shapefile",
        "crs": TARGET_CRS,
        "width": None,
        "height": None,
        "resolution_x": None,
        "resolution_y": None,
        "nodata": None,
        "bounds_left": float(osm_water.total_bounds[0]),
        "bounds_bottom": float(osm_water.total_bounds[1]),
        "bounds_right": float(osm_water.total_bounds[2]),
        "bounds_top": float(osm_water.total_bounds[3]),
    })

    notes = [
        {"item": "projection", "detail": "All distance, area, buffer and slope calculations use WGS 84 / UTM zone 49N (EPSG:32649)."},
        {"item": "slope", "detail": "Rise/run is calculated from 30 m projected DEM gradients. slope_pct=rise_run*100; slope_deg=atan(rise_run)*180/pi. ML uses percent slope, and steep-area thresholds follow the methodology at >10% and >15%."},
        {"item": "water", "detail": "OSM polygon classes water, reservoir, riverbank and dock define open water. The source is Geofabrik Guangdong dated 2026-04-15. OSM water is rasterized at 10 m and aggregated to 30 m land fraction."},
        {"item": "river_geometry", "detail": "River uses point-centred cumulative circles at 50-6000 m. Point slope is the nearest land pixel after excluding OSM open-water polygons."},
        {"item": "lake_geometry", "detail": "Lake uses the OSM open-water polygon covering the sampling point. Analysis geometry is buffer(lake, radius)-lake. Point slope is the mean percent slope in the land-side shoreline band, starting at 30 m and expanding to 60/90 m only when fewer than five valid pixels occur."},
        {"item": "lake_matching", "detail": "All 44 current Lake points fall inside an OSM open-water polygon; no point-circle fallback is used. Processing fails if a future Lake point has no OSM water polygon within 100 m."},
        {"item": "pixel_rule", "detail": "DEM and slope pixels intersecting the geometry are included with all_touched=True. Statistics use non-water valid pixels only."},
        {"item": "variance", "detail": "Population variance and standard deviation are calculated across valid terrain pixels (ddof=0)."},
        {"item": "delta_dem", "detail": "delta_dem_buffer_minus_point_m = mean_dem_buffer_m - point_dem_m."},
        {"item": "static_replication", "detail": "Terrain is site-static. Site values are copied to all Sample Special records sharing the same site."},
    ]

    feature_dict = pd.DataFrame(build_feature_dictionary(expected_features))
    comparison.to_csv(WORK_DIR / "coordinate_comparison.csv", index=False, encoding="utf-8-sig")
    point_df.to_csv(WORK_DIR / "site_point_process.csv", index=False, encoding="utf-8-sig")
    buffer_df.to_csv(WORK_DIR / "site_buffer_process_long.csv", index=False, encoding="utf-8-sig")
    lake_df.to_csv(WORK_DIR / "lake_geometry_qa.csv", index=False, encoding="utf-8-sig")
    site_wide.to_csv(WORK_DIR / "site_feature_wide.csv", index=False, encoding="utf-8-sig")
    sample_wide.to_csv(WORK_DIR / "sample_feature_wide.csv", index=False, encoding="utf-8-sig")
    sample_wide[["sample_id", "water_body_type", "site", "longitude", "latitude"]].to_csv(WORK_DIR / "sample_site_mapping.csv", index=False, encoding="utf-8-sig")
    feature_dict.to_csv(WORK_DIR / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metadata).to_csv(WORK_DIR / "raster_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(notes).to_csv(WORK_DIR / "method_notes.csv", index=False, encoding="utf-8-sig")
    with (WORK_DIR / "qa.json").open("w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
