"""Prepare the harmonised freshwater economy--ecology analysis tables.

Primary unit choices:
* GRQA water quality: station-year median (1990--2015).
* GLORICH electrical conductivity: station-year median (existing audited table).
* Fish: Freshwater Ecoregions of the World.
* Stream insects: HydroBASINS level-4 units from Liu et al. (2025).
* Algae: lake-level median chlorophyll-a from Filazzola et al. (2020).

The Kummu et al. 5-arc-min GDP-per-capita surface is the common exposure.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_原始数据"
AUDIT = ROOT / "02_数据审计"
CLEAN = ROOT / "03_清洗与匹配"
SOURCE = ROOT / "06_SourceData"
QA = ROOT / "09_QA"

OLD = ROOT.parent / "18_全球淡水经济发展与水生态宏观分析_2026-09-14"
OLD_REV = OLD / "11_修缮版_多类群与匹配增强"
GDP_NC = OLD / "01_原始数据" / "GDP_per_capita_PPP_1990_2015_v2.nc"
GDP_MEAN_TIF = OLD / "03_清洗与匹配" / "gdppc_mean_2010_2015_5arcmin.tif"
WORLD_SHP = OLD / "01_原始数据" / "NaturalEarth_110m" / "ne_110m_admin_0_countries.shp"
EC_TABLE = OLD_REV / "03_清洗与匹配" / "station_year_water_quality_expanded.csv.gz"
FRESHVERTS_GPKG = OLD_REV / "03_清洗与匹配" / "freshverts_ecoregion_economy.gpkg"


def ensure_dirs() -> None:
    for p in (AUDIT, CLEAN, SOURCE, QA):
        p.mkdir(parents=True, exist_ok=True)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP).to_crs(4326)
    world = world.loc[world["ADMIN"] != "Antarctica"].copy()
    world["country_code"] = world["WB_A3"].where(
        world["WB_A3"].notna() & (world["WB_A3"] != "-99"), world["ISO_A3"]
    )
    world["country_code"] = world["country_code"].replace("-99", np.nan)
    return world


def assign_country(frame: pd.DataFrame, world: gpd.GeoDataFrame,
                   lon: str = "longitude", lat: str = "latitude") -> pd.DataFrame:
    base = frame.reset_index(drop=True).copy()
    base["_row_id"] = np.arange(len(base))
    pts = gpd.GeoDataFrame(
        base, geometry=gpd.points_from_xy(base[lon], base[lat]), crs=4326
    )
    joined = gpd.sjoin(
        pts,
        world[["country_code", "ADMIN", "CONTINENT", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.sort_values("_row_id").drop_duplicates("_row_id")
    joined = joined.set_index("_row_id").reindex(base["_row_id"])
    out = base.drop(columns="_row_id")
    out["country_code"] = joined["country_code"].to_numpy()
    out["country_name"] = joined["ADMIN"].to_numpy()
    out["continent"] = joined["CONTINENT"].fillna("Unassigned").to_numpy()
    return out


def extract_gdp_points(frame: pd.DataFrame, lat_col: str, lon_col: str,
                       year_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Sample yearly GDPpc and use a 5 x 5-cell coastal fallback if necessary."""
    result = np.full(len(frame), np.nan)
    fallback = np.zeros(len(frame), dtype=bool)
    with h5py.File(GDP_NC, "r") as h5:
        lat = h5["latitude"][:]
        lon = h5["longitude"][:]
        years = h5["time"][:].astype(int)
        row = np.clip(
            np.rint((float(lat[0]) - frame[lat_col].to_numpy(float)) /
                    abs(float(lat[1] - lat[0]))).astype(int), 0, len(lat) - 1
        )
        col = np.clip(
            np.rint((frame[lon_col].to_numpy(float) - float(lon[0])) /
                    abs(float(lon[1] - lon[0]))).astype(int), 0, len(lon) - 1
        )
        year_values = frame[year_col].to_numpy()
        for year in np.unique(frame[year_col].dropna().astype(int)):
            if year not in years:
                continue
            positions = np.where(year_values == year)[0]
            layer = h5["GDP_per_capita_PPP"][int(np.where(years == year)[0][0]), :, :]
            vals = layer[row[positions], col[positions]].astype(float)
            bad = (~np.isfinite(vals)) | (vals <= 0)
            for local in np.where(bad)[0]:
                pos = positions[local]
                r0, c0 = row[pos], col[pos]
                window = layer[
                    max(0, r0 - 2):min(layer.shape[0], r0 + 3),
                    max(0, c0 - 2):min(layer.shape[1], c0 + 3),
                ]
                valid = window[np.isfinite(window) & (window > 0)]
                if valid.size:
                    vals[local] = float(np.median(valid))
                    fallback[pos] = True
            vals[(~np.isfinite(vals)) | (vals <= 0)] = np.nan
            result[positions] = vals
    return result, fallback


def sample_mean_gdp(frame: pd.DataFrame, lon_col: str, lat_col: str) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(len(frame), np.nan)
    fallback = np.zeros(len(frame), dtype=bool)
    with rasterio.open(GDP_MEAN_TIF) as src:
        coords = list(zip(frame[lon_col].astype(float), frame[lat_col].astype(float)))
        vals = np.array([x[0] for x in src.sample(coords)], dtype=float)
        nodata = src.nodata
        bad = (~np.isfinite(vals)) | (vals <= 0)
        if nodata is not None:
            bad |= vals == nodata
        for i in np.where(bad)[0]:
            r, c = src.index(coords[i][0], coords[i][1])
            window = src.read(
                1,
                window=rasterio.windows.Window(max(0, c - 2), max(0, r - 2), 5, 5),
                boundless=True,
                fill_value=src.nodata,
            ).astype(float)
            valid = window[np.isfinite(window) & (window > 0)]
            if nodata is not None:
                valid = valid[valid != nodata]
            if valid.size:
                vals[i] = float(np.median(valid))
                fallback[i] = True
        vals[(~np.isfinite(vals)) | (vals <= 0)] = np.nan
        result[:] = vals
    return result, fallback


def prepare_grqa(path: Path, endpoint: str, lower: float, upper: float,
                 world: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "obs_id", "lat_wgs84", "lon_wgs84", "obs_date", "site_id",
        "site_name", "site_country", "obs_value", "detection_limit_flag",
        "unit", "source", "obs_iqr_outlier",
    ]
    kept = []
    total = in_period = valid_coord = valid_numeric = plausible = 0
    outlier_count = detection_count = 0
    source_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(
        path, sep=";", usecols=usecols, chunksize=250_000,
        low_memory=False, encoding="utf-8",
    ):
        total += len(chunk)
        chunk["year"] = pd.to_numeric(chunk["obs_date"].astype(str).str[:4], errors="coerce")
        chunk["latitude"] = pd.to_numeric(chunk["lat_wgs84"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["lon_wgs84"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["obs_value"], errors="coerce")
        period = chunk["year"].between(1990, 2015)
        in_period += int(period.sum())
        coord = chunk["latitude"].between(-60, 90) & chunk["longitude"].between(-180, 180)
        valid_coord += int((period & coord).sum())
        numeric = np.isfinite(chunk["value"])
        valid_numeric += int((period & coord & numeric).sum())
        physical = chunk["value"].between(lower, upper)
        mask = period & coord & numeric & physical
        plausible += int(mask.sum())
        selected = chunk.loc[mask].copy()
        outlier_count += int(selected["obs_iqr_outlier"].astype(str).str.lower().eq("yes").sum())
        detection_count += int(selected["detection_limit_flag"].notna().sum())
        source_counts.update(selected["source"].fillna("Unknown").astype(str))
        unit_counts.update(selected["unit"].fillna("Unknown").astype(str))
        kept.append(selected)

    data = pd.concat(kept, ignore_index=True)
    data["source"] = data["source"].fillna("Unknown").astype(str)
    data["site_id"] = data["site_id"].astype(str).str.strip()
    data["unit_id"] = "GRQA_" + data["source"] + "_" + data["site_id"]
    data = data.drop_duplicates("obs_id")
    # GRQA has already de-duplicated cross-source identical observations. We retain
    # flagged distributional outliers after hard physical bounds and use medians.
    annual = data.groupby(["unit_id", "year"], as_index=False).agg(
        value=("value", "median"),
        n_observations=("value", "size"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        site_name=("site_name", "first"),
        source_country=("site_country", "first"),
        source=("source", "first"),
        outlier_fraction=("obs_iqr_outlier", lambda s: float(s.astype(str).str.lower().eq("yes").mean())),
        detection_limit_fraction=("detection_limit_flag", lambda s: float(s.notna().mean())),
    )
    annual["year"] = annual["year"].astype(int)
    annual["endpoint"] = endpoint
    annual = assign_country(annual, world)
    annual["gdppc"], annual["gdp_coastal_fallback"] = extract_gdp_points(
        annual, "latitude", "longitude", "year"
    )
    annual["spatial_block_10"] = (
        np.floor((annual["longitude"] + 180) / 10).astype(int).astype(str)
        + "_" + np.floor((annual["latitude"] + 90) / 10).astype(int).astype(str)
    )
    annual = annual.loc[annual["gdppc"].notna()].copy()
    audit = {
        "endpoint": endpoint,
        "raw_observations": total,
        "observations_1990_2015": in_period,
        "with_valid_coordinates": valid_coord,
        "with_numeric_values": valid_numeric,
        "within_physical_bounds": plausible,
        "flagged_iqr_outliers_retained": outlier_count,
        "detection_limit_flagged_retained": detection_count,
        "station_years_with_gdp": len(annual),
        "stations_with_gdp": int(annual["unit_id"].nunique()),
        "countries_spatially_assigned": int(annual["country_code"].nunique()),
        "sources": json.dumps(source_counts, ensure_ascii=False),
        "units": json.dumps(unit_counts, ensure_ascii=False),
        "hard_bounds": f"{lower} to {upper}",
    }
    return annual, audit


def prepare_ec(world: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    cols = [
        "STAT_ID", "year", "SpecCond25C", "n_SpecCond25C", "longitude", "latitude",
        "gdppc", "gdp_coastal_fallback", "coordinate_source",
    ]
    d = pd.read_csv(EC_TABLE, usecols=cols, low_memory=False)
    raw_rows = len(d)
    d["value"] = pd.to_numeric(d["SpecCond25C"], errors="coerce")
    d = d.loc[
        d["year"].between(1990, 2015)
        & d["value"].between(0.1, 200_000)
        & d["gdppc"].gt(0)
        & d["latitude"].between(-60, 90)
        & d["longitude"].between(-180, 180)
    ].copy()
    d["unit_id"] = "GLORICH_" + d["STAT_ID"].astype(str)
    d["endpoint"] = "EC"
    d["source"] = "GLORICH"
    d["n_observations"] = d["n_SpecCond25C"]
    d["outlier_fraction"] = np.nan
    d["detection_limit_fraction"] = np.nan
    d["site_name"] = d["STAT_ID"].astype(str)
    d["source_country"] = np.nan
    d = assign_country(d, world)
    d["spatial_block_10"] = (
        np.floor((d["longitude"] + 180) / 10).astype(int).astype(str)
        + "_" + np.floor((d["latitude"] + 90) / 10).astype(int).astype(str)
    )
    keep = [
        "unit_id", "year", "value", "n_observations", "latitude", "longitude",
        "site_name", "source_country", "source", "outlier_fraction",
        "detection_limit_fraction", "endpoint", "country_code", "country_name",
        "continent", "gdppc", "gdp_coastal_fallback", "spatial_block_10",
        "coordinate_source",
    ]
    d = d[keep].drop_duplicates(["unit_id", "year"])
    audit = {
        "endpoint": "EC",
        "raw_observations": raw_rows,
        "observations_1990_2015": int(len(d)),
        "with_valid_coordinates": int(len(d)),
        "with_numeric_values": int(len(d)),
        "within_physical_bounds": int(len(d)),
        "flagged_iqr_outliers_retained": np.nan,
        "detection_limit_flagged_retained": np.nan,
        "station_years_with_gdp": int(len(d)),
        "stations_with_gdp": int(d["unit_id"].nunique()),
        "countries_spatially_assigned": int(d["country_code"].nunique()),
        "sources": json.dumps({"GLORICH": len(d)}),
        "units": json.dumps({"uS/cm at 25 C": len(d)}),
        "hard_bounds": "0.1 to 200000",
    }
    return d, audit


def prepare_fish(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fish = gpd.read_file(FRESHVERTS_GPKG)
    fish = fish.loc[fish["fish_richness"].notna() & fish["gdppc"].gt(0)].copy()
    fish["unit_id"] = "FEOW_" + fish["FEOW_ID"].astype(str)
    fish["endpoint"] = "Fish"
    fish["value"] = fish["fish_richness"].astype(float)
    cent = fish.to_crs("ESRI:54030").geometry.centroid
    cent = gpd.GeoSeries(cent, crs="ESRI:54030").to_crs(4326)
    fish["longitude"] = cent.x.to_numpy()
    fish["latitude"] = cent.y.to_numpy()
    assigned = assign_country(pd.DataFrame(fish.drop(columns="geometry")), world)
    for col in ("country_code", "country_name", "continent"):
        fish[col] = assigned[col].to_numpy()
    fish["spatial_block_20"] = (
        np.floor((fish["longitude"] + 180) / 20).astype(int).astype(str)
        + "_" + np.floor((fish["latitude"] + 90) / 20).astype(int).astype(str)
    )
    return fish


def prepare_epto(world: gpd.GeoDataFrame) -> pd.DataFrame:
    d = pd.read_csv(RAW / "Liu2025_standardized_stream_insect_diversity.csv")
    numeric = [
        "TDq0", "Bio1", "Bio12", "Elev_range", "Area", "Sam_eff", "Time span",
        "Longitude", "Latitude",
    ]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.loc[
        d["TDq0"].gt(0) & d["Longitude"].between(-180, 180)
        & d["Latitude"].between(-60, 90)
    ].copy()
    d = d.rename(columns={"Longitude": "longitude", "Latitude": "latitude"})
    d["unit_id"] = "HYBAS4_" + d["HYBAS_ID"].astype(str)
    d["endpoint"] = "EPTO"
    d["value"] = d["TDq0"]
    d["gdppc"], d["gdp_coastal_fallback"] = sample_mean_gdp(d, "longitude", "latitude")
    d = d.loc[d["gdppc"].notna()].copy()
    d = assign_country(d, world)
    d["spatial_block_20"] = (
        np.floor((d["longitude"] + 180) / 20).astype(int).astype(str)
        + "_" + np.floor((d["latitude"] + 90) / 20).astype(int).astype(str)
    )
    return d


def prepare_algae(world: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = pd.read_csv(RAW / "Filazzola_ChlaData_final.csv", low_memory=False)
    for col in ["Year", "Lat", "Lon", "ChlValues", "SurfaceArea", "Depth.mean", "Depth.max"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    raw_rows = len(d)
    d = d.loc[
        d["Year"].between(1990, 2015) & d["ChlValues"].gt(0)
        & d["Lat"].between(-60, 90) & d["Lon"].between(-180, 180)
    ].copy()
    d = d.rename(columns={
        "Year": "year", "Lat": "latitude", "Lon": "longitude",
        "ChlValues": "value", "SurfaceArea": "surface_area",
        "Depth.mean": "mean_depth", "Depth.max": "max_depth",
    })
    fallback_id = (
        d["latitude"].round(4).astype(str) + "_" + d["longitude"].round(4).astype(str)
    )
    lake_name = d["UniqueLakeName"].astype("string").str.strip()
    d["unit_id"] = "LAKE_" + lake_name.fillna(fallback_id).astype(str)
    yearly = d.groupby(["unit_id", "year"], as_index=False).agg(
        value=("value", "median"), n_observations=("value", "size"),
        latitude=("latitude", "median"), longitude=("longitude", "median"),
        surface_area=("surface_area", "median"), mean_depth=("mean_depth", "median"),
        max_depth=("max_depth", "median"), study_count=("StudyID", "nunique"),
        below_detection_fraction=("Chla.flag", lambda s: float(s.notna().mean())),
    )
    yearly["year"] = yearly["year"].astype(int)
    yearly["gdppc"], yearly["gdp_coastal_fallback"] = extract_gdp_points(
        yearly, "latitude", "longitude", "year"
    )
    yearly = yearly.loc[yearly["gdppc"].notna()].copy()
    yearly = assign_country(yearly, world)
    lake = yearly.groupby("unit_id", as_index=False).agg(
        value=("value", "median"), n_observations=("n_observations", "sum"),
        n_years=("year", "nunique"), year=("year", "median"),
        latitude=("latitude", "median"), longitude=("longitude", "median"),
        surface_area=("surface_area", "median"), mean_depth=("mean_depth", "median"),
        max_depth=("max_depth", "median"), gdppc=("gdppc", "median"),
        country_code=("country_code", "first"), country_name=("country_name", "first"),
        continent=("continent", "first"),
        below_detection_fraction=("below_detection_fraction", "mean"),
    )
    lake["endpoint"] = "Algae"
    lake["spatial_block_20"] = (
        np.floor((lake["longitude"] + 180) / 20).astype(int).astype(str)
        + "_" + np.floor((lake["latitude"] + 90) / 20).astype(int).astype(str)
    )
    audit = {
        "endpoint": "Algae",
        "raw_observations": raw_rows,
        "observations_1990_2015_positive": int(len(d)),
        "lake_years_with_gdp": int(len(yearly)),
        "lakes_with_gdp": int(len(lake)),
        "countries_spatially_assigned": int(lake["country_code"].nunique()),
        "public_object_note": "Latest public KNB object; fewer public rows than the paper-wide compiled count",
    }
    return yearly, lake, audit


def save_outputs() -> None:
    ensure_dirs()
    world = load_world()

    dosat, a_do = prepare_grqa(RAW / "DOSAT_GRQA.csv", "DOSAT", 0, 200, world)
    tp, a_tp = prepare_grqa(RAW / "TP_GRQA.csv", "TP", 0, 100, world)
    ec, a_ec = prepare_ec(world)
    for name, d in (("DOSAT", dosat), ("TP", tp), ("EC", ec)):
        d.to_csv(CLEAN / f"water_{name}_station_year.csv.gz", index=False, compression="gzip")

    water = pd.concat([dosat, ec, tp], ignore_index=True, sort=False)
    water.to_csv(CLEAN / "water_all_station_year.csv.gz", index=False, compression="gzip")
    site = water.groupby(["endpoint", "unit_id"], as_index=False).agg(
        value=("value", "median"), gdppc=("gdppc", "median"),
        latitude=("latitude", "median"), longitude=("longitude", "median"),
        country_code=("country_code", "first"), country_name=("country_name", "first"),
        continent=("continent", "first"), n_years=("year", "nunique"),
        n_observations=("n_observations", "sum"), source=("source", "first"),
    )
    site.to_csv(CLEAN / "water_site_medians_long.csv.gz", index=False, compression="gzip")
    # De-duplicate shared physical locations across sources/endpoints for GDP-site map e.
    site["location_key"] = (
        site["latitude"].round(3).astype(str) + "_" + site["longitude"].round(3).astype(str)
    )
    all_sites = site.groupby("location_key", as_index=False).agg(
        gdppc=("gdppc", "median"), latitude=("latitude", "median"),
        longitude=("longitude", "median"), country_code=("country_code", "first"),
        country_name=("country_name", "first"), continent=("continent", "first"),
        endpoints=("endpoint", lambda s: "|".join(sorted(set(s)))),
    )
    all_sites.to_csv(CLEAN / "water_all_unique_sites_gdppc.csv.gz", index=False, compression="gzip")

    fish = prepare_fish(world)
    fish.to_file(CLEAN / "fish_units.gpkg", layer="fish", driver="GPKG")
    fish.drop(columns="geometry").to_csv(CLEAN / "fish_units.csv", index=False, encoding="utf-8-sig")
    fish[[
        "unit_id", "FEOW_ID", "all_richness", "amphibia_richness", "reptiles_richness",
        "gdppc", "longitude", "latitude", "country_code", "country_name", "continent",
    ]].to_csv(CLEAN / "supplementary_freshverts_units.csv", index=False, encoding="utf-8-sig")

    epto = prepare_epto(world)
    epto.to_csv(CLEAN / "epto_units.csv", index=False, encoding="utf-8-sig")
    algae_yearly, algae, a_algae = prepare_algae(world)
    algae_yearly.to_csv(CLEAN / "algae_lake_year.csv.gz", index=False, compression="gzip")
    algae.to_csv(CLEAN / "algae_units.csv", index=False, encoding="utf-8-sig")

    coverage = pd.DataFrame([
        {"domain": "water_quality", "endpoint": "DO saturation", "independent_unit": "station", "n_units": dosat.unit_id.nunique(), "n_rows": len(dosat), "n_countries": dosat.country_code.nunique(), "year_min": dosat.year.min(), "year_max": dosat.year.max()},
        {"domain": "water_quality", "endpoint": "EC", "independent_unit": "station", "n_units": ec.unit_id.nunique(), "n_rows": len(ec), "n_countries": ec.country_code.nunique(), "year_min": ec.year.min(), "year_max": ec.year.max()},
        {"domain": "water_quality", "endpoint": "TP", "independent_unit": "station", "n_units": tp.unit_id.nunique(), "n_rows": len(tp), "n_countries": tp.country_code.nunique(), "year_min": tp.year.min(), "year_max": tp.year.max()},
        {"domain": "aquatic_biota", "endpoint": "Fish richness", "independent_unit": "freshwater ecoregion", "n_units": len(fish), "n_rows": len(fish), "n_countries": fish.country_code.nunique(), "year_min": np.nan, "year_max": np.nan},
        {"domain": "aquatic_biota", "endpoint": "EPTO richness", "independent_unit": "HydroBASINS level-4 basin", "n_units": len(epto), "n_rows": len(epto), "n_countries": epto.country_code.nunique(), "year_min": np.nan, "year_max": np.nan},
        {"domain": "aquatic_biota", "endpoint": "Chlorophyll-a", "independent_unit": "lake", "n_units": len(algae), "n_rows": len(algae), "n_countries": algae.country_code.nunique(), "year_min": algae_yearly.year.min(), "year_max": algae_yearly.year.max()},
    ])
    coverage.to_csv(AUDIT / "analysis_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([a_do, a_ec, a_tp]).to_csv(AUDIT / "water_filter_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([a_algae]).to_csv(AUDIT / "algae_filter_audit.csv", index=False, encoding="utf-8-sig")

    inventory_paths = [
        RAW / "DOSAT_GRQA.csv", RAW / "TP_GRQA.csv", RAW / "Filazzola_ChlaData_final.csv",
        RAW / "Liu2025_standardized_stream_insect_diversity.csv", GDP_NC, GDP_MEAN_TIF,
        EC_TABLE, FRESHVERTS_GPKG, WORLD_SHP,
    ]
    inventory = pd.DataFrame([
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256sum(p)} for p in inventory_paths
    ])
    inventory.to_csv(AUDIT / "data_inventory_sha256.csv", index=False, encoding="utf-8-sig")

    summary = {
        "water_station_year_rows": int(len(water)),
        "water_unique_mapped_locations": int(len(all_sites)),
        "fish_ecoregions": int(len(fish)),
        "epto_basins": int(len(epto)),
        "algae_lakes": int(len(algae)),
    }
    (AUDIT / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    save_outputs()
