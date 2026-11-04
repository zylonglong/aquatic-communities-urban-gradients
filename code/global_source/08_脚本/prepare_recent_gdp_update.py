"""Prepare the 1990--2024 Kummu GDPpc v4 update for the freshwater analysis.

The primary exposure is the 2020--2022 mean of the authors' global ADM2
GDPpc (PPP, constant 2021 international USD) product.  The recent three-year
mean is sampled at every already-curated water-quality and aquatic-biota unit.
Observation-year GDPpc from the same v4 stack is retained for a temporal-
matching sensitivity analysis of water-quality records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "03_清洗与匹配"
UPDATE = ROOT / "11_近年GDP更新_5x5布局_2026-09-15"
RAW = UPDATE / "01_原始数据"
CLEAN = UPDATE / "02_匹配数据"
QA = UPDATE / "07_QA"

GDP_STACK = RAW / "rast_adm2_gdp_perCapita_1990_2024_30arcmin.tif"
GDP_RECENT = CLEAN / "gdppc_ppp2021_mean_2020_2022_30arcmin.tif"

WATER_ENDPOINTS = ("DOSAT", "EC", "TP", "NH4N", "NO3N")
BIOTA_FILES = ("fish_units.csv", "epto_units.csv", "algae_units.csv")
YEAR0 = 1990
RECENT_YEARS = (2020, 2021, 2022)


def ensure_dirs() -> None:
    for folder in (RAW, CLEAN, QA):
        folder.mkdir(parents=True, exist_ok=True)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_layer(layer: np.ndarray, nodata: float | None) -> np.ndarray:
    out = layer.astype("float64", copy=False)
    bad = ~np.isfinite(out) | (out <= 0)
    if nodata is not None and np.isfinite(nodata):
        bad |= out == nodata
    out = out.copy()
    out[bad] = np.nan
    return out


def band_for_year(year: int, count: int) -> int:
    band = int(year) - YEAR0 + 1
    if not 1 <= band <= count:
        raise ValueError(f"Year {year} is outside the {YEAR0}--{YEAR0 + count - 1} stack")
    return band


def build_recent_mean() -> dict:
    if not GDP_STACK.exists():
        raise FileNotFoundError(GDP_STACK)
    with rasterio.open(GDP_STACK) as src:
        if src.count < 33:
            raise RuntimeError(f"Expected at least 33 annual bands (1990--2022); found {src.count}")
        layers = [clean_layer(src.read(band_for_year(year, src.count)), src.nodata) for year in RECENT_YEARS]
        stack = np.stack(layers)
        valid = np.isfinite(stack)
        recent = np.divide(
            np.nansum(stack, axis=0), valid.sum(axis=0),
            out=np.full(stack.shape[1:], np.nan), where=valid.sum(axis=0) > 0,
        )
        meta = src.meta.copy()
        profile = {
            **meta,
            "count": 1,
            "dtype": "float32",
            "nodata": -9999.0,
            "compress": "deflate",
            "predictor": 3,
        }
        CLEAN.mkdir(parents=True, exist_ok=True)
        with rasterio.open(GDP_RECENT, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(recent), recent, -9999.0).astype("float32"), 1)
            dst.set_band_description(1, "GDPpc PPP mean 2020-2022, constant 2021 international USD")
            dst.update_tags(
                source="Kummu et al. GDPpc v4, Zenodo 10.5281/zenodo.18429133",
                years="2020,2021,2022",
                unit="constant 2021 international USD per person, PPP",
                aggregation="arithmetic mean across the three annual bands",
            )
        return {
            "stack_bands": int(src.count),
            "stack_years": f"{YEAR0}-{YEAR0 + src.count - 1}",
            "recent_years": list(RECENT_YEARS),
            "width": int(src.width),
            "height": int(src.height),
            "resolution_degrees": [abs(float(src.res[0])), abs(float(src.res[1]))],
            "crs": str(src.crs),
            "recent_valid_cells": int(np.isfinite(recent).sum()),
            "recent_min": float(np.nanmin(recent)),
            "recent_median": float(np.nanmedian(recent)),
            "recent_max": float(np.nanmax(recent)),
        }


def nearest_valid(layer: np.ndarray, row: int, col: int, max_radius: int = 6) -> tuple[float, int]:
    """Return the closest positive land-cell value and its search radius."""
    height, width = layer.shape
    for radius in range(1, max_radius + 1):
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        window = layer[r0:r1, c0:c1]
        valid = np.argwhere(np.isfinite(window) & (window > 0))
        if not len(valid):
            continue
        global_rows = valid[:, 0] + r0
        global_cols = valid[:, 1] + c0
        distance2 = (global_rows - row) ** 2 + (global_cols - col) ** 2
        closest = int(np.argmin(distance2))
        return float(layer[global_rows[closest], global_cols[closest]]), radius
    return np.nan, -1


def sample_layer(layer: np.ndarray, transform, frame: pd.DataFrame,
                 lon_col: str = "longitude", lat_col: str = "latitude") -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(frame[lon_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[lat_col], errors="coerce").to_numpy(float)
    values = np.full(len(frame), np.nan)
    radius_used = np.full(len(frame), -1, dtype=int)
    finite = np.isfinite(x) & np.isfinite(y) & (x >= -180) & (x <= 180) & (y >= -90) & (y <= 90)
    positions = np.where(finite)[0]
    if len(positions):
        rows, cols = rasterio.transform.rowcol(transform, x[positions], y[positions])
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        inside = (rows >= 0) & (rows < layer.shape[0]) & (cols >= 0) & (cols < layer.shape[1])
        inside_positions = positions[inside]
        values[inside_positions] = layer[rows[inside], cols[inside]]
        for pos, row, col in zip(inside_positions, rows[inside], cols[inside]):
            if not np.isfinite(values[pos]) or values[pos] <= 0:
                values[pos], radius_used[pos] = nearest_valid(layer, int(row), int(col))
            else:
                radius_used[pos] = 0
    return values, radius_used


def sample_recent(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(GDP_RECENT) as src:
        layer = clean_layer(src.read(1), src.nodata)
        return sample_layer(layer, src.transform, frame)


def sample_yearmatched(frame: pd.DataFrame, year_col: str = "year") -> tuple[np.ndarray, np.ndarray]:
    result = np.full(len(frame), np.nan)
    radius = np.full(len(frame), -1, dtype=int)
    years = pd.to_numeric(frame[year_col], errors="coerce")
    with rasterio.open(GDP_STACK) as src:
        for year in sorted(years.dropna().astype(int).unique()):
            if year < YEAR0 or year > YEAR0 + src.count - 1:
                continue
            pos = np.where(years.to_numpy() == year)[0]
            subset = frame.iloc[pos]
            layer = clean_layer(src.read(band_for_year(year, src.count)), src.nodata)
            values, used = sample_layer(layer, src.transform, subset)
            result[pos] = values
            radius[pos] = used
    return result, radius


def update_frame(frame: pd.DataFrame, yearmatched: bool) -> tuple[pd.DataFrame, dict]:
    d = frame.copy()
    if "gdppc" in d.columns:
        d["gdppc_legacy"] = pd.to_numeric(d["gdppc"], errors="coerce")
    recent, radius = sample_recent(d)
    d["gdppc"] = recent
    d["gdp_recent_fallback_cells"] = radius
    if yearmatched and "year" in d.columns:
        matched, matched_radius = sample_yearmatched(d)
        d["gdppc_yearmatched_v4"] = matched
        d["gdp_yearmatched_fallback_cells"] = matched_radius
    matched = np.isfinite(recent) & (recent > 0)
    d = d.loc[matched].copy()
    old = pd.to_numeric(d.get("gdppc_legacy"), errors="coerce")
    valid_old = old.notna() & old.gt(0) & d["gdppc"].gt(0)
    correlation = float(np.corrcoef(np.log(old[valid_old]), np.log(d.loc[valid_old, "gdppc"]))[0, 1]) if valid_old.sum() > 2 else np.nan
    audit = {
        "rows_input": int(len(frame)),
        "rows_recent_gdp_matched": int(len(d)),
        "direct_matches": int((d["gdp_recent_fallback_cells"] == 0).sum()),
        "coastal_fallback_matches": int((d["gdp_recent_fallback_cells"] > 0).sum()),
        "unmatched": int(len(frame) - len(d)),
        "log_old_new_correlation": correlation,
        "recent_gdppc_median": float(d["gdppc"].median()),
    }
    if "gdppc_yearmatched_v4" in d:
        audit["yearmatched_available"] = int(d["gdppc_yearmatched_v4"].notna().sum())
    return d, audit


def build_water_site_tables(water_frames: list[pd.DataFrame]) -> None:
    water = pd.concat(water_frames, ignore_index=True)
    site = water.groupby(["unit_id", "endpoint"], as_index=False).agg(
        value=("value", "median"),
        gdppc=("gdppc", "median"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        country_code=("country_code", "first"),
        country_name=("country_name", "first"),
        continent=("continent", "first"),
        n_station_years=("value", "size"),
    )
    site.to_csv(CLEAN / "water_site_medians_long.csv.gz", index=False, compression="gzip")
    all_sites = site.groupby("unit_id", as_index=False).agg(
        gdppc=("gdppc", "median"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        country_code=("country_code", "first"),
        country_name=("country_name", "first"),
        continent=("continent", "first"),
        n_endpoints=("endpoint", "nunique"),
    )
    all_sites["value"] = all_sites["gdppc"]
    all_sites.to_csv(CLEAN / "water_all_unique_sites_gdppc.csv.gz", index=False, compression="gzip")
    water.to_csv(CLEAN / "water_all_station_year.csv.gz", index=False, compression="gzip")


def run() -> None:
    ensure_dirs()
    raster_meta = build_recent_mean()
    audits: list[dict] = []
    water_frames: list[pd.DataFrame] = []
    for endpoint in WATER_ENDPOINTS:
        source = LEGACY / f"water_{endpoint}_station_year.csv.gz"
        frame = pd.read_csv(source, low_memory=False)
        updated, audit = update_frame(frame, yearmatched=True)
        updated.to_csv(CLEAN / source.name, index=False, compression="gzip")
        water_frames.append(updated)
        audits.append({"domain": "water_quality", "endpoint": endpoint, **audit})
    build_water_site_tables(water_frames)

    for filename in BIOTA_FILES:
        source = LEGACY / filename
        frame = pd.read_csv(source, low_memory=False)
        updated, audit = update_frame(frame, yearmatched=(filename == "algae_units.csv"))
        updated.to_csv(CLEAN / filename, index=False, encoding="utf-8-sig")
        endpoint = {"fish_units.csv": "Fish", "epto_units.csv": "EPTO", "algae_units.csv": "Algae"}[filename]
        audits.append({"domain": "aquatic_biota", "endpoint": endpoint, **audit})

    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(QA / "recent_gdp_matching_audit.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "dataset": "Kummu et al. global gridded GDPpc v4",
        "zenodo_record": "10.5281/zenodo.18429133",
        "source_file": GDP_STACK.name,
        "source_sha256": sha256sum(GDP_STACK),
        "unit": "constant 2021 international USD per person, PPP",
        "primary_exposure": "arithmetic mean of 2020, 2021 and 2022",
        "main_analysis_interpretation": "recent structural economic context; associational, not synchronous exposure",
        "temporal_sensitivity": "annual v4 GDPpc matched to water-quality observation year",
        "raster": raster_meta,
        "matching_audit": audits,
    }
    (QA / "recent_gdp_data_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"raster": raster_meta, "matching": audits}, ensure_ascii=False))


if __name__ == "__main__":
    run()
