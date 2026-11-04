# -*- coding: utf-8 -*-
"""Refresh station/record inputs after GIS coordinates change, then rerun analysis."""
from pathlib import Path
import importlib.util

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROC = OUT / "01_INPUT与过程数据"


def load_analysis_module():
    path = HERE / "02_build_wq_input_and_analyze.py"
    spec = importlib.util.spec_from_file_location("wq_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def registry_lookup(reg):
    keep = [
        "station_id", "spatial_id", "geometry_id", "waterbody_cluster_id",
        "spatial_waterbody_type", "spatial_longitude", "spatial_latitude",
        "spatial_quality", "waterbody_name_spatial", "station_name_canonical",
        "station_scope", "monitoring_agency",
    ]
    return reg[keep].copy().rename(columns={
        "spatial_id": "station_unit_spatial_id",
        "spatial_longitude": "longitude",
        "spatial_latitude": "latitude",
    })


def refresh_site(reg, static, pop):
    path = PROC / "WQ_Site_Special_13Ranges.csv"
    old = pd.read_csv(path, low_memory=False)
    lookup = registry_lookup(reg)
    spatial_cols = [c for c in lookup.columns if c != "station_id"]
    env_cols = [c for c in static.columns if c not in {"spatial_id", "water_body_type"}]
    pop_cols = [c for c in pop.columns if c.startswith("worldpop_")]
    drop_cols = [c for c in spatial_cols + ["spatial_id"] + env_cols + pop_cols + ["water_body_type"] if c in old.columns]
    base = old.drop(columns=drop_cols)
    site = base.merge(lookup, on="station_id", how="left", validate="1:1")
    site["spatial_id"] = site["geometry_id"]
    site = site.merge(static, on="spatial_id", how="left", validate="m:1")
    if "water_body_type" in site.columns:
        site = site.drop(columns=["water_body_type"])
    pop_site = pop.groupby("spatial_id", as_index=False)[pop_cols].mean()
    site = site.merge(pop_site, on="spatial_id", how="left", validate="m:1")
    site["n_station_units_same_geometry"] = site.groupby("spatial_id")["station_id"].transform("nunique")
    site["geometry_equal_weight"] = 1.0 / site["n_station_units_same_geometry"]
    if len(site) != 103 or site.geometry_id.nunique() != 56:
        raise ValueError("Refreshed site input does not contain 103 units and 56 geometries")
    if site[env_cols].isna().any().any():
        raise ValueError("Static environmental features are missing after site refresh")
    site.to_csv(path, index=False, encoding="utf-8-sig")
    return site


def refresh_record(reg, static, pop):
    path = PROC / "WQ_Record_Special_13Ranges.csv"
    old = pd.read_csv(path, low_memory=False)
    lookup = registry_lookup(reg)
    lookup = lookup.rename(columns={"station_unit_spatial_id": "spatial_id"})
    spatial_cols = [c for c in lookup.columns if c != "station_id"]
    env_cols = [c for c in static.columns if c not in {"spatial_id", "water_body_type"}]
    pop_cols = [c for c in pop.columns if c.startswith("worldpop_")]
    drop_cols = [c for c in spatial_cols + env_cols + pop_cols + ["water_body_type", "water_body_type_pop"] if c in old.columns]
    base = old.drop(columns=drop_cols)
    record = base.merge(lookup, on="station_id", how="left", validate="m:1")
    static_join = static.rename(columns={"spatial_id": "geometry_id", "water_body_type": "water_body_type"})
    record = record.merge(static_join, on="geometry_id", how="left", validate="m:1")
    if "monitoring_year" in record.columns:
        year = pd.to_numeric(record["monitoring_year"], errors="coerce")
    else:
        year = pd.to_datetime(record["monitoring_date_end"], errors="coerce").dt.year
    record["year"] = year
    pop_join = pop.rename(columns={"spatial_id": "geometry_id", "water_body_type": "water_body_type_pop"})
    record = record.merge(pop_join, on=["geometry_id", "year"], how="left", validate="m:1")
    if len(record) != len(old):
        raise ValueError("Record row count changed during refresh")
    record.to_csv(path, index=False, encoding="utf-8-sig")
    return record


def refresh_station_month_geometry(reg):
    mapping = reg.set_index("station_id")
    for filename in ["WQ_Class_StationUnit_Month.csv", "WQ_PollutionIndex_StationUnit_Month.csv"]:
        path = PROC / filename
        d = pd.read_csv(path, low_memory=False)
        d["spatial_id"] = d.station_id.map(mapping.spatial_id)
        d["geometry_id"] = d.station_id.map(mapping.geometry_id)
        d["waterbody_cluster_id"] = d.station_id.map(mapping.waterbody_cluster_id)
        d["spatial_waterbody_type"] = d.station_id.map(mapping.spatial_waterbody_type)
        d.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    reg = pd.read_csv(PROC / "WQ_Station_Spatial_Registry.csv", encoding="utf-8-sig")
    static = pd.read_csv(PROC / "WQ_Environmental_Features_Static_Wide.csv", low_memory=False)
    pop = pd.read_csv(PROC / "WQ_WorldPop_Annual_Wide.csv", low_memory=False)
    site = refresh_site(reg, static, pop)
    refresh_record(reg, static, pop)
    refresh_station_month_geometry(reg)

    m = load_analysis_module()
    scores, _, _, modules = m.build_gradients(site)
    assoc, best, indiv = m.analyse(site, scores, modules)
    overview = m.build_overview(site)
    print({
        "site_rows": len(site), "unique_geometries": site.spatial_id.nunique(),
        "gradient_rows": len(scores), "module_associations": len(assoc),
        "candidate_best_rows": len(best), "individual_associations": len(indiv),
        "overview_rows": len(overview),
    })


if __name__ == "__main__":
    main()
