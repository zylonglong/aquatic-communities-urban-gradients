"""Prepare and model supplementary GRQA ammonium- and nitrate-N data.

The supplement follows the primary water-quality pipeline: 1990--2015,
station-year medians, point-sampled yearly GDPpc, log-transformed responses,
natural/sampling-background adjustment, independent-unit and GDP-density
balancing, and spatial-block cross-validation of pre-specified trend forms.
Model selection is based on predictive performance and parsimony, never sign.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf

from prepare_global_macro_data import AUDIT, CLEAN, RAW, SOURCE, load_world, prepare_grqa
from fit_global_macro_models import (
    ANALYSIS,
    FittedEndpoint,
    adjusted_linear_effects,
    bh_fdr,
    common_columns,
    zscore,
)
from fit_density_balanced_trends import blocked_cv, fit_selected, formulas


ENDPOINTS = {
    "NH4N": {"file": RAW / "NH4N_GRQA.csv", "lower": 0.0, "upper": 100.0},
    "NO3N": {"file": RAW / "NO3N_GRQA.csv", "lower": 0.0, "upper": 100.0},
}


def fit_water_table(path: Path, endpoint: str) -> tuple[FittedEndpoint, dict]:
    d = pd.read_csv(path, low_memory=False)
    d = d.dropna(
        subset=["value", "gdppc", "year", "latitude", "unit_id", "spatial_block_10"]
    ).copy()
    d, meta = common_columns(d, endpoint, "log")
    d["year_z"], _, _ = zscore(d["year"])
    d["source"] = d["source"].fillna("Unknown").astype(str)
    formula = (
        "y_z ~ gdp_z + gdp2 + year_z + "
        "bs(abs_lat, df=4, degree=3, include_intercept=False) + "
        "C(continent) + C(source)"
    )
    reduced_formula = formula.replace("gdp_z + gdp2 + ", "")
    fit = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["spatial_block_10"]}
    )
    reduced = smf.ols(reduced_formula, data=d).fit()
    return FittedEndpoint(
        endpoint=endpoint,
        # Keep the primary domain label so the shared fitting code uses the
        # pre-specified 10-degree water-quality spatial blocks.
        domain="water_quality",
        data=d,
        fit=fit,
        reduced=reduced,
        gdp_mean=meta["gdp_mean"],
        gdp_sd=meta["gdp_sd"],
        y_transform="log",
        n_units=int(d["unit_id"].nunique()),
        n_clusters=int(d["spatial_block_10"].nunique()),
        formula=formula,
    ), meta


def main() -> None:
    for folder in (AUDIT, CLEAN, ANALYSIS, SOURCE):
        folder.mkdir(parents=True, exist_ok=True)
    world = load_world()
    audits: list[dict] = []
    site_maps: list[pd.DataFrame] = []
    models: list[FittedEndpoint] = []
    transforms: list[dict] = []
    audit_path = AUDIT / "supplementary_nitrogen_filter_audit.csv"
    prior_audit = pd.read_csv(audit_path).set_index("endpoint") if audit_path.exists() else None

    for endpoint, spec in ENDPOINTS.items():
        clean_path = CLEAN / f"water_{endpoint}_station_year.csv.gz"
        if clean_path.exists() and prior_audit is not None and endpoint in prior_audit.index:
            table = pd.read_csv(clean_path, low_memory=False)
            audit = prior_audit.loc[endpoint].to_dict()
            audit["endpoint"] = endpoint
        else:
            table, audit = prepare_grqa(
                spec["file"], endpoint, spec["lower"], spec["upper"], world
            )
            table.to_csv(clean_path, index=False, compression="gzip")
        audits.append(audit)
        site = table.groupby("unit_id", as_index=False).agg(
            value=("value", "median"),
            gdppc=("gdppc", "median"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            country_code=("country_code", "first"),
            country_name=("country_name", "first"),
            continent=("continent", "first"),
            source=("source", "first"),
            n_station_years=("year", "size"),
        )
        site["endpoint"] = endpoint
        site_maps.append(site)
        model, meta = fit_water_table(clean_path, endpoint)
        models.append(model)
        transforms.append(meta)

    pd.DataFrame(audits).to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    site_map = pd.concat(site_maps, ignore_index=True)
    site_map.to_csv(
        CLEAN / "supplementary_nitrogen_site_medians.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(transforms).to_csv(
        ANALYSIS / "supplementary_nitrogen_transform_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    curves: list[pd.DataFrame] = []
    points: list[pd.DataFrame] = []
    for model in models:
        candidates = formulas(model)
        cv, selected = blocked_cv(model, candidates)
        summary, curve, point = fit_selected(model, candidates[selected], selected)
        cv_rows.append(cv)
        summaries.append(summary)
        curves.append(curve)
        points.append(point)

    summary = pd.DataFrame(summaries)
    summary["joint_q"] = bh_fdr(summary["joint_p"])
    summary["multiplicity_family"] = "two exploratory supplementary nitrogen endpoints"
    summary.to_csv(
        ANALYSIS / "supplementary_nitrogen_density_balanced_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(cv_rows, ignore_index=True).to_csv(
        ANALYSIS / "supplementary_nitrogen_spatial_block_cv.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nitrogen_curves = pd.concat(curves, ignore_index=True)
    nitrogen_points = pd.concat(points, ignore_index=True)
    nitrogen_curves.to_csv(
        SOURCE / "Fig_S2_nitrogen_curves.csv", index=False, encoding="utf-8-sig"
    )
    nitrogen_points.to_csv(
        SOURCE / "Fig_S2_nitrogen_partial_residuals.csv.gz",
        index=False,
        compression="gzip",
    )

    # Natural-background-adjusted linear effects for the alternative main
    # figure. Its FDR family is defined across the six endpoints displayed
    # together, not across the two nitrogen endpoints alone.
    nitrogen_linear, nitrogen_partition = adjusted_linear_effects(models)
    nitrogen_linear.to_csv(
        ANALYSIS / "supplementary_nitrogen_natural_adjusted_linear_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nitrogen_partition.to_csv(
        ANALYSIS / "supplementary_nitrogen_gdp_background_variance_partition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retained = ["DOSAT", "Fish", "EPTO", "Algae"]
    primary_linear = pd.read_csv(ANALYSIS / "natural_adjusted_linear_gdp_effects.csv")
    alternative_linear = pd.concat(
        [primary_linear.loc[primary_linear["endpoint"].isin(retained)], nitrogen_linear],
        ignore_index=True,
    )
    alternative_linear["q_value"] = bh_fdr(alternative_linear["p_value"])
    alternative_linear.to_csv(
        ANALYSIS / "alternative_NH4N_NO3N_natural_adjusted_linear_gdp_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_trend = pd.read_csv(ANALYSIS / "density_balanced_selected_trend_summary.csv")
    alternative_trend = pd.concat(
        [primary_trend.loc[primary_trend["endpoint"].isin(retained)], summary],
        ignore_index=True,
    )
    alternative_trend["joint_q"] = bh_fdr(alternative_trend["joint_p"])
    alternative_trend["contrast_p"] = 2 * st.norm.sf(
        np.abs(alternative_trend["contrast_10_90"] / alternative_trend["contrast_se"])
    )
    alternative_trend["contrast_q"] = bh_fdr(alternative_trend["contrast_p"])
    alternative_trend["multiplicity_family"] = (
        "six endpoints in alternative NH4-N/NO3-N main figure"
    )
    alternative_trend.to_csv(
        ANALYSIS / "alternative_NH4N_NO3N_density_balanced_trend_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_points = pd.read_csv(SOURCE / "Fig_main_density_balanced_partial_residuals.csv.gz")
    primary_curves = pd.read_csv(SOURCE / "Fig_main_density_balanced_curves.csv")
    alternative_points = pd.concat(
        [primary_points.loc[primary_points["endpoint"].isin(retained)], nitrogen_points],
        ignore_index=True,
    )
    alternative_curves = pd.concat(
        [primary_curves.loc[primary_curves["endpoint"].isin(retained)], nitrogen_curves],
        ignore_index=True,
    )
    alternative_points.to_csv(
        SOURCE / "Fig_alt_NH4N_NO3N_density_balanced_partial_residuals.csv.gz",
        index=False,
        compression="gzip",
    )
    alternative_curves.to_csv(
        SOURCE / "Fig_alt_NH4N_NO3N_density_balanced_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_density = pd.read_csv(SOURCE / "Fig_main_balanced_gdp_density.csv.gz")
    nitrogen_density = nitrogen_points.groupby(["endpoint", "unit_id"], as_index=False)["gdppc"].median()
    nitrogen_density["domain"] = "water_quality"
    nitrogen_density["weight"] = nitrogen_density.groupby("endpoint")["unit_id"].transform(
        lambda s: 1.0 / len(s)
    )
    alternative_density = pd.concat(
        [primary_density.loc[primary_density["endpoint"].isin(retained)], nitrogen_density],
        ignore_index=True,
    )
    alternative_density.to_csv(
        SOURCE / "Fig_alt_NH4N_NO3N_balanced_gdp_density.csv.gz",
        index=False,
        compression="gzip",
    )

    dosat_sites = pd.read_csv(CLEAN / "water_site_medians_long.csv.gz")
    dosat_sites = dosat_sites.loc[dosat_sites["endpoint"] == "DOSAT"].copy()
    alternative_water_maps = pd.concat([dosat_sites, site_map], ignore_index=True, sort=False)
    alternative_water_maps.to_csv(
        CLEAN / "alternative_NH4N_NO3N_water_site_medians_long.csv.gz",
        index=False,
        compression="gzip",
    )
    site_locations = alternative_water_maps.dropna(
        subset=["latitude", "longitude", "gdppc"]
    ).copy()
    site_locations["lat_key"] = site_locations["latitude"].round(3)
    site_locations["lon_key"] = site_locations["longitude"].round(3)
    all_sites = site_locations.groupby(["lat_key", "lon_key"], as_index=False).agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        gdppc=("gdppc", "median"),
        country_code=("country_code", "first"),
        n_endpoint_records=("endpoint", "size"),
    )
    all_sites["value"] = all_sites["gdppc"]
    all_sites["unit_id"] = "ALT_SITE_" + all_sites.index.astype(str)
    all_sites.to_csv(
        CLEAN / "alternative_NH4N_NO3N_all_unique_sites_gdppc.csv.gz",
        index=False,
        compression="gzip",
    )
    print(json.dumps({
        "audit": audits,
        "models": summary[[
            "endpoint", "selected_model", "contrast_10_90", "contrast_ci_low",
            "contrast_ci_high", "joint_p", "joint_q", "n_observations",
            "n_independent_units",
        ]].to_dict("records"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
