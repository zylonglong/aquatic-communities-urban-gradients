"""Refit all eight GDP--freshwater models with recent Kummu GDPpc v4.

Main models use the 2020--2022 mean GDPpc at each ecological location.  Five
water-quality and three aquatic-biota endpoints are fitted with the same
background-adjustment, spatial-cluster uncertainty, density balancing and
spatial-block model-selection rules used in the audited preceding analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
UPDATE = ROOT / "11_近年GDP更新_5x5布局_2026-09-15"
CLEAN = UPDATE / "02_匹配数据"
ANALYSIS = UPDATE / "03_统计结果"
SOURCE = UPDATE / "05_SourceData"
QA = UPDATE / "07_QA"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_global_macro_models as fm  # noqa: E402


WATER_TRANSFORMS = {
    "DOSAT": "identity",
    "EC": "log",
    "TP": "log",
    "NH4N": "log",
    "NO3N": "log",
}


def configure_modules():
    for folder in (ANALYSIS, SOURCE, QA):
        folder.mkdir(parents=True, exist_ok=True)
    fm.CLEAN = CLEAN
    fm.ANALYSIS = ANALYSIS
    fm.SOURCE = SOURCE
    fm.QA = QA
    import fit_density_balanced_trends as dt
    dt.ANALYSIS = ANALYSIS
    dt.SOURCE = SOURCE
    dt.RNG = np.random.default_rng(20260916)
    return dt


def fit_models() -> tuple[list[fm.FittedEndpoint], list[dict]]:
    fitted = [fm.fit_water(endpoint, transform) for endpoint, transform in WATER_TRANSFORMS.items()]
    fitted.extend([fm.fit_fish(), fm.fit_epto(), fm.fit_algae()])
    return [item[0] for item in fitted], [item[1] for item in fitted]


def primary_outputs(models: list[fm.FittedEndpoint], transform_meta: list[dict]) -> None:
    summaries, coefficients, curves, points, densities = [], [], [], [], []
    for model in models:
        summary, coef, curve, point, density = fm.endpoint_outputs(model)
        summaries.append(summary)
        coefficients.append(coef)
        curves.append(curve)
        points.append(point)
        densities.append(density)

    summary = pd.DataFrame(summaries)
    summary["joint_gdp_q"] = fm.bh_fdr(summary["joint_gdp_p"])
    summary["contrast_q"] = fm.bh_fdr(summary["contrast_p"])
    coef = pd.concat(coefficients, ignore_index=True)
    coef["q_value_within_term"] = np.nan
    for term, idx in coef.groupby("term").groups.items():
        coef.loc[idx, "q_value_within_term"] = fm.bh_fdr(coef.loc[idx, "p_value"])

    linear, partition = fm.adjusted_linear_effects(models)
    sensitivity = fm.run_sensitivity(models)
    diagnostics = []
    for model in models:
        diagnostics.append({
            "domain": model.domain,
            "endpoint": model.endpoint,
            "n_observations": int(model.fit.nobs),
            "n_independent_units": int(model.n_units),
            "n_spatial_clusters": int(model.n_clusters),
            "n_parameters": int(len(model.fit.params)),
            "condition_number": float(model.fit.condition_number),
            "residual_mean": float(np.mean(model.fit.resid)),
            "residual_sd": float(np.std(model.fit.resid, ddof=1)),
            "residual_skew": float(st.skew(model.fit.resid, bias=False)),
            "residual_excess_kurtosis": float(st.kurtosis(model.fit.resid, fisher=True, bias=False)),
        })

    summary.to_csv(ANALYSIS / "primary_model_summary.csv", index=False, encoding="utf-8-sig")
    coef.to_csv(ANALYSIS / "primary_model_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.concat(curves, ignore_index=True).to_csv(
        SOURCE / "recent_GDP_adjusted_quadratic_curves.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(points, ignore_index=True).to_csv(
        SOURCE / "recent_GDP_observed_partial_residuals.csv.gz", index=False, compression="gzip"
    )
    pd.concat(densities, ignore_index=True).to_csv(
        SOURCE / "recent_GDP_balanced_density.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(transform_meta).to_csv(
        ANALYSIS / "transform_and_scaling_parameters.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity.to_csv(ANALYSIS / "model_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    linear.to_csv(ANALYSIS / "natural_adjusted_linear_gdp_effects.csv", index=False, encoding="utf-8-sig")
    partition.to_csv(ANALYSIS / "gdp_background_variance_partition.csv", index=False, encoding="utf-8-sig")
    linear.to_csv(SOURCE / "panel_b_adjusted_GDP_effects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(QA / "model_diagnostics.csv", index=False, encoding="utf-8-sig")

    q_audit = linear[["endpoint", "p_value", "q_value"]].sort_values("p_value").reset_index(drop=True)
    q_audit["rank"] = np.arange(1, len(q_audit) + 1)
    q_audit["m_tests"] = len(q_audit)
    q_audit["raw_bh_p_m_over_rank"] = q_audit["p_value"] * len(q_audit) / q_audit["rank"]
    q_audit["reverse_cumulative_minimum"] = (
        q_audit["raw_bh_p_m_over_rank"].iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    )
    q_audit.to_csv(QA / "BH_FDR_eight_endpoint_audit.csv", index=False, encoding="utf-8-sig")


def density_balanced_outputs(models: list[fm.FittedEndpoint], dt) -> None:
    cv_rows, summaries, curves, points = [], [], [], []
    for model in models:
        candidates = dt.formulas(model)
        cv, selected = dt.blocked_cv(model, candidates)
        summary, curve, point = dt.fit_selected(model, candidates[selected], selected)
        cv_rows.append(cv)
        summaries.append(summary)
        curves.append(curve)
        points.append(point)
    summary = pd.DataFrame(summaries)
    summary["joint_q"] = fm.bh_fdr(summary["joint_p"])
    contrast_p = pd.Series([
        2 * st.norm.sf(abs(row.contrast_10_90 / row.contrast_se))
        if row.contrast_se > 0 else np.nan
        for row in summary.itertuples()
    ])
    summary["contrast_p"] = contrast_p
    summary["contrast_q"] = fm.bh_fdr(contrast_p)
    summary.to_csv(ANALYSIS / "density_balanced_selected_trend_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(cv_rows, ignore_index=True).to_csv(
        ANALYSIS / "density_balanced_spatial_block_cv.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(curves, ignore_index=True).to_csv(
        SOURCE / "density_balanced_curves.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(points, ignore_index=True).to_csv(
        SOURCE / "density_balanced_partial_residuals.csv.gz", index=False, compression="gzip"
    )


def fit_yearmatched_water(endpoint: str, transform: str) -> fm.FittedEndpoint:
    d = pd.read_csv(CLEAN / f"water_{endpoint}_station_year.csv.gz", low_memory=False)
    d["gdppc"] = pd.to_numeric(d["gdppc_yearmatched_v4"], errors="coerce")
    d = d.dropna(subset=["value", "gdppc", "year", "latitude", "unit_id", "spatial_block_10"]).copy()
    d, meta = fm.common_columns(d, endpoint, transform)
    d["year_z"], _, _ = fm.zscore(d["year"])
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
    return fm.FittedEndpoint(
        endpoint=endpoint,
        domain="water_quality",
        data=d,
        fit=fit,
        reduced=reduced,
        gdp_mean=meta["gdp_mean"],
        gdp_sd=meta["gdp_sd"],
        y_transform=transform,
        n_units=int(d["unit_id"].nunique()),
        n_clusters=int(d["spatial_block_10"].nunique()),
        formula=formula,
    )


def temporal_sensitivity() -> None:
    models = [fit_yearmatched_water(endpoint, transform) for endpoint, transform in WATER_TRANSFORMS.items()]
    effects, partition = fm.adjusted_linear_effects(models)
    effects["exposure_definition"] = "annual GDPpc v4 matched to observation year"
    effects.to_csv(ANALYSIS / "sensitivity_yearmatched_water_linear_effects.csv", index=False, encoding="utf-8-sig")
    partition.to_csv(ANALYSIS / "sensitivity_yearmatched_water_variance_partition.csv", index=False, encoding="utf-8-sig")


def run() -> None:
    dt = configure_modules()
    models, meta = fit_models()
    primary_outputs(models, meta)
    density_balanced_outputs(models, dt)
    temporal_sensitivity()
    summary = pd.read_csv(ANALYSIS / "density_balanced_selected_trend_summary.csv")
    effects = pd.read_csv(ANALYSIS / "natural_adjusted_linear_gdp_effects.csv")
    machine = {
        "exposure": "GDPpc PPP mean 2020-2022 (constant 2021 international USD)",
        "selected_models": summary[["endpoint", "selected_model", "contrast_10_90", "contrast_ci_low", "contrast_ci_high", "joint_q", "weighted_delta_r_squared", "n_observations"]].to_dict("records"),
        "linear_effects": effects[["endpoint", "estimate", "ci_low", "ci_high", "p_value", "q_value", "partial_r_squared"]].to_dict("records"),
    }
    (ANALYSIS / "recent_gdp_model_summary.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(machine, ensure_ascii=False))


if __name__ == "__main__":
    run()
