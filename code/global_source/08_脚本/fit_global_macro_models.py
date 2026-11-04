"""Fit the six pre-specified GDP--freshwater response models.

The analysis is explicitly associational.  Each response is transformed for
distributional stability, standardized, and modelled against standardized
ln(GDPpc), allowing a quadratic term.  Geographic background is controlled in
all models; endpoint-specific natural/sampling covariates are added where the
public source supplies them.  Uncertainty is cluster-robust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "03_清洗与匹配"
ANALYSIS = ROOT / "04_统计分析"
SOURCE = ROOT / "06_SourceData"
QA = ROOT / "09_QA"


@dataclass
class FittedEndpoint:
    endpoint: str
    domain: str
    data: pd.DataFrame
    fit: object
    reduced: object
    gdp_mean: float
    gdp_sd: float
    y_transform: str
    n_units: int
    n_clusters: int
    formula: str


def ensure_dirs() -> None:
    for p in (ANALYSIS, SOURCE, QA):
        p.mkdir(parents=True, exist_ok=True)


def zscore(s: pd.Series) -> tuple[pd.Series, float, float]:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mean = float(s.mean())
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (s - mean) / sd, mean, sd


def bh_fdr(p: pd.Series) -> pd.Series:
    values = pd.to_numeric(p, errors="coerce").to_numpy(float)
    out = np.full(len(values), np.nan)
    valid = np.where(np.isfinite(values))[0]
    if not len(valid):
        return pd.Series(out, index=p.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0, 1)
    return pd.Series(out, index=p.index)


def safe_log(values: pd.Series) -> tuple[pd.Series, float]:
    v = pd.to_numeric(values, errors="coerce").astype(float)
    positive = v[v > 0]
    epsilon = float(positive.min() / 2) if len(positive) else 1e-8
    return np.log(v.clip(lower=0) + epsilon), epsilon


def common_columns(d: pd.DataFrame, endpoint: str, transform: str) -> tuple[pd.DataFrame, dict]:
    d = d.copy()
    raw = pd.to_numeric(d["value"], errors="coerce")
    if transform == "identity":
        transformed, epsilon = raw, np.nan
    elif transform == "log":
        transformed, epsilon = safe_log(raw)
    elif transform == "log1p":
        transformed, epsilon = np.log1p(raw.clip(lower=0)), 0.0
    else:
        raise ValueError(transform)
    d["y_z"], y_mean, y_sd = zscore(transformed)
    d["log_gdp"] = np.log(pd.to_numeric(d["gdppc"], errors="coerce"))
    d["gdp_z"], gdp_mean, gdp_sd = zscore(d["log_gdp"])
    d["gdp2"] = d["gdp_z"] ** 2
    d["abs_lat"] = pd.to_numeric(d["latitude"], errors="coerce").abs()
    d["continent"] = d["continent"].fillna("Unassigned").astype(str)
    return d, {
        "endpoint": endpoint, "transform": transform, "epsilon": epsilon,
        "y_mean": y_mean, "y_sd": y_sd,
        "gdp_mean": gdp_mean, "gdp_sd": gdp_sd,
    }


def fit_water(endpoint: str, transform: str) -> tuple[FittedEndpoint, dict]:
    d = pd.read_csv(CLEAN / f"water_{endpoint}_station_year.csv.gz", low_memory=False)
    d = d.dropna(subset=["value", "gdppc", "year", "latitude", "unit_id", "spatial_block_10"]).copy()
    d, meta = common_columns(d, endpoint, transform)
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
    result = FittedEndpoint(
        endpoint=endpoint, domain="water_quality", data=d, fit=fit, reduced=reduced,
        gdp_mean=meta["gdp_mean"], gdp_sd=meta["gdp_sd"],
        y_transform=transform, n_units=int(d["unit_id"].nunique()),
        n_clusters=int(d["spatial_block_10"].nunique()), formula=formula,
    )
    return result, meta


def fit_fish() -> tuple[FittedEndpoint, dict]:
    d = pd.read_csv(CLEAN / "fish_units.csv")
    d["value"] = pd.to_numeric(d["fish_richness"], errors="coerce")
    d = d.dropna(subset=["value", "gdppc", "area_km2", "latitude", "unit_id", "spatial_block_20"]).copy()
    d, meta = common_columns(d, "Fish", "log1p")
    d["realm"] = d["WWF_REALM"].fillna("Unassigned").astype(str)
    d["log_area_z"], _, _ = zscore(np.log(pd.to_numeric(d["area_km2"], errors="coerce")))
    formula = (
        "y_z ~ gdp_z + gdp2 + log_area_z + "
        "bs(abs_lat, df=4, degree=3, include_intercept=False) + C(realm)"
    )
    reduced_formula = formula.replace("gdp_z + gdp2 + ", "")
    fit = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["spatial_block_20"]}
    )
    reduced = smf.ols(reduced_formula, data=d).fit()
    return FittedEndpoint(
        endpoint="Fish", domain="aquatic_biota", data=d, fit=fit, reduced=reduced,
        gdp_mean=meta["gdp_mean"], gdp_sd=meta["gdp_sd"], y_transform="log1p",
        n_units=len(d), n_clusters=int(d["spatial_block_20"].nunique()), formula=formula,
    ), meta


def fit_epto() -> tuple[FittedEndpoint, dict]:
    d = pd.read_csv(CLEAN / "epto_units.csv")
    d["value"] = pd.to_numeric(d["TDq0"], errors="coerce")
    required = [
        "value", "gdppc", "latitude", "unit_id", "spatial_block_20",
        "Area", "Bio1", "Bio12", "Elev_range", "Sam_eff", "Time span",
    ]
    d = d.dropna(subset=required).copy()
    d, meta = common_columns(d, "EPTO", "log1p")
    for source, target, transform in [
        ("Area", "log_area_z", "log"),
        ("Bio1", "temp_z", "identity"),
        ("Bio12", "precip_z", "identity"),
        ("Elev_range", "elev_range_z", "log1p"),
        ("Sam_eff", "sampling_z", "log1p"),
        ("Time span", "time_span_z", "log1p"),
    ]:
        x = pd.to_numeric(d[source], errors="coerce")
        if transform == "log":
            x = np.log(x.clip(lower=1e-9))
        elif transform == "log1p":
            x = np.log1p(x.clip(lower=0))
        d[target], _, _ = zscore(x)
    formula = (
        "y_z ~ gdp_z + gdp2 + log_area_z + temp_z + precip_z + elev_range_z + "
        "sampling_z + time_span_z + bs(abs_lat, df=4, degree=3, include_intercept=False)"
    )
    reduced_formula = formula.replace("gdp_z + gdp2 + ", "")
    fit = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["spatial_block_20"]}
    )
    reduced = smf.ols(reduced_formula, data=d).fit()
    return FittedEndpoint(
        endpoint="EPTO", domain="aquatic_biota", data=d, fit=fit, reduced=reduced,
        gdp_mean=meta["gdp_mean"], gdp_sd=meta["gdp_sd"], y_transform="log1p",
        n_units=len(d), n_clusters=int(d["spatial_block_20"].nunique()), formula=formula,
    ), meta


def fit_algae() -> tuple[FittedEndpoint, dict]:
    d = pd.read_csv(CLEAN / "algae_units.csv")
    d = d.dropna(subset=["value", "gdppc", "year", "latitude", "unit_id", "spatial_block_20"]).copy()
    d, meta = common_columns(d, "Algae", "log")
    d["year_z"], _, _ = zscore(d["year"])
    d["surface_missing"] = d["surface_area"].isna().astype(int)
    surface = pd.to_numeric(d["surface_area"], errors="coerce")
    surface = surface.fillna(float(surface.median()))
    d["surface_z"], _, _ = zscore(np.log1p(surface.clip(lower=0)))
    formula = (
        "y_z ~ gdp_z + gdp2 + year_z + surface_z + surface_missing + "
        "bs(abs_lat, df=4, degree=3, include_intercept=False) + C(continent)"
    )
    reduced_formula = formula.replace("gdp_z + gdp2 + ", "")
    fit = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["spatial_block_20"]}
    )
    reduced = smf.ols(reduced_formula, data=d).fit()
    return FittedEndpoint(
        endpoint="Algae", domain="aquatic_biota", data=d, fit=fit, reduced=reduced,
        gdp_mean=meta["gdp_mean"], gdp_sd=meta["gdp_sd"], y_transform="log",
        n_units=len(d), n_clusters=int(d["spatial_block_20"].nunique()), formula=formula,
    ), meta


def endpoint_outputs(
    model: FittedEndpoint,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d, fit = model.data, model.fit
    terms = ["gdp_z", "gdp2"]
    cov = fit.cov_params().loc[terms, terms].to_numpy()
    beta = fit.params[terms].to_numpy()

    # Population-standardized marginal predictions.  Earlier versions plotted
    # only the two GDP columns and therefore forced both the estimate and its
    # standard error to zero at mean GDP.  Here the complete design vector is
    # retained, non-GDP columns are averaged over the analytical sample, and
    # uncertainty is propagated through the full robust covariance matrix.
    names = list(fit.model.exog_names)
    gdp_idx, gdp2_idx = names.index("gdp_z"), names.index("gdp2")
    xbar = np.asarray(fit.model.exog, dtype=float).mean(axis=0)
    grid_z = np.linspace(float(d["gdp_z"].min()), float(d["gdp_z"].max()), 240)
    X_grid = np.tile(xbar, (len(grid_z), 1))
    X_grid[:, gdp_idx] = grid_z
    X_grid[:, gdp2_idx] = grid_z ** 2
    params = fit.params.reindex(names).to_numpy(float)
    cov_full = fit.cov_params().loc[names, names].to_numpy(float)
    predicted = X_grid @ params
    se = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", X_grid, cov_full, X_grid)))
    curve = pd.DataFrame({
        "domain": model.domain, "endpoint": model.endpoint,
        "gdp_z": grid_z,
        "gdppc": np.exp(grid_z * model.gdp_sd + model.gdp_mean),
        "predicted_partial": predicted,
        "se": se, "ci_low": predicted - 1.96 * se, "ci_high": predicted + 1.96 * se,
        "prediction_definition": "population-standardized marginal prediction; non-GDP design columns fixed at sample means",
    })

    X_observed_gdp = np.tile(xbar, (len(d), 1))
    X_observed_gdp[:, gdp_idx] = d["gdp_z"].to_numpy(float)
    X_observed_gdp[:, gdp2_idx] = d["gdp2"].to_numpy(float)
    partial = np.asarray(fit.resid) + X_observed_gdp @ params
    point_cols = ["unit_id", "gdppc", "value", "latitude", "longitude", "gdp_z"]
    if "year" in d:
        point_cols.append("year")
    points = d[point_cols].copy()
    points["domain"] = model.domain
    points["endpoint"] = model.endpoint
    points["partial_response"] = np.asarray(partial)

    ci = fit.conf_int()
    coefficients = pd.DataFrame([
        {
            "domain": model.domain, "endpoint": model.endpoint, "term": term,
            "estimate": float(fit.params[term]), "se": float(fit.bse[term]),
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "p_value": float(fit.pvalues[term]),
        }
        for term in terms
    ])

    q10, q90 = d["gdp_z"].quantile([0.1, 0.9])
    contrast_vector = np.array([q90 - q10, q90 ** 2 - q10 ** 2])
    contrast = float(contrast_vector @ beta)
    contrast_se = float(np.sqrt(max(0, contrast_vector @ cov @ contrast_vector)))
    contrast_p = float(2 * st.norm.sf(abs(contrast / contrast_se))) if contrast_se > 0 else np.nan
    joint = fit.wald_test("gdp_z = 0, gdp2 = 0", scalar=True)
    joint_p = float(np.asarray(joint.pvalue).squeeze())
    delta_r2 = float(fit.rsquared - model.reduced.rsquared)
    summary = {
        "domain": model.domain, "endpoint": model.endpoint,
        "n_observations": int(fit.nobs), "n_independent_units": model.n_units,
        "n_clusters": model.n_clusters, "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj), "delta_r_squared": delta_r2,
        "gdp_10_90_contrast": contrast, "contrast_se": contrast_se,
        "contrast_ci_low": contrast - 1.96 * contrast_se,
        "contrast_ci_high": contrast + 1.96 * contrast_se,
        "contrast_p": contrast_p, "joint_gdp_p": joint_p,
        "gdp10": float(np.exp(q10 * model.gdp_sd + model.gdp_mean)),
        "gdp90": float(np.exp(q90 * model.gdp_sd + model.gdp_mean)),
        "formula": model.formula, "response_transform": model.y_transform,
    }
    density = d.groupby("unit_id", as_index=False)["gdppc"].median()
    density["domain"] = model.domain
    density["endpoint"] = model.endpoint
    density["weight"] = 1.0 / len(density)
    return summary, coefficients, curve, points, density


def adjusted_linear_effects(models: list[FittedEndpoint]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate comparable linear GDP effects and partition explained variance.

    The covariate set is kept unchanged.  The resulting standardized beta is
    the unique/direct associational component after adjustment; GDP-only R2 and
    the shared component retain the upstream association that overlaps with
    natural background rather than hiding it by deleting covariates.
    """
    effects, partitions = [], []
    for model in models:
        d = model.data.copy()
        linear_formula = model.formula.replace("gdp_z + gdp2", "gdp_z")
        background_formula = model.formula.replace("gdp_z + gdp2 + ", "")
        group_col = "spatial_block_10" if model.domain == "water_quality" else "spatial_block_20"
        fit = smf.ols(linear_formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[group_col]}
        )
        fit_plain = smf.ols(linear_formula, data=d).fit()
        background = smf.ols(background_formula, data=d).fit()
        gdp_only = smf.ols("y_z ~ gdp_z", data=d).fit()

        ci = fit.conf_int().loc["gdp_z"]
        unique_gdp = float(fit_plain.rsquared - background.rsquared)
        unique_background = float(fit_plain.rsquared - gdp_only.rsquared)
        shared = float(gdp_only.rsquared + background.rsquared - fit_plain.rsquared)
        denominator = max(1e-12, 1.0 - float(background.rsquared))

        names = list(fit_plain.model.exog_names)
        gdp_idx = names.index("gdp_z")
        x_background = np.delete(np.asarray(fit_plain.model.exog, dtype=float), gdp_idx, axis=1)
        aux = sm.OLS(d["gdp_z"].to_numpy(float), x_background).fit()
        vif = float(1.0 / max(1e-12, 1.0 - aux.rsquared))

        effects.append({
            "domain": model.domain,
            "endpoint": model.endpoint,
            "estimate": float(fit.params["gdp_z"]),
            "se": float(fit.bse["gdp_z"]),
            "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "p_value": float(fit.pvalues["gdp_z"]),
            "n_observations": int(fit.nobs),
            "n_independent_units": int(model.n_units),
            "n_spatial_clusters": int(d[group_col].nunique()),
            "partial_r_squared": unique_gdp / denominator,
            "unique_gdp_delta_r_squared": unique_gdp,
            "gdp_background_vif": vif,
            "formula": linear_formula,
            "robust_covariance": "10-degree spatial-block cluster" if model.domain == "water_quality" else "20-degree spatial-block cluster",
        })
        partitions.append({
            "domain": model.domain,
            "endpoint": model.endpoint,
            "gdp_only_r_squared": float(gdp_only.rsquared),
            "background_only_r_squared": float(background.rsquared),
            "full_r_squared": float(fit_plain.rsquared),
            "full_adjusted_r_squared": float(fit_plain.rsquared_adj),
            "unique_gdp_r_squared": unique_gdp,
            "unique_background_r_squared": unique_background,
            "shared_or_suppressed_r_squared": shared,
            "unexplained_fraction": 1.0 - float(fit_plain.rsquared),
            "partial_gdp_r_squared": unique_gdp / denominator,
            "gdp_background_vif": vif,
        })

    effects = pd.DataFrame(effects)
    effects["q_value"] = bh_fdr(effects["p_value"])
    return effects, pd.DataFrame(partitions)


def integrated_curves(models: list[FittedEndpoint]) -> pd.DataFrame:
    orientation = {"DOSAT": 1, "EC": -1, "TP": -1, "Fish": 1, "EPTO": 1, "Algae": -1}
    rows = []
    # Use the three-endpoint common-support interval separately for water and
    # aquatic biota. This avoids extrapolation while retaining an informative
    # GDP span in the integrated panel.
    for domain in sorted({m.domain for m in models}):
        domain_models = [m for m in models if m.domain == domain]
        lower = max(float(m.data["gdppc"].quantile(0.05)) for m in domain_models)
        upper = min(float(m.data["gdppc"].quantile(0.95)) for m in domain_models)
        if not upper > lower:
            raise RuntimeError(f"No shared GDP range for {domain}")
        grid = np.geomspace(lower, upper, 180)
        center = float(np.exp(np.mean(np.log([lower, upper]))))
        for m in domain_models:
            terms = ["gdp_z", "gdp2"]
            beta = m.fit.params[terms].to_numpy()
            cov = m.fit.cov_params().loc[terms, terms].to_numpy()
            z = (np.log(grid) - m.gdp_mean) / m.gdp_sd
            z0 = (np.log(center) - m.gdp_mean) / m.gdp_sd
            X = np.column_stack([z - z0, z ** 2 - z0 ** 2])
            est = orientation[m.endpoint] * (X @ beta)
            var = np.maximum(0, np.einsum("ij,jk,ik->i", X, cov, X))
            for gdppc, estimate, variance in zip(grid, est, var):
                rows.append({
                    "domain": m.domain, "endpoint": m.endpoint, "gdppc": gdppc,
                    "oriented_change": estimate, "variance": variance,
                    "orientation": orientation[m.endpoint], "reference_gdppc": center,
                })
    endpoint = pd.DataFrame(rows)
    composite = endpoint.groupby(["domain", "gdppc"], as_index=False).agg(
        composite_index=("oriented_change", "mean"),
        variance_sum=("variance", "sum"), n_endpoints=("endpoint", "nunique"),
    )
    composite["se"] = np.sqrt(composite["variance_sum"]) / composite["n_endpoints"]
    composite["ci_low"] = composite["composite_index"] - 1.96 * composite["se"]
    composite["ci_high"] = composite["composite_index"] + 1.96 * composite["se"]
    return endpoint, composite


def summarize_variant(model: FittedEndpoint, fit, data: pd.DataFrame,
                      variant: str, terms: list[str]) -> dict:
    beta = fit.params[terms].to_numpy()
    cov = fit.cov_params().loc[terms, terms].to_numpy()
    q10, q90 = data["gdp_z"].quantile([0.1, 0.9])
    if terms == ["gdp_z", "gdp2"]:
        vector = np.array([q90 - q10, q90 ** 2 - q10 ** 2])
        restriction = "gdp_z = 0, gdp2 = 0"
    else:
        vector = np.array([q90 - q10])
        restriction = "gdp_z = 0"
    contrast = float(vector @ beta)
    se = float(np.sqrt(max(0, vector @ cov @ vector)))
    joint_p = float(np.asarray(fit.wald_test(restriction, scalar=True).pvalue).squeeze())
    return {
        "domain": model.domain, "endpoint": model.endpoint, "variant": variant,
        "n_observations": int(fit.nobs), "contrast_10_90": contrast,
        "ci_low": contrast - 1.96 * se, "ci_high": contrast + 1.96 * se,
        "joint_p": joint_p,
    }


def run_sensitivity(models: list[FittedEndpoint]) -> pd.DataFrame:
    rows = []
    for model in models:
        d = model.data.copy()
        if model.domain == "water_quality":
            station = smf.ols(model.formula, data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["unit_id"]}
            )
            rows.append(summarize_variant(model, station, d, "station-cluster SE", ["gdp_z", "gdp2"]))
        else:
            hc3 = smf.ols(model.formula, data=d).fit(cov_type="HC3")
            rows.append(summarize_variant(model, hc3, d, "HC3 SE", ["gdp_z", "gdp2"]))

        linear_formula = model.formula.replace("gdp_z + gdp2", "gdp_z")
        if model.domain == "water_quality":
            linear = smf.ols(linear_formula, data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["spatial_block_10"]}
            )
        else:
            linear = smf.ols(linear_formula, data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["spatial_block_20"]}
            )
        rows.append(summarize_variant(model, linear, d, "linear GDP term", ["gdp_z"]))

        filtered = None
        label = None
        if model.endpoint in ("DOSAT", "TP") and "outlier_fraction" in d:
            filtered = d.loc[d["outlier_fraction"].fillna(0).eq(0)].copy()
            label = "GRQA IQR-flagged station-years excluded"
        elif model.endpoint == "Algae" and "below_detection_fraction" in d:
            filtered = d.loc[d["below_detection_fraction"].fillna(0).eq(0)].copy()
            label = "chlorophyll detection-flagged lakes excluded"
        if filtered is not None and len(filtered) >= 100:
            group = filtered["spatial_block_10"] if model.domain == "water_quality" else filtered["spatial_block_20"]
            robust = smf.ols(model.formula, data=filtered).fit(
                cov_type="cluster", cov_kwds={"groups": group}
            )
            rows.append(summarize_variant(model, robust, filtered, label, ["gdp_z", "gdp2"]))

    out = pd.DataFrame(rows)
    out["joint_q_within_variant"] = np.nan
    for variant, idx in out.groupby("variant").groups.items():
        out.loc[idx, "joint_q_within_variant"] = bh_fdr(out.loc[idx, "joint_p"])
    return out


def run() -> None:
    ensure_dirs()
    models_and_meta = [
        fit_water("DOSAT", "identity"),
        fit_water("EC", "log"),
        fit_water("TP", "log"),
        fit_fish(), fit_epto(), fit_algae(),
    ]
    models = [x[0] for x in models_and_meta]
    transform_meta = [x[1] for x in models_and_meta]

    summaries, coefficients, curves, points, densities = [], [], [], [], []
    for model in models:
        summary, coef, curve, point, density = endpoint_outputs(model)
        summaries.append(summary)
        coefficients.append(coef)
        curves.append(curve)
        points.append(point)
        densities.append(density)

    summary = pd.DataFrame(summaries)
    summary["joint_gdp_q"] = bh_fdr(summary["joint_gdp_p"])
    summary["contrast_q"] = bh_fdr(summary["contrast_p"])
    coef = pd.concat(coefficients, ignore_index=True)
    coef["q_value_within_term"] = np.nan
    for term, idx in coef.groupby("term").groups.items():
        coef.loc[idx, "q_value_within_term"] = bh_fdr(coef.loc[idx, "p_value"])
    curve = pd.concat(curves, ignore_index=True)
    point = pd.concat(points, ignore_index=True)
    density = pd.concat(densities, ignore_index=True)
    endpoint_integrated, composite = integrated_curves(models)
    sensitivity = run_sensitivity(models)
    linear_effects, variance_partition = adjusted_linear_effects(models)
    diagnostics = pd.DataFrame([
        {
            "domain": model.domain, "endpoint": model.endpoint,
            "n_observations": int(model.fit.nobs), "n_parameters": int(len(model.fit.params)),
            "n_clusters": model.n_clusters, "condition_number": float(model.fit.condition_number),
            "residual_mean": float(np.mean(model.fit.resid)),
            "residual_sd": float(np.std(model.fit.resid, ddof=1)),
            "residual_skew": float(st.skew(model.fit.resid, bias=False)),
            "residual_excess_kurtosis": float(st.kurtosis(model.fit.resid, fisher=True, bias=False)),
            "robust_covariance": "10-degree spatial-block cluster" if model.domain == "water_quality" else "20-degree spatial-block cluster",
        }
        for model in models
    ])

    summary.to_csv(ANALYSIS / "primary_model_summary.csv", index=False, encoding="utf-8-sig")
    coef.to_csv(ANALYSIS / "primary_model_coefficients.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(SOURCE / "Fig_main_adjusted_curves.csv", index=False, encoding="utf-8-sig")
    point.to_csv(SOURCE / "Fig_main_observed_partial_residuals.csv.gz", index=False, compression="gzip")
    density.to_csv(SOURCE / "Fig_main_balanced_gdp_density.csv.gz", index=False, compression="gzip")
    endpoint_integrated.to_csv(SOURCE / "Fig_main_integrated_endpoint_curves.csv", index=False, encoding="utf-8-sig")
    composite.to_csv(SOURCE / "Fig_main_integrated_composite_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(transform_meta).to_csv(ANALYSIS / "transform_and_scaling_parameters.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(ANALYSIS / "sensitivity_model_summary.csv", index=False, encoding="utf-8-sig")
    linear_effects.to_csv(ANALYSIS / "natural_adjusted_linear_gdp_effects.csv", index=False, encoding="utf-8-sig")
    variance_partition.to_csv(ANALYSIS / "gdp_background_variance_partition.csv", index=False, encoding="utf-8-sig")
    linear_effects.to_csv(SOURCE / "Fig_main_panel_d_adjusted_GDP_effects.csv", index=False, encoding="utf-8-sig")
    variance_partition.to_csv(SOURCE / "Fig_main_panel_d_variance_partition.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(QA / "model_diagnostics.csv", index=False, encoding="utf-8-sig")
    # Transparent Benjamini--Hochberg audit. Equal adjusted q values can arise
    # from the required reverse cumulative minimum and are not duplicated p
    # values or a coding error.
    q_audit = linear_effects[["endpoint", "p_value", "q_value"]].sort_values("p_value").reset_index(drop=True)
    q_audit["rank"] = np.arange(1, len(q_audit) + 1)
    q_audit["m_tests"] = len(q_audit)
    q_audit["raw_bh_p_m_over_rank"] = q_audit["p_value"] * len(q_audit) / q_audit["rank"]
    q_audit["reverse_cumulative_minimum"] = (
        q_audit["raw_bh_p_m_over_rank"].iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    )
    q_audit.to_csv(QA / "BH_FDR_linear_GDP_effects_audit.csv", index=False, encoding="utf-8-sig")

    machine = {
        row["endpoint"]: {
            "n": int(row["n_observations"]), "units": int(row["n_independent_units"]),
            "contrast": row["gdp_10_90_contrast"], "ci": [row["contrast_ci_low"], row["contrast_ci_high"]],
            "joint_q": row["joint_gdp_q"], "delta_r2": row["delta_r_squared"],
        }
        for row in summary.to_dict("records")
    }
    (ANALYSIS / "model_summary.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(machine, ensure_ascii=False))


if __name__ == "__main__":
    run()
