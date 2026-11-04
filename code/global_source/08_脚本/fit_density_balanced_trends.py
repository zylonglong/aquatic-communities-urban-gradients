"""Select and fit density-balanced GDP response curves for the six endpoints.

This is a descriptive trend layer for figure panels e--f. Each independent
unit contributes equal total weight and fixed-width GDP bins are reweighted
towards equal mass, with a conservative cap. Linear, quadratic and restricted
cubic B-spline candidates are compared using spatial-block cross-validation; the
simplest model within one standard error of the best score is retained.

The procedure never selects a model from the sign of its fitted trend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from patsy import build_design_matrices

from fit_global_macro_models import (
    ANALYSIS,
    SOURCE,
    FittedEndpoint,
    bh_fdr,
    fit_algae,
    fit_epto,
    fit_fish,
    fit_water,
)


RNG = np.random.default_rng(20260915)


def density_balanced_weights(data: pd.DataFrame, n_bins: int = 20, cap: float = 12.0) -> pd.Series:
    """Equalize independent-unit and broad GDP-gradient contributions."""
    d = data
    records_per_unit = d.groupby("unit_id")["unit_id"].transform("size").astype(float)
    unit_weight = 1.0 / records_per_unit
    edges = np.linspace(float(d["gdp_z"].min()), float(d["gdp_z"].max()), n_bins + 1)
    bin_id = pd.cut(d["gdp_z"], bins=edges, include_lowest=True, labels=False)
    bin_mass = pd.Series(unit_weight).groupby(bin_id).sum()
    target_mass = float(bin_mass[bin_mass > 0].median())
    multiplier = bin_id.map((target_mass / bin_mass).to_dict()).astype(float)
    weights = unit_weight.to_numpy() * multiplier.to_numpy()
    positive = weights[np.isfinite(weights) & (weights > 0)]
    ceiling = min(float(np.quantile(positive, 0.99)), float(np.median(positive) * cap))
    weights = np.minimum(weights, ceiling)
    weights = weights / np.mean(weights)
    return pd.Series(weights, index=d.index, name="density_weight")


def formulas(model: FittedEndpoint) -> dict[str, str]:
    lo, hi = float(model.data["gdp_z"].min()), float(model.data["gdp_z"].max())
    background = model.formula.replace("gdp_z + gdp2 + ", "").split("~", 1)[1].strip()
    if "bs(abs_lat, df=4, degree=3, include_intercept=False)" in background:
        lat_lo = float(model.data["abs_lat"].min())
        lat_hi = float(model.data["abs_lat"].max())
        background = background.replace(
            "bs(abs_lat, df=4, degree=3, include_intercept=False)",
            "bs(abs_lat, df=4, degree=3, include_intercept=False, "
            f"lower_bound={lat_lo!r}, upper_bound={lat_hi!r})",
        )
    spline3 = (
        f"bs(gdp_z, df=3, degree=3, include_intercept=False, "
        f"lower_bound={lo!r}, upper_bound={hi!r})"
    )
    spline4 = (
        f"bs(gdp_z, df=4, degree=3, include_intercept=False, "
        f"lower_bound={lo!r}, upper_bound={hi!r})"
    )
    return {
        "linear": f"y_z ~ gdp_z + {background}",
        "quadratic": f"y_z ~ gdp_z + gdp2 + {background}",
        "spline_df3": f"y_z ~ {spline3} + {background}",
        "spline_df4": f"y_z ~ {spline4} + {background}",
    }


def fold_assignments(data: pd.DataFrame, group_col: str, n_folds: int = 5) -> pd.Series:
    blocks = np.array(sorted(data[group_col].astype(str).unique()))
    RNG.shuffle(blocks)
    mapping = {block: i % n_folds for i, block in enumerate(blocks)}
    return data[group_col].astype(str).map(mapping).astype(int)


def blocked_cv(model: FittedEndpoint, candidates: dict[str, str]) -> tuple[pd.DataFrame, str]:
    d = model.data.copy()
    # Preserve the full-sample factor levels in every spatial fold. Without
    # this, a held-out block containing a category absent from its training
    # folds cannot be represented by patsy's design matrix.
    for column in ("continent", "source", "realm"):
        if column in d.columns:
            levels = sorted(d[column].astype(str).unique())
            d[column] = pd.Categorical(d[column].astype(str), categories=levels)
    d["density_weight"] = density_balanced_weights(d)
    group_col = "spatial_block_10" if model.domain == "water_quality" else "spatial_block_20"
    d["cv_fold"] = fold_assignments(d, group_col)
    rows = []
    for name, formula in candidates.items():
        for fold in sorted(d["cv_fold"].unique()):
            train = d.loc[d["cv_fold"] != fold]
            test = d.loc[d["cv_fold"] == fold]
            error = ""
            try:
                fit = smf.wls(formula, data=train, weights=train["density_weight"]).fit()
                predicted = fit.predict(test)
                residual = test["y_z"].to_numpy(float) - predicted.to_numpy(float)
                rmse = float(np.sqrt(np.average(residual ** 2, weights=test["density_weight"])))
            except Exception as exc:
                rmse = np.nan
                error = f"{type(exc).__name__}: {exc}"
            rows.append({
                "domain": model.domain, "endpoint": model.endpoint,
                "candidate": name, "fold": int(fold), "weighted_rmse": rmse,
                "n_train": len(train), "n_test": len(test), "error": error,
            })
    detail = pd.DataFrame(rows)
    scores = detail.groupby("candidate", as_index=False).agg(
        mean_rmse=("weighted_rmse", "mean"),
        sd_rmse=("weighted_rmse", "std"),
        valid_folds=("weighted_rmse", "count"),
    )
    scores["se_rmse"] = scores["sd_rmse"] / np.sqrt(scores["valid_folds"])
    if (scores["valid_folds"] < d["cv_fold"].nunique()).any():
        failed = scores.loc[scores["valid_folds"] < d["cv_fold"].nunique(), "candidate"].tolist()
        messages = detail.loc[detail["error"].ne(""), ["candidate", "fold", "error"]].to_dict("records")
        raise RuntimeError(
            f"Incomplete spatial-block CV for {model.endpoint}: {failed}; errors={messages[:8]}"
        )
    best = scores.loc[scores["mean_rmse"].idxmin()]
    threshold = float(best["mean_rmse"] + best["se_rmse"])
    complexity = ["linear", "quadratic", "spline_df3", "spline_df4"]
    eligible = set(scores.loc[scores["mean_rmse"] <= threshold, "candidate"])
    selected = next(name for name in complexity if name in eligible)
    detail = detail.merge(scores, on="candidate", how="left")
    detail["one_se_threshold"] = threshold
    detail["selected"] = detail["candidate"].eq(selected)
    return detail, selected


def gdp_columns(names: list[str]) -> list[int]:
    return [i for i, name in enumerate(names) if "gdp_z" in name or name == "gdp2"]


def prediction_matrix(fit, data: pd.DataFrame, grid_z: np.ndarray) -> tuple[np.ndarray, list[int]]:
    names = list(fit.model.exog_names)
    idx = gdp_columns(names)
    template = pd.concat([data.iloc[[0]]] * len(grid_z), ignore_index=True)
    template["gdp_z"] = grid_z
    template["gdp2"] = grid_z ** 2
    built = np.asarray(build_design_matrices([fit.model.data.design_info], template)[0], dtype=float)
    xbar = np.asarray(fit.model.exog, dtype=float).mean(axis=0)
    xgrid = np.tile(xbar, (len(grid_z), 1))
    xgrid[:, idx] = built[:, idx]
    return xgrid, idx


def fit_selected(model: FittedEndpoint, formula: str, selected: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = model.data.copy()
    d["density_weight"] = density_balanced_weights(d)
    group_col = "spatial_block_10" if model.domain == "water_quality" else "spatial_block_20"
    fit = smf.wls(formula, data=d, weights=d["density_weight"]).fit(
        cov_type="cluster", cov_kwds={"groups": d[group_col]}
    )
    unadjusted = smf.wls("y_z ~ gdp_z", data=d, weights=d["density_weight"]).fit(
        cov_type="cluster", cov_kwds={"groups": d[group_col]}
    )
    background_formula = model.formula.replace("gdp_z + gdp2 + ", "")
    reduced = smf.wls(background_formula, data=d, weights=d["density_weight"]).fit()
    grid_z = np.linspace(float(d["gdp_z"].min()), float(d["gdp_z"].max()), 240)
    xgrid, idx = prediction_matrix(fit, d, grid_z)
    names = list(fit.model.exog_names)
    params = fit.params.reindex(names).to_numpy(float)
    cov = fit.cov_params().loc[names, names].to_numpy(float)
    predicted = xgrid @ params
    se = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", xgrid, cov, xgrid)))
    curve = pd.DataFrame({
        "domain": model.domain, "endpoint": model.endpoint, "selected_model": selected,
        "gdp_z": grid_z, "gdppc": np.exp(grid_z * model.gdp_sd + model.gdp_mean),
        "predicted_partial": predicted, "se": se,
        "ci_low": predicted - 1.96 * se, "ci_high": predicted + 1.96 * se,
        "prediction_definition": "unit- and GDP-density-balanced population-standardized marginal prediction",
    })

    xobs = np.tile(np.asarray(fit.model.exog, dtype=float).mean(axis=0), (len(d), 1))
    xobs[:, idx] = np.asarray(fit.model.exog, dtype=float)[:, idx]
    partial = np.asarray(fit.resid) + xobs @ params
    point_cols = ["unit_id", "gdppc", "value", "latitude", "longitude", "gdp_z"]
    if "year" in d:
        point_cols.append("year")
    points = d[point_cols].copy()
    points["domain"] = model.domain
    points["endpoint"] = model.endpoint
    points["partial_response"] = partial
    points["density_weight"] = d["density_weight"].to_numpy(float)

    q10, q90 = d.groupby("unit_id")["gdp_z"].median().quantile([0.1, 0.9])
    x_contrast, _ = prediction_matrix(fit, d, np.array([q10, q90], dtype=float))
    vector = x_contrast[1] - x_contrast[0]
    contrast = float(vector @ params)
    contrast_se = float(np.sqrt(max(0, vector @ cov @ vector)))
    restriction = np.zeros((len(idx), len(names)))
    for row, column in enumerate(idx):
        restriction[row, column] = 1.0
    joint_p = float(np.asarray(fit.wald_test(restriction, scalar=True).pvalue).squeeze())
    summary = {
        "domain": model.domain, "endpoint": model.endpoint, "selected_model": selected,
        "n_observations": int(fit.nobs), "n_independent_units": int(model.n_units),
        "n_spatial_clusters": int(d[group_col].nunique()),
        "contrast_10_90": contrast, "contrast_se": contrast_se,
        "contrast_ci_low": contrast - 1.96 * contrast_se,
        "contrast_ci_high": contrast + 1.96 * contrast_se,
        "joint_p": joint_p, "weighted_r_squared": float(fit.rsquared),
        "weighted_adjusted_r_squared": float(fit.rsquared_adj),
        "weighted_delta_r_squared": float(fit.rsquared - reduced.rsquared),
        "density_balanced_unadjusted_beta": float(unadjusted.params["gdp_z"]),
        "density_balanced_unadjusted_p": float(unadjusted.pvalues["gdp_z"]),
        "density_balanced_unadjusted_r_squared": float(unadjusted.rsquared),
        "weight_min": float(d["density_weight"].min()),
        "weight_median": float(d["density_weight"].median()),
        "weight_max": float(d["density_weight"].max()),
        "formula": formula,
    }
    return summary, curve, points


def run() -> None:
    models = [
        fit_water("DOSAT", "identity")[0],
        fit_water("EC", "log")[0],
        fit_water("TP", "log")[0],
        fit_fish()[0], fit_epto()[0], fit_algae()[0],
    ]
    cv_rows, summaries, curves, points = [], [], [], []
    for model in models:
        candidate_formulas = formulas(model)
        cv, selected = blocked_cv(model, candidate_formulas)
        summary, curve, point = fit_selected(model, candidate_formulas[selected], selected)
        cv_rows.append(cv)
        summaries.append(summary)
        curves.append(curve)
        points.append(point)
    summary = pd.DataFrame(summaries)
    summary["joint_q"] = bh_fdr(summary["joint_p"])
    summary["contrast_q"] = bh_fdr(
        pd.Series([
            2 * st.norm.sf(abs(row.contrast_10_90 / row.contrast_se))
            if row.contrast_se > 0 else np.nan
            for row in summary.itertuples()
        ])
    )
    cv = pd.concat(cv_rows, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True)
    point = pd.concat(points, ignore_index=True)
    summary.to_csv(ANALYSIS / "density_balanced_selected_trend_summary.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(ANALYSIS / "density_balanced_spatial_block_cv.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(SOURCE / "Fig_main_density_balanced_curves.csv", index=False, encoding="utf-8-sig")
    point.to_csv(SOURCE / "Fig_main_density_balanced_partial_residuals.csv.gz", index=False, compression="gzip")
    print(json.dumps(summary[["endpoint", "selected_model", "contrast_10_90", "joint_p", "joint_q"]].to_dict("records")))


if __name__ == "__main__":
    run()
