from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
WATERS = ["River", "Lake"]
MODULES = ["terrain", "urbanization", "human_activity"]
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
TAXA = {
    "River": ["benthos", "fish", "macrophyte", "phytoplankton", "periphyton", "zooplankton"],
    "Lake": ["benthos", "fish", "macrophyte", "phytoplankton", "zooplankton"],
}
TAXON_CN = {
    "benthos": "底栖动物",
    "fish": "鱼类",
    "macrophyte": "水生植物",
    "phytoplankton": "浮游植物",
    "periphyton": "着生藻类",
    "zooplankton": "浮游动物",
}
MODULE_CN = {
    "terrain": "地形地貌",
    "urbanization": "城市化建设",
    "human_activity": "人类活动",
}
SUBVARIABLES = {
    "terrain": [
        ("mean_elevation", "平均高程", "Mean elevation", 1),
        ("relative_elevation", "相对高程位置", "Relative elevation", 1),
        ("terrain_relief", "地形起伏", "Terrain relief", 1),
        ("slope_intensity", "坡度强度", "Slope intensity", 1),
        ("steep_slope_cover", "陡坡覆盖", "Steep-slope cover", 1),
    ],
    "urbanization": [
        ("road_cover", "道路覆盖", "Road cover", 1),
        ("building_cover", "建筑覆盖", "Building cover", 1),
        ("traffic_facility_cover", "交通服务设施覆盖", "Traffic-facility cover", 1),
        ("transport_facility_cover", "运输设施覆盖", "Transport-facility cover", 1),
        ("urban_land_cover", "城市建设用地", "Urban land cover", 1),
        ("green_natural_buffer", "绿色/自然缓冲", "Green/natural buffer", -1),
    ],
    "human_activity": [
        ("api_expert_weighted", "加权API", "Expert-weighted API", 1),
        ("population_density", "人口密度", "Population density", 1),
    ],
}
EXPERT_WEIGHTS = {
    "P1": 1.60,
    "P2": 1.35,
    "P3": 1.00,
    "P4": 1.10,
    "P5": 0.95,
    "P6": 0.85,
    "P7": 0.35,
    "P8": 0.45,
}


def load_core(script_dir: Path):
    path = script_dir / "01_primary_multiscale_analysis.py"
    spec = importlib.util.spec_from_file_location("primary_core_subvars", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd


def average_z(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    cols = [c for c in columns if c in df and pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return pd.concat([zscore(df[c]) for c in cols], axis=1).mean(axis=1)


def range_tag(r: int) -> str:
    return f"{r:04d}m"


def build_weighted_api(site: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the prespecified expert-weighted API at all 13 frozen ranges.

    The original expert-weighted workbook contains 10 ranges and 127 sites.
    This reconstruction applies its published calculation rule to the frozen
    13-range raw P1-P8 matrix so that all 130 biological sites are retained.
    """
    rows = []
    weights = pd.Series(EXPERT_WEIGHTS, dtype=float)
    weights = weights / weights.sum()
    for r in RANGES:
        tag = range_tag(r)
        p_scores = {}
        for p in EXPERT_WEIGHTS:
            point = f"api_{tag}_{p}_point_density_per_km2"
            polygon = f"api_{tag}_{p}_polygon_area_pct"
            components = []
            if point in site and pd.to_numeric(site[point], errors="coerce").nunique() > 1:
                components.append(zscore(site[point]))
            if polygon in site and pd.to_numeric(site[polygon], errors="coerce").nunique() > 1:
                components.append(zscore(site[polygon]))
            score = pd.concat(components, axis=1).mean(axis=1) if components else pd.Series(np.nan, index=site.index)
            p_scores[p] = zscore(score)
        p_frame = pd.DataFrame(p_scores)
        api = p_frame.mul(weights, axis=1).sum(axis=1, min_count=4)
        for idx, value in api.items():
            rows.append(
                {
                    "row_index": idx,
                    "site": site.at[idx, "site"],
                    "water_body_type": site.at[idx, "water_body_type"],
                    "range_m": r,
                    "api_expert_weighted": value,
                }
            )
    return pd.DataFrame(rows)


def build_subvariable_long(site: pd.DataFrame, urban_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    api = build_weighted_api(site)
    urban = urban_long.copy()
    for src, dst in [
        ("road_area_m2", "road_cover"),
        ("building_area_m2", "building_cover"),
        ("traffic_area_m2", "traffic_facility_cover"),
        ("transport_area_m2", "transport_facility_cover"),
    ]:
        urban[dst] = 100 * pd.to_numeric(urban[src], errors="coerce") / pd.to_numeric(
            urban["analysis_land_area_m2"], errors="coerce"
        ).replace(0, np.nan)
    urban_keyed = urban.set_index(["water_body_type", "site_id", "buffer_m"])

    rows = []
    audit = []
    for water in WATERS:
        d = site[site.water_body_type.eq(water)].copy()
        for r in RANGES:
            tag = range_tag(r)
            terrain_values = {
                "mean_elevation": pd.to_numeric(d.get(f"terrain_{r}m_mean_dem_m"), errors="coerce"),
                "relative_elevation": pd.to_numeric(
                    d.get(f"terrain_{r}m_delta_dem_buffer_minus_point_m"), errors="coerce"
                ),
                "terrain_relief": average_z(
                    d, [f"terrain_{r}m_dem_range_m", f"terrain_{r}m_dem_std_m"]
                ),
                "slope_intensity": average_z(
                    d, [f"terrain_{r}m_mean_slope_pct", f"terrain_{r}m_max_slope_pct"]
                ),
                "steep_slope_cover": average_z(
                    d,
                    [
                        f"terrain_{r}m_slope_gt10pct_area_pct",
                        f"terrain_{r}m_slope_gt15pct_area_pct",
                    ],
                ),
            }
            urban_values = {}
            for key in [
                "road_cover",
                "building_cover",
                "traffic_facility_cover",
                "transport_facility_cover",
            ]:
                urban_values[key] = pd.Series(
                    [
                        urban_keyed.at[(water, sid, r), key]
                        if (water, sid, r) in urban_keyed.index
                        else np.nan
                        for sid in d.site
                    ],
                    index=d.index,
                )
            urban_values["urban_land_cover"] = pd.to_numeric(
                d.get(f"landuse_{tag}_urban_pct"), errors="coerce"
            )
            urban_values["green_natural_buffer"] = average_z(
                d,
                [
                    f"urbanization_{tag}_natural_land_pct",
                    f"landuse_{tag}_green_pct",
                ],
            )
            api_slice = api[
                api.water_body_type.eq(water) & api.range_m.eq(r)
            ].set_index("row_index")["api_expert_weighted"]
            human_values = {
                "api_expert_weighted": api_slice.reindex(d.index),
                "population_density": np.log1p(
                    pd.to_numeric(d.get(f"worldpop_{tag}_mean_density"), errors="coerce").clip(lower=0)
                ),
            }
            module_values = {
                "terrain": terrain_values,
                "urbanization": urban_values,
                "human_activity": human_values,
            }
            for module, definitions in SUBVARIABLES.items():
                for key, name_cn, name_en, orientation in definitions:
                    values = pd.to_numeric(module_values[module][key], errors="coerce")
                    for idx, value in values.items():
                        rows.append(
                            {
                                "row_index": idx,
                                "site": site.at[idx, "site"],
                                "water_body_type": water,
                                "range_m": r,
                                "environment_module": module,
                                "environment_module_cn": MODULE_CN[module],
                                "environment_subvariable": key,
                                "environment_subvariable_cn": name_cn,
                                "environment_subvariable_en": name_en,
                                "pressure_orientation_weight": orientation,
                                "value": value,
                            }
                        )
                    audit.append(
                        {
                            "water_body_type": water,
                            "range_m": r,
                            "environment_module": module,
                            "environment_module_cn": MODULE_CN[module],
                            "environment_subvariable": key,
                            "environment_subvariable_cn": name_cn,
                            "environment_subvariable_en": name_en,
                            "pressure_orientation_weight": orientation,
                            "n_sites": len(values),
                            "n_nonmissing": int(values.notna().sum()),
                            "n_unique": int(values.nunique(dropna=True)),
                            "min": float(values.min()) if values.notna().any() else np.nan,
                            "median": float(values.median()) if values.notna().any() else np.nan,
                            "max": float(values.max()) if values.notna().any() else np.nan,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(audit)


def api_reconstruction_validation(api_long: pd.DataFrame, improved_path: Path) -> pd.DataFrame:
    improved = pd.read_excel(improved_path, sheet_name="API_all_buffers_improved")
    available_ranges = sorted(set(improved.buffer_m.astype(int)) & set(RANGES))
    rec = api_long[api_long.range_m.isin(available_ranges)].copy()
    merged = improved.merge(
        rec,
        left_on=["site_id", "water_body_type", "buffer_m"],
        right_on=["site", "water_body_type", "range_m"],
        how="inner",
    )
    rows = []
    for r, g in merged.groupby("range_m"):
        rho = stats.spearmanr(
            g.API_expert_weighted, g.api_expert_weighted
        ).statistic
        rows.append(
            {
                "range_m": int(r),
                "n_matched_sites": len(g),
                "spearman_rho_rebuilt_vs_archived_expert_api": rho,
                "archived_site_n": int(improved[improved.buffer_m.eq(r)].site_id.nunique()),
                "frozen_raw_site_n": int(rec[rec.range_m.eq(r)].site.nunique()),
            }
        )
    return pd.DataFrame(rows)


def relative_weights(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Johnson relative weights for standardized residual-rank predictors."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    p = x.shape[1]
    if valid.sum() < p + 8:
        return np.full(p, np.nan), np.nan, np.nan
    xx = x[valid]
    yy = y[valid]
    xsd = xx.std(axis=0, ddof=1)
    if np.any(~np.isfinite(xsd)) or np.any(xsd <= 0):
        return np.full(p, np.nan), np.nan, np.nan
    xx = (xx - xx.mean(axis=0)) / xsd
    ysd = yy.std(ddof=1)
    if not np.isfinite(ysd) or ysd <= 0:
        return np.full(p, np.nan), np.nan, np.nan
    yy = (yy - yy.mean()) / ysd
    rxx = np.corrcoef(xx, rowvar=False)
    rxy = np.array([np.corrcoef(xx[:, j], yy)[0, 1] for j in range(p)])
    eigvals, eigvecs = np.linalg.eigh(rxx)
    eigvals = np.clip(eigvals, 1e-8, None)
    lam = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    beta = np.linalg.pinv(rxx, rcond=1e-8) @ rxy
    beta_star = lam @ beta
    weights = (lam * lam) @ (beta_star * beta_star)
    weights = np.clip(weights, 0, None)
    r2 = float(weights.sum())
    condition = float(np.sqrt(eigvals.max() / eigvals.min()))
    return weights, r2, condition


def prepare_residuals(
    site: pd.DataFrame,
    eligible_meta: pd.DataFrame,
    sub_long: pd.DataFrame,
    core,
):
    yres = {}
    xres = {}
    controls_by_water = {}
    for water in WATERS:
        d = site[site.water_body_type.eq(water)].copy()
        controls = core.controls_for(d)
        controls_by_water[water] = controls
        m = eligible_meta[eligible_meta.water_body_type.eq(water)]
        for c in m.column_name.unique():
            yres[(water, c)] = core.rank_residual(
                pd.to_numeric(d[c], errors="coerce"), controls
            )
        for module, definitions in SUBVARIABLES.items():
            for key, *_ in definitions:
                for r in RANGES:
                    s = sub_long[
                        sub_long.water_body_type.eq(water)
                        & sub_long.environment_module.eq(module)
                        & sub_long.environment_subvariable.eq(key)
                        & sub_long.range_m.eq(r)
                    ].set_index("row_index")["value"]
                    xres[(water, module, key, r)] = core.rank_residual(
                        s.reindex(d.index).to_numpy(float), controls
                    )
    return yres, xres, controls_by_water


def omnibus_unit(
    water: str,
    module: str,
    response_meta: pd.DataFrame,
    d: pd.DataFrame,
    yres: dict,
    xres: dict,
    core,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = SUBVARIABLES[module]
    subkeys = [x[0] for x in definitions]
    labels = {x[0]: x for x in definitions}
    mf, _ = core.select_family_representatives(d, response_meta)
    columns = list(mf.column_name)
    if not columns:
        return pd.DataFrame(), pd.DataFrame()
    Y = np.column_stack([yres[(water, c)] for c in columns])
    mask = np.isfinite(Y).astype(float)
    n_valid = mask.sum(axis=0)
    y_mean = np.nansum(Y, axis=0) / np.maximum(n_valid, 1)
    Y0 = np.where(np.isfinite(Y), Y - y_mean, 0.0)
    y_norm = np.sqrt(np.sum(Y0 * Y0, axis=0))
    valid_response = (n_valid >= 8) & (y_norm > 0)
    Y0 = Y0[:, valid_response]
    mask = mask[:, valid_response]
    n_valid = n_valid[valid_response]
    y_norm = y_norm[valid_response]
    columns = [c for c, keep in zip(columns, valid_response) if keep]
    if not columns:
        return pd.DataFrame(), pd.DataFrame()

    cell_keys = [(sub, r) for sub in subkeys for r in RANGES]
    X = np.column_stack([xres[(water, module, sub, r)] for sub, r in cell_keys])
    X = np.where(np.isfinite(X), X, 0.0)
    obs_rhos = np.full((len(columns), len(cell_keys)), np.nan)
    for j, (sub, r) in enumerate(cell_keys):
        for i, c in enumerate(columns):
            obs_rhos[i, j] = core.pearson_nan(
                xres[(water, module, sub, r)], yres[(water, c)]
            )[0]
    obs_stats = np.sqrt(np.nanmean(obs_rhos * obs_rhos, axis=0))
    perm_stats = np.full((n_perm, len(cell_keys)), np.nan)
    for b in range(n_perm):
        xp = X[rng.permutation(len(d)), :]
        numerator = Y0.T @ xp
        sum_x = mask.T @ xp
        sum_x2 = mask.T @ (xp * xp)
        x_ss = np.maximum(
            sum_x2 - (sum_x * sum_x) / np.maximum(n_valid[:, None], 1), 0
        )
        denom = y_norm[:, None] * np.sqrt(x_ss)
        rr = np.divide(
            numerator,
            denom,
            out=np.full_like(numerator, np.nan),
            where=denom > 0,
        )
        perm_stats[b] = np.sqrt(np.nanmean(rr * rr, axis=0))
    max_null = np.nanmax(perm_stats, axis=1)

    rows = []
    for j, (sub, r) in enumerate(cell_keys):
        vals = obs_rhos[:, j]
        obs = obs_stats[j]
        p_raw = (
            (1 + np.sum(perm_stats[:, j] >= obs)) / (n_perm + 1)
            if np.isfinite(obs)
            else np.nan
        )
        p_max = (
            (1 + np.sum(max_null >= obs)) / (n_perm + 1)
            if np.isfinite(obs)
            else np.nan
        )
        _, cn, en, orientation = labels[sub]
        rows.append(
            {
                "water_body_type": water,
                "environment_module": module,
                "environment_module_cn": MODULE_CN[module],
                "environment_subvariable": sub,
                "environment_subvariable_cn": cn,
                "environment_subvariable_en": en,
                "pressure_orientation_weight": orientation,
                "range_m": r,
                "n_response_variables": int(np.isfinite(vals).sum()),
                "n_response_variables_before_redundancy_control": len(response_meta),
                "rms_partial_rho": obs,
                "median_partial_rho": float(np.nanmedian(vals)),
                "pressure_aligned_median_partial_rho": float(np.nanmedian(vals)) * orientation,
                "positive_fraction": float(np.nanmean(vals > 0)),
                "permutation_p": p_raw,
                "subvariable_range_maxT_p": p_max,
            }
        )
    omnibus = pd.DataFrame(rows)
    omnibus["q_all_subvariables_ranges_within_unit"] = core.bh_fdr(
        omnibus.permutation_p
    )

    contribution_rows = []
    for r in RANGES:
        xmat = np.column_stack(
            [xres[(water, module, sub, r)] for sub in subkeys]
        )
        weights = []
        r2s = []
        conditions = []
        for c in columns:
            w, r2, condition = relative_weights(xmat, yres[(water, c)])
            weights.append(w)
            r2s.append(r2)
            conditions.append(condition)
        wmat = np.asarray(weights, float)
        mean_w = np.nanmean(wmat, axis=0)
        total = np.nansum(mean_w)
        shares = mean_w / total if np.isfinite(total) and total > 0 else np.full(len(subkeys), np.nan)
        rms_by_sub = omnibus[omnibus.range_m.eq(r)].set_index(
            "environment_subvariable"
        ).rms_partial_rho
        energies = np.array([rms_by_sub.get(s, np.nan) ** 2 for s in subkeys])
        etotal = np.nansum(energies)
        energy_shares = energies / etotal if etotal > 0 else np.full(len(subkeys), np.nan)
        for j, sub in enumerate(subkeys):
            contribution_rows.append(
                {
                    "water_body_type": water,
                    "environment_module": module,
                    "environment_subvariable": sub,
                    "range_m": r,
                    "mean_johnson_relative_weight": mean_w[j],
                    "johnson_relative_contribution_share": shares[j],
                    "marginal_rms2_contribution_share": energy_shares[j],
                    "mean_multivariable_R2_across_responses": float(np.nanmean(r2s)),
                    "median_condition_index": float(np.nanmedian(conditions)),
                    "n_relative_weight_response_models": int(
                        np.sum(np.isfinite(wmat[:, j]))
                    ),
                }
            )
    return omnibus, pd.DataFrame(contribution_rows)


def compute_response_associations(
    site: pd.DataFrame,
    eligible_meta: pd.DataFrame,
    yres: dict,
    xres: dict,
    controls_by_water: dict,
    core,
) -> pd.DataFrame:
    rows = []
    labels = {
        (module, key): (cn, en, orientation)
        for module, defs in SUBVARIABLES.items()
        for key, cn, en, orientation in defs
    }
    for water in WATERS:
        meta = eligible_meta[eligible_meta.water_body_type.eq(water)]
        k_controls = controls_by_water[water].shape[1]
        for _, m in meta.iterrows():
            for module, defs in SUBVARIABLES.items():
                for key, *_ in defs:
                    cn, en, orientation = labels[(module, key)]
                    for r in RANGES:
                        rho, n = core.pearson_nan(
                            xres[(water, module, key, r)],
                            yres[(water, m.column_name)],
                        )
                        p = core.partial_pvalue(rho, n, k_controls)
                        lo, hi = core.fisher_ci(rho, n, k_controls)
                        rows.append(
                            {
                                "water_body_type": water,
                                "environment_module": module,
                                "environment_module_cn": MODULE_CN[module],
                                "environment_subvariable": key,
                                "environment_subvariable_cn": cn,
                                "environment_subvariable_en": en,
                                "pressure_orientation_weight": orientation,
                                "response_family": m.response_family,
                                "response_family_cn": m.response_family_cn,
                                "taxon_group": m.taxon_group,
                                "taxon_group_cn": m.taxon_group_cn,
                                "column_name": m.column_name,
                                "variable_name_cn": m.variable_name_cn,
                                "range_m": r,
                                "partial_spearman_rho": rho,
                                "pressure_aligned_partial_rho": rho * orientation
                                if np.isfinite(rho)
                                else np.nan,
                                "ci95_low": lo,
                                "ci95_high": hi,
                                "p_value": p,
                                "n_sites": n,
                            }
                        )
    out = pd.DataFrame(rows)
    out["q_within_response_module_subvariables_ranges"] = np.nan
    for _, ix in out.groupby(
        ["water_body_type", "environment_module", "column_name"]
    ).groups.items():
        out.loc[ix, "q_within_response_module_subvariables_ranges"] = core.bh_fdr(
            out.loc[ix, "p_value"]
        )
    return out


def contiguous_range(values: pd.Series) -> tuple[str, int]:
    vals = sorted(set(pd.to_numeric(values, errors="coerce").dropna().astype(int)))
    if not vals:
        return "", 0
    pos = {r: i for i, r in enumerate(RANGES)}
    runs = [[vals[0]]]
    for r in vals[1:]:
        if pos.get(r, -9) == pos.get(runs[-1][-1], -99) + 1:
            runs[-1].append(r)
        else:
            runs.append([r])
    best = max(runs, key=len)
    return f"{best[0]}–{best[-1]}", len(best)


def summarize_trajectories(df: pd.DataFrame, unit_keys: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(unit_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values("range_m")
        peak = g.loc[g.rms_partial_rho.idxmax()]
        active = g.loc[g.rms_partial_rho.ge(0.15), "range_m"]
        response_range, response_n = contiguous_range(active)
        median_direction = float(np.nanmedian(g.median_partial_rho))
        if median_direction >= 0.05:
            direction = "正向占优"
        elif median_direction <= -0.05:
            direction = "负向占优"
        else:
            direction = "正负并存/近零"
        rows.append(
            {
                **dict(zip(unit_keys, keys)),
                "mean_rms_partial_rho": float(np.nanmean(g.rms_partial_rho)),
                "mean_median_partial_rho": float(np.nanmean(g.median_partial_rho)),
                "dominant_direction": direction,
                "mean_johnson_relative_contribution_share": float(
                    np.nanmean(g.johnson_relative_contribution_share)
                ),
                "peak_johnson_relative_contribution_share": float(
                    np.nanmax(g.johnson_relative_contribution_share)
                ),
                "peak_contribution_range_m": int(
                    g.loc[
                        g.johnson_relative_contribution_share.idxmax(), "range_m"
                    ]
                ),
                "peak_rms_partial_rho": float(peak.rms_partial_rho),
                "peak_effect_range_m": int(peak.range_m),
                "maxT_supported_range_n": int(
                    np.sum(g.subvariable_range_maxT_p <= 0.05)
                ),
                "maxT_supported_ranges": ";".join(
                    map(
                        str,
                        g.loc[
                            g.subvariable_range_maxT_p <= 0.05, "range_m"
                        ].astype(int),
                    )
                ),
                "response_range_m_abs_rms_ge_0.15": response_range,
                "response_range_n_abs_rms_ge_0.15": response_n,
                "mean_marginal_rms2_contribution_share": float(
                    np.nanmean(g.marginal_rms2_contribution_share)
                ),
                "importance_method_rank_concordance_spearman": stats.spearmanr(
                    g.johnson_relative_contribution_share,
                    g.marginal_rms2_contribution_share,
                    nan_policy="omit",
                ).statistic,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    raw_root = root.parent / "ML_Input_原始量_13尺度_2026-07-04"
    out = root / "14_三类环境子变量×生物五家族尺度分解_2026-07-27"
    for rel in [
        "00_配置与方法",
        "01_环境子变量审计",
        "02_五家族子变量尺度分解",
        "03_类群×五家族子变量尺度分解",
        "04_论文级图件/PNG",
        "04_论文级图件/PDF",
        "04_论文级图件/SVG",
        "04_论文级图件/TIFF",
        "05_SourceData",
        "06_QA",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    core = load_core(script_dir)
    site = pd.read_excel(
        raw_root / "ML_Input_Raw_13Ranges_Sample_Site.xlsx",
        sheet_name="Site_Special",
    )
    urban_long = pd.read_excel(
        raw_root / "Feature_Process_Urbanization_OSM_13Ranges.xlsx",
        sheet_name="Process_Long",
    )
    meta = core.read_dictionaries(raw_root)
    meta["taxon_group"] = meta.taxon_group.replace({"macrophytes": "macrophyte"})
    meta["taxon_group_cn"] = meta.taxon_group.map(TAXON_CN).fillna(
        meta.taxon_group_cn
    )
    availability = core.response_availability(site, meta)
    availability["taxon_group"] = availability.taxon_group.replace(
        {"macrophytes": "macrophyte"}
    )
    eligible_meta = []
    for water in WATERS:
        cols = set(
            availability.loc[
                availability.water_body_type.eq(water)
                & availability.primary_eligible,
                "column_name",
            ]
        )
        m = meta[meta.column_name.isin(cols)].copy()
        m["water_body_type"] = water
        eligible_meta.append(m)
    eligible_meta = pd.concat(eligible_meta, ignore_index=True)

    sub_long, audit = build_subvariable_long(site, urban_long)
    api_long = sub_long[
        sub_long.environment_subvariable.eq("api_expert_weighted")
    ][["row_index", "site", "water_body_type", "range_m", "value"]].rename(
        columns={"value": "api_expert_weighted"}
    )
    api_validation = api_reconstruction_validation(
        api_long,
        root.parents[1]
        / "OSM数据构建结果"
        / "POI_API_improved_all_buffers_20260602"
        / "improved_POI_API_all_buffers.xlsx",
    )

    sub_long.to_csv(
        out / "01_环境子变量审计" / "三类环境子变量_130站点_13尺度长表.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        out / "01_环境子变量审计" / "环境子变量可用性与分布审计.csv",
        index=False,
        encoding="utf-8-sig",
    )
    api_validation.to_csv(
        out / "01_环境子变量审计" / "加权API_重建与存档版本一致性.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eligible_meta.to_csv(
        out / "01_环境子变量审计" / "本分析生物响应变量字典.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yres, xres, controls = prepare_residuals(
        site, eligible_meta, sub_long, core
    )
    response_assoc = compute_response_associations(
        site, eligible_meta, yres, xres, controls, core
    )
    response_assoc.to_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "全部具体生物响应×环境子变量×13尺度_偏Spearman.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = np.random.default_rng(20260727)
    n_perm = 499
    family_omnibus = []
    family_contrib = []
    taxon_omnibus = []
    taxon_contrib = []
    for water in WATERS:
        d = site[site.water_body_type.eq(water)].copy()
        mw = eligible_meta[eligible_meta.water_body_type.eq(water)]
        for module in MODULES:
            for family in FAMILIES:
                mf = mw[mw.response_family.eq(family)]
                if not mf.empty:
                    om, ct = omnibus_unit(
                        water,
                        module,
                        mf,
                        d,
                        yres,
                        xres,
                        core,
                        rng,
                        n_perm,
                    )
                    if not om.empty:
                        om["response_family"] = family
                        om["response_family_cn"] = core.FAMILY_CN[family]
                        ct["response_family"] = family
                        ct["response_family_cn"] = core.FAMILY_CN[family]
                        family_omnibus.append(om)
                        family_contrib.append(ct)
            for taxon in TAXA[water]:
                for family in FAMILIES:
                    mf = mw[
                        mw.taxon_group.eq(taxon)
                        & mw.response_family.eq(family)
                    ]
                    if mf.empty:
                        continue
                    om, ct = omnibus_unit(
                        water,
                        module,
                        mf,
                        d,
                        yres,
                        xres,
                        core,
                        rng,
                        n_perm,
                    )
                    if not om.empty:
                        om["taxon_group"] = taxon
                        om["taxon_group_cn"] = TAXON_CN[taxon]
                        om["response_family"] = family
                        om["response_family_cn"] = core.FAMILY_CN[family]
                        ct["taxon_group"] = taxon
                        ct["taxon_group_cn"] = TAXON_CN[taxon]
                        ct["response_family"] = family
                        ct["response_family_cn"] = core.FAMILY_CN[family]
                        taxon_omnibus.append(om)
                        taxon_contrib.append(ct)

    family_omnibus = pd.concat(family_omnibus, ignore_index=True)
    family_contrib = pd.concat(family_contrib, ignore_index=True)
    family = family_omnibus.merge(
        family_contrib,
        on=[
            "water_body_type",
            "environment_module",
            "environment_subvariable",
            "range_m",
            "response_family",
            "response_family_cn",
        ],
        how="left",
    )
    taxon_omnibus = pd.concat(taxon_omnibus, ignore_index=True)
    taxon_contrib = pd.concat(taxon_contrib, ignore_index=True)
    taxon = taxon_omnibus.merge(
        taxon_contrib,
        on=[
            "water_body_type",
            "environment_module",
            "environment_subvariable",
            "range_m",
            "taxon_group",
            "taxon_group_cn",
            "response_family",
            "response_family_cn",
        ],
        how="left",
    )

    family.to_csv(
        out
        / "02_五家族子变量尺度分解"
        / "五家族×环境子变量×13尺度_共同置换与贡献份额.csv",
        index=False,
        encoding="utf-8-sig",
    )
    taxon.to_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "生物类群×五家族×环境子变量×13尺度_共同置换与贡献份额.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_summary = summarize_trajectories(
        family,
        [
            "water_body_type",
            "environment_module",
            "environment_module_cn",
            "response_family",
            "response_family_cn",
            "environment_subvariable",
            "environment_subvariable_cn",
            "environment_subvariable_en",
        ],
    )
    taxon_summary = summarize_trajectories(
        taxon,
        [
            "water_body_type",
            "environment_module",
            "environment_module_cn",
            "taxon_group",
            "taxon_group_cn",
            "response_family",
            "response_family_cn",
            "environment_subvariable",
            "environment_subvariable_cn",
            "environment_subvariable_en",
        ],
    )
    family_summary.to_csv(
        out / "02_五家族子变量尺度分解" / "五家族子变量尺度轨迹汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    taxon_summary.to_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "类群×五家族子变量尺度轨迹汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = (
        family.groupby(
            [
                "water_body_type",
                "environment_module",
                "environment_module_cn",
                "environment_subvariable",
                "environment_subvariable_cn",
                "environment_subvariable_en",
                "pressure_orientation_weight",
                "range_m",
            ],
            as_index=False,
        )
        .agg(
            family_equal_rms_partial_rho=(
                "rms_partial_rho",
                lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2))),
            ),
            family_equal_median_partial_rho=("median_partial_rho", "median"),
            family_equal_pressure_aligned_median_rho=(
                "pressure_aligned_median_partial_rho",
                "median",
            ),
            family_equal_johnson_contribution_share=(
                "johnson_relative_contribution_share",
                "mean",
            ),
            family_equal_marginal_rms2_share=(
                "marginal_rms2_contribution_share",
                "mean",
            ),
            maxT_supported_family_n=(
                "subvariable_range_maxT_p",
                lambda x: int(np.sum(np.asarray(x) <= 0.05)),
            ),
            evaluated_family_n=("response_family", "nunique"),
            median_condition_index=("median_condition_index", "median"),
        )
    )
    overall["family_equal_johnson_contribution_share"] = overall.groupby(
        ["water_body_type", "environment_module", "range_m"]
    )["family_equal_johnson_contribution_share"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.nan
    )
    overall.to_csv(
        out / "02_五家族子变量尺度分解" / "五家族等权_环境子变量总体贡献与方向.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_taxon = (
        taxon_summary.assign(
            ecological_priority=lambda x: x.mean_johnson_relative_contribution_share
            * x.mean_rms_partial_rho
        )
        .sort_values(
            [
                "water_body_type",
                "environment_module",
                "taxon_group",
                "response_family",
                "maxT_supported_range_n",
                "ecological_priority",
            ],
            ascending=[True, True, True, True, False, False],
        )
        .groupby(
            [
                "water_body_type",
                "environment_module",
                "taxon_group",
                "response_family",
            ],
            as_index=False,
        )
        .head(3)
    )
    top_taxon.to_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "每个类群×家族的前三位环境子变量.csv",
        index=False,
        encoding="utf-8-sig",
    )

    collinearity_rows = []
    for water in WATERS:
        for module, defs in SUBVARIABLES.items():
            keys = [x[0] for x in defs]
            for r in RANGES:
                mat = pd.DataFrame(
                    {
                        key: xres[(water, module, key, r)]
                        for key in keys
                    }
                )
                corr = mat.corr()
                tri = corr.where(
                    np.triu(np.ones(corr.shape), k=1).astype(bool)
                ).stack()
                max_corr = (
                    float(tri.abs().max()) if len(tri) else np.nan
                )
                eig = np.linalg.eigvalsh(
                    corr.fillna(0).to_numpy(float)
                )
                eig = np.clip(eig, 1e-8, None)
                collinearity_rows.append(
                    {
                        "water_body_type": water,
                        "environment_module": module,
                        "environment_module_cn": MODULE_CN[module],
                        "range_m": r,
                        "subvariable_n": len(keys),
                        "max_abs_spearman_between_subvariables": max_corr,
                        "condition_index": float(
                            np.sqrt(eig.max() / eig.min())
                        ),
                    }
                )
    collinearity = pd.DataFrame(collinearity_rows)
    collinearity.to_csv(
        out / "01_环境子变量审计" / "模块内子变量共线性_13尺度.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "analysis_date": "2026-07-27",
        "independent_unit": "site",
        "site_n": {"River": 86, "Lake": 44},
        "ranges_m": RANGES,
        "modules": MODULES,
        "subvariables": {
            module: [
                {
                    "key": key,
                    "name_cn": cn,
                    "name_en": en,
                    "pressure_orientation_weight": orientation,
                }
                for key, cn, en, orientation in defs
            ]
            for module, defs in SUBVARIABLES.items()
        },
        "api_expert_weights_raw": EXPERT_WEIGHTS,
        "api_rule": (
            "Within each range, z-score point density and polygon cover for each P group; "
            "average the two components; z-score P1-P8 group scores; combine using normalized "
            "expert weights. P1-P8 are not decomposed inferentially."
        ),
        "direction_statistic": (
            "Partial Spearman correlation after controlling longitude, latitude, squared terms, "
            "longitude-by-latitude and n_sample_records."
        ),
        "family_omnibus": (
            "RMS partial Spearman across nonredundant responses; signed median retains direction."
        ),
        "range_and_subvariable_control": (
            "499 common permutations with maxT across all subvariables and 13 nested ranges "
            "within each water × module × response unit."
        ),
        "contribution": (
            "Johnson relative weights on residual-rank predictors, averaged across nonredundant "
            "responses and normalized within module; descriptive allocation of shared explained "
            "variation, not unique causal attribution."
        ),
        "contribution_sensitivity": "Marginal RMS-squared share.",
        "urbanization_note": (
            "Road, building, traffic-facility and transport-facility covers are derived from the "
            "existing OSM Process_Long area fields. Total impervious cover is not entered together "
            "with its components to avoid deterministic double counting."
        ),
        "water_quality_scope": (
            "No deep water-quality subvariable analysis; ordinal water-quality grades remain "
            "comparative context only."
        ),
        "seed": 20260727,
        "permutations": n_perm,
    }
    (out / "00_配置与方法" / "环境子变量尺度分解分析配置.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "subvariable_long_rows": len(sub_long),
        "eligible_response_rows": len(eligible_meta),
        "response_association_rows": len(response_assoc),
        "family_omnibus_rows": len(family),
        "taxon_family_omnibus_rows": len(taxon),
        "family_summary_rows": len(family_summary),
        "taxon_summary_rows": len(taxon_summary),
        "api_validation_min_spearman": float(
            api_validation.spearman_rho_rebuilt_vs_archived_expert_api.min()
        ),
        "missing_urban_subvariable_cells": int(audit.n_nonmissing.lt(audit.n_sites).sum()),
    }
    (out / "06_QA" / "分析运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
