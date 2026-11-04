from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
FAMILIES = ["community_baseline", "alpha_diversity", "beta_diversity", "community_composition", "functional_indicator"]
TAXA = {
    "River": ["benthos", "fish", "macrophyte", "phytoplankton", "periphyton", "zooplankton"],
    "Lake": ["benthos", "fish", "macrophyte", "phytoplankton", "zooplankton"],
}
TAXON_CN = {
    "benthos": "底栖动物", "fish": "鱼类", "macrophyte": "水生植物",
    "phytoplankton": "浮游植物", "periphyton": "着生藻类", "zooplankton": "浮游动物",
}
DOMAINS = {
    "滨水区尺度（200–500 m）": (200, 500),
    "中尺度（1000–2500 m）": (1000, 2500),
    "景观尺度（3000–6000 m）": (3000, 6000),
}


def load_core(script_dir: Path):
    path = script_dir / "01_primary_multiscale_analysis.py"
    spec = importlib.util.spec_from_file_location("primary_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def normalize_taxon(series: pd.Series) -> pd.Series:
    return series.replace({"macrophytes": "macrophyte"})


def contiguous_runs(values):
    values = sorted(set(int(v) for v in values))
    if not values:
        return []
    pos = {r: i for i, r in enumerate(RANGES)}
    runs, current = [], [values[0]]
    for r in values[1:]:
        if pos[r] == pos[current[-1]] + 1:
            current.append(r)
        else:
            runs.append(current)
            current = [r]
    runs.append(current)
    return runs


def trajectory_summary(g: pd.DataFrame) -> dict:
    g = g.sort_values("range_m")
    rr = g.range_m.to_numpy(int)
    rms = g.rms_partial_rho.to_numpy(float)
    signed = g.median_partial_rho.to_numpy(float)
    active = np.isfinite(rms) & (rms >= 0.15)
    runs = contiguous_runs(rr[active])
    best = max(runs, key=len) if runs else []
    peak_i = int(np.nanargmax(rms)) if np.isfinite(rms).any() else 0
    strengths = {}
    for label, (lo, hi) in DOMAINS.items():
        m = (rr >= lo) & (rr <= hi)
        strengths[label] = float(np.nanmean(rms[m])) if m.any() else np.nan
    dominant = max(strengths, key=lambda k: -np.inf if not np.isfinite(strengths[k]) else strengths[k])
    has_pos = np.any(signed >= 0.10)
    has_neg = np.any(signed <= -0.10)
    if not active.any():
        shape = "弱响应/近零轨迹"
        scale_type = "无明确响应尺度"
    elif has_pos and has_neg:
        shape = "方向反转"
        scale_type = "跨尺度差异响应"
    elif len(best) >= 9:
        shape = "连续同向" if not (has_pos and has_neg) else "跨尺度异向"
        scale_type = "跨尺度连续响应"
    elif len(runs) >= 2:
        shape = "多峰或分段"
        scale_type = "跨尺度分段响应"
    elif 0 < peak_i < len(rms) - 1:
        shape = "中间峰值/转折"
        scale_type = dominant + "主导"
    else:
        shape = "局部单向"
        scale_type = dominant + "主导"
    positive = float(np.nanmedian(g.positive_fraction))
    if positive >= 0.65:
        direction = "正向占优"
    elif positive <= 0.35:
        direction = "负向占优"
    else:
        direction = "家族内正负并存"
    weights = np.where(np.isfinite(rms), rms, 0)
    centroid = float(np.sum(rr * weights) / np.sum(weights)) if np.sum(weights) > 0 else np.nan
    return {
        "dominant_scale_domain": dominant,
        "scale_response_type": scale_type,
        "trajectory_shape": shape,
        "direction_structure": direction,
        "median_positive_fraction": positive,
        "response_range_m": f"{best[0]}–{best[-1]}" if best else "",
        "response_range_n": len(best),
        "peak_range_m": int(rr[peak_i]),
        "peak_rms_partial_rho": float(rms[peak_i]),
        "peak_signed_median_rho": float(signed[peak_i]),
        "effect_weighted_scale_m": centroid,
        "nearshore_mean_rms": strengths["滨水区尺度（200–500 m）"],
        "mesoscale_mean_rms": strengths["中尺度（1000–2500 m）"],
        "landscape_mean_rms": strengths["景观尺度（3000–6000 m）"],
        "landscape_minus_nearshore_rms": strengths["景观尺度（3000–6000 m）"] - strengths["滨水区尺度（200–500 m）"],
        "maxT_supported_ranges": ";".join(map(str, g.loc[g.range_maxT_p <= 0.05, "range_m"].astype(int))),
        "broad_FDR_supported_ranges": ";".join(map(str, g.loc[g.q_module_all_taxon_family_ranges <= 0.05, "range_m"].astype(int))),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    out = script_dir.parent
    sub = out / "13_生物类群×五家族精细分析_2026-07-24"
    ml_root = out.parent / "ML_Input_原始量_13尺度_2026-07-04"
    core = load_core(script_dir)

    site = pd.read_excel(ml_root / "ML_Input_Raw_13Ranges_Sample_Site.xlsx", sheet_name="Site_Special")
    meta = core.read_dictionaries(ml_root)
    meta["taxon_group"] = normalize_taxon(meta.taxon_group)
    meta["taxon_group_cn"] = meta.taxon_group.map(TAXON_CN).fillna(meta.taxon_group_cn)
    availability = core.response_availability(site, meta)
    availability["taxon_group"] = normalize_taxon(availability.taxon_group)
    score_df, _, _ = core.build_module_scores(site)

    # Data audit: every defined taxon group is crossed with all five families.
    eligible_meta = []
    counts = []
    for water in ["River", "Lake"]:
        elig_cols = set(availability.loc[
            (availability.water_body_type == water) & availability.primary_eligible, "column_name"
        ])
        m = meta[meta.column_name.isin(elig_cols) & meta.taxon_group.isin(TAXA[water])].copy()
        m["water_body_type"] = water
        eligible_meta.append(m)
        for taxon in TAXA[water]:
            for family in FAMILIES:
                q = m[(m.taxon_group == taxon) & (m.response_family == family)]
                counts.append({
                    "water_body_type": water,
                    "taxon_group": taxon,
                    "taxon_group_cn": TAXON_CN[taxon],
                    "response_family": family,
                    "response_family_cn": core.FAMILY_CN[family],
                    "n_available_response_variables": len(q),
                    "has_structural_data": len(q) > 0,
                })
    eligible_meta = pd.concat(eligible_meta, ignore_index=True)
    counts = pd.DataFrame(counts)
    counts.to_csv(sub / "01_类群家族数据审计" / "River_Lake_生物类群×五家族可用变量数.csv", index=False, encoding="utf-8-sig")
    eligible_meta.to_csv(sub / "01_类群家族数据审计" / "类群家族主分析响应变量字典.csv", index=False, encoding="utf-8-sig")

    # Common-permutation omnibus analysis at taxon × family level.
    rng = np.random.default_rng(20260724)
    n_perm = 499
    rows, redundancy_rows = [], []
    for water in ["River", "Lake"]:
        d = site[site.water_body_type.eq(water)].copy()
        controls = core.controls_for(d)
        mwater = eligible_meta[eligible_meta.water_body_type.eq(water)]
        yres = {
            c: core.rank_residual(pd.to_numeric(d[c], errors="coerce"), controls)
            for c in mwater.column_name.unique()
        }
        for module in MODULES:
            score_mat = {}
            for r in RANGES:
                sc = score_df[
                    (score_df.water_body_type == water) &
                    (score_df.environment_module == module) &
                    (score_df.range_m == r)
                ].set_index("row_index").environment_score
                score_mat[r] = core.rank_residual(sc.reindex(d.index).to_numpy(float), controls)
            xscore = np.column_stack([score_mat[r] for r in RANGES])
            xscore = np.where(np.isfinite(xscore), xscore, 0.0)
            for taxon in TAXA[water]:
                for family in FAMILIES:
                    mf_all = mwater[(mwater.taxon_group == taxon) & (mwater.response_family == family)]
                    if mf_all.empty:
                        continue
                    mf, redmap = core.select_family_representatives(d, mf_all)
                    redmap["water_body_type"] = water
                    redmap["environment_module"] = module
                    redmap["taxon_group"] = taxon
                    redmap["taxon_group_cn"] = TAXON_CN[taxon]
                    redmap["response_family"] = family
                    redmap["response_family_cn"] = core.FAMILY_CN[family]
                    redundancy_rows.append(redmap)
                    columns = list(mf.column_name)
                    Y = np.column_stack([yres[c] for c in columns])
                    mask = np.isfinite(Y).astype(float)
                    n_valid = mask.sum(axis=0)
                    y_mean = np.nansum(Y, axis=0) / np.maximum(n_valid, 1)
                    Y0 = np.where(np.isfinite(Y), Y - y_mean, 0.0)
                    y_norm = np.sqrt(np.sum(Y0 * Y0, axis=0))
                    valid_response = (n_valid >= 8) & (y_norm > 0)
                    Y0, mask, n_valid, y_norm = (
                        Y0[:, valid_response], mask[:, valid_response],
                        n_valid[valid_response], y_norm[valid_response]
                    )
                    columns = [c for c, keep in zip(columns, valid_response) if keep]
                    obs_rhos = np.full((len(columns), len(RANGES)), np.nan)
                    for j, r in enumerate(RANGES):
                        for i, c in enumerate(columns):
                            obs_rhos[i, j] = core.pearson_nan(score_mat[r], yres[c])[0]
                    obs_stats = np.sqrt(np.nanmean(obs_rhos * obs_rhos, axis=0))
                    perm_stats = np.full((n_perm, len(RANGES)), np.nan)
                    for b in range(n_perm):
                        xp = xscore[rng.permutation(len(d)), :]
                        numerator = Y0.T @ xp
                        sum_x = mask.T @ xp
                        sum_x2 = mask.T @ (xp * xp)
                        x_ss = np.maximum(sum_x2 - (sum_x * sum_x) / np.maximum(n_valid[:, None], 1), 0)
                        denom = y_norm[:, None] * np.sqrt(x_ss)
                        rr = np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 0)
                        perm_stats[b] = np.sqrt(np.nanmean(rr * rr, axis=0))
                    max_null = np.nanmax(perm_stats, axis=1)
                    for j, r in enumerate(RANGES):
                        obs = obs_stats[j]
                        raw_p = (1 + np.sum(perm_stats[:, j] >= obs)) / (n_perm + 1)
                        max_p = (1 + np.sum(max_null >= obs)) / (n_perm + 1)
                        vals = obs_rhos[:, j]
                        rows.append({
                            "water_body_type": water,
                            "environment_module": module,
                            "environment_module_cn": core.MODULE_CN[module],
                            "taxon_group": taxon,
                            "taxon_group_cn": TAXON_CN[taxon],
                            "response_family": family,
                            "response_family_cn": core.FAMILY_CN[family],
                            "range_m": r,
                            "n_response_variables": int(np.isfinite(vals).sum()),
                            "n_response_variables_before_redundancy_control": len(mf_all),
                            "rms_partial_rho": obs,
                            "median_partial_rho": float(np.nanmedian(vals)),
                            "positive_fraction": float(np.nanmean(vals > 0)),
                            "permutation_p": raw_p,
                            "range_maxT_p": max_p,
                        })

    omnibus = pd.DataFrame(rows)
    omnibus["q_13ranges_within_taxon_family"] = np.nan
    for _, ix in omnibus.groupby(["water_body_type", "environment_module", "taxon_group", "response_family"]).groups.items():
        omnibus.loc[ix, "q_13ranges_within_taxon_family"] = core.bh_fdr(omnibus.loc[ix, "permutation_p"])
    omnibus["q_taxon_all_families_ranges"] = np.nan
    for _, ix in omnibus.groupby(["water_body_type", "environment_module", "taxon_group"]).groups.items():
        omnibus.loc[ix, "q_taxon_all_families_ranges"] = core.bh_fdr(omnibus.loc[ix, "permutation_p"])
    omnibus["q_module_all_taxon_family_ranges"] = np.nan
    for _, ix in omnibus.groupby(["water_body_type", "environment_module"]).groups.items():
        omnibus.loc[ix, "q_module_all_taxon_family_ranges"] = core.bh_fdr(omnibus.loc[ix, "permutation_p"])
    omnibus["scale_domain"] = pd.cut(
        omnibus.range_m, bins=[0, 500, 2500, 6000],
        labels=list(DOMAINS), include_lowest=True
    ).astype(str)
    omnibus.to_csv(
        sub / "02_类群家族整体尺度检验" / "生物类群×五家族×三环境模块_13尺度共同置换检验.csv",
        index=False, encoding="utf-8-sig"
    )
    redundancy = pd.concat(redundancy_rows, ignore_index=True).drop_duplicates()
    redundancy.to_csv(
        sub / "01_类群家族数据审计" / "类群家族整体检验_响应冗余控制映射.csv",
        index=False, encoding="utf-8-sig"
    )

    # Scale trajectories and cross-taxonomic response breadth.
    trajectories = []
    keys = ["water_body_type", "environment_module", "environment_module_cn",
            "taxon_group", "taxon_group_cn", "response_family", "response_family_cn"]
    for k, g in omnibus.groupby(keys):
        trajectories.append({**dict(zip(keys, k)), **trajectory_summary(g),
                             "n_response_variables": int(g.n_response_variables.max()),
                             "n_response_variables_before_redundancy_control": int(g.n_response_variables_before_redundancy_control.max())})
    trajectories = pd.DataFrame(trajectories)
    trajectories.to_csv(
        sub / "02_类群家族整体尺度检验" / "类群×家族_尺度轨迹与主导尺度汇总.csv",
        index=False, encoding="utf-8-sig"
    )
    breadth = omnibus.groupby(["water_body_type", "environment_module", "environment_module_cn",
                               "taxon_group", "taxon_group_cn"]).agg(
        mean_rms_all_family_ranges=("rms_partial_rho", "mean"),
        peak_rms=("rms_partial_rho", "max"),
        maxT_supported_cell_n=("range_maxT_p", lambda x: int(np.sum(np.asarray(x) <= 0.05))),
        broad_FDR_supported_cell_n=("q_module_all_taxon_family_ranges", lambda x: int(np.sum(np.asarray(x) <= 0.05))),
        supported_family_n=("response_family", lambda x: x[omnibus.loc[x.index, "range_maxT_p"] <= 0.05].nunique()),
    ).reset_index()
    breadth.to_csv(
        sub / "02_类群家族整体尺度检验" / "各类群对环境模块的响应广度.csv",
        index=False, encoding="utf-8-sig"
    )

    # Energy contribution: which taxon carries the aggregate response at each scale.
    energy = omnibus.copy()
    energy["family_equal_energy"] = energy.rms_partial_rho.pow(2)
    energy["variable_weighted_energy"] = energy.n_response_variables * energy.rms_partial_rho.pow(2)
    taxon_energy = energy.groupby(
        ["water_body_type", "environment_module", "environment_module_cn", "range_m", "taxon_group", "taxon_group_cn"],
        as_index=False
    ).agg(
        family_equal_energy=("family_equal_energy", "sum"),
        variable_weighted_energy=("variable_weighted_energy", "sum"),
    )
    equal_totals = taxon_energy.groupby(
        ["water_body_type", "environment_module", "range_m"]
    ).family_equal_energy.transform("sum")
    weighted_totals = taxon_energy.groupby(
        ["water_body_type", "environment_module", "range_m"]
    ).variable_weighted_energy.transform("sum")
    taxon_energy["taxon_response_energy_share"] = taxon_energy.family_equal_energy / equal_totals
    taxon_energy["taxon_variable_weighted_energy_share"] = taxon_energy.variable_weighted_energy / weighted_totals
    taxon_energy["is_dominant_taxon_at_range"] = taxon_energy.groupby(
        ["water_body_type", "environment_module", "range_m"]
    ).family_equal_energy.transform("max").eq(taxon_energy.family_equal_energy)
    taxon_energy.to_csv(
        sub / "02_类群家族整体尺度检验" / "各尺度总体响应的类群贡献份额.csv",
        index=False, encoding="utf-8-sig"
    )
    taxon_energy["scale_domain"] = pd.cut(
        taxon_energy.range_m,
        bins=[0, 500, 2500, np.inf],
        labels=["滨水区尺度（200–500 m）", "中尺度（1000–2500 m）", "景观尺度（3000–6000 m）"],
        include_lowest=True,
    )
    domain_energy = taxon_energy.groupby(
        [
            "water_body_type", "environment_module", "environment_module_cn",
            "scale_domain", "taxon_group", "taxon_group_cn",
        ],
        observed=True,
        as_index=False,
    ).agg(
        mean_taxon_response_energy_share=("taxon_response_energy_share", "mean"),
        min_taxon_response_energy_share=("taxon_response_energy_share", "min"),
        max_taxon_response_energy_share=("taxon_response_energy_share", "max"),
        mean_taxon_variable_weighted_energy_share=("taxon_variable_weighted_energy_share", "mean"),
    )
    domain_energy["domain_rank"] = domain_energy.groupby(
        ["water_body_type", "environment_module", "scale_domain"],
        observed=True,
    ).mean_taxon_response_energy_share.rank(method="dense", ascending=False).astype(int)
    domain_energy.to_csv(
        sub / "02_类群家族整体尺度检验" / "类群贡献份额_三尺度域汇总.csv",
        index=False, encoding="utf-8-sig"
    )

    # Existing response-wise statistics are reorganized within the defined taxa.
    assoc = pd.read_csv(out / "05_Q4_具体生物响应" / "全部具体响应_偏Spearman_分层FDR.csv")
    assoc["taxon_group"] = normalize_taxon(assoc.taxon_group)
    assoc = assoc[
        ((assoc.water_body_type == "River") & assoc.taxon_group.isin(TAXA["River"])) |
        ((assoc.water_body_type == "Lake") & assoc.taxon_group.isin(TAXA["Lake"]))
    ].copy()
    assoc = assoc.merge(
        omnibus[[
            "water_body_type", "environment_module", "taxon_group", "response_family", "range_m",
            "rms_partial_rho", "range_maxT_p", "q_taxon_all_families_ranges",
            "q_module_all_taxon_family_ranges"
        ]].rename(columns={
            "rms_partial_rho": "taxon_family_rms_partial_rho",
            "range_maxT_p": "taxon_family_range_maxT_p",
        }),
        on=["water_body_type", "environment_module", "taxon_group", "response_family", "range_m"],
        how="left"
    )
    assoc["joint_specific_taxon_family_support"] = (
        (assoc.q_within_response_13ranges <= 0.10) &
        (assoc.taxon_family_range_maxT_p <= 0.05)
    )
    assoc.to_csv(
        sub / "03_类群内具体响应" / "类群内全部具体响应_偏Spearman_13尺度.csv",
        index=False, encoding="utf-8-sig"
    )
    candidates = pd.read_csv(out / "05_Q4_具体生物响应" / "具体响应尺度类型与候选清单.csv")
    col_taxon = meta.drop_duplicates("column_name").set_index("column_name").taxon_group
    candidates["taxon_group"] = candidates.column_name.map(col_taxon)
    candidates["taxon_group_cn"] = candidates.taxon_group.map(TAXON_CN)
    candidates = candidates.dropna(subset=["taxon_group"]).copy()
    candidates["is_ordination_axis"] = candidates.column_name.str.contains("pcoa", case=False, na=False)
    joint = assoc.groupby(
        ["water_body_type", "environment_module", "taxon_group", "response_family", "column_name"],
        as_index=False
    ).agg(
        taxon_family_maxT_supported_n=("taxon_family_range_maxT_p", lambda x: int(np.sum(np.asarray(x) <= 0.05))),
        joint_specific_taxon_family_supported_n=("joint_specific_taxon_family_support", "sum"),
        taxon_family_peak_rms=("taxon_family_rms_partial_rho", "max"),
    )
    joint_ranges = (
        assoc[assoc.joint_specific_taxon_family_support]
        .groupby(["water_body_type", "environment_module", "taxon_group", "response_family", "column_name"])
        .range_m.apply(lambda x: ";".join(map(str, sorted(x.astype(int).unique()))))
        .rename("joint_supported_ranges").reset_index()
    )
    joint = joint.merge(
        joint_ranges,
        on=["water_body_type", "environment_module", "taxon_group", "response_family", "column_name"],
        how="left"
    )
    candidates = candidates.merge(
        joint,
        on=["water_body_type", "environment_module", "taxon_group", "response_family", "column_name"],
        how="left"
    )
    candidates["joint_supported_ranges"] = candidates.joint_supported_ranges.fillna("")
    candidates.to_csv(
        sub / "03_类群内具体响应" / "类群内具体响应_尺度类型完整表.csv",
        index=False, encoding="utf-8-sig"
    )
    representative = (
        candidates.sort_values(
            ["water_body_type", "environment_module", "taxon_group", "response_family",
             "joint_specific_taxon_family_supported_n", "taxon_family_maxT_supported_n",
             "screen_rank", "is_ordination_axis", "max_abs_rho"],
            ascending=[True, True, True, True, False, False, True, True, False]
        )
        .groupby(["water_body_type", "environment_module", "taxon_group", "response_family"], as_index=False)
        .head(3)
    )
    representative.to_csv(
        sub / "03_类群内具体响应" / "类群×家族×环境模块_前三位代表具体响应.csv",
        index=False, encoding="utf-8-sig"
    )

    # River–Lake descriptive comparison for the five common taxa.
    common = trajectories[trajectories.taxon_group.ne("periphyton")].copy()
    river = common[common.water_body_type.eq("River")].drop(columns=["water_body_type"]).add_prefix("river_")
    lake = common[common.water_body_type.eq("Lake")].drop(columns=["water_body_type"]).add_prefix("lake_")
    merge_keys_r = ["river_environment_module", "river_taxon_group", "river_response_family"]
    merge_keys_l = ["lake_environment_module", "lake_taxon_group", "lake_response_family"]
    comparison = river.merge(lake, left_on=merge_keys_r, right_on=merge_keys_l, how="inner")
    comparison["peak_rms_difference_river_minus_lake"] = (
        comparison.river_peak_rms_partial_rho - comparison.lake_peak_rms_partial_rho
    )
    comparison["weighted_scale_difference_m"] = (
        comparison.river_effect_weighted_scale_m - comparison.lake_effect_weighted_scale_m
    )
    comparison["dominant_scale_domain_concordant"] = (
        comparison.river_dominant_scale_domain == comparison.lake_dominant_scale_domain
    )
    comparison.to_csv(
        sub / "02_类群家族整体尺度检验" / "River_Lake共同类群×家族尺度指纹对照.csv",
        index=False, encoding="utf-8-sig"
    )

    # Excel digest.
    with pd.ExcelWriter(sub / "生物类群×五家族_13尺度核心结果.xlsx", engine="openpyxl") as xw:
        counts.to_excel(xw, sheet_name="类群家族可用性", index=False)
        omnibus.to_excel(xw, sheet_name="类群家族13尺度检验", index=False)
        trajectories.to_excel(xw, sheet_name="尺度轨迹汇总", index=False)
        breadth.to_excel(xw, sheet_name="类群响应广度", index=False)
        taxon_energy.to_excel(xw, sheet_name="类群逐尺度贡献", index=False)
        domain_energy.to_excel(xw, sheet_name="类群三尺度域贡献", index=False)
        representative.to_excel(xw, sheet_name="代表具体响应", index=False)
        assoc.to_excel(xw, sheet_name="具体响应13尺度", index=False)
        comparison.to_excel(xw, sheet_name="River_Lake指纹对照", index=False)

    config = {
        "analysis_date": "2026-07-24",
        "independent_unit": "site",
        "site_n": {"River": 86, "Lake": 44},
        "taxa": TAXA,
        "families": FAMILIES,
        "modules": MODULES,
        "ranges_m": RANGES,
        "permutations": n_perm,
        "seed": 20260724,
        "omnibus": "RMS partial Spearman across nonredundant response variables within each taxon × family",
        "spatial_controls": ["longitude", "latitude", "longitude^2", "latitude^2", "longitude×latitude", "n_sample_records"],
        "nested_range_control": "common-permutation maxT across 13 nested ranges",
        "multiplicity": [
            "BH within each taxon × family across 13 ranges",
            "BH within each taxon across five families × 13 ranges",
            "BH within each water × module across all taxon × family × range cells",
        ],
        "structural_missingness": "Lake periphyton is absent and never imputed; overall/other responses are excluded from taxon-specific inference.",
    }
    (sub / "00_配置与图件契约" / "taxon_family_analysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "omnibus_rows": len(omnibus),
        "trajectory_rows": len(trajectories),
        "specific_association_rows": len(assoc),
        "representative_rows": len(representative),
        "redundancy_map_rows": len(redundancy),
        "expected_crosses": {"River": 6 * 5 * 3 * 13, "Lake": 5 * 5 * 3 * 13},
    }
    (sub / "06_QA" / "分析运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
