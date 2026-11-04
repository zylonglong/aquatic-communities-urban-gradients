from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_CN = {"terrain": "地形地貌", "urbanization": "城市化建设", "human_activity": "人类直接活动"}
DRIVER_TO_MODULE = {v: k for k, v in MODULE_CN.items()}
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
FAMILY_CN = {
    "community_baseline": "群落基础特征",
    "alpha_diversity": "α多样性",
    "beta_diversity": "β多样性",
    "community_composition": "群落组成",
    "functional_indicator": "功能与指示群",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile_rank(values: pd.Series) -> pd.Series:
    n = int(values.notna().sum())
    if n <= 1:
        return pd.Series(np.nan, index=values.index)
    return (values.rank(method="average") - 1) / (n - 1)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return np.nan
    return float(frame.x.rank().corr(frame.y.rank()))


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def top_set(frame: pd.DataFrame, value: str, key: str, fraction: float = 0.25) -> set:
    k = max(1, int(math.ceil(len(frame) * fraction)))
    return set(frame.sort_values([value, key], ascending=[False, True]).head(k)[key])


def jaccard(a: set, b: set) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def range_text(values: pd.Series) -> str:
    vals = sorted(set(int(v) for v in values.dropna()))
    return ", ".join(map(str, vals)) if vals else ""


def classify_condition_indicator(name: str):
    text = name.lower()
    if re.search(r"(^|_)ept(_|$)|sensitive_proxy", text):
        return "sensitive_benthos", -1, "EPT或敏感类群升高通常表示生态状况改善；反向编码后正值表示退化增强。"
    if "_richness" in text and not re.search(r"griis_alien|griis_invasive", text):
        return None
    if re.search(
        r"tolerant_proxy|chironomidae|oligochaeta|griis_alien|griis_invasive|"
        r"bloom_forming|cyanobacteria|high_risk_algae",
        text,
    ):
        return "pressure_or_tolerant_indicator", 1, "耐受、滋扰藻华、蓝藻、外来或入侵指标升高通常表示生态压力增强。"
    return None


def load_inputs(root: Path):
    wq_path = root / "02_Q1_水质等级参照" / "水质等级_三类环境梯度_13尺度响应.csv"
    family_path = root / "03_Q2_水生生物整体响应" / "环境模块_生态维度_13尺度全局置换检验.csv"
    specific_path = root / "05_Q4_具体生物响应" / "全部具体响应_偏Spearman_分层FDR.csv"
    mapping_path = root / "五类水生生物变量对应关系_Q2分析流程.xlsx"
    wq = pd.read_csv(wq_path)
    wq = wq[wq["response"].eq("wq_class_score_mean")].copy()
    wq["environment_module"] = wq["driver"].map(DRIVER_TO_MODULE)
    family = pd.read_csv(family_path)
    specific = pd.read_csv(specific_path)
    return wq, family, specific, mapping_path, [wq_path, family_path, specific_path, mapping_path]


def build_strength(wq: pd.DataFrame, family: pd.DataFrame):
    keys = ["water_body_type", "environment_module", "range_m"]

    w = wq[keys + ["spearman_rho", "fdr_q", "n_spatial_units", "ci95_low", "ci95_high"]].copy()
    w["system"] = "water_quality"
    w["system_cn"] = "水质"
    w["native_strength"] = w["spearman_rho"].abs()
    w["signed_effect"] = w["spearman_rho"]
    w["strength_definition"] = "|Spearman ρ|（水质等级平均分，越高越差）"
    w["support_value"] = w["fdr_q"]
    w["support_method"] = "13-range FDR q"
    w["supported_05"] = w["fdr_q"] <= 0.05
    w["family_sd"] = np.nan
    w["family_iqr"] = np.nan
    w["family_min"] = np.nan
    w["family_max"] = np.nan
    w["analysis_n"] = w["n_spatial_units"]
    w["analysis_unit"] = "独立空间单元"

    biology = (
        family.groupby(keys, as_index=False)
        .agg(
            native_strength=("rms_partial_rho", "mean"),
            family_sd=("rms_partial_rho", lambda s: float(s.std(ddof=0))),
            family_iqr=("rms_partial_rho", iqr),
            family_min=("rms_partial_rho", "min"),
            family_max=("rms_partial_rho", "max"),
            family_n=("response_family", "nunique"),
            supported_family_n=("range_maxT_p", lambda s: int((s <= 0.05).sum())),
        )
    )
    biology["system"] = "biology_overall"
    biology["system_cn"] = "水生生物总体"
    biology["signed_effect"] = np.nan
    biology["strength_definition"] = "五类生物家族 RMS |partial ρ| 的等权均值"
    biology["support_value"] = np.nan
    biology["support_method"] = "不对总体均值另行检验；保留五类家族的 range-wise maxT"
    biology["supported_05"] = biology["supported_family_n"] > 0
    biology["analysis_n"] = np.where(biology.water_body_type.eq("River"), 86, 44)
    biology["analysis_unit"] = "生物监测site"
    biology["ci95_low"] = np.nan
    biology["ci95_high"] = np.nan

    cols = keys + [
        "system", "system_cn", "native_strength", "signed_effect", "strength_definition",
        "support_value", "support_method", "supported_05", "family_sd", "family_iqr",
        "family_min", "family_max", "analysis_n", "analysis_unit",
    ]
    cells = pd.concat([w[cols], biology[cols]], ignore_index=True)
    cells["environment_module_cn"] = cells.environment_module.map(MODULE_CN)
    cells["strength_percentile_39"] = cells.groupby(
        ["water_body_type", "system"], group_keys=False
    )["native_strength"].apply(percentile_rank)
    cells["strength_rank_39_desc"] = cells.groupby(["water_body_type", "system"])[
        "native_strength"
    ].rank(method="average", ascending=False)
    cells["strength_share_within_body_system"] = cells["native_strength"] / cells.groupby(
        ["water_body_type", "system"]
    )["native_strength"].transform("sum")
    cells["top_quartile_within_body_system"] = cells["strength_percentile_39"] >= 0.75
    cells["cell_key"] = cells.environment_module + "_" + cells.range_m.astype(str)
    cells = cells.sort_values(["system", "water_body_type", "environment_module", "range_m"])

    family_cells = family.copy()
    family_cells["response_family_cn"] = family_cells.response_family.map(FAMILY_CN)
    family_cells["family_strength_percentile_39"] = family_cells.groupby(
        ["water_body_type", "response_family"], group_keys=False
    )["rms_partial_rho"].apply(percentile_rank)
    family_cells["family_strength_rank_39_desc"] = family_cells.groupby(
        ["water_body_type", "response_family"]
    )["rms_partial_rho"].rank(method="average", ascending=False)
    family_cells["family_top_quartile"] = family_cells.family_strength_percentile_39 >= 0.75
    return cells, family_cells


def strength_summaries(cells: pd.DataFrame):
    body_rows = []
    for system, gsys in cells.groupby("system"):
        total = gsys.native_strength.sum()
        for body, g in gsys.groupby("water_body_type"):
            body_rows.append({
                "system": system,
                "system_cn": g.system_cn.iloc[0],
                "water_body_type": body,
                "n_driver_range_cells": len(g),
                "native_strength_mean": g.native_strength.mean(),
                "native_strength_median": g.native_strength.median(),
                "native_strength_iqr": iqr(g.native_strength),
                "native_strength_min": g.native_strength.min(),
                "native_strength_max": g.native_strength.max(),
                "share_of_system_total_across_two_waterbodies": g.native_strength.sum() / total,
                "interpretation": "仅在同一响应体系内描述River/Lake分布；不跨体系比较原始数值。",
            })
    body_summary = pd.DataFrame(body_rows)

    pair_rows = []
    for system, g in cells.groupby("system"):
        wide = g.pivot(index=["environment_module", "range_m"], columns="water_body_type", values="native_strength").dropna()
        pair_rows.append({
            "system": system,
            "system_cn": g.system_cn.iloc[0],
            "matched_cells_n": len(wide),
            "river_greater_fraction": float((wide.River > wide.Lake).mean()),
            "lake_greater_fraction": float((wide.Lake > wide.River).mean()),
            "median_paired_difference_river_minus_lake": float((wide.River - wide.Lake).median()),
            "mean_river": float(wide.River.mean()),
            "mean_lake": float(wide.Lake.mean()),
            "descriptive_mean_ratio_river_over_lake": float(wide.River.mean() / wide.Lake.mean()),
            "inference_boundary": "39个匹配单元来自嵌套range轨迹；结果为同一体系内描述性对照，不报告常规p值。",
        })
    body_pair = pd.DataFrame(pair_rows)

    driver_rows = []
    for (system, body), gb in cells.groupby(["system", "water_body_type"]):
        total = gb.native_strength.sum()
        tmp = []
        for module, g in gb.groupby("environment_module"):
            peak = g.loc[g.native_strength.idxmax()]
            tmp.append({
                "system": system,
                "system_cn": g.system_cn.iloc[0],
                "water_body_type": body,
                "environment_module": module,
                "environment_module_cn": MODULE_CN[module],
                "n_ranges": len(g),
                "native_strength_mean": g.native_strength.mean(),
                "native_strength_median": g.native_strength.median(),
                "native_strength_iqr": iqr(g.native_strength),
                "strength_allocation_share_within_body": g.native_strength.sum() / total,
                "mean_strength_percentile_39": g.strength_percentile_39.mean(),
                "top_quartile_cell_n": int(g.top_quartile_within_body_system.sum()),
                "peak_range_m": int(peak.range_m),
                "peak_native_strength": float(peak.native_strength),
            })
        tmp = pd.DataFrame(tmp)
        tmp["driver_rank_within_body_system"] = tmp.native_strength_mean.rank(method="min", ascending=False).astype(int)
        driver_rows.extend(tmp.to_dict("records"))
    driver_summary = pd.DataFrame(driver_rows)

    indirect = driver_summary.pivot(
        index=["water_body_type", "environment_module", "environment_module_cn"],
        columns="system",
        values=["strength_allocation_share_within_body", "driver_rank_within_body_system", "peak_range_m"],
    )
    indirect.columns = [f"{a}_{b}" for a, b in indirect.columns]
    indirect = indirect.reset_index()
    indirect["allocation_share_difference_biology_minus_wq_pp"] = 100 * (
        indirect["strength_allocation_share_within_body_biology_overall"]
        - indirect["strength_allocation_share_within_body_water_quality"]
    )
    indirect["same_driver_rank"] = (
        indirect["driver_rank_within_body_system_biology_overall"]
        == indirect["driver_rank_within_body_system_water_quality"]
    )
    indirect["interpretation"] = "比较各体系内部强度总量在三类因素间的分配比例；不是原始效应大小差。"

    pattern_rows, trajectory_rows = [], []
    for body in ["River", "Lake"]:
        w = cells[(cells.water_body_type == body) & (cells.system == "water_quality")]
        b = cells[(cells.water_body_type == body) & (cells.system == "biology_overall")]
        pair = w.merge(b, on=["water_body_type", "environment_module", "range_m", "cell_key"], suffixes=("_wq", "_bio"), validate="one_to_one")
        tw = top_set(pair, "strength_percentile_39_wq", "cell_key")
        tb = top_set(pair, "strength_percentile_39_bio", "cell_key")
        pattern_rows.append({
            "water_body_type": body,
            "n_cells": len(pair),
            "spearman_of_internal_strength_ranks_39cells": safe_spearman(pair.strength_percentile_39_wq, pair.strength_percentile_39_bio),
            "top_quartile_overlap_n": len(tw & tb),
            "top_quartile_jaccard": jaccard(tw, tb),
            "interpretation": "间接比较高响应位置和排序结构；不比较两套原始强度。",
        })
        for module, g in pair.groupby("environment_module"):
            g = g.sort_values("range_m")
            kw = top_set(g, "strength_percentile_39_wq", "range_m")
            kb = top_set(g, "strength_percentile_39_bio", "range_m")
            trajectory_rows.append({
                "water_body_type": body,
                "environment_module": module,
                "environment_module_cn": MODULE_CN[module],
                "n_nested_ranges": len(g),
                "range_trajectory_spearman_internal_ranks": safe_spearman(g.strength_percentile_39_wq, g.strength_percentile_39_bio),
                "top_quartile_range_overlap_n": len(kw & kb),
                "top_quartile_range_jaccard": jaccard(kw, kb),
                "wq_peak_range_m": int(g.loc[g.native_strength_wq.idxmax(), "range_m"]),
                "biology_peak_range_m": int(g.loc[g.native_strength_bio.idxmax(), "range_m"]),
                "inference_boundary": "13个range相互嵌套；相关和重叠指标仅作轨迹描述。",
            })
    return body_summary, body_pair, driver_summary, indirect, pd.DataFrame(pattern_rows), pd.DataFrame(trajectory_rows)


def build_direction(wq: pd.DataFrame, family_cells: pd.DataFrame, strength_cells: pd.DataFrame):
    keys = ["water_body_type", "environment_module", "range_m"]
    wq_strength = strength_cells[strength_cells.system.eq("water_quality")][
        keys + ["native_strength", "strength_percentile_39", "top_quartile_within_body_system"]
    ].rename(columns={
        "native_strength": "wq_strength_abs_rho",
        "strength_percentile_39": "wq_strength_percentile_39",
        "top_quartile_within_body_system": "wq_top_quartile",
    })
    w = wq[keys + ["spearman_rho", "fdr_q", "direction_interpretation", "n_spatial_units"]].merge(wq_strength, on=keys, validate="one_to_one")
    d = family_cells.merge(w, on=keys, validate="many_to_one")
    d["response_family_cn"] = d.response_family.map(FAMILY_CN)
    d["wq_direction_sign"] = np.sign(d.spearman_rho).astype(int)
    d["family_direction_sign"] = np.sign(d.median_partial_rho).astype(int)
    d["direction_relation"] = np.where(d.wq_direction_sign.eq(d.family_direction_sign), "concordant", "divergent")
    d["direction_relation_cn"] = d.direction_relation.map({"concordant": "同向", "divergent": "异向"})
    d["family_directional_coherence"] = np.maximum(d.positive_fraction, 1 - d.positive_fraction)
    d["wq_fdr_supported_05"] = d.fdr_q <= 0.05
    d["family_maxT_supported_05"] = d.range_maxT_p <= 0.05
    d["dual_supported_05"] = d.wq_fdr_supported_05 & d.family_maxT_supported_05
    d["both_top_quartile"] = d.wq_top_quartile & d.family_top_quartile
    d["priority_class"] = np.select(
        [
            d.dual_supported_05,
            d.family_maxT_supported_05 & d.both_top_quartile,
            d.wq_fdr_supported_05 & d.both_top_quartile,
            d.family_maxT_supported_05 | d.wq_fdr_supported_05,
        ],
        [
            "A_dual_corrected_support",
            "B_family_support_and_both_high_strength",
            "C_wq_support_and_both_high_strength",
            "D_single_endpoint_corrected_support",
        ],
        default="E_descriptive",
    )
    d["wq_condition_meaning"] = np.where(d.wq_direction_sign > 0, "梯度升高—水质变差", "梯度升高—水质改善")
    d["family_direction_scope"] = "五类家族中位partial ρ仅表示总体关联符号；除预先定向指标外不等同于生态改善或退化。"

    summary_rows = []
    for key, g in d.groupby(["water_body_type", "environment_module", "response_family", "response_family_cn"], sort=False):
        fam_sig = g[g.family_maxT_supported_05]
        wq_sig = g[g.wq_fdr_supported_05]
        dual = g[g.dual_supported_05]
        high = g[g.both_top_quartile]
        summary_rows.append({
            "water_body_type": key[0],
            "environment_module": key[1],
            "environment_module_cn": MODULE_CN[key[1]],
            "response_family": key[2],
            "response_family_cn": key[3],
            "n_ranges": len(g),
            "wq_positive_range_n": int((g.wq_direction_sign > 0).sum()),
            "wq_negative_range_n": int((g.wq_direction_sign < 0).sum()),
            "family_positive_range_n": int((g.family_direction_sign > 0).sum()),
            "family_negative_range_n": int((g.family_direction_sign < 0).sum()),
            "direction_concordance_fraction_all_ranges": float((g.direction_relation == "concordant").mean()),
            "mean_family_directional_coherence": float(g.family_directional_coherence.mean()),
            "wq_fdr_supported_range_n": len(wq_sig),
            "wq_fdr_supported_ranges_m": range_text(wq_sig.range_m),
            "family_maxT_supported_range_n": len(fam_sig),
            "family_maxT_supported_ranges_m": range_text(fam_sig.range_m),
            "dual_supported_range_n": len(dual),
            "dual_supported_ranges_m": range_text(dual.range_m),
            "dual_supported_concordance_fraction": float((dual.direction_relation == "concordant").mean()) if len(dual) else np.nan,
            "both_top_quartile_range_n": len(high),
            "both_top_quartile_concordance_fraction": float((high.direction_relation == "concordant").mean()) if len(high) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    priority = d[d.priority_class.ne("E_descriptive")].copy().sort_values(
        ["priority_class", "water_body_type", "environment_module", "response_family", "range_m"]
    )
    return d, summary, priority


def build_ecological_direction(mapping_path: Path, specific: pd.DataFrame, direction_cells: pd.DataFrame):
    reps = pd.read_excel(mapping_path, sheet_name="Q2最终代表变量").rename(columns={
        "水体类型": "water_body_type",
        "响应家族代码": "response_family",
        "响应家族中文": "response_family_cn",
        "生物类群代码": "taxon_group",
        "生物类群中文": "taxon_group_cn",
        "最终代表变量": "column_name",
        "变量中文名": "variable_name_cn_mapping",
    })
    rows = []
    for r in reps.itertuples(index=False):
        classified = classify_condition_indicator(str(r.column_name))
        if classified is None:
            continue
        category, polarity, rationale = classified
        rows.append({
            "water_body_type": r.water_body_type,
            "response_family": r.response_family,
            "response_family_cn": r.response_family_cn,
            "taxon_group": r.taxon_group,
            "taxon_group_cn": r.taxon_group_cn,
            "column_name": r.column_name,
            "variable_name_cn_mapping": r.variable_name_cn_mapping,
            "condition_indicator_category": category,
            "polarity_to_deterioration": polarity,
            "ecological_rationale": rationale,
        })
    dictionary = pd.DataFrame(rows).drop_duplicates(["water_body_type", "response_family", "column_name"])
    effects = specific.merge(
        dictionary,
        on=["water_body_type", "response_family", "taxon_group", "column_name"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_mapping"),
    )
    context_cols = [
        "water_body_type", "environment_module", "response_family", "range_m",
        "family_strength_percentile_39", "family_top_quartile", "spearman_rho",
        "wq_strength_percentile_39", "wq_top_quartile", "fdr_q",
    ]
    effects = effects.merge(direction_cells[context_cols], on=["water_body_type", "environment_module", "response_family", "range_m"], validate="many_to_one")
    effects["deterioration_oriented_partial_rho"] = effects.partial_spearman_rho * effects.polarity_to_deterioration
    effects["within_response_fdr_supported_05"] = effects.q_within_response_13ranges <= 0.05
    effects["familywide_fdr_supported_05"] = effects.q_within_family_all_responses_ranges <= 0.05
    effects["wq_fdr_supported_05"] = effects.fdr_q <= 0.05
    effects["ecological_direction_relation"] = np.where(
        np.sign(effects.deterioration_oriented_partial_rho) == np.sign(effects.spearman_rho),
        "concordant", "divergent"
    )
    effects["priority_ecological_record"] = effects.within_response_fdr_supported_05 & (
        effects.family_strength_percentile_39 >= 0.5
    )
    summary_rows = []
    for key, g in effects.groupby(["water_body_type", "environment_module"], sort=False):
        supported = g[g.within_response_fdr_supported_05]
        priority = g[g.priority_ecological_record]
        summary_rows.append({
            "water_body_type": key[0],
            "environment_module": key[1],
            "environment_module_cn": MODULE_CN[key[1]],
            "interpretable_indicator_range_records": len(g),
            "unique_interpretable_indicators": g.column_name.nunique(),
            "within_response_fdr_supported_records": len(supported),
            "supported_ecological_direction_concordance_fraction": float((supported.ecological_direction_relation == "concordant").mean()) if len(supported) else np.nan,
            "priority_strength_context_records": len(priority),
            "priority_direction_concordance_fraction": float((priority.ecological_direction_relation == "concordant").mean()) if len(priority) else np.nan,
            "familywide_fdr_supported_records": int(g.familywide_fdr_supported_05.sum()),
        })
    return dictionary, effects, pd.DataFrame(summary_rows)


def write_text_outputs(out: Path, summaries: dict, qa: dict):
    body = summaries["body"].set_index(["system", "water_body_type"])
    driver = summaries["driver"]
    pattern = summaries["pattern"].set_index("water_body_type")
    trajectory = summaries["trajectory"]
    direction = summaries["direction"]
    eco = summaries["eco"].set_index(["water_body_type", "environment_module"])

    def pct(x): return f"{100*x:.1f}%"
    lines = [
        "# Q3 统计学与生态学解读",
        "",
        "## 分析边界",
        "水质强度为水质等级平均分（越高越差）与环境梯度的 |Spearman ρ|；水生生物总体强度为五类家族 RMS |partial ρ| 的等权均值。两者不共享同一估计对象、模型或分析单位，因此不比较原始数值大小。分析先在各体系内部描述 River/Lake、三类因素和 13 个 range 的强度分布，再用内部百分位秩、分配比例和高响应位置进行间接对照。",
        "",
        "## 响应强度分布",
    ]
    for system in ["water_quality", "biology_overall"]:
        r = body.loc[(system, "River")]
        l = body.loc[(system, "Lake")]
        name = r.system_cn
        lines.append(
            f"- {name}体系内，River 的平均原生强度为 {r.native_strength_mean:.3f}，Lake 为 {l.native_strength_mean:.3f}；"
            f"River 占该体系两类水体累计强度的 {pct(r.share_of_system_total_across_two_waterbodies)}，Lake 占 {pct(l.share_of_system_total_across_two_waterbodies)}。这些数值只用于{name}体系内部的描述。"
        )
    for body_name in ["River", "Lake"]:
        g = driver[driver.water_body_type.eq(body_name)]
        w = g[g.system.eq("water_quality")].sort_values("driver_rank_within_body_system")
        b = g[g.system.eq("biology_overall")].sort_values("driver_rank_within_body_system")
        wtxt = " > ".join(f"{x.environment_module_cn}({pct(x.strength_allocation_share_within_body)})" for x in w.itertuples())
        btxt = " > ".join(f"{x.environment_module_cn}({pct(x.strength_allocation_share_within_body)})" for x in b.itertuples())
        lines.append(f"- {body_name}：水质强度分配排序为 {wtxt}；生物总体为 {btxt}。")
    lines += [
        f"- 39 个因素×range 单元的内部秩结构在 River 中 ρ = {pattern.loc['River'].spearman_of_internal_strength_ranks_39cells:.3f}，高响应四分位重叠 {int(pattern.loc['River'].top_quartile_overlap_n)} 个；在 Lake 中 ρ = {pattern.loc['Lake'].spearman_of_internal_strength_ranks_39cells:.3f}，重叠 {int(pattern.loc['Lake'].top_quartile_overlap_n)} 个。该 ρ 描述两套体系把高响应放在相同位置的程度，不是原始效应量相关的显著性检验。",
        "",
        "## 响应方向与强度背景",
    ]
    for body_name, module in [("Lake", "terrain"), ("Lake", "human_activity"), ("River", "terrain"), ("River", "urbanization")]:
        g = direction[(direction.water_body_type == body_name) & (direction.environment_module == module)]
        details = "; ".join(
            f"{x.response_family_cn}: {x.family_positive_range_n}正/{x.family_negative_range_n}负, maxT支持{x.family_maxT_supported_range_n}个range"
            for x in g.itertuples()
        )
        lines.append(f"- {body_name}—{MODULE_CN[module]}：{details}。")
    lines += [
        "- 水质方向可直接解释为水质状况：ρ>0 表示梯度升高伴随水质等级分数升高，即水质变差；ρ<0 表示水质改善。五类生物家族的中位 partial ρ 只表示该家族总体关联方向，不能机械地解释成改善或退化。",
        "- 生态学解释仅对预先具有明确极性的敏感、耐受、蓝藻/藻华和外来入侵指标进行。",
        "",
        "## Q3回答",
        "水质与生物响应既有空间位置上的趋同，也有水体、因素和range层面的重新分配。二者原始强度不可直接比较，但内部强度分布显示它们对环境梯度的敏感性重点并不相同；方向分析进一步表明，统计同向不必然等同于生态同义，显著且具有明确生态极性的响应才构成主要生态结论。",
    ]
    (out / "Q3_统计学与生态学解读及Q3回答.md").write_text("\n".join(lines), encoding="utf-8")

    method = """# Q3 分析方法说明

## 响应强度

水质和水生生物分别保留各自原始统计体系。水质单元强度定义为水质等级平均分与环境梯度之间 Spearman ρ 的绝对值；水生生物首先在每个水体、因素和 range 内计算五类生物家族 RMS |partial ρ| 的等权算术均值。等权均值使五类家族各贡献 20%，避免变量数量较多的家族主导总体结果；家族间标准差、四分位距和范围同时保留用于显示异质性。

分别在每种水体和响应体系内，对三类因素×13个range的39个强度值进行0–1百分位排序。跨体系只比较内部排序、强度分配比例、高响应单元重叠和range轨迹，不计算水质与生物原始效应的差值、比值或相等性检验。River/Lake及因素间的原生强度均值、中位数和累计比例仅在同一响应体系内解释。

## 响应方向

水质方向使用 Spearman ρ 的符号；因水质等级平均分越高表示水质越差，ρ>0表示梯度升高伴随水质变差，ρ<0表示水质改善。五类生物家族方向使用各家族 median partial ρ 的符号，并同时保留 positive fraction 表征家族内部方向一致程度。每个方向单元同时附带该端点在自身39单元中的强度百分位，以及水质13-range FDR和生物range-wise maxT支持状态。重点结果优先级依次为：两端点均经校正支持；单端点经校正支持且两端点均处于高强度四分位；其他单端点校正支持；描述性结果。

家族方向不具有统一的生态好坏含义。生态状态方向仅对预先定义的EPT/敏感类群、耐受类群、蓝藻/藻华及外来入侵指标进行定向编码。一般丰度、生物量、丰富度、PCoA和β多样性不被强行赋予改善—退化极性。

## 推断边界

13个range是累积且相互嵌套的空间范围，不能视为13个独立重复。因此，强度分布、轨迹相关、高响应重叠和峰值range均作为描述性多尺度结构证据，不附加常规独立样本p值。水质与生物使用不同样本层级和校正框架，结果不支持跨体系原始效应量大小比较或因果解释。
"""
    (out / "Q3_分析方法说明.md").write_text(method, encoding="utf-8")
    (out / "Q3_统计学与科学性QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    wq, family, specific, mapping_path, inputs = load_inputs(args.root)

    if len(wq) != 78 or len(family) != 390:
        raise ValueError(f"unexpected rows: wq={len(wq)}, family={len(family)}")
    if set(family.response_family.unique()) != set(FAMILIES):
        raise ValueError("five-family set mismatch")
    if family.groupby(["water_body_type", "environment_module", "range_m"]).response_family.nunique().ne(5).any():
        raise ValueError("not every cell has five families")

    cells, family_cells = build_strength(wq, family)
    body, body_pair, driver, indirect, pattern, trajectory = strength_summaries(cells)
    direction_cells, direction_summary, priority = build_direction(wq, family_cells, cells)
    eco_dict, eco_effects, eco_summary = build_ecological_direction(mapping_path, specific, direction_cells)

    outputs = {
        "Q3_强度分布_水质与生物总体逐range.csv": cells,
        "Q3_强度分布_五类生物逐range.csv": family_cells,
        "Q3_强度分布_水体内部汇总.csv": body,
        "Q3_强度分布_River_Lake配对描述.csv": body_pair,
        "Q3_强度分布_因素内部汇总.csv": driver,
        "Q3_强度分布_水质生物间接对照.csv": indirect,
        "Q3_强度分布_39单元结构对照.csv": pattern,
        "Q3_强度分布_range轨迹对照.csv": trajectory,
        "Q3_方向强度联合_五类逐range.csv": direction_cells,
        "Q3_方向强度联合_梯度家族汇总.csv": direction_summary,
        "Q3_方向强度联合_重点校正支持结果.csv": priority,
        "Q3_生态方向_指标字典.csv": eco_dict,
        "Q3_生态方向_指标效应明细.csv": eco_effects,
        "Q3_生态方向_梯度汇总.csv": eco_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")

    qa = {
        "scope": "Q3 only",
        "question": "To what extent do water-quality and biological response patterns converge or diverge across environmental gradients and ranges?",
        "estimands": {
            "water_quality": "|Spearman rho| of ordinal water-quality score (higher=worse)",
            "biology_family": "RMS |spatially adjusted partial Spearman rho|",
            "biology_overall": "equal-weight arithmetic mean of five family RMS values within each body × driver × range cell",
        },
        "cross_system_rule": "No raw WQ-versus-biology subtraction, ratio, equality test, or common-effect-scale claim.",
        "indirect_comparison": "Compare within-system distributions, 39-cell percentile ranks, allocation shares, top-response overlap, and range trajectories.",
        "direction_rule": "Direction uses raw signs; strength percentile and corrected support are shown as context, not collapsed into one score.",
        "multiplicity": {
            "water_quality": "13-range FDR q retained from Q1",
            "biology_family": "range-wise maxT p retained from Q2",
            "specific_indicator": "within-response 13-range FDR; family-wide FDR as strict sensitivity",
        },
        "nested_range_boundary": "13 cumulative ranges are dependent; all range-distribution comparisons are descriptive and carry no conventional independent-sample p value.",
        "checks": {
            "wq_rows_78": bool(len(wq) == 78),
            "family_rows_390": bool(len(family) == 390),
            "overall_strength_rows_156": bool(len(cells) == 156),
            "family_strength_rows_390": bool(len(family_cells) == 390),
            "direction_rows_390": bool(len(direction_cells) == 390),
            "five_families_each_cell": bool(family.groupby(["water_body_type", "environment_module", "range_m"]).response_family.nunique().eq(5).all()),
            "biology_overall_is_equal_mean_raw_rms": bool(
                np.allclose(
                    cells[cells.system.eq("biology_overall")].sort_values(["water_body_type", "environment_module", "range_m"]).native_strength,
                    family.groupby(["water_body_type", "environment_module", "range_m"]).rms_partial_rho.mean().sort_index().to_numpy(),
                )
            ),
            "all_internal_percentiles_0_1": bool(cells.strength_percentile_39.between(0, 1).all() and family_cells.family_strength_percentile_39.between(0, 1).all()),
            "39_cells_per_body_system": bool(cells.groupby(["water_body_type", "system"]).size().eq(39).all()),
            "13_ranges_per_profile": bool(cells.groupby(["water_body_type", "system", "environment_module"]).range_m.nunique().eq(13).all()),
            "no_cross_system_raw_gap_column": bool(not any("raw" in c.lower() and "difference" in c.lower() for c in indirect.columns)),
        },
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
    }
    write_text_outputs(out, {
        "body": body, "driver": driver, "pattern": pattern, "trajectory": trajectory,
        "direction": direction_summary, "eco": eco_summary,
    }, qa)
    print(json.dumps({"output": str(out), "files": len(outputs) + 3, "checks": qa["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
