from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
DOMAIN_BOUNDS = {
    "滨水区尺度（200–500 m）": (200, 500),
    "中尺度（1000–2500 m）": (1000, 2500),
    "景观尺度（3000–6000 m）": (3000, 6000),
}
MODULE_MAP = {"地形地貌": "terrain", "城市化建设": "urbanization", "人类活动压力": "human_activity", "人类直接活动": "human_activity"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def domain_of_range(r: int) -> str:
    for label, (lo, hi) in DOMAIN_BOUNDS.items():
        if lo <= int(r) <= hi:
            return label
    return "未定义"


def trajectory_summary(g: pd.DataFrame, value_col: str, signed_col: str | None = None) -> dict:
    g = g.sort_values("range_m")
    values = pd.to_numeric(g[value_col], errors="coerce").to_numpy(float)
    signed = pd.to_numeric(g[signed_col or value_col], errors="coerce").to_numpy(float)
    rr = g.range_m.to_numpy(int)
    active = np.isfinite(values) & (np.abs(values) >= 0.15)
    strengths = {}
    for label, (lo, hi) in DOMAIN_BOUNDS.items():
        m = (rr >= lo) & (rr <= hi)
        strengths[label] = float(np.nanmean(np.abs(values[m]))) if m.any() else np.nan
    dominant = max(strengths, key=lambda k: -np.inf if not np.isfinite(strengths[k]) else strengths[k])
    peak_i = int(np.nanargmax(np.abs(values))) if np.isfinite(values).any() else 0
    has_pos, has_neg = np.any(signed >= 0.15), np.any(signed <= -0.15)
    if not active.any():
        shape = "弱响应/近零轨迹"
    elif has_pos and has_neg:
        shape = "方向反转"
    elif active.sum() >= 9:
        shape = "跨尺度连续"
    elif 0 < peak_i < len(values) - 1:
        shape = "中间峰值/转折"
    else:
        shape = "局部单向"
    active_ranges = rr[active]
    return {
        "dominant_scale_domain": dominant,
        "trajectory_shape": shape,
        "peak_range_m": int(rr[peak_i]),
        "peak_value": float(values[peak_i]),
        "active_scale_min_m": int(active_ranges.min()) if active_ranges.size else np.nan,
        "active_scale_max_m": int(active_ranges.max()) if active_ranges.size else np.nan,
        **{f"{k.split('（')[0]}_mean_abs": v for k, v in strengths.items()},
    }


def main() -> None:
    out = Path(__file__).resolve().parents[1]
    processed = out.parents[1]
    ml_root = out.parent / "ML_Input_原始量_13尺度_2026-07-04"
    bio_input = ml_root / "ML_Input_Raw_13Ranges_Sample_Site.xlsx"
    wq_root = processed / "WQ_13尺度环境特征与水质响应分析_2026-07-20"
    wq_input = wq_root / "WQ_PublicWaterQuality_103Stations_13Ranges_INPUT_2026-07-20.xlsx"

    q1 = out / "02_Q1_水质等级参照"
    q2 = out / "03_Q2_水生生物整体响应"
    q3 = out / "04_Q3_水质与生物对照"
    q4 = out / "05_Q4_具体生物响应"
    q6 = out / "06_群落组成与Beta过程"
    audit = out / "01_数据审计"
    source = out / "09_SourceData"

    # Q1: carry the verified independent-geometry water-quality analysis into this frozen framework.
    wq_files = {
        "环境模块梯度与水质响应_13尺度.csv": "水质等级_三类环境梯度_13尺度响应.csv",
        "环境模块梯度与水质_候选最强尺度.csv": "水质等级_各梯度峰值尺度.csv",
        "水质类别上界赋值敏感性_13尺度.csv": "水质等级赋值敏感性_13尺度.csv",
        "水质响应空间自相关_MoranI.csv": "水质等级_空间自相关诊断.csv",
        "三类环境梯度跨尺度相关性.csv": "三类环境梯度跨尺度相关性.csv",
    }
    for src_name, dst_name in wq_files.items():
        shutil.copy2(wq_root / "02_统计结果" / src_name, q1 / dst_name)

    wq_all = pd.read_csv(q1 / "水质等级_三类环境梯度_13尺度响应.csv")
    wq = wq_all[wq_all.response.eq("wq_class_score_mean")].copy()
    wq.to_csv(q1 / "水质等级_三类环境梯度_13尺度响应.csv", index=False, encoding="utf-8-sig")
    wq["environment_module"] = wq.driver.map(MODULE_MAP)
    wq["scale_domain"] = wq.range_m.map(domain_of_range)
    wq_traj = []
    for keys, g in wq.groupby(["water_body_type", "driver", "environment_module"], dropna=False):
        wq_traj.append({**dict(zip(["water_body_type", "driver", "environment_module"], keys)),
                        **trajectory_summary(g, "spearman_rho")})
    pd.DataFrame(wq_traj).to_csv(q1 / "水质等级_响应尺度类型汇总.csv", index=False, encoding="utf-8-sig")
    grade_peaks = (wq.assign(abs_rho=wq.spearman_rho.abs())
                   .sort_values(["water_body_type", "driver", "abs_rho"], ascending=[True, True, False])
                   .groupby(["water_body_type", "driver"], as_index=False).head(1)
                   .drop(columns="abs_rho"))
    grade_peaks.to_csv(q1 / "水质等级_各梯度峰值尺度.csv", index=False, encoding="utf-8-sig")

    # Q2: family-level scale trajectories.
    family = pd.read_csv(q2 / "环境模块_生态维度_13尺度全局置换检验.csv")
    family["scale_domain"] = family.range_m.map(domain_of_range)
    fam_traj = []
    for keys, g in family.groupby(["water_body_type", "environment_module", "environment_module_cn", "response_family", "response_family_cn"]):
        fam_traj.append({**dict(zip(["water_body_type", "environment_module", "environment_module_cn", "response_family", "response_family_cn"], keys)),
                         **trajectory_summary(g, "rms_partial_rho", "median_partial_rho"),
                         "maxT_supported_ranges": ";".join(map(str, g.loc[g.range_maxT_p <= 0.05, "range_m"].astype(int))),
                         "n_response_variables": int(g.n_response_variables.max())})
    fam_traj = pd.DataFrame(fam_traj)
    fam_traj.to_csv(q2 / "五类生物响应_尺度轨迹汇总.csv", index=False, encoding="utf-8-sig")

    # Q3: descriptive, non-paired scale fingerprint comparison.
    bio_scale = family.groupby(["water_body_type", "environment_module", "range_m"]).agg(
        bio_family_mean_rms=("rms_partial_rho", "mean"),
        bio_family_median_signed_rho=("median_partial_rho", "median"),
        bio_supported_family_n=("range_maxT_p", lambda x: int(np.sum(np.asarray(x) <= 0.05))),
        bio_family_n=("response_family", "nunique"),
    ).reset_index()
    compare = wq.merge(bio_scale, on=["water_body_type", "environment_module", "range_m"], how="left")
    compare["scale_domain"] = compare.range_m.map(domain_of_range)
    compare["abs_effect_difference_wq_minus_bio"] = compare.spearman_rho.abs() - compare.bio_family_mean_rms
    compare["direction_concordance"] = np.where(
        np.sign(compare.spearman_rho) == np.sign(compare.bio_family_median_signed_rho), "同向", "异向"
    )
    compare.to_csv(q3 / "水质等级与水生生物_同尺度指纹对照.csv", index=False, encoding="utf-8-sig")
    comp_domain = compare.groupby(["water_body_type", "driver", "environment_module", "scale_domain"]).agg(
        water_quality_mean_rho=("spearman_rho", "mean"),
        water_quality_mean_abs_rho=("spearman_rho", lambda x: np.mean(np.abs(x))),
        biology_mean_rms=("bio_family_mean_rms", "mean"),
        biology_median_signed_rho=("bio_family_median_signed_rho", "median"),
        biology_supported_family_mean_n=("bio_supported_family_n", "mean"),
    ).reset_index()
    comp_domain.to_csv(q3 / "水质等级与水生生物_三尺度域概况.csv", index=False, encoding="utf-8-sig")

    # Q4 and community compact summaries.
    candidates = pd.read_csv(q4 / "具体响应尺度类型与候选清单.csv")
    candidates["is_predefined_candidate"] = candidates.screen_rank <= 3
    top = (candidates[candidates.is_predefined_candidate]
           .sort_values(["water_body_type", "environment_module", "response_family", "screen_rank", "max_abs_rho"],
                        ascending=[True, True, True, True, False])
           .groupby(["water_body_type", "environment_module", "response_family"], as_index=False).head(3))
    top.to_csv(q4 / "各水体_环境模块_生物家族代表响应.csv", index=False, encoding="utf-8-sig")

    community = pd.read_csv(q6 / "原始分类单元矩阵_Hellinger偏RDA与群落离散度.csv")
    community["scale_domain"] = community.range_m.map(domain_of_range)
    community["composition_supported_q05"] = community.q_13ranges_within_group_module_rda <= 0.05
    community["dispersion_supported_q05"] = community.q_13ranges_within_group_module_dispersion <= 0.05
    best_comm = (community.sort_values(["water_body_type", "taxon_group", "environment_module", "permutation_p", "partial_rda_r2"],
                                      ascending=[True, True, True, True, False])
                 .groupby(["water_body_type", "taxon_group", "environment_module"], as_index=False).head(1))
    best_comm.to_csv(q6 / "各类群_环境模块_群落组成峰值尺度.csv", index=False, encoding="utf-8-sig")

    # Audit and source data.
    site = pd.read_excel(bio_input, sheet_name="Site_Special")
    sample = pd.read_excel(bio_input, sheet_name="Sample_Special")
    wq_overview = pd.read_csv(wq_root / "01_INPUT与过程数据" / "WQ_INPUT_数据概览.csv")
    audit_rows = [
        {"dataset": "Aquatic biology Site_Special", "analysis_role": "primary independent-site analysis",
         "n_rows": len(site), "n_columns": site.shape[1], "notes": str(site.water_body_type.value_counts().to_dict())},
        {"dataset": "Aquatic biology Sample_Special", "analysis_role": "temporal/hydrological sensitivity",
         "n_rows": len(sample), "n_columns": sample.shape[1], "notes": str(sample.water_body_type.value_counts().to_dict())},
        {"dataset": "Water-quality grade", "analysis_role": "regional comparative reference",
         "n_rows": int(wq.n_spatial_units.max()), "n_columns": len(wq.columns),
         "notes": "Independent geometry counts differ by water type; see Q1 tables."},
    ]
    pd.DataFrame(audit_rows).to_csv(audit / "核心数据规模与分析角色.csv", index=False, encoding="utf-8-sig")
    wq_overview.to_csv(audit / "水质输入数据概览_原始记录.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"input_file": bio_input.name, "sha256": sha256(bio_input), "bytes": bio_input.stat().st_size},
        {"input_file": wq_input.name, "sha256": sha256(wq_input), "bytes": wq_input.stat().st_size},
    ]).to_csv(audit / "主输入文件_SHA256.csv", index=False, encoding="utf-8-sig")

    # Canonical source tables used by plotting.
    wq.to_csv(source / "Figure02_Q1_water_quality_scale.csv", index=False, encoding="utf-8-sig")
    family.to_csv(source / "Figure03_Q2_biology_family_scale.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(source / "Figure04_Q3_wq_biology_comparison.csv", index=False, encoding="utf-8-sig")
    top.to_csv(source / "Figure05_Q4_representative_responses.csv", index=False, encoding="utf-8-sig")
    community.to_csv(source / "Figure06_community_composition.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "analysis_date": "2026-07-23",
        "bio_site_rows": len(site),
        "bio_sample_rows": len(sample),
        "water_quality_scale_rows": len(wq),
        "biology_family_scale_rows": len(family),
        "specific_response_trajectory_rows": len(candidates),
        "community_test_rows": len(community),
        "interpretation_rule": "scale-response trajectories; no discovery-robustness-extrapolation hierarchy",
    }
    (out / "11_脚本与QA" / "整合运行摘要.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
