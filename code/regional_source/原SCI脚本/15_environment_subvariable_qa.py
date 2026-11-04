from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULE_COUNTS = {"terrain": 5, "urbanization": 6, "human_activity": 2}
FAMILIES = {
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
}
TAXA = {
    "River": {"benthos", "fish", "macrophyte", "phytoplankton", "periphyton", "zooplankton"},
    "Lake": {"benthos", "fish", "macrophyte", "phytoplankton", "zooplankton"},
}


def record(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out = root / "14_三类环境子变量×生物五家族尺度分解_2026-07-27"
    qa_dir = out / "06_QA"
    qa_dir.mkdir(parents=True, exist_ok=True)

    family = pd.read_csv(
        out / "02_五家族子变量尺度分解" / "五家族×环境子变量×13尺度_共同置换与贡献份额.csv"
    )
    taxon = pd.read_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "生物类群×五家族×环境子变量×13尺度_共同置换与贡献份额.csv"
    )
    summary = pd.read_csv(
        out / "03_类群×五家族子变量尺度分解" / "类群×五家族子变量尺度轨迹汇总.csv"
    )
    env = pd.read_csv(out / "01_环境子变量审计" / "三类环境子变量_130站点_13尺度长表.csv")
    api = pd.read_csv(out / "01_环境子变量审计" / "加权API_重建与存档版本一致性.csv")
    col = pd.read_csv(out / "01_环境子变量审计" / "模块内子变量共线性_13尺度.csv")

    checks: list[dict] = []
    expected_family = 2 * len(FAMILIES) * sum(MODULE_COUNTS.values()) * len(RANGES)
    expected_taxon = sum(len(v) for v in TAXA.values()) * len(FAMILIES) * sum(MODULE_COUNTS.values()) * len(RANGES)
    record(checks, "五家族结果行数", len(family) == expected_family, f"{len(family)} / {expected_family}")
    record(checks, "类群×五家族结果行数", len(taxon) == expected_taxon, f"{len(taxon)} / {expected_taxon}")
    record(checks, "13个range完整", set(family.range_m) == set(RANGES) and set(taxon.range_m) == set(RANGES), str(sorted(set(taxon.range_m))))
    record(checks, "五类响应家族完整", set(family.response_family) == FAMILIES and set(taxon.response_family) == FAMILIES, ", ".join(sorted(set(taxon.response_family))))
    taxa_ok = all(set(taxon.loc[taxon.water_body_type.eq(w), "taxon_group"]) == taxa for w, taxa in TAXA.items())
    record(checks, "River/Lake生物类群完整", taxa_ok, "; ".join(f"{w}:{len(v)}" for w, v in TAXA.items()))

    family_sums = family.groupby(
        ["water_body_type", "response_family", "environment_module", "range_m"]
    )["johnson_relative_contribution_share"].sum()
    taxon_sums = taxon.groupby(
        ["water_body_type", "taxon_group", "response_family", "environment_module", "range_m"]
    )["johnson_relative_contribution_share"].sum()
    record(checks, "五家族Johnson贡献份额归一", np.allclose(family_sums, 1, atol=1e-8), f"最大偏差={float((family_sums - 1).abs().max()):.3g}")
    record(checks, "类群家族Johnson贡献份额归一", np.allclose(taxon_sums, 1, atol=1e-8), f"最大偏差={float((taxon_sums - 1).abs().max()):.3g}")

    numeric_cols = [
        "rms_partial_rho",
        "median_partial_rho",
        "permutation_p",
        "subvariable_range_maxT_p",
        "q_all_subvariables_ranges_within_unit",
        "johnson_relative_contribution_share",
    ]
    no_missing = not family[numeric_cols].isna().any().any() and not taxon[numeric_cols].isna().any().any()
    record(checks, "核心统计量无缺失", no_missing, f"family NA={int(family[numeric_cols].isna().sum().sum())}; taxon NA={int(taxon[numeric_cols].isna().sum().sum())}")
    pcols = ["permutation_p", "subvariable_range_maxT_p", "q_all_subvariables_ranges_within_unit"]
    p_ok = all(family[c].between(0, 1).all() and taxon[c].between(0, 1).all() for c in pcols)
    record(checks, "P/q值位于[0,1]", p_ok, ", ".join(pcols))

    env_expected = 130 * len(RANGES) * sum(MODULE_COUNTS.values())
    record(checks, "130站点×13尺度环境长表完整", len(env) == env_expected, f"{len(env)} / {env_expected}")
    record(checks, "环境子变量无缺失", int(env.value.isna().sum()) == 0, f"NA={int(env.value.isna().sum())}")
    min_api_rho = float(api.spearman_rho_rebuilt_vs_archived_expert_api.min())
    record(checks, "重建API与存档API一致", min_api_rho >= 0.99, f"最小Spearman ρ={min_api_rho:.4f}")

    rank_concordance = summary.importance_method_rank_concordance_spearman.dropna()
    sensitivity = {
        "n": int(len(rank_concordance)),
        "median": float(rank_concordance.median()),
        "q25": float(rank_concordance.quantile(0.25)),
        "q75": float(rank_concordance.quantile(0.75)),
        "fraction_nonnegative": float((rank_concordance >= 0).mean()),
        "fraction_ge_0_5": float((rank_concordance >= 0.5).mean()),
    }
    record(
        checks,
        "Johnson权重与边际RMS²排序敏感性",
        sensitivity["median"] >= 0.5,
        f"Spearman中位数={sensitivity['median']:.3f}, IQR={sensitivity['q25']:.3f}–{sensitivity['q75']:.3f}",
    )

    col_summary = (
        col.groupby(["water_body_type", "environment_module"])
        .agg(
            max_pairwise_rho=("max_abs_spearman_between_subvariables", "max"),
            median_pairwise_rho=("max_abs_spearman_between_subvariables", "median"),
            max_condition_index=("condition_index", "max"),
            median_condition_index=("condition_index", "median"),
        )
        .reset_index()
    )
    col_summary.to_csv(qa_dir / "环境模块共线性摘要.csv", index=False, encoding="utf-8-sig")

    figure_rows = []
    png_dir = out / "04_论文级图件" / "PNG"
    expected_basenames = [f"FigureS{i}" for i in range(13, 20)]
    for path in sorted(png_dir.glob("FigureS*.png")):
        with Image.open(path) as im:
            figure_rows.append(
                {
                    "file": path.name,
                    "width_px": im.width,
                    "height_px": im.height,
                    "mode": im.mode,
                    "size_mb": path.stat().st_size / 1024**2,
                }
            )
    figure_qa = pd.DataFrame(figure_rows)
    figure_qa.to_csv(qa_dir / "图件像素与文件审计.csv", index=False, encoding="utf-8-sig")
    names_ok = all(any(row["file"].startswith(prefix) for row in figure_rows) for prefix in expected_basenames)
    dimensions_ok = len(figure_rows) == 7 and all(row["width_px"] >= 3000 and row["height_px"] >= 1400 for row in figure_rows)
    record(checks, "Figure S13–S19齐全", names_ok and len(figure_rows) == 7, f"PNG={len(figure_rows)}")
    record(checks, "图件满足高分辨率输出", dimensions_ok, "; ".join(f"{r['file']} {r['width_px']}×{r['height_px']}" for r in figure_rows))
    vector_ok = all(
        len(list((out / "04_论文级图件" / ext).glob("FigureS*"))) == 7
        for ext in ["PDF", "SVG", "TIFF"]
    )
    record(checks, "PDF/SVG/TIFF格式齐全", vector_ok, "每种格式均应为7幅")

    all_passed = all(c["passed"] for c in checks)
    report = {
        "all_passed": all_passed,
        "checks": checks,
        "importance_sensitivity": sensitivity,
        "row_counts": {
            "family": int(len(family)),
            "taxon_family": int(len(taxon)),
            "environment_long": int(len(env)),
        },
    }
    (qa_dir / "统计与图件QA.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 三类环境子变量尺度分解：统计与图件 QA",
        "",
        f"总判定：{'通过' if all_passed else '未完全通过'}。",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|---|---:|---|",
    ]
    for c in checks:
        lines.append(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 子变量贡献是相关预测因子之间共享解释变异的描述性分配，不等同于独立效应或因果归因。",
            "- maxT 在每个水体类型×响应单元内，联合控制全部环境子变量与13个嵌套range；BH q作为补充。",
            "- 模块内共线性较高时，以Johnson相对权重为主、边际RMS²排序为敏感性参照，并同时报告方向与尺度轨迹。",
            "- 水质仅保留为整体对照，本扩展分析聚焦水生生物五家族及其类群分解。",
        ]
    )
    (qa_dir / "统计与图件QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(checks).to_csv(qa_dir / "统计与图件QA检查表.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"all_passed": all_passed, "checks": len(checks), "sensitivity": sensitivity}, ensure_ascii=False))


if __name__ == "__main__":
    main()
