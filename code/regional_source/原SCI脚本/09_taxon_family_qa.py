from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "13_生物类群×五家族精细分析_2026-07-24"
REPORT = ROOT / "10_结果解读报告" / "城市水生态四科学问题_尺度响应分析与论文级解读报告.md"
RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
FAMILIES = {
    "community_baseline", "alpha_diversity", "beta_diversity",
    "community_composition", "functional_indicator",
}
TAXA = {
    "River": {"benthos", "fish", "macrophyte", "phytoplankton", "periphyton", "zooplankton"},
    "Lake": {"benthos", "fish", "macrophyte", "phytoplankton", "zooplankton"},
}


def check(condition: bool, label: str, results: list[dict]) -> None:
    results.append({"check": label, "status": "PASS" if condition else "FAIL"})
    if not condition:
        raise AssertionError(label)


def main() -> None:
    results: list[dict] = []
    omnibus = pd.read_csv(
        SUB / "02_类群家族整体尺度检验" / "生物类群×五家族×三环境模块_13尺度共同置换检验.csv"
    )
    trajectories = pd.read_csv(
        SUB / "02_类群家族整体尺度检验" / "类群×家族_尺度轨迹与主导尺度汇总.csv"
    )
    associations = pd.read_csv(
        SUB / "03_类群内具体响应" / "类群内全部具体响应_偏Spearman_13尺度.csv"
    )
    representatives = pd.read_csv(
        SUB / "03_类群内具体响应" / "类群×家族×环境模块_前三位代表具体响应.csv"
    )
    counts = pd.read_csv(SUB / "01_类群家族数据审计" / "River_Lake_生物类群×五家族可用变量数.csv")

    check(len(omnibus) == 2145, "整体检验行数为2145", results)
    check(
        omnibus.groupby(["water_body_type", "environment_module", "taxon_group", "response_family"]).size().eq(13).all(),
        "每个水体×模块×类群×家族均包含13个range",
        results,
    )
    check(set(omnibus.range_m) == set(RANGES), "13个range完整且取值正确", results)
    check(set(omnibus.response_family) == FAMILIES, "五个响应家族完整", results)
    for water, taxa in TAXA.items():
        check(
            set(omnibus.loc[omnibus.water_body_type.eq(water), "taxon_group"]) == taxa,
            f"{water}生物类群集合正确",
            results,
        )
    check(len(trajectories) == 165, "尺度轨迹汇总行数为165", results)
    check(len(associations) == 16107, "具体响应13尺度记录为16107条", results)
    check(len(representatives) == 486, "前三位代表具体响应为486条", results)
    check(
        counts.groupby(["water_body_type", "taxon_group"]).response_family.nunique().eq(5).all(),
        "每个纳入类群均覆盖五个家族",
        results,
    )

    for col in ["range_maxT_p", "q_taxon_all_families_ranges", "q_module_all_taxon_family_ranges"]:
        check(omnibus[col].between(0, 1, inclusive="both").all(), f"{col}位于[0,1]", results)
    check(omnibus.rms_partial_rho.between(0, 1, inclusive="both").all(), "整体RMS位于[0,1]", results)
    check(omnibus.median_partial_rho.abs().le(1).all(), "整体signed median rho位于[-1,1]", results)
    for col in [
        "partial_spearman_rho", "ci95_low", "ci95_high",
        "p_value", "q_within_response_13ranges",
        "q_within_family_all_responses_ranges", "taxon_family_range_maxT_p",
        "q_taxon_all_families_ranges", "q_module_all_taxon_family_ranges",
    ]:
        lower, upper = (-1, 1) if "rho" in col or col.startswith("ci95") else (0, 1)
        check(associations[col].between(lower, upper, inclusive="both").all(), f"{col}数值边界正确", results)
    expected_joint = (
        associations.q_within_response_13ranges.le(0.10)
        & associations.taxon_family_range_maxT_p.le(0.05)
    )
    check(
        associations.joint_specific_taxon_family_support.astype(bool).eq(expected_joint).all(),
        "具体响应联合支持标记逻辑正确",
        results,
    )

    main_png = sorted((SUB / "04_论文级图件" / "PNG").glob("Figure*.png"))
    main_formats = {
        ext: len(list((SUB / "04_论文级图件" / ext).glob(f"Figure*.{ext.lower()}")))
        for ext in ["PNG", "TIFF", "PDF", "SVG"]
    }
    check(len(main_png) == 5, "Figure08–12共5张PNG主图", results)
    check(all(n == 5 for n in main_formats.values()), "Figure08–12四种主图格式齐全", results)
    check(len(list((SUB / "04_论文级图件" / "类群分面PNG").glob("*.png"))) == 11, "11张类群PNG分面图齐全", results)
    check(len(list((SUB / "04_论文级图件" / "类群分面PDF").glob("*.pdf"))) == 11, "11张类群PDF分面图齐全", results)
    image_sizes = {}
    for path in main_png:
        with Image.open(path) as im:
            image_sizes[path.name] = {"width": im.width, "height": im.height}
            check(im.width >= 3000 and im.height >= 1800, f"{path.name}像素尺寸满足论文级输出", results)

    source_files = list((SUB / "05_SourceData").glob("Figure*.csv"))
    check(len(source_files) >= 7, "主图SourceData文件齐全", results)
    report_text = REPORT.read_text(encoding="utf-8")
    for number in range(8, 13):
        check(f"Figure {number:02d}" in report_text, f"原报告已嵌入Figure {number:02d}", results)
    check("生物类群 × 五家族的精细尺度响应" in report_text, "原报告已加入精细分析章节", results)
    check("发现—稳健—可外推" in report_text, "报告保留尺度证据而非外推等级语义", results)
    check(all(p.stat().st_size > 0 for p in SUB.rglob("*") if p.is_file()), "增补目录无空文件", results)

    summary = {
        "status": "PASS",
        "checks": len(results),
        "omnibus_rows": len(omnibus),
        "trajectory_rows": len(trajectories),
        "association_rows": len(associations),
        "representative_rows": len(representatives),
        "main_figure_formats": main_formats,
        "main_png_dimensions": image_sizes,
        "taxon_card_png_n": len(list((SUB / "04_论文级图件" / "类群分面PNG").glob("*.png"))),
        "taxon_card_pdf_n": len(list((SUB / "04_论文级图件" / "类群分面PDF").glob("*.pdf"))),
        "results": results,
    }
    (SUB / "06_QA" / "生物类群×五家族_完整QA.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [
        "# 生物类群 × 五家族精细分析 QA",
        "",
        f"**状态：** {summary['status']}",
        f"**检查项：** {summary['checks']}",
        f"**整体检验：** {len(omnibus):,} 行",
        f"**具体响应：** {len(associations):,} 行",
        f"**主图：** Figure 08–12，PNG/TIFF/PDF/SVG 各 {main_formats['PNG']} 张",
        f"**类群分面：** PNG/PDF 各 {summary['taxon_card_png_n']} 张",
        "",
        "## 检查明细",
        "",
        *[f"- {r['status']}｜{r['check']}" for r in results],
        "",
    ]
    (SUB / "06_QA" / "生物类群×五家族_完整QA.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results", "main_png_dimensions"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
