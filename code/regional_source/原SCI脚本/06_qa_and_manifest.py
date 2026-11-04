from __future__ import annotations

import hashlib
import json
import platform
import py_compile
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn
import sklearn
import statsmodels
from PIL import Image


RANGES = {200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out = Path(__file__).resolve().parents[1]
    qa_dir = out / "11_脚本与QA"
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # Syntax and reproducibility.
    scripts = sorted(qa_dir.glob("*.py"))
    for p in scripts:
        try:
            py_compile.compile(str(p), doraise=True)
            check(f"python_syntax:{p.name}", True, "compiled")
        except Exception as exc:
            check(f"python_syntax:{p.name}", False, str(exc))

    # Core statistical tables.
    wq = pd.read_csv(out / "02_Q1_水质等级参照" / "水质等级_三类环境梯度_13尺度响应.csv")
    fam = pd.read_csv(out / "03_Q2_水生生物整体响应" / "环境模块_生态维度_13尺度全局置换检验.csv")
    assoc = pd.read_csv(out / "05_Q4_具体生物响应" / "全部具体响应_偏Spearman_分层FDR.csv")
    cand = pd.read_csv(out / "05_Q4_具体生物响应" / "具体响应尺度类型与候选清单.csv")
    comm = pd.read_csv(out / "06_群落组成与Beta过程" / "原始分类单元矩阵_Hellinger偏RDA与群落离散度.csv")

    check("Q1_only_water_quality_grade", set(wq.response) == {"wq_class_score_mean"}, str(wq.response.value_counts().to_dict()))
    check("Q1_expected_rows", len(wq) == 78, f"rows={len(wq)}")
    check("Q1_all_13_ranges", set(wq.range_m.astype(int)) == RANGES, str(sorted(wq.range_m.unique())))
    check("Q2_expected_rows", len(fam) == 390, f"rows={len(fam)}")
    check("Q4_association_rows", len(assoc) == 16575, f"rows={len(assoc)}")
    check("Q4_scale_summary_rows", len(cand) == 1275, f"rows={len(cand)}")
    check("community_rows", len(comm) == 429, f"rows={len(comm)}")
    check("correlations_in_bounds", bool((assoc.partial_spearman_rho.abs() <= 1 + 1e-10).all()), "partial rho")
    check("p_q_in_bounds", bool(
        ((fam.permutation_p >= 0) & (fam.permutation_p <= 1) &
         (fam.range_maxT_p >= 0) & (fam.range_maxT_p <= 1)).all()), "family p/maxT")
    check("scale_language_present", {"dominant_scale_domain", "scale_response_type", "trajectory_shape"}.issubset(cand.columns),
          ",".join(cand.columns))

    # Figure files and basic visual integrity.
    figure_names = [
        "Figure01_研究设计与数据覆盖",
        "Figure02_Q1_水质等级尺度响应参照",
        "Figure03_Q2_五类生物响应尺度谱",
        "Figure04_Q3_水质与生物尺度指纹对照",
        "Figure05_Q4_具体生物响应range",
        "Figure06_群落组成与Beta过程",
        "Figure07_River_Lake时间空间敏感性",
    ]
    fig_rows = []
    for name in figure_names:
        for folder, ext in [("PNG", "png"), ("TIFF", "tiff"), ("PDF", "pdf"), ("SVG", "svg")]:
            p = out / "08_论文级图件" / folder / f"{name}.{ext}"
            exists = p.exists() and p.stat().st_size > 10_000
            width = height = np.nan
            if exists and ext in {"png", "tiff"}:
                with Image.open(p) as img:
                    width, height = img.size
                    exists = exists and width >= 1800 and height >= 1200
            if exists and ext == "svg":
                text = p.read_text(encoding="utf-8", errors="ignore")
                exists = "<text" in text and "svg" in text[:500].lower()
            fig_rows.append({
                "figure": name, "format": ext.upper(), "passed": exists,
                "bytes": p.stat().st_size if p.exists() else 0, "width_px": width, "height_px": height,
            })
    figqa = pd.DataFrame(fig_rows)
    figqa.to_csv(qa_dir / "图件格式完整性检查.csv", index=False, encoding="utf-8-sig")
    check("all_main_figure_formats_pass", bool(figqa.passed.all()), f"{int(figqa.passed.sum())}/{len(figqa)}")

    # Report integrity.
    report = out / "10_结果解读报告" / "城市水生态四科学问题_尺度响应分析与论文级解读报告.md"
    text = report.read_text(encoding="utf-8")
    check("report_has_all_figures", all(name.split("_")[0] in text for name in figure_names), f"chars={len(text)}")
    check("report_has_four_questions", all(f"科学问题 {i}" in text for i in range(1, 5)), "Q1-Q4 headings")
    check("report_no_mojibake", "�" not in text, "UTF-8")

    # Environment.
    env = {
        "analysis_date": "2026-07-23",
        "platform": platform.platform(),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "seed_primary": 20260723,
    }
    (qa_dir / "软件环境.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    qa = pd.DataFrame(checks)
    qa.to_csv(qa_dir / "最终统计与交付QA.csv", index=False, encoding="utf-8-sig")
    summary = {
        "checks_total": len(qa),
        "checks_passed": int(qa.passed.sum()),
        "checks_failed": int((~qa.passed).sum()),
        "main_figure_artifacts": len(figqa),
        "status": "PASS" if qa.passed.all() else "FAIL",
    }
    # Write QA outputs before hashing so the manifest covers every deliverable except itself.
    summary["files_manifested"] = sum(
        1 for p in out.rglob("*") if p.is_file() and p.name != "文件清单_SHA256.csv"
    )
    (qa_dir / "最终QA摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "文件清单_SHA256.csv":
            files.append({
                "relative_path": str(p.relative_to(out)),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            })
    pd.DataFrame(files).to_csv(qa_dir / "文件清单_SHA256.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not qa.passed.all():
        print(qa.loc[~qa.passed].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
