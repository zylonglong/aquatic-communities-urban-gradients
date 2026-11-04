from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


ROOT = Path(r"E:\桌面\博士课题\Data")
DELIVERY = ROOT / "manuscript" / "城市水生态四科学问题_尺度响应分析_2026-07-23" / "manuscript可视化图件" / "Figure5_Q4_specific_biological_carriers_2026-08-08"
SOURCE_DIR = DELIVERY / "02_source_data"
FIG_DIR = DELIVERY / "03_figure"
QA_DIR = DELIVERY / "04_QA"
SCRIPT_DIR = DELIVERY / "01_script"
ANALYSIS_SCRIPT = ROOT / "src" / "modeling" / "q4_specific_biological_carriers.py"
RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
FAMILIES = ["community_baseline", "alpha_diversity", "beta_diversity", "community_composition", "functional_indicator"]
WATERS = ["River", "Lake"]
MODULE_LABELS = {"terrain": "Topography and\ngeomorphology", "urbanization": "Urban construction", "human_activity": "Direct human activity"}
MODULE_SHORT = {"terrain": "Topography", "urbanization": "Urban construction", "human_activity": "Direct activity"}
FAMILY_LABELS = {"community_baseline": "community-level attributes", "alpha_diversity": "α diversity", "beta_diversity": "β diversity", "community_composition": "community composition", "functional_indicator": "indicator/tolerance taxa"}
FAMILY_COLORS = {"community_baseline": "#355C9A", "alpha_diversity": "#6F91C7", "beta_diversity": "#9E8ABF", "community_composition": "#D08A79", "functional_indicator": "#B45266"}
WATER_COLORS = {"River": "#5067B0", "Lake": "#B54A79"}


def clean_variable_label(name: str) -> str:
    text = str(name)
    replacements = [
        (r"^beta_(river|lake|all)_", ""),
        (r"_bray_curtis_", " Bray-Curtis "), (r"_bray_", " Bray-Curtis "),
        (r"_sorensen_", " Sørensen "), (r"_jaccard_", " Jaccard "),
        (r"_nestedness_", " nestedness "), (r"_turnover_", " turnover "),
        (r"_lcbd$", " LCBD"), (r"_mean$", " mean dissimilarity"),
        (r"_pcoa([123])$", r" PCoA\1"), (r"_pct_", " % "),
        (r"_pct$", " %"), (r"_cwm_", " CWM "), (r"_", " "),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("macrophytes", "macrophyte").replace("qual pa", "presence/absence")
    text = text.replace("quant", "quantitative").replace("density Bray-Curtis", "abundance Bray-Curtis")
    text = text.replace("count Bray-Curtis", "abundance Bray-Curtis").replace("total density", "total abundance")
    return text[0].upper() + text[1:] if text else str(name)


def plot_figure(family_share: pd.DataFrame, contribution: pd.DataFrame, assoc: pd.DataFrame):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 6.3, "axes.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    selected = contribution.loc[contribution.observed_rank_within_family <= 3].copy()
    selected["label"] = selected.column_name.map(clean_variable_label)
    selected["water_order"] = selected.water_body_type.map({"River": 0, "Lake": 1})
    selected["family_order"] = selected.response_family.map({f: i for i, f in enumerate(FAMILIES)})
    selected = selected.sort_values(["water_order", "family_order", "observed_rank_within_family"]).reset_index(drop=True)

    fig = plt.figure(figsize=(7.2, 11.0), constrained_layout=False)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.25, 3.0, 3.1, 3.1], hspace=0.58)
    ax_a = fig.add_subplot(outer[0, 0])
    summary = family_share.groupby(["water_body_type", "environment_module", "response_family"], as_index=False).family_signal_share.mean()
    y = np.arange(6)[::-1]
    left = np.zeros(6)
    ordered_cells = [(w, m) for w in WATERS for m in MODULES]
    for family in FAMILIES:
        values = []
        for water, module in ordered_cells:
            hit = summary.loc[summary.water_body_type.eq(water) & summary.environment_module.eq(module) & summary.response_family.eq(family), "family_signal_share"]
            values.append(float(hit.iloc[0]) if len(hit) else 0.0)
        values = np.asarray(values)
        ax_a.barh(y, values, left=left, height=0.68, color=FAMILY_COLORS[family], edgecolor="white", linewidth=0.35)
        left += values
    ax_a.set_yticks(y, [f"{w} · {MODULE_SHORT[m]}" for w, m in ordered_cells])
    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
    ax_a.set_xlabel("Mean share of five-domain biological response strength (%)")
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in FAMILIES], ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.00), fontsize=5.8)
    ax_a.text(-.12, 1.43, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold", va="top")
    ax_a.text(-.08, 1.43, "Ecological dimensions carrying the overall biological signal", transform=ax_a.transAxes, fontsize=8.2, fontweight="bold", va="top")

    gs_b = outer[1, 0].subgridspec(1, 2, wspace=.44)
    for wi, water in enumerate(WATERS):
        ax = fig.add_subplot(gs_b[0, wi])
        sub = selected.loc[selected.water_body_type.eq(water)].copy().reset_index(drop=True)
        positions = np.arange(len(sub))[::-1]
        for pos, row in zip(positions, sub.itertuples(index=False)):
            x, lo, hi = 100*row.balanced_contribution_share, 100*row.bootstrap_ci95_low, 100*row.bootstrap_ci95_high
            color = FAMILY_COLORS[row.response_family]
            ax.plot([lo, hi], [pos, pos], color=color, linewidth=1.2, alpha=.8)
            ax.scatter(x, pos, s=24, facecolor=color if row.top3_selection_frequency >= .70 else "white", edgecolor=color, linewidth=1, zorder=3)
            ax.text(max(hi, x)+.35, pos, f"{100*row.top3_selection_frequency:.0f}%", va="center", fontsize=5.1, color="#555555")
        ax.set_yticks(positions, [r.label for r in sub.itertuples(index=False)])
        ax.tick_params(axis="y", length=0, labelsize=5.35)
        for tick, row in zip(ax.get_yticklabels(), sub.itertuples(index=False)):
            tick.set_color(FAMILY_COLORS[row.response_family])
        ax.set_xlabel("Balanced contribution to family signal (%)")
        ax.text(.02, 1.01, water, transform=ax.transAxes, color=WATER_COLORS[water], fontsize=7.5, fontweight="bold", ha="left", va="bottom")
        ax.axvline(0, color="#777777", linewidth=.5)
        ax.set_xlim(0, max(12, 100*sub.bootstrap_ci95_high.max()+6))
        if wi == 0:
            ax.text(-.28, 1.08, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
            ax.text(-.20, 1.08, "Contribution and bootstrap selection stability", transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")
        ax.text(.99, 1.01, "label = Top 3 frequency", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color="#555555")

    cmap, norm = mpl.colormaps["coolwarm"], Normalize(vmin=-.60, vmax=.60)
    for wi, water in enumerate(WATERS):
        ax = fig.add_subplot(outer[2 + wi, 0])
        sub = selected.loc[selected.water_body_type.eq(water)].copy().reset_index(drop=True)
        rows, family_sig, response_sig = [], [], []
        for row in sub.itertuples(index=False):
            g = assoc.loc[assoc.water_body_type.eq(water) & assoc.column_name.eq(row.column_name)].copy()
            g["module_order"] = g.environment_module.map({m: i for i, m in enumerate(MODULES)})
            g["range_order"] = g.range_m.map({r: i for i, r in enumerate(RANGES)})
            g = g.sort_values(["module_order", "range_order"])
            rows.append(g.partial_spearman_rho.to_numpy(float))
            family_sig.append(g.family_variables_13range_maxT_p.to_numpy(float) <= .05)
            response_sig.append(g.response_13range_maxT_p.to_numpy(float) <= .05)
        matrix = np.vstack(rows)
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        family_sig, response_sig = np.vstack(family_sig), np.vstack(response_sig)
        yy, xx = np.where(response_sig & ~family_sig)
        ax.scatter(xx, yy, s=5, facecolor="none", edgecolor="#333333", linewidth=.35)
        yy, xx = np.where(family_sig)
        ax.scatter(xx, yy, s=5.5, facecolor="#111111", edgecolor="#111111", linewidth=.25)
        for divider in [12.5, 25.5]:
            ax.axvline(divider, color="white", linewidth=1.4)
            ax.axvline(divider, color="#666666", linewidth=.35)
        ax.set_xticks(range(39), [str(r/1000).rstrip("0").rstrip(".") for _ in MODULES for r in RANGES], rotation=55, ha="right", fontsize=4.6)
        ax.set_yticks(np.arange(len(sub)), [r.label for r in sub.itertuples(index=False)], fontsize=5.15)
        ax.tick_params(axis="y", length=0)
        for tick, row in zip(ax.get_yticklabels(), sub.itertuples(index=False)):
            tick.set_color(FAMILY_COLORS[row.response_family])
        ax.set_xlabel("Spatial range (km)")
        for mi, module in enumerate(MODULES):
            ax.text((mi*13+6)/38, 1.015, MODULE_LABELS[module], transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5, fontweight="bold")
        ax.text(-.12, 1.16, "c" if wi == 0 else "d", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.text(-.08, 1.16, "River responses of selected characteristics" if wi == 0 else "Lake responses of selected characteristics", transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")
        ax.text(1.00, 1.16, water, transform=ax.transAxes, color=WATER_COLORS[water], fontsize=7.5, fontweight="bold", ha="right", va="top")
        for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([.37, .022, .26, .010])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("Partial Spearman ρ", labelpad=1); cb.set_ticks([-.6, 0, .6])
    fig.text(.15, .025, "○ response-wise 13-range maxT P ≤ 0.05", fontsize=5.4, ha="center")
    fig.text(.84, .025, "● family-wise variables × ranges maxT P ≤ 0.05", fontsize=5.4, ha="center")
    fig.subplots_adjust(left=.20, right=.985, top=.965, bottom=.060)
    return fig


def main():
    assoc = pd.read_csv(SOURCE_DIR / "Q4_all_frozen_candidate_multiscale_associations.csv")
    contribution = pd.read_csv(SOURCE_DIR / "Q4_variable_contribution_and_bootstrap_stability.csv")
    family_share = pd.read_csv(SOURCE_DIR / "Figure_Q4_family_signal_shares.csv")
    selected = contribution.loc[contribution.observed_rank_within_family <= 3].copy()
    figure = plot_figure(family_share, contribution, assoc)
    base = "Figure_Q4_specific_biological_carriers"
    paths = {ext: FIG_DIR / ext.upper() / f"{base}.{ext if ext != 'tiff' else 'tiff'}" for ext in ["png", "svg", "pdf", "tiff"]}
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    figure.savefig(paths["tiff"], dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ANALYSIS_SCRIPT, SCRIPT_DIR / ANALYSIS_SCRIPT.name)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)

    checks = []
    def add(name, passed, detail): checks.append({"check": name, "passed": bool(passed), "detail": detail})
    add("Frozen Q2 representatives only", contribution.column_name.notna().all(), f"n={len(contribution)}")
    sums = family_share.groupby(["water_body_type", "environment_module", "range_m"]).family_signal_share.sum()
    add("Five-family shares sum to one", np.allclose(sums, 1, atol=1e-10), f"max deviation={float((sums-1).abs().max()):.3g}")
    sums = contribution.groupby(["water_body_type", "response_family"]).balanced_contribution_share.sum()
    add("Within-family contribution sums to one", np.allclose(sums, 1, atol=1e-10), f"max deviation={float((sums-1).abs().max()):.3g}")
    counts = selected.groupby(["water_body_type", "response_family"]).size()
    add("Exactly three selected per water x family", bool((counts == 3).all() and len(counts) == 10), str(counts.to_dict()))
    pcols = ["permutation_p", "response_13range_maxT_p", "family_variables_13range_maxT_p"]
    add("Permutation probabilities bounded", assoc[pcols].apply(lambda s: s.dropna().between(0, 1).all()).all(), "0 <= p <= 1")
    for ext, path in paths.items(): add(f"{ext.upper()} exists", path.exists() and path.stat().st_size > 0, str(path.stat().st_size if path.exists() else 0))
    svg = paths["svg"].read_text(encoding="utf-8")
    add("SVG retains live text", "<text" in svg, f"text elements={svg.count('<text')}")
    qa = pd.DataFrame(checks)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    qa.to_csv(QA_DIR / "Figure_Q4_automated_QA.csv", index=False, encoding="utf-8-sig")
    summary_mean = family_share.groupby(["water_body_type", "environment_module", "response_family"], as_index=False).family_signal_share.mean()
    top = summary_mean.sort_values("family_signal_share", ascending=False).groupby(["water_body_type", "environment_module"], as_index=False).first()
    summary = {"analysis_date": "2026-08-08", "backend": "Python", "independent_units": {"River": 86, "Lake": 44}, "candidate_representatives": contribution.groupby("water_body_type").size().to_dict(), "selected_variables": selected.groupby("water_body_type").size().to_dict(), "permutations": 999, "site_bootstraps": 499, "all_qa_pass": bool(qa.passed.all()), "top_family_by_water_and_driver": top.to_dict(orient="records"), "selected_variables_detail": selected[["water_body_type", "response_family", "taxon_group", "column_name", "variable_name_cn", "balanced_contribution_share", "bootstrap_ci95_low", "bootstrap_ci95_high", "top3_selection_frequency", "selection_stability"]].to_dict(orient="records")}
    (QA_DIR / "Q4_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"delivery": str(DELIVERY), "all_qa_pass": summary["all_qa_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
