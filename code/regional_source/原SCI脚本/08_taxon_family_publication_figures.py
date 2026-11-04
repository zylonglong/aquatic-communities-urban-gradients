from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
XKM = np.array(RANGES) / 1000
XLABELS = ["0.2", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6"]
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_CN = {"terrain": "地形地貌", "urbanization": "城市化建设", "human_activity": "人类活动压力"}
MODULE_SHORT = {"terrain": "地形", "urbanization": "城市化", "human_activity": "人类活动"}
FAMILIES = ["community_baseline", "alpha_diversity", "beta_diversity", "community_composition", "functional_indicator"]
FAMILY_CN = {
    "community_baseline": "群落基础", "alpha_diversity": "α多样性", "beta_diversity": "β多样性",
    "community_composition": "群落组成", "functional_indicator": "功能/指示",
}
TAXA = {
    "River": ["benthos", "fish", "macrophyte", "phytoplankton", "periphyton", "zooplankton"],
    "Lake": ["benthos", "fish", "macrophyte", "phytoplankton", "zooplankton"],
}
TAXON_CN = {
    "benthos": "底栖动物", "fish": "鱼类", "macrophyte": "水生植物",
    "phytoplankton": "浮游植物", "periphyton": "着生藻类", "zooplankton": "浮游动物",
}
TAXON_COLOR = {
    "benthos": "#7A6F9B", "fish": "#2B7A9B", "macrophyte": "#4C9A6A",
    "phytoplankton": "#C79A37", "periphyton": "#B86745", "zooplankton": "#D16D86",
}
WATER_COLOR = {"River": "#2878B5", "Lake": "#2A9D8F"}


def setup_style():
    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.2,
        "axes.linewidth": .75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def save_main(fig, sub: Path, name: str):
    for folder, ext, dpi in [("PNG", "png", 400), ("TIFF", "tiff", 600), ("PDF", "pdf", 300), ("SVG", "svg", 300)]:
        fig.savefig(sub / "04_论文级图件" / folder / f"{name}.{ext}", dpi=dpi,
                    bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)


def panel(ax, label):
    ax.text(-.10, 1.035, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            ha="left", va="bottom")


def scale_background(ax, alpha=.78):
    """Show each measured cumulative range without grouping them into domains."""
    for x in range(len(RANGES)):
        ax.axvline(x, color="#E6E8EB", lw=.38, alpha=.9, zorder=0)


def bubble_counts(ax, d, water, letter):
    taxa = TAXA[water]
    q = d[d.water_body_type.eq(water)].set_index(["taxon_group", "response_family"])
    vmax = d.n_available_response_variables.max()
    cmap = mpl.colormaps["Blues"] if water == "River" else mpl.colormaps["BuGn"]
    norm = Normalize(0, vmax)
    for yi, taxon in enumerate(taxa):
        for xi, fam in enumerate(FAMILIES):
            n = int(q.loc[(taxon, fam), "n_available_response_variables"])
            ax.scatter(xi, yi, s=22 + 8.5 * n, color=cmap(norm(n)), edgecolor="white", lw=.6)
            ax.text(xi, yi, str(n), color="white" if n >= 8 else "#333333",
                    ha="center", va="center", fontsize=6.2, fontweight="bold")
    ax.set_xticks(range(5), [FAMILY_CN[f] for f in FAMILIES], rotation=22, ha="right")
    ax.set_yticks(range(len(taxa)), [TAXON_CN[t] for t in taxa])
    ax.set_xlim(-.55, 4.55); ax.set_ylim(len(taxa) - .45, -.55)
    ax.set_title(f"{water}｜类群 × 家族可用响应变量数", loc="left", fontweight="bold")
    ax.set_xlabel("圆面积与数字均表示变量数")
    panel(ax, letter)


def breadth_panel(ax, d, water, letter):
    taxa = TAXA[water]
    q = d[d.water_body_type.eq(water)].set_index(["taxon_group", "environment_module"])
    vmax = max(.32, d.mean_rms_all_family_ranges.quantile(.98))
    cmap = mpl.colormaps["YlOrRd"]; norm = Normalize(0.08, vmax)
    for yi, taxon in enumerate(taxa):
        for xi, module in enumerate(MODULES):
            r = q.loc[(taxon, module)]
            n = int(r.maxT_supported_cell_n)
            ax.scatter(xi, yi, s=32 + 7.0 * n, color=cmap(norm(r.mean_rms_all_family_ranges)),
                       edgecolor="#222222" if r.broad_FDR_supported_cell_n > 0 else "white",
                       lw=.9 if r.broad_FDR_supported_cell_n > 0 else .5)
            ax.text(xi, yi, str(int(r.supported_family_n)), ha="center", va="center",
                    color="white" if r.mean_rms_all_family_ranges > .17 else "#333333",
                    fontsize=6.2, fontweight="bold")
    ax.set_xticks(range(3), [MODULE_CN[m] for m in MODULES], rotation=15, ha="right")
    ax.set_yticks(range(len(taxa)), [TAXON_CN[t] for t in taxa])
    ax.set_xlim(-.55, 2.55); ax.set_ylim(len(taxa) - .45, -.55)
    ax.set_title(f"{water}｜类群对环境模块的响应广度", loc="left", fontweight="bold")
    ax.set_xlabel("圆面积=maxT支持的家族×range单元数；数字=获得支持的家族数")
    panel(ax, letter)
    return norm, cmap


def figure08(sub):
    counts = pd.read_csv(sub / "01_类群家族数据审计" / "River_Lake_生物类群×五家族可用变量数.csv")
    breadth = pd.read_csv(sub / "02_类群家族整体尺度检验" / "各类群对环境模块的响应广度.csv")
    fig = plt.figure(figsize=(12.3, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.02], hspace=.42, wspace=.34)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    bubble_counts(axes[0], counts, "River", "a")
    bubble_counts(axes[1], counts, "Lake", "b")
    norm, cmap = breadth_panel(axes[2], breadth, "River", "c")
    breadth_panel(axes[3], breadth, "Lake", "d")
    cax = fig.add_axes([.32, .035, .36, .016])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("类群在全部5家族 × 13 range中的平均 RMS partial ρ")
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="#DD8452", mec="white", ms=6, label="圆面积：maxT支持单元数"),
        Line2D([0], [0], marker="o", ls="", mfc="#DD8452", mec="#222222", ms=6, label="黑色外圈：模块内广义FDR存在支持"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97), ncol=2)
    fig.suptitle("Figure 08｜生物类群 × 五家族的数据覆盖与环境响应广度", y=1.005, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.90, bottom=.12)
    counts.to_csv(sub / "05_SourceData" / "Figure08a_b_taxon_family_counts.csv", index=False, encoding="utf-8-sig")
    breadth.to_csv(sub / "05_SourceData" / "Figure08c_d_taxon_response_breadth.csv", index=False, encoding="utf-8-sig")
    save_main(fig, sub, "Figure08_类群家族数据覆盖与响应广度")


def atlas_panel(ax, omnibus, water, module, show_y, letter):
    taxa = TAXA[water]
    rows = [(taxon, family) for taxon in taxa for family in FAMILIES]
    labels = [f"{TAXON_CN[t]}｜{FAMILY_CN[f]}" for t, f in rows]
    q = omnibus[(omnibus.water_body_type == water) & (omnibus.environment_module == module)]
    cmap = mpl.colormaps["RdBu_r"]; norm = TwoSlopeNorm(vmin=-.50, vcenter=0, vmax=.50)
    scale_background(ax)
    key_to_y = {k: i for i, k in enumerate(rows)}
    for _, r in q.iterrows():
        x = RANGES.index(int(r.range_m)); y = key_to_y[(r.taxon_group, r.response_family)]
        ax.scatter(x, y, s=17 + 430 * r.rms_partial_rho ** 2, color=cmap(norm(r.median_partial_rho)),
                   edgecolor="#202020" if r.range_maxT_p <= .05 else "white",
                   lw=1.0 if r.range_maxT_p <= .05 else .35, zorder=3)
        if r.q_module_all_taxon_family_ranges <= .05:
            ax.text(x, y, "•", color="white", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=4)
    ax.set_xticks(range(13), XLABELS, rotation=45, ha="right")
    ax.set_xlabel("空间范围（km）")
    ax.set_xlim(-.5, 12.5); ax.set_ylim(len(rows) - .45, -.55)
    ax.set_title(MODULE_CN[module], loc="left", fontweight="bold", color="#333333")
    if show_y:
        ax.set_yticks(range(len(rows)), labels)
    else:
        ax.tick_params(axis="y", labelleft=False)
    for k in range(1, len(taxa)):
        ax.axhline(k * len(FAMILIES) - .5, color="#8B8B8B", lw=.65)
    panel(ax, letter)


def atlas_figure(sub, water, number):
    omnibus = pd.read_csv(sub / "02_类群家族整体尺度检验" / "生物类群×五家族×三环境模块_13尺度共同置换检验.csv")
    height = 11.3 if water == "River" else 9.7
    fig, axes = plt.subplots(1, 3, figsize=(14.0, height), sharex=True, sharey=True)
    for j, module in enumerate(MODULES):
        atlas_panel(axes[j], omnibus, water, module, j == 0, chr(ord("a") + j))
    taxa = TAXA[water]
    rows = [(taxon, family) for taxon in taxa for family in FAMILIES]
    axes[0].set_yticks(
        range(len(rows)),
        [f"{TAXON_CN[taxon]}｜{FAMILY_CN[family]}" for taxon, family in rows],
    )
    cax = fig.add_axes([.31, .035, .38, .016])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-.50, vcenter=0, vmax=.50), cmap="RdBu_r"),
                      cax=cax, orientation="horizontal")
    cb.set_label("类群—家族内非冗余响应的 signed median partial Spearman ρ")
    handles = [
        plt.scatter([], [], s=17 + 430 * .15 ** 2, color="#999999", edgecolor="white", label="RMS=0.15"),
        plt.scatter([], [], s=17 + 430 * .30 ** 2, color="#999999", edgecolor="white", label="RMS=0.30"),
        plt.scatter([], [], s=42, facecolor="white", edgecolor="#202020", lw=1, label="13-range maxT p≤0.05"),
        plt.scatter([], [], s=42, facecolor="#777777", edgecolor="#202020", lw=1, label="白点：模块内广义FDR q≤0.05"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97), ncol=4)
    fig.suptitle(f"Figure {number}｜{water} 生物类群 × 五家族的三环境模块 13-range 响应图谱",
                 y=1.005, fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=.18, right=.985, top=.91, bottom=.10, wspace=.08)
    omnibus[omnibus.water_body_type.eq(water)].to_csv(
        sub / "05_SourceData" / f"Figure{number}_{water}_taxon_family_atlas.csv",
        index=False, encoding="utf-8-sig"
    )
    save_main(fig, sub, f"Figure{number}_{water}_类群家族13尺度图谱")


def figure11(sub):
    d = pd.read_csv(sub / "02_类群家族整体尺度检验" / "各尺度总体响应的类群贡献份额.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.7), sharex=True, sharey=True)
    for i, water in enumerate(["River", "Lake"]):
        taxa = TAXA[water]
        for j, module in enumerate(MODULES):
            ax = axes[i, j]
            q = d[(d.water_body_type == water) & (d.environment_module == module)]
            pivot = q.pivot(index="taxon_group", columns="range_m", values="taxon_response_energy_share").reindex(index=taxa, columns=RANGES).fillna(0)
            ax.stackplot(XKM, pivot.to_numpy(), colors=[TAXON_COLOR[t] for t in taxa],
                         labels=[TAXON_CN[t] for t in taxa], alpha=.90, linewidth=.25, edgecolor="white")
            ax.axvline(.5, color="white", lw=1.1); ax.axvline(2.5, color="white", lw=1.1)
            ax.set_xlim(.2, 6); ax.set_ylim(0, 1)
            ax.set_title(f"{water} · {MODULE_CN[module]}", loc="left", fontweight="bold")
            if i == 1:
                ax.set_xlabel("空间范围（km）")
            if j == 0:
                ax.set_ylabel("五家族等权的响应能量份额")
            panel(ax, chr(ord("a") + i * 3 + j))
    handles = [Patch(facecolor=TAXON_COLOR[t], label=TAXON_CN[t]) for t in TAXA["River"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=6)
    fig.suptitle("Figure 11｜三类环境因素的整体响应由哪些生物类群承载？", y=1.03, fontsize=11, fontweight="bold")
    fig.text(.5, .012, "注：每个类群先等权汇总五个家族的 RMS²，再换算为同一 water × module × range 内的比例；该比例是描述性分解，不是因果贡献。",
             ha="center", fontsize=7, color="#555555")
    fig.subplots_adjust(top=.89, bottom=.12, left=.08, right=.985, wspace=.14, hspace=.25)
    d.to_csv(sub / "05_SourceData" / "Figure11_taxon_response_energy_share.csv", index=False, encoding="utf-8-sig")
    save_main(fig, sub, "Figure11_环境响应的类群贡献份额")


def short_label(value, n=19):
    s = str(value).replace("·总·", "·").replace("总体·", "")
    return s if len(s) <= n else s[:n - 1] + "…"


def choose_specific(sub, water):
    reps = pd.read_csv(sub / "03_类群内具体响应" / "类群×家族×环境模块_前三位代表具体响应.csv")
    q = reps[reps.water_body_type.eq(water)].copy()
    q = q.sort_values(
        ["taxon_group", "joint_specific_taxon_family_supported_n", "taxon_family_maxT_supported_n",
         "screen_rank", "is_ordination_axis", "max_abs_rho"],
        ascending=[True, False, False, True, True, False]
    )
    # At most two variables per taxon and prefer two different response families.
    selected = []
    for taxon in TAXA[water]:
        z = q[q.taxon_group.eq(taxon)]
        first_by_family = z.groupby("response_family", as_index=False).head(1)
        first_by_family = first_by_family.sort_values(
            ["joint_specific_taxon_family_supported_n", "taxon_family_maxT_supported_n", "screen_rank", "max_abs_rho"],
            ascending=[False, False, True, False]
        )
        selected.append(first_by_family.head(2))
    return pd.concat(selected, ignore_index=True)


def specific_panel(ax, long, selected, water, letter, right_labels=False):
    rows = []
    labels = []
    for _, r in selected.iterrows():
        key = (r.environment_module, r.taxon_group, r.response_family, r.column_name)
        rows.append(key)
        labels.append(f"{MODULE_SHORT[r.environment_module]}｜{TAXON_CN[r.taxon_group]}｜{FAMILY_CN[r.response_family]}｜{short_label(r.variable_name_cn)}")
    rowmap = {k: i for i, k in enumerate(rows)}
    scale_background(ax)
    cmap = mpl.colormaps["RdBu_r"]; norm = TwoSlopeNorm(vmin=-.60, vcenter=0, vmax=.60)
    for _, r in long.iterrows():
        key = (r.environment_module, r.taxon_group, r.response_family, r.column_name)
        if key not in rowmap:
            continue
        x = RANGES.index(int(r.range_m)); y = rowmap[key]
        ax.scatter(x, y, s=35, color=cmap(norm(r.partial_spearman_rho)),
                   edgecolor="#202020" if r.q_within_response_13ranges <= .10 else "white",
                   lw=.9 if r.q_within_response_13ranges <= .10 else .3)
        if bool(r.joint_specific_taxon_family_support):
            ax.text(x, y, "•", color="white", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(13), XLABELS, rotation=45, ha="right"); ax.set_xlabel("空间范围（km）")
    ax.set_xlim(-.5, 12.5); ax.set_ylim(len(rows) - .45, -.55)
    ax.set_yticks(range(len(rows)), labels)
    if right_labels:
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False, pad=4)
    ax.set_title(f"{water}｜每个类群两个不同家族的代表具体响应", loc="left", fontweight="bold")
    for k in range(2, len(rows), 2):
        ax.axhline(k - .5, color="#8B8B8B", lw=.6)
    panel(ax, letter)


def figure12(sub):
    assoc = pd.read_csv(sub / "03_类群内具体响应" / "类群内全部具体响应_偏Spearman_13尺度.csv")
    sr = choose_specific(sub, "River")
    sl = choose_specific(sub, "Lake")
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.0), sharex=True)
    specific_panel(axes[0], assoc[assoc.water_body_type.eq("River")], sr, "River", "a")
    specific_panel(axes[1], assoc[assoc.water_body_type.eq("Lake")], sl, "Lake", "b", right_labels=True)
    cax = fig.add_axes([.31, .045, .38, .017])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-.60, vcenter=0, vmax=.60), cmap="RdBu_r"),
                      cax=cax, orientation="horizontal")
    cb.set_label("具体响应的偏 Spearman ρ")
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="#999999", mec="#202020", ms=6, label="黑色外圈：响应内13 range q≤0.10"),
        Line2D([0], [0], marker="o", ls="", mfc="#777777", mec="#202020", ms=6, label="白点：同时满足具体响应q≤0.10与类群—家族maxT p≤0.05"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .975), ncol=2)
    fig.suptitle("Figure 12｜类群—家族整体尺度谱对应的具体生物响应", y=1.015, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.90, bottom=.14, left=.16, right=.84, wspace=.18)
    selected = pd.concat([sr.assign(panel="River"), sl.assign(panel="Lake")], ignore_index=True)
    selected.to_csv(sub / "05_SourceData" / "Figure12_selected_specific_responses.csv", index=False, encoding="utf-8-sig")
    assoc.merge(
        selected[["water_body_type", "environment_module", "taxon_group", "response_family", "column_name"]].drop_duplicates(),
        on=["water_body_type", "environment_module", "taxon_group", "response_family", "column_name"], how="inner"
    ).to_csv(sub / "05_SourceData" / "Figure12_selected_response_scale_data.csv", index=False, encoding="utf-8-sig")
    save_main(fig, sub, "Figure12_类群内代表具体响应")


def taxon_card(sub, omnibus, water, taxon):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.3), sharex=True, sharey=True)
    q = omnibus[(omnibus.water_body_type == water) & (omnibus.taxon_group == taxon)]
    cmap = mpl.colormaps["RdBu_r"]; norm = TwoSlopeNorm(vmin=-.50, vcenter=0, vmax=.50)
    for j, module in enumerate(MODULES):
        ax = axes[j]; scale_background(ax)
        z = q[q.environment_module.eq(module)]
        for _, r in z.iterrows():
            x = RANGES.index(int(r.range_m)); y = FAMILIES.index(r.response_family)
            ax.scatter(x, y, s=18 + 430 * r.rms_partial_rho ** 2, color=cmap(norm(r.median_partial_rho)),
                       edgecolor="#202020" if r.range_maxT_p <= .05 else "white",
                       lw=.9 if r.range_maxT_p <= .05 else .3)
            if r.q_module_all_taxon_family_ranges <= .05:
                ax.text(x, y, "•", color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")
        ax.set_xticks(range(13), XLABELS, rotation=45, ha="right"); ax.set_xlabel("range（km）")
        ax.set_xlim(-.5, 12.5); ax.set_ylim(4.45, -.55)
        ax.set_title(MODULE_CN[module], loc="left", fontweight="bold")
        if j == 0:
            ax.set_yticks(range(5), [FAMILY_CN[f] for f in FAMILIES])
        else:
            ax.tick_params(axis="y", labelleft=False)
        panel(ax, chr(ord("a") + j))
    axes[0].set_yticks(range(5), [FAMILY_CN[f] for f in FAMILIES])
    fig.suptitle(f"{water} · {TAXON_CN[taxon]}｜五家族 × 三环境模块的13-range响应", y=1.035, fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=.10, right=.985, top=.84, bottom=.20, wspace=.08)
    name = f"{water}_{TAXON_CN[taxon]}_五家族13尺度分面"
    fig.savefig(sub / "04_论文级图件" / "类群分面PNG" / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=.04)
    fig.savefig(sub / "04_论文级图件" / "类群分面PDF" / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    setup_style()
    out = Path(__file__).resolve().parents[1]
    sub = out / "13_生物类群×五家族精细分析_2026-07-24"
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        figure08(sub)
        atlas_figure(sub, "River", "09")
        atlas_figure(sub, "Lake", "10")
        figure11(sub)
        figure12(sub)
    elif target == "river-atlas":
        atlas_figure(sub, "River", "09")
    elif target == "lake-atlas":
        atlas_figure(sub, "Lake", "10")
    elif target != "taxon-cards":
        raise ValueError("target must be all, river-atlas, lake-atlas, or taxon-cards")
    if target in {"all", "taxon-cards"}:
        omnibus = pd.read_csv(sub / "02_类群家族整体尺度检验" / "生物类群×五家族×三环境模块_13尺度共同置换检验.csv")
        for water in ["River", "Lake"]:
            for taxon in TAXA[water]:
                taxon_card(sub, omnibus, water, taxon)
    files = sorted(str(p.relative_to(sub)) for p in (sub / "04_论文级图件").rglob("*") if p.is_file())
    summary = {"main_figures": 5, "taxon_cards": 11, "figure_files": len(files), "files": files}
    (sub / "06_QA" / "图件生成清单.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
