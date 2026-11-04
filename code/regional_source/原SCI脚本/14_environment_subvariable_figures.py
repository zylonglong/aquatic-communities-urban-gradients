from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_CN = {
    "terrain": "地形地貌",
    "urbanization": "城市化建设",
    "human_activity": "人类活动",
}
FAMILY_ORDER = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
FAMILY_SHORT = {
    "community_baseline": "基础特征",
    "alpha_diversity": "α多样性",
    "beta_diversity": "β多样性",
    "community_composition": "群落组成",
    "functional_indicator": "功能/指示",
}
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
SUB_ORDER = {
    "terrain": [
        "mean_elevation",
        "relative_elevation",
        "terrain_relief",
        "slope_intensity",
        "steep_slope_cover",
    ],
    "urbanization": [
        "road_cover",
        "building_cover",
        "traffic_facility_cover",
        "transport_facility_cover",
        "urban_land_cover",
        "green_natural_buffer",
    ],
    "human_activity": ["api_expert_weighted", "population_density"],
}
SUB_CN = {
    "mean_elevation": "平均高程",
    "relative_elevation": "相对高程",
    "terrain_relief": "地形起伏",
    "slope_intensity": "坡度强度",
    "steep_slope_cover": "陡坡覆盖",
    "road_cover": "道路覆盖",
    "building_cover": "建筑覆盖",
    "traffic_facility_cover": "交通设施",
    "transport_facility_cover": "运输设施",
    "urban_land_cover": "建设用地",
    "green_natural_buffer": "绿色/自然缓冲",
    "api_expert_weighted": "加权API",
    "population_density": "人口密度",
}
COLORS = {
    "mean_elevation": "#0F4D92",
    "relative_elevation": "#3775BA",
    "terrain_relief": "#42949E",
    "slope_intensity": "#7656A5",
    "steep_slope_cover": "#9A4D8E",
    "road_cover": "#B64342",
    "building_cover": "#E07A5F",
    "traffic_facility_cover": "#F2A65A",
    "transport_facility_cover": "#8C6D62",
    "urban_land_cover": "#D1495B",
    "green_natural_buffer": "#3A8D5D",
    "api_expert_weighted": "#B64342",
    "population_density": "#0F4D92",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "Arial", "sans-serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_scale_bands(ax) -> None:
    """Show all 13 cumulative ranges without imposing categorical boundaries."""
    for range_m in RANGES:
        ax.axvline(
            range_m / 1000,
            color="#E6E8EB",
            lw=0.38,
            alpha=0.9,
            zorder=0,
        )


def save_figure(fig, out: Path, basename: str) -> list[str]:
    paths = []
    specs = [
        ("PNG", "png", 600),
        ("PDF", "pdf", 300),
        ("SVG", "svg", 300),
        ("TIFF", "tiff", 600),
    ]
    for folder, ext, dpi in specs:
        path = out / "04_论文级图件" / folder / f"{basename}.{ext}"
        kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.06}
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def figure_overall_profiles(overall: pd.DataFrame, out: Path) -> list[str]:
    fig = plt.figure(figsize=(16.8, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.35, 0.78, 1.35, 0.78],
        hspace=0.18,
        wspace=0.18,
    )
    letters = iter("abcdefghijkl")
    for wi, water in enumerate(["River", "Lake"]):
        for mi, module in enumerate(MODULES):
            ax_share = fig.add_subplot(gs[wi * 2, mi])
            ax_dir = fig.add_subplot(gs[wi * 2 + 1, mi], sharex=ax_share)
            panel_label(ax_share, next(letters))
            panel_label(ax_dir, next(letters))
            q = overall[
                overall.water_body_type.eq(water)
                & overall.environment_module.eq(module)
            ]
            x = np.asarray(RANGES) / 1000
            shares = []
            directions = []
            for sub in SUB_ORDER[module]:
                g = q[q.environment_subvariable.eq(sub)].set_index("range_m")
                shares.append(
                    g.reindex(RANGES).family_equal_johnson_contribution_share.to_numpy()
                )
                directions.append(
                    g.reindex(RANGES).family_equal_median_partial_rho.to_numpy()
                )
            add_scale_bands(ax_share)
            ax_share.stackplot(
                x,
                shares,
                colors=[COLORS[s] for s in SUB_ORDER[module]],
                alpha=0.88,
                linewidth=0.35,
                edgecolor="white",
                labels=[SUB_CN[s] for s in SUB_ORDER[module]],
            )
            ax_share.set_ylim(0, 1)
            ax_share.set_ylabel("相对贡献份额")
            ax_share.set_title(f"{water} · {MODULE_CN[module]}")
            ax_share.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2 if len(SUB_ORDER[module]) > 3 else 1,
                handlelength=1.2,
                columnspacing=0.8,
            )
            ax_share.tick_params(labelbottom=False)
            add_scale_bands(ax_dir)
            for sub, y in zip(SUB_ORDER[module], directions):
                ax_dir.plot(
                    x,
                    y,
                    marker="o",
                    ms=2.8,
                    lw=1.35,
                    color=COLORS[sub],
                    label=SUB_CN[sub],
                )
            ax_dir.axhline(0, color="#4D4D4D", lw=0.75, ls="--")
            max_abs = np.nanmax(np.abs(np.concatenate(directions)))
            lim = max(0.20, min(0.55, math_ceil(max_abs + 0.04, 0.05)))
            ax_dir.set_ylim(-lim, lim)
            ax_dir.set_ylabel("signed median ρ")
            ax_dir.set_xlabel("累积 range（km）" if wi == 1 else "")
            ax_dir.set_xticks(np.asarray(RANGES) / 1000)
            ax_dir.tick_params(axis="x", labelrotation=60, labelsize=5.8)
            ax_dir.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.75)
    fig.suptitle(
        "Figure S13 | 三类环境因素内部子变量的贡献份额与响应方向沿 range 的变化",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(fig, out, "FigureS13_环境子变量总体贡献与方向轨迹")


def math_ceil(value: float, step: float) -> float:
    return np.ceil(value / step) * step


def bubble_atlas(family: pd.DataFrame, water: str, out: Path, number: int) -> list[str]:
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(17.2, 10.5),
        sharex=True,
        constrained_layout=True,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.40, vcenter=0, vmax=0.40)
    cmap = plt.get_cmap("RdBu_r")
    for mi, module in enumerate(MODULES):
        for fi, family_key in enumerate(FAMILY_ORDER):
            ax = axes[mi, fi]
            panel_label(ax, chr(ord("a") + mi * 5 + fi))
            add_scale_bands(ax)
            q = family[
                family.water_body_type.eq(water)
                & family.environment_module.eq(module)
                & family.response_family.eq(family_key)
            ]
            suborder = SUB_ORDER[module]
            for yi, sub in enumerate(suborder):
                g = q[q.environment_subvariable.eq(sub)].set_index("range_m").reindex(RANGES)
                x = np.asarray(RANGES) / 1000
                y = np.full(len(RANGES), yi)
                size = 15 + 330 * g.johnson_relative_contribution_share.to_numpy()
                edge = np.where(
                    g.subvariable_range_maxT_p.to_numpy() <= 0.05,
                    "#111111",
                    "#FFFFFF",
                )
                lw = np.where(
                    g.subvariable_range_maxT_p.to_numpy() <= 0.05, 0.9, 0.25
                )
                ax.scatter(
                    x,
                    y,
                    s=size,
                    c=g.median_partial_rho.to_numpy(),
                    cmap=cmap,
                    norm=norm,
                    edgecolors=edge,
                    linewidths=lw,
                    alpha=0.92,
                )
            ax.set_ylim(len(suborder) - 0.5, -0.5)
            ax.set_yticks(range(len(suborder)))
            ax.set_yticklabels([SUB_CN[s] for s in suborder] if fi == 0 else [])
            if mi == 0:
                ax.set_title(FAMILY_SHORT[family_key])
            if fi == 0:
                ax.set_ylabel(MODULE_CN[module], fontweight="bold")
            if mi == 2:
                ax.set_xlabel("range（km）")
            ax.set_xticks(np.asarray(RANGES) / 1000)
            ax.tick_params(axis="x", labelrotation=60, labelsize=5.4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.035,
        aspect=45,
    )
    cbar.set_label("signed median partial Spearman ρ")
    size_handles = [
        plt.scatter([], [], s=15 + 330 * v, color="#8DA0CB", edgecolor="white")
        for v in [0.1, 0.3, 0.5]
    ]
    support_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#DDDDDD",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=7,
        label="maxT p≤0.05",
    )
    fig.legend(
        size_handles + [support_handle],
        ["贡献份额 0.10", "0.30", "0.50", "maxT p≤0.05"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=4,
    )
    fig.suptitle(
        f"Figure S{number} | {water} 五类生物响应对环境子变量的13-range响应图谱",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(
        fig, out, f"FigureS{number}_{water}_五家族环境子变量13尺度图谱"
    )


def taxon_family_matrix(
    summary: pd.DataFrame, water: str, out: Path, number: int
) -> list[str]:
    taxa = TAXA[water]
    rows = [(t, f) for t in taxa for f in FAMILY_ORDER]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.4, 13.4),
        sharey=True,
        constrained_layout=True,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.35)
    cmap = plt.get_cmap("RdBu_r")
    for mi, module in enumerate(MODULES):
        ax = axes[mi]
        panel_label(ax, chr(ord("a") + mi))
        subs = SUB_ORDER[module]
        q = summary[
            summary.water_body_type.eq(water)
            & summary.environment_module.eq(module)
        ]
        for yi, (taxon, family_key) in enumerate(rows):
            for xi, sub in enumerate(subs):
                g = q[
                    q.taxon_group.eq(taxon)
                    & q.response_family.eq(family_key)
                    & q.environment_subvariable.eq(sub)
                ]
                if g.empty:
                    continue
                row = g.iloc[0]
                edge = "black" if row.maxT_supported_range_n > 0 else "white"
                lw = 0.9 if row.maxT_supported_range_n > 0 else 0.25
                ax.scatter(
                    xi,
                    yi,
                    s=20 + 420 * row.mean_johnson_relative_contribution_share,
                    c=[row.mean_median_partial_rho],
                    cmap=cmap,
                    norm=norm,
                    edgecolors=edge,
                    linewidths=lw,
                )
        ax.set_xlim(-0.6, len(subs) - 0.4)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels([SUB_CN[s] for s in subs], rotation=35, ha="right")
        ax.set_title(MODULE_CN[module])
        for boundary in np.arange(5, len(rows), 5):
            ax.axhline(boundary - 0.5, color="#808080", lw=0.55)
        ax.grid(axis="x", color="#E5E5E5", lw=0.4)
    labels = [
        f"{TAXON_CN[t]} · {FAMILY_SHORT[f]}"
        for t, f in rows
    ]
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(labels)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.025,
        aspect=45,
    )
    cbar.set_label("13 range 平均 signed median partial ρ")
    handles = [
        plt.scatter([], [], s=20 + 420 * v, color="#8DA0CB", edgecolor="white")
        for v in [0.15, 0.35, 0.60]
    ]
    support = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#DDDDDD",
        markeredgecolor="black",
        markersize=7,
    )
    fig.legend(
        handles + [support],
        ["平均贡献 0.15", "0.35", "0.60", "至少1个 range：maxT p≤0.05"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.008),
        ncol=4,
    )
    fig.suptitle(
        f"Figure S{number} | {water} 生物类群 × 五家族的环境子变量贡献与方向",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(
        fig, out, f"FigureS{number}_{water}_类群家族环境子变量贡献矩阵"
    )


def selected_trajectories(
    taxon: pd.DataFrame, summary: pd.DataFrame, out: Path
) -> tuple[list[str], pd.DataFrame]:
    chosen = []
    for water in ["River", "Lake"]:
        for module in MODULES:
            q = summary[
                summary.water_body_type.eq(water)
                & summary.environment_module.eq(module)
                & summary.maxT_supported_range_n.gt(0)
            ].copy()
            q["priority"] = (
                q.mean_johnson_relative_contribution_share
                * q.mean_rms_partial_rho
                * (1 + q.maxT_supported_range_n / 13)
            )
            q = q.sort_values("priority", ascending=False)
            seen_taxa = set()
            take = []
            for _, row in q.iterrows():
                if row.taxon_group in seen_taxa and len(take) < 1:
                    continue
                take.append(row)
                seen_taxa.add(row.taxon_group)
                if len(take) == 2:
                    break
            chosen.extend(take)
    chosen = pd.DataFrame(chosen)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.6, 8.4),
        sharex=True,
        constrained_layout=True,
    )
    for wi, water in enumerate(["River", "Lake"]):
        for mi, module in enumerate(MODULES):
            ax = axes[wi, mi]
            panel_label(ax, chr(ord("a") + wi * 3 + mi))
            add_scale_bands(ax)
            q = chosen[
                chosen.water_body_type.eq(water)
                & chosen.environment_module.eq(module)
            ]
            for j, (_, sel) in enumerate(q.iterrows()):
                g = taxon[
                    taxon.water_body_type.eq(water)
                    & taxon.environment_module.eq(module)
                    & taxon.taxon_group.eq(sel.taxon_group)
                    & taxon.response_family.eq(sel.response_family)
                    & taxon.environment_subvariable.eq(
                        sel.environment_subvariable
                    )
                ].sort_values("range_m")
                color = COLORS[sel.environment_subvariable]
                line_styles = ["-", "--"]
                markers = ["o", "s"]
                label = (
                    f"{sel.taxon_group_cn}·{FAMILY_SHORT[sel.response_family]}—"
                    f"{SUB_CN[sel.environment_subvariable]} "
                    f"(平均贡献 {sel.mean_johnson_relative_contribution_share:.2f})"
                )
                ax.plot(
                    g.range_m / 1000,
                    g.median_partial_rho,
                    color=color,
                    marker=markers[j % len(markers)],
                    linestyle=line_styles[j % len(line_styles)],
                    lw=1.7,
                    ms=3.8,
                    label=label,
                )
                supported = g.subvariable_range_maxT_p <= 0.05
                ax.scatter(
                    g.loc[supported, "range_m"] / 1000,
                    g.loc[supported, "median_partial_rho"],
                    s=42,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=4,
                )
            ax.axhline(0, color="#4D4D4D", lw=0.75, ls="--")
            ax.set_ylim(-0.55, 0.55)
            ax.set_title(f"{water} · {MODULE_CN[module]}")
            ax.set_ylabel("signed median partial ρ")
            ax.set_xlabel("累积 range（km）")
            ax.set_xticks(np.asarray(RANGES) / 1000)
            ax.tick_params(axis="x", labelrotation=60, labelsize=5.8)
            ax.grid(axis="y", color="#DDDDDD", lw=0.4)
            ax.legend(loc="best", fontsize=6.8)
    fig.suptitle(
        "Figure S18 | 贡献度与跨尺度统计支持共同指向的代表性生态响应通道",
        fontsize=13,
        fontweight="bold",
    )
    return (
        save_figure(fig, out, "FigureS18_代表性子变量类群家族尺度轨迹"),
        chosen,
    )


def collinearity_figure(collinearity: pd.DataFrame, out: Path) -> list[str]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 7.6),
        sharex=True,
        constrained_layout=True,
    )
    for wi, water in enumerate(["River", "Lake"]):
        for mi, module in enumerate(MODULES):
            ax = axes[wi, mi]
            panel_label(ax, chr(ord("a") + wi * 3 + mi))
            q = collinearity[
                collinearity.water_body_type.eq(water)
                & collinearity.environment_module.eq(module)
            ].sort_values("range_m")
            x = q.range_m / 1000
            ax.plot(
                x,
                q.max_abs_spearman_between_subvariables,
                color="#0F4D92",
                marker="o",
                ms=3.2,
                lw=1.6,
                label="最大 |ρ|",
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("模块内最大 |Spearman ρ|", color="#0F4D92")
            ax.tick_params(axis="y", labelcolor="#0F4D92")
            ax2 = ax.twinx()
            ax2.plot(
                x,
                q.condition_index,
                color="#B64342",
                marker="s",
                ms=2.8,
                lw=1.3,
                label="条件指数",
            )
            ax2.set_ylabel("条件指数", color="#B64342")
            ax2.tick_params(axis="y", labelcolor="#B64342")
            ax.set_title(f"{water} · {MODULE_CN[module]}")
            ax.set_xlabel("range（km）")
            ax.set_xticks(np.asarray(RANGES) / 1000)
            ax.tick_params(axis="x", labelrotation=60, labelsize=5.8)
            ax.grid(axis="y", color="#DDDDDD", lw=0.4)
    fig.suptitle(
        "Figure S19 | 环境子变量共线性随空间 range 的变化",
        fontsize=13,
        fontweight="bold",
    )
    return save_figure(fig, out, "FigureS19_环境子变量共线性诊断")


def main() -> None:
    style()
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    out = root / "14_三类环境子变量×生物五家族尺度分解_2026-07-27"
    family = pd.read_csv(
        out
        / "02_五家族子变量尺度分解"
        / "五家族×环境子变量×13尺度_共同置换与贡献份额.csv"
    )
    overall = pd.read_csv(
        out / "02_五家族子变量尺度分解" / "五家族等权_环境子变量总体贡献与方向.csv"
    )
    taxon = pd.read_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "生物类群×五家族×环境子变量×13尺度_共同置换与贡献份额.csv"
    )
    summary = pd.read_csv(
        out
        / "03_类群×五家族子变量尺度分解"
        / "类群×五家族子变量尺度轨迹汇总.csv"
    )
    collinearity = pd.read_csv(
        out / "01_环境子变量审计" / "模块内子变量共线性_13尺度.csv"
    )
    manifest = []
    manifest += figure_overall_profiles(overall, out)
    manifest += bubble_atlas(family, "River", out, 14)
    manifest += bubble_atlas(family, "Lake", out, 15)
    manifest += taxon_family_matrix(summary, "River", out, 16)
    manifest += taxon_family_matrix(summary, "Lake", out, 17)
    paths, chosen = selected_trajectories(taxon, summary, out)
    manifest += paths
    manifest += collinearity_figure(collinearity, out)

    source_dir = out / "05_SourceData"
    overall.to_csv(
        source_dir / "FigureS13_environment_subvariable_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    family[family.water_body_type.eq("River")].to_csv(
        source_dir / "FigureS14_River_family_subvariable_atlas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    family[family.water_body_type.eq("Lake")].to_csv(
        source_dir / "FigureS15_Lake_family_subvariable_atlas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary[summary.water_body_type.eq("River")].to_csv(
        source_dir / "FigureS16_River_taxon_family_subvariable_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary[summary.water_body_type.eq("Lake")].to_csv(
        source_dir / "FigureS17_Lake_taxon_family_subvariable_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    chosen.to_csv(
        source_dir / "FigureS18_selected_trajectory_keys.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_keys = chosen[
        [
            "water_body_type",
            "environment_module",
            "taxon_group",
            "response_family",
            "environment_subvariable",
        ]
    ]
    selected_source = taxon.merge(
        selected_keys,
        on=[
            "water_body_type",
            "environment_module",
            "taxon_group",
            "response_family",
            "environment_subvariable",
        ],
        how="inner",
    )
    selected_source.to_csv(
        source_dir / "FigureS18_selected_trajectory_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    collinearity.to_csv(
        source_dir / "FigureS19_collinearity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out / "06_QA" / "图件生成清单.json").write_text(
        json.dumps(
            {
                "figure_files": manifest,
                "figure_file_n": len(manifest),
                "figure_basename_n": len(manifest) // 4,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "figure_file_n": len(manifest),
                "figure_basename_n": len(manifest) // 4,
                "selected_trajectory_n": len(chosen),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
