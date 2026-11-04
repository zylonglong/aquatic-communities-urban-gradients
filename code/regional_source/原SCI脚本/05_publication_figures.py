from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
XLABELS = ["0.2", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6"]
WATERS = ["River", "Lake"]
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_CN = {"terrain": "地形地貌", "urbanization": "城市化建设", "human_activity": "人类活动压力"}
MODULE_SHORT = {"terrain": "地形", "urbanization": "城市化", "human_activity": "人类活动"}
MODULE_COLOR = {"terrain": "#2F7D62", "urbanization": "#D05A4E", "human_activity": "#5D6FB5"}
WATER_COLOR = {"River": "#246B9A", "Lake": "#1B9E8A"}
FAMILY_ORDER = ["community_baseline", "alpha_diversity", "beta_diversity", "community_composition", "functional_indicator"]
FAMILY_CN = {
    "community_baseline": "群落基础",
    "alpha_diversity": "α多样性",
    "beta_diversity": "β多样性",
    "community_composition": "群落组成",
    "functional_indicator": "功能/指示",
}
TAXON_ORDER = ["benthos", "fish", "macrophyte", "phytoplankton", "periphyton", "zooplankton"]
TAXON_CN = {
    "benthos": "底栖动物", "fish": "鱼类", "macrophyte": "水生植物",
    "phytoplankton": "浮游植物", "periphyton": "着生藻类", "zooplankton": "浮游动物",
}


def setup_style() -> None:
    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_all(fig: plt.Figure, out: Path, name: str) -> None:
    specs = [("PNG", "png", 400), ("TIFF", "tiff", 600), ("PDF", "pdf", 300), ("SVG", "svg", 300)]
    for folder, ext, dpi in specs:
        fig.savefig(out / "08_论文级图件" / folder / f"{name}.{ext}", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def panel(ax, letter: str) -> None:
    ax.text(-0.10, 1.035, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left")


def add_scale_domains(ax, alpha=1.0) -> None:
    """Show the 13 observed cumulative ranges without imposing scale domains."""
    for x in range(len(RANGES)):
        ax.axvline(x, color="#E6E8EB", lw=0.38, alpha=0.9, zorder=0)


def short_label(s: str, max_len=20) -> str:
    s = str(s).replace("·总体", "").replace("·总·", "·")
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def figure1(out: Path) -> None:
    audit = out / "01_数据审计"
    cov = pd.read_csv(audit / "样本年份水文期覆盖.csv")
    avail = pd.read_csv(audit / "River_Lake响应变量可用性与结构性缺失.csv")
    avail["primary_eligible"] = avail.primary_eligible.astype(str).str.lower().eq("true")
    counts = (avail[avail.primary_eligible]
              .groupby(["water_body_type", "response_family"]).size().reset_index(name="n"))

    fig = plt.figure(figsize=(12.0, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[1, 1], wspace=0.30, hspace=0.42)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    panel(ax0, "a")
    ax0.set_title("四科学问题与分层统计路径", loc="left", fontweight="bold", pad=8)
    xcols = [0.03, 0.37, 0.70]
    for x, title, color in zip(xcols, ["环境梯度", "生态响应", "尺度解释"], ["#ECE9E2", "#E8F1F5", "#EEF0E8"]):
        ax0.add_patch(FancyBboxPatch((x, 0.72), 0.27, 0.18, boxstyle="round,pad=0.012,rounding_size=0.015",
                                     facecolor=color, edgecolor="#555555", lw=0.8))
        ax0.text(x + 0.135, 0.855, title, ha="center", va="center", fontweight="bold", fontsize=9)
    ax0.text(0.165, 0.785, "地形地貌\n城市化建设\n人类活动压力", ha="center", va="center", linespacing=1.5)
    ax0.text(0.505, 0.785, "水质等级（参照）\n5类水生生物特征（主体）", ha="center", va="center", linespacing=1.5)
    ax0.text(
        0.835,
        0.785,
        "13 个递增累积 range\n0.2、0.5、1.0…6.0 km\n逐 range 比较",
        ha="center",
        va="center",
        linespacing=1.45,
    )
    for x1, x2 in [(0.30, 0.37), (0.64, 0.70)]:
        ax0.annotate("", xy=(x2, 0.81), xytext=(x1, 0.81), arrowprops=dict(arrowstyle="->", lw=1, color="#555555"))

    qboxes = [
        ("Q1", "水质等级的整体响应", "独立空间几何\n区域比较参照"),
        ("Q2", "五类生物特征整体响应", "Site 为独立单位\n家族共同置换"),
        ("Q3", "水质—生物尺度指纹差异", "非配对区域比较\n不作因果中介"),
        ("Q4", "具体生物特征与驱动变量", "分层筛选\n群落与β过程"),
    ]
    for i, (q, title, note) in enumerate(qboxes):
        y = 0.53 - i * 0.135
        color = ["#F4E7E2", "#E5EFF4", "#ECEAF5", "#E7F1EA"][i]
        ax0.add_patch(FancyBboxPatch((0.05, y), 0.88, 0.095, boxstyle="round,pad=0.012,rounding_size=0.012",
                                     facecolor=color, edgecolor="#777777", lw=0.65))
        ax0.text(0.085, y + 0.048, q, ha="center", va="center", fontweight="bold", color="#333333")
        ax0.text(0.15, y + 0.062, title, ha="left", va="center", fontweight="bold")
        ax0.text(0.73, y + 0.048, note, ha="center", va="center", fontsize=7.1, color="#444444")

    panel(ax1, "b")
    ax1.set_title("独立站点与重复调查记录", loc="left", fontweight="bold")
    x = np.arange(2)
    sites = [86, 44]; samples = [409, 155]
    ax1.bar(x - 0.18, sites, 0.36, color=[WATER_COLOR["River"], WATER_COLOR["Lake"]], alpha=0.95, label="Site（主分析）")
    ax1.bar(x + 0.18, samples, 0.36, color=[WATER_COLOR["River"], WATER_COLOR["Lake"]], alpha=0.30,
            edgecolor=[WATER_COLOR["River"], WATER_COLOR["Lake"]], label="Sample（时间敏感性）")
    for i, (a, b) in enumerate(zip(sites, samples)):
        ax1.text(i - .18, a + 8, str(a), ha="center", fontsize=7.5)
        ax1.text(i + .18, b + 8, str(b), ha="center", fontsize=7.5)
    ax1.set_xticks(x, ["River", "Lake"]); ax1.set_ylabel("数量（n）")
    ax1.set_ylim(0, 470); ax1.legend(loc="upper right")

    panel(ax2, "c")
    ax2.set_title("五类生物响应变量的主分析覆盖", loc="left", fontweight="bold")
    pivot = counts.pivot(index="water_body_type", columns="response_family", values="n").reindex(index=WATERS, columns=FAMILY_ORDER).fillna(0)
    for yi, w in enumerate(WATERS):
        for xi, fam in enumerate(FAMILY_ORDER):
            n = pivot.loc[w, fam]
            ax2.scatter(xi, yi, s=34 + n * 5.0, color=WATER_COLOR[w], alpha=.82, edgecolor="white", lw=.6)
            ax2.text(xi, yi, f"{int(n)}", ha="center", va="center", color="white", fontsize=6.3, fontweight="bold")
    ax2.set_xticks(range(5), [FAMILY_CN[f] for f in FAMILY_ORDER], rotation=22, ha="right")
    ax2.set_yticks([0, 1], ["River", "Lake"]); ax2.set_xlim(-.6, 4.6); ax2.set_ylim(1.55, -.55)
    ax2.set_xlabel("圆面积表示可用变量数；数字为变量数")
    sns.despine(ax=ax2, left=True)
    save_all(fig, out, "Figure01_研究设计与数据覆盖")


def figure2(out: Path) -> None:
    wq = pd.read_csv(out / "09_SourceData" / "Figure02_Q1_water_quality_scale.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True, sharey=True)
    for i, water in enumerate(WATERS):
        for j, mod in enumerate(MODULES):
            ax = axes[i, j]; add_scale_domains(ax, .80)
            g = wq[(wq.water_body_type == water) & (wq.environment_module == mod)].sort_values("range_m")
            x = np.arange(len(g))
            ax.fill_between(x, g.ci95_low, g.ci95_high, color=MODULE_COLOR[mod], alpha=.16, lw=0)
            ax.plot(x, g.spearman_rho, color=MODULE_COLOR[mod], lw=1.7, marker="o", ms=4.2,
                    markerfacecolor="white", markeredgewidth=.9)
            sig = g.fdr_q <= .05
            ax.scatter(x[sig], g.loc[sig, "spearman_rho"], s=28, color=MODULE_COLOR[mod], zorder=4)
            ax.axhline(0, color="#666666", lw=.7, ls="--")
            ax.set_ylim(-1.0, 1.0); ax.set_xlim(-.45, 12.45)
            ax.set_title(f"{water} · {MODULE_CN[mod]}", loc="left", fontweight="bold")
            ax.text(.98, .05, f"n={int(g.n_spatial_units.iloc[0])} 独立空间单元", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.8, color="#555555")
            if i == 1:
                ax.set_xticks(range(13), XLABELS, rotation=45, ha="right")
                ax.set_xlabel("空间范围（km）")
            if j == 0:
                ax.set_ylabel("Spearman ρ\n（水质等级越高表示越差）")
            panel(ax, chr(ord("a") + i * 3 + j))
    legend = [
        Line2D([0], [0], color="#555555", marker="o", mfc="white", label="ρ 与 95% bootstrap CI"),
        Line2D([0], [0], color="#555555", marker="o", mfc="#555555", ls="", label="13 range FDR q ≤ 0.05"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, 1.015), ncol=2, frameon=False)
    fig.suptitle("Q1｜水质等级对三类环境梯度的区域尺度响应参照", y=1.055, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.88, wspace=.16, hspace=.25)
    save_all(fig, out, "Figure02_Q1_水质等级尺度响应参照")


def family_panel(ax, data: pd.DataFrame, water: str, letter: str) -> None:
    rows = [(m, f) for m in MODULES for f in FAMILY_ORDER]
    ylabels = [f"{MODULE_SHORT[m]}｜{FAMILY_CN[f]}" for m, f in rows]
    add_scale_domains(ax, .72)
    cmap = mpl.colormaps["RdBu_r"]
    norm = TwoSlopeNorm(vmin=-.40, vcenter=0, vmax=.40)
    for yi, (m, f) in enumerate(rows):
        g = data[(data.water_body_type == water) & (data.environment_module == m) &
                 (data.response_family == f)].set_index("range_m").reindex(RANGES)
        for xi, r in enumerate(RANGES):
            row = g.loc[r]
            val = row.median_partial_rho
            rms = row.rms_partial_rho
            ax.scatter(xi, yi, s=18 + 350 * rms * rms, color=cmap(norm(val)),
                       edgecolor="#202020" if row.range_maxT_p <= .05 else "white",
                       lw=1.0 if row.range_maxT_p <= .05 else .35, zorder=3)
    ax.set_yticks(range(len(rows)), ylabels)
    ax.set_xticks(range(13), XLABELS, rotation=45, ha="right")
    ax.set_xlabel("空间范围（km）")
    ax.set_xlim(-.5, 12.5); ax.set_ylim(len(rows) - .45, -.55)
    ax.set_title(f"{water}｜点色=signed median ρ；点面积=家族 RMS |ρ|", loc="left", fontweight="bold")
    for y in [4.5, 9.5]:
        ax.axhline(y, color="#888888", lw=.7)
    panel(ax, letter)


def figure3(out: Path) -> None:
    data = pd.read_csv(out / "09_SourceData" / "Figure03_Q2_biology_family_scale.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 8.5), sharex=True, sharey=True)
    family_panel(axes[0], data, "River", "a")
    family_panel(axes[1], data, "Lake", "b")
    cmap = mpl.colormaps["RdBu_r"]; norm = TwoSlopeNorm(vmin=-.40, vcenter=0, vmax=.40)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([.27, .045, .46, .018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("家族内具体响应的 signed median partial Spearman ρ")
    size_handles = [plt.scatter([], [], s=18 + 350 * v * v, facecolor="#888888", edgecolor="white", label=f"RMS={v:.1f}") for v in [.1, .2, .3]]
    ring = plt.scatter([], [], s=34, facecolor="white", edgecolor="#202020", lw=1, label="共同置换 maxT p ≤ 0.05")
    fig.legend(handles=size_handles + [ring], loc="upper center", bbox_to_anchor=(.5, .975), ncol=4, frameon=False)
    fig.suptitle("Q2｜五类水生生物特征对三类环境梯度的 13-range 响应谱", y=1.015, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.91, bottom=.17, wspace=.28)
    save_all(fig, out, "Figure03_Q2_五类生物响应尺度谱")


def figure4(out: Path) -> None:
    d = pd.read_csv(out / "09_SourceData" / "Figure04_Q3_wq_biology_comparison.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=True, sharey=True)
    for i, water in enumerate(WATERS):
        for j, mod in enumerate(MODULES):
            ax = axes[i, j]; add_scale_domains(ax, .78)
            g = d[(d.water_body_type == water) & (d.environment_module == mod)].sort_values("range_m")
            x = np.arange(len(g))
            ax.plot(x, g.spearman_rho, color="#333333", lw=1.6, marker="o", ms=4.0,
                    mfc="white", label="水质等级 ρ")
            ax.plot(x, g.bio_family_median_signed_rho, color=MODULE_COLOR[mod], lw=1.7, marker="s", ms=3.8,
                    mfc=MODULE_COLOR[mod], label="生物家族 median ρ")
            ax.scatter(x, np.zeros_like(x) - .68, s=12 + 16 * g.bio_supported_family_n,
                       color=MODULE_COLOR[mod], alpha=.75, clip_on=False)
            ax.axhline(0, color="#777777", lw=.7, ls="--")
            ax.set_ylim(-.75, .75); ax.set_xlim(-.45, 12.45)
            ax.set_title(f"{water} · {MODULE_CN[mod]}", loc="left", fontweight="bold")
            if i == 1:
                ax.set_xticks(range(13), XLABELS, rotation=45, ha="right"); ax.set_xlabel("空间范围（km）")
            if j == 0:
                ax.set_ylabel("同尺度响应方向（ρ）")
            panel(ax, chr(ord("a") + i * 3 + j))
    handles = [
        Line2D([0], [0], color="#333333", marker="o", mfc="white", label="水质等级 Spearman ρ"),
        Line2D([0], [0], color="#777777", marker="s", label="5个生物家族 signed median ρ"),
        Line2D([0], [0], color="#777777", marker="o", ls="", ms=5, label="底部圆面积=该尺度 maxT 支持的生物家族数"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99), ncol=3, frameon=False)
    fig.suptitle("Q3｜水质等级与水生生物对环境梯度的区域尺度指纹比较", y=1.035, fontsize=11, fontweight="bold")
    fig.text(.5, .01, "注：两套数据的独立空间单元不同，本图为区域尺度指纹对照，不是站点配对或因果中介分析。",
             ha="center", fontsize=7.2, color="#555555")
    fig.subplots_adjust(top=.90, bottom=.14, wspace=.16, hspace=.25)
    save_all(fig, out, "Figure04_Q3_水质与生物尺度指纹对照")


def selected_response_matrix(out: Path, water: str) -> tuple[pd.DataFrame, list[str]]:
    top = pd.read_csv(out / "05_Q4_具体生物响应" / "各水体_环境模块_生物家族代表响应.csv")
    assoc = pd.read_csv(out / "05_Q4_具体生物响应" / "全部具体响应_偏Spearman_分层FDR.csv")
    sel = (top[top.water_body_type == water]
           .sort_values(["environment_module", "response_family", "screen_rank", "max_abs_rho"])
           .groupby(["environment_module", "response_family"], as_index=False).head(1))
    long = assoc.merge(sel[["water_body_type", "environment_module", "response_family", "column_name"]],
                       on=["water_body_type", "environment_module", "response_family", "column_name"], how="inner")
    row_order, labels = [], []
    for m in MODULES:
        for f in FAMILY_ORDER:
            q = sel[(sel.environment_module == m) & (sel.response_family == f)]
            if q.empty:
                continue
            c = q.iloc[0].column_name
            row_order.append((m, f, c))
            labels.append(f"{MODULE_SHORT[m]}｜{FAMILY_CN[f]}｜{short_label(q.iloc[0].variable_name_cn, 17)}")
    long["row_key"] = list(zip(long.environment_module, long.response_family, long.column_name))
    long["row_index"] = long.row_key.map({k: i for i, k in enumerate(row_order)})
    return long.dropna(subset=["row_index"]), labels


def response_panel(ax, d: pd.DataFrame, labels: list[str], title: str, letter: str) -> None:
    add_scale_domains(ax, .78)
    cmap = mpl.colormaps["RdBu_r"]; norm = TwoSlopeNorm(vmin=-.55, vcenter=0, vmax=.55)
    for _, r in d.iterrows():
        x = RANGES.index(int(r.range_m)); y = int(r.row_index)
        ax.scatter(x, y, s=33, color=cmap(norm(r.partial_spearman_rho)),
                   edgecolor="#202020" if r.q_within_response_13ranges <= .10 else "white",
                   lw=.9 if r.q_within_response_13ranges <= .10 else .3)
        if r.q_within_family_all_responses_ranges <= .05:
            ax.text(x, y, "•", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(13), XLABELS, rotation=45, ha="right"); ax.set_xlabel("空间范围（km）")
    ax.set_xlim(-.5, 12.5); ax.set_ylim(len(labels) - .45, -.55)
    ax.set_title(title, loc="left", fontweight="bold")
    for y in [4.5, 9.5]:
        ax.axhline(y, color="#888888", lw=.7)
    panel(ax, letter)


def figure5(out: Path) -> None:
    dr, lr = selected_response_matrix(out, "River"), selected_response_matrix(out, "Lake")
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 9.0), sharex=True)
    response_panel(axes[0], dr[0], dr[1], "River｜每个环境模块 × 生物家族的代表响应", "a")
    response_panel(axes[1], lr[0], lr[1], "Lake｜每个环境模块 × 生物家族的代表响应", "b")
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="y", labelright=True, labelleft=False, pad=4)
    sm = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-.55, vcenter=0, vmax=.55), cmap="RdBu_r")
    cax = fig.add_axes([.27, .045, .46, .018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("偏 Spearman ρ（控制经纬度二次项及调查次数）")
    handles = [
        Line2D([0], [0], marker="o", ms=6, color="none", mfc="#BBBBBB", mec="#202020",
               label="黑色外圈：该响应在13个range内 q ≤ 0.10"),
        Line2D([0], [0], marker="o", ms=6, color="none", mfc="#777777", mec="#202020",
               label="白色中心点：家族全部响应×尺度 q ≤ 0.05"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=2, frameon=False)
    fig.suptitle("Q4｜具体水生生物特征对三类环境因素的响应 range 与轨迹", y=1.005, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.91, bottom=.17, left=.16, right=.84, wspace=.18)
    pd.concat([dr[0].assign(panel="River"), lr[0].assign(panel="Lake")]).to_csv(
        out / "09_SourceData" / "Figure05_Q4_selected_response_scale_matrix.csv", index=False, encoding="utf-8-sig")
    save_all(fig, out, "Figure05_Q4_具体生物响应range")


def community_panel(ax, d: pd.DataFrame, water: str, letter: str) -> None:
    groups = [g for g in TAXON_ORDER if g in set(d.loc[d.water_body_type == water, "taxon_group"])]
    rows = [(m, g) for m in MODULES for g in groups]
    labels = [f"{MODULE_SHORT[m]}｜{TAXON_CN[g]}" for m, g in rows]
    add_scale_domains(ax, .78)
    vmax = max(.06, float(d.partial_rda_r2.quantile(.98)))
    cmap = mpl.colormaps["YlGnBu"]; norm = Normalize(0, vmax)
    for yi, (m, group) in enumerate(rows):
        q = d[(d.water_body_type == water) & (d.environment_module == m) & (d.taxon_group == group)]
        for _, r in q.iterrows():
            xi = RANGES.index(int(r.range_m))
            ax.scatter(xi, yi, s=24 + 900 * r.partial_rda_r2, color=cmap(norm(r.partial_rda_r2)),
                       edgecolor="#202020" if r.q_13ranges_within_group_module_rda <= .10 else "white",
                       lw=.9 if r.q_13ranges_within_group_module_rda <= .10 else .3)
            if r.q_13ranges_within_group_module_dispersion <= .10:
                ax.scatter(xi, yi - .28, marker="v", s=14, color="#C44E52", edgecolor="white", lw=.25, zorder=5)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks(range(13), XLABELS, rotation=45, ha="right"); ax.set_xlabel("空间范围（km）")
    ax.set_xlim(-.5, 12.5); ax.set_ylim(len(rows) - .45, -.65)
    ax.set_title(f"{water}｜Hellinger 偏 RDA", loc="left", fontweight="bold")
    for k in range(1, 3):
        ax.axhline(k * len(groups) - .5, color="#888888", lw=.7)
    panel(ax, letter)


def figure6(out: Path) -> None:
    d = pd.read_csv(out / "09_SourceData" / "Figure06_community_composition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 9.2), sharex=True)
    community_panel(axes[0], d, "River", "a")
    community_panel(axes[1], d, "Lake", "b")
    vmax = max(.06, float(d.partial_rda_r2.quantile(.98)))
    sm = mpl.cm.ScalarMappable(norm=Normalize(0, vmax), cmap="YlGnBu")
    cax = fig.add_axes([.27, .045, .46, .018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("环境模块解释的群落组成变异：partial RDA R²")
    handles = [
        Line2D([0], [0], marker="o", ms=6, color="none", mfc="#7FB7C9", mec="#202020",
               label="黑色外圈：组成检验在13个range内 q ≤ 0.10"),
        Line2D([0], [0], marker="v", ms=5, color="none", mfc="#C44E52", mec="white",
               label="红色三角：群落离散度在13个range内 q ≤ 0.10"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=2, frameon=False)
    fig.suptitle("群落组成与 β 过程｜环境梯度对应的群落中心与离散度变化", y=1.005, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.91, bottom=.17, left=.12, right=.98, wspace=.30)
    save_all(fig, out, "Figure06_群落组成与Beta过程")


def figure7(out: Path) -> None:
    sdir = out / "07_River_Lake时间空间敏感性"
    diff = pd.read_csv(sdir / "River_Lake标准化效应差_FisherZ检验.csv")
    temp = pd.read_csv(sdir / "Sample层年份水文期混合模型稳健性.csv")
    cv = pd.read_csv(sdir / "代表响应_空间CV汇总.csv")

    dtop = (diff.assign(absdiff=lambda x: x.rho_difference_river_minus_lake.abs())
            .sort_values(["q_within_module_all_common_responses_ranges", "absdiff"], ascending=[True, False])
            .head(12).sort_values("rho_difference_river_minus_lake"))
    dtop["label"] = dtop.apply(lambda r: f"{MODULE_SHORT[r.environment_module]}｜{short_label(r.variable_name_cn, 16)}｜{r.range_m/1000:g} km", axis=1)
    tmain = temp[temp.term == "environment_score"].copy()
    tmain = tmain.sort_values("q_value").head(12).sort_values("estimate")
    tmain["label"] = tmain.apply(lambda r: f"{r.water_body_type}·{MODULE_SHORT[r.environment_module]}｜{short_label(r.response_name_cn, 15)}", axis=1)

    fig = plt.figure(figsize=(12.2, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], wspace=.38, hspace=.38)
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, :])

    y = np.arange(len(dtop))
    for yi, (_, r) in enumerate(dtop.iterrows()):
        ax0.plot([r.river_rho, r.lake_rho], [yi, yi], color="#B8B8B8", lw=1.2)
    ax0.scatter(dtop.river_rho, y, color=WATER_COLOR["River"], s=27, label="River")
    ax0.scatter(dtop.lake_rho, y, color=WATER_COLOR["Lake"], marker="s", s=25, label="Lake")
    ax0.axvline(0, color="#777777", lw=.7, ls="--")
    ax0.set_yticks(y, dtop.label); ax0.set_xlabel("偏 Spearman ρ")
    ax0.set_title("River–Lake 响应效应对照（差异最突出12项）", loc="left", fontweight="bold")
    ax0.legend(loc="lower right"); panel(ax0, "a")

    y2 = np.arange(len(tmain))
    colors = [WATER_COLOR[w] for w in tmain.water_body_type]
    ax1.errorbar(tmain.estimate, y2, xerr=1.96 * tmain.se, fmt="none", ecolor="#999999", elinewidth=1, capsize=2)
    ax1.scatter(tmain.estimate, y2, c=colors, s=28, edgecolor=["#202020" if q <= .10 else "white" for q in tmain.q_value], lw=.9)
    ax1.axvline(0, color="#777777", lw=.7, ls="--")
    ax1.set_yticks(y2, tmain.label); ax1.set_xlabel("环境梯度固定效应（±95% CI）")
    ax1.set_title("重复调查的年份/水文期敏感性", loc="left", fontweight="bold")
    panel(ax1, "b")

    plot_cv = cv.copy()
    plot_cv["module_cn"] = plot_cv.environment_module.map(MODULE_SHORT)
    sns.boxplot(data=plot_cv, x="environment_module", y="delta_cv_r2_median", hue="water_body_type",
                order=MODULES, hue_order=WATERS, palette=WATER_COLOR, width=.55, fliersize=0, linewidth=.8, ax=ax2)
    sns.stripplot(data=plot_cv, x="environment_module", y="delta_cv_r2_median", hue="water_body_type",
                  order=MODULES, hue_order=WATERS, dodge=True, palette=WATER_COLOR, size=2.3, alpha=.45, ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], title="水体类型", ncol=2, loc="upper right")
    ax2.axhline(0, color="#777777", lw=.7, ls="--")
    ax2.set_xticks(range(3), [MODULE_CN[m] for m in MODULES]); ax2.set_xlabel("")
    ax2.set_ylabel("空间阻塞诊断 ΔR²\n（环境模块相对空间基线）")
    ax2.set_title("残余空间结构诊断（不作为预测或外推目标）", loc="left", fontweight="bold")
    panel(ax2, "c")
    fig.suptitle("River–Lake、时间重复调查与残余空间结构敏感性", y=.995, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=.93, bottom=.08, left=.19, right=.98)
    dtop.to_csv(out / "09_SourceData" / "Figure07a_River_Lake_effects.csv", index=False, encoding="utf-8-sig")
    tmain.to_csv(out / "09_SourceData" / "Figure07b_temporal_sensitivity.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(out / "09_SourceData" / "Figure07c_spatial_diagnostic.csv", index=False, encoding="utf-8-sig")
    save_all(fig, out, "Figure07_River_Lake时间空间敏感性")


def supplementary(out: Path) -> None:
    c = pd.read_csv(out / "05_Q4_具体生物响应" / "具体响应尺度类型与候选清单.csv")
    tab = c.groupby(["water_body_type", "environment_module", "scale_response_type"]).size().reset_index(name="n")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    types = list(tab.groupby("scale_response_type").n.sum().sort_values(ascending=False).index)
    for ax, water, letter in zip(axes, WATERS, ["a", "b"]):
        q = tab[tab.water_body_type == water]
        bottom = np.zeros(3)
        palette = sns.color_palette("Set2", n_colors=len(types))
        for typ, col in zip(types, palette):
            vals = q[q.scale_response_type == typ].set_index("environment_module").n.reindex(MODULES).fillna(0).to_numpy()
            ax.bar(range(3), vals, bottom=bottom, color=col, width=.68, label=typ)
            bottom += vals
        ax.set_xticks(range(3), [MODULE_CN[m] for m in MODULES])
        ax.set_title(water, loc="left", fontweight="bold"); ax.set_ylabel("具体响应变量数")
        panel(ax, letter)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(.99, .5), frameon=False)
    fig.suptitle("Supplementary Figure S1｜具体生物响应的尺度类型构成", y=1.02, fontweight="bold")
    fig.subplots_adjust(right=.76, wspace=.18)
    for folder, ext, dpi in [("PNG", "png", 400), ("TIFF", "tiff", 600), ("PDF", "pdf", 300), ("SVG", "svg", 300)]:
        fig.savefig(out / "12_补充材料" / f"FigureS1_具体响应尺度类型构成.{ext}", dpi=dpi,
                    bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)
    tab.to_csv(out / "09_SourceData" / "FigureS1_scale_type_counts.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    setup_style()
    out = Path(__file__).resolve().parents[1]
    for fn in [figure1, figure2, figure3, figure4, figure5, figure6, figure7, supplementary]:
        fn(out)
    generated = sorted(str(p.relative_to(out)) for p in (out / "08_论文级图件").rglob("*") if p.is_file())
    (out / "11_脚本与QA" / "图件生成清单.json").write_text(
        json.dumps({"n_main_files": len(generated), "files": generated}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"main_figure_files": len(generated), "main_figures": 7, "supplementary_figures": 1},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
