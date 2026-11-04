from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "02_source_data" / "Figure03_Q2_biology_family_scale.csv"
FIG_DIR = ROOT / "03_figure"
QA_DIR = ROOT / "04_QA"

MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_TITLES = {
    "terrain": "Topography and geomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
FAMILY_LABELS = {
    "community_baseline": "community-level attributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "community composition",
    "functional_indicator": "indicator/tolerance taxa",
}
FAMILY_PLOT_LABELS = {
    "community_baseline": "community-level\nattributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "community\ncomposition",
    "functional_indicator": "indicator/tolerance\ntaxa",
}
FAMILY_LABELS_CN = {
    "community_baseline": "群落整体规模和状态",
    "alpha_diversity": "局域多样性结构",
    "beta_diversity": "空间周转结构",
    "community_composition": "分类组成结构",
    "functional_indicator": "生态功能/环境响应结构",
}
WATERS = ["River", "Lake"]
WATER_COLORS = {"River": "#4F5DA9", "Lake": "#B04A78"}
RANGES_M = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
RANGE_LABELS = ["0.2", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6"]
MAXT_THRESHOLD = 0.05

COOLWARM = mpl.colormaps["coolwarm"]
SIGNED_LIMIT = 0.25
SIGNED_NORM = mpl.colors.TwoSlopeNorm(vmin=-SIGNED_LIMIT, vcenter=0.0, vmax=SIGNED_LIMIT)
SIZE_MIN = 4.0
SIZE_MAX = 68.0
RMS_MIN = 0.06
RMS_MAX = 0.31

Y_POSITIONS = {
    **{("River", family): 10.0 - i for i, family in enumerate(FAMILIES)},
    **{("Lake", family): 4.25 - i for i, family in enumerate(FAMILIES)},
}


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.6,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.8,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_and_validate():
    data = pd.read_csv(DATA_FILE)
    required = {
        "water_body_type",
        "environment_module",
        "response_family",
        "range_m",
        "rms_partial_rho",
        "median_partial_rho",
        "range_maxT_p",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Missing values found in required plotting fields.")
    key = ["water_body_type", "environment_module", "response_family", "range_m"]
    if data.duplicated(key).any():
        raise ValueError("Duplicate water x module x family x range rows found.")
    expected_rows = len(WATERS) * len(MODULES) * len(FAMILIES) * len(RANGES_M)
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(data)}.")
    counts = data.groupby(["water_body_type", "environment_module", "response_family"]).size()
    if not (counts == len(RANGES_M)).all():
        raise ValueError(f"Every trajectory must contain 13 ranges; observed:\n{counts}")
    if set(data["water_body_type"]) != set(WATERS):
        raise ValueError("Unexpected water-body labels.")
    if set(data["environment_module"]) != set(MODULES):
        raise ValueError("Unexpected environmental-module labels.")
    if set(data["response_family"]) != set(FAMILIES):
        raise ValueError("Unexpected response-family labels.")
    if set(data["range_m"].astype(int)) != set(RANGES_M):
        raise ValueError("Unexpected spatial ranges.")
    if not data["rms_partial_rho"].between(0, 1).all():
        raise ValueError("RMS partial rho falls outside [0, 1].")
    if not data["median_partial_rho"].between(-1, 1).all():
        raise ValueError("Signed median partial rho falls outside [-1, 1].")
    if not data["range_maxT_p"].between(0, 1).all():
        raise ValueError("Range-wise maxT P falls outside [0, 1].")
    return data


def bubble_sizes(rms_values):
    values = np.asarray(rms_values, dtype=float)
    scaled = np.clip((values - RMS_MIN) / (RMS_MAX - RMS_MIN), 0, 1)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * scaled ** 1.30


def add_gradient_connector(ax, x, y, rgba, areas, steps=18):
    """Interpolate colour and line width between adjacent bubble centres.

    At each endpoint the line width is exactly one-third of the corresponding
    scatter-circle diameter, where marker area is expressed in points squared.
    """
    segments, colours, widths = [], [], []
    diameters = 2.0 * np.sqrt(np.asarray(areas, dtype=float) / np.pi)
    endpoint_widths = diameters / 3.0
    for i in range(len(x) - 1):
        xx = np.linspace(x[i], x[i + 1], steps + 1)
        c0, c1 = np.asarray(rgba[i]).copy(), np.asarray(rgba[i + 1]).copy()
        for j in range(steps):
            frac = (j + 0.5) / steps
            segments.append([(xx[j], y), (xx[j + 1], y)])
            colour = c0 * (1 - frac) + c1 * frac
            colour[3] = 0.72
            colours.append(colour)
            widths.append(endpoint_widths[i] * (1 - frac) + endpoint_widths[i + 1] * frac)
    ax.add_collection(
        LineCollection(
            segments,
            colors=colours,
            linewidths=widths,
            capstyle="round",
            zorder=2.5,
        )
    )


def plot_module(ax, data, module, show_left_labels):
    subset_module = data[data["environment_module"].eq(module)]
    ax.axhline(5.15, color="#BFC6CE", linewidth=0.65, zorder=0)

    for water in WATERS:
        for family in FAMILIES:
            q = subset_module[
                subset_module["water_body_type"].eq(water)
                & subset_module["response_family"].eq(family)
            ].set_index("range_m").reindex(RANGES_M)
            x = np.arange(len(RANGES_M), dtype=float)
            y = Y_POSITIONS[(water, family)]
            rms = q["rms_partial_rho"].to_numpy(dtype=float)
            signed = q["median_partial_rho"].to_numpy(dtype=float)
            max_t = q["range_maxT_p"].to_numpy(dtype=float)
            sizes = bubble_sizes(rms)
            rgba = COOLWARM(SIGNED_NORM(signed))

            supported = max_t <= MAXT_THRESHOLD
            add_gradient_connector(ax, x, y, rgba, sizes)
            if supported.any():
                ax.scatter(
                    x[supported],
                    np.full(supported.sum(), y),
                    s=sizes[supported] + 10.0,
                    marker="o",
                    facecolors="#111111",
                    edgecolors="none",
                    linewidths=0,
                    zorder=3.4,
                )
            ax.scatter(
                x,
                np.full_like(x, y),
                s=sizes,
                c=signed,
                cmap=COOLWARM,
                norm=SIGNED_NORM,
                edgecolors="none",
                linewidths=0,
                alpha=0.98,
                zorder=3.6,
                clip_on=False,
            )

    ax.set_xlim(-0.62, 12.62)
    ax.set_ylim(-0.35, 10.45)
    ax.set_xticks(np.arange(len(RANGE_LABELS)))
    ax.set_xticklabels(RANGE_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=2.3, pad=2.0, colors="#3D434A")
    ax.set_xlabel("Spatial range (km)", labelpad=3.5)
    ax.set_title(MODULE_TITLES[module], pad=4.0, fontsize=6.7, fontweight="semibold", color="#252A30")
    if show_left_labels:
        yticks = [Y_POSITIONS[(water, family)] for water in WATERS for family in FAMILIES]
        ylabels = [FAMILY_PLOT_LABELS[family] for water in WATERS for family in FAMILIES]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="y", length=0, pad=5.5, colors="#202428", labelright=False, labelleft=True)
    else:
        ax.tick_params(axis="y", length=0, labelleft=False)

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#4A4F55")
    ax.spines["bottom"].set_linewidth(0.65)
    ax.set_facecolor("white")


def add_water_group_labels(fig, right_ax):
    bbox = right_ax.get_position()
    y_min, y_max = right_ax.get_ylim()

    def fig_y(y):
        return bbox.y0 + (y - y_min) / (y_max - y_min) * bbox.height

    x = bbox.x1 + 0.028
    fig.text(
        x,
        fig_y(np.mean([Y_POSITIONS[("River", f)] for f in FAMILIES])),
        "River",
        rotation=90,
        ha="center",
        va="center",
        color="#111111",
        fontsize=7.2,
        fontweight="bold",
    )
    fig.text(
        x,
        fig_y(np.mean([Y_POSITIONS[("Lake", f)] for f in FAMILIES])),
        "Lake",
        rotation=90,
        ha="center",
        va="center",
        color="#111111",
        fontsize=7.2,
        fontweight="bold",
    )


def compute_integrated_rms(data):
    integrated = (
        data.groupby(["water_body_type", "environment_module", "range_m"], as_index=False)
        .agg(
            n_response_domains=("response_family", "nunique"),
            integrated_rms=("rms_partial_rho", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            domain_min_rms=("rms_partial_rho", "min"),
            domain_max_rms=("rms_partial_rho", "max"),
        )
    )
    if not integrated["n_response_domains"].eq(5).all():
        raise ValueError("Panel b requires exactly five response domains in every water x module x range unit.")
    return integrated


def plot_integrated_trend(ax, integrated, module, show_ylabels):
    x = np.arange(len(RANGES_M), dtype=float)
    x_smooth = np.linspace(x.min(), x.max(), 500)
    for water in WATERS:
        q = (
            integrated[
                integrated["water_body_type"].eq(water)
                & integrated["environment_module"].eq(module)
            ]
            .set_index("range_m")
            .reindex(RANGES_M)
        )
        y = q["integrated_rms"].to_numpy(dtype=float)
        lower = q["domain_min_rms"].to_numpy(dtype=float)
        upper = q["domain_max_rms"].to_numpy(dtype=float)
        y_smooth = PchipInterpolator(x, y)(x_smooth)
        lower_smooth = PchipInterpolator(x, lower)(x_smooth)
        upper_smooth = PchipInterpolator(x, upper)(x_smooth)
        lower_smooth = np.minimum(lower_smooth, y_smooth)
        upper_smooth = np.maximum(upper_smooth, y_smooth)
        ax.fill_between(
            x_smooth,
            lower_smooth,
            upper_smooth,
            color=WATER_COLORS[water],
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_smooth,
            y_smooth,
            color=WATER_COLORS[water],
            linewidth=1.25,
            zorder=3,
        )
        ax.plot(
            x,
            y,
            linestyle="none",
            color=WATER_COLORS[water],
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgecolor=WATER_COLORS[water],
            markeredgewidth=0.65,
            zorder=4,
        )
    ax.set_xlim(-0.35, 12.35)
    ax.set_ylim(0.045, 0.32)
    ax.set_xticks(x)
    ax.set_xticklabels(RANGE_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Spatial range (km)", labelpad=3.5)
    ax.set_title(MODULE_TITLES[module], pad=4.5, fontsize=7.2, fontweight="semibold", color="#252A30")
    ax.tick_params(axis="both", colors="#3D434A")
    if show_ylabels:
        ax.set_ylabel("Overall RMS strength", labelpad=5.5)
        handles = [
            Line2D([0], [0], color=WATER_COLORS[water], marker="o", markersize=3.5,
                   markerfacecolor="white", markeredgewidth=0.65, linewidth=1.25, label=water)
            for water in WATERS
        ]
        handles.append(
            Patch(
                facecolor="#9AA0A6",
                edgecolor="none",
                alpha=0.24,
                label="Response-domain min–max",
            )
        )
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=6.1,
            handlelength=1.45,
            handletextpad=0.35,
            labelspacing=0.25,
            borderaxespad=0.25,
        )
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4F55")
    ax.spines["bottom"].set_color("#4A4F55")


def add_bubble_legends(legend_ax):
    legend_ax.set_axis_off()
    examples = [0.10, 0.20, 0.30]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=float(np.sqrt(bubble_sizes([value])[0])),
            markerfacecolor="#7E8791",
            markeredgecolor="none",
            label=f"{value:.2f}",
        )
        for value in examples
    ]
    size_legend = legend_ax.legend(
        handles=size_handles,
        title="Response strength (RMS partial ρ)",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.84),
        ncol=3,
        fontsize=6.2,
        title_fontsize=6.6,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
        labelspacing=0.35,
    )
    size_legend._legend_box.align = "left"
    legend_ax.add_artist(size_legend)

    support_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="#111111",
        markeredgewidth=0.9,
        label="Range-wise maxT P ≤ 0.05",
    )
    legend_ax.legend(
        handles=[support_handle],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.93),
        fontsize=6.2,
        handletextpad=0.35,
    )

    cax = legend_ax.inset_axes([0.79, 0.59, 0.18, 0.20])
    cb = legend_ax.figure.colorbar(
        mpl.cm.ScalarMappable(norm=SIGNED_NORM, cmap=COOLWARM),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_ticks([-0.2, 0.0, 0.2])
    cb.ax.tick_params(labelsize=6.2, length=1.8, width=0.45, pad=1.4)
    cb.outline.set_linewidth(0.45)
    cb.set_label("Signed median partial Spearman ρ", fontsize=6.7, labelpad=1.8)
    legend_ax.text(0.775, 0.69, "Negative", transform=legend_ax.transAxes, ha="right", va="center", fontsize=6.0, color=COOLWARM(0.05))
    legend_ax.text(0.985, 0.69, "Positive", transform=legend_ax.transAxes, ha="left", va="center", fontsize=6.0, color=COOLWARM(0.95))


def add_waterbody_legend(legend_ax):
    legend_ax.set_axis_off()
    legend_ax.text(0.5, 0.82, "Spatial range (km)", transform=legend_ax.transAxes,
                   ha="center", va="center", fontsize=7.4)
    handles = [
        Line2D([0], [0], color=WATER_COLORS[water], marker="o", markersize=3.8,
               markerfacecolor="white", markeredgewidth=0.7, linewidth=1.25, label=water)
        for water in WATERS
    ]
    legend_ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=6.5,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.5,
    )


def save_outputs(fig):
    stem = "Figure_Q2_aquatic_community_response_domains_bubbles"
    outputs = {
        "PNG": (FIG_DIR / "PNG" / f"{stem}.png", {"dpi": 600}),
        "SVG": (FIG_DIR / "SVG" / f"{stem}.svg", {}),
        "PDF": (FIG_DIR / "PDF" / f"{stem}.pdf", {}),
        "TIFF": (FIG_DIR / "TIFF" / f"{stem}.tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    }
    for _, (output, kwargs) in outputs.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.025, **kwargs)
    return outputs


def write_qa(data, integrated, outputs):
    QA_DIR.mkdir(parents=True, exist_ok=True)
    visualized = data[
        [
            "water_body_type",
            "environment_module",
            "response_family",
            "range_m",
            "rms_partial_rho",
            "median_partial_rho",
            "range_maxT_p",
        ]
    ].copy()
    visualized["figure_response_label"] = visualized["response_family"].map(FAMILY_LABELS)
    visualized["figure_response_label_cn"] = visualized["response_family"].map(FAMILY_LABELS_CN)
    visualized["bubble_area_points_squared"] = bubble_sizes(visualized["rms_partial_rho"])
    visualized["maxT_black_ring"] = visualized["range_maxT_p"] <= MAXT_THRESHOLD
    visualized.to_csv(QA_DIR / "all_values_visualized.csv", index=False, encoding="utf-8-sig")
    integrated.to_csv(QA_DIR / "integrated_rms_by_range.csv", index=False, encoding="utf-8-sig")

    summary = (
        visualized.groupby(["water_body_type", "environment_module", "response_family"], as_index=False)
        .agg(
            n_ranges=("range_m", "size"),
            max_rms=("rms_partial_rho", "max"),
            min_signed_median=("median_partial_rho", "min"),
            max_signed_median=("median_partial_rho", "max"),
            maxT_supported_ranges=("maxT_black_ring", "sum"),
        )
    )
    summary.to_csv(QA_DIR / "trajectory_and_support_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "DATA QA: PASS",
        f"Source rows: {len(data)}",
        "Expected panel-a structure: 3 modules x 2 water bodies x 5 response domains x 13 ranges = 390 rows",
        "Duplicate key rows: 0",
        "Missing required plotting values: 0",
        f"Range-wise maxT P <= 0.05 black rings: {int((data['range_maxT_p'] <= MAXT_THRESHOLD).sum())}",
        f"RMS range: {data['rms_partial_rho'].min():.6f} to {data['rms_partial_rho'].max():.6f}",
        f"Signed median range: {data['median_partial_rho'].min():.6f} to {data['median_partial_rho'].max():.6f}",
        f"Bubble area range: {SIZE_MIN:.1f} to {SIZE_MAX:.1f} points squared",
        "Bubble and connector colour: classic coolwarm mapped to signed median partial Spearman rho",
        "Connector width: smooth interpolation between one-third of adjacent bubble diameters",
        "Significance overlay: continuous black outer ring for range-wise maxT P <= 0.05; the ring and bubble fill are drawn above connectors",
        "Panel a plot background: white with no alternating row fills",
        "Panel b integrated RMS formula: sqrt(mean(five family-level RMS values squared)) within each water x module x range",
        "Panel b display: three aligned environmental-module trend panels comparing River and Lake directly",
        "Panel b envelope: minimum-to-maximum family-level RMS across the five heterogeneous response domains; descriptive range, not a confidence interval",
        "Panel b smoothing: shape-preserving PCHIP through the 13 calculated nodes for the centre line and min-max envelope; original nodes retained as markers",
        "Panel b grid lines: removed",
        "Backend: Python/matplotlib only",
        "SVG/PDF text export: editable text retained",
        "Raster export: PNG and LZW-compressed TIFF at 600 dpi",
        "",
        "EXPORTS",
    ]
    for kind, (output, _) in outputs.items():
        lines.append(f"{kind}: {output.relative_to(ROOT)} ({output.stat().st_size / 1024:.1f} KiB)")
    with Image.open(outputs["PNG"][0]) as image:
        lines.append(f"PNG dimensions: {image.size[0]} x {image.size[1]} px")
        lines.append(f"PNG resolution metadata: {image.info.get('dpi')}")
    with Image.open(outputs["TIFF"][0]) as image:
        lines.append(f"TIFF dimensions: {image.size[0]} x {image.size[1]} px")
        lines.append(f"TIFF compression: {image.info.get('compression')}")
        lines.append(f"TIFF resolution metadata: {image.info.get('dpi')}")
    svg_text = outputs["SVG"][0].read_text(encoding="utf-8")
    live_text = len(re.findall(r"<text\b", svg_text))
    lines.append(f"SVG live-text elements: {live_text}")
    lines.append(f"SVG editable text check: {'PASS' if live_text > 0 else 'FAIL'}")
    (QA_DIR / "data_and_export_QA.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_style()
    data = load_and_validate()
    integrated = compute_integrated_rms(data)

    fig = plt.figure(figsize=(7.20, 6.05))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.05, 1.00],
        left=0.22,
        right=0.94,
        top=0.93,
        bottom=0.09,
        hspace=0.16,
    )
    a_block = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.18], hspace=0.30)
    top = a_block[0].subgridspec(1, 3, wspace=0.18)
    axes_a = []
    for i, module in enumerate(MODULES):
        ax = fig.add_subplot(top[0, i])
        axes_a.append(ax)
        plot_module(ax, data, module, show_left_labels=(i == 0))

    a_legend_ax = fig.add_subplot(a_block[1])
    add_bubble_legends(a_legend_ax)

    b_grid = outer[1].subgridspec(1, 3, wspace=0.18)
    axes_b = []
    for i, module in enumerate(MODULES):
        ax = fig.add_subplot(b_grid[0, i], sharey=axes_b[0] if axes_b else None)
        axes_b.append(ax)
        plot_integrated_trend(ax, integrated, module, show_ylabels=(i == 0))

    add_water_group_labels(fig, axes_a[-1])

    fig.text(0.22, 0.973, "a", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#111111")
    fig.text(0.239, 0.973, "Scale-dependent response domains", ha="left", va="center", fontsize=8.0, fontweight="semibold", color="#30353B")
    b_y = axes_b[0].get_position().y1 + 0.048
    fig.text(0.22, b_y, "b", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#111111")
    fig.text(0.239, b_y, "Integrated response strength across domains", ha="left", va="center", fontsize=8.0, fontweight="semibold", color="#30353B")

    outputs = save_outputs(fig)
    write_qa(data, integrated, outputs)
    plt.close(fig)
    print("Q2 composite bubble and integrated-trend figure written successfully.")
    for kind, (output, _) in outputs.items():
        print(f"{kind}: {output}")


if __name__ == "__main__":
    main()
