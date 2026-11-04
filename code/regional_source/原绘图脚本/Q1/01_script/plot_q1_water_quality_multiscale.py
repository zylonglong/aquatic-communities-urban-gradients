from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "02_source_data" / "Figure02_Q1_water_quality_scale.csv"
FIG_DIR = ROOT / "03_figure"
QA_DIR = ROOT / "04_QA"

WATER_COLORS = {"River": "#4F5DA9", "Lake": "#B04A78"}
WATER_MARKERS = {"River": "o", "Lake": "o"}
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_TITLES = {
    "terrain": "Topography and geomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
MODULE_COLORS = {
    "terrain": "#3F7F6D",
    "urbanization": "#A95D52",
    "human_activity": "#6D6293",
}
P_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "permutation_evidence_purples",
    mpl.colormaps["Purples"](np.linspace(0.35, 0.92, 256)),
)
SIGNIFICANCE_LEVEL = 0.05
SIZE_MIN = 12.0
SIZE_MAX = 60.0
P_MARKER_COLOR = "#D39B2A"
Q_MARKER_COLOR = "#4B1E5D"
RANGE_LABELS = ["0.2", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6"]


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.6,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "lines.linewidth": 1.3,
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
        "range_m",
        "n_spatial_units",
        "spearman_rho",
        "ci95_low",
        "ci95_high",
        "permutation_p",
        "fdr_q",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Missing values found in required plotting fields.")
    if data.duplicated(["water_body_type", "environment_module", "range_m"]).any():
        raise ValueError("Duplicate water-body x module x range rows found.")
    counts = data.groupby(["water_body_type", "environment_module"]).size()
    if not (counts == 13).all():
        raise ValueError(f"Each trajectory must contain 13 ranges; observed:\n{counts}")
    if not (
        (data["ci95_low"] <= data["spearman_rho"])
        & (data["spearman_rho"] <= data["ci95_high"])
    ).all():
        raise ValueError("At least one estimate lies outside its 95% CI.")
    if not data["permutation_p"].between(0, 1).all():
        raise ValueError("Permutation P values fall outside [0, 1].")
    if not data["fdr_q"].between(0, 1).all():
        raise ValueError("FDR q values fall outside [0, 1].")
    return data


def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    ax.tick_params(colors="#3F3F3F")


def evidence_strength(values):
    values = np.asarray(values, dtype=float)
    return -np.log10(np.clip(values, np.finfo(float).tiny, 1.0))


def bubble_sizes(q_values, q_strength_max):
    strength = evidence_strength(q_values)
    normalized = np.clip(strength / q_strength_max, 0, 1)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * normalized ** 0.78


def add_effect_legend(ax):
    ci_handle = (
        Patch(facecolor=WATER_COLORS["River"], edgecolor="none", alpha=0.18),
        Patch(facecolor=WATER_COLORS["Lake"], edgecolor="none", alpha=0.18),
    )
    effect_handles = [
        Line2D(
            [0], [0], color=WATER_COLORS["River"], marker=WATER_MARKERS["River"],
            markersize=4.6, markerfacecolor="white", markeredgewidth=0.9,
            linewidth=1.45, label="River (n = 19)"
        ),
        Line2D(
            [0], [0], color=WATER_COLORS["Lake"], marker=WATER_MARKERS["Lake"],
            markersize=4.6, markerfacecolor="white", markeredgewidth=0.9,
            linewidth=1.45, label="Lake (n = 35)"
        ),
        Line2D(
            [0], [0], color=P_MARKER_COLOR, marker="o", markersize=5.1,
            markerfacecolor="none", markeredgecolor=P_MARKER_COLOR,
            markeredgewidth=1.2, linewidth=0, label="Permutation P ≤ 0.05"
        ),
        Line2D(
            [0], [0], color=Q_MARKER_COLOR, marker="o", markersize=3.8,
            markerfacecolor=Q_MARKER_COLOR, markeredgecolor="white",
            markeredgewidth=0.35, linewidth=0, label="FDR q ≤ 0.05"
        ),
        ci_handle,
    ]
    effect_labels = [
        "River (n = 19)", "Lake (n = 35)", "Permutation P ≤ 0.05",
        "FDR q ≤ 0.05", "95% CI",
    ]
    ax.legend(
        handles=effect_handles, labels=effect_labels,
        loc="upper left", bbox_to_anchor=(0.012, 0.985), ncol=2,
        fontsize=6.25, handlelength=1.35, columnspacing=0.65,
        handletextpad=0.35, labelspacing=0.34, borderaxespad=0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    )


def add_bubble_key(ax, p_strength_max, q_strength_max, p_min):
    ax.set_axis_off()
    q_examples = [0.80, 0.20, 0.05]
    q_example_sizes = bubble_sizes(q_examples, q_strength_max)
    ax.text(
        0.02, 0.49, "FDR q — bubble size (smaller = larger)",
        transform=ax.transAxes, ha="left", va="center", fontsize=7.0,
    )
    example_x = [0.31, 0.41, 0.51]
    ax.scatter(
        example_x, [0.49] * 3, s=q_example_sizes,
        transform=ax.transAxes, facecolor=P_CMAP(0.55),
        edgecolor="none", linewidth=0, clip_on=False,
    )
    for x, q in zip(example_x, q_examples):
        ax.text(
            x + 0.014, 0.49, f"q = {q:.2f}", transform=ax.transAxes,
            ha="left", va="center", fontsize=6.8,
        )

    color_ax = ax.inset_axes([0.64, 0.37, 0.30, 0.24])
    norm = mpl.colors.Normalize(vmin=0, vmax=p_strength_max)
    colorbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=P_CMAP),
        cax=color_ax, orientation="horizontal",
    )
    threshold_strength = evidence_strength([SIGNIFICANCE_LEVEL])[0]
    ticks = [0, threshold_strength, p_strength_max]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(["1", "0.05", f"{p_min:.4f}".lstrip("0")])
    colorbar.ax.tick_params(labelsize=6.5, length=1.8, pad=1.4, width=0.45)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label("Permutation P  (darker = smaller)", fontsize=7.0, labelpad=1.5)
    colorbar.ax.xaxis.set_label_position("top")


def add_gradient_connector(ax, x, y, rgba, bubble_areas, steps=18):
    segments = []
    colours = []
    widths = []
    diameters = np.sqrt(np.asarray(bubble_areas, dtype=float))
    for index in range(len(x) - 1):
        x_steps = np.linspace(x[index], x[index + 1], steps + 1)
        c0 = np.asarray(rgba[index])
        c1 = np.asarray(rgba[index + 1])
        width0 = diameters[index] / 3.0
        width1 = diameters[index + 1] / 3.0
        for step in range(steps):
            segments.append(
                [(x_steps[step], y), (x_steps[step + 1], y)]
            )
            fraction = (step + 0.5) / steps
            colours.append(c0 * (1 - fraction) + c1 * fraction)
            widths.append(width0 * (1 - fraction) + width1 * fraction)
    collection = LineCollection(
        segments, colors=colours, linewidths=widths,
        capstyle="round", zorder=1,
    )
    ax.add_collection(collection)


def plot_bubble_panel(
    ax, module, data, p_strength_max, q_strength_max, show_row_labels
):
    norm = mpl.colors.Normalize(vmin=0, vmax=p_strength_max)
    y_positions = {"River": 0.43, "Lake": 0.24}

    for water in ["River", "Lake"]:
        subset = data[
            (data["environment_module"] == module)
            & (data["water_body_type"] == water)
        ].sort_values("range_m")
        x = np.arange(len(subset), dtype=float)
        y = np.full(len(subset), y_positions[water], dtype=float)
        p_strength = evidence_strength(subset["permutation_p"].to_numpy())
        sizes = bubble_sizes(subset["fdr_q"].to_numpy(), q_strength_max)
        rgba = P_CMAP(norm(p_strength))

        add_gradient_connector(ax, x, y_positions[water], rgba, sizes)
        ax.scatter(
            x, y, s=sizes, c=p_strength, cmap=P_CMAP, norm=norm,
            edgecolors="none", linewidths=0, zorder=3, clip_on=False,
        )

    ax.set_xlim(-0.68, len(RANGE_LABELS) - 0.32)
    ax.set_ylim(0.10, 0.58)
    ax.set_xticks(np.arange(len(RANGE_LABELS)))
    ax.set_xticklabels(
        RANGE_LABELS, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", length=2.6, pad=2.5, width=0.55)
    ax.set_xlabel("Spatial range (km)", labelpad=5)
    if show_row_labels:
        ax.set_yticks([0.43, 0.24])
        ax.set_yticklabels(["River", "Lake"])
        ax.tick_params(axis="y", length=0, pad=5)
        for tick, water in zip(ax.get_yticklabels(), ["River", "Lake"]):
            tick.set_color(WATER_COLORS[water])
            tick.set_fontweight("semibold")
    else:
        ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#4A4A4A")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_facecolor("white")


def plot_effect_panel(ax, module, data):
    ax.axhline(0, color="#777777", linewidth=0.75,
               linestyle=(0, (2.2, 2.2)), zorder=1)
    for water in ["River", "Lake"]:
        subset = data[
            (data["environment_module"] == module)
            & (data["water_body_type"] == water)
        ].sort_values("range_m")
        x = np.arange(len(subset), dtype=float)
        y = subset["spearman_rho"].to_numpy()
        low = subset["ci95_low"].to_numpy()
        high = subset["ci95_high"].to_numpy()
        color = WATER_COLORS[water]
        marker = WATER_MARKERS[water]

        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(
            x, y, color=color, marker=marker, markersize=3.05,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=0.75, linewidth=1.3, zorder=3,
        )

        p_sig = subset["permutation_p"].to_numpy() <= SIGNIFICANCE_LEVEL
        q_sig = subset["fdr_q"].to_numpy() <= SIGNIFICANCE_LEVEL
        if p_sig.any():
            ax.scatter(
                x[p_sig], y[p_sig], s=38, marker="o", facecolors="none",
                edgecolors=P_MARKER_COLOR, linewidths=1.25, zorder=5,
            )
        if q_sig.any():
            ax.scatter(
                x[q_sig], y[q_sig], s=22, marker="o",
                facecolors=color, edgecolors="white",
                linewidths=0.45, zorder=6,
            )

    ax.set_xlim(-0.68, len(RANGE_LABELS) - 0.32)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(len(RANGE_LABELS)))
    ax.set_xticklabels(
        RANGE_LABELS, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", length=2.8, pad=2.5, width=0.6)
    ax.set_xlabel("Spatial range (km)", labelpad=5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title(
        MODULE_TITLES[module], pad=7, fontsize=8.2,
        color="#262626", fontweight="semibold",
    )
    style_axis(ax)


def save_outputs(fig):
    base = "Figure_Q1_water_quality_multiscale_redesign"
    outputs = {
        "PNG": (FIG_DIR / "PNG" / f"{base}.png", {"dpi": 600}),
        "SVG": (FIG_DIR / "SVG" / f"{base}.svg", {}),
        "PDF": (FIG_DIR / "PDF" / f"{base}.pdf", {}),
        "TIFF": (
            FIG_DIR / "TIFF" / f"{base}.tiff",
            {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        ),
    }
    for _, (path, kwargs) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **kwargs)
    return outputs


def write_qa(data, outputs, p_strength_max, q_strength_max):
    QA_DIR.mkdir(parents=True, exist_ok=True)
    evidence_values = data[
        [
            "environment_module", "water_body_type", "range_m",
            "spearman_rho", "ci95_low", "ci95_high", "permutation_p", "fdr_q",
        ]
    ].copy()
    evidence_values["neg_log10_permutation_p"] = evidence_strength(
        evidence_values["permutation_p"]
    )
    evidence_values["neg_log10_fdr_q"] = evidence_strength(evidence_values["fdr_q"])
    evidence_values["bubble_area_points_squared"] = bubble_sizes(
        evidence_values["fdr_q"], q_strength_max
    )
    evidence_values["categorical_range_position"] = (
        evidence_values.sort_values("range_m")
        .groupby(["environment_module", "water_body_type"])
        .cumcount()
        .sort_index()
    )
    evidence_values.to_csv(QA_DIR / "all_range_values_visualized.csv", index=False)

    trajectory_summary = (
        data.assign(
            fdr_supported=data["fdr_q"] <= SIGNIFICANCE_LEVEL,
            permutation_supported=data["permutation_p"] <= SIGNIFICANCE_LEVEL,
            positive=data["spearman_rho"] > 0,
        )
        .groupby(["environment_module", "water_body_type"], as_index=False)
        .agg(
            n_ranges=("range_m", "size"),
            positive_ranges=("positive", "sum"),
            permutation_supported_ranges=("permutation_supported", "sum"),
            fdr_supported_ranges=("fdr_supported", "sum"),
        )
    )
    trajectory_summary.to_csv(QA_DIR / "trajectory_evidence_summary.csv", index=False)

    lines = [
        "DATA QA: PASS",
        f"Source rows: {len(data)}",
        "Expected structure: 2 water bodies x 3 modules x 13 ranges = 78 rows",
        "Missing required values: 0",
        "Duplicate water-body x module x range rows: 0",
        "All estimates lie within their reported 95% CIs: yes",
        "All P and q values lie within [0, 1]: yes",
        f"Permutation-P colour scale maximum: {p_strength_max:.4f} on -log10(P) scale",
        f"FDR-q size scale maximum: {q_strength_max:.4f} on -log10(q) scale",
        "Bubble size direction: smaller q = larger bubble",
        f"Bubble area limits: {SIZE_MIN:.0f} to {SIZE_MAX:.0f} points squared",
        "Bubble colour direction: smaller permutation P = darker fill",
        "Range axis: 13 equally spaced categorical groups",
        f"Rows with 0.05 < permutation P <= 0.10: {((data['permutation_p'] > 0.05) & (data['permutation_p'] <= 0.10)).sum()}",
        f"Rows with 0.05 < FDR q <= 0.10: {((data['fdr_q'] > 0.05) & (data['fdr_q'] <= 0.10)).sum()}",
        "A 0.10 threshold is not marked as confirmatory because it was not prespecified.",
        "FDR q <= 0.05 marker: filled circle in waterbody colour",
        "Range labels and axis title: shown below both Spearman and bubble panels",
        "Cross-panel correspondence lines: removed",
        "Bubble and connector outlines: removed",
        "Permutation-P colormap: truncated Purples (0.35-0.92) to avoid near-white bubbles",
        "Connector width: smooth interpolation between one-third of adjacent bubble diameters",
        "Bubble-row spacing: compressed",
        "Spearman y labels: numeric labels shown only in panel a; axis title simplified to Spearman rho",
        "Panel hierarchy: three Spearman plots grouped as panel a; three bubble strips grouped as panel b",
        "Panel subtitles: Scale-dependent associations; Statistical evidence across scales",
        "Legend layout: Spearman legend inside panel a; bubble legend below all panels",
        "Plot backgrounds: white; no scale-domain fills",
        "Panel d and the peak-statistics table: removed",
        "",
        "EXPORTS",
    ]
    for kind, (path, _) in outputs.items():
        lines.append(
            f"{kind}: {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KiB)"
        )

    png_path = outputs["PNG"][0]
    tiff_path = outputs["TIFF"][0]
    svg_path = outputs["SVG"][0]
    with Image.open(png_path) as image:
        lines.append(f"PNG dimensions: {image.size[0]} x {image.size[1]} px")
        lines.append(f"PNG resolution metadata: {image.info.get('dpi')}")
    with Image.open(tiff_path) as image:
        lines.append(f"TIFF dimensions: {image.size[0]} x {image.size[1]} px")
        lines.append(f"TIFF compression: {image.info.get('compression')}")
        lines.append(f"TIFF resolution metadata: {image.info.get('dpi')}")
    svg_text = svg_path.read_text(encoding="utf-8")
    live_text_count = len(re.findall(r"<text\b", svg_text))
    lines.append(f"SVG live-text elements: {live_text_count}")
    lines.append(f"SVG editable text check: {'PASS' if live_text_count > 0 else 'FAIL'}")
    (QA_DIR / "data_and_export_QA.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_style()
    data = load_and_validate()
    p_strength_max = evidence_strength(data["permutation_p"]).max()
    q_strength_max = evidence_strength(data["fdr_q"]).max()
    p_min = data["permutation_p"].min()

    fig = plt.figure(figsize=(7.20, 5.70))
    outer = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[3.85, 0.62],
        left=0.105, right=0.985, top=0.900, bottom=0.055, hspace=0.24,
    )

    columns = outer[0].subgridspec(1, 3, wspace=0.17)
    effect_axes = []
    bubble_axes = []
    for i, module in enumerate(MODULES):
        column = columns[0, i].subgridspec(
            2, 1, height_ratios=[1.55, 0.54], hspace=0.46,
        )
        effect_ax = fig.add_subplot(
            column[0, 0], sharey=effect_axes[0] if effect_axes else None,
        )
        bubble_ax = fig.add_subplot(column[1, 0])
        effect_axes.append(effect_ax)
        bubble_axes.append(bubble_ax)

        plot_effect_panel(effect_ax, module, data)
        plot_bubble_panel(
            bubble_ax, module, data, p_strength_max, q_strength_max,
            show_row_labels=(i == 0),
        )

        if i == 0:
            effect_ax.set_ylabel("Spearman ρ", labelpad=7)
            add_effect_legend(effect_ax)
        effect_ax.tick_params(axis="y", labelleft=(i == 0))
        effect_ax.spines["left"].set_visible(True)

    group_x = 0.105
    fig.text(
        group_x, 0.965, "a", ha="left", va="center",
        fontsize=8.8, fontweight="bold", color="#111111",
    )
    fig.text(
        group_x + 0.017, 0.965,
        "Scale-dependent associations",
        ha="left", va="center", fontsize=8.2,
        fontweight="semibold", color="#333333",
    )
    bubble_heading_y = bubble_axes[0].get_position().y1 + 0.010
    fig.text(
        group_x, bubble_heading_y, "b", ha="left", va="center",
        fontsize=8.8, fontweight="bold", color="#111111", zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.1},
    )
    fig.text(
        group_x + 0.017, bubble_heading_y,
        "Statistical evidence across scales",
        ha="left", va="center", fontsize=8.2,
        fontweight="semibold", color="#333333", zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.1},
    )

    bubble_key_ax = fig.add_subplot(outer[1])
    add_bubble_key(bubble_key_ax, p_strength_max, q_strength_max, p_min)

    outputs = save_outputs(fig)
    write_qa(data, outputs, p_strength_max, q_strength_max)
    plt.close(fig)
    print("Bubble-evidence figure and QA outputs written successfully.")
    for kind, (path, _) in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
