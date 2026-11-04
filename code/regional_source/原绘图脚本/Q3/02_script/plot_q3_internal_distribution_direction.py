from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageOps
from scipy.stats import spearmanr


RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
MODULE_LABEL = {
    "terrain": "Topography and geomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
MODULE_SHORT = {
    "terrain": "Topography",
    "urbanization": "Urbanization",
    "human_activity": "Human activity",
}
MODULE_PANEL_A = {
    "terrain": "Topography and\ngeomorphology",
    "urbanization": "Urban\nconstruction",
    "human_activity": "Direct human\nactivity",
}
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
FAMILY_LABEL = {
    "community_baseline": "community-level attributes",
    "alpha_diversity": r"$\alpha$ diversity",
    "beta_diversity": r"$\beta$ diversity",
    "community_composition": "community composition",
    "functional_indicator": "indicator/tolerance taxa",
}

# River and Lake exactly match manuscript Figure 03.
COLORS = {
    "river": "#4F5DA9",
    "lake": "#B04A78",
    "terrain": "#8A6746",
    "urbanization": "#667F98",
    "human_activity": "#E07A5F",
    "text": "#20242A",
    "muted": "#667085",
    "light": "#D7DCE3",
}

# Use the central 80% of the original Purples colour span to avoid both the
# near-white and near-black extremes while preserving ordinal interpretation.
_PURPLES_BASE = mpl.colormaps["Purples"]
PURPLES_80 = mpl.colors.LinearSegmentedColormap.from_list(
    "Purples_central_80", _PURPLES_BASE(np.linspace(0.10, 0.90, 256))
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def find_csv(data_dir: Path, suffix: str) -> Path:
    matches = sorted(data_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV ending in {suffix!r}; found {matches}")
    return matches[0]


def panel_heading(
    fig: plt.Figure,
    ax: plt.Axes,
    letter: str,
    title: str,
    x_shift: float = 0.0,
    y_offset: float = 0.013,
) -> None:
    box = ax.get_position()
    y = box.y1 + y_offset
    fig.text(box.x0 - 0.033 + x_shift, y, letter, ha="left", va="bottom", fontsize=9.4, fontweight="bold", color="#111111")
    fig.text(box.x0 - 0.010 + x_shift, y, title, ha="left", va="bottom", fontsize=8.0, fontweight="semibold", color=COLORS["text"])


def luminance(hex_colour: str) -> float:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def heatmap_strength(ax: plt.Axes, cells: pd.DataFrame, system: str, title: str, show_y: bool) -> mpl.image.AxesImage:
    order = [(body, module) for body in ["River", "Lake"] for module in MODULES]
    matrix = np.full((6, 13), np.nan)
    for i, (body, module) in enumerate(order):
        group = cells[
            (cells["system"] == system)
            & (cells["water_body_type"] == body)
            & (cells["environment_module"] == module)
        ].set_index("range_m")
        matrix[i] = group.loc[RANGES, "strength_percentile_39"].to_numpy(float)

    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=PURPLES_80, interpolation="nearest")
    ax.set_title(title, pad=4.2, fontsize=6.7, fontweight="semibold")
    ax.set_xticks(range(13), [f"{r / 1000:g}" for r in RANGES])
    ax.set_xlabel("Spatial range (km)")
    labels = [MODULE_PANEL_A[module] for body, module in order]
    ax.set_yticks(range(6), labels if show_y else [])
    if show_y:
        # Contiguous driver-colour segments separate the long manuscript labels
        # from the heat map without consuming additional panel width.
        for row, (_, module) in enumerate(order):
            ax.add_line(
                Line2D(
                    [-0.004, -0.004],
                    [row - 0.5, row + 0.5],
                    transform=ax.get_yaxis_transform(),
                    color=COLORS[module],
                    lw=2.4,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=6,
                )
            )
        ax.text(
            -0.34,
            1.0,
            "River",
            transform=ax.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="semibold",
            color=COLORS["text"],
            clip_on=False,
        )
        ax.text(
            -0.34,
            4.0,
            "Lake",
            transform=ax.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="semibold",
            color=COLORS["text"],
            clip_on=False,
        )
    ax.axhline(2.5, color="white", lw=1.8)
    for y in range(7):
        ax.axhline(y - 0.5, color="white", lw=0.40)
    for x in range(14):
        ax.axvline(x - 0.5, color="white", lw=0.30)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def annotate_segment(ax: plt.Axes, left: float, width: float, y: float, colour: str, text: str, size: float = 6.8) -> None:
    text_colour = "white" if luminance(colour) < 0.58 else COLORS["text"]
    ax.text(left + width / 2, y, text, ha="center", va="center", fontsize=size, fontweight="semibold", color=text_colour)


def allocation_panel(ax: plt.Axes, body_summary: pd.DataFrame, driver_summary: pd.DataFrame) -> None:
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(0.15, 6.55)

    ax.text(0, 6.30, "Across waterbodies", ha="left", va="center", fontsize=6.7, fontweight="semibold", color=COLORS["text"])
    for y, label, system in [(5.70, "Water quality", "water_quality"), (4.82, "Aquatic biota", "biology_overall")]:
        summary = body_summary[body_summary["system"] == system].set_index("water_body_type")
        river = 100 * float(summary.loc["River", "share_of_system_total_across_two_waterbodies"])
        lake = 100 - river
        ax.barh(y, river, height=0.78, color=COLORS["river"], edgecolor="white", lw=0.5)
        ax.barh(y, lake, left=river, height=0.78, color=COLORS["lake"], edgecolor="white", lw=0.5)
        ax.text(-2.2, y, label, ha="right", va="center", fontsize=6.6, color=COLORS["text"])
        annotate_segment(ax, 0, river, y, COLORS["river"], f"River {river:.1f}%", 7.0)
        annotate_segment(ax, river, lake, y, COLORS["lake"], f"Lake {lake:.1f}%", 7.0)

    ax.text(
        0,
        4.05,
        "Across drivers within each waterbody and response system",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="semibold",
        color=COLORS["text"],
    )
    combos = [
        ("River · water quality", "River", "water_quality", 3.45),
        ("River · aquatic biota", "River", "biology_overall", 2.70),
        ("Lake · water quality", "Lake", "water_quality", 1.75),
        ("Lake · aquatic biota", "Lake", "biology_overall", 1.00),
    ]
    for label, body, system, y in combos:
        group = driver_summary[
            (driver_summary["water_body_type"] == body) & (driver_summary["system"] == system)
        ].set_index("environment_module")
        left = 0.0
        ax.text(-2.2, y, label, ha="right", va="center", fontsize=6.4, color=COLORS["text"])
        for module in MODULES:
            value = 100 * float(group.loc[module, "strength_allocation_share_within_body"])
            colour = COLORS[module]
            ax.barh(y, value, left=left, height=0.76, color=colour, edgecolor="white", lw=0.5)
            annotate_segment(ax, left, value, y, colour, f"{value:.1f}%", 6.7)
            left += value

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def allocation_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    def item(y: float, colour: str, label: str) -> None:
        ax.add_patch(
            plt.Rectangle(
                (0.04, y - 0.0275),
                0.13,
                0.055,
                transform=ax.transAxes,
                facecolor=colour,
                edgecolor="none",
                clip_on=False,
            )
        )
        ax.text(
            0.22,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.1,
            linespacing=0.90,
        )

    ax.text(0.04, 0.93, "Waterbody", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, fontweight="semibold")
    item(0.78, COLORS["river"], "River")
    item(0.66, COLORS["lake"], "Lake")
    ax.text(0.04, 0.54, "Environmental driver", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, fontweight="semibold")
    item(0.40, COLORS["terrain"], "Topography and\ngeomorphology")
    item(0.26, COLORS["urbanization"], "Urban\nconstruction")
    item(0.12, COLORS["human_activity"], "Direct human\nactivity")


def paired_rank_data(cells: pd.DataFrame) -> pd.DataFrame:
    paired = cells.pivot_table(
        index=["water_body_type", "environment_module", "range_m", "cell_key"],
        columns="system",
        values="strength_rank_39_desc",
        aggfunc="first",
    ).reset_index()
    paired.columns.name = None
    paired = paired.rename(
        columns={"water_quality": "water_quality_rank", "biology_overall": "biota_rank"}
    )
    if paired[["water_quality_rank", "biota_rank"]].isna().any().any():
        raise ValueError("Panel c requires paired water-quality and biotic ranks for every cell.")
    return paired


def paired_driver_summary(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for body in ["River", "Lake"]:
        for module in MODULES:
            group = paired[
                (paired["water_body_type"] == body)
                & (paired["environment_module"] == module)
            ].sort_values("range_m")
            records.append(
                {
                    "water_body_type": body,
                    "environment_module": module,
                    "n_nested_ranges": int(len(group)),
                    "water_quality_median_rank": float(group["water_quality_rank"].median()),
                    "biota_median_rank": float(group["biota_rank"].median()),
                    "range_trajectory_spearman_internal_ranks": float(
                        spearmanr(group["water_quality_rank"], group["biota_rank"]).statistic
                    ),
                    "inference": "descriptive trajectory summary; the 13 cumulative ranges are nested",
                }
            )
    return pd.DataFrame.from_records(records)


def paired_rank_panel(
    ax: plt.Axes,
    paired: pd.DataFrame,
    driver_summary: pd.DataFrame,
    body: str,
    show_y: bool,
) -> None:
    subset = paired[paired["water_body_type"] == body]
    for module in MODULES:
        group = subset[subset["environment_module"] == module].sort_values("range_m")
        colour = COLORS[module]
        for row in group.itertuples(index=False):
            ax.plot(
                [0, 1],
                [row.water_quality_rank, row.biota_rank],
                color=colour,
                lw=0.65,
                alpha=0.22,
                solid_capstyle="round",
                zorder=1,
            )
        ax.scatter(
            np.zeros(len(group)),
            group["water_quality_rank"],
            s=13,
            marker="o",
            facecolor=colour,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            np.ones(len(group)),
            group["biota_rank"],
            s=14,
            marker="D",
            facecolor=colour,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

        medians = [float(group["water_quality_rank"].median()), float(group["biota_rank"].median())]
        rho = float(
            driver_summary[
                (driver_summary["water_body_type"] == body)
                & (driver_summary["environment_module"] == module)
            ]["range_trajectory_spearman_internal_ranks"].iloc[0]
        )
        width = 1.4 + 3.7 * abs(rho)
        style = "-" if rho >= 0 else (0, (3.0, 1.6))
        median_x = [0.08, 0.92]
        ax.plot(median_x, medians, color=colour, lw=width, ls=style, alpha=1.0, zorder=5, solid_capstyle="round")
        ax.scatter([median_x[0]], [medians[0]], s=39, marker="o", facecolor=colour, edgecolor="#111111", linewidth=0.7, zorder=6)
        ax.scatter([median_x[1]], [medians[1]], s=40, marker="D", facecolor=colour, edgecolor="#111111", linewidth=0.7, zorder=6)
        y_mid = float(np.mean(medians))
        label_offset = {
            ("River", "terrain"): -3.40,
            ("River", "urbanization"): 3.30,
            ("River", "human_activity"): -3.10,
            ("Lake", "terrain"): 3.20,
            ("Lake", "urbanization"): -3.20,
            ("Lake", "human_activity"): 3.00,
        }[(body, module)]
        ax.text(
            0.50,
            y_mid + label_offset,
            rf"$\rho$={rho:+.2f}",
            ha="center",
            va="center",
            fontsize=5.5,
            color=colour,
            fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
            zorder=7,
        )

    ax.set_xlim(-0.23, 1.23)
    ax.set_ylim(40, 0)
    ax.set_xticks([0, 1], ["Water quality", "Aquatic biota"])
    ax.set_yticks([1, 10, 20, 30, 39] if show_y else [])
    if show_y:
        ax.set_ylabel("Within-system strength rank\n(1 = strongest)")
    ax.text(
        0.50,
        0.975,
        body,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="semibold",
        color=COLORS["text"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
        zorder=8,
    )
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.2)
    ax.spines["left"].set_visible(show_y)
    ax.spines["bottom"].set_color("#111111")
    ax.spines["bottom"].set_linewidth(0.75)


def paired_rank_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    driver_labels = {
        "terrain": "Topography and\ngeomorphology",
        "urbanization": "Urban\nconstruction",
        "human_activity": "Direct human\nactivity",
    }
    driver_handles = [
        Line2D([0], [0], color=COLORS[m], marker="o", markersize=4.0, lw=1.3, label=driver_labels[m])
        for m in MODULES
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="#444444", markerfacecolor="white", markersize=4.5, lw=0, label="Water quality rank"),
        Line2D([0], [0], marker="D", color="#444444", markerfacecolor="white", markersize=4.1, lw=0, label="Aquatic-biota rank"),
        Line2D([0, 1], [0, 0], color="#777777", lw=0.8, alpha=0.35, label="Same driver × range\npair"),
        Line2D([0, 1], [0, 0], color="#444444", lw=1.4, label=r"Median $|\rho|=0.0$"),
        Line2D([0, 1], [0, 0], color="#444444", lw=3.25, label=r"Median $|\rho|=0.5$"),
        Line2D([0, 1], [0, 0], color="#444444", lw=5.1, label=r"Median $|\rho|=1.0$"),
        Line2D([0, 1], [0, 0], color="#444444", lw=2.0, ls=(0, (3, 1.6)), label=r"Dashed: $\rho<0$"),
    ]
    ax.legend(
        handles=[*driver_handles, *marker_handles],
        loc="center left",
        bbox_to_anchor=(0.0, 0.50),
        handlelength=2.2,
        labelspacing=0.53,
        borderaxespad=0,
        frameon=False,
        fontsize=6.2,
    )


def build_concordance(direction: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for body in ["River", "Lake"]:
        for module in MODULES:
            for family in FAMILIES:
                group = direction[
                    (direction["water_body_type"] == body)
                    & (direction["environment_module"] == module)
                    & (direction["response_family"] == family)
                ].sort_values("range_m")
                if group["range_m"].tolist() != RANGES:
                    raise ValueError(f"Missing or unordered ranges for {body}, {module}, {family}")
                wq = group["wq_strength_percentile_39"].to_numpy(float)
                bio = group["family_strength_percentile_39"].to_numpy(float)
                rho = float(spearmanr(wq, bio).statistic)
                joint = np.sqrt(np.clip(wq, 0, 1) * np.clip(bio, 0, 1))
                sign_product = group["wq_direction_sign"].to_numpy(float) * group["family_direction_sign"].to_numpy(float)
                direction_score = float(np.average(sign_product, weights=joint)) if joint.sum() > 0 else float(np.mean(sign_product))
                records.append(
                    {
                        "water_body_type": body,
                        "environment_module": module,
                        "response_family": family,
                        "strength_trajectory_spearman_rho": rho,
                        "strength_weighted_direction_agreement": direction_score,
                        "mean_joint_percentile_strength": float(joint.mean()),
                        "dual_supported_range_n": int(group["dual_supported_05"].astype(bool).sum()),
                        "family_maxT_supported_range_n": int(group["family_maxT_supported_05"].astype(bool).sum()),
                        "wq_fdr_supported_range_n": int(group["wq_fdr_supported_05"].astype(bool).sum()),
                        "n_nested_ranges": len(group),
                    }
                )
    return pd.DataFrame.from_records(records)


def family_positions() -> tuple[dict[tuple[str, str], float], list[float], list[str], dict[str, float]]:
    positions: dict[tuple[str, str], float] = {}
    ticks: list[float] = []
    labels: list[str] = []
    module_top: dict[str, float] = {}
    y = 17.0
    for module in MODULES:
        module_top[module] = y + 0.55
        for family in FAMILIES:
            positions[(module, family)] = y
            ticks.append(y)
            labels.append(FAMILY_LABEL[family])
            y -= 1.0
        y -= 0.9
    return positions, ticks, labels, module_top


def bubble_area(joint_strength: float) -> float:
    return 17.0 + 78.0 * float(joint_strength)


def concordance_panel(ax: plt.Axes, summary: pd.DataFrame, body: str, show_y: bool) -> mpl.cm.ScalarMappable:
    positions, ticks, labels, _ = family_positions()
    cmap = mpl.colormaps["coolwarm"]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    subset = summary[summary["water_body_type"] == body]

    for module in MODULES:
        block_y = [positions[(module, family)] for family in FAMILIES]
        block_bottom = min(block_y) - 0.43
        block_top = max(block_y) + 0.43
        ax.vlines(0, block_bottom, block_top, color="#111111", lw=0.90, zorder=0)
        ax.plot(
            [-0.035, -0.035],
            [block_bottom, block_top],
            transform=ax.get_yaxis_transform(),
            color=COLORS[module],
            lw=4.2,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )
        for family in FAMILIES:
            row = subset[
                (subset["environment_module"] == module) & (subset["response_family"] == family)
            ].iloc[0]
            y = positions[(module, family)]
            x = float(row["strength_trajectory_spearman_rho"])
            direction_score = float(row["strength_weighted_direction_agreement"])
            colour = cmap(norm(direction_score))
            ax.hlines(y, 0, x, color=colour, lw=1.55, alpha=0.50, zorder=1)
            edge = "#111111" if int(row["dual_supported_range_n"]) > 0 else "white"
            edge_width = 1.0 if int(row["dual_supported_range_n"]) > 0 else 0.45
            ax.scatter(
                x,
                y,
                s=bubble_area(float(row["mean_joint_percentile_strength"])),
                facecolor=colour,
                edgecolor=edge,
                linewidth=edge_width,
                zorder=3,
            )

    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-0.1, 18.0)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"Strength-trajectory agreement, Spearman $\rho$")
    ax.set_yticks(ticks, labels if show_y else [])
    ax.set_title(body, pad=3.0, fontweight="semibold")
    ax.tick_params(axis="y", length=0, pad=10.0)
    ax.tick_params(axis="x", length=2.5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#111111")
    ax.spines["bottom"].set_linewidth(0.75)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def concordance_legend(ax: plt.Axes, mappable: mpl.cm.ScalarMappable) -> None:
    ax.set_axis_off()
    colour_ax = ax.inset_axes([0.02, 0.80, 0.13, 0.18])
    colourbar = ax.figure.colorbar(mappable, cax=colour_ax, orientation="vertical")
    colourbar.set_ticks([-1, 0, 1], labels=["Opposite", "Mixed", "Same sign"])
    colourbar.ax.tick_params(labelsize=6.1, length=2.0)
    colourbar.set_label("Weighted direction\nagreement", fontsize=6.2, labelpad=4)

    ax.text(
        0.02,
        0.69,
        "Environmental driver",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        fontweight="semibold",
    )
    driver_ys = {"terrain": 0.60, "urbanization": 0.51, "human_activity": 0.42}
    driver_labels = {
        "terrain": "Topography and\ngeomorphology",
        "urbanization": "Urban\nconstruction",
        "human_activity": "Direct human\nactivity",
    }
    for module in MODULES:
        y = driver_ys[module]
        ax.add_patch(
            plt.Rectangle(
                (0.02, y - 0.018),
                0.13,
                0.036,
                transform=ax.transAxes,
                facecolor=COLORS[module],
                edgecolor="none",
                clip_on=False,
            )
        )
        ax.text(
            0.20,
            y,
            driver_labels[module],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.1,
            linespacing=0.90,
        )

    ax.text(
        0.02,
        0.32,
        "Mean joint percentile\nstrength (point area; %)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        fontweight="bold",
        linespacing=0.95,
    )
    symbol_x = 0.085
    label_x = 0.20
    for value, y in zip([0.25, 0.50, 0.75], [0.215, 0.155, 0.095]):
        ax.scatter(
            [symbol_x],
            [y],
            s=bubble_area(value),
            transform=ax.transAxes,
            facecolor="#BFC5CE",
            edgecolor="white",
            linewidth=0.45,
            clip_on=False,
            zorder=3,
        )
        ax.text(
            label_x,
            y,
            f"{100 * value:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.1,
        )
    ax.scatter(
        [symbol_x],
        [0.035],
        s=42,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.0,
        clip_on=False,
        zorder=3,
    )
    ax.text(
        label_x,
        0.035,
        "≥1 dual-supported\nrange",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        linespacing=0.92,
    )


def ecological_panel(ax: plt.Axes, eco: pd.DataFrame) -> None:
    order = [(body, module) for body in ["River", "Lake"] for module in MODULES]
    labels = [f"{body[0]} · {MODULE_SHORT[module]}" for body, module in order]
    y = np.arange(len(order))[::-1]
    ax.set_xlim(-0.6, 20.6)
    ax.set_ylim(-0.7, len(order) - 0.3)
    for yi, (body, module) in zip(y, order):
        row = eco[(eco["water_body_type"] == body) & (eco["environment_module"] == module)].iloc[0]
        n_records = int(row["within_response_fdr_supported_records"])
        if n_records:
            colour = COLORS[body.lower()]
            ax.scatter(n_records, yi, s=28 + 6 * n_records, color=colour, edgecolor="white", linewidth=0.65, zorder=3)
            ax.text(n_records, yi, str(n_records), ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
        else:
            ax.scatter(0, yi, s=18, facecolor="white", edgecolor="#A7AFBA", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Ecologically oriented indicator × range records with within-response FDR q ≤ 0.05")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["river"], markeredgecolor="white", label="River: supported and concordant"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["lake"], markeredgecolor="white", label="Lake: supported and concordant"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#A7AFBA", label="No supported record"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        ncol=1,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def make_figure(data_dir: Path, output_dir: Path) -> None:
    setup_style()
    cells = pd.read_csv(find_csv(data_dir, "水质与生物总体逐range.csv"))
    body_summary = pd.read_csv(find_csv(data_dir, "水体内部汇总.csv"))
    driver_summary = pd.read_csv(find_csv(data_dir, "因素内部汇总.csv"))
    direction = pd.read_csv(find_csv(data_dir, "五类逐range.csv"))

    paired = paired_rank_data(cells)
    paired_summary = paired_driver_summary(paired)
    concordance = build_concordance(direction)
    paired.to_csv(data_dir / "Figure_Q3_panel_c_paired_internal_ranks.csv", index=False, encoding="utf-8-sig")
    paired_summary.to_csv(data_dir / "Figure_Q3_panel_c_driver_rank_summary.csv", index=False, encoding="utf-8-sig")
    concordance.to_csv(data_dir / "Figure_Q3_panel_d_direction_strength_concordance.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.80, 8.55))
    grid = fig.add_gridspec(
        4,
        16,
        height_ratios=[1.28, 1.74, 2.15, 2.72],
        hspace=0.32,
        wspace=0.58,
        left=0.190,
        right=0.982,
        top=0.958,
        bottom=0.058,
    )

    grid_a = grid[0, 0:13].subgridspec(1, 2, wspace=0.08)
    ax_a1 = fig.add_subplot(grid_a[0, 0])
    ax_a2 = fig.add_subplot(grid_a[0, 1])
    ax_aleg = fig.add_subplot(grid[0, 13:16])
    ax_aleg.set_axis_off()
    image_a = heatmap_strength(ax_a1, cells, "water_quality", r"Water quality: $|\rho|$ rank", True)
    heatmap_strength(ax_a2, cells, "biology_overall", "Aquatic biota: mean family RMS rank", False)
    ax_acb = ax_aleg.inset_axes([0.04, 0.00, 0.13, 1.00])
    colourbar_a = fig.colorbar(image_a, cax=ax_acb, orientation="vertical")
    colourbar_a.set_ticks([0, 0.5, 1], labels=["0", "50", "100"])
    colourbar_a.ax.tick_params(labelsize=6.1, length=2.0)
    colourbar_a.set_label("Strength percentile\n(within system)", fontsize=6.2, labelpad=3.0)

    ax_b = fig.add_subplot(grid[1, 0:13])
    ax_bleg = fig.add_subplot(grid[1, 13:16])
    allocation_panel(ax_b, body_summary, driver_summary)
    allocation_legend(ax_bleg)

    grid_c = grid[2, 0:13].subgridspec(1, 2, wspace=0.16)
    ax_c1 = fig.add_subplot(grid_c[0, 0])
    ax_c2 = fig.add_subplot(grid_c[0, 1])
    ax_cleg = fig.add_subplot(grid[2, 13:16])
    for axis in (ax_c1, ax_c2, ax_cleg):
        box = axis.get_position()
        axis.set_position([box.x0, box.y0 + 0.019, box.width, box.height])
    paired_rank_panel(ax_c1, paired, paired_summary, "River", True)
    paired_rank_panel(ax_c2, paired, paired_summary, "Lake", False)
    paired_rank_legend(ax_cleg)

    grid_d = grid[3, 0:13].subgridspec(1, 2, wspace=0.16)
    ax_d1 = fig.add_subplot(grid_d[0, 0])
    ax_d2 = fig.add_subplot(grid_d[0, 1])
    ax_dleg = fig.add_subplot(grid[3, 13:16])
    for axis in (ax_d1, ax_d2, ax_dleg):
        box = axis.get_position()
        axis.set_position([box.x0, box.y0 + 0.019, box.width, box.height])
    mappable = concordance_panel(ax_d1, concordance, "River", True)
    concordance_panel(ax_d2, concordance, "Lake", False)
    concordance_legend(ax_dleg, mappable)

    fig.canvas.draw()
    panel_heading(fig, ax_a1, "a", "Within-system response-strength distributions", y_offset=0.020)
    panel_heading(fig, ax_b, "b", "Allocation of response strength within each system", y_offset=0.005)
    panel_heading(fig, ax_c1, "c", "Paired water-quality and aquatic-biota strength ranks", y_offset=0.014)
    panel_heading(fig, ax_d1, "d", "Family-specific direction and strength concordance", y_offset=0.020)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "PNG": output_dir / "PNG" / "Figure_Q3_strength_distribution_and_direction.png",
        "PDF": output_dir / "PDF" / "Figure_Q3_strength_distribution_and_direction.pdf",
        "SVG": output_dir / "SVG" / "Figure_Q3_strength_distribution_and_direction.svg",
        "TIFF": output_dir / "TIFF" / "Figure_Q3_strength_distribution_and_direction.tiff",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["PNG"], dpi=600)
    fig.savefig(outputs["PDF"])
    fig.savefig(outputs["SVG"])
    fig.savefig(outputs["TIFF"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    grayscale_path = output_dir.parent / "04_QA" / "Figure_Q3_grayscale_QA.png"
    grayscale_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(outputs["PNG"]) as preview:
        ImageOps.grayscale(preview).save(grayscale_path)

    qa = {
        "backend": "Python/matplotlib only",
        "figure_size_inches": [round(float(v), 2) for v in fig.get_size_inches()],
        "figure_size_mm": [round(float(v) * 25.4, 2) for v in fig.get_size_inches()],
        "panels": ["a", "b", "c", "d"],
        "panel_e_removed": True,
        "formats": {key: str(path.relative_to(output_dir.parent)) for key, path in outputs.items()},
        "strength_cell_rows": int(len(cells)),
        "paired_rank_rows": int(len(paired)),
        "paired_driver_summary_rows": int(len(paired_summary)),
        "concordance_rows": int(len(concordance)),
        "direction_rows": int(len(direction)),
        "all_13_ranges_present": sorted(direction["range_m"].unique().tolist()) == RANGES,
        "all_five_families_present": set(direction["response_family"].unique()) == set(FAMILIES),
        "river_lake_colours_match_figure_03": {"River": COLORS["river"], "Lake": COLORS["lake"]},
        "top_quartile_overlap_displayed": False,
        "panel_c_contract": {
            "thin_lines": "same driver x range paired ranks at two aligned x positions",
            "opaque_lines": "driver-specific median ranks shifted inward",
            "line_width": "1.4 + 3.7 x absolute descriptive Spearman rho across 13 nested ranges",
            "line_width_legend": "explicit examples shown for absolute rho = 0.0, 0.5 and 1.0",
            "negative_rho": "dashed opaque line; signed rho annotated away from the median line",
        },
        "panel_d_contract": {
            "x": "descriptive Spearman rho of WQ and family strength-percentile trajectories",
            "colour": "joint-strength-weighted mean sign product across 13 ranges",
            "area": "mean geometric joint percentile strength",
            "black_outline": "at least one dual-supported range",
            "inference": "descriptive because ranges are nested",
        },
        "grey_explanatory_notes_in_panels_a_d": False,
        "response_system_term": "Aquatic biota",
        "panel_a_colourbar_height": "matches the heat-map plotting height",
        "panel_a_colour_span": "central 80% of the original Purples colormap",
        "panel_a_driver_bars": "contiguous manuscript-colour segments between the full driver labels and the heat map",
        "panel_b_legend": "environmental-driver section shifted upward and fully contained within the panel-b legend axis",
        "panel_c_titles": "River and Lake titles are inside the top of each plotting region",
        "panel_abc_headings": "all three main headings are lowered toward their plotting regions",
        "panel_c_heading": "lowered toward panel c while c and d rows are shifted upward to reduce the b-c gap",
        "panel_cd_bottom_axes": "black, 0.75 pt",
        "panel_d_zero_reference": "black and segmented by environmental-driver block",
        "panel_d_driver_labels": "full manuscript terms appear in the right legend only",
        "panel_d_driver_bars": "repeated beside both River and Lake plotting regions",
        "panel_d_connector_width": "1.55 pt",
        "panel_d_legend_layout": "taller direction colourbar and non-overlapping manual sections; size symbols align to driver swatches",
        "panel_d_size_legend_gap": "bold two-line title; four symbol rows use enlarged, even spacing and remain inside the legend axis",
        "png_export_dpi": 600,
        "legend_alignment": "all legends occupy grid columns 13:16",
        "plot_alignment": "all data panels occupy grid columns 0:13",
    }
    qa_dir = output_dir.parent / "04_QA"
    (qa_dir / "Figure_Q3_automated_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    make_figure(arguments.data, arguments.output)
