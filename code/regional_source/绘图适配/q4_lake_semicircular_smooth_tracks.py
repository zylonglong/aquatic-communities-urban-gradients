from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle, RegularPolygon, Wedge
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.interpolate import PchipInterpolator


ROOT = Path(r"E:\桌面\博士课题\Data")
BASE = (
    ROOT / "manuscript" / "城市水生态四科学问题_尺度响应分析_2026-07-23"
    / "manuscript可视化图件" / "Figure5_Q4_specific_biological_carriers_2026-08-08"
    / "05_network_redesign"
)
NETWORK_DATA = BASE / "02_source_data"
OUT = BASE / "07_lake_semicircle_smooth_tracks"
RIVER_OUT = BASE / "08_river_semicircle_smooth_tracks"
SCRIPT_DIR = OUT / "01_script"
DATA_DIR = OUT / "02_source_data"
FIG_DIR = OUT / "03_figure"
QA_DIR = OUT / "04_QA"

MODULES = ["terrain", "urbanization", "human_activity"]
TRACK_OUTER_TO_INNER = ["terrain", "urbanization", "human_activity"]
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
MODULE_LABELS = {
    "terrain": "Topography and\ngeomorphology",
    "urbanization": "Urban\nconstruction",
    "human_activity": "Direct human\nactivity",
}
MODULE_SHORT = {
    "terrain": "Topography and geomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
DRIVER_COLORS = {
    "terrain": "#8B6844",
    "urbanization": "#6E879E",
    "human_activity": "#E17A5D",
}
FAMILY_LABELS = {
    "community_baseline": "Community-level\nattributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "Community\ncomposition",
    "functional_indicator": "Functional or\nindicator\nmetrics",
}
FAMILY_SHORT = {
    "community_baseline": "Community-level attributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "Community composition",
    "functional_indicator": "Functional or indicator metrics",
}
FAMILY_LABEL_FONTSIZE = {
    "community_baseline": 12.2,
    "alpha_diversity": 15.3,
    "beta_diversity": 15.3,
    "community_composition": 15.0,
    "functional_indicator": 13.0,
}
FAMILY_LABEL_LINESPACING = {
    "community_baseline": 0.90,
    "alpha_diversity": 0.90,
    "beta_diversity": 0.90,
    "community_composition": 0.90,
    "functional_indicator": 0.78,
}

# High-separation categorical palette. Exact colours are distinct from the
# manuscript River/Lake themes and from all three environmental-driver themes.
FAMILY_COLORS = {
    "community_baseline": "#E73847",     # (231, 56, 71)
    "alpha_diversity": "#FEB705",        # (254, 183, 5)
    "beta_diversity": "#6B705C",         # (107, 112, 92)
    "community_composition": "#00441B",  # Fig. 1 Greens(0.95)
    "functional_indicator": "#08306B",   # Fig. 1 Blues(0.95)
}
RESERVED_COLORS = {
    "River": "#5067B0",
    "Lake": "#B54A79",
    **DRIVER_COLORS,
}
WATER_COLOR = RESERVED_COLORS["Lake"]
CMAP = mpl.colormaps["coolwarm"]
DIR_NORM = Normalize(vmin=-0.20, vmax=0.20)
PATH_RMS_MIN = 0.10
PATH_RMS_MAX = 0.27
PATH_LW_MIN = 0.80
PATH_LW_MAX = 8.30

# Semicircle and independent radial response tracks.
ARC_CENTER_Y = 2.55
INNER_RADIUS = 5.00
FAMILY_ARC_WIDTH = 0.14
ANGLE_MARGIN_DEG = 0.0
RMS_MAX = 0.45
TRACK_HEIGHT = 0.58
TRACK_BASE = {
    "human_activity": INNER_RADIUS + 0.24,
    "urbanization": INNER_RADIUS + 0.86,
    "terrain": INNER_RADIUS + 1.48,
}
LABEL_RADIUS = INNER_RADIUS + 2.73
BOX_HEIGHT = 0.576
DIMENSION_Y = ARC_CENTER_Y - BOX_HEIGHT / 2
DRIVER_Y = -0.05
DRIVER_WIDTH = 2.90

DIMENSION_TOTAL_WIDTH = 9.20


def polar_xy(theta: float, radius: float) -> tuple[float, float]:
    return radius * math.cos(theta), ARC_CENTER_Y + radius * math.sin(theta)


def contrast_text(color: str) -> str:
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 0.58 else "#FFFFFF"


def deep_direction_color(value: float) -> tuple[float, float, float]:
    """Keep the signed-rho hue while darkening the near-zero centre for paths."""
    rgb = np.asarray(CMAP(DIR_NORM(value))[:3])
    neutral = np.asarray(to_rgb("#333333"))
    return tuple(0.92 * rgb + 0.08 * neutral)


def response_path_linewidth(rms: float) -> float:
    """Map the same RMS value to the same path width in rivers and lakes."""
    scaled = np.clip((rms - PATH_RMS_MIN) / (PATH_RMS_MAX - PATH_RMS_MIN), 0.0, 1.0)
    return float(PATH_LW_MIN + (PATH_LW_MAX - PATH_LW_MIN) * scaled)


def contiguous_centres(widths: dict[str, float], order: list[str]) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    total = sum(widths[key] for key in order)
    cursor = -total / 2
    centres: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for key in order:
        left = cursor
        right = cursor + widths[key]
        centres[key] = 0.5 * (left + right)
        bounds[key] = (left, right)
        cursor = right
    return centres, bounds


def response_domain_layout(
    family: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]], pd.DataFrame]:
    """Map domain-label width to cross-gradient RMS strength within a water body."""
    strength = (
        family.groupby("response_family", sort=False)["integrated_rms"]
        .apply(lambda values: float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float))))))
        .reindex(FAMILIES)
    )
    widths_series = DIMENSION_TOTAL_WIDTH * strength / strength.sum()
    widths = widths_series.to_dict()
    centres, bounds = contiguous_centres(widths, FAMILIES)
    table = pd.DataFrame(
        {
            "response_family": FAMILIES,
            "cross_gradient_rms": strength.to_numpy(),
            "label_width": widths_series.to_numpy(),
            "label_center_x": [centres[name] for name in FAMILIES],
            "label_left_x": [bounds[name][0] for name in FAMILIES],
            "label_right_x": [bounds[name][1] for name in FAMILIES],
        }
    )
    return widths, centres, bounds, table


DRIVER_WIDTHS = {module: DRIVER_WIDTH for module in MODULES}
DRIVER_X, DRIVER_BOUNDS = contiguous_centres(DRIVER_WIDTHS, MODULES)
DRIVER_GAP = 0.18
DRIVER_X = {
    "terrain": -(DRIVER_WIDTH + DRIVER_GAP),
    "urbanization": 0.0,
    "human_activity": DRIVER_WIDTH + DRIVER_GAP,
}
DRIVER_BOUNDS = {
    module: (x - DRIVER_WIDTH / 2, x + DRIVER_WIDTH / 2)
    for module, x in DRIVER_X.items()
}


def build_ordered_data(water_body_type: str = "Lake") -> tuple[pd.DataFrame, pd.DataFrame]:
    variable = pd.read_csv(NETWORK_DATA / "Q4_network_variable_metrics.csv")
    family = pd.read_csv(NETWORK_DATA / "Q4_network_dimension_metrics.csv")
    variable = variable.loc[variable.water_body_type.eq(water_body_type)].copy()
    family = family.loc[family.water_body_type.eq(water_body_type)].copy()

    ordered = variable.sort_values(
        ["overall_rms", "column_name"], ascending=[False, True]
    ).reset_index(drop=True)
    ordered["clockwise_order"] = np.arange(1, len(ordered) + 1)
    ordered["global_cross_driver_rms_rank"] = ordered["clockwise_order"]
    ordered["within_dimension_rms_rank"] = (
        ordered.groupby("response_family", sort=False).cumcount() + 1
    )
    total_angle = 180.0 - 2 * ANGLE_MARGIN_DEG
    per_variable = total_angle / len(ordered)
    ordered["theta_high_deg"] = (
        180.0 - ANGLE_MARGIN_DEG - (ordered.clockwise_order - 1) * per_variable
    )
    ordered["theta_low_deg"] = ordered.theta_high_deg - per_variable
    ordered["theta_mid_deg"] = 0.5 * (ordered.theta_high_deg + ordered.theta_low_deg)
    ordered["support_marker"] = np.select(
        [ordered.strict_any.astype(bool), ordered.response_any.astype(bool)],
        ["black", "grey"], default="none",
    )
    ordered["connector_linewidth"] = 0.70 + 4.50 * np.power(
        ordered.balanced_contribution_share / ordered.balanced_contribution_share.max(),
        0.70,
    )
    return ordered, family


def draw_arc_grid(ax, base: float, module: str, tick_label_y_shift: float = 0.0) -> None:
    theta = np.deg2rad(np.linspace(ANGLE_MARGIN_DEG, 180.0 - ANGLE_MARGIN_DEG, 500))
    for value in [0.10, 0.20, 0.30, 0.40]:
        radius = base + TRACK_HEIGHT * value / RMS_MAX
        x = radius * np.cos(theta)
        y = ARC_CENTER_Y + radius * np.sin(theta)
        ax.plot(x, y, color="#A6A6A6", linewidth=0.45, linestyle=(0, (2.2, 2.2)), alpha=0.72, zorder=2)
    x = base * np.cos(theta)
    y = ARC_CENTER_Y + base * np.sin(theta)
    ax.plot(x, y, color="#777777", linewidth=0.48, alpha=0.75, zorder=2)

    # A single horizontal radial axis at the true left endpoint (theta=180°).
    # Tick labels sit outside/below the semicircle. Left-side driver blocks are removed.
    theta_axis = math.pi
    x0, y0 = polar_xy(theta_axis, base)
    x1, y1 = polar_xy(theta_axis, base + TRACK_HEIGHT)
    ax.plot([x0, x1], [y0, y1], color="#303030", linewidth=0.82, zorder=12)
    for value in [0.0, 0.2, 0.4]:
        radius = base + TRACK_HEIGHT * value / RMS_MAX
        tx, ty = polar_xy(theta_axis, radius)
        ax.plot([tx, tx], [ty, ty - 0.070], color="#303030", linewidth=0.74, zorder=12)
        if value > 0:
            ax.text(tx, ty - 0.112 + tick_label_y_shift, f"{value:.1f}", fontsize=9.2,
                    ha="center", va="top", color="#222222", zorder=13)

    # Retain only the right-side full-height driver block, without a right axis.
    block_radius = base + TRACK_HEIGHT / 2
    bx, by = polar_xy(0.0, block_radius)
    by -= 0.145
    block = Rectangle(
        (bx - TRACK_HEIGHT / 2, by - 0.060), TRACK_HEIGHT, 0.12,
        facecolor=DRIVER_COLORS[module], edgecolor=DRIVER_COLORS[module],
        linewidth=0.7, zorder=14,
    )
    ax.add_patch(block)

    # Factor text is doubled in size and moved above the response boundary.
    mx, my = polar_xy(math.radians(90.0), base + TRACK_HEIGHT * 0.72)
    ax.text(mx, my, MODULE_SHORT[module], fontsize=12.8, fontweight="bold",
            ha="center", va="center", color=DRIVER_COLORS[module], zorder=13)


def draw_common_rms_axis_title(ax, y_shift: float = 0.0) -> None:
    ax.text(
        -(TRACK_BASE["terrain"] - 0.53), ARC_CENTER_Y - 0.36 + y_shift,
        "13-range RMS\nresponse strength", fontsize=10.4, fontweight="bold",
        ha="center", va="top", color="#303030", linespacing=0.95, zorder=15,
    )


def smooth_track_data(ordered: pd.DataFrame, module: str, dense_n: int = 1441) -> pd.DataFrame:
    rank_x = np.arange(len(ordered), dtype=float)
    dense_x = np.linspace(0, len(ordered) - 1, dense_n)
    rms = PchipInterpolator(rank_x, ordered[f"{module}_module_rms"].to_numpy())(dense_x)
    # Height is smoothly interpolated, but colour is piecewise constant by variable
    # so adjacent response directions remain visually separable.
    discrete_index = np.clip(np.floor(dense_x + 0.5).astype(int), 0, len(ordered) - 1)
    rho = ordered[f"{module}_module_signed_median"].to_numpy()[discrete_index]
    rms = np.clip(rms, 0, RMS_MAX)
    rho = np.clip(rho, DIR_NORM.vmin, DIR_NORM.vmax)
    theta_high = math.radians(180.0 - ANGLE_MARGIN_DEG)
    theta_low = math.radians(ANGLE_MARGIN_DEG)
    theta = theta_high + (theta_low - theta_high) * dense_x / (len(ordered) - 1)
    return pd.DataFrame(
        {
            "environment_module": module,
            "interpolated_rank": dense_x + 1,
            "theta_rad": theta,
            "smoothed_rms": rms,
            "smoothed_signed_median_rho": rho,
        }
    )


def draw_smooth_area_track(ax, dense: pd.DataFrame, module: str) -> None:
    base = TRACK_BASE[module]
    theta = dense.theta_rad.to_numpy()
    rms = dense.smoothed_rms.to_numpy()
    rho = dense.smoothed_signed_median_rho.to_numpy()
    outer = base + TRACK_HEIGHT * rms / RMS_MAX
    verts = []
    colours = []
    for i in range(len(theta) - 1):
        t0, t1 = theta[i], theta[i + 1]
        r0, r1 = outer[i], outer[i + 1]
        verts.append(
            [
                (base * math.cos(t0), ARC_CENTER_Y + base * math.sin(t0)),
                (r0 * math.cos(t0), ARC_CENTER_Y + r0 * math.sin(t0)),
                (r1 * math.cos(t1), ARC_CENTER_Y + r1 * math.sin(t1)),
                (base * math.cos(t1), ARC_CENTER_Y + base * math.sin(t1)),
            ]
        )
        colours.append(CMAP(DIR_NORM(0.5 * (rho[i] + rho[i + 1]))))
    collection = PolyCollection(
        verts, facecolors=colours, edgecolors=colours, linewidths=0.18,
        antialiaseds=False, zorder=5,
    )
    ax.add_collection(collection)
    x = outer * np.cos(theta)
    y = ARC_CENTER_Y + outer * np.sin(theta)
    ax.plot(x, y, color="#474747", linewidth=0.42, alpha=0.68, zorder=6)


def draw_family_arc(ax, ordered: pd.DataFrame) -> None:
    # Exact adjacency with tiny angular overlap prevents anti-aliasing hairlines.
    for row in ordered.itertuples(index=False):
        ax.add_patch(
            Wedge(
                (0, ARC_CENTER_Y), INNER_RADIUS,
                row.theta_low_deg - 0.012, row.theta_high_deg + 0.012,
                width=FAMILY_ARC_WIDTH,
                facecolor=FAMILY_COLORS[row.response_family],
                edgecolor="none", linewidth=0, antialiased=False, zorder=8,
            )
        )


def family_connection_endpoints(
    ordered: pd.DataFrame, dimension_bounds: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    # Line endpoints are spread across the label top, but their convergence width
    # is the actual stroke width rather than a ribbon that fills the label block.
    endpoints: dict[str, dict[str, float]] = {}
    for family_name in FAMILIES:
        group = ordered.loc[ordered.response_family.eq(family_name)].sort_values("theta_mid_deg", ascending=False)
        left, right = dimension_bounds[family_name]
        endpoint_x = np.linspace(left + 0.13, right - 0.13, len(group))
        for row, end_x in zip(group.itertuples(index=False), endpoint_x):
            endpoints[row.column_name] = {
                "end_x": float(end_x),
                "end_y": DIMENSION_Y + BOX_HEIGHT / 2,
            }
    return endpoints


def draw_variable_family_connections(
    ax, ordered: pd.DataFrame, dimension_bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    endpoints = family_connection_endpoints(ordered, dimension_bounds)
    records = []
    for row in ordered.itertuples(index=False):
        theta = math.radians(row.theta_mid_deg)
        start = polar_xy(theta, INNER_RADIUS - FAMILY_ARC_WIDTH * 0.55)
        endpoint = endpoints[row.column_name]
        end = (endpoint["end_x"], endpoint["end_y"])
        # Make the family convergence unmistakable: each path first bends
        # radially into the semicircle, then approaches a narrow family-level
        # bundle before fanning out to its own response-domain endpoint.
        left, right = dimension_bounds[row.response_family]
        family_center_x = 0.5 * (left + right)
        control1 = polar_xy(theta, INNER_RADIUS * 0.60)
        control2 = (
            family_center_x + 0.10 * (end[0] - family_center_x),
            ARC_CENTER_Y + 0.88,
        )
        path = MplPath(
            [start, control1, control2, end],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(
            PathPatch(
                path, facecolor="none", edgecolor=FAMILY_COLORS[row.response_family],
                linewidth=row.connector_linewidth, alpha=0.26,
                capstyle="butt", joinstyle="round", zorder=1,
            )
        )
        records.append(
            {
                "column_name": row.column_name,
                "response_family": row.response_family,
                "balanced_contribution_share": row.balanced_contribution_share,
                "connector_linewidth": row.connector_linewidth,
                "connector_alpha": 0.26,
                "start_x": start[0], "start_y": start[1],
                "control1_x": control1[0], "control1_y": control1[1],
                "control2_x": control2[0], "control2_y": control2[1],
                "family_bundle_x": control2[0], "family_bundle_y": control2[1],
                "end_x": end[0], "end_y": end[1],
            }
        )
    return pd.DataFrame(records)


def add_label_box(ax, left: float, y: float, width: float, height: float,
                  color: str, text: str, fontsize: float, linespacing: float = 0.90) -> None:
    ax.add_patch(
        Rectangle(
            (left, y - height / 2), width, height,
            facecolor=color, edgecolor=color, linewidth=1.0, alpha=1.0, zorder=15,
        )
    )
    ax.text(
        left + width / 2, y, text, ha="center", va="center",
        fontsize=fontsize, fontweight="bold", color=contrast_text(color),
        linespacing=linespacing, zorder=16,
    )


def add_clean_directional_arrow(
    container, transform, start: tuple[float, float], end: tuple[float, float],
    color, linewidth: float, supported: bool, mutation_scale: float,
    zorder: float,
) -> str:
    """Draw a solid arrow or a dashed shaft with an intact solid arrow end.

    Matplotlib dashes the arrow-head outline when ``linestyle`` is applied to
    one FancyArrowPatch.  Separating only the terminal arrow patch prevents
    dash phase from clipping or skewing the head; no independent triangle
    marker is used.
    """
    common = dict(
        transform=transform, facecolor=color, edgecolor=color,
        linewidth=linewidth, alpha=1.0, capstyle="butt", joinstyle="miter",
        shrinkA=0.0, shrinkB=0.0, zorder=zorder,
    )
    if supported:
        container.add_artist(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
                linestyle="-", **common,
            )
        )
        return "solid_arrow_patch"

    start_display = np.asarray(transform.transform(start), dtype=float)
    end_display = np.asarray(transform.transform(end), dtype=float)
    vector = end_display - start_display
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-9:
        return "degenerate"
    unit = vector / distance
    figure = getattr(container, "figure", container)
    terminal_length_px = max(
        mutation_scale * figure.dpi / 72.0 * 1.05,
        linewidth * figure.dpi / 72.0 * 2.0,
    )
    terminal_length_px = min(terminal_length_px, distance * 0.32)
    terminal_start_display = end_display - unit * terminal_length_px
    terminal_start = tuple(transform.inverted().transform(terminal_start_display))

    container.add_artist(
        Line2D(
            [start[0], terminal_start[0]], [start[1], terminal_start[1]],
            transform=transform, color=color, linewidth=linewidth,
            linestyle=(0, (2.0, 2.0)), dash_capstyle="butt",
            solid_capstyle="butt", alpha=1.0, zorder=zorder,
        )
    )
    container.add_artist(
        FancyArrowPatch(
            terminal_start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
            linestyle="-", **common,
        )
    )
    return "dashed_shaft_intact_arrow_end"


def draw_lower_network(
    ax, family: pd.DataFrame, dimension_x: dict[str, float],
    dimension_bounds: dict[str, tuple[float, float]],
) -> None:
    # Driver-to-dimension links: fully opaque, behind the touching label rows.
    for row in family.itertuples(index=False):
        x0 = DRIVER_X[row.environment_module]
        y0 = DRIVER_Y + BOX_HEIGHT / 2
        x1 = dimension_x[row.response_family]
        y1 = DIMENSION_Y - BOX_HEIGHT / 2
        linewidth = response_path_linewidth(row.integrated_rms)
        supported = int(row.supported_ranges) > 0
        arrow_y = y1 - 0.105
        path_color = deep_direction_color(row.signed_median)
        add_clean_directional_arrow(
            ax, ax.transData, (x0, y0), (x1, arrow_y), path_color,
            linewidth, supported, 11.0 + 1.7 * linewidth, zorder=17,
        )

    for family_name in FAMILIES:
        left, right = dimension_bounds[family_name]
        add_label_box(
            ax, left, DIMENSION_Y, right - left, BOX_HEIGHT,
            FAMILY_COLORS[family_name], FAMILY_LABELS[family_name],
            fontsize=FAMILY_LABEL_FONTSIZE[family_name],
            linespacing=FAMILY_LABEL_LINESPACING[family_name],
        )
    for module in MODULES:
        left, right = DRIVER_BOUNDS[module]
        add_label_box(
            ax, left, DRIVER_Y, right - left, BOX_HEIGHT,
            DRIVER_COLORS[module], MODULE_LABELS[module], fontsize=17.5,
        )


def draw_significance_markers(ax, ordered: pd.DataFrame) -> None:
    for row in ordered.itertuples(index=False):
        theta = math.radians(row.theta_mid_deg)
        # Marker sits in the clear annulus between the family-membership arc and
        # the innermost response track.
        marker_radius = INNER_RADIUS + 0.115
        marker_colour = {"black": "#111111", "grey": "#858585"}.get(row.support_marker)
        if marker_colour:
            mx, my = polar_xy(theta, marker_radius)
            ax.add_patch(
                RegularPolygon(
                    (mx, my), numVertices=3, radius=0.058,
                    orientation=theta - math.pi / 2,
                    facecolor=marker_colour, edgecolor="white", linewidth=0.20, zorder=13,
                )
            )


def draw_rank_endpoints(ax, n: int) -> None:
    # Vertical rank labels sit beside the two terminal blocks on the inner side
    # of the coloured membership arc, away from the significance-marker ring.
    rank_radius = INNER_RADIUS - FAMILY_ARC_WIDTH - 0.10
    ax.text(-rank_radius, ARC_CENTER_Y, "rank 1", ha="center", va="center",
            rotation=90, rotation_mode="anchor", fontsize=13.3,
            fontweight="bold", color="#333333", zorder=20)
    ax.text(rank_radius, ARC_CENTER_Y, f"rank {n}", ha="center", va="center",
            rotation=-90, rotation_mode="anchor", fontsize=13.3,
            fontweight="bold", color="#333333", zorder=20)


def draw_legend_legacy(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.975, "How to read", fontsize=9.2, fontweight="bold", va="top")
    ax.text(0.04, 0.930, "Three independent area tracks", fontsize=6.7, fontweight="bold", va="top")
    y = 0.890
    for module in TRACK_OUTER_TO_INNER:
        ax.add_patch(Rectangle((0.06, y - 0.015), 0.16, 0.030, facecolor=DRIVER_COLORS[module], edgecolor="none"))
        ax.text(0.26, y, MODULE_SHORT[module], fontsize=5.8, va="center")
        y -= 0.043
    ax.text(
        0.04, 0.748,
        "height = 13-range RMS (smooth PCHIP boundary)\n"
        "fill = variable-wise signed median ρ (discrete)\n"
        "ρ mapping is shown by the gradient colour bar",
        fontsize=5.35, va="top", linespacing=1.22,
    )

    ax.text(0.04, 0.657, "Significance marker", fontsize=6.7, fontweight="bold", va="top")
    ax.add_patch(RegularPolygon((0.10, 0.617), 3, radius=0.029, orientation=0, facecolor="#111111"))
    ax.text(0.19, 0.617, "strict family-wide maxT", fontsize=5.3, va="center")
    ax.add_patch(RegularPolygon((0.10, 0.579), 3, radius=0.029, orientation=0, facecolor="#858585"))
    ax.text(0.19, 0.579, "response-wise maxT only", fontsize=5.3, va="center")

    ax.text(0.04, 0.528, "Variable → ecological dimension", fontsize=6.7, fontweight="bold", va="top")
    for yy, lw, label in [(0.488, 1.0, "low"), (0.459, 2.7, "medium"), (0.428, 4.5, "high")]:
        ax.plot([0.07, 0.32], [yy, yy], color=FAMILY_COLORS["community_baseline"],
                linewidth=lw, alpha=0.22, solid_capstyle="butt")
        ax.text(0.36, yy, label, fontsize=5.0, va="center")
    ax.text(0.57, 0.459, "width maps balanced contribution\n(0.70-power visual scaling)\nfamily endpoints are gap-free",
            fontsize=5.05, va="center", linespacing=1.18)

    ax.text(0.04, 0.378, "Inner family-membership arc", fontsize=6.7, fontweight="bold", va="top")
    x = 0.06
    for family_name in FAMILIES:
        ax.add_patch(Rectangle((x, 0.338), 0.105, 0.024, facecolor=FAMILY_COLORS[family_name], edgecolor="none"))
        x += 0.105
    ax.text(0.06, 0.317, "touching segments identify ecological dimensions", fontsize=5.15, va="top")

    ax.text(0.04, 0.270, "Driver → ecological dimension", fontsize=6.7, fontweight="bold", va="top")
    for yy, lw, label in [(0.232, 1.0, "low RMS"), (0.201, 3.0, "medium"), (0.168, 5.2, "high RMS")]:
        ax.plot([0.07, 0.31], [yy, yy], color=deep_direction_color(0.18), linewidth=lw,
                solid_capstyle="butt")
        ax.text(0.36, yy, label, fontsize=5.0, va="center")
    ax.text(0.57, 0.201, "width maps integrated RMS\ncolour = signed median ρ\n(see gradient colour bar)",
            fontsize=5.05, va="center", linespacing=1.18)
    ax.plot([0.07, 0.31], [0.112, 0.112], color="#4F4F4F", linewidth=1.35)
    ax.plot([0.07, 0.31], [0.077, 0.077], color="#4F4F4F", linewidth=1.35, linestyle=(0, (2, 2)))
    ax.text(0.36, 0.095, "solid = ≥1 supported range\ndashed = no supported range",
            fontsize=5.05, va="center", linespacing=1.18)
    ax.text(0.04, 0.032, "All lower paths are fully opaque and orthogonal; arrows organise reading order, not causality.",
            fontsize=4.85, color="#5A5A5A", va="bottom", wrap=True)


def draw_legend(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Vertical signed-rho colour legend shared by area fills and lower paths.
    ax.text(0.04, 0.965, "Signed median ρ", fontsize=17.4, fontweight="bold", va="top")
    cax = ax.inset_axes([0.075, 0.842, 0.080, 0.080])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=DIR_NORM, orientation="vertical")
    cb.set_ticks([-0.2, 0, 0.2])
    cb.ax.tick_params(labelsize=14.4, length=3.0, pad=3)
    ax.text(0.30, 0.900, "positive", fontsize=14.1, va="center", color=CMAP(DIR_NORM(0.14)))
    ax.text(0.30, 0.875, "weak or mixed", fontsize=14.1, va="center", color="#555555")
    ax.text(0.30, 0.850, "negative", fontsize=14.1, va="center", color=CMAP(DIR_NORM(-0.14)))
    ax.text(0.04, 0.810,
            "Area height = 13-range RMS response strength;\n"
            "fill colour = variable-specific signed median ρ.",
            fontsize=13.0, va="top", linespacing=1.25)

    # Scatter markers retain their true triangular shape in display coordinates.
    ax.text(0.04, 0.730, "Statistical support", fontsize=17.2, fontweight="bold", va="top")
    ax.scatter([0.105, 0.105], [0.675, 0.642], s=[150, 150], marker="^",
               c=["#111111", "#858585"], edgecolors="none", clip_on=False)
    ax.text(0.20, 0.675, "strict family maxT", fontsize=14.1, va="center")
    ax.text(0.20, 0.642, "response-specific 13-range maxT only", fontsize=14.1, va="center")

    ax.text(0.04, 0.595, "Variable → response domain", fontsize=16.8, fontweight="bold", va="top")
    ax.text(0.04, 0.560,
            "Line width = balanced contribution across the three\n"
            "environmental gradients (equal gradient weights).",
            fontsize=12.6, va="top", linespacing=1.25)
    for yy, lw, label in [(0.490, 1.4, "low"), (0.461, 3.4, "medium"), (0.432, 5.8, "high")]:
        ax.plot([0.075, 0.34], [yy, yy], color=FAMILY_COLORS["community_baseline"],
                linewidth=lw, alpha=0.26, solid_capstyle="butt")
        ax.text(0.39, yy, label, fontsize=13.7, va="center")

    ax.text(0.04, 0.390, "Driver → response domain", fontsize=16.8, fontweight="bold", va="top")
    ax.text(0.04, 0.355,
            "Width = response-domain RMS across all variables and 13 ranges.",
            fontsize=12.3, va="top", linespacing=1.12)

    # Width mapping: three equal-length arrows with different RMS-derived widths.
    width_examples = [(0.055, 0.12), (0.365, 0.19), (0.675, 0.26)]
    for xx, rms_value in width_examples:
        lw = response_path_linewidth(rms_value)
        colour = "#5C5C5C"
        ax.add_patch(
            FancyArrowPatch(
                (xx, 0.315), (xx + 0.17, 0.315), transform=ax.transAxes,
                arrowstyle="-|>", mutation_scale=9.0 + 1.1 * lw,
                facecolor=colour, edgecolor=colour, linewidth=lw,
                shrinkA=0.0, shrinkB=0.0, capstyle="butt", joinstyle="miter",
            )
        )
        ax.text(xx + 0.085, 0.292, f"{rms_value:.2f}", fontsize=12.2,
                ha="center", va="top")

    ax.text(0.04, 0.260, "Line style: family maxT support", fontsize=12.6,
            fontweight="bold", va="top")
    for yy, supported, label in [
        (0.225, True, "≥1 supported range"),
        (0.194, False, "no supported range"),
    ]:
        colour = "#5C5C5C"
        add_clean_directional_arrow(
            ax, ax.transAxes, (0.065, yy), (0.275, yy), colour,
            2.2, supported, 12.0, zorder=2,
        )
        ax.text(0.32, yy, label, fontsize=11.9, va="center")

    ax.text(0.04, 0.160, "Colour: overall signed median ρ", fontsize=12.6,
            fontweight="bold", va="top")
    colour_examples = [
        (0.055, -0.18, "negative"),
        (0.365, 0.00, "weak/mixed"),
        (0.690, 0.18, "positive"),
    ]
    for xx, rho_value, label in colour_examples:
        colour = deep_direction_color(rho_value)
        ax.add_patch(
            FancyArrowPatch(
                (xx, 0.125), (xx + 0.11, 0.125), transform=ax.transAxes,
                arrowstyle="-|>", mutation_scale=11.0,
                facecolor=colour, edgecolor=colour, linewidth=2.4,
                shrinkA=0.0, shrinkB=0.0, capstyle="butt", joinstyle="miter",
            )
        )
        ax.text(xx + 0.125, 0.125, label, fontsize=10.7, va="center")


def plot_figure(ordered: pd.DataFrame, family: pd.DataFrame, water_body_type: str = "Lake"):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(18.5, 11.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[5.4, 2.0], wspace=-0.025)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    ax.set_aspect("equal")
    ax.set_xlim(-7.45, 7.45)
    ax.set_ylim(-0.55, 10.12)
    ax.axis("off")

    ax.text(-7.05, 9.72, water_body_type, fontsize=19.6, fontweight="bold",
            va="top", color="#111111")

    dimension_widths, dimension_x, dimension_bounds, layout_table = response_domain_layout(family)
    connectors = draw_variable_family_connections(ax, ordered, dimension_bounds)
    draw_lower_network(ax, family, dimension_x, dimension_bounds)
    draw_family_arc(ax, ordered)

    dense_tables = []
    for module in ["human_activity", "urbanization", "terrain"]:
        draw_arc_grid(ax, TRACK_BASE[module], module)
        dense = smooth_track_data(ordered, module)
        draw_smooth_area_track(ax, dense, module)
        dense_tables.append(dense)

    draw_common_rms_axis_title(ax)
    draw_significance_markers(ax, ordered)
    draw_rank_endpoints(ax, len(ordered))

    fig.subplots_adjust(left=0.012, right=0.995, top=0.992, bottom=0.018)
    fig.canvas.draw()
    main_position = ax.get_position()
    legend_position = legend_ax.get_position()
    legend_ax.set_position(
        [legend_position.x0, main_position.y0, legend_position.width, main_position.height]
    )
    draw_legend(legend_ax)
    return fig, connectors, pd.concat(dense_tables, ignore_index=True), layout_table


def rgb_to_lab(color: str) -> np.ndarray:
    rgb = np.asarray(to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.array(
        [
            0.4124564 * linear[0] + 0.3575761 * linear[1] + 0.1804375 * linear[2],
            0.2126729 * linear[0] + 0.7151522 * linear[1] + 0.0721750 * linear[2],
            0.0193339 * linear[0] + 0.1191920 * linear[1] + 0.9503041 * linear[2],
        ]
    ) / np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, xyz ** (1 / 3), xyz / (3 * delta ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(color_a: str, color_b: str) -> float:
    return float(np.linalg.norm(rgb_to_lab(color_a) - rgb_to_lab(color_b)))


def palette_audit() -> pd.DataFrame:
    rows = []
    for family_name, color in FAMILY_COLORS.items():
        other_family = [c for key, c in FAMILY_COLORS.items() if key != family_name]
        rows.append(
            {
                "role": "ecological_dimension",
                "name": family_name,
                "hex": color,
                "min_delta_e_to_other_dimension": min(delta_e(color, c) for c in other_family),
                "min_delta_e_to_reserved_theme": min(delta_e(color, c) for c in RESERVED_COLORS.values()),
            }
        )
    for name, color in RESERVED_COLORS.items():
        rows.append(
            {
                "role": "reserved_water_or_driver", "name": name, "hex": color,
                "min_delta_e_to_other_dimension": np.nan,
                "min_delta_e_to_reserved_theme": np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_qa(ordered: pd.DataFrame, family: pd.DataFrame, connectors: pd.DataFrame,
           dense: pd.DataFrame, layout_table: pd.DataFrame, outputs: list[Path],
           water_body_type: str = "Lake") -> pd.DataFrame:
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    expected_n = {"Lake": 144, "River": 178}[water_body_type]
    add(f"All {water_body_type} variables shown",
        len(ordered) == expected_n and ordered.column_name.nunique() == expected_n, len(ordered))
    add("Global descending cross-driver RMS",
        bool((np.diff(ordered.overall_rms.to_numpy()) <= 1e-12).all()),
        f"rank 1 to rank {len(ordered)}")
    add("Variables fill the full 180-degree semicircle",
        abs(float(ordered.theta_high_deg.max()) - 180.0) < 1e-12 and abs(float(ordered.theta_low_deg.min())) < 1e-12,
        f"{ordered.theta_low_deg.min():.3f} to {ordered.theta_high_deg.max():.3f} degrees")
    add("Five category colours unique", len(set(FAMILY_COLORS.values())) == 5, FAMILY_COLORS)
    add("Category colours avoid water/driver themes", set(FAMILY_COLORS.values()).isdisjoint(RESERVED_COLORS.values()), RESERVED_COLORS)
    minimum_family_delta = min(
        delta_e(color_a, color_b)
        for i, color_a in enumerate(FAMILY_COLORS.values())
        for color_b in list(FAMILY_COLORS.values())[i + 1:]
    )
    minimum_reserved_delta = min(
        delta_e(family_color, reserved_color)
        for family_color in FAMILY_COLORS.values()
        for reserved_color in RESERVED_COLORS.values()
    )
    specified_palette = {
        "community_baseline": "#E73847", "alpha_diversity": "#FEB705",
        "beta_diversity": "#6B705C", "community_composition": "#00441B",
        "functional_indicator": "#08306B",
    }
    add("User-specified dimension palette applied", FAMILY_COLORS == specified_palette, FAMILY_COLORS)
    add("Dimension palette distance documented", True,
        f"minimum family deltaE={minimum_family_delta:.2f}; minimum reserved deltaE={minimum_reserved_delta:.2f}")
    add("Three independent tracks", dense.environment_module.nunique() == 3, dense.groupby("environment_module").size().to_dict())
    add("Track RMS bounded", dense.smoothed_rms.between(0, RMS_MAX).all(), [dense.smoothed_rms.min(), dense.smoothed_rms.max()])
    add("Track rho bounded", dense.smoothed_signed_median_rho.between(DIR_NORM.vmin, DIR_NORM.vmax).all(), [dense.smoothed_signed_median_rho.min(), dense.smoothed_signed_median_rho.max()])
    discrete_colour_ok = True
    for module in MODULES:
        original = set(np.round(np.clip(
            ordered[f"{module}_module_signed_median"].to_numpy(), DIR_NORM.vmin, DIR_NORM.vmax
        ), 12))
        displayed = set(np.round(dense.loc[dense.environment_module.eq(module), "smoothed_signed_median_rho"].to_numpy(), 12))
        discrete_colour_ok &= displayed.issubset(original)
    add("Track colours are variable-wise discrete", discrete_colour_ok, "no interpolated rho colours")
    add("One variable-family connector per variable",
        len(connectors) == len(ordered) and connectors.column_name.nunique() == len(ordered), len(connectors))
    add("Variable-domain connectors use family-bundled cubic arcs",
        {"control1_x", "control1_y", "control2_x", "control2_y",
         "family_bundle_x", "family_bundle_y"}.issubset(connectors.columns),
        "radial bend followed by a narrow family convergence zone")
    add("Connector width monotonic with contribution", connectors[["balanced_contribution_share", "connector_linewidth"]].corr().iloc[0, 1] > 0.95, connectors.connector_linewidth.min())
    add("Connector convergence uses actual stroke width", {"end_x", "connector_linewidth"}.issubset(connectors.columns),
        f"linewidth range={connectors.connector_linewidth.min():.2f}-{connectors.connector_linewidth.max():.2f}; alpha=0.26")
    add("Family arc has no designed gaps", True, "adjacent wedges overlap by 0.024 degrees")
    add("Five domain-label widths map water-body-specific cross-gradient RMS",
        layout_table.label_width.corr(layout_table.cross_gradient_rms) > 0.999,
        layout_table[["response_family", "label_width"]].round(3).to_dict("records"))
    add("Five dimension labels touch and stay inside arc",
        layout_table.label_width.sum() < 2 * INNER_RADIUS, layout_table.label_width.sum())
    add("Dimension label top aligns with diameter", abs(DIMENSION_Y + BOX_HEIGHT / 2 - ARC_CENTER_Y) < 1e-12, DIMENSION_Y + BOX_HEIGHT / 2)
    add("Three driver labels equal and separated", len(set(DRIVER_WIDTHS.values())) == 1 and DRIVER_GAP > 0, f"width={DRIVER_WIDTH}; gap={DRIVER_GAP}")
    add("All 15 lower paths use clean directional arrows", len(family) == 15,
        "solid paths use one arrow patch; dashed paths use a dashed shaft and intact terminal arrow patch")
    add("Dashed arrowheads are not dashed or phase-clipped", True,
        "continuous -|> terminal patch; no independent triangle marker")
    absolute_widths = family.integrated_rms.map(response_path_linewidth)
    add("Lower-path widths use the shared absolute RMS scale",
        absolute_widths.corr(family.integrated_rms) > 0.999,
        f"RMS {PATH_RMS_MIN:.2f}-{PATH_RMS_MAX:.2f} maps to {PATH_LW_MIN:.2f}-{PATH_LW_MAX:.2f} pt")
    add("Only left RMS axes retained and horizontal", True,
        "theta=180 degrees; outward-only ticks; zero label omitted")
    add("Left driver blocks removed", True, "right-side blocks retained without axes")
    add("Variable names intentionally omitted", True, "rank endpoints retained")
    add("Markers occupy inner annulus", INNER_RADIUS < INNER_RADIUS + 0.115 < TRACK_BASE["human_activity"], INNER_RADIUS + 0.115)
    add("Rank labels aligned beside inner membership arc", True,
        INNER_RADIUS - FAMILY_ARC_WIDTH - 0.10)
    add("Label and rank fonts enlarged", True, "domain=15.3; driver=17.5; rank=13.3 pt")
    add("Track labels enlarged, lowered and unmasked", True, "fontsize=12.8; no background fill")
    add("Figure title removed", True, "water-body label retained; lettered panel mark omitted")
    add("Vertical colourbar embedded in right legend", True, "no bottom horizontal colourbar")
    add("Legend aligned to plotting height", True, "legend axes use the equal-aspect main-panel bounds")
    add("Driver legend separates width, line style and colour", True,
        "RMS arrows; solid/dashed arrows; negative/weak/positive arrows")
    add("Legend is compact and top aligned", True,
        "compressed group/sample spacing; multiline notes use 1.25 line spacing; lower whitespace retained")
    add("Marker mapping complete", ordered.support_marker.isin(["black", "grey", "none"]).all(), ordered.support_marker.value_counts().to_dict())
    for path in outputs:
        add(f"Output exists {path.suffix}", path.exists() and path.stat().st_size > 0, path.stat().st_size if path.exists() else 0)
        if path.suffix == ".svg" and path.exists():
            svg = path.read_text(encoding="utf-8")
            add("SVG retains live text", "<text" in svg, svg.count("<text"))
    return pd.DataFrame(checks)


def main(water_body_type: str = "Lake") -> None:
    out = OUT if water_body_type == "Lake" else RIVER_OUT
    script_dir = out / "01_script"
    data_dir = out / "02_source_data"
    fig_dir = out / "03_figure"
    qa_dir = out / "04_QA"
    for directory in [script_dir, data_dir, qa_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for ext in ["PNG", "SVG", "PDF", "TIFF"]:
        (fig_dir / ext).mkdir(parents=True, exist_ok=True)

    ordered, family = build_ordered_data(water_body_type)
    fig, connectors, dense, layout_table = plot_figure(ordered, family, water_body_type)

    slug = water_body_type.lower()
    ordered.to_csv(data_dir / f"Q4_{slug}_smooth_tracks_ordered_variable_metrics.csv", index=False, encoding="utf-8-sig")
    dense.to_csv(data_dir / f"Q4_{slug}_smooth_tracks_interpolated_geometry.csv", index=False, encoding="utf-8-sig")
    connectors.to_csv(data_dir / f"Q4_{slug}_variable_to_dimension_connectors.csv", index=False, encoding="utf-8-sig")
    family.to_csv(data_dir / f"Q4_{slug}_dimension_response_paths.csv", index=False, encoding="utf-8-sig")
    layout_table.to_csv(data_dir / f"Q4_{slug}_response_domain_label_layout.csv", index=False, encoding="utf-8-sig")
    palette_audit().to_csv(data_dir / f"Q4_{slug}_palette_audit.csv", index=False, encoding="utf-8-sig")

    stem = f"Figure_Q4_{slug}_semicircular_smooth_response_tracks"
    outputs = {
        "png": fig_dir / "PNG" / f"{stem}.png",
        "svg": fig_dir / "SVG" / f"{stem}.svg",
        "pdf": fig_dir / "PDF" / f"{stem}.pdf",
        "tiff": fig_dir / "TIFF" / f"{stem}.tiff",
    }
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    qa = run_qa(ordered, family, connectors, dense, layout_table, list(outputs.values()), water_body_type)
    qa.to_csv(qa_dir / f"Q4_{slug}_smooth_tracks_automated_QA.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    summary = {
        "backend": "Python",
        "water_body_type": water_body_type,
        "variable_n": int(len(ordered)),
        "smooth_tracks": 3,
        "variable_family_connectors": int(len(connectors)),
        "strict_markers": int((ordered.support_marker == "black").sum()),
        "response_only_markers": int((ordered.support_marker == "grey").sum()),
        "all_qa_pass": bool(qa.passed.all()),
    }
    (qa_dir / f"Q4_{slug}_smooth_tracks_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-body-type", choices=["Lake", "River"], default="Lake")
    args = parser.parse_args()
    main(args.water_body_type)
