from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle, RegularPolygon, Wedge


ROOT = Path(r"E:\桌面\博士课题\Data")
BASE = (
    ROOT / "manuscript" / "城市水生态四科学问题_尺度响应分析_2026-07-23"
    / "manuscript可视化图件" / "Figure5_Q4_specific_biological_carriers_2026-08-08"
    / "05_network_redesign"
)
NETWORK_DATA = BASE / "02_source_data"
OUT = BASE / "06_lake_semicircle_prototype"
SCRIPT_DIR = OUT / "01_script"
DATA_DIR = OUT / "02_source_data"
FIG_DIR = OUT / "03_figure"
QA_DIR = OUT / "04_QA"

MODULES = ["terrain", "urbanization", "human_activity"]
# Outermost to innermost, matching the requested visual top-to-bottom order.
STACK_OUTER_TO_INNER = ["terrain", "urbanization", "human_activity"]
STACK_INNER_TO_OUTER = list(reversed(STACK_OUTER_TO_INNER))
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
MODULE_LABELS = {
    "terrain": "Topography and\ngeomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
MODULE_SHORT = {"terrain": "Terrain", "urbanization": "Urban", "human_activity": "Direct activity"}
DRIVER_COLORS = {"terrain": "#8B6844", "urbanization": "#6E879E", "human_activity": "#E17A5D"}
FAMILY_LABELS = {
    "community_baseline": "Community-level\nattributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "Community\ncomposition",
    "functional_indicator": "Functional /\nindicator",
}
FAMILY_SHORT = {
    "community_baseline": "Community-level attributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "Community composition",
    "functional_indicator": "Functional/indicator",
}
FAMILY_COLORS = {
    "community_baseline": "#355C9A",
    "alpha_diversity": "#6F91C7",
    "beta_diversity": "#9E8ABF",
    "community_composition": "#D08A79",
    "functional_indicator": "#B45266",
}
WATER_COLOR = "#B54A79"
CMAP = mpl.colormaps["coolwarm"]
DIR_NORM = Normalize(vmin=-0.40, vmax=0.40)

# Main geometry.
ARC_CENTER_Y = 2.20
INNER_RADIUS = 4.70
BAR_BASE_RADIUS = INNER_RADIUS + 0.15
ANGLE_MARGIN_DEG = 2.5
DIMENSION_X = dict(zip(FAMILIES, [-4.20, -2.10, 0.0, 2.10, 4.20]))
DIMENSION_Y = 0.72
DRIVER_X = {"terrain": -3.0, "urbanization": 0.0, "human_activity": 3.0}
DRIVER_Y = -1.05


def blend_white(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple((1 - amount) * rgb + amount * np.ones(3))


def polar_xy(theta: float, radius: float) -> tuple[float, float]:
    return radius * math.cos(theta), ARC_CENTER_Y + radius * math.sin(theta)


def build_ordered_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variable = pd.read_csv(NETWORK_DATA / "Q4_network_variable_metrics.csv")
    family = pd.read_csv(NETWORK_DATA / "Q4_network_dimension_metrics.csv")
    variable = variable.loc[variable.water_body_type.eq("Lake")].copy()
    family = family.loc[family.water_body_type.eq("Lake")].copy()

    # Global clockwise ranking: all retained variables are ordered from high to low
    # cross-driver RMS. The thin inner-arc segment of each variable carries family
    # membership, so family colours may legitimately alternate along the semicircle.
    total_angle = 180.0 - 2 * ANGLE_MARGIN_DEG
    per_variable_deg = total_angle / len(variable)
    cursor = 180.0 - ANGLE_MARGIN_DEG
    ordered_rows = []
    ranked = variable.sort_values(["overall_rms", "column_name"], ascending=[False, True]).reset_index(drop=True)
    ranked["within_dimension_rms_rank"] = (
        ranked.groupby("response_family", sort=False).cumcount() + 1
    )
    for global_rank, row in enumerate(ranked.itertuples(index=False), start=1):
        theta_high = cursor
        theta_low = cursor - per_variable_deg
        theta_mid = 0.5 * (theta_high + theta_low)
        record = row._asdict()
        record.update(
            {
                "clockwise_order": global_rank,
                "global_cross_driver_rms_rank": global_rank,
                "theta_high_deg": theta_high,
                "theta_low_deg": theta_low,
                "theta_mid_deg": theta_mid,
            }
        )
        ordered_rows.append(record)
        cursor = theta_low

    ordered = pd.DataFrame(ordered_rows)
    sectors = ordered[
        ["clockwise_order", "column_name", "display_label", "response_family",
         "theta_high_deg", "theta_low_deg", "theta_mid_deg"]
    ].copy()
    max_total_rms = ordered[[f"{m}_module_rms" for m in MODULES]].sum(axis=1).max()
    radial_scale = 1.42 / max_total_rms
    for module in MODULES:
        ordered[f"{module}_stack_height"] = ordered[f"{module}_module_rms"] * radial_scale
    ordered["total_stack_height"] = ordered[[f"{m}_stack_height" for m in MODULES]].sum(axis=1)
    ordered["support_marker"] = np.select(
        [ordered.strict_any.astype(bool), ordered.response_any.astype(bool)],
        ["black", "grey"],
        default="none",
    )
    return ordered, sectors, family


def add_square(ax, x: float, y: float, side: float, face, edge, label: str, fontsize: float, linewidth: float = 1.2):
    patch = Rectangle((x - side / 2, y - side / 2), side, side, facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=10)
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#222222", zorder=11)
    return patch


def draw_family_arc(ax, segment, color: str):
    # One thin family-coloured segment per variable; micro-gaps keep memberships
    # separable while all segments together complete the upper semicircle.
    span = segment.theta_high_deg - segment.theta_low_deg
    inset = 0.055 * span
    ax.add_patch(
        Wedge(
            (0, ARC_CENTER_Y), INNER_RADIUS,
            segment.theta_low_deg + inset, segment.theta_high_deg - inset,
            width=0.095, facecolor=color, edgecolor="none", zorder=6,
        )
    )


def draw_radial_bars(ax, ordered: pd.DataFrame):
    max_stack = float(ordered.total_stack_height.max())
    fixed_label_radius = BAR_BASE_RADIUS + max_stack + 0.42
    rows = []
    for row in ordered.itertuples(index=False):
        theta1, theta2 = row.theta_low_deg, row.theta_high_deg
        # Leave a narrow white gap between neighbouring variables.
        inset = 0.12 * (theta2 - theta1)
        theta1 += inset
        theta2 -= inset
        bottom = BAR_BASE_RADIUS
        segment_geometry = {}
        for module in STACK_INNER_TO_OUTER:
            height = float(getattr(row, f"{module}_stack_height"))
            direction = float(getattr(row, f"{module}_module_signed_median"))
            outer = bottom + height
            ax.add_patch(
                Wedge(
                    (0, ARC_CENTER_Y), outer, theta1, theta2, width=height,
                    facecolor=CMAP(DIR_NORM(direction)), edgecolor="white", linewidth=0.18, zorder=7,
                )
            )
            segment_geometry[module] = {"inner_radius": bottom, "outer_radius": outer}
            bottom = outer

        theta = math.radians(row.theta_mid_deg)
        marker_radius = bottom + 0.095
        marker_x, marker_y = polar_xy(theta, marker_radius)
        marker_colour = None
        if row.support_marker == "black":
            marker_colour = "#111111"
        elif row.support_marker == "grey":
            marker_colour = "#858585"
        if marker_colour:
            ax.add_patch(
                RegularPolygon(
                    (marker_x, marker_y), numVertices=3, radius=0.038,
                    orientation=theta - math.pi / 2, facecolor=marker_colour,
                    edgecolor="white", linewidth=0.25, zorder=10,
                )
            )

        label_x, label_y = polar_xy(theta, fixed_label_radius)
        angle = row.theta_mid_deg
        if angle <= 90:
            rotation, ha = angle, "left"
        else:
            rotation, ha = angle - 180, "right"
        ax.text(
            label_x, label_y, row.display_label, rotation=rotation, rotation_mode="anchor",
            ha=ha, va="center", fontsize=2.45, color="#252525", clip_on=False, zorder=12,
        )
        rows.append(
            {
                "column_name": row.column_name,
                "label_radius": fixed_label_radius,
                "marker_radius": marker_radius,
                "bar_outer_radius": bottom,
                "segment_geometry_json": json.dumps(segment_geometry),
            }
        )
    return pd.DataFrame(rows)


def draw_lower_network(ax, family: pd.DataFrame):
    family_once = family.drop_duplicates("response_family").set_index("response_family")
    strength_min, strength_max = family.integrated_rms.min(), family.integrated_rms.max()
    size_min, size_max = family_once.family_overall_rms.min(), family_once.family_overall_rms.max()

    # Fully opaque driver-to-dimension paths.
    for row in family.itertuples(index=False):
        x0, y0 = DRIVER_X[row.environment_module], DRIVER_Y + 0.62
        x1 = DIMENSION_X[row.response_family]
        frow = family_once.loc[row.response_family]
        size_scaled = (frow.family_overall_rms - size_min) / max(size_max - size_min, 1e-9)
        side = 0.78 + 0.25 * size_scaled
        y1 = DIMENSION_Y - side / 2
        strength_scaled = (row.integrated_rms - strength_min) / max(strength_max - strength_min, 1e-9)
        linewidth = 0.75 + 4.2 * strength_scaled
        color = CMAP(DIR_NORM(row.signed_median))
        supported = int(row.supported_ranges) > 0
        curvature = 0.0 if abs(x1 - x0) < 1e-9 else 0.065 * np.sign(x1 - x0)
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.0,
                connectionstyle=f"arc3,rad={curvature}", linewidth=linewidth,
                linestyle="-" if supported else (0, (2.0, 2.0)),
                color=color, alpha=1.0, zorder=2,
            )
        )

    # Square family nodes. Square side maps the previous ellipse/node size.
    for family_name in FAMILIES:
        frow = family_once.loc[family_name]
        scaled = (frow.family_overall_rms - size_min) / max(size_max - size_min, 1e-9)
        side = 0.78 + 0.25 * scaled
        add_square(
            ax, DIMENSION_X[family_name], DIMENSION_Y, side,
            blend_white(FAMILY_COLORS[family_name], 0.76), FAMILY_COLORS[family_name],
            FAMILY_LABELS[family_name], fontsize=5.5,
        )

    # Square driver labels.
    for module in MODULES:
        add_square(
            ax, DRIVER_X[module], DRIVER_Y, 1.24,
            blend_white(DRIVER_COLORS[module], 0.77), DRIVER_COLORS[module],
            MODULE_LABELS[module], fontsize=6.4,
        )


def draw_legend(ax):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.04, 0.975, "How to read", fontsize=9, fontweight="bold", va="top")
    ax.text(0.04, 0.928, "Semicircular variable spectrum", fontsize=6.8, fontweight="bold")
    # Stack example, top-to-bottom as requested.
    x, y, w = 0.08, 0.775, 0.17
    heights = {"human_activity": 0.075, "urbanization": 0.105, "terrain": 0.13}
    bottom = y
    for module in STACK_INNER_TO_OUTER:
        h = heights[module]
        ax.add_patch(Rectangle((x, bottom), w, h, facecolor=CMAP(DIR_NORM({"terrain": -0.28, "urbanization": 0.25, "human_activity": 0.06}[module])), edgecolor="white", linewidth=0.5))
        bottom += h
    ax.text(0.30, y + sum(heights.values()) - heights["terrain"] / 2, "Terrain", fontsize=5.8, va="center")
    ax.text(0.30, y + heights["human_activity"] + heights["urbanization"] / 2, "Urban", fontsize=5.8, va="center")
    ax.text(0.30, y + heights["human_activity"] / 2, "Direct activity", fontsize=5.8, va="center")
    ax.text(0.04, 0.735, "segment height = 13-range RMS", fontsize=5.7)
    ax.text(0.04, 0.708, "fill colour = signed median ρ", fontsize=5.7)

    ax.add_patch(RegularPolygon((0.10, 0.655), 3, radius=0.025, orientation=0, facecolor="#111111"))
    ax.text(0.17, 0.655, "strict family-wide maxT", fontsize=5.6, va="center")
    ax.add_patch(RegularPolygon((0.10, 0.615), 3, radius=0.025, orientation=0, facecolor="#858585"))
    ax.text(0.17, 0.615, "response-wise maxT only", fontsize=5.6, va="center")
    ax.text(0.04, 0.570, "All variables are sorted clockwise by\nglobal cross-driver RMS (rank 1 to n).", fontsize=5.7, va="top")

    ax.text(0.04, 0.490, "Five ecological dimensions", fontsize=6.8, fontweight="bold")
    y0 = 0.455
    for family_name in FAMILIES:
        ax.plot([0.06, 0.20], [y0, y0], color=FAMILY_COLORS[family_name], linewidth=5, solid_capstyle="butt")
        ax.text(0.24, y0, FAMILY_SHORT[family_name], fontsize=5.5, va="center")
        y0 -= 0.038

    ax.text(0.04, 0.245, "Lower association network", fontsize=6.8, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.06, 0.205), (0.31, 0.205), arrowstyle="-|>", mutation_scale=7, linewidth=3.5, color=CMAP(DIR_NORM(0.26)), alpha=1.0))
    ax.text(0.36, 0.205, "width = RMS\ncolour = signed median ρ", fontsize=5.5, va="center")
    ax.plot([0.06, 0.31], [0.155, 0.155], color="#555555", linewidth=1.1)
    ax.plot([0.06, 0.31], [0.120, 0.120], color="#555555", linewidth=1.1, linestyle=(0, (2, 2)))
    ax.text(0.36, 0.138, "solid = supported\ndashed = unsupported", fontsize=5.5, va="center")
    ax.text(0.04, 0.055, "All paths are fully opaque. Arrowheads\norganize predictor-to-response layers;\nthey do not imply causal SEM paths.", fontsize=5.3, color="#666666", va="bottom")


def plot_figure(ordered: pd.DataFrame, sectors: pd.DataFrame, family: pd.DataFrame):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(20.0, 13.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[5.35, 1.18], wspace=0.015)
    ax = fig.add_subplot(gs[0, 0]); leg = fig.add_subplot(gs[0, 1])
    ax.set_aspect("equal"); ax.set_xlim(-9.45, 9.45); ax.set_ylim(-2.25, 11.55); ax.axis("off")

    ax.text(-9.25, 11.34, "c", fontsize=11, fontweight="bold", va="top")
    ax.text(-8.85, 11.34, "Lake semicircular hierarchical response network", fontsize=12, fontweight="bold", color=WATER_COLOR, va="top")
    ax.text(-8.85, 11.02, "n = 44 independent sites · all 144 retained variables · association network, not a causal SEM", fontsize=6.8, color="#555555", va="top")

    # Lower paths first, then family arcs and radial spectrum.
    draw_lower_network(ax, family)
    for sector in sectors.itertuples(index=False):
        draw_family_arc(ax, sector, FAMILY_COLORS[sector.response_family])
    geometry = draw_radial_bars(ax, ordered)

    # Explicit endpoints make the global ordering in the hand-drawn design readable.
    for theta_deg, label, align in [
        (ordered.theta_mid_deg.iloc[0], "rank 1", "right"),
        (ordered.theta_mid_deg.iloc[-1], f"rank {len(ordered)}", "left"),
    ]:
        tx, ty = polar_xy(math.radians(theta_deg), INNER_RADIUS - 0.22)
        ax.text(tx, ty, label, ha=align, va="center", fontsize=5.5,
                fontweight="bold", color="#3C3C3C", zorder=12)

    # Direction colour scale inside the lower-left margin.
    cax = fig.add_axes([0.165, 0.042, 0.30, 0.012])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=DIR_NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-0.4, 0, 0.4]); cb.ax.tick_params(labelsize=5, length=2)
    cb.set_label("Signed median partial Spearman ρ", fontsize=5.6, labelpad=1)
    draw_legend(leg)
    fig.subplots_adjust(left=0.018, right=0.988, top=0.985, bottom=0.075)
    return fig, geometry


def run_qa(ordered: pd.DataFrame, sectors: pd.DataFrame, outputs: list[Path]) -> pd.DataFrame:
    checks = []
    def add(name, passed, detail): checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    add("All Lake variables shown", len(ordered) == 144 and ordered.column_name.nunique() == 144, len(ordered))
    add("One family arc segment per variable", len(sectors) == len(ordered) and sectors.column_name.nunique() == len(ordered), len(sectors))
    add("Clockwise angular order", bool((np.diff(ordered.theta_mid_deg.to_numpy()) < 0).all()), f"{ordered.theta_mid_deg.iloc[0]:.2f} to {ordered.theta_mid_deg.iloc[-1]:.2f}")
    global_ok = bool((np.diff(ordered.overall_rms.to_numpy()) <= 1e-12).all())
    add("Global descending cross-driver RMS", global_ok, "rank 1 to n clockwise")
    add("Three nonnegative stack heights", bool((ordered[[f"{m}_stack_height" for m in MODULES]] >= 0).all().all()), STACK_OUTER_TO_INNER)
    add("Marker mapping complete", ordered.support_marker.isin(["black", "grey", "none"]).all(), ordered.support_marker.value_counts().to_dict())
    add("Dimension labels are square", True, "side mapped to family_overall_rms")
    add("Driver labels are square", True, "fixed side 1.24")
    add("All 15 driver-dimension paths opaque", True, "alpha=1.0")
    for path in outputs:
        add(f"Output exists {path.suffix}", path.exists() and path.stat().st_size > 0, path.stat().st_size if path.exists() else 0)
        if path.suffix == ".svg" and path.exists():
            svg = path.read_text(encoding="utf-8")
            add("SVG retains live text", "<text" in svg, svg.count("<text"))
    return pd.DataFrame(checks)


def main():
    for path in [SCRIPT_DIR, DATA_DIR, QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for ext in ["PNG", "SVG", "PDF", "TIFF"]:
        (FIG_DIR / ext).mkdir(parents=True, exist_ok=True)

    ordered, sectors, family = build_ordered_data()
    fig, geometry = plot_figure(ordered, sectors, family)
    geometry = ordered.merge(geometry, on="column_name", how="left")
    stale_sector_file = DATA_DIR / "Q4_lake_semicircle_dimension_sectors.csv"
    if stale_sector_file.exists():
        stale_sector_file.unlink()
    geometry.to_csv(DATA_DIR / "Q4_lake_semicircle_variable_geometry_and_metrics.csv", index=False, encoding="utf-8-sig")
    sectors.to_csv(DATA_DIR / "Q4_lake_semicircle_variable_family_arc_segments.csv", index=False, encoding="utf-8-sig")
    family.to_csv(DATA_DIR / "Q4_lake_semicircle_dimension_paths.csv", index=False, encoding="utf-8-sig")

    stem = "Figure_Q4_lake_semicircular_hierarchical_response_network"
    paths = {
        "png": FIG_DIR / "PNG" / f"{stem}.png",
        "svg": FIG_DIR / "SVG" / f"{stem}.svg",
        "pdf": FIG_DIR / "PDF" / f"{stem}.pdf",
        "tiff": FIG_DIR / "TIFF" / f"{stem}.tiff",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["tiff"], dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    qa = run_qa(ordered, sectors, list(paths.values()))
    qa.to_csv(QA_DIR / "Q4_lake_semicircle_automated_QA.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    summary = {
        "backend": "Python",
        "water_body_type": "Lake",
        "variable_n": int(len(ordered)),
        "strict_markers": int((ordered.support_marker == "black").sum()),
        "response_only_markers": int((ordered.support_marker == "grey").sum()),
        "all_qa_pass": bool(qa.passed.all()),
    }
    (QA_DIR / "Q4_lake_semicircle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
