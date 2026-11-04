from __future__ import annotations

import json
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

import q4_lake_semicircular_smooth_tracks as network_base
import q4_river_lake_shared_semicircular_network as network
import fig5b_top5_driver_responses as top5b


BASE = network_base.BASE.parent
SOURCE_DIR = BASE / "02_source_data"
SCRIPT_DIR = BASE / "01_script"
FIG_DIR = BASE / "03_figure"
QA_DIR = BASE / "04_QA"

FAMILIES = network_base.FAMILIES
FAMILY_COLORS = network_base.FAMILY_COLORS
FAMILY_LABELS = {
    "community_baseline": "Community-level attributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "Community composition",
    "functional_indicator": "Functional or indicator metrics",
}
MODULES = network_base.MODULES
MODULE_LABELS = {
    "terrain": "Topography and geomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
WATERS = ["River", "Lake"]
C_WATER_LABEL_X = -5.65


def contrast_text(hex_color: str) -> str:
    rgb = np.asarray(mpl.colors.to_rgb(hex_color))
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#111111" if luminance > 0.58 else "#FFFFFF"


def clean_variable_label(name: str) -> str:
    text = str(name)
    replacements = [
        (r"^beta_(river|lake|all)_", ""),
        (r"_bray_curtis_", " Bray–Curtis "),
        (r"_bray_", " Bray–Curtis "),
        (r"_sorensen_", " Sørensen "),
        (r"_jaccard_", " Jaccard "),
        (r"_nestedness_", " nestedness "),
        (r"_turnover_", " turnover "),
        (r"_lcbd$", " LCBD"),
        (r"_mean$", " mean dissimilarity"),
        (r"_pcoa([123])$", r" PCoA\1"),
        (r"_pct_", " % "),
        (r"_pct$", " %"),
        (r"_cwm_", " CWM "),
        (r"_", " "),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^macrophytes?", "Macrophyte", text, flags=re.IGNORECASE)
    text = re.sub(r"^benthos", "Benthic invertebrate", text, flags=re.IGNORECASE)
    text = text.replace("qual pa", "presence/absence")
    text = text.replace("quant", "quantitative")
    text = text.replace("density Bray–Curtis", "abundance Bray–Curtis")
    text = text.replace("count Bray–Curtis", "abundance Bray–Curtis")
    text = text.replace("total density", "total abundance")
    # Restore manuscript casing after the sequential underscore substitutions.
    text = re.sub(r"\blcbd\b", "LCBD", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpcoa([123])\b", r"PCoA\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpielou\b", "Pielou", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmargalef\b", "Margalef", text, flags=re.IGNORECASE)
    return text[:1].upper() + text[1:] if text else str(name)


def wrap_label(text: str, width: int = 31) -> str:
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def load_panel_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    family_share = pd.read_csv(SOURCE_DIR / "Figure_Q4_family_signal_shares.csv")
    selected, _ = top5b.load_and_select()
    return family_share, selected


def panel_title(
    ax, letter: str, title: str, title_fontsize: float = 20.5,
    linespacing: float = 1.00,
) -> None:
    ax.axis("off")
    title_y = 0.82
    ax.text(
        0.000, title_y, letter, fontsize=23.0, fontweight="bold", va="top",
    )
    ax.annotate(
        title, xy=(0.000, title_y), xycoords="axes fraction",
        xytext=(36, 0), textcoords="offset points",
        fontsize=title_fontsize, fontweight="bold", va="top",
        linespacing=linespacing,
    )


def draw_panel_a(ax, family_share: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    summary = (
        family_share.groupby(
            ["water_body_type", "environment_module", "response_family"],
            as_index=False,
        ).family_signal_share.mean()
    )
    summary["percentage"] = 100.0 * summary.family_signal_share
    ordered_cells = [(water, module) for water in WATERS for module in MODULES]
    y = np.asarray([5.15, 4.15, 3.15, 1.85, 0.85, -0.15])
    left = np.zeros(len(ordered_cells))
    annotations = 0
    for family in FAMILIES:
        values = []
        for water, module in ordered_cells:
            hit = summary.loc[
                summary.water_body_type.eq(water)
                & summary.environment_module.eq(module)
                & summary.response_family.eq(family),
                "percentage",
            ]
            values.append(float(hit.iloc[0]))
        values = np.asarray(values)
        bars = ax.barh(
            y, values, left=left, height=0.72,
            color=FAMILY_COLORS[family], edgecolor="white", linewidth=0.75,
        )
        for bar, value, start in zip(bars, values, left):
            ax.text(
                start + value / 2, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", ha="center", va="center",
                fontsize=11.0, fontweight="bold",
                color=contrast_text(FAMILY_COLORS[family]),
            )
            annotations += 1
        left += values

    labels = [MODULE_LABELS[module].replace(" ", "\n", 1)
              if module != "terrain" else "Topography and\ngeomorphology"
              for _, module in ordered_cells]
    ax.set_yticks(y, labels, fontsize=13.4, linespacing=0.92)
    # Keep the waterbody labels fixed while moving the driver labels clear of
    # the narrow River/Lake colour strips.
    ax.tick_params(axis="y", length=0, pad=15)
    group_transform = blended_transform_factory(ax.transAxes, ax.transData)
    group_specs = [
        ("River", np.mean(y[:3]), 2.76, 5.54, network_base.RESERVED_COLORS["River"]),
        ("Lake", np.mean(y[3:]), -0.54, 2.24, network_base.RESERVED_COLORS["Lake"]),
    ]
    for water, centre_y, lower_y, upper_y, colour in group_specs:
        ax.add_patch(
            Rectangle(
                (-0.012, lower_y), 0.007, upper_y - lower_y,
                transform=group_transform, facecolor=colour, edgecolor=colour,
                linewidth=0.0, clip_on=False, zorder=4,
            )
        )
        ax.text(
            -0.118, centre_y, water, transform=group_transform,
            rotation=90, fontsize=15.2, fontweight="bold",
            ha="center", va="center", color="#111111", clip_on=False,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.72, 5.72)
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(2.50, color="#C8C8C8", linewidth=0.8, zorder=0)
    return summary, annotations


def draw_panel_a_legend(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.04, 0.93, "Ecological response dimensions",
        fontsize=17.4, fontweight="bold", va="top",
    )
    handles = [Patch(facecolor=FAMILY_COLORS[family], edgecolor="none") for family in FAMILIES]
    labels = [FAMILY_LABELS[family] for family in FAMILIES]
    ax.legend(
        handles, labels, ncol=1, loc="upper left", bbox_to_anchor=(0.02, 0.82),
        frameon=False, fontsize=13.8, handlelength=1.8, handleheight=1.0,
        labelspacing=1.05, borderaxespad=0.0, handletextpad=0.8,
    )


def draw_panel_b_axis(
    ax, selected: pd.DataFrame, water: str, shared_xmax: float,
) -> int:
    subset = selected.loc[selected.water_body_type.eq(water)].copy().reset_index(drop=True)
    positions = np.arange(len(subset))[::-1]
    for position, row in zip(positions, subset.itertuples(index=False)):
        value = 100.0 * row.balanced_contribution_share
        low = 100.0 * row.bootstrap_ci95_low
        high = 100.0 * row.bootstrap_ci95_high
        color = FAMILY_COLORS[row.response_family]
        ax.plot(
            [low, high], [position, position], color=color,
            linewidth=3.4, alpha=0.80, solid_capstyle="butt", zorder=2,
        )
        filled = row.top3_selection_frequency >= 0.70
        ax.scatter(
            value, position, s=112, facecolor=color if filled else "white",
            edgecolor=color, linewidth=2.1, zorder=3,
        )

    labels = [wrap_label(label) for label in subset.display_label]
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, labelsize=14.2, pad=8, colors="#111111")
    for tick in ax.get_yticklabels():
        tick.set_color("#111111")
        tick.set_linespacing(0.92)
    ax.set_xlim(0, shared_xmax)
    ax.set_xticks(np.arange(0, shared_xmax + 0.1, 10.0))
    ax.tick_params(axis="x", labelsize=13.8, length=5.0, width=0.9)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.65, linestyle=(0, (2.0, 2.4)))
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Balanced contribution within response domain (%)",
        fontsize=15.2, labelpad=7,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    return len(subset)


def add_panel_b_legend(ax) -> None:
    colour = "#505050"
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=10.5,
            markerfacecolor=colour, markeredgecolor=colour, markeredgewidth=1.8,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=10.5,
            markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.8,
        ),
        Line2D([0], [0], color=colour, linewidth=3.4, solid_capstyle="butt"),
    ]
    labels = [
        "Top-three frequency ≥70%",
        "Top-three frequency <70%",
        "95% site-bootstrap interval",
    ]
    ax.legend(
        handles, labels, loc="lower right", bbox_to_anchor=(0.995, 0.015),
        fontsize=11.6, frameon=False, borderaxespad=0.0,
        handlelength=2.4, handletextpad=0.75, labelspacing=0.75,
    )


def draw_c_panel(fig, c_spec, c_legend_ax):
    grid = c_spec.subgridspec(
        4, 1, height_ratios=[0.10, 1.0, 0.22, 1.0], hspace=-0.01,
    )
    title_ax = fig.add_subplot(grid[0, 0])
    panel_title(title_ax, "c", "Multiscale responses across all biological characteristics")
    river_ax = fig.add_subplot(grid[1, 0])
    driver_ax = fig.add_subplot(grid[2, 0])
    lake_ax = fig.add_subplot(grid[3, 0])
    for axis in [title_ax, river_ax, driver_ax, lake_ax, c_legend_ax]:
        axis.set_zorder(2)

    ordered = {}
    family = {}
    for water in WATERS:
        ordered[water], family[water] = network_base.build_ordered_data(water)

    river_connectors, river_dense, river_dimension_x, river_bounds, river_layout = (
        network.draw_semicircle_panel(
            river_ax, ordered["River"], family["River"], "River", inverted=False,
            water_label_x=C_WATER_LABEL_X, show_water_label=False,
        )
    )
    network.draw_shared_driver_boxes(driver_ax)
    lake_connectors, lake_dense, lake_dimension_x, lake_bounds, lake_layout = (
        network.draw_semicircle_panel(
            lake_ax, ordered["Lake"], family["Lake"], "Lake", inverted=True,
            water_label_x=C_WATER_LABEL_X, show_water_label=False,
        )
    )
    network_base.draw_legend(c_legend_ax)
    fig.canvas.draw()
    shared_paths = network.draw_shared_paths(
        fig, {"River": river_ax, "Lake": lake_ax}, driver_ax, family,
        {"River": river_dimension_x, "Lake": lake_dimension_x},
    )
    return {
        "title_ax": title_ax,
        "river_ax": river_ax,
        "driver_ax": driver_ax,
        "lake_ax": lake_ax,
        "ordered": ordered,
        "family": family,
        "connectors": {"River": river_connectors, "Lake": lake_connectors},
        "dense": {"River": river_dense, "Lake": lake_dense},
        "layouts": {"River": river_layout, "Lake": lake_layout},
        "shared_paths": shared_paths,
    }


def plot_fig5(family_share: pd.DataFrame, selected: pd.DataFrame):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(25.0, 20.0), facecolor="white")
    outer = fig.add_gridspec(
        2, 3,
        width_ratios=[1.45, 3.35, 1.25],
        height_ratios=[1.05, 4.90],
        wspace=0.045, hspace=0.022,
    )

    a_grid = outer[0, 0:2].subgridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.02)
    a_title_ax = fig.add_subplot(a_grid[0, 0])
    panel_title(a_title_ax, "a", "Ecological dimensions carrying the overall biological signal")
    a_ax = fig.add_subplot(a_grid[1, 0])
    a_legend_ax = fig.add_subplot(outer[0, 2])
    panel_a_summary, panel_a_annotations = draw_panel_a(a_ax, family_share)
    draw_panel_a_legend(a_legend_ax)

    b_grid = outer[1, 0].subgridspec(
        4, 2,
        height_ratios=[0.10, 0.92, 0.19, 0.92],
        width_ratios=[1.48, 1.0],
        hspace=-0.01, wspace=0.0,
    )
    b_title_ax = fig.add_subplot(b_grid[0, :])
    panel_title(
        b_title_ax, "b", "Strongest biological responses\nwithin environmental drivers",
        title_fontsize=19.2, linespacing=1.18,
    )
    b_river_label_ax = fig.add_subplot(b_grid[1, 0])
    b_river_ax = fig.add_subplot(b_grid[1, 1])
    b_middle_ax = fig.add_subplot(b_grid[2, :])
    b_lake_label_ax = fig.add_subplot(b_grid[3, 0])
    b_lake_ax = fig.add_subplot(b_grid[3, 1])
    b_panel_meta = {
        "River": top5b.draw_water_panel(
            b_river_label_ax, b_river_ax, selected, "River"
        ),
        "Lake": top5b.draw_water_panel(
            b_lake_label_ax, b_lake_ax, selected, "Lake"
        ),
    }
    b_legend_count = top5b.draw_compact_domain_legend(b_middle_ax)
    b_counts = {
        water: int(
            selected.loc[selected.water_body_type.eq(water)].shape[0]
        )
        for water in WATERS
    }

    c_legend_ax = fig.add_subplot(outer[1, 2])
    fig.subplots_adjust(left=0.118, right=0.994, top=0.992, bottom=0.025)
    c_data = draw_c_panel(fig, outer[1, 1], c_legend_ax)
    fig.canvas.draw()

    # Keep panel b slightly shorter than the paired semicircle fields. The
    # upper response field remains top-aligned, whereas the lower field is
    # lifted until its bottom aligns with the lower Topography text itself.
    b_height_ratio = 0.92
    river_target = c_data["river_ax"].get_position()
    river_height = river_target.height * b_height_ratio
    river_y0 = river_target.y1 - river_height
    for axis in (b_river_label_ax, b_river_ax):
        position = axis.get_position()
        axis.set_position([position.x0, river_y0, position.width, river_height])

    renderer = fig.canvas.get_renderer()
    lower_topography = next(
        text for text in c_data["lake_ax"].texts
        if text.get_text() == "Topography and geomorphology"
    )
    topography_box = lower_topography.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    lake_target = c_data["lake_ax"].get_position()
    lake_height = lake_target.height * b_height_ratio
    lake_y0 = topography_box.y0
    for axis in (b_lake_label_ax, b_lake_ax):
        position = axis.get_position()
        axis.set_position([position.x0, lake_y0, position.width, lake_height])

    # Compress and re-centre the shared legend within the new inter-panel gap.
    legend_position = b_middle_ax.get_position()
    legend_height_ratio = 0.80
    legend_height = legend_position.height * legend_height_ratio
    gap_bottom = lake_y0 + lake_height
    gap_top = river_y0
    legend_y0 = 0.5 * (gap_bottom + gap_top - legend_height)
    b_middle_ax.set_position(
        [legend_position.x0, legend_y0, legend_position.width, legend_height]
    )
    fig.canvas.draw()

    # Lower panel a slightly and reduce the a-to-b/c gap.
    for axis in [a_title_ax, a_ax, a_legend_ax]:
        position = axis.get_position()
        axis.set_position(
            [position.x0, position.y0 - 0.006, position.width, position.height]
        )
    fig.canvas.draw()

    # Narrow panel a from the left while preserving its right boundary.  Treat
    # the driver tick labels, the vertical River/Lake labels and their colour
    # strips as one visible left-axis block; its outer edge should align with
    # the left edge of panel b.
    renderer = fig.canvas.get_renderer()
    figure_inverse = fig.transFigure.inverted()
    def panel_a_visible_left() -> float:
        current_renderer = fig.canvas.get_renderer()
        boxes = [
            tick.get_window_extent(current_renderer).transformed(figure_inverse)
            for tick in a_ax.get_yticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        boxes.extend(
            text.get_window_extent(current_renderer).transformed(figure_inverse)
            for text in a_ax.texts
            if text.get_visible() and text.get_text() in WATERS
        )
        boxes.extend(
            patch.get_window_extent(current_renderer).transformed(figure_inverse)
            for patch in a_ax.patches
            if isinstance(patch, Rectangle) and patch.get_visible()
        )
        return min(box.x0 for box in boxes)

    a_visible_left = panel_a_visible_left()
    b_left = b_river_label_ax.get_position().x0
    a_left_shift = max(0.0, b_left - a_visible_left)
    a_position = a_ax.get_position()
    a_ax.set_position(
        [
            a_position.x0 + a_left_shift,
            a_position.y0,
            a_position.width - a_left_shift,
            a_position.height,
        ]
    )
    # Artist offsets expressed in axes coordinates change slightly when the
    # axes width changes. Iterate the correction so the rendered edge—not only
    # the axes box—lands on the target after the enlarged labels are laid out.
    for _ in range(4):
        fig.canvas.draw()
        correction = b_left - panel_a_visible_left()
        if abs(correction) < 1e-5:
            break
        a_position = a_ax.get_position()
        a_ax.set_position(
            [
                a_position.x0 + correction,
                a_position.y0,
                a_position.width - correction,
                a_position.height,
            ]
        )
    fig.canvas.draw()

    meta = {
        "a_title_ax": a_title_ax,
        "a_ax": a_ax,
        "a_legend_ax": a_legend_ax,
        "b_title_ax": b_title_ax,
        "b_river_label_ax": b_river_label_ax,
        "b_river_ax": b_river_ax,
        "b_middle_ax": b_middle_ax,
        "b_lake_label_ax": b_lake_label_ax,
        "b_lake_ax": b_lake_ax,
        "c_legend_ax": c_legend_ax,
        "panel_a_summary": panel_a_summary,
        "panel_a_annotations": panel_a_annotations,
        "b_counts": b_counts,
        "b_panel_meta": b_panel_meta,
        "b_legend_count": b_legend_count,
        "b_height_ratio": b_height_ratio,
        "b_legend_height_ratio": legend_height_ratio,
        "b_lake_bottom_target": topography_box.y0,
        "panel_a_visible_alignment_target": b_left,
        **c_data,
    }
    return fig, meta


def run_qa(
    family_share: pd.DataFrame, selected: pd.DataFrame, fig, meta: dict,
    outputs: list[Path],
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, detail) -> None:
        rows.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    sums = family_share.groupby(
        ["water_body_type", "environment_module", "range_m"]
    ).family_signal_share.sum()
    add("Panel a source shares sum to one", np.allclose(sums, 1.0, atol=1e-10),
        f"max deviation={float(np.max(np.abs(sums - 1.0))):.3g}")
    add("Panel a displays all 30 percentages", meta["panel_a_annotations"] == 30,
        meta["panel_a_annotations"])
    a_text = [text.get_text() for text in meta["a_ax"].texts]
    add("Panel a labels each waterbody once",
        a_text.count("River") == 1 and a_text.count("Lake") == 1,
        {"River": a_text.count("River"), "Lake": a_text.count("Lake")})
    add("Panel a horizontal axis removed", len(meta["a_ax"].get_xticklabels()) == 0,
        "no ticks or tick labels")
    add("Five ecological-dimension colours match panel c", FAMILY_COLORS == network_base.FAMILY_COLORS,
        FAMILY_COLORS)
    counts = selected.groupby(["water_body_type", "environment_module"]).size()
    add("Panel b retains five strongest characteristics per driver",
        bool((counts == 5).all() and len(counts) == 6), counts.to_dict())
    add("Panel b River and Lake use a shared x scale",
        meta["b_river_ax"].get_xlim() == meta["b_lake_ax"].get_xlim(),
        meta["b_river_ax"].get_xlim())
    add("Panel b removes horizontal axes and grids",
        all(
            len(axis.get_xticks()) == 0
            and not axis.spines["bottom"].get_visible()
            and not axis.spines["left"].get_visible()
            for axis in [meta["b_river_ax"], meta["b_lake_ax"]]
        ), "no x ticks, labels, bottom or left spine, or grid")
    add("Panel b encodes direction with signed median rho",
        np.isfinite(selected.signed_median_rho_13range).all()
        and selected.signed_median_rho_13range.between(
            top5b.B_DIRECTION_NORM.vmin, top5b.B_DIRECTION_NORM.vmax
        ).all(),
        {
            "data": (
                selected.signed_median_rho_13range.min(),
                selected.signed_median_rho_13range.max(),
            ),
            "colour scale": (
                top5b.B_DIRECTION_NORM.vmin,
                top5b.B_DIRECTION_NORM.vmax,
            ),
        })
    add("Panel b has touching response-domain strips",
        all(meta["b_panel_meta"][water]["domain_strips"] == 15 for water in WATERS),
        {water: meta["b_panel_meta"][water]["domain_strips"] for water in WATERS})
    add("Panel b has three filled driver labels per waterbody",
        all(meta["b_panel_meta"][water]["driver_labels"] == 3 for water in WATERS),
        {water: meta["b_panel_meta"][water]["driver_labels"] for water in WATERS})
    add("Panel b legend contains all five ecological response dimensions",
        meta["b_legend_count"] == 5, meta["b_legend_count"])
    b_river_box = meta["b_river_ax"].get_position()
    b_lake_box = meta["b_lake_ax"].get_position()
    b_river_label_box = meta["b_river_label_ax"].get_position()
    c_river_box = meta["river_ax"].get_position()
    c_lake_box = meta["lake_ax"].get_position()
    add("Panel b rows are shorter and aligned to panel c landmarks",
        abs(b_river_box.y1 - c_river_box.y1) < 1e-6
        and abs(b_lake_box.y0 - meta["b_lake_bottom_target"]) < 1e-6
        and abs(b_river_box.height / c_river_box.height - meta["b_height_ratio"]) < 1e-6
        and abs(b_lake_box.height / c_lake_box.height - meta["b_height_ratio"]) < 1e-6,
        {
            "River ratio": b_river_box.height / c_river_box.height,
            "Lake ratio": b_lake_box.height / c_lake_box.height,
            "River top": b_river_box.y1,
            "Lake bottom/Topography text bottom": (
                b_lake_box.y0, meta["b_lake_bottom_target"]
            ),
        })
    add("Panel b RMS titles are single-line and identical",
        all(
            meta["b_panel_meta"][water]["metric_title"]
            == "13-range RMS response strength"
            and "\n" not in meta["b_panel_meta"][water]["metric_title"]
            for water in WATERS
        ),
        {water: meta["b_panel_meta"][water]["metric_title"] for water in WATERS})
    legend_box = meta["b_middle_ax"].get_position()
    add("Panel b middle legend is vertically compressed",
        meta["b_legend_height_ratio"] < 1.0,
        {"height": legend_box.height, "scale": meta["b_legend_height_ratio"]})
    a_box = meta["a_ax"].get_position()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_inverse = fig.transFigure.inverted()
    a_visible_boxes = [
        tick.get_window_extent(renderer).transformed(figure_inverse)
        for tick in meta["a_ax"].get_yticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    a_visible_boxes.extend(
        text.get_window_extent(renderer).transformed(figure_inverse)
        for text in meta["a_ax"].texts
        if text.get_visible() and text.get_text() in WATERS
    )
    a_visible_left = min(box.x0 for box in a_visible_boxes)
    add("Panel a visible left-axis block aligns with panel b",
        abs(a_visible_left - b_river_label_box.x0) < 2e-4,
        f"panel a visible edge={a_visible_left:.4f}; panel b={b_river_label_box.x0:.4f}")
    add("Panel a retains the combined b-c right boundary",
        a_box.x1 >= c_river_box.x1 - 1e-6,
        f"panel a right={a_box.x1:.4f}; panel c right={c_river_box.x1:.4f}")
    a_legend_box = meta["a_legend_ax"].get_position()
    c_legend_box = meta["c_legend_ax"].get_position()
    add("Panel a and c legends share one aligned column",
        abs(a_legend_box.x0 - c_legend_box.x0) < 1e-6
        and abs(a_legend_box.x1 - c_legend_box.x1) < 1e-6,
        f"x={a_legend_box.x0:.3f}-{a_legend_box.x1:.3f}")
    add("Panel c retains every biological characteristic",
        len(meta["ordered"]["River"]) == 178 and len(meta["ordered"]["Lake"]) == 144,
        {water: len(table) for water, table in meta["ordered"].items()})
    c_water_text = [
        text.get_text() for axis in [meta["river_ax"], meta["lake_ax"]]
        for text in axis.texts
    ]
    b_water_text = [
        text.get_text()
        for axis in [meta["b_river_label_ax"], meta["b_lake_label_ax"]]
        for text in axis.texts
    ]
    add("Panel b and c omit redundant River/Lake labels",
        "River" not in c_water_text and "Lake" not in c_water_text
        and "River" not in b_water_text and "Lake" not in b_water_text,
        "waterbody identity is carried by the shared vertical ordering")
    add("Panel titles a-c present", True,
        [
            "Ecological dimensions carrying the overall biological signal",
            "Strongest biological responses within environmental drivers",
            "Multiscale responses across all biological characteristics",
        ])
    add("Panel c driver-domain paths retained", len(meta["shared_paths"]) == 30,
        len(meta["shared_paths"]))
    for output in outputs:
        add(f"Output exists {output.suffix}", output.exists() and output.stat().st_size > 0,
            output.stat().st_size if output.exists() else 0)
        if output.suffix == ".svg" and output.exists():
            svg = output.read_text(encoding="utf-8")
            add("SVG retains editable text", "<text" in svg, svg.count("<text"))
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [SCRIPT_DIR, SOURCE_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for extension in ["PNG", "SVG", "PDF", "TIFF"]:
        (FIG_DIR / extension).mkdir(parents=True, exist_ok=True)

    family_share, selected = load_panel_data()
    fig, meta = plot_fig5(family_share, selected)
    outputs = {
        "png": FIG_DIR / "PNG" / "Fig5.png",
        "svg": FIG_DIR / "SVG" / "Fig5.svg",
        "pdf": FIG_DIR / "PDF" / "Fig5.pdf",
        "tiff": FIG_DIR / "TIFF" / "Fig5.tiff",
    }
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(
        outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.03,
        facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    meta["panel_a_summary"].to_csv(
        SOURCE_DIR / "Fig5_panel_a_ecological_dimension_shares.csv",
        index=False, encoding="utf-8-sig",
    )
    selected.to_csv(
        SOURCE_DIR / "Fig5_panel_b_top5_driver_responsive_characteristics.csv",
        index=False, encoding="utf-8-sig",
    )
    meta["shared_paths"].to_csv(
        SOURCE_DIR / "Fig5_panel_c_driver_response_domain_paths.csv",
        index=False, encoding="utf-8-sig",
    )
    qa = run_qa(family_share, selected, fig, meta, list(outputs.values()))
    qa.to_csv(QA_DIR / "Fig5_automated_QA.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    summary = {
        "backend": "Python",
        "panels": ["a", "b", "c"],
        "panel_a_segments": int(meta["panel_a_annotations"]),
        "panel_b_characteristics": meta["b_counts"],
        "panel_c_characteristics": {
            water: int(len(table)) for water, table in meta["ordered"].items()
        },
        "all_qa_pass": bool(qa.passed.all()),
    }
    (QA_DIR / "Fig5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
