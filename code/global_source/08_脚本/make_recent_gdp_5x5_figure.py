"""Build the recent-GDP 4-row x 5-column manuscript figure and atomic panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
UPDATE = ROOT / "11_近年GDP更新_5x5布局_2026-09-15"
CLEAN = UPDATE / "02_匹配数据"
ANALYSIS = UPDATE / "03_统计结果"
FIGURE = UPDATE / "04_图件"
ATOMIC = FIGURE / "子图"
SOURCE = UPDATE / "05_SourceData"
QA = UPDATE / "07_QA"
GDP_TIF = CLEAN / "gdppc_ppp2021_mean_2020_2022_30arcmin.tif"

OLD = ROOT.parent / "18_全球淡水经济发展与水生态宏观分析_2026-09-14"
WORLD_SHP = OLD / "01_原始数据" / "NaturalEarth_110m" / "ne_110m_admin_0_countries.shp"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import make_final_composite_figure as base  # noqa: E402


GDP_CMAP = base.GDP_CMAP
GDP_POINT_CMAP = base.GDP_POINT_CMAP
WATER_POINT_CMAP = base.WATER_POINT_CMAP
BIOTA_POINT_CMAP = base.BIOTA_POINT_CMAP
COLORS = base.COLORS
LABELS = base.ENDPOINT_LABELS

WATER = ["DOSAT", "EC", "TP", "NH4N", "NO3N"]
BIOTA = ["Fish", "EPTO", "Algae"]
FOREST_ORDER = BIOTA + WATER
FONT_SCALE = 1.5


def fs(points: float) -> float:
    """Apply the requested uniform 1.5× font enlargement."""
    return points * FONT_SCALE

def style() -> None:
    base.style()
    mpl.rcParams.update({
        "font.size": fs(6.3),
        "axes.labelsize": fs(6.1),
        "xtick.labelsize": fs(5.35),
        "ytick.labelsize": fs(5.35),
        "axes.linewidth": 0.55,
    })


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP).to_crs(4326)
    return world.loc[world["ADMIN"] != "Antarctica"].copy()


def load_inputs() -> dict[str, pd.DataFrame]:
    water = pd.read_csv(CLEAN / "water_site_medians_long.csv.gz")
    all_water = pd.read_csv(CLEAN / "water_all_unique_sites_gdppc.csv.gz")
    all_water["value"] = all_water["gdppc"]
    return {
        "water": water,
        "all_water": all_water,
        "Fish": pd.read_csv(CLEAN / "fish_units.csv"),
        "EPTO": pd.read_csv(CLEAN / "epto_units.csv"),
        "Algae": pd.read_csv(CLEAN / "algae_units.csv"),
        "points": pd.read_csv(SOURCE / "density_balanced_partial_residuals.csv.gz"),
        "curves": pd.read_csv(SOURCE / "density_balanced_curves.csv"),
        "trend": pd.read_csv(ANALYSIS / "density_balanced_selected_trend_summary.csv"),
        "linear": pd.read_csv(ANALYSIS / "natural_adjusted_linear_gdp_effects.csv"),
        "density": pd.read_csv(SOURCE / "recent_GDP_balanced_density.csv.gz"),
    }


def format_raw(value: float) -> str:
    return base.format_raw(value)


def robust_norm(values: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> Normalize:
    return base.robust_norm(values, lo=lo, hi=hi)


def map_base(ax: plt.Axes, world: gpd.GeoDataFrame) -> None:
    world.plot(ax=ax, color=COLORS["lightgray"], edgecolor=COLORS["border"], linewidth=0.18, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_axis_off()


def add_colorbar(fig: plt.Figure, ax: plt.Axes, mappable, label: str, inverse,
                 standalone: bool = False) -> None:
    width = "82%" if standalone else "88%"
    y_offset = -0.155 if standalone else -0.135
    height = "5.6%" if standalone else "5.5%"
    cax = inset_axes(
        ax, width=width, height=height, loc="lower center",
        bbox_to_anchor=(0, y_offset, 1, 1), bbox_transform=ax.transAxes, borderpad=0,
    )
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.locator = MaxNLocator(nbins=3)
    cb.update_ticks()
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_raw(float(inverse(x)))))
    cb.ax.tick_params(
        length=1.7, pad=1.0,
        labelsize=fs(6.7 if standalone else 5.05),
    )
    cb.outline.set_linewidth(0.35)
    cb.set_label(
        label, labelpad=1.25, fontsize=fs(7.1 if standalone else 5.35),
        fontweight="bold",
    )


def point_size(name: str, n: int, standalone: bool) -> float:
    multiplier = 2.2 if standalone else 1.0
    if name == "GDP_sites":
        return 0.11 * multiplier
    if name in WATER:
        return (0.13 if n > 20000 else 0.20) * multiplier
    if name in ("Fish", "EPTO"):
        return 1.75 * multiplier
    return 0.52 * multiplier


def plot_point_map(fig: plt.Figure, ax: plt.Axes, world: gpd.GeoDataFrame,
                   data: pd.DataFrame, name: str, cmap, fixed_norm: Normalize | None = None,
                   standalone: bool = False) -> tuple[pd.DataFrame, Normalize]:
    d = data.dropna(subset=["longitude", "latitude", "value"]).copy()
    transformed, inverse, label = base.transform_values(name, d["value"])
    if name == "GDP_sites":
        label = "GDPpc (2021 int$ PPP)"
    d["display_transformed_value"] = transformed
    norm = fixed_norm or robust_norm(transformed)
    map_base(ax, world)
    points = ax.scatter(
        d["longitude"], d["latitude"], c=transformed, cmap=cmap, norm=norm,
        s=point_size(name, len(d), standalone), alpha=0.85, linewidths=0,
        rasterized=True, zorder=2,
    )
    add_colorbar(fig, ax, points, label, inverse, standalone=standalone)
    d["layer"] = name
    d["display_transform"] = {
        "DOSAT": "identity", "EC": "log10", "TP": "log10", "NH4N": "log10",
        "NO3N": "log10", "GDP_sites": "log10", "Fish": "log1p", "EPTO": "log1p",
        "Algae": "log10 after mg/L to ug/L",
    }[name]
    return d, norm


def read_gdp_raster() -> tuple[np.ndarray, list[float], pd.DataFrame, Normalize]:
    with rasterio.open(GDP_TIF) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr <= 0] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        step = 2
        sampled = arr[::step, ::step]
        rows = np.arange(0, arr.shape[0], step)
        cols = np.arange(0, arr.shape[1], step)
        lon = np.array([rasterio.transform.xy(src.transform, 0, int(c), offset="center")[0] for c in cols])
        lat = np.array([rasterio.transform.xy(src.transform, int(r), 0, offset="center")[1] for r in rows])
    transformed = np.log10(arr)
    norm = robust_norm(transformed)
    xx, yy = np.meshgrid(lon, lat)
    source = pd.DataFrame({
        "longitude": xx.ravel(), "latitude": yy.ravel(), "gdppc": sampled.ravel(),
        "display_transformed_value": np.log10(sampled).ravel(), "display_transform": "log10",
    }).dropna(subset=["gdppc"])
    return transformed, extent, source, norm


def plot_gdp_grid(fig: plt.Figure, ax: plt.Axes, world: gpd.GeoDataFrame,
                  raster, standalone: bool = False) -> Normalize:
    transformed, extent, _, norm = raster
    map_base(ax, world)
    image = ax.imshow(
        transformed, extent=extent, origin="upper", cmap=GDP_CMAP, norm=norm,
        interpolation="nearest", rasterized=True, zorder=1,
    )
    world.boundary.plot(ax=ax, color=COLORS["border"], linewidth=0.18, zorder=2)
    add_colorbar(fig, ax, image, "GDPpc (2021 int$ PPP)", lambda x: 10 ** x, standalone=standalone)
    return norm


def pretty_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"={value:.3f}"


def log_tick(value, pos=None) -> str:
    if value <= 0:
        return ""
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def plot_association(ax: plt.Axes, endpoint: str, inputs: dict[str, pd.DataFrame],
                     color: str, fill_color: str, standalone: bool = False) -> None:
    points = inputs["points"].loc[inputs["points"]["endpoint"] == endpoint].sort_values("gdppc")
    curve = inputs["curves"].loc[inputs["curves"]["endpoint"] == endpoint].sort_values("gdppc")
    trend = inputs["trend"].loc[inputs["trend"]["endpoint"] == endpoint].iloc[0]
    large = len(points) > 5000
    ax.scatter(
        points["gdppc"], points["partial_response"],
        s=(0.40 if large else 2.2) * (1.6 if standalone else 1.0),
        color=COLORS["gray"], alpha=0.06 if large else 0.24,
        linewidths=0, rasterized=True, zorder=1,
    )
    ax.fill_between(
        curve["gdppc"], curve["ci_low"], curve["ci_high"],
        color=fill_color, alpha=0.32, linewidth=0, zorder=2,
    )
    ax.plot(curve["gdppc"], curve["predicted_partial"], color=color,
            lw=2.0 if standalone else 1.45, zorder=3)
    ax.axhline(0, color="#AAB2BB", lw=0.45, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(float(curve["gdppc"].min()), float(curve["gdppc"].max()))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick))
    ax.set_xlabel(
        "GDPpc (2021 int$ PPP)", labelpad=1.3,
        fontsize=fs(7.4 if standalone else 5.65),
    )
    ax.set_ylabel(
        "Adjusted response (SD)", labelpad=1.8,
        fontsize=fs(7.4 if standalone else 5.65),
    )
    if standalone:
        annotation = (
            f"ΔP10–P90={trend['contrast_10_90']:+.2f} "
            f"[{trend['contrast_ci_low']:+.2f}, {trend['contrast_ci_high']:+.2f}]\n"
            f"q{pretty_p(float(trend['joint_q']))}; ΔR²={trend['weighted_delta_r_squared']:.3f}; "
            f"n={int(trend['n_observations']):,}"
        )
    else:
        annotation = (
            f"ΔP10–P90={trend['contrast_10_90']:+.2f}\n"
            f"95% CI [{trend['contrast_ci_low']:+.2f}, {trend['contrast_ci_high']:+.2f}]\n"
            f"q{pretty_p(float(trend['joint_q']))}; ΔR²={trend['weighted_delta_r_squared']:.3f}; "
            f"n={int(trend['n_observations']):,}"
        )
    ax.text(
        0.018, 0.98, annotation, transform=ax.transAxes, ha="left", va="top",
        fontsize=fs(7.4 if standalone else 5.65), color=COLORS["ink"], linespacing=1.22,
        zorder=5,
    )
    ax.text(
        0.025, 0.055, LABELS[endpoint], transform=ax.transAxes,
        ha="left", va="bottom", fontsize=fs(8.2 if standalone else 6.15),
        fontweight="bold", color=color, zorder=6,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.35, alpha=0.65)


def plot_effect_forest(ax: plt.Axes, linear: pd.DataFrame, standalone: bool = False) -> None:
    data = linear.set_index("endpoint").loc[FOREST_ORDER].reset_index()
    y = np.arange(len(data))[::-1]
    low, high = float(data["ci_low"].min()), float(data["ci_high"].max())
    span = max(0.1, high - low)
    x_low, x_high = low - 0.10 * span, high + 0.10 * span
    ax.set_xlim(x_low, x_high)
    for yi, row in zip(y, data.to_dict("records")):
        color = COLORS["water"] if row["domain"] == "water_quality" else COLORS["biota"]
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color,
                lw=3.2 if standalone else 2.5, solid_capstyle="round")
        ax.scatter(row["estimate"], yi, s=28 if standalone else 17,
                   color=color, edgecolor="white", linewidth=0.55, zorder=3)
        label = (
            f"{LABELS[row['endpoint']]}   "
            f"p{pretty_p(float(row['p_value']))}; q{pretty_p(float(row['q_value']))}"
        )
        ax.text(
            float(row["ci_low"]), yi + 0.20, label, transform=ax.transData,
            ha="left", va="bottom", fontsize=fs(7.2 if standalone else 5.2),
            fontweight="bold", color=COLORS["ink"], clip_on=False,
        )
    ax.axvline(0, color="#8B949E", lw=0.75, ls="--", zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.42, len(data) - 0.22)
    ax.set_xlabel("Standardized GDP effect (95% CI)", labelpad=1.4 if standalone else 0.35,
                  fontsize=fs(7.4 if standalone else 5.55))
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.4, alpha=0.65)


def plot_gdp_density(ax: plt.Axes, density: pd.DataFrame, standalone: bool = False) -> None:
    data = density.dropna(subset=["gdppc", "weight"]).copy()
    weights = data["weight"].to_numpy(float) / data["endpoint"].nunique()
    log_gdp = np.log(data["gdppc"].to_numpy(float))
    log_grid = np.linspace(float(log_gdp.min()), float(log_gdp.max()), 320)
    grid = np.exp(log_grid)
    den = gaussian_kde(log_gdp, weights=weights)(log_grid)
    ax.fill_betweenx(grid, 0, den, color="#E88BB0", alpha=0.52, linewidth=0)
    ax.plot(den, grid, color=COLORS["gdp"], lw=1.6 if standalone else 1.25)
    ax.set_yscale("log")
    ax.set_ylim(grid.max(), grid.min())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
    ax.set_ylabel("GDPpc (2021 int$ PPP)", labelpad=1.4,
                  fontsize=fs(7.4 if standalone else 5.55))
    ax.set_xticks([])
    title_x = 0.16 if not standalone else 0.08
    density_title = "GDPpc distribution" if standalone else "GDPpc\ndistribution"
    ax.text(title_x, 0.96, density_title, transform=ax.transAxes,
            ha="left", va="top", fontsize=fs(7.6 if standalone else 5.35),
            linespacing=1.05,
            fontweight="bold", color=COLORS["ink"])
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def group_title(fig: plt.Figure, axes: list[plt.Axes], letter: str, text: str,
                offset: float = 0.010, x_override: float | None = None) -> None:
    boxes = [axis.get_position() for axis in axes]
    fig.text(
        x_override if x_override is not None else min(box.x0 for box in boxes),
        max(box.y1 for box in boxes) + offset,
        f"{letter}  {text}", ha="left", va="bottom", fontsize=fs(8.4),
        fontweight="bold", color=COLORS["ink"],
    )


def export(fig: plt.Figure, stem: str, folder: Path, tight: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.045} if tight else {}
    fig.savefig(folder / f"{stem}.png", dpi=600, **kwargs)
    fig.savefig(folder / f"{stem}.pdf", dpi=300, **kwargs)
    fig.savefig(folder / f"{stem}.svg", dpi=300, **kwargs)
    fig.savefig(
        folder / f"{stem}.tiff", dpi=600,
        pil_kwargs={"compression": "tiff_lzw"}, **kwargs,
    )


def make_composite(world: gpd.GeoDataFrame, inputs: dict[str, pd.DataFrame], raster) -> list[pd.DataFrame]:
    fig = plt.figure(figsize=(12.2, 7.35))
    gs = fig.add_gridspec(
        4, 5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.20, 1.0, 1.20],
        left=0.040, right=0.995, bottom=0.077, top=0.942,
        wspace=0.16, hspace=0.43,
    )
    top_map_grid = gs[0, :].subgridspec(1, 5, wspace=0.06)
    ax_gdp_grid = fig.add_subplot(top_map_grid[0, 0])
    ax_gdp_sites = fig.add_subplot(top_map_grid[0, 1])
    b_grid = gs[1, 0:2].subgridspec(1, 2, width_ratios=[1.42, 0.58], wspace=0.25)
    ax_forest = fig.add_subplot(b_grid[0, 0])
    ax_density = fig.add_subplot(b_grid[0, 1])
    bio_maps = [fig.add_subplot(top_map_grid[0, column]) for column in range(2, 5)]
    bio_model_grid = gs[1, 2:5].subgridspec(
        1, 4, width_ratios=[0.05, 1, 1, 1], wspace=0.14,
    )
    bio_models = [fig.add_subplot(bio_model_grid[0, column]) for column in range(1, 4)]
    water_map_grid = gs[2, :].subgridspec(1, 5, wspace=0.06)
    water_maps = [fig.add_subplot(water_map_grid[0, column]) for column in range(5)]
    water_model_grid = gs[3, :].subgridspec(
        1, 6, width_ratios=[0.08, 1, 1, 1, 1, 1], wspace=0.14,
    )
    water_models = [fig.add_subplot(water_model_grid[0, column]) for column in range(1, 6)]

    # Add a little breathing room below the b/d row without reopening all
    # inter-row gaps.  Moving both lower rows preserves the e-to-f spacing.
    lower_rows_shift = 0.010
    for axis in water_maps + water_models:
        box = axis.get_position()
        axis.set_position([box.x0, box.y0 - lower_rows_shift, box.width, box.height])

    map_sources: list[pd.DataFrame] = []
    gdp_norm = plot_gdp_grid(fig, ax_gdp_grid, world, raster)
    source, _ = plot_point_map(
        fig, ax_gdp_sites, world, inputs["all_water"], "GDP_sites", GDP_POINT_CMAP,
        fixed_norm=gdp_norm,
    )
    map_sources.append(source)
    plot_effect_forest(ax_forest, inputs["linear"])
    plot_gdp_density(ax_density, inputs["density"])

    for ax_map, ax_model, endpoint in zip(bio_maps, bio_models, BIOTA):
        source, _ = plot_point_map(fig, ax_map, world, inputs[endpoint], endpoint, BIOTA_POINT_CMAP)
        map_sources.append(source)
        plot_association(ax_model, endpoint, inputs, COLORS["biota"], COLORS["biota_fill"])

    for ax_map, ax_model, endpoint in zip(water_maps, water_models, WATER):
        frame = inputs["water"].loc[inputs["water"]["endpoint"] == endpoint]
        source, _ = plot_point_map(fig, ax_map, world, frame, endpoint, WATER_POINT_CMAP)
        map_sources.append(source)
        plot_association(ax_model, endpoint, inputs, COLORS["water"], COLORS["water_fill"])

    # Use one external y-axis title per response group. A compact spacer at
    # the start of each nested grid accommodates the title without extending
    # left of the aligned group heading; repeated titles are removed.
    bio_models[0].yaxis.set_label_coords(-0.14, 0.5)
    water_models[0].yaxis.set_label_coords(-0.14, 0.5)
    for axis in bio_models[1:] + water_models[1:]:
        axis.set_ylabel("")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def align_ylabel_left(axis: plt.Axes, target_x: float) -> None:
        """Align the rendered ylabel's left edge to a figure x coordinate."""
        label = axis.yaxis.label
        label_box = label.get_window_extent(renderer=renderer).transformed(
            fig.transFigure.inverted()
        )
        current_x, current_y = label.get_position()
        shift_axes = (target_x - label_box.x0) / axis.get_position().width
        axis.yaxis.set_label_coords(current_x + shift_axes, current_y)

    bio_group_x = min(axis.get_position().x0 for axis in bio_maps)
    water_group_x = min(axis.get_position().x0 for axis in water_maps)
    align_ylabel_left(bio_models[0], bio_group_x)
    align_ylabel_left(water_models[0], water_group_x)
    fig.canvas.draw()
    group_title(fig, [ax_gdp_grid, ax_gdp_sites], "a", "Recent economic context")
    group_title(fig, [ax_forest, ax_density], "b", "Adjusted GDP effects", offset=0.008)
    group_title(fig, bio_maps, "c", "Aquatic-biota observations")
    group_title(
        fig, bio_models, "d", "Biotic responses",
        x_override=bio_group_x,
    )
    group_title(fig, water_maps, "e", "Water-quality observations")
    group_title(
        fig, water_models, "f", "Water-quality responses",
        x_override=water_group_x,
    )
    export(fig, "Fig_recent_GDPpc_global_freshwater_4x5", FIGURE)
    plt.close(fig)
    return map_sources


def export_atomic_panels(world: gpd.GeoDataFrame, inputs: dict[str, pd.DataFrame], raster) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    gdp_norm = plot_gdp_grid(fig, ax, world, raster, standalone=True)
    export(fig, "01_GDPpc_grid_2020_2022", ATOMIC, tight=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    plot_point_map(fig, ax, world, inputs["all_water"], "GDP_sites", GDP_POINT_CMAP,
                   fixed_norm=gdp_norm, standalone=True)
    export(fig, "02_GDPpc_water_sites", ATOMIC, tight=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    plot_effect_forest(ax, inputs["linear"], standalone=True)
    ax.set_title(
        "Background-adjusted GDP effects", loc="left",
        fontsize=fs(9.0), fontweight="bold",
    )
    export(fig, "03_adjusted_GDP_effects", ATOMIC, tight=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.7, 5.2))
    plot_gdp_density(ax, inputs["density"], standalone=True)
    export(fig, "04_endpoint_balanced_GDPpc_density", ATOMIC, tight=True)
    plt.close(fig)

    number = 5
    for endpoint in BIOTA + WATER:
        fig, ax = plt.subplots(figsize=(5.1, 3.0))
        if endpoint in BIOTA:
            frame, cmap = inputs[endpoint], BIOTA_POINT_CMAP
        else:
            frame = inputs["water"].loc[inputs["water"]["endpoint"] == endpoint]
            cmap = WATER_POINT_CMAP
        plot_point_map(fig, ax, world, frame, endpoint, cmap, standalone=True)
        export(fig, f"{number:02d}_{endpoint}_global_map", ATOMIC, tight=True)
        plt.close(fig)
        number += 1

        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        color = COLORS["biota"] if endpoint in BIOTA else COLORS["water"]
        fill = COLORS["biota_fill"] if endpoint in BIOTA else COLORS["water_fill"]
        plot_association(ax, endpoint, inputs, color, fill, standalone=True)
        export(fig, f"{number:02d}_{endpoint}_GDP_response", ATOMIC, tight=True)
        plt.close(fig)
        number += 1


def text_overlap_audit(image_stem: str) -> None:
    # The detailed visual inspection is completed on the rendered PNG.  This
    # machine-readable record fixes the intended grid and review checklist.
    record = {
        "figure": image_stem,
        "layout": "4 rows x 5 columns; coordinates interpreted as (column, row)",
        "font_scale_relative_to_previous": 1.5,
        "rendered_png_pixels": [7320, 4410],
        "map_column_spacing": "compact nested map grids (wspace=0.06)",
        "model_endpoint_labels": "inside lower-left of each model axis",
        "annotation_background": "none",
        "groups": {
            "a": "row 1, columns 1-2: two GDP maps",
            "b": "row 2, columns 1-2: effect forest and GDP density",
            "c": "row 1, columns 3-5: three biota maps",
            "d": "row 2, columns 3-5: three equal-width biota models",
            "e": "row 3, columns 1-5: five water-quality maps",
            "f": "row 4, columns 1-5: five water-quality models with the same width as biota models",
        },
        "manual_QA_required": [
            "group titles do not overlap axes or colourbars",
            "map panels contain no internal subtitles; colourbar titles identify variables",
            "model annotations remain inside axes",
            "forest labels remain anchored above their corresponding 95% CI segments",
            "transparent annotations remain legible over plotted observations",
            "all confidence bands span the plotted GDP support",
            "axis labels and tick labels are legible at final output size",
        ],
    }
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "figure_layout_and_overlap_checklist.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run() -> None:
    style()
    for folder in (FIGURE, ATOMIC, SOURCE, QA):
        folder.mkdir(parents=True, exist_ok=True)
    world = load_world()
    inputs = load_inputs()
    raster = read_gdp_raster()
    map_sources = make_composite(world, inputs, raster)
    export_atomic_panels(world, inputs, raster)
    raster[2].to_csv(SOURCE / "map_GDPpc_grid_display_sample.csv.gz", index=False, compression="gzip")
    for frame in map_sources:
        name = str(frame["layer"].iloc[0])
        frame.to_csv(SOURCE / f"map_{name}_points.csv.gz", index=False, compression="gzip")
    text_overlap_audit("Fig_recent_GDPpc_global_freshwater_4x5")
    print(json.dumps({
        "main_figure": str(FIGURE / "Fig_recent_GDPpc_global_freshwater_4x5.png"),
        "atomic_folder": str(ATOMIC),
        "n_atomic_png": len(list(ATOMIC.glob("*.png"))),
    }, ensure_ascii=False))


if __name__ == "__main__":
    run()
