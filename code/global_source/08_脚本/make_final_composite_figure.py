"""Create the revised Nature-style global freshwater macroecology figure.

The eight maps are observational: only the GDP raster is continuous, whereas
site/unit data are shown as independent coloured points over administrative
boundaries. The lower panels show natural-background-adjusted GDP effects and
population-standardized marginal predictions with full-covariance 95% CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "03_清洗与匹配"
ANALYSIS = ROOT / "04_统计分析"
FIG = ROOT / "05_最终图件"
INDIVIDUAL = FIG / "独立子图"
ATOMIC = INDIVIDUAL / "单图"
SUPPLEMENT_ATOMIC = ATOMIC / "补充材料"
SOURCE = ROOT / "06_SourceData"
QA = ROOT / "09_QA"
OLD = ROOT.parent / "18_全球淡水经济发展与水生态宏观分析_2026-09-14"
GDP_TIF = OLD / "03_清洗与匹配" / "gdppc_mean_2010_2015_5arcmin.tif"
WORLD_SHP = OLD / "01_原始数据" / "NaturalEarth_110m" / "ne_110m_admin_0_countries.shp"

GDP_CMAP = LinearSegmentedColormap.from_list(
    "white_pink", ["#FFFFFF", "#FBE8EF", "#F5B9CE", "#E56B9F", "#B51659"]
)
WATER_CMAP = LinearSegmentedColormap.from_list(
    "yellow_brown", ["#FFFDF1", "#F7E6A7", "#D9A441", "#9B6416", "#633A0B"]
)
BIOTA_CMAP = LinearSegmentedColormap.from_list(
    "biota_green", ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#075B2C"]
)


def truncate_colormap(cmap, minimum: float = 0.25):
    """Remove the palest quarter so low-valued points remain visible."""
    return LinearSegmentedColormap.from_list(
        f"{cmap.name}_upper75", cmap(np.linspace(minimum, 1.0, 256))
    )


GDP_POINT_CMAP = truncate_colormap(GDP_CMAP)
WATER_POINT_CMAP = truncate_colormap(WATER_CMAP)
BIOTA_POINT_CMAP = truncate_colormap(BIOTA_CMAP)

COLORS = {
    "gdp": "#C62E6B",
    "water": "#9B6416",
    "water_fill": "#E9C66D",
    "biota": "#238B45",
    "biota_fill": "#8FD19A",
    "gray": "#64748B",
    "lightgray": "#F2F3F4",
    "border": "#AEB5BC",
    "grid": "#D9DEE3",
    "ink": "#17212B",
}

ENDPOINT_LABELS = {
    "DOSAT": "DO saturation",
    "EC": "EC",
    "TP": "TP",
    "Fish": "Fish richness",
    "EPTO": "EPTO richness",
    "Algae": "Chlorophyll a",
    "NH4N": r"NH$_4$–N",
    "NO3N": r"NO$_3$–N",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 5.8,
        "axes.titlesize": 6.4,
        "axes.labelsize": 5.6,
        "xtick.labelsize": 4.8,
        "ytick.labelsize": 4.8,
        "legend.fontsize": 4.8,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
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
        "points": pd.read_csv(SOURCE / "Fig_main_density_balanced_partial_residuals.csv.gz"),
        "curves": pd.read_csv(SOURCE / "Fig_main_density_balanced_curves.csv"),
        "summary": pd.read_csv(ANALYSIS / "primary_model_summary.csv"),
        "linear": pd.read_csv(ANALYSIS / "natural_adjusted_linear_gdp_effects.csv"),
        "trend": pd.read_csv(ANALYSIS / "density_balanced_selected_trend_summary.csv"),
        "density": pd.read_csv(SOURCE / "Fig_main_balanced_gdp_density.csv.gz"),
    }


def load_alternative_inputs() -> dict[str, pd.DataFrame]:
    """Inputs for the main-figure variant replacing EC/TP with NH4-N/NO3-N."""
    water = pd.read_csv(CLEAN / "alternative_NH4N_NO3N_water_site_medians_long.csv.gz")
    all_water = pd.read_csv(CLEAN / "alternative_NH4N_NO3N_all_unique_sites_gdppc.csv.gz")
    all_water["value"] = all_water["gdppc"]
    return {
        "water": water,
        "all_water": all_water,
        "Fish": pd.read_csv(CLEAN / "fish_units.csv"),
        "EPTO": pd.read_csv(CLEAN / "epto_units.csv"),
        "Algae": pd.read_csv(CLEAN / "algae_units.csv"),
        "points": pd.read_csv(SOURCE / "Fig_alt_NH4N_NO3N_density_balanced_partial_residuals.csv.gz"),
        "curves": pd.read_csv(SOURCE / "Fig_alt_NH4N_NO3N_density_balanced_curves.csv"),
        "linear": pd.read_csv(ANALYSIS / "alternative_NH4N_NO3N_natural_adjusted_linear_gdp_effects.csv"),
        "trend": pd.read_csv(ANALYSIS / "alternative_NH4N_NO3N_density_balanced_trend_summary.csv"),
        "density": pd.read_csv(SOURCE / "Fig_alt_NH4N_NO3N_balanced_gdp_density.csv.gz"),
    }


def transform_values(name: str, values: pd.Series | np.ndarray) -> tuple[np.ndarray, callable, str]:
    """Return monotonic display transform, inverse, and raw-unit label."""
    v = np.asarray(values, dtype=float)
    if name in ("GDP_grid", "GDP_sites"):
        return np.log10(np.clip(v, 1, None)), lambda x: 10 ** x, "GDPpc (PPP)"
    if name == "DOSAT":
        return v, lambda x: x, "DO saturation (%)"
    if name == "EC":
        return np.log10(np.clip(v, 0.1, None)), lambda x: 10 ** x, r"EC ($\mu$S cm$^{-1}$)"
    if name == "TP":
        return np.log10(np.clip(v, 1e-5, None)), lambda x: 10 ** x, r"TP (mg L$^{-1}$)"
    if name == "NH4N":
        return np.log10(np.clip(v, 1e-5, None)), lambda x: 10 ** x, r"NH$_4$–N (mg L$^{-1}$)"
    if name == "NO3N":
        return np.log10(np.clip(v, 1e-5, None)), lambda x: 10 ** x, r"NO$_3$–N (mg L$^{-1}$)"
    if name == "Fish":
        return np.log1p(np.clip(v, 0, None)), lambda x: np.expm1(x), "Fish richness"
    if name == "EPTO":
        return np.log1p(np.clip(v, 0, None)), lambda x: np.expm1(x), "EPTO richness"
    if name == "Algae":
        ug = v * 1000.0
        return np.log10(np.clip(ug, 1e-4, None)), lambda x: 10 ** x, r"Chlorophyll a ($\mu$g L$^{-1}$)"
    if name in ("AllFreshVerts", "Amphibia", "Reptiles"):
        labels = {
            "AllFreshVerts": "Freshwater-vertebrate richness",
            "Amphibia": "Amphibian richness",
            "Reptiles": "Reptile richness",
        }
        return np.log1p(np.clip(v, 0, None)), lambda x: np.expm1(x), labels[name]
    raise KeyError(name)


def format_raw(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def robust_norm(values: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> Normalize:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    vmin, vmax = np.quantile(finite, [lo, hi])
    if not vmax > vmin:
        vmax = vmin + 1.0
    return Normalize(float(vmin), float(vmax), clip=True)


def map_base(ax: plt.Axes, world: gpd.GeoDataFrame) -> None:
    world.plot(ax=ax, color=COLORS["lightgray"], edgecolor=COLORS["border"], linewidth=0.20, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def add_horizontal_colorbar(fig: plt.Figure, ax: plt.Axes, mappable, label: str, inverse) -> None:
    cax = inset_axes(
        ax, width="84%", height="5.5%", loc="lower center",
        bbox_to_anchor=(0, -0.145, 1, 1), bbox_transform=ax.transAxes, borderpad=0,
    )
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.locator = MaxNLocator(nbins=3)
    cb.update_ticks()
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_raw(float(inverse(x)))))
    cb.ax.tick_params(length=1.4, pad=0.8, labelsize=4.1)
    cb.outline.set_linewidth(0.35)
    cb.set_label(label, labelpad=0.7, fontsize=4.4)


def point_size(name: str, n: int, standalone: bool = False) -> float:
    scale = 1.65 if standalone else 1.0
    if name == "GDP_sites":
        return 0.12 * scale
    if name in ("DOSAT", "EC", "TP", "NH4N", "NO3N"):
        return (0.18 if n > 20000 else 0.28) * scale
    if name in ("Fish", "EPTO"):
        return 2.2 * scale
    return 0.75 * scale


def plot_point_map(fig: plt.Figure, ax: plt.Axes, world: gpd.GeoDataFrame,
                   data: pd.DataFrame, name: str, cmap, fixed_norm: Normalize | None = None,
                   standalone: bool = False) -> tuple[pd.DataFrame, Normalize]:
    d = data.dropna(subset=["longitude", "latitude", "value"]).copy()
    transformed, inverse, label = transform_values(name, d["value"])
    d["display_transformed_value"] = transformed
    norm = fixed_norm or robust_norm(transformed)
    map_base(ax, world)
    points = ax.scatter(
        d["longitude"], d["latitude"], c=transformed,
        cmap=cmap, norm=norm, s=point_size(name, len(d), standalone),
        alpha=0.82, linewidths=0, rasterized=True, zorder=2,
    )
    add_horizontal_colorbar(fig, ax, points, label, inverse)
    d["layer"] = name
    d["display_transform"] = {
        "DOSAT": "identity", "EC": "log10", "TP": "log10",
        "NH4N": "log10", "NO3N": "log10",
        "GDP_sites": "log10", "Fish": "log1p", "EPTO": "log1p",
        "Algae": "log10 after mg/L to ug/L",
    }.get(name, "log1p")
    return d, norm


def read_gdp_raster() -> tuple[np.ndarray, list[float], pd.DataFrame, Normalize]:
    with rasterio.open(GDP_TIF) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr <= 0] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        step = 4
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


def plot_gdp_raster(fig: plt.Figure, ax: plt.Axes, world: gpd.GeoDataFrame,
                    raster: tuple[np.ndarray, list[float], pd.DataFrame, Normalize]) -> Normalize:
    transformed, extent, _, norm = raster
    map_base(ax, world)
    image = ax.imshow(
        transformed, extent=extent, origin="upper", cmap=GDP_CMAP, norm=norm,
        interpolation="nearest", rasterized=True, zorder=1,
    )
    world.boundary.plot(ax=ax, color=COLORS["border"], linewidth=0.18, zorder=2)
    add_horizontal_colorbar(fig, ax, image, "GDPpc (PPP)", lambda x: 10 ** x)
    return norm


def pretty_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"={p:.3f}"


def log_tick(value, pos=None) -> str:
    if value <= 0:
        return ""
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def plot_association(ax: plt.Axes, endpoint: str, ylabel: str,
                     inputs: dict[str, pd.DataFrame], color: str, fill_color: str,
                     annotate: bool = True) -> None:
    p = inputs["points"].loc[inputs["points"]["endpoint"] == endpoint].sort_values("gdppc")
    c = inputs["curves"].loc[inputs["curves"]["endpoint"] == endpoint].sort_values("gdppc")
    trend = inputs["trend"].loc[inputs["trend"]["endpoint"] == endpoint].iloc[0]
    ax.scatter(
        p["gdppc"], p["partial_response"], s=0.34 if len(p) > 5000 else 2.0,
        color=COLORS["gray"], alpha=0.055 if len(p) > 5000 else 0.24,
        linewidths=0, rasterized=True, zorder=1,
    )
    ax.fill_between(c["gdppc"], c["ci_low"], c["ci_high"], color=fill_color, alpha=0.30, linewidth=0, zorder=2)
    ax.plot(c["gdppc"], c["predicted_partial"], color=color, lw=1.45, zorder=3)
    ax.axhline(0, color="#AAB2BB", lw=0.45, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(float(c["gdppc"].min()), float(c["gdppc"].max()))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick))
    ax.set_xlabel("GDPpc (PPP)", labelpad=1.0)
    ax.set_ylabel(ylabel + " (adj. SD)", labelpad=1.4)
    if annotate:
        annotation = (
            f"Δ10–90={trend['contrast_10_90']:+.2f} "
            f"[{trend['contrast_ci_low']:+.2f}, {trend['contrast_ci_high']:+.2f}]\n"
            f"q{pretty_p(float(trend['joint_q']))}; ΔR²={trend['weighted_delta_r_squared']:.3f}; "
            f"n={int(trend['n_observations']):,}"
        )
        ax.text(
            0.025, 0.97, annotation, transform=ax.transAxes, ha="left", va="top",
            fontsize=5.15, color=COLORS["ink"], linespacing=1.10,
            bbox={"facecolor": "white", "alpha": 0.74, "edgecolor": "none", "pad": 0.8},
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.35, alpha=0.65)


def plot_effect_forest(ax: plt.Axes, linear: pd.DataFrame, annotate: bool = True,
                       order: list[str] | None = None) -> None:
    order = order or ["DOSAT", "EC", "TP", "Fish", "EPTO", "Algae"]
    d = linear.set_index("endpoint").loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    for yi, row in zip(y, d.to_dict("records")):
        color = COLORS["water"] if row["domain"] == "water_quality" else COLORS["biota"]
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color, lw=2.7, solid_capstyle="round")
        ax.scatter(row["estimate"], yi, s=22, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(
            row["ci_low"], yi + 0.22, ENDPOINT_LABELS[row["endpoint"]],
            ha="left", va="bottom", fontsize=5.35, fontweight="bold",
            color=COLORS["ink"], clip_on=False,
        )
        if annotate:
            ax.text(0.985, yi,
                    f"p{pretty_p(float(row['p_value']))}; q{pretty_p(float(row['q_value']))}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=4.8, color=COLORS["ink"],
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 0.5})
    ax.axvline(0, color="#8B949E", lw=0.75, ls="--", zorder=0)
    low, high = float(d["ci_low"].min()), float(d["ci_high"].max())
    span = max(0.1, high - low)
    ax.set_xlim(low - 0.10 * span, high + 0.28 * span)
    ax.set_yticks([])
    ax.set_ylim(-0.35, len(d) - 0.38)
    ax.set_xlabel("GDP effect (95% CI)", labelpad=1.6, fontsize=5.1)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.4, alpha=0.65)


def plot_gdp_density(ax: plt.Axes, density: pd.DataFrame) -> None:
    d = density.dropna(subset=["gdppc", "weight"]).copy()
    weights = d["weight"].to_numpy(float) / d["endpoint"].nunique()
    log_gdp = np.log(d["gdppc"].to_numpy(float))
    log_grid = np.linspace(float(log_gdp.min()), float(log_gdp.max()), 320)
    grid = np.exp(log_grid)
    den = gaussian_kde(log_gdp, weights=weights)(log_grid)
    ax.fill_between(grid, 0, den, color="#E88BB0", alpha=0.52, linewidth=0)
    ax.plot(grid, den, color=COLORS["gdp"], lw=1.15)
    ax.set_xscale("log")
    ax.set_xlim(grid.min(), grid.max())
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick))
    ax.set_xlabel("GDPpc (PPP)", labelpad=1.5)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)


def group_title(fig: plt.Figure, axes: list[plt.Axes], letter: str, text: str,
                offset: float = 0.012, fontsize: float = 7.2) -> None:
    boxes = [ax.get_position() for ax in axes]
    x = min(b.x0 for b in boxes)
    y = max(b.y1 for b in boxes) + offset
    fig.text(x, y, f"{letter}  {text}", ha="left", va="bottom", fontsize=fontsize,
             fontweight="bold", color=COLORS["ink"])


def export(fig: plt.Figure, stem: str, folder: Path = FIG, tight: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.04} if tight else {}
    fig.savefig(folder / f"{stem}.png", dpi=600, **kwargs)
    fig.savefig(folder / f"{stem}.pdf", dpi=300, **kwargs)
    fig.savefig(folder / f"{stem}.svg", dpi=300, **kwargs)
    fig.savefig(folder / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kwargs)


def make_main() -> tuple[gpd.GeoDataFrame, dict[str, pd.DataFrame], tuple, list[pd.DataFrame]]:
    world = load_world()
    inputs = load_inputs()
    raster = read_gdp_raster()
    fig = plt.figure(figsize=(7.2, 5.05))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.15, 1.45],
        left=0.038, right=0.994, bottom=0.078, top=0.955, hspace=0.11,
    )
    maps_block = outer[0, 0].subgridspec(2, 1, hspace=0.28)
    maps_top = maps_block[0, 0].subgridspec(1, 4, wspace=0.08)
    maps_second = maps_block[1, 0].subgridspec(1, 4, wspace=0.08)
    lower = outer[1, 0].subgridspec(2, 4, height_ratios=[1, 1], wspace=0.29, hspace=0.54)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(maps_top[0, i]) for i in range(4)]
    ax_e, ax_f, ax_g, ax_h = [fig.add_subplot(maps_second[0, i]) for i in range(4)]
    dgrid = lower[:, 0].subgridspec(2, 1, height_ratios=[0.74, 0.26], hspace=0.42)
    ax_forest, ax_density = fig.add_subplot(dgrid[0, 0]), fig.add_subplot(dgrid[1, 0])
    ax_w = [fig.add_subplot(lower[0, i]) for i in range(1, 4)]
    ax_bio = [fig.add_subplot(lower[1, i]) for i in range(1, 4)]

    gdp_norm = plot_gdp_raster(fig, ax_a, world, raster)
    observed_sources: list[pd.DataFrame] = []
    for ax, endpoint in zip([ax_b, ax_c, ax_d], ["DOSAT", "EC", "TP"]):
        source, _ = plot_point_map(fig, ax, world,
                                   inputs["water"].loc[inputs["water"]["endpoint"] == endpoint],
                                   endpoint, WATER_POINT_CMAP)
        observed_sources.append(source)
    source, _ = plot_point_map(fig, ax_e, world, inputs["all_water"], "GDP_sites", GDP_POINT_CMAP, fixed_norm=gdp_norm)
    observed_sources.append(source)
    for ax, endpoint in zip([ax_f, ax_g, ax_h], ["Fish", "EPTO", "Algae"]):
        source, _ = plot_point_map(fig, ax, world, inputs[endpoint], endpoint, BIOTA_POINT_CMAP)
        observed_sources.append(source)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_anchor("N")
    for ax in (ax_e, ax_f, ax_g, ax_h):
        ax.set_anchor("N")

    plot_effect_forest(ax_forest, inputs["linear"])
    plot_gdp_density(ax_density, inputs["density"])
    for ax, endpoint in zip(ax_w, ["DOSAT", "EC", "TP"]):
        plot_association(ax, endpoint, ENDPOINT_LABELS[endpoint], inputs, COLORS["water"], COLORS["water_fill"])
    for ax, endpoint in zip(ax_bio, ["Fish", "EPTO", "Algae"]):
        plot_association(ax, endpoint, ENDPOINT_LABELS[endpoint], inputs, COLORS["biota"], COLORS["biota_fill"])

    fig.canvas.draw()
    group_title(fig, [ax_a], "a", "Global economic context")
    group_title(fig, [ax_b, ax_c, ax_d], "b", "Global water-quality observations")
    group_title(fig, [ax_f, ax_g, ax_h], "c", "Global aquatic-biota observations", offset=0.003)
    group_title(fig, [ax_forest], "d", "Background-adjusted GDP effects", offset=0.012, fontsize=6.7)
    group_title(fig, ax_w, "e", "Water-quality responses", offset=0.012, fontsize=6.7)
    group_title(fig, ax_bio, "f", "Aquatic-biota responses", offset=0.012, fontsize=6.7)
    export(fig, "Fig_main_global_freshwater_economy_ecology")
    export(fig, "Fig_main_global_freshwater_economy_ecology_EC_TP")
    plt.close(fig)

    raster[2].to_csv(SOURCE / "Fig_main_GDPpc_display_grid.csv.gz", index=False, compression="gzip")
    common_cols = ["layer", "longitude", "latitude", "country_code", "gdppc", "value", "unit_id",
                   "display_transformed_value", "display_transform"]
    observed = pd.concat(observed_sources, ignore_index=True, sort=False)
    observed[[c for c in common_cols if c in observed.columns]].to_csv(
        SOURCE / "Fig_main_point_map_observations.csv.gz", index=False, compression="gzip")
    return world, inputs, raster, observed_sources


def make_alternative_main(world: gpd.GeoDataFrame, raster: tuple) -> None:
    """Create the full NH4-N/NO3-N alternative without overwriting EC/TP."""
    inputs = load_alternative_inputs()
    water_endpoints = ["DOSAT", "NH4N", "NO3N"]
    effect_order = ["DOSAT", "NH4N", "NO3N", "Fish", "EPTO", "Algae"]
    fig = plt.figure(figsize=(7.2, 5.05))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.15, 1.45],
        left=0.038, right=0.994, bottom=0.078, top=0.955, hspace=0.11,
    )
    maps_block = outer[0, 0].subgridspec(2, 1, hspace=0.28)
    maps_top = maps_block[0, 0].subgridspec(1, 4, wspace=0.08)
    maps_second = maps_block[1, 0].subgridspec(1, 4, wspace=0.08)
    lower = outer[1, 0].subgridspec(
        2, 4, height_ratios=[1, 1], wspace=0.29, hspace=0.54
    )
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(maps_top[0, i]) for i in range(4)]
    ax_e, ax_f, ax_g, ax_h = [fig.add_subplot(maps_second[0, i]) for i in range(4)]
    dgrid = lower[:, 0].subgridspec(2, 1, height_ratios=[0.74, 0.26], hspace=0.42)
    ax_forest, ax_density = fig.add_subplot(dgrid[0, 0]), fig.add_subplot(dgrid[1, 0])
    ax_w = [fig.add_subplot(lower[0, i]) for i in range(1, 4)]
    ax_bio = [fig.add_subplot(lower[1, i]) for i in range(1, 4)]

    gdp_norm = plot_gdp_raster(fig, ax_a, world, raster)
    observed_sources: list[pd.DataFrame] = []
    for ax, endpoint in zip([ax_b, ax_c, ax_d], water_endpoints):
        source, _ = plot_point_map(
            fig, ax, world,
            inputs["water"].loc[inputs["water"]["endpoint"] == endpoint],
            endpoint, WATER_POINT_CMAP,
        )
        observed_sources.append(source)
    source, _ = plot_point_map(
        fig, ax_e, world, inputs["all_water"], "GDP_sites",
        GDP_POINT_CMAP, fixed_norm=gdp_norm,
    )
    observed_sources.append(source)
    for ax, endpoint in zip([ax_f, ax_g, ax_h], ["Fish", "EPTO", "Algae"]):
        source, _ = plot_point_map(
            fig, ax, world, inputs[endpoint], endpoint, BIOTA_POINT_CMAP
        )
        observed_sources.append(source)
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h):
        ax.set_anchor("N")

    plot_effect_forest(ax_forest, inputs["linear"], order=effect_order)
    plot_gdp_density(ax_density, inputs["density"])
    for ax, endpoint in zip(ax_w, water_endpoints):
        plot_association(
            ax, endpoint, ENDPOINT_LABELS[endpoint], inputs,
            COLORS["water"], COLORS["water_fill"],
        )
    for ax, endpoint in zip(ax_bio, ["Fish", "EPTO", "Algae"]):
        plot_association(
            ax, endpoint, ENDPOINT_LABELS[endpoint], inputs,
            COLORS["biota"], COLORS["biota_fill"],
        )

    fig.canvas.draw()
    group_title(fig, [ax_a], "a", "Global economic context")
    group_title(fig, [ax_b, ax_c, ax_d], "b", "Global water-quality observations")
    group_title(fig, [ax_f, ax_g, ax_h], "c", "Global aquatic-biota observations", offset=0.003)
    group_title(fig, [ax_forest], "d", "Background-adjusted GDP effects", offset=0.012, fontsize=6.7)
    group_title(fig, ax_w, "e", "Water-quality responses", offset=0.012, fontsize=6.7)
    group_title(fig, ax_bio, "f", "Aquatic-biota responses", offset=0.012, fontsize=6.7)
    export(fig, "Fig_main_global_freshwater_economy_ecology_NH4N_NO3N")
    plt.close(fig)

    observed = pd.concat(observed_sources, ignore_index=True, sort=False)
    common_cols = [
        "layer", "longitude", "latitude", "country_code", "gdppc", "value",
        "unit_id", "display_transformed_value", "display_transform",
    ]
    observed[[c for c in common_cols if c in observed.columns]].to_csv(
        SOURCE / "Fig_alt_NH4N_NO3N_point_map_observations.csv.gz",
        index=False,
        compression="gzip",
    )


def make_group_panels(world: gpd.GeoDataFrame, inputs: dict[str, pd.DataFrame], raster: tuple) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25))
    norm = plot_gdp_raster(fig, axes[0], world, raster)
    plot_point_map(fig, axes[1], world, inputs["all_water"], "GDP_sites", GDP_POINT_CMAP, fixed_norm=norm, standalone=True)
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.16, top=0.88, wspace=0.10)
    group_title(fig, list(axes), "a", "Global economic context", offset=0.025, fontsize=8.0)
    export(fig, "Panel_a_global_economic_context", INDIVIDUAL)
    plt.close(fig)

    for letter, title, endpoints, family, cmap in [
        ("b", "Global water-quality observations", ["DOSAT", "EC", "TP"], "water", WATER_POINT_CMAP),
        ("c", "Global aquatic-biota observations", ["Fish", "EPTO", "Algae"], "biota", BIOTA_POINT_CMAP),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
        for ax, endpoint in zip(axes, endpoints):
            frame = inputs["water"].loc[inputs["water"]["endpoint"] == endpoint] if family == "water" else inputs[endpoint]
            plot_point_map(fig, ax, world, frame, endpoint, cmap, standalone=True)
        fig.subplots_adjust(left=0.018, right=0.997, bottom=0.17, top=0.86, wspace=0.10)
        group_title(fig, list(axes), letter, title, offset=0.026, fontsize=8.0)
        stem = "Panel_b_global_water_quality" if letter == "b" else "Panel_c_global_aquatic_biota"
        export(fig, stem, INDIVIDUAL)
        plt.close(fig)

    fig = plt.figure(figsize=(3.8, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 0.28], left=0.11, right=0.98, bottom=0.12, top=0.89, hspace=0.48)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    plot_effect_forest(ax1, inputs["linear"])
    plot_gdp_density(ax2, inputs["density"])
    group_title(fig, [ax1], "d", "Background-adjusted GDP effects", offset=0.032, fontsize=8.0)
    export(fig, "Panel_d_adjusted_GDP_effects", INDIVIDUAL)
    plt.close(fig)

    for letter, title, endpoints, color, fill in [
        ("e", "Water-quality responses", ["DOSAT", "EC", "TP"], COLORS["water"], COLORS["water_fill"]),
        ("f", "Aquatic-biota responses", ["Fish", "EPTO", "Algae"], COLORS["biota"], COLORS["biota_fill"]),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
        for ax, endpoint in zip(axes, endpoints):
            plot_association(ax, endpoint, ENDPOINT_LABELS[endpoint], inputs, color, fill)
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.84, wspace=0.33)
        group_title(fig, list(axes), letter, title, offset=0.030, fontsize=8.0)
        stem = "Panel_e_water_quality_responses" if letter == "e" else "Panel_f_aquatic_biota_responses"
        export(fig, stem, INDIVIDUAL)
        plt.close(fig)


def make_atomic_panels(world: gpd.GeoDataFrame, inputs: dict[str, pd.DataFrame], raster: tuple) -> None:
    map_specs = [
        ("01_GDPpc_grid", "GDP_grid", None, GDP_CMAP),
        ("02_GDPpc_sites", "GDP_sites", inputs["all_water"], GDP_POINT_CMAP),
        ("03_DO_saturation_sites", "DOSAT", inputs["water"].loc[inputs["water"]["endpoint"] == "DOSAT"], WATER_POINT_CMAP),
        ("04_EC_sites", "EC", inputs["water"].loc[inputs["water"]["endpoint"] == "EC"], WATER_POINT_CMAP),
        ("05_TP_sites", "TP", inputs["water"].loc[inputs["water"]["endpoint"] == "TP"], WATER_POINT_CMAP),
        ("06_Fish_richness_units", "Fish", inputs["Fish"], BIOTA_POINT_CMAP),
        ("07_EPTO_richness_units", "EPTO", inputs["EPTO"], BIOTA_POINT_CMAP),
        ("08_Chlorophyll_a_lakes", "Algae", inputs["Algae"], BIOTA_POINT_CMAP),
    ]
    gdp_norm = raster[3]
    for stem, name, frame, cmap in map_specs:
        fig, ax = plt.subplots(figsize=(4.0, 2.35))
        if name == "GDP_grid":
            plot_gdp_raster(fig, ax, world, raster)
            title = "GDPpc grid"
        else:
            plot_point_map(fig, ax, world, frame, name, cmap,
                           fixed_norm=gdp_norm if name == "GDP_sites" else None, standalone=True)
            title = transform_values(name, frame["value"])[2].split(" (")[0]
        ax.text(0.0, 1.025, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")
        fig.subplots_adjust(left=0.02, right=0.995, bottom=0.15, top=0.91)
        export(fig, stem, ATOMIC)
        plt.close(fig)

    fig = plt.figure(figsize=(4.0, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 0.28], left=0.11, right=0.98, bottom=0.12, top=0.90, hspace=0.48)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    plot_effect_forest(ax1, inputs["linear"])
    plot_gdp_density(ax2, inputs["density"])
    ax1.text(0.0, 1.035, "Adjusted GDP effects", transform=ax1.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    export(fig, "09_adjusted_GDP_effects", ATOMIC)
    plt.close(fig)

    for number, endpoint in enumerate(["DOSAT", "EC", "TP", "Fish", "EPTO", "Algae"], start=10):
        domain_water = endpoint in ("DOSAT", "EC", "TP")
        color = COLORS["water"] if domain_water else COLORS["biota"]
        fill = COLORS["water_fill"] if domain_water else COLORS["biota_fill"]
        fig, ax = plt.subplots(figsize=(3.6, 2.35))
        plot_association(ax, endpoint, ENDPOINT_LABELS[endpoint], inputs, color, fill)
        ax.text(0.0, 1.035, ENDPOINT_LABELS[endpoint], transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.23, top=0.88)
        export(fig, f"{number:02d}_{endpoint}_GDP_response", ATOMIC)
        plt.close(fig)


def make_supplement(world: gpd.GeoDataFrame) -> None:
    d = pd.read_csv(CLEAN / "supplementary_freshverts_units.csv")
    specs = [
        ("all_richness", "AllFreshVerts", "Freshwater-associated vertebrates"),
        ("amphibia_richness", "Amphibia", "Amphibians"),
        ("reptiles_richness", "Reptiles", "Reptiles"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
    sources = []
    for ax, (column, name, label) in zip(axes, specs):
        frame = d.rename(columns={column: "value"}).copy()
        source, _ = plot_point_map(fig, ax, world, frame, name, BIOTA_POINT_CMAP, standalone=True)
        sources.append(source)
        ax.text(0.0, 1.025, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, fontweight="bold")
    fig.subplots_adjust(left=0.018, right=0.997, bottom=0.17, top=0.88, wspace=0.10)
    export(fig, "Fig_S1_freshwater_associated_vertebrates")
    plt.close(fig)
    pd.concat(sources, ignore_index=True).to_csv(
        SOURCE / "Fig_S1_point_map_observations.csv.gz", index=False, compression="gzip")


def make_nitrogen_supplement(world: gpd.GeoDataFrame) -> None:
    """Map and model the two requested supplementary nitrogen parameters."""
    map_data = pd.read_csv(CLEAN / "supplementary_nitrogen_site_medians.csv.gz")
    nitrogen_inputs = {
        "points": pd.read_csv(SOURCE / "Fig_S2_nitrogen_partial_residuals.csv.gz"),
        "curves": pd.read_csv(SOURCE / "Fig_S2_nitrogen_curves.csv"),
        "trend": pd.read_csv(ANALYSIS / "supplementary_nitrogen_density_balanced_summary.csv"),
    }
    fig = plt.figure(figsize=(7.2, 4.15))
    gs = fig.add_gridspec(
        2, 2, left=0.075, right=0.995, bottom=0.105, top=0.94,
        height_ratios=[0.95, 1.0], wspace=0.23, hspace=0.58,
    )
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    response_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    map_sources = []
    for ax, endpoint in zip(map_axes, ["NH4N", "NO3N"]):
        frame = map_data.loc[map_data["endpoint"] == endpoint]
        source, _ = plot_point_map(
            fig, ax, world, frame, endpoint, WATER_POINT_CMAP, standalone=True
        )
        map_sources.append(source)
        ax.text(
            0.0, 1.025, ENDPOINT_LABELS[endpoint], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.8, fontweight="bold",
        )
    for ax, endpoint in zip(response_axes, ["NH4N", "NO3N"]):
        plot_association(
            ax, endpoint, ENDPOINT_LABELS[endpoint], nitrogen_inputs,
            COLORS["water"], COLORS["water_fill"],
        )
    fig.canvas.draw()
    group_title(fig, map_axes, "a", "Global nitrogen observations", offset=0.035, fontsize=8.0)
    group_title(fig, response_axes, "b", "Density-balanced GDP responses", offset=0.035, fontsize=8.0)
    export(fig, "Fig_S2_nitrogen_GDP_responses")
    plt.close(fig)
    pd.concat(map_sources, ignore_index=True).to_csv(
        SOURCE / "Fig_S2_nitrogen_map_observations.csv.gz",
        index=False,
        compression="gzip",
    )

    SUPPLEMENT_ATOMIC.mkdir(parents=True, exist_ok=True)
    for number, endpoint in enumerate(["NH4N", "NO3N"], start=1):
        frame = map_data.loc[map_data["endpoint"] == endpoint]
        fig, ax = plt.subplots(figsize=(4.0, 2.35))
        plot_point_map(
            fig, ax, world, frame, endpoint, WATER_POINT_CMAP, standalone=True
        )
        ax.text(
            0.0, 1.025, ENDPOINT_LABELS[endpoint], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.0, fontweight="bold",
        )
        fig.subplots_adjust(left=0.02, right=0.995, bottom=0.15, top=0.91)
        export(fig, f"S2_{number:02d}_{endpoint}_sites", SUPPLEMENT_ATOMIC)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.6, 2.35))
        plot_association(
            ax, endpoint, ENDPOINT_LABELS[endpoint], nitrogen_inputs,
            COLORS["water"], COLORS["water_fill"],
        )
        ax.text(
            0.0, 1.035, ENDPOINT_LABELS[endpoint], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.0, fontweight="bold",
        )
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.23, top=0.88)
        export(fig, f"S2_{number + 2:02d}_{endpoint}_GDP_response", SUPPLEMENT_ATOMIC)
        plt.close(fig)


def write_qa() -> None:
    from PIL import Image
    records = []
    for path in sorted(FIG.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".png", ".pdf", ".svg", ".tiff"):
            continue
        row = {"file": str(path.relative_to(FIG)), "bytes": path.stat().st_size}
        if path.suffix.lower() in (".png", ".tiff"):
            with Image.open(path) as image:
                row.update({"width_px": image.width, "height_px": image.height, "mode": image.mode})
        records.append(row)
    pd.DataFrame(records).to_csv(QA / "figure_export_QA.csv", index=False, encoding="utf-8-sig")
    (QA / "figure_layout_contract_check.json").write_text(json.dumps({
        "main_grid": "4 columns x 4 logical rows with compact inter-group spacing",
        "maps": "rows 1-2: GDP grid, DO saturation, EC, TP; GDP-coloured matched sites, fish, EPTO, chlorophyll-a",
        "site_map_geometry": "independent coloured points only; no interpolation; no zero-filled countries",
        "point_colour_range": "upper 75% of each family colour map; minimum points are not white",
        "effect_panel": "column 1, rows 3-4; adjusted standardized linear GDP beta and 95% CI plus separate GDP density marginal",
        "associations": "water row 3 columns 2-4; aquatic biota row 4 columns 2-4",
        "group_labels": ["a", "b", "c", "d", "e", "f"],
        "observed_points_drawn": True,
        "prediction_ci_drawn": True,
        "prediction_definition": "population-standardized marginal prediction with full robust covariance",
        "map_display_values": "raw observations with monotonic colour transform and raw-unit colourbar labels; never z scores",
        "supplement": "NH4-N and NO3-N point maps plus density-balanced selected GDP trends, with atomic exports",
        "alternative_main": "EC and TP replaced consistently by NH4-N and NO3-N in maps, effects, trends and density inputs",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    style()
    for folder in (FIG, INDIVIDUAL, ATOMIC, SUPPLEMENT_ATOMIC, SOURCE, QA):
        folder.mkdir(parents=True, exist_ok=True)
    world, inputs, raster, _ = make_main()
    make_alternative_main(world, raster)
    make_group_panels(world, inputs, raster)
    make_atomic_panels(world, inputs, raster)
    make_supplement(world)
    make_nitrogen_supplement(world)
    write_qa()
    print(json.dumps({
        "main": str(FIG / "Fig_main_global_freshwater_economy_ecology.png"),
        "group_panels": 6,
        "atomic_panels": 19,
        "main_variants": 2,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
