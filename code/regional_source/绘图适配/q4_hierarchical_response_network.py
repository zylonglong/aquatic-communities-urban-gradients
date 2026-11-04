from __future__ import annotations

import json
import math
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge


ROOT = Path(r"E:\桌面\博士课题\Data")
ANALYSIS_ROOT = ROOT / "manuscript" / "城市水生态四科学问题_尺度响应分析_2026-07-23"
BASE = ANALYSIS_ROOT / "manuscript可视化图件" / "Figure5_Q4_specific_biological_carriers_2026-08-08"
SOURCE = BASE / "02_source_data"
Q2_FAMILY = ANALYSIS_ROOT / "03_Q2_水生生物整体响应" / "环境模块_生态维度_13尺度全局置换检验.csv"
OUT = BASE / "05_network_redesign"
SCRIPT_DIR = OUT / "01_script"
DATA_DIR = OUT / "02_source_data"
FIG_DIR = OUT / "03_figure"
QA_DIR = OUT / "04_QA"

MODULES = ["terrain", "urbanization", "human_activity"]
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
MODULE_SHORT = {
    "terrain": "Topography",
    "urbanization": "Urban construction",
    "human_activity": "Direct activity",
}
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
WATER_COLORS = {"River": "#5067B0", "Lake": "#B54A79"}
WATER_N = {"River": 86, "Lake": 44}
CMAP = mpl.colormaps["coolwarm"]
DIR_NORM = Normalize(vmin=-0.40, vmax=0.40)


def blend_with_white(color: str, amount: float = 0.88) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple((1 - amount) * rgb + amount * np.ones(3))


def clean_label(name: str) -> str:
    aliases = {
        "zooplankton_total_density": "Zooplankton total density",
        "zooplankton_total_biomass": "Zooplankton total biomass",
        "fish_total_biomass": "Fish total biomass",
        "phytoplankton_total_density": "Phytoplankton total density",
        "macrophytes_total_density": "Macrophyte total density",
        "overall_richness": "Overall richness",
        "periphyton_margalef": "Periphyton Margalef richness",
        "fish_mean_total_length": "Fish mean total length",
        "fish_median_total_length": "Fish median total length",
        "periphyton_high_profile_diatom_biomass_pct": "High-profile diatom biomass (%)",
        "phytoplankton_chlorophyta_richness": "Chlorophyta richness",
    }
    if name in aliases:
        return aliases[name]
    text = str(name)
    text = re.sub(r"^beta_(river|lake|all)_", "", text, flags=re.I)
    replacements = [
        ("_bray_curtis_", " Bray-Curtis "), ("_bray_", " Bray-Curtis "),
        ("_sorensen_", " Sørensen "), ("_jaccard_", " Jaccard "),
        ("_nestedness_", " nestedness "), ("_turnover_", " turnover "),
        ("_lcbd", " LCBD"), ("_pcoa", " PCoA"), ("_pct", " %"), ("_", " "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("qual pa", "presence/absence").replace("density Bray-Curtis", "abundance Bray-Curtis")
    return text[:1].upper() + text[1:]


def prepare_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    assoc = pd.read_csv(SOURCE / "Q4_all_frozen_candidate_multiscale_associations.csv")
    contrib = pd.read_csv(SOURCE / "Q4_variable_contribution_and_bootstrap_stability.csv")
    q2 = pd.read_csv(Q2_FAMILY)

    identity = [
        "water_body_type", "response_family", "response_family_cn", "taxon_group",
        "taxon_group_cn", "column_name", "variable_name_cn",
    ]
    overall = (
        assoc.groupby(identity, as_index=False)
        .agg(
            overall_rms=("partial_spearman_rho", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))),
            strict_min_p=("family_variables_13range_maxT_p", "min"),
            response_min_p=("response_13range_maxT_p", "min"),
            strict_cells=("family_variables_13range_maxT_p", lambda x: int(np.sum(np.asarray(x, float) <= 0.05))),
            response_cells=("response_13range_maxT_p", lambda x: int(np.sum(np.asarray(x, float) <= 0.05))),
        )
    )
    module = (
        assoc.groupby(identity + ["environment_module"], as_index=False)
        .agg(
            module_rms=("partial_spearman_rho", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))),
            module_signed_mean=("partial_spearman_rho", "mean"),
            module_signed_median=("partial_spearman_rho", "median"),
            module_strict_ranges=("family_variables_13range_maxT_p", lambda x: int(np.sum(np.asarray(x, float) <= 0.05))),
            module_response_ranges=("response_13range_maxT_p", lambda x: int(np.sum(np.asarray(x, float) <= 0.05))),
            peak_abs_rho=("partial_spearman_rho", lambda x: float(np.nanmax(np.abs(np.asarray(x, float))))),
        )
    )
    wide_parts = []
    for metric in ["module_rms", "module_signed_mean", "module_signed_median", "module_strict_ranges", "module_response_ranges", "peak_abs_rho"]:
        w = module.pivot(index=identity, columns="environment_module", values=metric)
        w.columns = [f"{m}_{metric}" for m in w.columns]
        wide_parts.append(w)
    variable = pd.concat(wide_parts, axis=1).reset_index().merge(overall, on=identity, how="left")
    keep = [
        "water_body_type", "response_family", "column_name", "balanced_contribution_share",
        "observed_rank_within_family", "top3_selection_frequency", "selection_stability",
        "bootstrap_ci95_low", "bootstrap_ci95_high",
    ]
    variable = variable.merge(contrib[keep], on=["water_body_type", "response_family", "column_name"], how="left")
    variable["strict_any"] = variable.strict_min_p <= 0.05
    variable["response_any"] = variable.response_min_p <= 0.05
    variable["display_label"] = variable.column_name.map(clean_label)

    family = (
        assoc.groupby(["water_body_type", "response_family", "environment_module"], as_index=False)
        .agg(
            integrated_rms=("partial_spearman_rho", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))),
            signed_mean=("partial_spearman_rho", "mean"),
            signed_median=("partial_spearman_rho", "median"),
            n_variables=("column_name", "nunique"),
        )
    )
    support = (
        q2.groupby(["water_body_type", "response_family", "environment_module"], as_index=False)
        .agg(
            supported_ranges=("range_maxT_p", lambda x: int(np.sum(np.asarray(x, float) <= 0.05))),
            family_rms_across_ranges=("rms_partial_rho", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))),
        )
    )
    family = family.merge(support, on=["water_body_type", "response_family", "environment_module"], how="left")
    overall_family = (
        variable.groupby(["water_body_type", "response_family"], as_index=False)
        .agg(
            family_overall_rms=("overall_rms", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))),
            candidate_n=("column_name", "nunique"),
        )
    )
    top3 = (
        variable.loc[variable.observed_rank_within_family <= 3]
        .groupby(["water_body_type", "response_family"], as_index=False)
        .balanced_contribution_share.sum()
        .rename(columns={"balanced_contribution_share": "top3_cumulative_share"})
    )
    effective = (
        variable.groupby(["water_body_type", "response_family"])
        .balanced_contribution_share.apply(lambda x: float(1.0 / np.sum(np.asarray(x, float) ** 2)))
        .reset_index(name="effective_contributor_n")
    )
    family = (
        family.merge(overall_family, on=["water_body_type", "response_family"], how="left")
        .merge(top3, on=["water_body_type", "response_family"], how="left")
        .merge(effective, on=["water_body_type", "response_family"], how="left")
    )
    return variable, family


def choose_highlights(sub: pd.DataFrame, maximum: int = 12) -> pd.DataFrame:
    chosen: set[str] = set()
    # Ensure every family is represented, prioritising strict evidence and then contribution.
    for family in FAMILIES:
        g = sub.loc[sub.response_family.eq(family)].copy()
        strict = g.loc[g.strict_any].sort_values(["strict_cells", "overall_rms", "balanced_contribution_share"], ascending=False)
        row = strict.iloc[0] if len(strict) else g.sort_values("balanced_contribution_share", ascending=False).iloc[0]
        chosen.add(row.column_name)
    ranked = sub.sort_values(["strict_any", "strict_cells", "overall_rms", "balanced_contribution_share"], ascending=False)
    for name in ranked.column_name:
        if len(chosen) >= maximum:
            break
        if bool(ranked.loc[ranked.column_name.eq(name), "strict_any"].iloc[0]):
            chosen.add(name)
    out = sub.loc[sub.column_name.isin(chosen)].copy()
    out["family_order"] = out.response_family.map({f: i for i, f in enumerate(FAMILIES)})
    out = out.sort_values(["family_order", "strict_any", "strict_cells", "balanced_contribution_share"], ascending=[True, False, False, False]).reset_index(drop=True)
    out["highlight_id"] = np.arange(1, len(out) + 1)
    return out


def grid_positions(n: int, center_x: float, width: float = 0.155, y_low: float = 0.085, y_high: float = 0.535):
    ncols = int(np.clip(round(math.sqrt(n * 0.43)), 3, 7))
    nrows = int(math.ceil(n / ncols))
    xs = np.linspace(center_x - width / 2, center_x + width / 2, ncols)
    ys = np.linspace(y_high, y_low, nrows)
    positions = []
    for i in range(n):
        row, col = divmod(i, ncols)
        x = xs[col]
        if row % 2 and ncols > 1:
            x += (xs[1] - xs[0]) * 0.22
            x = min(center_x + width / 2, x)
        positions.append((x, ys[row]))
    return positions


def draw_wedge_node(ax, x: float, y: float, radius: float, row, highlight_id: int | None):
    # Fixed sector positions: top = terrain, lower right = urban, lower left = direct activity.
    sectors = {
        "terrain": (30, 150),
        "human_activity": (150, 270),
        "urbanization": (270, 390),
    }
    module_rms = {module: float(getattr(row, f"{module}_module_rms")) for module in MODULES}
    local_max = max(max(module_rms.values()), 1e-9)
    ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor="none", zorder=5))
    for module, (theta1, theta2) in sectors.items():
        value = float(getattr(row, f"{module}_module_signed_median"))
        sector_radius = radius * (0.42 + 0.58 * module_rms[module] / local_max)
        ax.add_patch(Wedge((x, y), sector_radius, theta1, theta2, facecolor=CMAP(DIR_NORM(value)), edgecolor="none", zorder=6))
    edge = "#111111" if bool(row.strict_any) else ("#6A6A6A" if bool(row.response_any) else "white")
    lw = 1.05 if bool(row.strict_any) else (0.45 if bool(row.response_any) else 0.25)
    ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=edge, linewidth=lw, zorder=7))
    if highlight_id is not None:
        ax.text(
            x + radius * 0.82, y + radius * 0.82, str(highlight_id),
            fontsize=4.6, fontweight="bold", ha="center", va="center", zorder=9,
            bbox={"boxstyle": "circle,pad=0.12", "facecolor": "white", "edgecolor": "#222222", "linewidth": 0.35},
        )


def plot_network(water: str, variable: pd.DataFrame, family: pd.DataFrame):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7, "axes.linewidth": 0.75, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    v = variable.loc[variable.water_body_type.eq(water)].copy()
    f = family.loc[family.water_body_type.eq(water)].copy()
    highlights = choose_highlights(v)
    highlight_map = highlights.set_index("column_name").highlight_id.to_dict()

    fig = plt.figure(figsize=(12.0, 8.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.85, 1.55], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    leg = fig.add_subplot(gs[0, 1]); leg.set_xlim(0, 1); leg.set_ylim(0, 1); leg.axis("off")

    ax.text(0.00, 0.995, "b" if water == "River" else "c", fontsize=10, fontweight="bold", va="top")
    ax.text(0.035, 0.995, f"{water} hierarchical response network", fontsize=11, fontweight="bold", va="top", color=WATER_COLORS[water])
    ax.text(0.035, 0.962, f"n = {WATER_N[water]} independent sites · association network, not a causal SEM", fontsize=6.6, color="#555555", va="top")

    driver_x = {"terrain": 0.20, "urbanization": 0.50, "human_activity": 0.80}
    family_x = dict(zip(FAMILIES, [0.09, 0.295, 0.50, 0.705, 0.91]))
    driver_y, family_y = 0.905, 0.705

    # Driver nodes.
    for module in MODULES:
        x = driver_x[module]
        ax.add_patch(FancyBboxPatch((x - 0.075, driver_y - 0.030), 0.15, 0.060, boxstyle="round,pad=0.008,rounding_size=0.015", facecolor=blend_with_white(DRIVER_COLORS[module], 0.76), edgecolor=DRIVER_COLORS[module], linewidth=1.2, zorder=8))
        ax.text(x, driver_y, MODULE_LABELS[module], ha="center", va="center", fontsize=6.8, fontweight="bold", color="#222222", zorder=9)

    # Family-level association paths.
    strength_min, strength_max = f.integrated_rms.min(), f.integrated_rms.max()
    for row in f.itertuples(index=False):
        x0, x1 = driver_x[row.environment_module], family_x[row.response_family]
        scaled = (row.integrated_rms - strength_min) / max(strength_max - strength_min, 1e-9)
        width = 0.8 + 4.5 * scaled
        color = CMAP(DIR_NORM(row.signed_median))
        supported = int(row.supported_ranges) > 0
        rad = 0.18 * (x1 - x0)
        arrow = FancyArrowPatch(
            (x0, driver_y - 0.034), (x1, family_y + 0.036),
            arrowstyle="-|>", mutation_scale=6.5, connectionstyle=f"arc3,rad={rad}",
            linewidth=width, color=color, alpha=0.88 if supported else 0.34,
            linestyle="-" if supported else (0, (2.0, 2.0)), zorder=2,
        )
        ax.add_patch(arrow)

    # Family groups and variable nodes.
    overall_min, overall_max = v.overall_rms.min(), v.overall_rms.max()
    family_strength_min, family_strength_max = f.family_overall_rms.min(), f.family_overall_rms.max()
    node_position_rows = []
    for family_name in FAMILIES:
        cx = family_x[family_name]
        family_v = v.loc[v.response_family.eq(family_name)].sort_values(["balanced_contribution_share", "overall_rms"], ascending=False).reset_index(drop=True)
        family_f = f.loc[f.response_family.eq(family_name)].iloc[0]
        ax.add_patch(FancyBboxPatch((cx - 0.087, 0.055), 0.174, 0.505, boxstyle="round,pad=0.006,rounding_size=0.010", facecolor=blend_with_white(FAMILY_COLORS[family_name], 0.92), edgecolor=blend_with_white(FAMILY_COLORS[family_name], 0.45), linewidth=0.7, zorder=0))
        fs = (family_f.family_overall_rms - family_strength_min) / max(family_strength_max - family_strength_min, 1e-9)
        family_radius = 0.029 + 0.012 * fs
        ax.add_patch(Circle((cx, family_y), family_radius, facecolor=FAMILY_COLORS[family_name], edgecolor="white", linewidth=1.0, zorder=7))
        ax.text(cx, family_y, FAMILY_LABELS[family_name], ha="center", va="center", fontsize=5.4, fontweight="bold", color="white", zorder=8)
        top3_share = float(family_f.top3_cumulative_share)
        ax.text(cx, 0.610, f"n = {len(family_v)} · Top 3 = {100 * top3_share:.1f}% · Nₑ = {family_f.effective_contributor_n:.1f}", ha="center", va="center", fontsize=5.25, color="#444444")

        positions = grid_positions(len(family_v), cx)
        contribution_max = family_v.balanced_contribution_share.max()
        for (x, y), row in zip(positions, family_v.itertuples(index=False)):
            edge_scale = math.sqrt(max(row.balanced_contribution_share, 0) / max(contribution_max, 1e-9))
            ax.plot([cx, x], [family_y - family_radius * 0.75, y], color=FAMILY_COLORS[family_name], linewidth=0.12 + 1.45 * edge_scale, alpha=0.09 + 0.27 * edge_scale, zorder=1)
            size_scale = (row.overall_rms - overall_min) / max(overall_max - overall_min, 1e-9)
            radius = 0.0042 + 0.0047 * math.sqrt(max(size_scale, 0))
            draw_wedge_node(ax, x, y, radius, row, highlight_map.get(row.column_name))
            node_position_rows.append({"water_body_type": water, "response_family": family_name, "column_name": row.column_name, "x": x, "y": y, "radius": radius, "highlight_id": highlight_map.get(row.column_name)})

    # Network footer and colour scale.
    ax.text(0.50, 0.020, "All retained variables are shown; labels identify selected review anchors only", ha="center", va="center", fontsize=5.8, color="#555555")
    cax = fig.add_axes([0.17, 0.028, 0.34, 0.014])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=DIR_NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-0.4, 0, 0.4]); cb.ax.tick_params(labelsize=5, length=2)
    cb.set_label("Signed median partial Spearman ρ", fontsize=5.5, labelpad=1)

    # Explanation panel.
    leg.text(0.03, 0.985, "How to read the network", fontsize=8.5, fontweight="bold", va="top")
    leg.text(0.03, 0.944, "Driver → dimension", fontsize=6.7, fontweight="bold")
    leg.add_patch(FancyArrowPatch((0.05, 0.913), (0.42, 0.913), arrowstyle="-|>", mutation_scale=7, linewidth=3.5, color=CMAP(DIR_NORM(0.28))))
    leg.text(0.47, 0.913, "width = RMS strength\ncolour = signed median ρ", fontsize=5.8, va="center")
    leg.plot([0.05, 0.42], [0.865, 0.865], color="#777777", linewidth=1.2)
    leg.plot([0.05, 0.42], [0.835, 0.835], color="#777777", linewidth=1.2, linestyle=(0, (2, 2)))
    leg.text(0.47, 0.850, "solid = ≥1 family maxT range\ndashed = no supported range", fontsize=5.7, va="center")
    leg.text(0.03, 0.790, "Dimension → variable", fontsize=6.7, fontweight="bold")
    leg.plot([0.05, 0.42], [0.756, 0.756], color=FAMILY_COLORS["community_baseline"], linewidth=2.4, alpha=0.55)
    leg.text(0.47, 0.756, "width = within-dimension\nbalanced contribution", fontsize=5.8, va="center")

    # Wedge example.
    glyph_x, glyph_y, glyph_r = 0.18, 0.680, 0.047
    fake = type("Fake", (), {
        "terrain_module_signed_median": -0.28, "urbanization_module_signed_median": 0.32,
        "human_activity_module_signed_median": 0.05, "terrain_module_rms": 0.30,
        "urbanization_module_rms": 0.24, "human_activity_module_rms": 0.10,
        "strict_any": True, "response_any": True,
    })()
    draw_wedge_node(leg, glyph_x, glyph_y, glyph_r, fake, None)
    leg.text(0.18, 0.744, "terrain", ha="center", fontsize=5.2)
    leg.text(0.28, 0.650, "urban", ha="center", fontsize=5.2)
    leg.text(0.07, 0.650, "direct", ha="center", fontsize=5.2)
    leg.text(0.47, 0.685, "sector length = driver RMS\nsector colour = signed median ρ", fontsize=5.8, va="center")
    leg.add_patch(Circle((0.10, 0.600), 0.014, facecolor="#BDBDBD", edgecolor="white"))
    leg.add_patch(Circle((0.19, 0.600), 0.028, facecolor="#BDBDBD", edgecolor="white"))
    leg.text(0.47, 0.600, "outer size = cross-driver RMS", fontsize=5.8, va="center")
    leg.add_patch(Circle((0.10, 0.550), 0.022, facecolor="white", edgecolor="#111111", linewidth=1.2))
    leg.text(0.47, 0.550, "black ring = ≥1 strict\nfamily-wide maxT range", fontsize=5.8, va="center")

    leg.axhline(0.505, color="#D0D0D0", linewidth=0.7)
    leg.text(0.03, 0.480, "Highlighted characteristics", fontsize=7.2, fontweight="bold", va="top")
    y = 0.447
    line_step = 0.034 if len(highlights) <= 8 else 0.029
    for row in highlights.itertuples(index=False):
        color = FAMILY_COLORS[row.response_family]
        leg.text(0.035, y, str(row.highlight_id), fontsize=5.6, fontweight="bold", ha="center", va="top", bbox={"boxstyle": "circle,pad=0.16", "facecolor": "white", "edgecolor": color, "linewidth": 0.8})
        label = "\n".join(textwrap.wrap(row.display_label, width=31))
        leg.text(0.085, y + 0.002, label, fontsize=5.45, color="#222222", va="top")
        y -= line_step * (1.0 + 0.72 * label.count("\n"))

    leg.text(0.03, 0.020, "Arrowheads organize predictor-to-response layers;\nthey do not represent tested causal paths.", fontsize=5.3, color="#666666", va="bottom")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.975, bottom=0.060)
    return fig, highlights, pd.DataFrame(node_position_rows)


def run_qa(variable: pd.DataFrame, family: pd.DataFrame, positions: pd.DataFrame, outputs: list[Path]) -> pd.DataFrame:
    checks = []
    def add(name, passed, detail): checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    add("All River variables represented", len(positions.loc[positions.water_body_type.eq("River")]) == 178, len(positions.loc[positions.water_body_type.eq("River")]))
    add("All Lake variables represented", len(positions.loc[positions.water_body_type.eq("Lake")]) == 144, len(positions.loc[positions.water_body_type.eq("Lake")]))
    add("Three module metrics per variable", all(f"{m}_module_signed_mean" in variable.columns for m in MODULES), "3 sectors")
    add("Five dimensions per waterbody", bool((family.groupby("water_body_type").response_family.nunique() == 5).all()), family.groupby("water_body_type").response_family.nunique().to_dict())
    sums = variable.groupby(["water_body_type", "response_family"]).balanced_contribution_share.sum()
    add("Within-dimension contributions sum to one", np.allclose(sums, 1, atol=1e-10), float((sums - 1).abs().max()))
    add("No variable position duplicates within family", not positions.duplicated(["water_body_type", "response_family", "x", "y"]).any(), "unique layout positions")
    for path in outputs:
        add(f"Output exists: {path.suffix}", path.exists() and path.stat().st_size > 0, path.stat().st_size if path.exists() else 0)
        if path.suffix == ".svg" and path.exists():
            text = path.read_text(encoding="utf-8")
            add("SVG live text", "<text" in text, text.count("<text"))
    return pd.DataFrame(checks)


def main():
    for path in [SCRIPT_DIR, DATA_DIR, QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for ext in ["PNG", "SVG", "PDF", "TIFF"]:
        (FIG_DIR / ext).mkdir(parents=True, exist_ok=True)

    variable, family = prepare_metrics()
    variable.to_csv(DATA_DIR / "Q4_network_variable_metrics.csv", index=False, encoding="utf-8-sig")
    family.to_csv(DATA_DIR / "Q4_network_dimension_metrics.csv", index=False, encoding="utf-8-sig")
    all_positions = []
    all_highlights = []
    outputs = []
    for water in ["River", "Lake"]:
        fig, highlights, positions = plot_network(water, variable, family)
        stem = f"Figure_Q4_{water.lower()}_hierarchical_response_network"
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
        outputs.extend(paths.values())
        highlights.assign(water_body_type=water).to_csv(DATA_DIR / f"Q4_{water.lower()}_highlighted_characteristics.csv", index=False, encoding="utf-8-sig")
        all_positions.append(positions)
        all_highlights.append(highlights.assign(water_body_type=water))

    positions = pd.concat(all_positions, ignore_index=True)
    positions.to_csv(DATA_DIR / "Q4_network_node_positions.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_highlights, ignore_index=True).to_csv(DATA_DIR / "Q4_network_highlighted_characteristics.csv", index=False, encoding="utf-8-sig")
    qa = run_qa(variable, family, positions, outputs)
    qa.to_csv(QA_DIR / "Q4_hierarchical_network_automated_QA.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    summary = {
        "backend": "Python",
        "figure_type": "non-causal hierarchical response network",
        "variable_nodes": variable.groupby("water_body_type").size().to_dict(),
        "strict_variable_nodes": variable.groupby("water_body_type").strict_any.sum().astype(int).to_dict(),
        "all_qa_pass": bool(qa.passed.all()),
    }
    (QA_DIR / "Q4_hierarchical_network_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
