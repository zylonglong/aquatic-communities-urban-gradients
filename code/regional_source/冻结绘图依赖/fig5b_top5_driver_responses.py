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
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import q4_lake_semicircular_smooth_tracks as base


FIG5_ROOT = base.BASE.parent
NETWORK_SOURCE = base.BASE / "09_river_lake_shared" / "02_source_data"
ASSOCIATION_SOURCE = (
    FIG5_ROOT / "02_source_data" / "Q4_all_frozen_candidate_multiscale_associations.csv"
)
OUT = FIG5_ROOT / "06_panel_b_top5_driver_responses"
SCRIPT_DIR = OUT / "01_script"
DATA_DIR = OUT / "02_source_data"
FIG_DIR = OUT / "03_figure"
QA_DIR = OUT / "04_QA"

WATERS = ["River", "Lake"]
MODULES = ["terrain", "urbanization", "human_activity"]
TOP_N = 5
GROUP_GAP = 0.0
X_MAX = 0.50
B_DIRECTION_NORM = Normalize(vmin=-0.50, vmax=0.50)

MODULE_LABELS = {
    "terrain": "Topography and geomorphology",
    "urbanization": "Urban construction",
    "human_activity": "Direct human activity",
}
MODULE_PLOT_LABELS = {
    "terrain": "Topography and\ngeomorphology",
    "urbanization": "Urban\nconstruction",
    "human_activity": "Direct human\nactivity",
}
FAMILY_LABELS = {
    "community_baseline": "Community-level attributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "Community composition",
    "functional_indicator": "Functional or indicator metrics",
}


def publication_label(column_name: str) -> str:
    text = str(column_name)
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
    text = re.sub(r"\blcbd\b", "LCBD", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpcoa([123])\b", r"PCoA\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpielou\b", "Pielou", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmargalef\b", "Margalef", text, flags=re.IGNORECASE)
    text = text[:1].upper() + text[1:] if text else str(column_name)
    text = re.sub(
        r"^Benthic invertebrate qual (.+)$",
        r"Benthic invertebrate \1 (qualitative sample)",
        text,
    )
    text = text.replace("Phytoplankton chlorophyta richness", "Chlorophyta richness")
    text = text.replace("biomass weighted", "biomass-weighted")
    text = text.replace("high profile", "high-profile")
    text = text.replace("low profile", "low-profile")
    text = re.sub(r" %$", " (%)", text)
    return text


def wrap_label(text: str, width: int = 43) -> str:
    if len(text) <= width:
        return text
    return "\n".join(
        textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    )


def load_and_select() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    boundary_records = []
    for water in WATERS:
        metrics = pd.read_csv(
            NETWORK_SOURCE / f"Q4_{water.lower()}_ordered_variable_metrics.csv"
        )
        if metrics.column_name.duplicated().any():
            raise ValueError(f"Duplicate retained characteristics in {water}")

        for module in MODULES:
            rms_col = f"{module}_module_rms"
            peak_col = f"{module}_peak_abs_rho"
            strict_col = f"{module}_module_strict_ranges"
            response_col = f"{module}_module_response_ranges"
            direction_col = f"{module}_module_signed_median"
            ranked = metrics.sort_values(
                [rms_col, peak_col, "column_name"],
                ascending=[False, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            top = ranked.head(TOP_N).copy()
            top["environment_module"] = module
            top["driver_label"] = MODULE_LABELS[module]
            top["selection_rank"] = np.arange(1, TOP_N + 1)
            top["response_rms_13range"] = top[rms_col]
            top["signed_median_rho_13range"] = top[direction_col]
            top["peak_abs_rho"] = top[peak_col]
            top["strict_supported_ranges"] = top[strict_col].astype(int)
            top["response_supported_ranges"] = top[response_col].astype(int)
            top["candidate_pool_n"] = len(metrics)
            top["display_label"] = top.column_name.map(publication_label)
            top["family_label"] = top.response_family.map(FAMILY_LABELS)
            selected_frames.append(top)

            rank5 = float(ranked.loc[TOP_N - 1, rms_col])
            rank6 = float(ranked.loc[TOP_N, rms_col])
            boundary_records.append(
                {
                    "water_body_type": water,
                    "environment_module": module,
                    "candidate_pool_n": len(metrics),
                    "rank5_rms": rank5,
                    "rank6_rms": rank6,
                    "rank5_minus_rank6": rank5 - rank6,
                }
            )

    selected = pd.concat(selected_frames, ignore_index=True)
    selected["water_order"] = selected.water_body_type.map({"River": 0, "Lake": 1})
    selected["module_order"] = selected.environment_module.map(
        {module: index for index, module in enumerate(MODULES)}
    )
    selected = selected.sort_values(
        ["water_order", "module_order", "selection_rank"]
    ).reset_index(drop=True)
    return selected, pd.DataFrame(boundary_records)


def row_positions() -> tuple[dict[str, np.ndarray], float]:
    positions = {
        module: index * (TOP_N + GROUP_GAP) + np.arange(TOP_N, dtype=float)
        for index, module in enumerate(MODULES)
    }
    ymax = float(positions[MODULES[-1]][-1] + 0.50)
    return positions, ymax


def draw_compact_domain_legend(ax: plt.Axes) -> int:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.92, "Ecological response dimensions",
        fontsize=12.4, fontweight="bold", ha="center", va="top",
    )
    x_positions = np.linspace(0.08, 0.92, len(base.FAMILIES))
    compact_labels = {
        "community_baseline": "Community-level\nattributes",
        "alpha_diversity": "α diversity",
        "beta_diversity": "β diversity",
        "community_composition": "Community\ncomposition",
        "functional_indicator": "Functional or\nindicator metrics",
    }
    for x, family in zip(x_positions, base.FAMILIES):
        ax.add_patch(
            Rectangle(
                (x - 0.034, 0.58), 0.068, 0.16,
                facecolor=base.FAMILY_COLORS[family],
                edgecolor=base.FAMILY_COLORS[family], linewidth=0.0,
            )
        )
        ax.text(
            x, 0.50, compact_labels[family], fontsize=9.7,
            ha="center", va="top", linespacing=0.92, color="#222222",
        )
    ax.text(
        0.5, 0.190, "Line/point colour: 13-range signed median ρ",
        fontsize=9.9, fontweight="bold", ha="center", va="center",
    )
    colour_ax = ax.inset_axes([0.25, 0.025, 0.50, 0.085])
    colour_bar = mpl.colorbar.ColorbarBase(
        colour_ax, cmap=base.CMAP, norm=B_DIRECTION_NORM,
        orientation="horizontal", ticks=[-0.5, 0.0, 0.5],
    )
    colour_bar.outline.set_linewidth(0.55)
    colour_ax.tick_params(axis="x", labelsize=8.9, length=2.0, pad=1.2)
    return len(base.FAMILIES)


def draw_water_panel(
    label_ax: plt.Axes,
    plot_ax: plt.Axes,
    selected: pd.DataFrame,
    water: str,
) -> dict:
    rows, ymax = row_positions()
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(ymax, -1.45)
    label_ax.axis("off")
    plot_ax.set_xlim(0, X_MAX + 0.01)
    plot_ax.set_ylim(ymax, -1.45)
    plot_ax.set_yticks([])
    plot_ax.set_xticks([])
    plot_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    plot_ax.grid(False)
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.spines["bottom"].set_visible(False)
    plot_ax.spines["left"].set_visible(False)
    plot_ax.text(
        0.50, 0.985, "13-range RMS response strength",
        transform=plot_ax.transAxes, fontsize=9.6, fontweight="bold",
        ha="center", va="top", color="#222222",
    )

    plotted = 0
    domain_strips = 0
    driver_labels = 0
    for module in MODULES:
        subset = selected.loc[
            selected.water_body_type.eq(water)
            & selected.environment_module.eq(module)
        ].sort_values("selection_rank")
        ys = rows[module]
        driver_color = base.DRIVER_COLORS[module]

        label_ax.add_patch(
            Rectangle(
                (0.0, ys[0] - 0.5), 0.145, TOP_N,
                facecolor=driver_color, edgecolor=driver_color,
                linewidth=0.8, alpha=1.0, zorder=2,
            )
        )
        label_ax.text(
            0.0725, float(np.mean(ys)), MODULE_PLOT_LABELS[module],
            rotation=90, rotation_mode="anchor", ha="center", va="center",
            fontsize=12.2, fontweight="bold", color=base.contrast_text(driver_color),
            linespacing=1.12,
            zorder=3,
        )
        driver_labels += 1

        for y, row in zip(ys, subset.itertuples(index=False)):
            # RMS and direction are deliberately orthogonal encodings:
            # line length/point position = unsigned 13-range RMS strength;
            # coolwarm colour = signed median rho across the same 13 ranges.
            direction_color = base.CMAP(
                B_DIRECTION_NORM(row.signed_median_rho_13range)
            )
            label_ax.text(
                0.895, y, wrap_label(row.display_label, width=24),
                fontsize=10.3, ha="right", va="center", color="#111111",
                linespacing=1.08,
            )
            label_ax.add_patch(
                Rectangle(
                    (0.925, y - 0.5), 0.075, 1.0,
                    facecolor=base.FAMILY_COLORS[row.response_family],
                    edgecolor=base.FAMILY_COLORS[row.response_family],
                    linewidth=0.0, zorder=2,
                )
            )
            domain_strips += 1
            plot_ax.plot(
                [0.0, row.response_rms_13range], [y, y],
                color=direction_color, linewidth=6.8, alpha=1.0,
                solid_capstyle="butt", zorder=2,
            )
            plot_ax.scatter(
                row.response_rms_13range, y, s=205,
                facecolor=direction_color, edgecolor="white", linewidth=1.2, zorder=3,
            )
            plot_ax.text(
                row.response_rms_13range + 0.026, y,
                f"{row.response_rms_13range:.3f}",
                fontsize=9.8, ha="left", va="center", color="#333333",
            )
            plotted += 1

    return {
        "plotted": plotted,
        "xlim": plot_ax.get_xlim(),
        "ymax": ymax,
        "domain_strips": domain_strips,
        "driver_labels": driver_labels,
        "metric_title": "13-range RMS response strength",
        "direction_norm": (B_DIRECTION_NORM.vmin, B_DIRECTION_NORM.vmax),
        "left_spine_visible": plot_ax.spines["left"].get_visible(),
    }


def plot_panel_b(selected: pd.DataFrame) -> tuple[plt.Figure, dict]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10.0,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(5.15, 15.0), facecolor="white")
    grid = fig.add_gridspec(
        4, 2,
        height_ratios=[0.10, 0.92, 0.19, 0.92],
        width_ratios=[1.48, 1.0],
        left=0.025, right=0.985, top=0.992, bottom=0.018,
        hspace=0.015, wspace=0.0,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_y = 0.82
    title_ax.text(0.0, title_y, "b", fontsize=20.5, fontweight="bold", va="top")
    title_ax.annotate(
        "Strongest biological responses\nwithin environmental drivers",
        xy=(0.0, title_y), xycoords="axes fraction",
        xytext=(36, 0), textcoords="offset points",
        fontsize=16.6, fontweight="bold", va="top", linespacing=1.18,
    )

    axes = {}
    panel_meta = {}
    for row_index, water in [(1, "River"), (3, "Lake")]:
        label_ax = fig.add_subplot(grid[row_index, 0])
        plot_ax = fig.add_subplot(grid[row_index, 1])
        axes[water] = {"label": label_ax, "plot": plot_ax}
        panel_meta[water] = draw_water_panel(label_ax, plot_ax, selected, water)

    legend_ax = fig.add_subplot(grid[2, :])
    legend_count = draw_compact_domain_legend(legend_ax)
    return fig, {
        "title_ax": title_ax,
        "legend_ax": legend_ax,
        "legend_count": legend_count,
        "axes": axes,
        "panels": panel_meta,
    }


def run_qa(
    selected: pd.DataFrame,
    boundary: pd.DataFrame,
    associations: pd.DataFrame,
    meta: dict,
    outputs: list[Path],
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, detail) -> None:
        rows.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    counts = selected.groupby(["water_body_type", "environment_module"]).size()
    add("Exactly five characteristics per waterbody-driver group",
        bool((counts == TOP_N).all()) and len(counts) == 6, counts.to_dict())
    ranks = selected.groupby(["water_body_type", "environment_module"])[
        "selection_rank"
    ].apply(list)
    add("Each group contains ranks 1-5",
        all(values == [1, 2, 3, 4, 5] for values in ranks), ranks.to_dict())
    add("Thirty plotted characteristic-driver entries",
        len(selected) == 30 and sum(v["plotted"] for v in meta["panels"].values()) == 30,
        len(selected))
    add("Shared River-Lake x scale",
        meta["panels"]["River"]["xlim"] == meta["panels"]["Lake"]["xlim"],
        meta["panels"]["River"]["xlim"])
    add("Horizontal axes and grids removed",
        all(
            len(meta["axes"][water]["plot"].get_xticks()) == 0
            and not meta["axes"][water]["plot"].spines["bottom"].get_visible()
            and not meta["axes"][water]["plot"].spines["left"].get_visible()
            for water in WATERS
        ), "no ticks, labels, bottom or left spine, or grid")
    add("Each waterbody has 15 touching ecological-dimension strips",
        all(meta["panels"][water]["domain_strips"] == 15 for water in WATERS),
        {water: meta["panels"][water]["domain_strips"] for water in WATERS})
    add("Each waterbody has three filled driver labels",
        all(meta["panels"][water]["driver_labels"] == 3 for water in WATERS),
        {water: meta["panels"][water]["driver_labels"] for water in WATERS})
    add("Five-domain legend placed between waterbody panels",
        meta["legend_count"] == 5, meta["legend_count"])
    add("RMS titles use one line above both response fields",
        all(
            meta["panels"][water]["metric_title"]
            == "13-range RMS response strength"
            and "\n" not in meta["panels"][water]["metric_title"]
            for water in WATERS
        ),
        {water: meta["panels"][water]["metric_title"] for water in WATERS})
    add("RMS values are finite and within plotting range",
        np.isfinite(selected.response_rms_13range).all()
        and selected.response_rms_13range.between(0, X_MAX).all(),
        (selected.response_rms_13range.min(), selected.response_rms_13range.max()))
    add("All five ecological-dimension colours available",
        set(selected.response_family).issubset(base.FAMILY_COLORS)
        and set(base.FAMILIES).issubset(base.FAMILY_COLORS),
        {family: base.FAMILY_COLORS[family] for family in base.FAMILIES})
    add("All three driver-label colours available",
        set(MODULES).issubset(base.DRIVER_COLORS),
        {module: base.DRIVER_COLORS[module] for module in MODULES})
    add("Signed median rho available for all plotted entries",
        np.isfinite(selected.signed_median_rho_13range).all(),
        (selected.signed_median_rho_13range.min(),
         selected.signed_median_rho_13range.max()))
    add("Signed median rho colour scale does not clip selected values",
        selected.signed_median_rho_13range.between(
            B_DIRECTION_NORM.vmin, B_DIRECTION_NORM.vmax
        ).all(),
        {
            "data": (
                selected.signed_median_rho_13range.min(),
                selected.signed_median_rho_13range.max(),
            ),
            "scale": (B_DIRECTION_NORM.vmin, B_DIRECTION_NORM.vmax),
        })
    add("Rank boundary audit covers six groups",
        len(boundary) == 6 and (boundary.rank5_minus_rank6 >= 0).all(),
        boundary.rank5_minus_rank6.min())

    group = associations.groupby(
        ["water_body_type", "column_name", "environment_module"]
    ).agg(
        range_n=("range_m", "size"),
        unique_range_n=("range_m", "nunique"),
        recomputed_rms=(
            "partial_spearman_rho",
            lambda values: float(np.sqrt(np.mean(np.square(values)))),
        ),
        recomputed_signed_median=("partial_spearman_rho", "median"),
    ).reset_index()
    audit = selected.merge(
        group,
        on=["water_body_type", "column_name", "environment_module"],
        how="left",
        validate="one_to_one",
    )
    max_diff = float(
        np.max(np.abs(audit.response_rms_13range - audit.recomputed_rms))
    )
    add("Every selected RMS uses exactly 13 unique ranges",
        bool((audit.range_n == 13).all() and (audit.unique_range_n == 13).all()),
        {"min": int(audit.range_n.min()), "max": int(audit.range_n.max())})
    add("Displayed RMS matches recomputation from range-level rho",
        max_diff < 1e-12, max_diff)
    direction_diff = float(
        np.max(
            np.abs(
                audit.signed_median_rho_13range
                - audit.recomputed_signed_median
            )
        )
    )
    add("Direction colour statistic matches recomputed signed median rho",
        direction_diff < 1e-12, direction_diff)

    for output in outputs:
        add(
            f"Output exists {output.suffix}",
            output.exists() and output.stat().st_size > 0,
            output.stat().st_size if output.exists() else 0,
        )
        if output.suffix == ".svg" and output.exists():
            svg = output.read_text(encoding="utf-8")
            add("SVG retains editable text", "<text" in svg, svg.count("<text"))
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [SCRIPT_DIR, DATA_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for extension in ["PNG", "SVG", "PDF", "TIFF"]:
        (FIG_DIR / extension).mkdir(parents=True, exist_ok=True)

    selected, boundary = load_and_select()
    associations = pd.read_csv(ASSOCIATION_SOURCE)
    fig, meta = plot_panel_b(selected)
    stem = "Fig5b_top5_driver_responsive_characteristics"
    outputs = {
        "png": FIG_DIR / "PNG" / f"{stem}.png",
        "svg": FIG_DIR / "SVG" / f"{stem}.svg",
        "pdf": FIG_DIR / "PDF" / f"{stem}.pdf",
        "tiff": FIG_DIR / "TIFF" / f"{stem}.tiff",
    }
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    source_columns = [
        "water_body_type", "environment_module", "driver_label",
        "selection_rank", "candidate_pool_n", "response_family", "family_label",
        "taxon_group", "column_name", "display_label", "response_rms_13range",
        "signed_median_rho_13range", "peak_abs_rho",
        "strict_supported_ranges", "response_supported_ranges",
    ]
    selected[source_columns].to_csv(
        DATA_DIR / "Fig5b_top5_driver_responsive_characteristics.csv",
        index=False, encoding="utf-8-sig",
    )
    boundary.to_csv(
        DATA_DIR / "Fig5b_rank5_rank6_boundary_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    qa = run_qa(selected, boundary, associations, meta, list(outputs.values()))
    qa.to_csv(QA_DIR / "Fig5b_automated_QA.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    summary = {
        "backend": "Python",
        "selection": "top 5 within each waterbody x driver by 13-range RMS",
        "candidate_pools": {"River": 178, "Lake": 144},
        "plotted_entries": int(len(selected)),
        "minimum_rank5_rank6_gap": float(boundary.rank5_minus_rank6.min()),
        "all_qa_pass": bool(qa.passed.all()),
    }
    (QA_DIR / "Fig5b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
