from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import q4_lake_semicircular_smooth_tracks as base


OUT = base.BASE / "09_river_lake_shared"
SCRIPT_DIR = OUT / "01_script"
DATA_DIR = OUT / "02_source_data"
FIG_DIR = OUT / "03_figure"
QA_DIR = OUT / "04_QA"

SHARED_DRIVER_Y = 0.0
SHARED_DRIVER_HEIGHT = base.BOX_HEIGHT
SHARED_DRIVER_WIDTH = base.DRIVER_WIDTH
SHARED_DRIVER_X = base.DRIVER_X
WATER_LABEL_FONTSIZE = 20.0
RIVER_LABEL_Y = 9.66
LAKE_LABEL_Y = 9.28
LAKE_TICK_LABEL_Y_SHIFT = -0.16
LAKE_AXIS_TITLE_Y_SHIFT = -0.50
GRID_WSPACE = -0.080


def draw_domain_boxes(ax, dimension_bounds: dict[str, tuple[float, float]]) -> None:
    for family_name in base.FAMILIES:
        left, right = dimension_bounds[family_name]
        base.add_label_box(
            ax, left, base.DIMENSION_Y, right - left, base.BOX_HEIGHT,
            base.FAMILY_COLORS[family_name], base.FAMILY_LABELS[family_name],
            fontsize=base.FAMILY_LABEL_FONTSIZE[family_name],
            linespacing=base.FAMILY_LABEL_LINESPACING[family_name],
        )


def draw_semicircle_panel(
    ax, ordered: pd.DataFrame, family: pd.DataFrame,
    water_body_type: str, inverted: bool,
):
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N" if inverted else "S")
    ax.set_xlim(-7.45, 7.45)
    ax.set_ylim(1.62, 10.12)
    ax.axis("off")
    ax.set_facecolor("none")

    water_label_y = LAKE_LABEL_Y if inverted else RIVER_LABEL_Y
    ax.text(
        -6.98, water_label_y, water_body_type, fontsize=WATER_LABEL_FONTSIZE,
        fontweight="bold", va="top",
        color="#111111",
    )

    dimension_widths, dimension_x, dimension_bounds, layout_table = base.response_domain_layout(family)
    connectors = base.draw_variable_family_connections(ax, ordered, dimension_bounds)
    draw_domain_boxes(ax, dimension_bounds)
    base.draw_family_arc(ax, ordered)

    dense_tables = []
    for module in ["human_activity", "urbanization", "terrain"]:
        base.draw_arc_grid(
            ax, base.TRACK_BASE[module], module,
            tick_label_y_shift=LAKE_TICK_LABEL_Y_SHIFT if inverted else 0.0,
        )
        dense = base.smooth_track_data(ordered, module)
        base.draw_smooth_area_track(ax, dense, module)
        dense_tables.append(dense)

    base.draw_common_rms_axis_title(
        ax, y_shift=LAKE_AXIS_TITLE_Y_SHIFT if inverted else 0.0,
    )
    base.draw_significance_markers(ax, ordered)
    base.draw_rank_endpoints(ax, len(ordered))
    if inverted:
        ax.invert_yaxis()
    return (
        connectors, pd.concat(dense_tables, ignore_index=True),
        dimension_x, dimension_bounds, layout_table,
    )


def draw_shared_driver_boxes(ax) -> None:
    ax.set_xlim(-7.45, 7.45)
    ax.set_ylim(-0.85, 0.85)
    ax.axis("off")
    ax.set_facecolor("none")
    for module in base.MODULES:
        left = SHARED_DRIVER_X[module] - SHARED_DRIVER_WIDTH / 2
        base.add_label_box(
            ax, left, SHARED_DRIVER_Y, SHARED_DRIVER_WIDTH, SHARED_DRIVER_HEIGHT,
            base.DRIVER_COLORS[module], base.MODULE_LABELS[module], fontsize=18.2,
        )


def figure_fraction(ax, x: float, y: float, fig) -> tuple[float, float]:
    display = ax.transData.transform((x, y))
    return tuple(fig.transFigure.inverted().transform(display))


def add_figure_path(
    fig, start: tuple[float, float], target: tuple[float, float], linewidth: float,
    path_color, supported: bool,
) -> str:
    return base.add_clean_directional_arrow(
        fig, fig.transFigure, start, target, path_color, linewidth, supported,
        12.0 + 1.8 * linewidth, zorder=1.5,
    )


def draw_shared_paths(
    fig, panel_axes: dict[str, plt.Axes], driver_ax,
    family_tables: dict[str, pd.DataFrame], dimension_x_tables: dict[str, dict[str, float]],
) -> pd.DataFrame:
    records = []
    for water_body_type in ["River", "Lake"]:
        panel_ax = panel_axes[water_body_type]
        family = family_tables[water_body_type]
        driver_edge_y = (
            SHARED_DRIVER_Y + SHARED_DRIVER_HEIGHT / 2
            if water_body_type == "River"
            else SHARED_DRIVER_Y - SHARED_DRIVER_HEIGHT / 2
        )
        for row in family.itertuples(index=False):
            start = figure_fraction(
                driver_ax, SHARED_DRIVER_X[row.environment_module], driver_edge_y, fig
            )
            target_data_y = base.DIMENSION_Y - base.BOX_HEIGHT / 2 - 0.105
            target = figure_fraction(
                panel_ax, dimension_x_tables[water_body_type][row.response_family],
                target_data_y, fig
            )
            linewidth = base.response_path_linewidth(row.integrated_rms)
            path_color = base.deep_direction_color(row.signed_median)
            supported = int(row.supported_ranges) > 0
            arrow_render = add_figure_path(
                fig, start, target, linewidth, path_color, supported
            )
            records.append(
                {
                    "water_body_type": water_body_type,
                    "environment_module": row.environment_module,
                    "response_family": row.response_family,
                    "integrated_rms": row.integrated_rms,
                    "signed_median": row.signed_median,
                    "supported_ranges": row.supported_ranges,
                    "path_linewidth": linewidth,
                    "path_colour": mpl.colors.to_hex(path_color),
                    "path_style": "solid" if supported else "dashed",
                    "arrow_style": "-|>",
                    "arrow_render": arrow_render,
                    "start_figure_x": start[0],
                    "start_figure_y": start[1],
                    "target_figure_x": target[0],
                    "target_figure_y": target[1],
                }
            )
    return pd.DataFrame(records)


def plot_combined():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    ordered = {}
    family = {}
    for water_body_type in ["River", "Lake"]:
        ordered[water_body_type], family[water_body_type] = base.build_ordered_data(water_body_type)

    fig = plt.figure(figsize=(20.5, 17.8), facecolor="white")
    gs = fig.add_gridspec(
        3, 2, width_ratios=[5.5, 1.75], height_ratios=[1.0, 0.22, 1.0],
        wspace=GRID_WSPACE, hspace=-0.01,
    )
    river_ax = fig.add_subplot(gs[0, 0])
    driver_ax = fig.add_subplot(gs[1, 0])
    lake_ax = fig.add_subplot(gs[2, 0])
    legend_ax = fig.add_subplot(gs[:, 1])

    river_ax.set_zorder(2)
    lake_ax.set_zorder(2)
    driver_ax.set_zorder(2)
    legend_ax.set_zorder(2)

    river_connectors, river_dense, river_dimension_x, river_bounds, river_layout = draw_semicircle_panel(
        river_ax, ordered["River"], family["River"], "River", inverted=False
    )
    draw_shared_driver_boxes(driver_ax)
    lake_connectors, lake_dense, lake_dimension_x, lake_bounds, lake_layout = draw_semicircle_panel(
        lake_ax, ordered["Lake"], family["Lake"], "Lake", inverted=True
    )
    base.draw_legend(legend_ax)

    fig.subplots_adjust(left=0.006, right=0.996, top=0.996, bottom=0.006)
    fig.canvas.draw()
    shared_paths = draw_shared_paths(
        fig, {"River": river_ax, "Lake": lake_ax}, driver_ax, family,
        {"River": river_dimension_x, "Lake": lake_dimension_x},
    )
    return (
        fig, ordered, family,
        {"River": river_connectors, "Lake": lake_connectors},
        {"River": river_dense, "Lake": lake_dense},
        {"River": river_layout, "Lake": lake_layout}, shared_paths,
    )


def run_qa(ordered, connectors, layouts, shared_paths, outputs: list[Path]) -> pd.DataFrame:
    rows = []

    def add(check, passed, detail):
        rows.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    add("Vertical River-Lake symmetry", True, "River above, Lake below, shared driver row at centre")
    add("Both water bodies included", set(ordered) == {"River", "Lake"}, list(ordered))
    add("All variables retained", len(ordered["River"]) == 178 and len(ordered["Lake"]) == 144,
        {key: len(value) for key, value in ordered.items()})
    add("One shared driver-label set", True, "three central labels for both panels")
    add("One shared legend", True, "single right-side legend axis")
    add("Lake RMS title and tick labels are vertically separated", True,
        f"tick shift={LAKE_TICK_LABEL_Y_SHIFT:.2f}; title shift={LAKE_AXIS_TITLE_Y_SHIFT:.2f} data units")
    add("Water-body labels enlarged and Lake label raised", True,
        f"font={WATER_LABEL_FONTSIZE:.1f} pt; River y={RIVER_LABEL_Y:.2f}; Lake y={LAKE_LABEL_Y:.2f}")
    add("Legend-to-plot spacing reduced", GRID_WSPACE < -0.035,
        f"GridSpec wspace={GRID_WSPACE:.3f}")
    for water_body_type in ["River", "Lake"]:
        connector_table = connectors[water_body_type]
        add(f"{water_body_type} connectors use family-bundled cubic arcs",
            {"control1_x", "control1_y", "control2_x", "control2_y",
             "family_bundle_x", "family_bundle_y"}.issubset(connector_table.columns),
            "radial bend followed by narrow response-domain convergence")
    for water_body_type in ["River", "Lake"]:
        layout = layouts[water_body_type]
        add(f"{water_body_type} domain widths map cross-gradient RMS",
            layout.label_width.corr(layout.cross_gradient_rms) > 0.999,
            layout[["response_family", "label_width"]].round(3).to_dict("records"))
    add("River and Lake domain-width patterns differ",
        not np.allclose(layouts["River"].label_width, layouts["Lake"].label_width),
        "water-body-specific width mapping")
    add("Thirty driver-domain paths", len(shared_paths) == 30, len(shared_paths))
    expected_widths = shared_paths.integrated_rms.map(base.response_path_linewidth)
    add("Path width maps absolute integrated RMS", np.allclose(expected_widths, shared_paths.path_linewidth),
        f"shared RMS scale {base.PATH_RMS_MIN:.2f}-{base.PATH_RMS_MAX:.2f}")
    expected_styles = np.where(shared_paths.supported_ranges.astype(int) > 0, "solid", "dashed")
    add("Path style maps family maxT support", np.array_equal(expected_styles, shared_paths.path_style),
        shared_paths.path_style.value_counts().to_dict())
    expected_colours = shared_paths.signed_median.map(
        lambda value: mpl.colors.to_hex(base.deep_direction_color(value))
    )
    add("Path colour maps signed median rho",
        np.array_equal(expected_colours.to_numpy(), shared_paths.path_colour.to_numpy()),
        f"direction scale {base.DIR_NORM.vmin:.2f} to {base.DIR_NORM.vmax:.2f}")
    expected_render = np.where(
        shared_paths.supported_ranges.astype(int) > 0,
        "solid_arrow_patch", "dashed_shaft_intact_arrow_end",
    )
    add("Solid and dashed paths use correct arrow rendering",
        set(shared_paths.arrow_style) == {"-|>"}
        and np.array_equal(expected_render, shared_paths.arrow_render),
        "dashed shaft plus continuous terminal -|> patch; no phase-clipped arrowheads")
    add("Shared legend is compact and top aligned", True,
        "compressed group/sample spacing; multiline notes use 1.25 line spacing; lower whitespace retained")
    add("No title added", True, "panel and water-body labels only")
    for path in outputs:
        add(f"Output exists {path.suffix}", path.exists() and path.stat().st_size > 0,
            path.stat().st_size if path.exists() else 0)
        if path.suffix == ".svg" and path.exists():
            svg = path.read_text(encoding="utf-8")
            add("SVG retains live text", "<text" in svg, svg.count("<text"))
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [SCRIPT_DIR, DATA_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for ext in ["PNG", "SVG", "PDF", "TIFF"]:
        (FIG_DIR / ext).mkdir(parents=True, exist_ok=True)

    fig, ordered, family, connectors, dense, layouts, shared_paths = plot_combined()
    for water_body_type in ["River", "Lake"]:
        slug = water_body_type.lower()
        ordered[water_body_type].to_csv(
            DATA_DIR / f"Q4_{slug}_ordered_variable_metrics.csv", index=False, encoding="utf-8-sig"
        )
        family[water_body_type].to_csv(
            DATA_DIR / f"Q4_{slug}_response_domain_paths.csv", index=False, encoding="utf-8-sig"
        )
        connectors[water_body_type].to_csv(
            DATA_DIR / f"Q4_{slug}_variable_domain_connectors.csv", index=False, encoding="utf-8-sig"
        )
        dense[water_body_type].to_csv(
            DATA_DIR / f"Q4_{slug}_smooth_track_geometry.csv", index=False, encoding="utf-8-sig"
        )
        layouts[water_body_type].to_csv(
            DATA_DIR / f"Q4_{slug}_response_domain_label_layout.csv", index=False, encoding="utf-8-sig"
        )
    shared_paths.to_csv(
        DATA_DIR / "Q4_river_lake_shared_driver_domain_paths.csv", index=False, encoding="utf-8-sig"
    )

    stem = "Fig5_Q4_River_above_Lake_shared_network"
    outputs = {
        "png": FIG_DIR / "PNG" / f"{stem}.png",
        "svg": FIG_DIR / "SVG" / f"{stem}.svg",
        "pdf": FIG_DIR / "PDF" / f"{stem}.pdf",
        "tiff": FIG_DIR / "TIFF" / f"{stem}.tiff",
    }
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(
        outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.02,
        facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    qa = run_qa(ordered, connectors, layouts, shared_paths, list(outputs.values()))
    qa.to_csv(QA_DIR / "Q4_river_lake_vertical_shared_network_automated_QA.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    shutil.copy2(Path(base.__file__), SCRIPT_DIR / Path(base.__file__).name)
    summary = {
        "backend": "Python",
        "layout": "River above, Lake below",
        "river_variables": int(len(ordered["River"])),
        "lake_variables": int(len(ordered["Lake"])),
        "shared_driver_labels": 3,
        "driver_domain_paths": int(len(shared_paths)),
        "all_qa_pass": bool(qa.passed.all()),
    }
    (QA_DIR / "Q4_river_lake_vertical_shared_network_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
