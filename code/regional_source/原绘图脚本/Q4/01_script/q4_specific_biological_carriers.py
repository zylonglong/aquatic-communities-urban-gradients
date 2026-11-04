from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy import stats


ROOT = Path(r"E:\桌面\博士课题\Data")
ANALYSIS_ROOT = ROOT / "manuscript" / "城市水生态四科学问题_尺度响应分析_2026-07-23"
RAW_BOOK = (
    ROOT
    / "data"
    / "processed"
    / "训练数据汇总"
    / "ML_Input_原始量_13尺度_2026-07-04"
    / "ML_Input_Raw_13Ranges_Sample_Site.xlsx"
)
MAPPING_BOOK = ANALYSIS_ROOT / "五类水生生物变量对应关系_Q2分析流程.xlsx"
CORE_SCRIPT = ANALYSIS_ROOT / "11_脚本与QA" / "01_primary_multiscale_analysis.py"

DELIVERY = (
    ANALYSIS_ROOT
    / "manuscript可视化图件"
    / "Figure5_Q4_specific_biological_carriers_2026-08-08"
)
SCRIPT_DIR = DELIVERY / "01_script"
SOURCE_DIR = DELIVERY / "02_source_data"
FIG_DIR = DELIVERY / "03_figure"
QA_DIR = DELIVERY / "04_QA"

RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
MODULES = ["terrain", "urbanization", "human_activity"]
FAMILIES = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
WATERS = ["River", "Lake"]

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
FAMILY_LABELS = {
    "community_baseline": "community-level attributes",
    "alpha_diversity": "α diversity",
    "beta_diversity": "β diversity",
    "community_composition": "community composition",
    "functional_indicator": "indicator/tolerance taxa",
}
FAMILY_COLORS = {
    "community_baseline": "#355C9A",
    "alpha_diversity": "#6F91C7",
    "beta_diversity": "#9E8ABF",
    "community_composition": "#D08A79",
    "functional_indicator": "#B45266",
}
WATER_COLORS = {"River": "#5067B0", "Lake": "#B54A79"}
DRIVER_COLORS = {
    "terrain": "#8B6844",
    "urbanization": "#6E879E",
    "human_activity": "#E17A5D",
}

N_PERM = 999
N_BOOT = 499
SEED = 20260808


def load_core():
    spec = importlib.util.spec_from_file_location("q4_core", CORE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bh_fdr(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    back = np.empty(len(ranked))
    back[order] = np.clip(adjusted, 0, 1)
    q[ok] = back
    return q


def pairwise_corr_matrix(y_matrix: np.ndarray, x_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson correlation between residual-rank matrices."""
    n_y = y_matrix.shape[1]
    n_x = x_matrix.shape[1]
    rho = np.full((n_y, n_x), np.nan)
    n_obs = np.zeros((n_y, n_x), dtype=int)
    for i in range(n_y):
        y = y_matrix[:, i]
        for j in range(n_x):
            x = x_matrix[:, j]
            ok = np.isfinite(y) & np.isfinite(x)
            n_obs[i, j] = int(ok.sum())
            if ok.sum() >= 8 and np.std(y[ok]) > 0 and np.std(x[ok]) > 0:
                rho[i, j] = np.corrcoef(y[ok], x[ok])[0, 1]
    return rho, n_obs


def fast_perm_corr(y_matrix: np.ndarray, x_matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Fast pairwise-complete correlations after a common row permutation of X."""
    y = y_matrix
    x = x_matrix[permutation, :]
    mask_y = np.isfinite(y).astype(float)
    mask_x = np.isfinite(x).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)
    x0 = np.where(np.isfinite(x), x, 0.0)
    n = mask_y.T @ mask_x
    sum_y = y0.T @ mask_x
    sum_x = mask_y.T @ x0
    sum_y2 = (y0 * y0).T @ mask_x
    sum_x2 = mask_y.T @ (x0 * x0)
    sum_xy = y0.T @ x0
    numerator = sum_xy - (sum_y * sum_x) / np.maximum(n, 1)
    y_ss = np.maximum(sum_y2 - (sum_y * sum_y) / np.maximum(n, 1), 0)
    x_ss = np.maximum(sum_x2 - (sum_x * sum_x) / np.maximum(n, 1), 0)
    denominator = np.sqrt(y_ss * x_ss)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=(denominator > 0) & (n >= 8),
    )


def partial_pvalue(rho: float, n: int, n_controls: int = 6) -> float:
    if not np.isfinite(rho) or abs(rho) >= 1 or n <= n_controls + 2:
        return np.nan
    df = n - n_controls - 2
    t_stat = rho * math.sqrt(df / max(1e-15, 1 - rho * rho))
    return float(2 * stats.t.sf(abs(t_stat), df))


def normalize_taxon(value: str) -> str:
    v = str(value)
    return "macrophyte" if v == "macrophytes" else v


def clean_variable_label(name: str) -> str:
    text = name
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
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("macrophytes", "macrophyte")
    text = text.replace("qual pa", "presence/absence")
    text = text.replace("quant", "quantitative")
    text = text.replace("density Bray–Curtis", "abundance Bray–Curtis")
    text = text.replace("count Bray–Curtis", "abundance Bray–Curtis")
    text = text.replace("total density", "total abundance")
    return text[0].upper() + text[1:] if text else name


def controls_for(core, frame: pd.DataFrame) -> np.ndarray:
    return core.controls_for(frame)


def residual_matrices(core, frame: pd.DataFrame, variables: list[str], score_lookup: dict) -> tuple[np.ndarray, np.ndarray]:
    controls = controls_for(core, frame)
    y_cols = []
    for variable in variables:
        y = pd.to_numeric(frame[variable], errors="coerce").to_numpy(float)
        y_cols.append(core.rank_residual(y, controls))
    x_cols = []
    for module in MODULES:
        for range_m in RANGES:
            score = score_lookup[(module, range_m)].reindex(frame.index).to_numpy(float)
            x_cols.append(core.rank_residual(score, controls))
    return np.column_stack(y_cols), np.column_stack(x_cols)


def contribution_from_rho(rho: np.ndarray, variable_families: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-module contribution; each module contributes one third within family."""
    variable_families = np.asarray(variable_families)
    contribution_rows = []
    module_rows = []
    for family in FAMILIES:
        idx = np.where(variable_families == family)[0]
        if not len(idx):
            continue
        module_shares = []
        for mi, module in enumerate(MODULES):
            cols = slice(mi * len(RANGES), (mi + 1) * len(RANGES))
            energy = np.nansum(rho[idx, cols] ** 2, axis=1)
            total = np.nansum(energy)
            share = energy / total if total > 0 else np.repeat(np.nan, len(energy))
            module_shares.append(share)
            for local_i, global_i in enumerate(idx):
                module_rows.append(
                    {
                        "variable_index": int(global_i),
                        "response_family": family,
                        "environment_module": module,
                        "module_contribution_share": float(share[local_i]),
                    }
                )
        balanced = np.nanmean(np.column_stack(module_shares), axis=1)
        balanced = balanced / np.nansum(balanced)
        for local_i, global_i in enumerate(idx):
            contribution_rows.append(
                {
                    "variable_index": int(global_i),
                    "response_family": family,
                    "balanced_contribution_share": float(balanced[local_i]),
                }
            )
    return pd.DataFrame(contribution_rows), pd.DataFrame(module_rows)


def calculate_water(core, site: pd.DataFrame, reps: pd.DataFrame, scores: pd.DataFrame, water: str):
    rng = np.random.default_rng(SEED + (0 if water == "River" else 10000))
    frame = site.loc[site.water_body_type.eq(water)].copy()
    meta = reps.loc[reps.water_body_type.eq(water)].copy()
    meta = meta.drop_duplicates(["response_family", "column_name"]).reset_index(drop=True)
    meta = meta.loc[meta.column_name.isin(frame.columns)].copy().reset_index(drop=True)
    # Guard against aliases that survived the Q2 mapping but contain exactly the
    # same site-level values. Keeping both would double-count one ecological signal.
    kept_rows = []
    duplicate_rows = []
    for family, group in meta.groupby("response_family", sort=False):
        kept_columns = []
        for idx in group.index:
            column = meta.at[idx, "column_name"]
            values = pd.to_numeric(frame[column], errors="coerce")
            duplicate_of = None
            for prior in kept_columns:
                prior_values = pd.to_numeric(frame[prior], errors="coerce")
                same_missingness = values.isna().equals(prior_values.isna())
                observed = values.notna() & prior_values.notna()
                if same_missingness and observed.sum() >= 8 and np.allclose(
                    values[observed], prior_values[observed], rtol=0, atol=1e-12
                ):
                    duplicate_of = prior
                    break
            if duplicate_of is None:
                kept_rows.append(idx)
                kept_columns.append(column)
            else:
                duplicate_rows.append(
                    {
                        "water_body_type": water,
                        "response_family": family,
                        "dropped_exact_duplicate": column,
                        "retained_variable": duplicate_of,
                        "n_sites_compared": int(observed.sum()),
                    }
                )
    meta = meta.loc[kept_rows].reset_index(drop=True)
    duplicate_audit = pd.DataFrame(duplicate_rows)
    variables = meta.column_name.tolist()
    families = meta.response_family.tolist()
    score_lookup = {}
    water_scores = scores.loc[scores.water_body_type.eq(water)]
    for module in MODULES:
        for range_m in RANGES:
            score_lookup[(module, range_m)] = (
                water_scores.loc[
                    water_scores.environment_module.eq(module) & water_scores.range_m.eq(range_m)
                ]
                .set_index("row_index")["environment_score"]
            )

    y_res, x_res = residual_matrices(core, frame, variables, score_lookup)
    observed_rho, n_obs = pairwise_corr_matrix(y_res, x_res)
    contribution, contribution_module = contribution_from_rho(observed_rho, families)
    meta_for_merge = (
        meta.reset_index()
        .rename(columns={"index": "variable_index"})
        .drop(columns=["response_family"])
    )
    contribution = contribution.merge(meta_for_merge, on="variable_index", how="left")
    contribution_module = contribution_module.merge(meta_for_merge, on="variable_index", how="left")
    contribution["observed_rank_within_family"] = contribution.groupby("response_family")[
        "balanced_contribution_share"
    ].rank(method="first", ascending=False)
    selected = contribution.loc[contribution.observed_rank_within_family <= 3].copy()
    selected_indices = selected.variable_index.astype(int).tolist()

    boot_contrib = np.full((N_BOOT, len(variables)), np.nan)
    boot_rank = np.full((N_BOOT, len(variables)), np.nan)
    boot_selected_rho = np.full((N_BOOT, len(selected_indices), len(MODULES) * len(RANGES)), np.nan)
    for b in range(N_BOOT):
        sample_pos = rng.integers(0, len(frame), len(frame))
        boot_frame = frame.iloc[sample_pos].copy()
        boot_frame.index = np.arange(len(boot_frame))
        boot_scores = {}
        original_index = frame.index.to_numpy()[sample_pos]
        for key, series in score_lookup.items():
            boot_scores[key] = pd.Series(series.reindex(original_index).to_numpy(), index=boot_frame.index)
        yb, xb = residual_matrices(core, boot_frame, variables, boot_scores)
        rb, _ = pairwise_corr_matrix(yb, xb)
        cb, _ = contribution_from_rho(rb, families)
        for row in cb.itertuples(index=False):
            boot_contrib[b, row.variable_index] = row.balanced_contribution_share
        for family in FAMILIES:
            idx = np.where(np.asarray(families) == family)[0]
            if len(idx):
                vals = boot_contrib[b, idx]
                order = np.argsort(np.where(np.isfinite(vals), -vals, np.inf))
                ranks = np.empty(len(idx), dtype=float)
                ranks[order] = np.arange(1, len(idx) + 1)
                boot_rank[b, idx] = ranks
        boot_selected_rho[b, :, :] = rb[selected_indices, :]

    contribution["bootstrap_median_contribution"] = np.nanmedian(boot_contrib, axis=0)
    contribution["bootstrap_ci95_low"] = np.nanpercentile(boot_contrib, 2.5, axis=0)
    contribution["bootstrap_ci95_high"] = np.nanpercentile(boot_contrib, 97.5, axis=0)
    contribution["bootstrap_rank_median"] = np.nanmedian(boot_rank, axis=0)
    contribution["bootstrap_rank_iqr"] = np.nanpercentile(boot_rank, 75, axis=0) - np.nanpercentile(
        boot_rank, 25, axis=0
    )
    contribution["top3_selection_frequency"] = np.nanmean(boot_rank <= 3, axis=0)
    contribution["selection_stability"] = pd.cut(
        contribution.top3_selection_frequency,
        bins=[-np.inf, 0.49, 0.69, np.inf],
        labels=["low", "moderate", "high"],
    ).astype(str)

    # Common-permutation maxT: response-wise across 13 ranges and family-wise
    # across every frozen Q2 representative response and all 13 ranges.
    perm_raw = np.full_like(observed_rho, np.nan)
    perm_response_maxt = np.full_like(observed_rho, np.nan)
    perm_family_maxt = np.full_like(observed_rho, np.nan)
    for family in FAMILIES:
        y_idx = np.where(np.asarray(families) == family)[0]
        if not len(y_idx):
            continue
        for mi, module in enumerate(MODULES):
            x_idx = np.arange(mi * len(RANGES), (mi + 1) * len(RANGES))
            obs = np.abs(observed_rho[np.ix_(y_idx, x_idx)])
            exceed_raw = np.zeros_like(obs, dtype=int)
            exceed_response = np.zeros_like(obs, dtype=int)
            exceed_family = np.zeros_like(obs, dtype=int)
            for _ in range(N_PERM):
                rp = np.abs(fast_perm_corr(y_res[:, y_idx], x_res[:, x_idx], rng.permutation(len(frame))))
                exceed_raw += rp >= obs
                response_max = np.nanmax(rp, axis=1)[:, None]
                family_max = float(np.nanmax(rp))
                exceed_response += response_max >= obs
                exceed_family += family_max >= obs
            perm_raw[np.ix_(y_idx, x_idx)] = (1 + exceed_raw) / (N_PERM + 1)
            perm_response_maxt[np.ix_(y_idx, x_idx)] = (1 + exceed_response) / (N_PERM + 1)
            perm_family_maxt[np.ix_(y_idx, x_idx)] = (1 + exceed_family) / (N_PERM + 1)

    assoc_rows = []
    selected_pos = {idx: pos for pos, idx in enumerate(selected_indices)}
    for vi, row in meta.iterrows():
        for mi, module in enumerate(MODULES):
            for ri, range_m in enumerate(RANGES):
                ci_low = ci_high = np.nan
                if vi in selected_pos:
                    values = boot_selected_rho[:, selected_pos[vi], mi * len(RANGES) + ri]
                    if np.isfinite(values).any():
                        ci_low, ci_high = np.nanpercentile(values, [2.5, 97.5])
                rho = observed_rho[vi, mi * len(RANGES) + ri]
                assoc_rows.append(
                    {
                        "water_body_type": water,
                        "response_family": row.response_family,
                        "response_family_cn": row.response_family_cn,
                        "taxon_group": row.taxon_group,
                        "taxon_group_cn": row.taxon_group_cn,
                        "column_name": row.column_name,
                        "variable_name_cn": row.variable_name_cn,
                        "environment_module": module,
                        "range_m": range_m,
                        "partial_spearman_rho": rho,
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "n_sites": int(n_obs[vi, mi * len(RANGES) + ri]),
                        "parametric_p": partial_pvalue(rho, int(n_obs[vi, mi * len(RANGES) + ri])),
                        "permutation_p": perm_raw[vi, mi * len(RANGES) + ri],
                        "response_13range_maxT_p": perm_response_maxt[vi, mi * len(RANGES) + ri],
                        "family_variables_13range_maxT_p": perm_family_maxt[vi, mi * len(RANGES) + ri],
                        "selected_top3": vi in selected_pos,
                    }
                )
    assoc = pd.DataFrame(assoc_rows)
    assoc["q_within_response_13ranges"] = np.nan
    for _, ix in assoc.groupby(["environment_module", "column_name"]).groups.items():
        assoc.loc[ix, "q_within_response_13ranges"] = bh_fdr(assoc.loc[ix, "parametric_p"].to_numpy())
    assoc["q_within_family_all_representatives_ranges"] = np.nan
    for _, ix in assoc.groupby(["environment_module", "response_family"]).groups.items():
        assoc.loc[ix, "q_within_family_all_representatives_ranges"] = bh_fdr(
            assoc.loc[ix, "parametric_p"].to_numpy()
        )

    # Family share of the total five-domain biological signal.
    family_rows = []
    for module in MODULES:
        for range_m in RANGES:
            cell = assoc.loc[
                assoc.environment_module.eq(module) & assoc.range_m.eq(range_m)
            ]
            energy = {}
            for family in FAMILIES:
                values = cell.loc[cell.response_family.eq(family), "partial_spearman_rho"].to_numpy(float)
                energy[family] = float(np.nanmean(values * values)) if np.isfinite(values).any() else np.nan
            total = np.nansum(list(energy.values()))
            for family, value in energy.items():
                family_rows.append(
                    {
                        "water_body_type": water,
                        "environment_module": module,
                        "range_m": range_m,
                        "response_family": family,
                        "rms_partial_rho": math.sqrt(value) if np.isfinite(value) else np.nan,
                        "family_signal_share": value / total if total > 0 else np.nan,
                    }
                )
    family_share = pd.DataFrame(family_rows)

    # Taxon shares are first averaged within taxon x family, then normalized
    # among taxa so that groups with more derived variables do not gain weight.
    taxon_rows = []
    for (module, range_m, family), cell in assoc.groupby(
        ["environment_module", "range_m", "response_family"]
    ):
        taxon_energy = (
            cell.assign(energy=cell.partial_spearman_rho**2)
            .groupby(["taxon_group", "taxon_group_cn"], as_index=False)
            .energy.mean()
        )
        total = taxon_energy.energy.sum()
        for row in taxon_energy.itertuples(index=False):
            taxon_rows.append(
                {
                    "water_body_type": water,
                    "environment_module": module,
                    "range_m": range_m,
                    "response_family": family,
                    "taxon_group": row.taxon_group,
                    "taxon_group_cn": row.taxon_group_cn,
                    "taxon_signal_share_within_family": row.energy / total if total > 0 else np.nan,
                }
            )
    taxon_share = pd.DataFrame(taxon_rows)

    return assoc, contribution, contribution_module, family_share, taxon_share, duplicate_audit


def prepare_metadata(mapping: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "水体类型": "water_body_type",
        "响应家族代码": "response_family",
        "响应家族中文": "response_family_cn",
        "生物类群代码": "taxon_group",
        "生物类群中文": "taxon_group_cn",
        "最终代表变量": "column_name",
        "变量中文名": "variable_name_cn",
        "代表的候选变量数": "represented_candidate_n",
        "非缺失site数": "mapping_nonmissing_sites",
    }
    out = mapping.rename(columns=rename).copy()
    out["taxon_group"] = out.taxon_group.map(normalize_taxon)
    out.loc[out.taxon_group.eq("macrophyte"), "taxon_group_cn"] = "水生植物"
    return out[
        [
            "water_body_type",
            "response_family",
            "response_family_cn",
            "taxon_group",
            "taxon_group_cn",
            "column_name",
            "variable_name_cn",
            "represented_candidate_n",
            "mapping_nonmissing_sites",
        ]
    ]


def plot_figure(family_share: pd.DataFrame, contribution: pd.DataFrame, assoc: pd.DataFrame):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.3,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    selected = contribution.loc[contribution.observed_rank_within_family <= 3].copy()
    selected["label"] = selected.column_name.map(clean_variable_label)
    selected["water_order"] = selected.water_body_type.map({"River": 0, "Lake": 1})
    selected["family_order"] = selected.response_family.map({f: i for i, f in enumerate(FAMILIES)})
    selected = selected.sort_values(
        ["water_order", "family_order", "observed_rank_within_family"]
    ).reset_index(drop=True)

    fig = plt.figure(figsize=(7.2, 9.0), constrained_layout=False)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.35, 3.2, 5.0], hspace=0.34)

    # a: mean ecological-domain carrier shares, range-averaged within each driver.
    ax_a = fig.add_subplot(outer[0, 0])
    summary = (
        family_share.groupby(["water_body_type", "environment_module", "response_family"], as_index=False)
        .family_signal_share.mean()
    )
    y_labels = []
    y = np.arange(6)[::-1]
    left = np.zeros(6)
    ordered_cells = [(w, m) for w in WATERS for m in MODULES]
    for family in FAMILIES:
        vals = []
        for water, module in ordered_cells:
            hit = summary.loc[
                summary.water_body_type.eq(water)
                & summary.environment_module.eq(module)
                & summary.response_family.eq(family),
                "family_signal_share",
            ]
            vals.append(float(hit.iloc[0]) if len(hit) else 0.0)
        vals = np.asarray(vals)
        ax_a.barh(y, vals, left=left, height=0.68, color=FAMILY_COLORS[family], edgecolor="white", linewidth=0.35)
        left += vals
    for water, module in ordered_cells:
        y_labels.append(f"{water} · {MODULE_SHORT[module]}")
    ax_a.set_yticks(y, y_labels)
    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax_a.set_xlabel("Mean share of five-domain biological response strength (%)")
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in FAMILIES]
    ax_a.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=5.8)
    ax_a.text(-0.12, 1.14, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold", va="top")
    ax_a.text(-0.08, 1.14, "Ecological dimensions carrying the overall biological signal", transform=ax_a.transAxes, fontsize=8.2, fontweight="bold", va="top")

    # b: contribution and bootstrap selection stability.
    gs_b = outer[1, 0].subgridspec(1, 2, wspace=0.44)
    for wi, water in enumerate(WATERS):
        ax = fig.add_subplot(gs_b[0, wi])
        sub = selected.loc[selected.water_body_type.eq(water)].copy().reset_index(drop=True)
        positions = np.arange(len(sub))[::-1]
        for pos, row in zip(positions, sub.itertuples(index=False)):
            x = 100 * row.balanced_contribution_share
            lo = 100 * row.bootstrap_ci95_low
            hi = 100 * row.bootstrap_ci95_high
            color = FAMILY_COLORS[row.response_family]
            ax.plot([lo, hi], [pos, pos], color=color, linewidth=1.2, alpha=0.8)
            face = color if row.top3_selection_frequency >= 0.70 else "white"
            ax.scatter(x, pos, s=24, facecolor=face, edgecolor=color, linewidth=1.0, zorder=3)
            ax.text(
                max(hi, x) + 0.35,
                pos,
                f"{100 * row.top3_selection_frequency:.0f}%",
                va="center",
                fontsize=5.1,
                color="#555555",
            )
        labels = [f"{FAMILY_LABELS[r.response_family]} · {r.label}" for r in sub.itertuples(index=False)]
        ax.set_yticks(positions, labels)
        ax.tick_params(axis="y", length=0, labelsize=5.35)
        ax.set_xlabel("Balanced contribution to family signal (%)")
        ax.set_title(water, color=WATER_COLORS[water], fontsize=7.5, fontweight="bold", pad=3)
        ax.axvline(0, color="#777777", linewidth=0.5)
        xmax = max(12.0, 100 * sub.bootstrap_ci95_high.max() + 6)
        ax.set_xlim(0, xmax)
        if wi == 0:
            ax.text(-0.28, 1.08, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
            ax.text(-0.20, 1.08, "Contribution and bootstrap selection stability", transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")
        ax.text(0.99, 1.01, "label = Top 3 frequency", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color="#555555")

    # c,d: multiscale heatmaps for the selected variables.
    gs_c = outer[2, 0].subgridspec(1, 2, wspace=0.46)
    cmap = mpl.colormaps["coolwarm"]
    norm = Normalize(vmin=-0.60, vmax=0.60)
    for wi, water in enumerate(WATERS):
        ax = fig.add_subplot(gs_c[0, wi])
        sub = selected.loc[selected.water_body_type.eq(water)].copy().reset_index(drop=True)
        rows = []
        maxT = []
        response_maxT = []
        for row in sub.itertuples(index=False):
            g = assoc.loc[assoc.water_body_type.eq(water) & assoc.column_name.eq(row.column_name)].copy()
            g["module_order"] = g.environment_module.map({m: i for i, m in enumerate(MODULES)})
            g["range_order"] = g.range_m.map({r: i for i, r in enumerate(RANGES)})
            g = g.sort_values(["module_order", "range_order"])
            rows.append(g.partial_spearman_rho.to_numpy(float))
            maxT.append(g.family_variables_13range_maxT_p.to_numpy(float) <= 0.05)
            response_maxT.append(g.response_13range_maxT_p.to_numpy(float) <= 0.05)
        matrix = np.vstack(rows)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        family_maxT = np.vstack(maxT)
        response_support = np.vstack(response_maxT)
        yy, xx = np.where(response_support & ~family_maxT)
        ax.scatter(xx, yy, s=5, facecolor="none", edgecolor="#333333", linewidth=0.35)
        yy, xx = np.where(family_maxT)
        ax.scatter(xx, yy, s=5.5, facecolor="#111111", edgecolor="#111111", linewidth=0.25)
        for divider in [12.5, 25.5]:
            ax.axvline(divider, color="white", linewidth=1.4)
            ax.axvline(divider, color="#666666", linewidth=0.35)
        tick_positions = []
        tick_labels = []
        for mi, module in enumerate(MODULES):
            for ri, range_m in enumerate(RANGES):
                tick_positions.append(mi * len(RANGES) + ri)
                tick_labels.append(str(range_m / 1000).rstrip("0").rstrip("."))
        ax.set_xticks(tick_positions, tick_labels, rotation=55, ha="right", fontsize=4.6)
        labels = [f"{FAMILY_LABELS[r.response_family]} · {r.label}" for r in sub.itertuples(index=False)]
        ax.set_yticks(np.arange(len(sub)), labels, fontsize=5.15)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Spatial range (km)")
        for mi, module in enumerate(MODULES):
            ax.text(mi * 13 + 6, -1.15, MODULE_LABELS[module], ha="center", va="bottom", fontsize=5.5, fontweight="bold", color=DRIVER_COLORS[module])
        ax.set_title(water, color=WATER_COLORS[water], fontsize=7.5, fontweight="bold", pad=17)
        panel = "c" if wi == 0 else "d"
        title = "River trait-scale responses" if wi == 0 else "Lake trait-scale responses"
        ax.text(-0.28, 1.09, panel, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.text(-0.20, 1.09, title, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_axes([0.37, 0.035, 0.26, 0.013])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("Partial Spearman ρ", labelpad=1)
    cb.set_ticks([-0.6, 0, 0.6])
    fig.text(0.14, 0.041, "○ response-wise 13-range maxT P ≤ 0.05", fontsize=5.4, ha="center")
    fig.text(0.82, 0.041, "● family-wise variables × ranges maxT P ≤ 0.05", fontsize=5.4, ha="center")
    fig.subplots_adjust(left=0.21, right=0.985, top=0.96, bottom=0.075)
    return fig


def run_qa(assoc, contribution, family_share, selected, png_path: Path, svg_path: Path, tiff_path: Path):
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("Frozen Q2 representatives only", contribution.column_name.notna().all(), f"n={len(contribution)}")
    share_sums = family_share.groupby(["water_body_type", "environment_module", "range_m"]).family_signal_share.sum()
    add("Five-family shares sum to one", np.allclose(share_sums, 1, atol=1e-10), f"max deviation={float((share_sums-1).abs().max()):.3g}")
    contrib_sums = contribution.groupby(["water_body_type", "response_family"]).balanced_contribution_share.sum()
    add("Within-family contribution sums to one", np.allclose(contrib_sums, 1, atol=1e-10), f"max deviation={float((contrib_sums-1).abs().max()):.3g}")
    selected_counts = selected.groupby(["water_body_type", "response_family"]).size()
    add("Exactly three selected per water x family", bool((selected_counts == 3).all() and len(selected_counts) == 10), selected_counts.to_dict())
    add("Permutation probabilities bounded", assoc[["permutation_p", "response_13range_maxT_p", "family_variables_13range_maxT_p"]].apply(lambda s: s.dropna().between(0, 1).all()).all(), "0 <= p <= 1")
    add("Selected CIs available", assoc.loc[assoc.selected_top3, ["bootstrap_ci95_low", "bootstrap_ci95_high"]].notna().all().all(), f"rows={int(assoc.selected_top3.sum())}")
    add("PNG exists", png_path.exists() and png_path.stat().st_size > 0, str(png_path.stat().st_size if png_path.exists() else 0))
    add("SVG exists", svg_path.exists() and svg_path.stat().st_size > 0, str(svg_path.stat().st_size if svg_path.exists() else 0))
    add("TIFF exists", tiff_path.exists() and tiff_path.stat().st_size > 0, str(tiff_path.stat().st_size if tiff_path.exists() else 0))
    if svg_path.exists():
        text = svg_path.read_text(encoding="utf-8")
        add("SVG retains live text", "<text" in text, f"text elements={text.count('<text')}")
    qa = pd.DataFrame(checks)
    qa.to_csv(QA_DIR / "Figure_Q4_automated_QA.csv", index=False, encoding="utf-8-sig")
    return qa


def main():
    for path in [SCRIPT_DIR, SOURCE_DIR, FIG_DIR / "PNG", FIG_DIR / "SVG", FIG_DIR / "PDF", FIG_DIR / "TIFF", QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    core = load_core()
    site = pd.read_excel(RAW_BOOK, sheet_name="Site_Special")
    mapping = pd.read_excel(MAPPING_BOOK, sheet_name="Q2最终代表变量")
    reps = prepare_metadata(mapping)
    scores, score_meta, score_components = core.build_module_scores(site)

    all_assoc = []
    all_contribution = []
    all_contribution_module = []
    all_family_share = []
    all_taxon_share = []
    all_duplicate_audit = []
    for water in WATERS:
        result = calculate_water(core, site, reps, scores, water)
        assoc, contribution, contribution_module, family_share, taxon_share, duplicate_audit = result
        all_assoc.append(assoc)
        all_contribution.append(contribution)
        all_contribution_module.append(contribution_module)
        all_family_share.append(family_share)
        all_taxon_share.append(taxon_share)
        all_duplicate_audit.append(duplicate_audit)

    assoc = pd.concat(all_assoc, ignore_index=True)
    contribution = pd.concat(all_contribution, ignore_index=True)
    contribution_module = pd.concat(all_contribution_module, ignore_index=True)
    family_share = pd.concat(all_family_share, ignore_index=True)
    taxon_share = pd.concat(all_taxon_share, ignore_index=True)
    duplicate_audit = pd.concat(all_duplicate_audit, ignore_index=True)
    selected = contribution.loc[contribution.observed_rank_within_family <= 3].copy()
    selected_assoc = assoc.loc[assoc.selected_top3].copy()

    assoc.to_csv(SOURCE_DIR / "Q4_all_frozen_candidate_multiscale_associations.csv", index=False, encoding="utf-8-sig")
    contribution.to_csv(SOURCE_DIR / "Q4_variable_contribution_and_bootstrap_stability.csv", index=False, encoding="utf-8-sig")
    contribution_module.to_csv(SOURCE_DIR / "Q4_module_specific_variable_contributions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(SOURCE_DIR / "Q4_selected_top3_variables_per_family.csv", index=False, encoding="utf-8-sig")
    selected_assoc.to_csv(SOURCE_DIR / "Figure_Q4_selected_variable_multiscale_source_data.csv", index=False, encoding="utf-8-sig")
    family_share.to_csv(SOURCE_DIR / "Figure_Q4_family_signal_shares.csv", index=False, encoding="utf-8-sig")
    taxon_share.to_csv(SOURCE_DIR / "Q4_taxon_signal_shares_within_family.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(SOURCE_DIR / "Q4_rebuilt_environment_module_scores.csv", index=False, encoding="utf-8-sig")
    score_meta.to_csv(QA_DIR / "Q4_environment_score_definition_audit.csv", index=False, encoding="utf-8-sig")
    score_components.to_csv(QA_DIR / "Q4_environment_score_component_audit.csv", index=False, encoding="utf-8-sig")
    duplicate_audit.to_csv(QA_DIR / "Q4_exact_duplicate_representative_audit.csv", index=False, encoding="utf-8-sig")

    if "--compute-only" in sys.argv:
        shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
        print(
            json.dumps(
                {
                    "mode": "compute-only",
                    "candidate_representatives": contribution.groupby("water_body_type").size().to_dict(),
                    "exact_duplicates_removed": int(len(duplicate_audit)),
                },
                ensure_ascii=False,
            )
        )
        return

    figure = plot_figure(family_share, contribution, assoc)
    base = "Figure_Q4_specific_biological_carriers"
    png_path = FIG_DIR / "PNG" / f"{base}.png"
    svg_path = FIG_DIR / "SVG" / f"{base}.svg"
    pdf_path = FIG_DIR / "PDF" / f"{base}.pdf"
    tiff_path = FIG_DIR / "TIFF" / f"{base}.tiff"
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)

    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    qa = run_qa(assoc, contribution, family_share, selected, png_path, svg_path, tiff_path)

    family_summary = (
        family_share.groupby(["water_body_type", "environment_module", "response_family"], as_index=False)
        .family_signal_share.mean()
    )
    top_families = (
        family_summary.sort_values("family_signal_share", ascending=False)
        .groupby(["water_body_type", "environment_module"], as_index=False)
        .first()
    )
    selected_summary = selected[
        [
            "water_body_type",
            "response_family",
            "taxon_group",
            "column_name",
            "variable_name_cn",
            "balanced_contribution_share",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "top3_selection_frequency",
            "selection_stability",
        ]
    ].copy()
    summary = {
        "analysis_date": "2026-08-08",
        "backend": "Python",
        "independent_units": site.groupby("water_body_type").size().to_dict(),
        "candidate_representatives": contribution.groupby("water_body_type").size().to_dict(),
        "selected_variables": selected.groupby("water_body_type").size().to_dict(),
        "permutations": N_PERM,
        "site_bootstraps": N_BOOT,
        "all_qa_pass": bool(qa.passed.all()),
        "top_family_by_water_and_driver": top_families.to_dict(orient="records"),
        "selected_variables_detail": selected_summary.to_dict(orient="records"),
    }
    (QA_DIR / "Q4_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"delivery": str(DELIVERY), **{k: summary[k] for k in ["independent_units", "candidate_representatives", "selected_variables", "all_qa_pass"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
