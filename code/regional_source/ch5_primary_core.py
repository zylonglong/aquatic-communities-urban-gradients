from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=RuntimeWarning)

RANGES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
WATER_TYPES = ["River", "Lake"]
MODULES = ["terrain", "urbanization", "human_activity"]
FAMILY_ORDER = [
    "community_baseline",
    "alpha_diversity",
    "beta_diversity",
    "community_composition",
    "functional_indicator",
]
FAMILY_CN = {
    "community_baseline": "群落基础特征",
    "alpha_diversity": "α多样性",
    "beta_diversity": "β多样性",
    "community_composition": "群落组成",
    "functional_indicator": "功能与指示群",
}
MODULE_CN = {
    "terrain": "地形地貌",
    "urbanization": "城市化建设",
    "human_activity": "人类直接活动",
}
TAXON_CN = {
    "benthos": "底栖动物",
    "fish": "鱼类",
    "macrophytes": "水生植物",
    "macrophyte": "水生植物",
    "phytoplankton": "浮游植物",
    "periphyton": "着生藻类",
    "zooplankton": "浮游动物",
    "overall": "总体",
}


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    back = np.empty(m)
    back[order] = adj
    q[ok] = back
    return q


def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    if not np.isfinite(iqr) or iqr <= 0:
        sd = x.std(ddof=0)
        iqr = sd if np.isfinite(sd) and sd > 0 else 1.0
    return (x - med) / iqr


def rank_residual(y, controls):
    y = np.asarray(y, dtype=float)
    c = np.asarray(controls, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(c), axis=1)
    out = np.full(len(y), np.nan)
    if valid.sum() < c.shape[1] + 5:
        return out
    yr = stats.rankdata(y[valid], method="average")
    cc = c[valid]
    X = np.column_stack([np.ones(valid.sum()), cc])
    coef = np.linalg.lstsq(X, yr, rcond=None)[0]
    out[valid] = yr - X @ coef
    return out


def pearson_nan(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 8:
        return np.nan, n
    xx, yy = x[ok], y[ok]
    if np.std(xx) == 0 or np.std(yy) == 0:
        return np.nan, n
    return float(np.corrcoef(xx, yy)[0, 1]), n


def partial_pvalue(r, n, k):
    if not np.isfinite(r) or n <= k + 3 or abs(r) >= 1:
        return np.nan
    df = n - k - 2
    t = r * math.sqrt(df / max(1e-15, 1 - r * r))
    return float(2 * stats.t.sf(abs(t), df))


def fisher_ci(r, n, k, alpha=0.05):
    if not np.isfinite(r) or abs(r) >= 0.999999 or n <= k + 5:
        return np.nan, np.nan
    se = 1 / math.sqrt(max(1, n - k - 3))
    z = np.arctanh(r)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def infer_taxon(column):
    c = column.lower()
    for key in ["benthos", "fish", "macrophytes", "macrophyte", "phytoplankton", "periphyton", "zooplankton", "overall"]:
        if c.startswith(key + "_") or f"_{key}_" in c:
            return "macrophyte" if key == "macrophytes" else key
    return "other"


def read_dictionaries(root: Path):
    specs = [
        ("Feature_Process_Aquatic_Biology_Raw.xlsx", "aquatic"),
        ("Feature_Process_Beta_Diversity_Raw.xlsx", "beta"),
        ("Feature_Process_Biological_Response_Traits_Raw.xlsx", "traits"),
    ]
    frames = []
    for filename, source in specs:
        d = pd.read_excel(root / filename, sheet_name="Feature_Dictionary")
        d["source_dictionary"] = source
        frames.append(d)
    meta = pd.concat(frames, ignore_index=True, sort=False)
    family = []
    for _, row in meta.iterrows():
        col = str(row["column_name"])
        src = row["source_dictionary"]
        cat = str(row.get("feature_category", ""))
        if src == "aquatic":
            if any(x in col.lower() for x in ["richness", "shannon", "margalef", "pielou", "simpson"]):
                family.append("alpha_diversity")
            else:
                family.append("community_baseline")
        elif src == "beta":
            family.append("beta_diversity")
        elif cat == "community_composition_axis":
            family.append("community_composition")
        else:
            family.append("functional_indicator")
    meta["response_family"] = family
    meta["response_family_cn"] = meta["response_family"].map(FAMILY_CN)
    if "taxon_group" not in meta:
        meta["taxon_group"] = np.nan
    meta["taxon_group"] = [
        infer_taxon(c) if pd.isna(t) else str(t)
        for c, t in zip(meta["column_name"], meta["taxon_group"])
    ]
    meta["taxon_group_cn"] = meta["taxon_group"].map(TAXON_CN).fillna(meta["taxon_group"])
    return meta


def range_tag(r):
    return f"{r:04d}m"


def build_module_scores(df: pd.DataFrame):
    scores = []
    score_meta = []
    component_long = []
    for water in WATER_TYPES:
        idx = df["water_body_type"].eq(water)
        sub = df.loc[idx]
        for r in RANGES:
            tag = range_tag(r)
            # Terrain: a fixed, interpretable average of elevation/relief/slope subdimensions.
            tprefix = f"terrain_{r}m_"
            tvars = [c for c in df.columns if c.startswith(tprefix)]
            groups = {
                "地形位置": [c for c in tvars if any(k in c for k in ["mean_dem_m", "delta_dem_buffer_minus_point_m"])],
                "地形起伏": [c for c in tvars if any(k in c for k in ["dem_range_m", "dem_std_m"])],
                "坡度强度": [c for c in tvars if any(k in c for k in ["mean_slope_pct", "max_slope_pct"])],
                "陡坡覆盖": [c for c in tvars if "slope_gt" in c],
            }
            terrain_components = []
            for g, cols in groups.items():
                cols = [c for c in cols if c in sub and pd.to_numeric(sub[c], errors="coerce").nunique() > 1]
                if cols:
                    z = pd.concat([robust_z(sub[c]) for c in cols], axis=1).mean(axis=1)
                    terrain_components.append(z)
                    for c in cols:
                        component_long.append((water, r, "terrain", g, c, 1))
            tscore = pd.concat(terrain_components, axis=1).mean(axis=1) if terrain_components else pd.Series(np.nan, index=sub.index)

            # Urban construction: positive built dimensions minus natural-buffer dimensions.
            pos = [
                f"urbanization_{tag}_impervious_pct", f"urbanization_{tag}_building_pct",
                f"landuse_{tag}_urban_pct",
            ]
            neg = [f"urbanization_{tag}_natural_land_pct", f"landuse_{tag}_green_pct"]
            pos = [c for c in pos if c in sub and pd.to_numeric(sub[c], errors="coerce").nunique() > 1]
            neg = [c for c in neg if c in sub and pd.to_numeric(sub[c], errors="coerce").nunique() > 1]
            uz = []
            for c in pos:
                uz.append(robust_z(sub[c])); component_long.append((water, r, "urbanization", "建设地表", c, 1))
            for c in neg:
                uz.append(-robust_z(sub[c])); component_long.append((water, r, "urbanization", "自然缓冲（反向）", c, -1))
            uscore = pd.concat(uz, axis=1).mean(axis=1) if uz else pd.Series(np.nan, index=sub.index)

            # Human activity: equal-weight mechanism-group intensity after log1p and robust scaling.
            hvars = [c for c in df.columns if c.startswith(f"api_{tag}_")]
            hvars += [f"worldpop_{tag}_mean_density"]
            hz = []
            for c in hvars:
                x = pd.to_numeric(sub[c], errors="coerce").clip(lower=0)
                if x.nunique() > 1:
                    hz.append(robust_z(np.log1p(x)))
                    mechanism = re.search(r"_(P[1-8])_", c)
                    component_long.append((water, r, "human_activity", mechanism.group(1) if mechanism else "API", c, 1))
            hscore = pd.concat(hz, axis=1).mean(axis=1) if hz else pd.Series(np.nan, index=sub.index)

            for module, sc, variables in [
                ("terrain", tscore, sum(groups.values(), [])),
                ("urbanization", uscore, pos + neg),
                ("human_activity", hscore, hvars),
            ]:
                sc = robust_z(sc)
                for i, value in sc.items():
                    scores.append((i, water, r, module, value))
                score_meta.append({
                    "water_body_type": water,
                    "range_m": r,
                    "environment_module": module,
                    "environment_module_cn": MODULE_CN[module],
                    "n_variables": len(set(variables)),
                    "variables": ";".join(sorted(set(variables))),
                    "n_nonmissing": int(sc.notna().sum()),
                    "score_rule": {
                        "terrain": "四个地形机制子组分别Robust-z后等权平均，值越高表示高程/起伏/坡度综合增强",
                        "urbanization": "不透水、建筑、建设用地为正，自然用地与绿色植被为负；人口已移至直接人类活动；Robust-z等权平均",
                        "human_activity": "P1–P8点密度、面占用及人口密度先log1p再Robust-z并等权平均",
                    }[module],
                })
    score_df = pd.DataFrame(scores, columns=["row_index", "water_body_type", "range_m", "environment_module", "environment_score"])
    comp_df = pd.DataFrame(component_long, columns=["water_body_type", "range_m", "environment_module", "mechanism_group", "variable", "direction_weight"])
    return score_df, pd.DataFrame(score_meta), comp_df


def response_availability(site, response_meta):
    rows = []
    for water in WATER_TYPES:
        d = site.loc[site.water_body_type.eq(water)]
        for _, m in response_meta.iterrows():
            c = m.column_name
            x = pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(dtype=float)
            reason = "available"
            if x.notna().sum() == 0:
                if water == "Lake" and (m.taxon_group == "periphyton" or str(m.get("beta_scope", "")) == "river"):
                    reason = "structural_not_sampled_or_not_applicable"
                elif water == "River" and str(m.get("beta_scope", "")) == "lake":
                    reason = "structural_not_applicable"
                else:
                    reason = "all_missing"
            elif m.response_family == "beta_diversity":
                scope = str(m.get("beta_scope", ""))
                if scope not in [water.lower()]:
                    reason = "sensitivity_scope_not_primary"
            rows.append({
                "water_body_type": water, "column_name": c,
                "variable_name_cn": m.variable_name_cn,
                "response_family": m.response_family,
                "response_family_cn": m.response_family_cn,
                "taxon_group": m.taxon_group, "taxon_group_cn": m.taxon_group_cn,
                "n_total_sites": len(d), "n_nonmissing": int(x.notna().sum()),
                "missing_pct": float(x.isna().mean() * 100) if len(x) else np.nan,
                "n_unique": int(x.nunique(dropna=True)),
                "zero_pct_nonmissing": float((x.dropna() == 0).mean() * 100) if x.notna().any() else np.nan,
                "skew": float(x.dropna().skew()) if x.notna().sum() >= 3 else np.nan,
                "availability_status": reason,
                "primary_eligible": bool(reason == "available" and x.notna().sum() >= max(20, math.ceil(0.5 * len(d))) and x.nunique(dropna=True) >= 3),
            })
    return pd.DataFrame(rows)


def controls_for(d):
    lon = robust_z(d["longitude"]).to_numpy()
    lat = robust_z(d["latitude"]).to_numpy()
    nrec = robust_z(d.get("n_sample_records", pd.Series(1, index=d.index))).fillna(0).to_numpy()
    return np.column_stack([lon, lat, lon * lon, lat * lat, lon * lat, nrec])


def select_family_representatives(d, mf, threshold=0.95):
    """Select nonredundant responses for the family omnibus statistic only."""
    selected, mapping = [], []
    for taxon, mt in mf.groupby("taxon_group", dropna=False):
        order = sorted(
            mt.column_name,
            key=lambda c: (-pd.to_numeric(d[c], errors="coerce").notna().sum(), len(str(c)), str(c)),
        )
        keep = []
        for c in order:
            xc = pd.to_numeric(d[c], errors="coerce")
            representative = None
            for k in keep:
                xk = pd.to_numeric(d[k], errors="coerce")
                ok = xc.notna() & xk.notna()
                rho = stats.spearmanr(xc[ok], xk[ok]).statistic if ok.sum() >= 10 else np.nan
                if np.isfinite(rho) and abs(rho) >= threshold:
                    representative = k
                    break
            if representative is None:
                keep.append(c)
                representative = c
            mapping.append({
                "column_name": c,
                "omnibus_representative": representative,
                "is_omnibus_representative": c == representative,
                "redundancy_threshold_abs_spearman": threshold,
            })
        selected.extend(keep)
    return mf[mf.column_name.isin(selected)].copy(), pd.DataFrame(mapping)


def compute_associations(site, response_meta, availability, score_df):
    all_rows = []
    family_rows = []
    redundancy_rows = []
    rng = np.random.default_rng(20260722)
    n_perm = 499
    for water in WATER_TYPES:
        d = site.loc[site.water_body_type.eq(water)].copy()
        controls = controls_for(d)
        eligible = availability[(availability.water_body_type == water) & availability.primary_eligible]
        meta = response_meta.merge(eligible[["column_name"]], on="column_name", how="inner")
        # Cache spatially adjusted rank residuals once because responses do not change with range.
        yres = {c: rank_residual(pd.to_numeric(d[c], errors="coerce"), controls) for c in meta.column_name}
        for module in MODULES:
            score_mat = {}
            for r in RANGES:
                sc = score_df[(score_df.water_body_type == water) & (score_df.environment_module == module) & (score_df.range_m == r)].set_index("row_index")["environment_score"]
                x = sc.reindex(d.index).to_numpy(dtype=float)
                score_mat[r] = rank_residual(x, controls)
            for family in FAMILY_ORDER:
                mf_all = meta[meta.response_family == family]
                if mf_all.empty:
                    continue
                mf, redmap = select_family_representatives(d, mf_all)
                redmap["water_body_type"] = water
                redmap["response_family"] = family
                redundancy_rows.append(redmap)
                # Matrix representation permits exact pairwise-missing Pearson
                # correlations for all responses/scales in one linear-algebra pass.
                Y = np.column_stack([yres[c] for c in mf.column_name])
                mask = np.isfinite(Y).astype(float)
                n_valid = mask.sum(axis=0)
                y_mean = np.nansum(Y, axis=0) / np.maximum(n_valid, 1)
                Y0 = np.where(np.isfinite(Y), Y - y_mean, 0.0)
                y_norm = np.sqrt(np.sum(Y0 * Y0, axis=0))
                Xscore = np.column_stack([score_mat[r] for r in RANGES])
                # Scores are expected complete; residual-control failures are
                # conservatively replaced by zero only for the omnibus matrix.
                Xscore = np.where(np.isfinite(Xscore), Xscore, 0.0)
                obs_stats = []
                for r in RANGES:
                    rs = []
                    for _, m in mf_all.iterrows():
                        rr, n = pearson_nan(score_mat[r], yres[m.column_name])
                        p = partial_pvalue(rr, n, controls.shape[1])
                        lo, hi = fisher_ci(rr, n, controls.shape[1])
                        rs.append(rr)
                        all_rows.append({
                            "water_body_type": water,
                            "environment_module": module,
                            "environment_module_cn": MODULE_CN[module],
                            "response_family": family,
                            "response_family_cn": FAMILY_CN[family],
                            "taxon_group": m.taxon_group,
                            "taxon_group_cn": m.taxon_group_cn,
                            "column_name": m.column_name,
                            "variable_name_cn": m.variable_name_cn,
                            "range_m": r, "partial_spearman_rho": rr,
                            "ci95_low": lo, "ci95_high": hi,
                            "p_value": p, "n_sites": n,
                        })
                    arr = np.asarray(rs, dtype=float)
                    obs = float(np.sqrt(np.nanmean(arr * arr))) if np.isfinite(arr).any() else np.nan
                    obs_stats.append(obs)
                # maxT correction across the 13 nested scales using common row permutations.
                perm_stats = np.full((n_perm, len(RANGES)), np.nan)
                for b in range(n_perm):
                    perm = rng.permutation(len(d))
                    xp = Xscore[perm, :]
                    numerator = Y0.T @ xp
                    sum_x = mask.T @ xp
                    sum_x2 = mask.T @ (xp * xp)
                    x_ss = np.maximum(sum_x2 - (sum_x * sum_x) / np.maximum(n_valid[:, None], 1), 0)
                    denominator = y_norm[:, None] * np.sqrt(x_ss)
                    rr = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
                    perm_stats[b, :] = np.sqrt(np.nanmean(rr * rr, axis=0))
                max_null = np.nanmax(perm_stats, axis=1)
                for j, r in enumerate(RANGES):
                    obs = obs_stats[j]
                    p_raw = (1 + np.sum(perm_stats[:, j] >= obs)) / (n_perm + 1) if np.isfinite(obs) else np.nan
                    p_maxt = (1 + np.sum(max_null >= obs)) / (n_perm + 1) if np.isfinite(obs) else np.nan
                    rep_names = set(mf.column_name)
                    vals = [x["partial_spearman_rho"] for x in all_rows if x["water_body_type"] == water and x["environment_module"] == module and x["response_family"] == family and x["range_m"] == r and x["column_name"] in rep_names]
                    va = np.asarray(vals, dtype=float)
                    family_rows.append({
                        "water_body_type": water, "environment_module": module,
                        "environment_module_cn": MODULE_CN[module], "response_family": family,
                        "response_family_cn": FAMILY_CN[family], "range_m": r,
                        "n_response_variables": int(np.isfinite(va).sum()),
                        "n_response_variables_before_redundancy_control": int(len(mf_all)),
                        "rms_partial_rho": obs,
                        "median_partial_rho": float(np.nanmedian(va)) if np.isfinite(va).any() else np.nan,
                        "positive_fraction": float(np.nanmean(va > 0)) if np.isfinite(va).any() else np.nan,
                        "permutation_p": p_raw, "range_maxT_p": p_maxt,
                    })
    assoc = pd.DataFrame(all_rows)
    family = pd.DataFrame(family_rows)
    # Hierarchical FDR columns.
    assoc["q_within_response_13ranges"] = np.nan
    for _, ix in assoc.groupby(["water_body_type", "environment_module", "column_name"]).groups.items():
        assoc.loc[ix, "q_within_response_13ranges"] = bh_fdr(assoc.loc[ix, "p_value"])
    assoc["q_within_family_all_responses_ranges"] = np.nan
    for _, ix in assoc.groupby(["water_body_type", "environment_module", "response_family"]).groups.items():
        assoc.loc[ix, "q_within_family_all_responses_ranges"] = bh_fdr(assoc.loc[ix, "p_value"])
    family["q_13ranges_within_family"] = np.nan
    for _, ix in family.groupby(["water_body_type", "environment_module", "response_family"]).groups.items():
        family.loc[ix, "q_13ranges_within_family"] = bh_fdr(family.loc[ix, "permutation_p"])
    family["q_module_all_families_ranges"] = np.nan
    for _, ix in family.groupby(["water_body_type", "environment_module"]).groups.items():
        family.loc[ix, "q_module_all_families_ranges"] = bh_fdr(family.loc[ix, "permutation_p"])
    redundancy = pd.concat(redundancy_rows, ignore_index=True).drop_duplicates() if redundancy_rows else pd.DataFrame()
    return assoc, family, redundancy


def contiguous_runs(ranges):
    ranges = sorted(set(int(x) for x in ranges))
    if not ranges:
        return []
    positions = {r: i for i, r in enumerate(RANGES)}
    runs, cur = [], [ranges[0]]
    for r in ranges[1:]:
        if positions[r] == positions[cur[-1]] + 1:
            cur.append(r)
        else:
            runs.append(cur); cur = [r]
    runs.append(cur)
    return runs


def classify_scale_trajectory(g):
    """Describe the response by spatial scale, without an evidence hierarchy."""
    g = g.sort_values("range_m")
    rho = g.partial_spearman_rho.to_numpy(float)
    ranges = g.range_m.to_numpy(int)
    active = np.isfinite(rho) & (np.abs(rho) >= 0.15)
    active_ranges = ranges[active].tolist()
    runs = contiguous_runs(active_ranges)
    best_run = max(runs, key=len) if runs else []
    peak_i = int(np.nanargmax(np.abs(rho))) if np.isfinite(rho).any() else 0
    peak_range = int(ranges[peak_i])

    domains = {
        "滨水区尺度（200–500 m）": (200, 500),
        "中尺度（1000–2500 m）": (1000, 2500),
        "景观尺度（3000–6000 m）": (3000, 6000),
    }
    domain_strength = {}
    for label, (lo, hi) in domains.items():
        mask = (ranges >= lo) & (ranges <= hi) & np.isfinite(rho)
        domain_strength[label] = float(np.nanmean(np.abs(rho[mask]))) if mask.any() else np.nan
    dominant_domain = max(domain_strength, key=lambda k: -np.inf if not np.isfinite(domain_strength[k]) else domain_strength[k])

    has_pos = np.any(rho >= 0.15)
    has_neg = np.any(rho <= -0.15)
    if not active.any():
        trajectory = "弱响应/近零轨迹"
        scale_type = "无明确响应尺度"
    elif has_pos and has_neg:
        trajectory = "方向反转"
        scale_type = "跨尺度差异响应"
    elif len(best_run) >= 9:
        trajectory = "连续同向"
        scale_type = "跨尺度连续响应"
    elif len(runs) >= 2:
        trajectory = "多峰或分段"
        scale_type = "跨尺度分段响应"
    elif peak_i not in (0, len(rho) - 1) and len(best_run) >= 3:
        left = np.nanmean(np.abs(rho[:peak_i + 1]))
        right = np.nanmean(np.abs(rho[peak_i:]))
        trajectory = "中间峰值/转折" if np.isfinite(left + right) else "单峰"
        scale_type = dominant_domain + "主导"
    else:
        trajectory = "局部单向"
        scale_type = dominant_domain + "主导"

    return {
        "active_ranges_absrho_ge_0.15": ";".join(map(str, active_ranges)),
        "response_range_m": f"{best_run[0]}–{best_run[-1]}" if best_run else "",
        "response_range_n": len(best_run),
        "dominant_scale_domain": dominant_domain,
        "scale_response_type": scale_type,
        "trajectory_shape": trajectory,
        "nearshore_mean_abs_rho": domain_strength["滨水区尺度（200–500 m）"],
        "mesoscale_mean_abs_rho": domain_strength["中尺度（1000–2500 m）"],
        "landscape_mean_abs_rho": domain_strength["景观尺度（3000–6000 m）"],
        "peak_range_m": peak_range,
    }


def summarize_candidates(assoc, family):
    famkey = family.set_index(["water_body_type", "environment_module", "response_family", "range_m"])
    rows = []
    for keys, g in assoc.groupby(["water_body_type", "environment_module", "response_family", "column_name", "variable_name_cn", "taxon_group_cn"]):
        water, module, fam, col, name_cn, taxon_cn = keys
        g = g.sort_values("range_m")
        median_sign = np.sign(g.partial_spearman_rho.median())
        if median_sign == 0 or not np.isfinite(median_sign):
            median_sign = 1
        support = g[(np.sign(g.partial_spearman_rho) == median_sign) & (g.q_within_response_13ranges <= 0.10)]
        confirm = g[(np.sign(g.partial_spearman_rho) == median_sign) & (g.q_within_family_all_responses_ranges <= 0.05)]
        sruns = contiguous_runs(support.range_m)
        cruns = contiguous_runs(confirm.range_m)
        peak_idx = g.partial_spearman_rho.abs().idxmax() if g.partial_spearman_rho.notna().any() else None
        peak = g.loc[peak_idx] if peak_idx is not None else None
        fam_supported_ranges = []
        for r in g.range_m:
            try:
                fr = famkey.loc[(water, module, fam, r)]
                if float(fr["range_maxT_p"]) <= 0.05:
                    fam_supported_ranges.append(int(r))
            except Exception:
                pass
        statistical_flag = "未达预设筛查阈值"
        if cruns and max(map(len, cruns)) >= 2 and fam_supported_ranges:
            statistical_flag = "响应层FDR与家族maxT均支持"
        elif sruns and max(map(len, sruns)) >= 2 and fam_supported_ranges:
            statistical_flag = "响应层FDR与家族maxT部分支持"
        elif np.sum(g.partial_spearman_rho.abs() >= 0.15) >= 3:
            statistical_flag = "连续效应量达到阈值"
        elif np.nanmax(g.partial_spearman_rho.abs()) >= 0.20:
            statistical_flag = "单尺度探索性效应"
        scale_desc = classify_scale_trajectory(g)
        rows.append({
            "water_body_type": water, "environment_module": module,
            "environment_module_cn": MODULE_CN[module], "response_family": fam,
            "response_family_cn": FAMILY_CN[fam], "taxon_group_cn": taxon_cn,
            "column_name": col, "variable_name_cn": name_cn,
            "dominant_direction": "正" if median_sign > 0 else "负",
            "median_rho_13ranges": float(g.partial_spearman_rho.median()),
            "max_abs_rho": float(g.partial_spearman_rho.abs().max()),
            "peak_rho": float(peak.partial_spearman_rho) if peak is not None else np.nan,
            "peak_q_response": float(peak.q_within_response_13ranges) if peak is not None else np.nan,
            "family_maxT_supported_ranges": ";".join(map(str, fam_supported_ranges)),
            "statistical_flag": statistical_flag,
            **scale_desc,
        })
    out = pd.DataFrame(rows)
    rankmap = {
        "响应层FDR与家族maxT均支持": 1,
        "响应层FDR与家族maxT部分支持": 2,
        "连续效应量达到阈值": 3,
        "单尺度探索性效应": 4,
        "未达预设筛查阈值": 5,
    }
    out["screen_rank"] = out.statistical_flag.map(rankmap)
    return out.sort_values(["screen_rank", "max_abs_rho"], ascending=[True, False])


def main():
    script = Path(__file__).resolve()
    out = script.parents[1]
    ml_root = out.parent / "ML_Input_原始量_13尺度_2026-07-04"
    ml_path = ml_root / "ML_Input_Raw_13Ranges_Sample_Site.xlsx"
    site = pd.read_excel(ml_path, sheet_name="Site_Special")
    sample = pd.read_excel(ml_path, sheet_name="Sample_Special")
    response_meta = read_dictionaries(ml_root)
    availability = response_availability(site, response_meta)
    score_df, score_meta, components = build_module_scores(site)

    # Core audit tables.
    audit_dir = out / "01_数据审计"
    source_dir = out / "09_SourceData"
    audit_dir.mkdir(exist_ok=True); source_dir.mkdir(exist_ok=True)
    response_meta.to_csv(audit_dir / "响应变量字典与生态维度.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(audit_dir / "River_Lake响应变量可用性与结构性缺失.csv", index=False, encoding="utf-8-sig")
    score_meta.to_csv(audit_dir / "环境模块固定指数构建清单.csv", index=False, encoding="utf-8-sig")
    components.to_csv(audit_dir / "环境指数分量与方向权重.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(source_dir / "环境模块13尺度固定指数.csv", index=False, encoding="utf-8-sig")

    coverage = sample.groupby(["water_body_type", "sampling_year", "flow_period"]).agg(
        n_samples=("sample_id", "size"), n_sites=("site", "nunique")
    ).reset_index()
    coverage.to_csv(audit_dir / "样本年份水文期覆盖.csv", index=False, encoding="utf-8-sig")
    site_cov = site[["site_special_id", "water_body_type", "site", "longitude", "latitude", "n_sample_records", "sampling_years", "flow_periods"]].copy()
    site_cov.to_csv(audit_dir / "独立Site及调查覆盖.csv", index=False, encoding="utf-8-sig")

    assoc, family, redundancy = compute_associations(site, response_meta, availability, score_df)
    candidates = summarize_candidates(assoc, family)
    redundancy.to_csv(audit_dir / "家族全局检验响应冗余控制映射.csv", index=False, encoding="utf-8-sig")
    global_dir = out / "03_Q2_水生生物整体响应"
    detail_dir = out / "05_Q4_具体生物响应"
    global_dir.mkdir(exist_ok=True); detail_dir.mkdir(exist_ok=True)
    family.to_csv(global_dir / "环境模块_生态维度_13尺度全局置换检验.csv", index=False, encoding="utf-8-sig")
    assoc.to_csv(detail_dir / "全部具体响应_偏Spearman_分层FDR.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(detail_dir / "具体响应尺度类型与候选清单.csv", index=False, encoding="utf-8-sig")

    # Excel digest for manual review.
    with pd.ExcelWriter(out / "水生生物_三类环境因素_13尺度核心统计结果.xlsx", engine="openpyxl") as xw:
        coverage.to_excel(xw, sheet_name="样本覆盖", index=False)
        availability.to_excel(xw, sheet_name="响应可用性", index=False)
        score_meta.to_excel(xw, sheet_name="环境指数定义", index=False)
        family.to_excel(xw, sheet_name="家族级全局效应", index=False)
        candidates.to_excel(xw, sheet_name="具体响应尺度类型", index=False)

    config = {
        "analysis_date": "2026-07-23",
        "primary_matrix": str(ml_path),
        "site_n": site.groupby("water_body_type").size().to_dict(),
        "sample_n": sample.groupby("water_body_type").size().to_dict(),
        "ranges_m": RANGES,
        "water_types": WATER_TYPES,
        "modules": MODULES,
        "response_families": FAMILY_ORDER,
        "independent_unit_primary": "site",
        "spatial_controls": ["longitude", "latitude", "longitude^2", "latitude^2", "longitude×latitude", "n_sample_records"],
        "omnibus_statistic": "RMS of response-wise partial Spearman correlations",
        "permutations": 499,
        "nested_range_control": "maxT across all 13 ranges using common permutations",
        "seed": 20260723,
        "notes": [
            "The frozen four-question scale-response framework dated 2026-07-23 is authoritative.",
            "Water-quality grade is a comparative regional reference; aquatic biology is the deep-analysis target.",
            "Interpretation is organized by response range and trajectory, not by discovery-robustness-extrapolation tiers.",
            "Beta-diversity primary analysis uses water-type-specific scope; all-scope variables are retained as sensitivity-only.",
            "Lake periphyton absence and opposite-water beta scopes are structural missingness and are never imputed.",
        ],
    }
    cfgdir = out / "00_方案与配置"; cfgdir.mkdir(exist_ok=True)
    (cfgdir / "analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out), "availability_rows": len(availability),
        "association_rows": len(assoc), "family_rows": len(family),
        "scale_response_types": candidates.scale_response_type.value_counts().to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
