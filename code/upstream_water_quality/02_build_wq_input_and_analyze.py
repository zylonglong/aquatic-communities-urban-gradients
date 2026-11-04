# -*- coding: utf-8 -*-
"""Assemble WQ input matrices and analyse scale-dependent responses."""
from __future__ import annotations

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew, rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"E:\桌面\博士课题\Data")
OUT = Path(__file__).resolve().parent.parent
WQROOT = OUT
PROC = OUT / "01_INPUT与过程数据"
RES = OUT / "02_统计结果"
QA = OUT / "06_QA"
for p in [PROC, RES, QA]:
    p.mkdir(parents=True, exist_ok=True)

SCALES = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
RNG = np.random.default_rng(20260720)


def boolify(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def class_score(x):
    s = "" if pd.isna(x) else str(x).strip().replace("类", "")
    s = s.replace("≤", "")
    return {"Ⅰ": 1.0, "Ⅱ": 2.0, "Ⅲ": 3.0, "Ⅳ": 4.0, "Ⅴ": 5.0, "劣Ⅴ": 6.0}.get(s, np.nan)


def bh_q(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def permutation_spearman(x, y, n_perm=4999):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    rho = float(np.dot(rx, ry))
    permuted = np.vstack([RNG.permutation(ry) for _ in range(n_perm)])
    r_perm = permuted @ rx
    exceed = int(np.sum(np.abs(r_perm) >= abs(rho) - 1e-12))
    return rho, (exceed + 1) / (n_perm + 1)


def bootstrap_spearman(x, y, n_boot=999):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(x), len(x))
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        vals.append(spearmanr(x[idx], y[idx]).statistic)
    if len(vals) < 100:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def jackknife_sign_stability(x, y, full_rho):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6 or not np.isfinite(full_rho) or full_rho == 0:
        return np.nan
    signs = []
    for i in range(len(x)):
        r = spearmanr(np.delete(x, i), np.delete(y, i)).statistic
        if np.isfinite(r) and r != 0:
            signs.append(np.sign(r) == np.sign(full_rho))
    return float(np.mean(signs)) if signs else np.nan


def build_input():
    long = pd.read_csv(WQROOT / r"02_结构化数据\深圳及周边公共水质_标准长表.csv", low_memory=False)
    wide = pd.read_csv(WQROOT / r"02_结构化数据\深圳及周边公共水质_站点时间宽表.csv", low_memory=False)
    reg = pd.read_csv(PROC / "WQ_Station_Spatial_Registry.csv")
    static = pd.read_csv(PROC / "WQ_Environmental_Features_Static_Wide.csv")
    pop = pd.read_csv(PROC / "WQ_WorldPop_Annual_Wide.csv")
    reg["spatial_eligible"] = reg["spatial_eligible"].map(boolify)

    # Record-special matrix: original publication unit and time retained.
    wide["monitoring_date_start"] = pd.to_datetime(wide["monitoring_date_start"], errors="coerce")
    wide["monitoring_year"] = wide["monitoring_date_start"].dt.year
    meta_cols = ["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id",
                 "waterbody_name_spatial", "spatial_waterbody_type", "spatial_longitude",
                 "spatial_latitude", "spatial_quality", "spatial_eligible", "station_scope", "monitoring_agency"]
    record = wide.merge(reg[meta_cols], on="station_id", how="left", validate="m:1")
    record = record.merge(static, on="spatial_id", how="left", suffixes=("", "_env"), validate="m:1")
    record = record.merge(pop, left_on=["spatial_id", "monitoring_year"], right_on=["spatial_id", "year"], how="left",
                          suffixes=("", "_pop"), validate="m:1")
    record.to_csv(PROC / "WQ_Record_Special_13Ranges.csv", index=False, encoding="utf-8-sig")

    long["include_primary_bool"] = long["include_primary"].map(boolify)
    long = long[long.include_primary_bool].copy()
    long = long.merge(reg[meta_cols], on="station_id", how="left", validate="m:1")
    long["monitoring_date_end"] = pd.to_datetime(long["monitoring_date_end"], errors="coerce")
    long["month"] = long["monitoring_date_end"].dt.to_period("M").astype(str)
    long["class_score"] = long["water_quality_class"].map(class_score)

    # Equal-month weighting is first performed within each of the 103 public
    # station/source units. Repeated source units at the same physical geometry
    # remain visible here and are collapsed only at the inferential stage.
    cls = long[(long.indicator_name_std == "水质类别") & long.spatial_eligible & long.class_score.notna()].copy()
    cls_month = cls.groupby(["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id",
                             "spatial_waterbody_type", "month"], as_index=False).agg(
        class_score=("class_score", "mean"), n_source_records=("record_id", "size"))
    cls_site = cls_month.groupby(["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id",
                                  "spatial_waterbody_type"], as_index=False).agg(
        wq_class_score_mean=("class_score", "mean"), wq_class_score_median=("class_score", "median"),
        wq_class_score_sd=("class_score", "std"), wq_class_score_best=("class_score", "min"),
        wq_class_score_worst=("class_score", "max"), wq_class_n_months=("month", "nunique"),
        wq_class_start=("month", "min"), wq_class_end=("month", "max"))
    frac = cls_month.assign(good=lambda x: x.class_score <= 2, poor=lambda x: x.class_score >= 4).groupby("station_id").agg(
        wq_good_le_II_fraction=("good", "mean"), wq_poor_ge_IV_fraction=("poor", "mean")).reset_index()
    cls_site = cls_site.merge(frac, on="station_id", how="left")

    pol = long[(long.indicator_name_std == "综合污染指数") & long.spatial_eligible].copy()
    pol["value_numeric"] = pd.to_numeric(pol["value_numeric"], errors="coerce")
    pol = pol[pol.value_numeric.notna()]
    pol_month = pol.groupby(["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id",
                             "spatial_waterbody_type", "month"], as_index=False).agg(
        pollution_index=("value_numeric", "mean"), n_source_records=("record_id", "size"))
    pol_site = pol_month.groupby(["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id",
                                  "spatial_waterbody_type"], as_index=False).agg(
        pollution_index_mean=("pollution_index", "mean"), pollution_index_median=("pollution_index", "median"),
        pollution_index_sd=("pollution_index", "std"), pollution_index_max=("pollution_index", "max"),
        pollution_index_n_months=("month", "nunique"), pollution_index_start=("month", "min"), pollution_index_end=("month", "max"))

    coords = reg[reg.spatial_eligible][[
        "station_id", "spatial_id", "geometry_id", "waterbody_cluster_id", "spatial_waterbody_type",
        "spatial_longitude", "spatial_latitude", "spatial_quality", "waterbody_name_spatial",
        "station_name_canonical", "station_scope", "monitoring_agency",
    ]].copy().rename(columns={"spatial_longitude": "longitude", "spatial_latitude": "latitude"})
    site = coords.merge(cls_site, on=["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id", "spatial_waterbody_type"], how="left", validate="1:1")
    site = site.merge(pol_site, on=["station_id", "spatial_id", "geometry_id", "waterbody_cluster_id", "spatial_waterbody_type"], how="left", validate="1:1")
    # The repaired registry keeps ``spatial_id`` as the public station/source
    # unit identifier, while every GIS feature row is keyed by the unique
    # physical ``geometry_id``.  Preserve both meanings explicitly and use the
    # physical geometry as the inferential/feature-join key.  Without this
    # remapping, all environmental features become missing and duplicated
    # station publications could be mistaken for independent spatial samples.
    site = site.rename(columns={"spatial_id": "station_unit_spatial_id"})
    site["spatial_id"] = site["geometry_id"]
    site = site.merge(static, on="spatial_id", how="left", validate="m:1")
    # Resolve the type field after merge and add 2021-2025 mean population.
    if "water_body_type" in site.columns:
        site = site.drop(columns=["water_body_type"])
    pop_cols = [c for c in pop if c.startswith("worldpop_")]
    pop_site = pop.groupby("spatial_id", as_index=False)[pop_cols].mean()
    site = site.merge(pop_site, on="spatial_id", how="left", validate="m:1")
    site["n_station_units_same_geometry"] = site.groupby("spatial_id")["station_id"].transform("nunique")
    site["geometry_equal_weight"] = 1.0 / site["n_station_units_same_geometry"]
    site.to_csv(PROC / "WQ_Site_Special_13Ranges.csv", index=False, encoding="utf-8-sig")
    cls_month.to_csv(PROC / "WQ_Class_StationUnit_Month.csv", index=False, encoding="utf-8-sig")
    pol_month.to_csv(PROC / "WQ_PollutionIndex_StationUnit_Month.csv", index=False, encoding="utf-8-sig")
    return record, site, cls_month, pol_month


def feature_modules(site):
    modules = {}
    for scale in SCALES:
        s = f"{scale:04d}m"
        modules[("城市化建设", scale)] = [c for c in [
            f"urbanization_{s}_impervious_pct", f"urbanization_{s}_building_pct",
            f"urbanization_{s}_natural_land_pct", f"landuse_{s}_urban_pct",
            f"landuse_{s}_agriculture_pct", f"landuse_{s}_green_pct"] if c in site]
        modules[("人类直接活动", scale)] = [c for c in site if c.startswith(f"api_{s}_")] + ([f"worldpop_{s}_mean_density"] if f"worldpop_{s}_mean_density" in site else [])
        modules[("地形地貌", scale)] = [c for c in [
            f"terrain_{s}_mean_dem_m", f"terrain_{s}_dem_range_m", f"terrain_{s}_dem_std_m",
            f"terrain_{s}_delta_dem_buffer_minus_point_m", f"terrain_{s}_mean_slope_pct",
            f"terrain_{s}_max_slope_pct", f"terrain_{s}_slope_gt10pct_area_pct",
            f"terrain_{s}_slope_gt15pct_area_pct"] if c in site]
    return modules


def transform_for_gradient(df, cols):
    arr = pd.DataFrame(index=df.index)
    rules = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        miss = float(x.isna().mean())
        if miss > 0.20 or x.nunique(dropna=True) < 3:
            rules.append((c, "drop_missing_or_low_variance", miss))
            continue
        med = x.median()
        x = x.fillna(med)
        if c.endswith("_pct") or "area_pct" in c:
            z = np.arcsin(np.sqrt(np.clip(x / 100.0, 0, 1)))
            rule = "arcsin_sqrt_percent"
        elif "density" in c and np.nanmin(x) >= 0:
            z = np.log1p(x)
            rule = "log1p"
        elif np.nanmin(x) >= 0 and abs(skew(x, nan_policy="omit")) > 1.5:
            z = np.log1p(x)
            rule = "log1p_skew"
        else:
            z = x
            rule = "raw"
        arr[c] = z
        rules.append((c, rule, miss))
    if arr.shape[1] == 0:
        return arr, rules
    # Greedy redundancy control; keep ecologically ordered columns.
    kept = []
    for c in arr.columns:
        if not kept:
            kept.append(c); continue
        cors = arr[kept].corrwith(arr[c], method="spearman").abs()
        if not (cors > 0.85).any():
            kept.append(c)
        else:
            rules.append((c, "drop_redundant_abs_rho_gt_0.85", 0.0))
    return arr[kept], rules


def orient_gradient(scores, zdf, driver):
    anchor = []
    signs = []
    for c in zdf.columns:
        if driver == "城市化建设":
            sign = -1 if ("natural" in c or "green" in c) else 1
        else:
            sign = 1
        anchor.append(zdf[c].to_numpy() * sign)
        signs.append(sign)
    if not anchor:
        return scores, np.nan
    a = np.nanmean(np.vstack(anchor), axis=0)
    r = spearmanr(scores, a).statistic
    return (-scores if np.isfinite(r) and r < 0 else scores), r


def build_gradients(site):
    modules = feature_modules(site)
    score_rows, loading_rows, transform_rows = [], [], []
    for water_type in ["Lake", "River"]:
        # Environmental gradients are properties of physical geometries.  The
        # public registry can contain several station/source IDs for the same
        # waterbody geometry, so PCA must use each geometry exactly once.
        sub = (site[site.spatial_waterbody_type.eq(water_type)]
               .drop_duplicates("spatial_id").copy().reset_index(drop=True))
        for (driver, scale), cols in modules.items():
            arr, rules = transform_for_gradient(sub, cols)
            for c, rule, missing in rules:
                transform_rows.append({"water_body_type": water_type, "driver": driver, "range_m": scale,
                                       "feature": c, "rule": rule, "missing_fraction": missing})
            if arr.shape[1] < 2 or len(sub) < 6:
                continue
            scaler = RobustScaler().fit(arr)
            z = pd.DataFrame(scaler.transform(arr), columns=arr.columns, index=arr.index)
            pca = PCA(n_components=1).fit(z)
            score = pca.transform(z)[:, 0]
            score, orientation_corr = orient_gradient(score, z, driver)
            # If orientation flips, loadings must flip as well.
            raw_anchor = np.nanmean(np.vstack([z[c].to_numpy() * (-1 if driver == "城市化建设" and ("natural" in c or "green" in c) else 1) for c in z]), axis=0)
            flip = np.isfinite(spearmanr(pca.transform(z)[:, 0], raw_anchor).statistic) and spearmanr(pca.transform(z)[:, 0], raw_anchor).statistic < 0
            for sid, sc in zip(sub.spatial_id, score):
                score_rows.append({"spatial_id": sid, "water_body_type": water_type, "driver": driver,
                                   "range_m": scale, "gradient_score": sc,
                                   "pc1_explained_variance": pca.explained_variance_ratio_[0],
                                   "n_features": arr.shape[1]})
            for c, load in zip(arr.columns, pca.components_[0]):
                loading_rows.append({"water_body_type": water_type, "driver": driver, "range_m": scale,
                                     "feature": c, "loading": -load if flip else load,
                                     "pc1_explained_variance": pca.explained_variance_ratio_[0]})
    scores = pd.DataFrame(score_rows)
    loadings = pd.DataFrame(loading_rows)
    transforms = pd.DataFrame(transform_rows)
    scores.to_csv(RES / "环境模块梯度得分_13尺度.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(RES / "环境模块PCA载荷_13尺度.csv", index=False, encoding="utf-8-sig")
    transforms.to_csv(RES / "环境变量转换与冗余筛选.csv", index=False, encoding="utf-8-sig")
    return scores, loadings, transforms, modules


def analyse(site, scores, modules):
    outcomes = [
        ("Lake", "wq_class_score_mean", "水质类别平均分（越高越差）", 6),
        ("River", "wq_class_score_mean", "水质类别平均分（越高越差）", 6),
        ("River", "pollution_index_mean", "综合污染指数均值（越高越差）", 6),
    ]
    assoc = []
    for wt, ycol, ycn, min_n in outcomes:
        raw_y = site[site.spatial_waterbody_type.eq(wt)][["station_id", "spatial_id", ycol]].dropna()
        # All station/source units contribute to the response, but the
        # inferential unit is the unique physical geometry.  Equal averaging
        # prevents duplicated publication units from inflating significance.
        ydat = raw_y.groupby("spatial_id", as_index=False).agg(
            **{ycol: (ycol, "mean")},
            n_station_units_raw=("station_id", "nunique"),
        )
        n_station_units_raw = int(raw_y.station_id.nunique())
        for driver in ["城市化建设", "人类直接活动", "地形地貌"]:
            for scale in SCALES:
                g = scores[(scores.water_body_type == wt) & (scores.driver == driver) & (scores.range_m == scale)]
                d = ydat.merge(g[["spatial_id", "gradient_score", "pc1_explained_variance", "n_features"]], on="spatial_id", how="inner")
                if len(d) < min_n:
                    continue
                rho, p = permutation_spearman(d.gradient_score, d[ycol])
                lo, hi = bootstrap_spearman(d.gradient_score, d[ycol])
                assoc.append({"water_body_type": wt, "response": ycol, "response_cn": ycn, "driver": driver,
                              "range_m": scale, "n_station_units_raw": n_station_units_raw,
                              "n_spatial_units": len(d), "spearman_rho": rho,
                              "ci95_low": lo, "ci95_high": hi, "permutation_p": p,
                              "jackknife_sign_stability": jackknife_sign_stability(d.gradient_score, d[ycol], rho),
                              "pc1_explained_variance": d.pc1_explained_variance.iloc[0], "n_features": d.n_features.iloc[0]})
    assoc = pd.DataFrame(assoc)
    if len(assoc):
        assoc["fdr_q"] = assoc.groupby(["water_body_type", "response", "driver"])["permutation_p"].transform(bh_q)
        assoc["direction_interpretation"] = np.where(assoc.spearman_rho > 0, "梯度升高对应水质变差", "梯度升高对应水质改善")
    assoc.to_csv(RES / "环境模块梯度与水质响应_13尺度.csv", index=False, encoding="utf-8-sig")

    best = (assoc.assign(abs_rho=assoc.spearman_rho.abs()).sort_values(["water_body_type", "response", "driver", "abs_rho"], ascending=[True, True, True, False])
            .groupby(["water_body_type", "response", "driver"], as_index=False).head(1)) if len(assoc) else pd.DataFrame()
    best.to_csv(RES / "环境模块梯度与水质_候选最强尺度.csv", index=False, encoding="utf-8-sig")

    # Individual-feature scan. Correlation is rank-based, so transformation
    # does not change rho; the transform rules are retained for modelling.
    indiv = []
    for wt, ycol, ycn, min_n in outcomes:
        raw = site[site.spatial_waterbody_type.eq(wt)].copy()
        n_station_units_raw = int(raw.loc[raw[ycol].notna(), "station_id"].nunique())
        response = (raw[["spatial_id", ycol]].dropna()
                    .groupby("spatial_id", as_index=False)[ycol].mean())
        predictors = raw.drop_duplicates("spatial_id").set_index("spatial_id")
        for (driver, scale), cols in modules.items():
            for c in cols:
                if c not in predictors.columns:
                    continue
                d = response.merge(predictors[[c]].reset_index(), on="spatial_id", how="inner")
                d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
                d[c] = pd.to_numeric(d[c], errors="coerce")
                d = d.dropna()
                if len(d) < min_n or d[c].nunique() < 3 or d[ycol].nunique() < 3:
                    continue
                rho, p = permutation_spearman(d[c], d[ycol], n_perm=1999)
                indiv.append({"water_body_type": wt, "response": ycol, "response_cn": ycn, "driver": driver,
                              "range_m": scale, "feature": c,
                              "n_station_units_raw": n_station_units_raw,
                              "n_spatial_units": len(d),
                              "spearman_rho": rho, "permutation_p": p})
    indiv = pd.DataFrame(indiv)
    if len(indiv):
        indiv["fdr_q"] = indiv.groupby(["water_body_type", "response", "driver"])["permutation_p"].transform(bh_q)
        indiv["abs_rho"] = indiv.spearman_rho.abs()
    indiv.to_csv(RES / "单变量水质响应筛查_全尺度.csv", index=False, encoding="utf-8-sig")
    return assoc, best, indiv


def run_qa(record, site):
    static = pd.read_csv(PROC / "WQ_Environmental_Features_Static_Wide.csv")
    env = [c for c in static if c not in ["spatial_id", "water_body_type"]]
    missing = static.melt(id_vars=["spatial_id", "water_body_type"], value_vars=env, var_name="feature", value_name="value")
    missing = missing[missing.value.isna()]
    missing.to_csv(QA / "静态特征缺失单元格.csv", index=False, encoding="utf-8-sig")
    pct_cols = [c for c in env if (
        c.startswith("urbanization_") or c.startswith("landuse_") or
        (c.startswith("api_") and c.endswith("polygon_area_pct")) or
        (c.startswith("terrain_") and ("slope_gt10pct_area_pct" in c or "slope_gt15pct_area_pct" in c))
    )]
    pct = static.melt(id_vars=["spatial_id", "water_body_type"], value_vars=pct_cols, var_name="feature", value_name="value")
    pct["out_of_0_100"] = (pct.value < -1e-8) | (pct.value > 100 + 1e-8)
    pct[pct.out_of_0_100].to_csv(QA / "面积比例0至100范围异常.csv", index=False, encoding="utf-8-sig")
    lake = site[site.spatial_waterbody_type == "Lake"]
    river = site[site.spatial_waterbody_type == "River"]
    overview = pd.DataFrame([
        ["record_special_rows", len(record)], ["station_source_units", site.station_id.nunique()],
        ["unique_spatial_geometries", site.spatial_id.nunique()],
        ["unique_waterbody_clusters", site.waterbody_cluster_id.nunique()],
        ["lake_station_source_units", lake.station_id.nunique()],
        ["river_station_source_units", river.station_id.nunique()],
        ["lake_unique_spatial_geometries", lake.spatial_id.nunique()],
        ["river_unique_spatial_geometries", river.spatial_id.nunique()],
        ["static_environment_features", len(env)], ["population_features", 13],
        ["static_missing_cells", len(missing)], ["percent_out_of_range_cells", int(pct.out_of_0_100.sum())],
        ["lake_class_station_units_n", int(lake.wq_class_score_mean.notna().sum())],
        ["lake_class_spatial_geometries_n", int(lake.loc[lake.wq_class_score_mean.notna(), "spatial_id"].nunique())],
        ["river_class_station_units_n", int(river.wq_class_score_mean.notna().sum())],
        ["river_class_spatial_geometries_n", int(river.loc[river.wq_class_score_mean.notna(), "spatial_id"].nunique())],
        ["river_pollution_station_units_n", int(river.pollution_index_mean.notna().sum())],
        ["river_pollution_spatial_geometries_n", int(river.loc[river.pollution_index_mean.notna(), "spatial_id"].nunique())],
    ], columns=["item", "value"])
    overview.to_csv(QA / "WQ_INPUT与分析数据概览.csv", index=False, encoding="utf-8-sig")
    return overview


def main():
    record, site, cls_month, pol_month = build_input()
    scores, loadings, transforms, modules = build_gradients(site)
    assoc, best, indiv = analyse(site, scores, modules)
    overview = run_qa(record, site)
    summary = {"record_rows": len(record), "site_rows": len(site), "gradient_rows": len(scores),
               "module_association_rows": len(assoc), "individual_association_rows": len(indiv)}
    (QA / "分析运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
