from __future__ import annotations

import math
import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "processed" / "训练数据汇总"
SUMMARY_DIR = ROOT / "data" / "processed" / "Summary"
SAMPLE_DIR = TRAIN_DIR / "Sample_Special"
SITE_DIR = TRAIN_DIR / "Site_Special"

TODAY = date.today().isoformat()
PY_ENGINE = "openpyxl"

SAMPLE_RENAMES = {
    "Feature_Database_API.xlsx": "Feature_Database_API_Sample_Special.xlsx",
    "Feature_Database_Aquatic_Biology.xlsx": "Feature_Database_Aquatic_Biology_Sample_Special.xlsx",
    "Feature_Database_Beta.xlsx": "Feature_Database_Beta_Sample_Special.xlsx",
    "Feature_Database_LandUse_WorldCover.xlsx": "Feature_Database_LandUse_WorldCover_Sample_Special.xlsx",
    "Feature_Database_Urbanization_OSM.xlsx": "Feature_Database_Urbanization_OSM_Sample_Special.xlsx",
    "Feature_Database_WorldPop.xlsx": "Feature_Database_WorldPop_Sample_Special.xlsx",
    "Feature_Process_API.xlsx": "Feature_Process_API_Sample_Special.xlsx",
    "Feature_Process_Aquatic_Biology.xlsx": "Feature_Process_Aquatic_Biology_Sample_Special.xlsx",
    "Feature_Process_Beta_Visualization_Index_Update.xlsx": "Feature_Process_Beta_Visualization_Index_Update_Sample_Special.xlsx",
    "Feature_Process_LandUse_WorldCover.xlsx": "Feature_Process_LandUse_WorldCover_Sample_Special.xlsx",
    "Feature_Process_Urbanization_OSM.xlsx": "Feature_Process_Urbanization_OSM_Sample_Special.xlsx",
    "Feature_Process_WorldPop.xlsx": "Feature_Process_WorldPop_Sample_Special.xlsx",
    "ML_Input_Sample_Feature_Matrix.xlsx": "ML_Input_Sample_Special_Feature_Matrix.xlsx",
    "机器学习特征构建与数据处理方法综合说明.docx": "机器学习特征构建与数据处理方法综合说明_Sample_Special.docx",
}

TAXON_RULES = {
    "benthos": {
        "diversity_sheet": "Benthos_Qualitative",
        "diversity_value": None,
        "diversity_mode": "presence/absence",
        "totals_sheet": "Benthos",
        "density_col": "Density",
        "biomass_col": "Biomass",
        "total_mode": "sum(Density), sum(Biomass)",
        "applicable": "Lake and River",
    },
    "fish": {
        "diversity_sheet": "Fish",
        "diversity_value": "__record_count__",
        "diversity_mode": "individual record count",
        "totals_sheet": "Fish",
        "density_col": "__record_count__",
        "biomass_col": "Weight",
        "total_mode": "count(records), sum(Weight)",
        "applicable": "Lake and River",
    },
    "macrophytes": {
        "diversity_sheet": "Macrophytes",
        "diversity_value": "Density",
        "diversity_mode": "sum(Density)",
        "totals_sheet": "Macrophytes",
        "density_col": "Density",
        "biomass_col": "Biomass",
        "total_mode": "sum(Density), sum(Biomass)",
        "applicable": "Lake and River",
    },
    "phytoplankton": {
        "diversity_sheet": "Phytoplankton",
        "diversity_value": "Raw value",
        "diversity_mode": "sum(Raw value)",
        "totals_sheet": "Phytoplankton",
        "density_col": "Density",
        "biomass_col": "Biomass",
        "total_mode": "sum(Density), sum(Biomass)",
        "applicable": "Lake and River",
    },
    "periphyton": {
        "diversity_sheet": "Periphyton",
        "diversity_value": "Raw value",
        "diversity_mode": "sum(Raw value); River only",
        "totals_sheet": "Periphyton",
        "density_col": "Density",
        "biomass_col": "Biomass",
        "total_mode": "sum(Density), sum(Biomass); River only",
        "applicable": "River only",
    },
    "zooplankton": {
        "diversity_sheet": "Zooplankton",
        "diversity_value": "Density",
        "diversity_mode": "sum(Density)",
        "totals_sheet": "Zooplankton",
        "density_col": "Density",
        "biomass_col": "Biomass",
        "total_mode": "sum(Density), sum(Biomass)",
        "applicable": "Lake and River",
    },
}

BASE_SAMPLE_COLUMNS = [
    "sample_id",
    "water_body_type",
    "sampling_time",
    "sampling_year",
    "flow_period",
    "site",
]
SITE_INFO_COLUMNS = [
    "site_special_id",
    "water_body_type",
    "site",
    "longitude",
    "latitude",
    "n_sample_records",
    "sampling_years",
    "flow_periods",
]

RANGES = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000]


def safe_rename_sample_files() -> pd.DataFrame:
    records: list[dict[str, str]] = []
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for old_name, new_name in SAMPLE_RENAMES.items():
        old = SAMPLE_DIR / old_name
        new = SAMPLE_DIR / new_name
        if old.exists() and not new.exists():
            old.rename(new)
            status = "renamed"
        elif new.exists():
            status = "already_sample_special"
        elif old.exists() and new.exists():
            status = "both_exist"
        else:
            status = "missing_old_name"
        records.append({"old_name": old_name, "new_name": new_name, "status": status})
    return pd.DataFrame(records)


def p_sample(name: str) -> Path:
    sample_name = SAMPLE_RENAMES.get(name, name)
    path = SAMPLE_DIR / sample_name
    if path.exists():
        return path
    fallback = SAMPLE_DIR / name
    if fallback.exists():
        return fallback
    legacy = TRAIN_DIR / sample_name
    if legacy.exists():
        return legacy
    legacy_fallback = TRAIN_DIR / name
    if legacy_fallback.exists():
        return legacy_fallback
    raise FileNotFoundError(f"Cannot find {name} or {sample_name}")


def parse_sampling_time(value: object) -> tuple[str, str]:
    text = "" if pd.isna(value) else str(value)
    year_match = re.search(r"(20\d{2})", text)
    year = year_match.group(1) if year_match else ""
    lower = text.lower()
    if (
        "high-flow" in lower
        or "high flow" in lower
        or "hjgh-flow" in lower
        or "hjgh flow" in lower
        or re.search(r"\bhigh\b", lower)
        or re.search(r"\bhjgh\b", lower)
        or "丰" in text
    ):
        flow = "high-flow"
    elif "low-flow" in lower or "low flow" in lower or re.search(r"\blow\b", lower) or "枯" in text:
        flow = "low-flow"
    else:
        flow = ""
    return year, flow


def make_sample_id(water_body_type: object, site: object, sampling_time: object) -> str:
    year, flow = parse_sampling_time(sampling_time)
    flow_short = "high" if flow == "high-flow" else "low" if flow == "low-flow" else "unknown"
    return f"{str(water_body_type).strip()}_{str(site).strip()}_{year}_{flow_short}"


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def taxon_key(df: pd.DataFrame) -> pd.Series:
    sci = df.get("Scientific name")
    species = df.get("Species")
    if sci is None:
        return species.fillna("").astype(str).str.strip()
    key = sci.fillna("").astype(str).str.strip()
    if species is not None:
        species_key = species.fillna("").astype(str).str.strip()
        key = key.mask(key.eq("") | key.str.lower().eq("nan"), species_key)
    return key


def add_sample_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["water_body_type"] = out["Water body type"].astype(str).str.strip()
    out["sampling_time"] = out["Sampling time"].astype(str).str.strip()
    out["site"] = out["Site"].astype(str).str.strip()
    parsed = out["sampling_time"].map(parse_sampling_time)
    out["sampling_year"] = parsed.map(lambda x: x[0])
    out["flow_period"] = parsed.map(lambda x: x[1])
    out["sample_id"] = [
        make_sample_id(wt, site, st)
        for wt, site, st in zip(out["water_body_type"], out["site"], out["sampling_time"])
    ]
    return out


def diversity_metrics(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return {"richness": 0, "shannon": 0.0, "margalef": 0.0, "pielou": 0.0, "simpson": 0.0}
    total = float(arr.sum())
    richness = int(arr.size)
    p = arr / total if total > 0 else arr
    shannon = float(-np.sum(p * np.log(p))) if total > 0 else 0.0
    margalef = float((richness - 1) / math.log(total)) if total > 1 and richness > 1 else 0.0
    pielou = float(shannon / math.log(richness)) if richness > 1 else 0.0
    simpson = float(1 - np.sum(p**2)) if total > 0 else 0.0
    return {
        "richness": richness,
        "shannon": shannon,
        "margalef": margalef,
        "pielou": pielou,
        "simpson": simpson,
    }


def build_sample_universe(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sheet, df in sheets.items():
        if sheet == "Water_Type_QA":
            continue
        d = add_sample_fields(df)
        frames.append(d[BASE_SAMPLE_COLUMNS])
    universe = pd.concat(frames, ignore_index=True)
    universe = universe[BASE_SAMPLE_COLUMNS].drop_duplicates("sample_id")
    universe = universe.sort_values(["water_body_type", "site", "sampling_year", "flow_period"]).reset_index(drop=True)
    return universe


def build_abundance_long(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group, rule in TAXON_RULES.items():
        df = add_sample_fields(sheets[rule["diversity_sheet"]])
        df["taxon_key"] = taxon_key(df)
        df = df[df["taxon_key"].ne("")]
        if group == "periphyton":
            df = df[df["water_body_type"].eq("River")]
        value_col = rule["diversity_value"]
        if value_col is None:
            abund = (
                df.drop_duplicates(["sample_id", "taxon_key"])
                .assign(abundance=1.0)
                .groupby(["sample_id", "taxon_key"], as_index=False)["abundance"]
                .sum()
            )
        elif value_col == "__record_count__":
            abund = (
                df.groupby(["sample_id", "taxon_key"], as_index=False)
                .size()
                .rename(columns={"size": "abundance"})
            )
        else:
            d = df.copy()
            d["abundance"] = clean_numeric(d[value_col]).fillna(0)
            abund = d.groupby(["sample_id", "taxon_key"], as_index=False)["abundance"].sum()
        abund = abund[abund["abundance"] > 0]
        meta = df[BASE_SAMPLE_COLUMNS].drop_duplicates("sample_id")
        abund = abund.merge(meta, on="sample_id", how="left")
        abund["taxon_group"] = group
        abund["abundance_mode"] = rule["diversity_mode"]
        abund["overall_taxon_key"] = group + "::" + abund["taxon_key"].astype(str)
        frames.append(
            abund[
                BASE_SAMPLE_COLUMNS
                + ["taxon_group", "taxon_key", "overall_taxon_key", "abundance", "abundance_mode"]
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["sample_id", "taxon_group", "taxon_key"]).reset_index(drop=True)


def build_taxon_diversity(abundance_long: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (sample_id, group), sub in abundance_long.groupby(["sample_id", "taxon_group"]):
        row_meta = sub.iloc[0].to_dict()
        metrics = diversity_metrics(sub["abundance"])
        records.append(
            {
                "sample_id": sample_id,
                "taxon_group": group,
                "water_body_type": row_meta["water_body_type"],
                "sampling_time": row_meta["sampling_time"],
                "sampling_year": row_meta["sampling_year"],
                "flow_period": row_meta["flow_period"],
                "site": row_meta["site"],
                **metrics,
                "abundance_mode": row_meta["abundance_mode"],
            }
        )
    return pd.DataFrame(records).sort_values(["sample_id", "taxon_group"]).reset_index(drop=True)


def build_taxon_totals(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group, rule in TAXON_RULES.items():
        df = add_sample_fields(sheets[rule["totals_sheet"]])
        if group == "periphyton":
            df = df[df["water_body_type"].eq("River")]
        if rule["density_col"] == "__record_count__":
            density = df.groupby("sample_id").size().rename("total_density")
        else:
            density = df.assign(_density=clean_numeric(df[rule["density_col"]]).fillna(0)).groupby("sample_id")["_density"].sum()
            density.name = "total_density"
        biomass_col = rule["biomass_col"]
        biomass = df.assign(_biomass=clean_numeric(df[biomass_col]).fillna(0)).groupby("sample_id")["_biomass"].sum()
        biomass.name = "total_biomass"
        totals = pd.concat([density, biomass], axis=1).reset_index()
        meta = df[BASE_SAMPLE_COLUMNS].drop_duplicates("sample_id")
        totals = totals.merge(meta, on="sample_id", how="left")
        for _, row in totals.iterrows():
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "taxon_group": group,
                    "water_body_type": row["water_body_type"],
                    "sampling_time": row["sampling_time"],
                    "sampling_year": row["sampling_year"],
                    "flow_period": row["flow_period"],
                    "site": row["site"],
                    "total_density": row["total_density"],
                    "total_biomass": row["total_biomass"],
                    "total_mode": rule["total_mode"],
                }
            )
    return pd.DataFrame(records).sort_values(["sample_id", "taxon_group"]).reset_index(drop=True)


def build_overall_diversity(abundance_long: pd.DataFrame, sample_universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, sub in abundance_long.groupby("sample_id"):
        meta = sample_universe.loc[sample_universe["sample_id"].eq(sample_id)]
        if meta.empty:
            continue
        meta_row = meta.iloc[0].to_dict()
        combined = sub.groupby("overall_taxon_key", as_index=False)["abundance"].sum()
        metrics = diversity_metrics(combined["abundance"])
        rows.append(
            {
                "sample_id": sample_id,
                "water_body_type": meta_row["water_body_type"],
                "sampling_time": meta_row["sampling_time"],
                "sampling_year": meta_row["sampling_year"],
                "flow_period": meta_row["flow_period"],
                "site": meta_row["site"],
                "overall_richness": metrics["richness"],
                "overall_shannon": metrics["shannon"],
                "overall_margalef": metrics["margalef"],
                "overall_pielou": metrics["pielou"],
                "overall_simpson": metrics["simpson"],
                "total_richness": metrics["richness"],
            }
        )
    out = pd.DataFrame(rows)
    return sample_universe.merge(out, on=BASE_SAMPLE_COLUMNS, how="left")


def pivot_diversity(sample_universe: pd.DataFrame, taxon_div: pd.DataFrame) -> pd.DataFrame:
    feature_parts = []
    for group in TAXON_RULES:
        sub = taxon_div[taxon_div["taxon_group"].eq(group)]
        wide = sub.pivot(index="sample_id", columns="taxon_group", values=["richness", "shannon", "margalef", "pielou", "simpson"])
        if wide.empty:
            continue
        wide.columns = [f"{grp}_{metric}" for metric, grp in wide.columns]
        feature_parts.append(wide)
    combined = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame(index=sample_universe["sample_id"])
    out = sample_universe.merge(combined.reset_index(), on="sample_id", how="left")
    feature_cols = [c for c in out.columns if c not in BASE_SAMPLE_COLUMNS]
    for col in feature_cols:
        if col.startswith("periphyton_"):
            out.loc[out["water_body_type"].eq("Lake"), col] = np.nan
        out.loc[out[col].isna() & ~out["water_body_type"].eq("Lake"), col] = out.loc[
            out[col].isna() & ~out["water_body_type"].eq("Lake"), col
        ].fillna(0)
        if not col.startswith("periphyton_"):
            out[col] = out[col].fillna(0)
    return out


def pivot_totals(sample_universe: pd.DataFrame, taxon_totals: pd.DataFrame) -> pd.DataFrame:
    feature_parts = []
    for group in TAXON_RULES:
        sub = taxon_totals[taxon_totals["taxon_group"].eq(group)]
        if sub.empty:
            continue
        wide = sub.pivot(index="sample_id", columns="taxon_group", values=["total_density", "total_biomass"])
        wide.columns = [f"{grp}_{metric}" for metric, grp in wide.columns]
        feature_parts.append(wide)
    combined = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame(index=sample_universe["sample_id"])
    out = sample_universe.merge(combined.reset_index(), on="sample_id", how="left")
    feature_cols = [c for c in out.columns if c not in BASE_SAMPLE_COLUMNS]
    for col in feature_cols:
        if col.startswith("periphyton_"):
            out.loc[out["water_body_type"].eq("Lake"), col] = np.nan
        out.loc[out[col].isna() & ~out["water_body_type"].eq("Lake"), col] = out.loc[
            out[col].isna() & ~out["water_body_type"].eq("Lake"), col
        ].fillna(0)
        if not col.startswith("periphyton_"):
            out[col] = out[col].fillna(0)
    return out


def robust_params(values: pd.Series) -> tuple[float, float, float, float, bool]:
    x = np.log10(pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float) + 1)
    median = float(np.nanmedian(x))
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    iqr = q3 - q1
    fallback = False
    if not np.isfinite(iqr) or iqr == 0:
        iqr = float(np.nanstd(x))
        fallback = True
    if not np.isfinite(iqr) or iqr == 0:
        iqr = 1.0
        fallback = True
    return median, q1, q3, iqr, fallback


def standardize_totals(totals_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = totals_wide[BASE_SAMPLE_COLUMNS].copy()
    records: list[dict[str, object]] = []
    for group in TAXON_RULES:
        for metric in ["total_density", "total_biomass"]:
            source_col = f"{group}_{metric}"
            if source_col not in totals_wide.columns:
                continue
            output_col = f"{source_col}_log10p1_robust"
            values = totals_wide[source_col].copy()
            applicable = pd.Series(True, index=totals_wide.index)
            if TAXON_RULES[group]["applicable"] == "River only":
                applicable = totals_wide["water_body_type"].eq("River")
            train_values = values[applicable].fillna(0)
            median, q1, q3, iqr, fallback = robust_params(train_values)
            numeric_values = clean_numeric(values).fillna(0)
            transformed = (np.log10(numeric_values + 1) - median) / iqr
            transformed = transformed.astype(float)
            transformed.loc[~applicable] = np.nan
            out[output_col] = transformed
            records.append(
                {
                    "source_column": source_col,
                    "standardized_column": output_col,
                    "group": group,
                    "metric": metric.replace("total_", ""),
                    "applicable_scope": TAXON_RULES[group]["applicable"],
                    "missing_handling": "missing values imputed as 0 before log transform; not applicable values kept as NA",
                    "transform": "log10(x + 1)",
                    "scaling": "(log10(x+1) - median) / IQR",
                    "median_log10p1": median,
                    "q1_log10p1": q1,
                    "q3_log10p1": q3,
                    "iqr_used": iqr,
                    "iqr_fallback_to_sd_or_1": fallback,
                    "n_applicable_samples": int(applicable.sum()),
                    "n_raw_nonmissing": int(values[applicable].notna().sum()),
                }
            )
    return out, pd.DataFrame(records)


def aggregate_site_mean(sample_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    grouped = sample_df.groupby(["water_body_type", "site"], as_index=False)[feature_cols].mean(numeric_only=True)
    return grouped


def build_site_metadata(sample_input: pd.DataFrame) -> pd.DataFrame:
    agg = (
        sample_input.groupby(["water_body_type", "site"], as_index=False)
        .agg(
            longitude=("longitude", "first"),
            latitude=("latitude", "first"),
            n_sample_records=("sample_id", "nunique"),
            sampling_years=("sampling_year", lambda x: ";".join(sorted({str(v) for v in x if pd.notna(v)}))),
            flow_periods=("flow_period", lambda x: ";".join(sorted({str(v) for v in x if pd.notna(v)}))),
        )
        .sort_values(["water_body_type", "site"])
        .reset_index(drop=True)
    )
    agg.insert(0, "site_special_id", agg["water_body_type"].astype(str) + "_" + agg["site"].astype(str))
    return agg[SITE_INFO_COLUMNS]


def build_aquatic_site_features(rename_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = SUMMARY_DIR / "Aquatic_Biological_Database.xlsx"
    sheets = pd.read_excel(source, sheet_name=None, engine=PY_ENGINE)
    sample_universe = build_sample_universe(sheets)
    abundance_long = build_abundance_long(sheets)
    taxon_div = build_taxon_diversity(abundance_long)
    taxon_totals = build_taxon_totals(sheets)
    overall = build_overall_diversity(abundance_long, sample_universe)
    div_wide = pivot_diversity(sample_universe, taxon_div)
    totals_wide = pivot_totals(sample_universe, taxon_totals)
    totals_std, std_params = standardize_totals(totals_wide)

    diversity_feature_cols = []
    for group in TAXON_RULES:
        diversity_feature_cols += [
            f"{group}_richness",
            f"{group}_shannon",
            f"{group}_margalef",
            f"{group}_pielou",
            f"{group}_simpson",
        ]
    overall_cols = ["overall_richness", "total_richness", "overall_shannon", "overall_margalef", "overall_pielou", "overall_simpson"]
    total_std_cols = [c for c in totals_std.columns if c.endswith("_log10p1_robust")]

    sample_features = sample_universe.copy()
    sample_features = sample_features.merge(div_wide[["sample_id"] + diversity_feature_cols], on="sample_id", how="left")
    sample_features = sample_features.merge(overall[["sample_id"] + overall_cols], on="sample_id", how="left")
    sample_features = sample_features.merge(totals_std[["sample_id"] + total_std_cols], on="sample_id", how="left")

    # Match the current ML input convention: 48 aquatic columns.
    aquatic_cols = diversity_feature_cols + overall_cols + total_std_cols
    site_mean = aggregate_site_mean(sample_features, aquatic_cols)
    site_mean = site_mean.rename(columns={"site": "site"})

    site_sample_summary = (
        sample_universe.groupby(["water_body_type", "site"], as_index=False)
        .agg(
            n_sample_records=("sample_id", "nunique"),
            sampling_times=("sampling_time", lambda x: "; ".join(sorted({str(v) for v in x}))),
            sampling_years=("sampling_year", lambda x: ";".join(sorted({str(v) for v in x}))),
            flow_periods=("flow_period", lambda x: ";".join(sorted({str(v) for v in x}))),
        )
        .sort_values(["water_body_type", "site"])
    )

    source_qa = []
    for name, df in sheets.items():
        source_qa.append(
            {
                "sheet_name": name,
                "rows": len(df),
                "columns": len(df.columns),
                "water_type_errors": "",
            }
        )
    source_qa = pd.DataFrame(source_qa)

    process_qa = pd.DataFrame(
        [
            ["source_database", str(source), "raw aquatic biological database"],
            ["sample_rows_recomputed", len(sample_universe), "unique point x sampling time samples"],
            ["site_rows", site_mean.shape[0], "unique sites after site-special aggregation"],
            ["aquatic_feature_columns", len(aquatic_cols), "48 columns expected"],
            ["aggregation_rule", "mean across sample-level features within site", "do not merge raw multi-time species records"],
            ["sample_special_rename_log_rows", len(rename_log), "file rename trace"],
        ],
        columns=["check_item", "value", "note"],
    )

    feature_dictionary = pd.DataFrame(
        [{"column_name": c, "role": "site_info" if c in ["water_body_type", "site"] else "aquatic_biology_site_mean_feature"} for c in site_mean.columns]
    )

    db_path = SITE_DIR / "Feature_Database_Aquatic_Biology_Site_Special.xlsx"
    process_path = SITE_DIR / "Feature_Process_Aquatic_Biology_Site_Special.xlsx"
    with pd.ExcelWriter(process_path, engine=PY_ENGINE) as writer:
        source_qa.to_excel(writer, sheet_name="Source_Sheet_QA", index=False)
        abundance_long.to_excel(writer, sheet_name="Taxon_Abundance_Long_Sample", index=False)
        taxon_div.to_excel(writer, sheet_name="Taxon_Diversity_Long_Sample", index=False)
        taxon_totals.to_excel(writer, sheet_name="Taxon_Totals_Long_Sample", index=False)
        overall.to_excel(writer, sheet_name="Overall_Diversity_Sample", index=False)
        sample_features.to_excel(writer, sheet_name="Sample_Level_Features_Rebuilt", index=False)
        site_sample_summary.to_excel(writer, sheet_name="Site_Sample_Summary", index=False)
        site_mean.to_excel(writer, sheet_name="Site_Mean_Features", index=False)
        std_params.to_excel(writer, sheet_name="Standardization_Params", index=False)
        process_qa.to_excel(writer, sheet_name="Process_QA", index=False)
        rename_log.to_excel(writer, sheet_name="Sample_Special_Rename_Log", index=False)
    with pd.ExcelWriter(db_path, engine=PY_ENGINE) as writer:
        site_mean.to_excel(writer, sheet_name="input_feature", index=False)
        feature_dictionary.to_excel(writer, sheet_name="Feature_Dictionary", index=False)
        std_params.to_excel(writer, sheet_name="Standardization_Params", index=False)
        process_qa.to_excel(writer, sheet_name="QA_Checks", index=False)
    return site_mean, sample_features, process_qa


def beta_site_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = p_sample("Feature_Database_Beta.xlsx")
    beta_scaled = pd.read_excel(source, sheet_name="input_feature_scaled_optional", engine=PY_ENGINE)
    beta_raw = pd.read_excel(source, sheet_name="input_feature", engine=PY_ENGINE)
    beta_long = pd.read_excel(source, sheet_name="Beta_Features_Long", engine=PY_ENGINE)
    beta_feature_cols = [c for c in beta_scaled.columns if c not in BASE_SAMPLE_COLUMNS]
    beta_raw_cols = [c for c in beta_raw.columns if c not in BASE_SAMPLE_COLUMNS]

    site_scaled = aggregate_site_mean(beta_scaled, beta_feature_cols)
    site_raw = aggregate_site_mean(beta_raw, beta_raw_cols)
    site_long = (
        beta_long.groupby(["water_body_type", "site", "taxon_group", "beta_scope", "distance_metric"], as_index=False)
        .agg(
            mean_dissimilarity_mean=("mean_dissimilarity", "mean"),
            LCBD_mean=("LCBD", "mean"),
            n_sample_records_with_feature=("sample_id", "nunique"),
            n_samples_in_matrix=("n_samples_in_matrix", "first"),
            n_species_in_matrix=("n_species_in_matrix", "first"),
        )
        .sort_values(["water_body_type", "site", "taxon_group", "beta_scope", "distance_metric"])
    )

    process_qa = pd.DataFrame(
        [
            ["source_beta_database", str(source), "sample-special beta feature database"],
            ["source_sheet_for_ml", "input_feature_scaled_optional", "consistent with current 577-row ML input"],
            ["site_rows", site_scaled.shape[0], "unique sites"],
            ["beta_feature_columns", len(beta_feature_cols), "64 mean dissimilarity + 64 LCBD/log robust columns"],
            ["aggregation_rule", "mean across sample-level beta features within site", "sampling-frequency-aware site summary"],
        ],
        columns=["check_item", "value", "note"],
    )
    feature_dictionary = pd.DataFrame(
        [
            {
                "column_name": c,
                "role": "site_info" if c in ["water_body_type", "site"] else ("beta_feature_lcbd" if "lcbd" in c.lower() else "beta_feature_mean_dissimilarity"),
            }
            for c in site_scaled.columns
        ]
    )
    process_path = SITE_DIR / "Feature_Process_Beta_Site_Special.xlsx"
    db_path = SITE_DIR / "Feature_Database_Beta_Site_Special.xlsx"
    with pd.ExcelWriter(process_path, engine=PY_ENGINE) as writer:
        beta_long.to_excel(writer, sheet_name="Beta_Features_Long_Sample", index=False)
        site_long.to_excel(writer, sheet_name="Beta_Features_Long_SiteMean", index=False)
        site_raw.to_excel(writer, sheet_name="Site_Features_Wide_RawMean", index=False)
        site_scaled.to_excel(writer, sheet_name="Site_Features_Wide_ScaledMean", index=False)
        process_qa.to_excel(writer, sheet_name="Process_QA", index=False)
    with pd.ExcelWriter(db_path, engine=PY_ENGINE) as writer:
        site_scaled.to_excel(writer, sheet_name="input_feature", index=False)
        site_raw.to_excel(writer, sheet_name="input_feature_raw_lcbd_mean", index=False)
        feature_dictionary.to_excel(writer, sheet_name="Feature_Dictionary", index=False)
        process_qa.to_excel(writer, sheet_name="QA_Checks", index=False)
    return site_scaled, process_qa


def flatten_api_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = p_sample("Feature_Database_API.xlsx")
    api = pd.read_excel(source, sheet_name="API_Features_Long", engine=PY_ENGINE)
    value_cols = ["API_expert_weighted"] + [f"P{i}_z" for i in range(1, 9)]
    wide_parts = []
    for col in value_cols:
        part = api.pivot_table(index=["water_body_type", "site_id", "longitude", "latitude"], columns="buffer_m", values=col, aggfunc="first")
        part.columns = [f"{col}_{int(b):04d}m" for b in part.columns]
        wide_parts.append(part)
    wide = pd.concat(wide_parts, axis=1).reset_index().rename(columns={"site_id": "site"})
    wide = wide.sort_values(["water_body_type", "site"]).reset_index(drop=True)
    qa = pd.DataFrame(
        [
            ["source_api_database", str(source), "sample-special API database"],
            ["site_rows", wide.shape[0], "unique sites"],
            ["api_feature_columns", wide.shape[1] - 4, "10 buffers x 9 API features"],
            ["aggregation_rule", "direct site-level pivot from site x buffer API table", "static spatial feature"],
        ],
        columns=["check_item", "value", "note"],
    )
    return wide, qa


def copy_site_feature_database(source_name: str, sheet: str, output_stem: str, note: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = p_sample(source_name)
    df = pd.read_excel(source, sheet_name=sheet, engine=PY_ENGINE)
    if "site_id" in df.columns:
        df = df.rename(columns={"site_id": "site"})
    df = df.sort_values(["water_body_type", "site"]).reset_index(drop=True)
    qa = pd.DataFrame(
        [
            ["source_database", str(source), note],
            ["site_rows", df.shape[0], "unique sites"],
            ["feature_columns", df.shape[1] - 4, "excluding site, water_body_type, longitude, latitude"],
            ["aggregation_rule", "direct site-level values from sample-special database", "static spatial feature"],
        ],
        columns=["check_item", "value", "note"],
    )
    db_path = SITE_DIR / f"Feature_Database_{output_stem}_Site_Special.xlsx"
    with pd.ExcelWriter(db_path, engine=PY_ENGINE) as writer:
        df.to_excel(writer, sheet_name="feature_data", index=False)
        qa.to_excel(writer, sheet_name="QA_Checks", index=False)
    return df, qa


def copy_process_with_site_note(source_name: str, output_stem: str, qa: pd.DataFrame) -> None:
    source = p_sample(source_name)
    out = SITE_DIR / f"Feature_Process_{output_stem}_Site_Special.xlsx"
    sheets = pd.read_excel(source, sheet_name=None, engine=PY_ENGINE)
    with pd.ExcelWriter(out, engine=PY_ENGINE) as writer:
        pd.DataFrame(
            [
                ["site_special_process_note", "Values are site-level or site-year values reused from Sample_Special source.", ""],
                ["source_process_file", str(source), ""],
            ],
            columns=["check_item", "value", "note"],
        ).to_excel(writer, sheet_name="Site_Special_Note", index=False)
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
        qa.to_excel(writer, sheet_name="Site_Special_QA", index=False)


def worldpop_site_features(site_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = p_sample("Feature_Process_WorldPop.xlsx")
    siteyear = pd.read_excel(source, sheet_name="WorldPop_SiteYear_Features", engine=PY_ENGINE)
    if "site_id" in siteyear.columns:
        siteyear = siteyear.rename(columns={"site_id": "site"})
    pop_cols = [c for c in siteyear.columns if c.startswith("worldpop_")]
    site = siteyear.groupby(["water_body_type", "site"], as_index=False)[pop_cols].mean(numeric_only=True)
    coords = site_meta[["water_body_type", "site", "longitude", "latitude"]]
    site = coords.merge(site, on=["water_body_type", "site"], how="left")
    qa = pd.DataFrame(
        [
            ["source_process", str(source), "sample-special WorldPop process workbook"],
            ["source_sheet", "WorldPop_SiteYear_Features", "unique site x sampling_year rows"],
            ["site_rows", site.shape[0], "unique sites"],
            ["worldpop_feature_columns", len(pop_cols), "10 buffers"],
            ["aggregation_rule", "mean across available site-year WorldPop values", "avoids duplicating high/low periods within the same year"],
            ["missing_cells", int(site[pop_cols].isna().sum().sum()), "should be 0"],
        ],
        columns=["check_item", "value", "note"],
    )
    db_path = SITE_DIR / "Feature_Database_WorldPop_Site_Special.xlsx"
    process_path = SITE_DIR / "Feature_Process_WorldPop_Site_Special.xlsx"
    with pd.ExcelWriter(db_path, engine=PY_ENGINE) as writer:
        site.to_excel(writer, sheet_name="WorldPop_Site_Feature_Data", index=False)
        qa.to_excel(writer, sheet_name="QA_Checks", index=False)
    with pd.ExcelWriter(process_path, engine=PY_ENGINE) as writer:
        siteyear.to_excel(writer, sheet_name="WorldPop_SiteYear_Features", index=False)
        site.to_excel(writer, sheet_name="WorldPop_SiteMean_Features", index=False)
        qa.to_excel(writer, sheet_name="Process_QA", index=False)
    return site, qa


def build_ml_input(
    site_meta: pd.DataFrame,
    aquatic: pd.DataFrame,
    beta: pd.DataFrame,
    api: pd.DataFrame,
    landuse: pd.DataFrame,
    urbanization: pd.DataFrame,
    worldpop: pd.DataFrame,
    module_qas: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    matrix = site_meta.copy()
    module_order = [
        ("aquatic", aquatic),
        ("api", api),
        ("beta", beta),
        ("worldpop", worldpop),
        ("landuse", landuse),
        ("urbanization", urbanization),
    ]
    unmatched_records = []
    for module, df in module_order:
        drop_cols = [c for c in ["longitude", "latitude"] if c in df.columns]
        df2 = df.drop(columns=drop_cols, errors="ignore")
        before = len(matrix)
        matrix = matrix.merge(df2, on=["water_body_type", "site"], how="left", suffixes=("", f"_{module}"))
        unmatched = matrix.loc[matrix.filter(regex=f"_{module}$").isna().all(axis=1)] if any(c.endswith(f"_{module}") for c in matrix.columns) else pd.DataFrame()
        unmatched_records.append({"module": module, "rows_before": before, "rows_after": len(matrix), "unmatched_rows": len(unmatched)})

    feature_roles = []
    for c in matrix.columns:
        if c in SITE_INFO_COLUMNS:
            role = "site_info"
        elif c.startswith("beta_"):
            role = "beta_feature_lcbd" if "lcbd" in c.lower() else "beta_feature_mean_dissimilarity"
        elif c.startswith("API_") or re.match(r"^P[1-8]_z_", c):
            role = "api_feature"
        elif c.startswith("worldpop_"):
            role = "worldpop_population_density_feature"
        elif c.startswith("landuse_"):
            role = "landuse_worldcover_feature"
        elif c.startswith("urbanization_"):
            role = "urbanization_osm_feature"
        else:
            role = "aquatic_biology_site_mean_feature"
        feature_roles.append({"column_name": c, "role": role})
    feature_dictionary = pd.DataFrame(feature_roles)

    qa_rows = [
        ["matrix_scope", "Site_Special", "one row per unique sampling site"],
        ["site_rows", matrix.shape[0], "expected 130"],
        ["total_columns", matrix.shape[1], "site info + feature variables"],
        ["duplicate_site_special_id", int(matrix["site_special_id"].duplicated().sum()), "should be 0"],
        ["feature_columns", matrix.shape[1] - len(SITE_INFO_COLUMNS), "model candidate features"],
        ["created_date", TODAY, ""],
    ]
    for module, qa in module_qas.items():
        for _, row in qa.iterrows():
            qa_rows.append([f"{module}_{row['check_item']}", row["value"], row["note"]])
    merge_qa = pd.DataFrame(qa_rows, columns=["check_item", "value", "note"])
    unmatched = pd.DataFrame(unmatched_records)

    out = SITE_DIR / "ML_Input_Site_Special_Feature_Matrix.xlsx"
    with pd.ExcelWriter(out, engine=PY_ENGINE) as writer:
        matrix.to_excel(writer, sheet_name="Model_Input_Matrix", index=False)
        feature_dictionary.to_excel(writer, sheet_name="Feature_Dictionary", index=False)
        merge_qa.to_excel(writer, sheet_name="Merge_QA", index=False)
        unmatched.to_excel(writer, sheet_name="Merge_Module_QA", index=False)
    return matrix


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        tag = qn(f"w:{edge}")
        node = tc_mar.find(tag)
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    if sum(widths_dxa) != 9360:
        raise ValueError(f"Table widths must total 9360 DXA, got {sum(widths_dxa)}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    for tag_name, attrs in (
        ("w:tblW", {"w:w": "9360", "w:type": "dxa"}),
        ("w:tblInd", {"w:w": "120", "w:type": "dxa"}),
        ("w:tblLayout", {"w:type": "fixed"}),
    ):
        tag = qn(tag_name)
        node = tbl_pr.find(tag)
        if node is None:
            node = OxmlElement(tag_name)
            tbl_pr.append(node)
        for key, value in attrs.items():
            node.set(qn(key), value)

    tbl_grid = table._tbl.tblGrid
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in widths_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        tbl_grid.append(grid_col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[index]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell.width = Inches(width / 1440)
            set_cell_margins(cell)


def set_run_font(run, western: str = "Calibri", east_asia: str = "Microsoft YaHei") -> None:
    run.font.name = western
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    r_fonts.set(qn("w:ascii"), western)
    r_fonts.set(qn("w:hAnsi"), western)
    r_fonts.set(qn("w:eastAsia"), east_asia)


def format_table_cell(cell, font_size: float, bold: bool = False) -> None:
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.05
        for run in paragraph.runs:
            set_run_font(run)
            run.font.size = Pt(font_size)
            run.bold = bold


def add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[object]],
    widths_dxa: list[int] | None = None,
    font_size: float = 9.0,
) -> None:
    widths_dxa = widths_dxa or [9360 // len(headers)] * len(headers)
    widths_dxa[-1] += 9360 - sum(widths_dxa)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = str(h)
        set_cell_shading(cell, "E8EEF5")
        format_table_cell(cell, font_size, bold=True)
    set_repeat_table_header(table.rows[0])
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = "" if pd.isna(value) else str(value)
            format_table_cell(cells[i], font_size)
    set_table_geometry(table, widths_dxa)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def add_equation(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="Equation")


def add_labeled_paragraph(doc: Document, label: str, text: str) -> None:
    paragraph = doc.add_paragraph()
    label_run = paragraph.add_run(label)
    label_run.bold = True
    set_run_font(label_run)
    text_run = paragraph.add_run(text)
    set_run_font(text_run)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    before = paragraph.add_run("第 ")
    set_run_font(before)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    field_begin_run = OxmlElement("w:r")
    field_begin_run.append(begin)
    field_instr_run = OxmlElement("w:r")
    field_instr_run.append(instr)
    field_end_run = OxmlElement("w:r")
    field_end_run.append(end)
    paragraph._p.append(field_begin_run)
    paragraph._p.append(field_instr_run)
    paragraph._p.append(field_end_run)
    after = paragraph.add_run(" 页")
    set_run_font(after)
    for run in paragraph.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor.from_string("6B7280")


def style_doc(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title_style = styles["Title"]
    title_style.font.name = "Calibri"
    title_style.font.size = Pt(20)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor.from_string("0B2545")
    title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_style.paragraph_format.space_before = Pt(0)
    title_style.paragraph_format.space_after = Pt(5)
    title_style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    title_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    title_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    subtitle_style = styles["Subtitle"]
    subtitle_style.font.name = "Calibri"
    subtitle_style.font.size = Pt(10.5)
    subtitle_style.font.italic = True
    subtitle_style.font.color.rgb = RGBColor.from_string("5E6B78")
    subtitle_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_style.paragraph_format.space_before = Pt(0)
    subtitle_style.paragraph_format.space_after = Pt(12)
    subtitle_style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    subtitle_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    subtitle_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for style_name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 18, 10),
        ("Heading 2", 13, "2E74B5", 14, 7),
        ("Heading 3", 12, "1F4D78", 10, 5),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    equation = styles.add_style("Equation", WD_STYLE_TYPE.PARAGRAPH)
    equation.base_style = normal
    equation.font.name = "Cambria Math"
    equation.font.size = Pt(10.5)
    equation.font.color.rgb = RGBColor.from_string("0B2545")
    equation.paragraph_format.left_indent = Inches(0.25)
    equation.paragraph_format.space_before = Pt(2)
    equation.paragraph_format.space_after = Pt(5)
    equation.paragraph_format.line_spacing = 1.0
    equation._element.rPr.rFonts.set(qn("w:ascii"), "Cambria Math")
    equation._element.rPr.rFonts.set(qn("w:hAnsi"), "Cambria Math")
    equation._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    caption = styles.add_style("Caption Small", WD_STYLE_TYPE.PARAGRAPH)
    caption.base_style = normal
    caption.font.name = "Calibri"
    caption.font.size = Pt(9)
    caption.font.bold = True
    caption.font.color.rgb = RGBColor.from_string("475569")
    caption.paragraph_format.space_before = Pt(6)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.keep_with_next = True
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    header = section.header.paragraphs[0]
    header.text = "Site Special | 机器学习特征构建与数据处理方法"
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header.paragraph_format.space_after = Pt(0)
    for run in header.runs:
        set_run_font(run)
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor.from_string("64748B")
    add_page_number(section.footer.paragraphs[0])


def _build_method_doc_legacy(matrix: pd.DataFrame, module_counts: dict[str, int]) -> None:
    doc = Document()
    style_doc(doc)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Site Special 机器学习特征构建与数据处理方法")
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor.from_string("0B2545")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("深圳水生态多源特征数据库 | 站点独立版本").italic = True

    doc.add_heading("文档说明", level=1)
    doc.add_paragraph(
        "本文件说明 Site_Special 版本机器学习输入矩阵的构建方法。该版本以采样点为独立样本单位，"
        "将 577 个点位 × 采样时间样本整合为 130 个点位样本，适用于以空间独立点位为有效样本量的建模与解释。"
    )
    add_table(
        doc,
        ["项目", "当前结果"],
        [
            ["矩阵文件", "ML_Input_Site_Special_Feature_Matrix.xlsx"],
            ["样本单位", "site / 采样点"],
            ["点位数", matrix.shape[0]],
            ["总列数", matrix.shape[1]],
            ["特征列数", matrix.shape[1] - len(SITE_INFO_COLUMNS)],
            ["生成日期", TODAY],
        ],
    )

    doc.add_heading("1 数据整合原则", level=1)
    doc.add_paragraph(
        "Sample_Special 版本保留每一次点位 × 采样时间作为独立样本。Site_Special 版本不重新合并多次采样的原始物种记录，"
        "因为同一站点不同年份或水期的水体状态、群落结构和采样频次并不一致。为避免把多时间群落误当成一次综合采样，"
        "本版本先按原 Sample_Special 方法计算每个样本的生物指标，再对同一 site 的样本级特征求均值。"
    )
    doc.add_paragraph(
        "空间静态特征如 API、WorldCover 和 OSM 城市化指标本来就是点位级特征，因此直接从 Sample_Special 过程或数据库中转换为站点宽表。"
        "WorldPop 为年度人口密度特征，本版本使用 site × year 过程表按站点求均值，避免同一年丰水期和枯水期重复加权。"
    )

    doc.add_heading("2 水生生物特征", level=1)
    doc.add_paragraph(
        "水生生物原始来源为 Aquatic_Biological_Database.xlsx。底栖动物、鱼类、水生植物、浮游植物、着生藻类和浮游动物按现有规则计算样本级 alpha 多样性、总密度和总生物量。"
    )
    add_table(
        doc,
        ["类群", "多样性输入", "总密度/总生物量", "Site_Special 汇总"],
        [
            ["Benthos", "Qualitative 出现记录", "Density / Biomass 求和", "同一 site 样本级特征均值"],
            ["Fish", "物种个体记录数", "个体数 / Weight 求和", "同一 site 样本级特征均值"],
            ["Macrophytes", "Density", "Density / Biomass 求和", "同一 site 样本级特征均值"],
            ["Phytoplankton", "Raw value", "Density / Biomass 求和", "同一 site 样本级特征均值"],
            ["Periphyton", "Raw value, River only", "Density / Biomass 求和", "Lake 保持不适用缺失"],
            ["Zooplankton", "Density", "Density / Biomass 求和", "同一 site 样本级特征均值"],
        ],
    )
    doc.add_paragraph(
        "多样性指标包括 richness、Shannon、Margalef、Pielou 和 Simpson；总密度和总生物量采用 log10(x+1) 后 robust scaling。"
        "Site_Special 数据库保留 Feature_Process_Aquatic_Biology_Site_Special.xlsx 作为追溯文件。"
    )

    doc.add_heading("3 Beta 多样性特征", level=1)
    doc.add_paragraph(
        "Beta 特征来自 Sample_Special 版本的 577 样本分解结果。每个样本已有 Mean dissimilarity 和 LCBD；Site_Special 版本直接对同一站点的样本级 beta 特征求均值。"
        "最终 ML input 使用与 Sample_Special 主矩阵一致的 input_feature_scaled_optional，即 Mean dissimilarity 保持原值，LCBD 使用 log10(LCBD+1e-12) 后 robust scaling 的列。"
    )

    doc.add_heading("4 空间与人口特征", level=1)
    add_table(
        doc,
        ["模块", "Site_Special 处理方式", "特征数"],
        [
            ["OSM API", "从 site × buffer 长表透视为站点宽表，保留 API_expert_weighted 与 P1_z-P8_z", module_counts["api"]],
            ["WorldCover", "直接使用点位级 10 尺度 × 4 类组合比例", module_counts["landuse"]],
            ["WorldPop", "按 site × year 过程表对采样年份求站点均值", module_counts["worldpop"]],
            ["OSM 城市化", "直接使用点位级不透水、建筑和自然用地比例", module_counts["urbanization"]],
        ],
    )
    doc.add_paragraph(
        "DEM 尚未包含在当前 Sample_Special 主矩阵中，因此本次 Site_Special 主矩阵未新增 DEM 列；若后续建立 DEM Feature_Database，可按同一站点级规则追加。"
    )

    doc.add_heading("5 输出文件", level=1)
    add_table(
        doc,
        ["文件", "用途"],
        [
            ["Feature_Process_Aquatic_Biology_Site_Special.xlsx", "水生生物样本级复算、站点均值和 QA"],
            ["Feature_Database_Aquatic_Biology_Site_Special.xlsx", "站点级水生生物输入特征"],
            ["Feature_Process_Beta_Site_Special.xlsx", "Beta 样本级长表与站点均值追溯"],
            ["Feature_Database_Beta_Site_Special.xlsx", "站点级 Beta 输入特征"],
            ["Feature_Process/API/LandUse/WorldPop/Urbanization_*_Site_Special.xlsx", "空间与人口特征追溯过程"],
            ["ML_Input_Site_Special_Feature_Matrix.xlsx", "最终 130 站点 ML input 主矩阵"],
        ],
    )

    doc.add_heading("6 建模使用建议", level=1)
    doc.add_paragraph(
        "Site_Special 版本将有效样本量定义为 130 个空间独立点位，适合分析静态空间压力梯度与多年平均生物状态之间的关系。"
        "如果模型目标关注年际或丰枯水期动态，应优先使用 Sample_Special 版本并采用按 site 分组的交叉验证；如果目标关注点位尺度空间格局，Site_Special 版本更直接。"
    )
    path = SITE_DIR / "机器学习特征构建与数据处理方法综合说明_Site_Special.docx"
    doc.save(path)


def build_method_doc(matrix: pd.DataFrame, module_counts: dict[str, int]) -> None:
    """Build the detailed, reproducible Site Special methods manual."""
    aquatic_process = SITE_DIR / "Feature_Process_Aquatic_Biology_Site_Special.xlsx"
    beta_process = SITE_DIR / "Feature_Process_Beta_Site_Special.xlsx"
    api_process = SITE_DIR / "Feature_Process_API_Site_Special.xlsx"
    worldcover_process = SITE_DIR / "Feature_Process_LandUse_WorldCover_Site_Special.xlsx"
    worldpop_process = SITE_DIR / "Feature_Process_WorldPop_Site_Special.xlsx"
    urban_process = SITE_DIR / "Feature_Process_Urbanization_OSM_Site_Special.xlsx"
    matrix_path = SITE_DIR / "ML_Input_Site_Special_Feature_Matrix.xlsx"

    aquatic_params = pd.read_excel(aquatic_process, sheet_name="Standardization_Params", engine=PY_ENGINE)
    beta_site_long = pd.read_excel(beta_process, sheet_name="Beta_Features_Long_SiteMean", engine=PY_ENGINE)
    api_groups = pd.read_excel(api_process, sheet_name="Group_Definitions", engine=PY_ENGINE)
    api_weights = pd.read_excel(api_process, sheet_name="Weight_Schemes", engine=PY_ENGINE)
    api_sensitivity = pd.read_excel(api_process, sheet_name="Sensitivity_Correlations", engine=PY_ENGINE)
    worldcover_rules = pd.read_excel(worldcover_process, sheet_name="Combination_Rules", engine=PY_ENGINE)
    worldpop_siteyear = pd.read_excel(worldpop_process, sheet_name="WorldPop_SiteYear_Features", engine=PY_ENGINE)
    urban_fclass = pd.read_excel(urban_process, sheet_name="Fclass_Design", engine=PY_ENGINE)
    road_widths = pd.read_excel(urban_process, sheet_name="Road_Width_Assumptions", engine=PY_ENGINE)
    urban_long = pd.read_excel(urban_process, sheet_name="Urbanization_Process_Long", engine=PY_ENGINE)
    feature_dictionary = pd.read_excel(matrix_path, sheet_name="Feature_Dictionary", engine=PY_ENGINE)

    doc = Document()
    style_doc(doc)
    doc.core_properties.title = "Site Special 机器学习特征构建与数据处理方法"
    doc.core_properties.subject = "130 个空间独立采样点的特征矩阵构建与质量控制"
    doc.core_properties.keywords = "Site Special; aquatic biology; beta diversity; OSM; WorldCover; WorldPop"

    title = doc.add_paragraph(style="Title")
    title.add_run("Site Special 机器学习特征构建与数据处理方法")

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("深圳水生态多源特征数据库 | 130 个空间独立采样点版本")

    doc.add_heading("文档说明", level=1)
    doc.add_paragraph(
        "本说明以当前实际生成的 Site_Special 过程文件、特征数据库和 ML Input 主矩阵为唯一版本基准，"
        "用于数据流程追溯、SCI 手稿方法撰写和后续复算。Site Special 的独立样本单位是采样点，而不是"
        "“点位 × 采样时间”。它将 Sample Special 中的 577 个时间样本整合为 130 个空间点位，但不把"
        "同一站点多次采样的原始物种记录合并成一个虚拟群落。"
    )
    doc.add_paragraph("表 1  Site Special 主矩阵版本基准", style="Caption Small")
    add_table(
        doc,
        ["项目", "实际结果"],
        [
            ["主矩阵", "ML_Input_Site_Special_Feature_Matrix.xlsx / Model_Input_Matrix"],
            ["来源矩阵", "577 个 Sample Special 点位 × 时间样本"],
            ["Site Special 样本", f"{matrix.shape[0]} 个点位：River 86，Lake 44"],
            ["矩阵规模", f"{matrix.shape[0]} 行 × {matrix.shape[1]} 列；8 个站点信息列 + {matrix.shape[1] - 8} 个特征列"],
            ["空间尺度", "50、100、200、500、1000、2000、3000、4000、5000、6000 m"],
            ["版本日期", TODAY],
        ],
        [2200, 7160],
    )
    add_labeled_paragraph(
        doc,
        "与 Sample Special 的关系：",
        "Sample Special 文档负责说明 577 个独立时间样本的原始特征计算；本文件在重述必要原始算法的同时，"
        "重点说明这些结果如何被转换为 130 个站点独立样本。两套矩阵同时保留，互不替代。",
    )

    doc.add_heading("1 站点独立矩阵的定义与整合层级", level=1)
    doc.add_heading("1.1 两级样本标识", level=2)
    doc.add_paragraph(
        "Sample Special 的样本键由水体类型、点位、采样年份和丰枯水期共同确定；同一 site 在不同年份或"
        "不同水期属于不同样本。Site Special 仅以 water_body_type + site 确定一行，避免 River 与 Lake 中"
        "可能出现的同名点位发生错误连接。"
    )
    add_equation(doc, "sample_id = water_body_type + site + sampling_year + flow_period")
    add_equation(doc, "site_special_id = water_body_type + '_' + site")
    doc.add_paragraph(
        "最终站点信息列包括 site_special_id、water_body_type、site、longitude、latitude、n_sample_records、"
        "sampling_years 和 flow_periods。经纬度取 Sample Special 中该点位的第一条一致记录；"
        "n_sample_records 为该点位唯一 sample_id 数；年份和水期以去重后分号连接的方式保存。"
    )

    doc.add_heading("1.2 为什么不直接合并原始物种记录", level=2)
    doc.add_paragraph(
        "同一点位不同年份或水期的水文条件、群落结构和采样频次不同。若把多次采样的物种记录直接拼接后"
        "重新计算 Richness、Shannon 或 β 多样性，会把“重复观测”误作“一次更大采样”，并使采样次数多的"
        "站点得到更高的检出概率。因此本流程采用两级计算：先严格按 577 个独立样本计算生物学指标和必要变换，"
        "再对同一站点的样本级特征求算术平均。"
    )
    add_equation(doc, "x̄(s,k) = [1 / n(s,k)] Σₜ x(s,t,k)")
    doc.add_paragraph(
        "其中 x(s,t,k) 为站点 s 在采样时间 t 的第 k 个样本级特征，n(s,k) 为该特征的非缺失采样次数。"
        "每次有效采样等权；pandas 分组均值自动忽略 NA。最终每个站点只占模型中的一行，所以站点之间等权，"
        "但采样次数较多的站点均值通常估计更稳定。"
    )

    doc.add_heading("1.3 实际采样频次", level=2)
    sample_frequency = matrix["n_sample_records"].value_counts().sort_index()
    sample_frequency_rows = [[int(n), int(count)] for n, count in sample_frequency.items()]
    doc.add_paragraph("表 2  130 个站点的时间样本数分布", style="Caption Small")
    add_table(doc, ["每站点样本数", "站点数"], sample_frequency_rows, [3000, 6360])
    doc.add_paragraph(
        f"每站点样本数范围为 {int(matrix['n_sample_records'].min())}-{int(matrix['n_sample_records'].max())}，"
        f"中位数为 {matrix['n_sample_records'].median():.0f}，均值为 {matrix['n_sample_records'].mean():.3f}。"
        "124 个站点同时覆盖丰水期和枯水期，5 个站点仅有丰水期，1 个站点仅有枯水期。"
    )

    doc.add_heading("1.4 统一空间尺度与 River/Lake 范围", level=2)
    doc.add_paragraph(
        "API、WorldCover、WorldPop 和 OSM 城市化均使用 10 个累计空间尺度。River 以采样点为圆心构建"
        "累计圆形缓冲区；Lake 先按 site 匹配完整湖泊多边形，再取湖泊边界向外扩张的陆域环带。"
    )
    add_equation(doc, "B = {50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000} m")
    add_equation(doc, "Ω(s,b) = Buffer(Point(s), b)  [River]")
    add_equation(doc, "Ω(s,b) = Buffer(LakePolygon(s), b) \\ LakePolygon(s)  [Lake]")
    doc.add_paragraph(
        "上述范围是累计尺度，不是相邻环带。不同模块对水体分母的处理并不完全相同：API 和 OSM 城市化"
        "进一步扣除 OSM water_a 水面；WorldCover 的最终组合比例保留水体像元并使用全部有效像元作分母；"
        "WorldPop 在相同几何范围内统计年度人口密度。"
    )

    doc.add_heading("2 水生生物特征：样本级复算后求站点均值", level=1)
    doc.add_heading("2.1 原始来源、样本全集和分类单元键", level=2)
    doc.add_paragraph(
        "原始来源为 data/processed/Summary/Aquatic_Biological_Database.xlsx。读取 Benthos、"
        "Benthos_Qualitative、Fish、Macrophytes、Phytoplankton、Periphyton 和 Zooplankton 工作表；"
        "Water_Type_QA 不参与特征计算。所有工作表的 Water body type、Site 和 Sampling time 被标准化为"
        "water_body_type、site、sampling_year 和 flow_period，再组合为 sample_id。原始拼写 hjgh-flow period"
        "按 high-flow 处理，避免被字符串中的 low 误判为枯水期。去重后样本全集为 577 个，其中 River 409、Lake 168。"
    )
    doc.add_paragraph(
        "分类单元优先使用 Scientific name；若为空，则回退到 Species。空分类单元被剔除。用于跨类群整体指标时，"
        "分类单元键写成 taxon_group::taxon_key，以防不同生物类群出现同名标签后被错误合并。"
    )

    doc.add_heading("2.2 六类群的实际丰度与总量口径", level=2)
    doc.add_paragraph("表 3  样本级水生生物输入口径", style="Caption Small")
    add_table(
        doc,
        ["类群", "多样性丰度 nᵢ", "总密度", "总生物量", "适用范围"],
        [
            ["底栖动物", "Benthos_Qualitative；同一样本-分类单元去重后记 1", "Benthos Density 求和", "Benthos Biomass 求和", "River + Lake"],
            ["鱼类", "Fish 中同一样本-分类单元的个体记录数", "样本内 Fish 记录条数", "Weight 求和", "River + Lake"],
            ["水生植物", "Density 按分类单元求和", "Density 求和", "Biomass 求和", "River + Lake"],
            ["浮游植物", "Raw value 按分类单元求和", "Density 求和", "Biomass 求和", "River + Lake"],
            ["着生藻类", "Raw value 按分类单元求和", "Density 求和", "Biomass 求和", "River only"],
            ["浮游动物", "Density 按分类单元求和", "Density 求和", "Biomass 求和", "River + Lake"],
        ],
        [1200, 3000, 1700, 1700, 1760],
        font_size=8.6,
    )
    doc.add_paragraph(
        "数值列先转换为 numeric，无法解析或缺失的数值按 0 参与样本内求和；多样性矩阵仅保留丰度大于 0 的"
        "分类单元。对适用水体中没有检出记录的类群，五个多样性指标和总量按 0 处理。着生藻类不适用于 Lake，"
        "相应值保持 NA，不用 0 代替。"
    )

    doc.add_heading("2.3 α 多样性与整体指标", level=2)
    doc.add_paragraph(
        "对每个独立样本和生物类群，设正丰度分类单元数为 S，第 i 个分类单元丰度为 nᵢ，总丰度为 N，"
        "相对丰度 pᵢ = nᵢ/N。仅 nᵢ > 0 的分类单元进入计算。"
    )
    add_equation(doc, "Richness = S")
    add_equation(doc, "Shannon H′ = -Σᵢ pᵢ ln(pᵢ)")
    add_equation(doc, "Margalef d = (S - 1) / ln(N),  当 N > 1 且 S > 1；否则为 0")
    add_equation(doc, "Pielou J′ = H′ / ln(S),  当 S > 1；否则为 0")
    add_equation(doc, "Simpson = 1 - Σᵢ pᵢ²")
    doc.add_paragraph(
        "每个类群形成 5 个指标，共 30 列。随后在同一样本内合并所有可用的 group::taxon 及其丰度，计算"
        "overall_richness、overall_shannon、overall_margalef、overall_pielou 和 overall_simpson；"
        "total_richness 与 overall_richness 在当前实现中取同一个合并后 Richness 值。由于 group 前缀保证"
        "不同类群不重名，该 Richness 等价于各类群 Richness 之和。其他 overall 指数混合了不同类群的原始丰度"
        "量纲，保留用于与 Sample Special 一致的探索性输入，解释时应比单类群指标更谨慎。"
    )

    doc.add_heading("2.4 总密度和总生物量的变换与参数", level=2)
    doc.add_paragraph(
        "六类群的总密度和总生物量共 12 个变量。在 577 个样本层面先做 log10(x+1)，再按该变量适用样本的"
        "中位数和四分位距进行 robust scaling；着生藻类参数只用 409 个 River 样本拟合，其余变量使用全部 577 个样本。"
    )
    add_equation(doc, "x_log = log₁₀(x + 1)")
    add_equation(doc, "x_robust = [x_log - median(x_log)] / [Q₃(x_log) - Q₁(x_log)]")
    doc.add_paragraph(
        "若 IQR 为 0，程序依次回退到标准差和 1；本次 12 个变量均未触发回退。适用范围内缺失值在变换前按 0 处理，"
        "不适用值保留 NA。多样性指标不在此阶段缩放。下表参数直接来自 Standardization_Params。"
    )
    parameter_rows = []
    for _, row in aquatic_params.iterrows():
        parameter_rows.append(
            [
                row["source_column"],
                "River" if row["applicable_scope"] == "River only" else "River + Lake",
                f"{row['median_log10p1']:.6f}",
                f"{row['q1_log10p1']:.6f}",
                f"{row['q3_log10p1']:.6f}",
                f"{row['iqr_used']:.6f}",
            ]
        )
    doc.add_paragraph("表 4  12 个总量变量的实际 robust scaling 参数", style="Caption Small")
    add_table(
        doc,
        ["原始总量变量", "参数范围", "Median", "Q1", "Q3", "IQR"],
        parameter_rows,
        [2900, 1200, 1300, 1300, 1300, 1360],
        font_size=8.2,
    )

    doc.add_heading("2.5 从 577 样本到 130 站点", level=2)
    doc.add_paragraph(
        "样本级 48 列由 30 个类群多样性指标、6 个 overall/total 指标和 12 个已变换总量指标组成。"
        "Site_Mean_Features 对每一列分别按 water_body_type + site 求非缺失算术均值。这里平均的是已经在样本层面"
        "计算并标准化后的特征，不是先把多时间原始 Density、Biomass 或物种表合并再计算。"
    )
    add_equation(doc, "AquaticSiteFeature(s,k) = mean_t[SampleFeature(s,t,k)]")
    doc.add_paragraph(
        "River 和 Lake 的 5 个通用类群及 overall 指标无缺失；Lake 的 44 个站点在 5 个着生藻类多样性列和"
        "2 个着生藻类总量变换列上均为 NA，共 44 × 7 = 308 个结构性缺失单元格。该缺失表示“不适用”，"
        "不能在建模前简单填 0。"
    )
    add_labeled_paragraph(
        doc,
        "追溯位置：",
        "Feature_Process_Aquatic_Biology_Site_Special.xlsx 依次保存原始工作表 QA、样本级分类单元丰度长表、"
        "类群多样性、类群总量、整体多样性、577 行复算宽表、站点采样摘要、130 行站点均值、标准化参数和过程 QA。",
    )

    doc.add_heading("3 β 多样性：沿用样本级分解并按站点均值汇总", level=1)
    doc.add_heading("3.1 Sample Special 中的距离矩阵", level=2)
    doc.add_paragraph(
        "β 特征不在 Site Special 中重新构建“站点 × 物种”矩阵，而是读取 Sample Special 的"
        "Feature_Database_Beta_Sample_Special.xlsx。各类群先建立 577 个时间样本中的样本 × 分类单元矩阵；"
        "丰度矩阵用于 Bray-Curtis，有无矩阵用于 Sørensen 及 turnover/nestedness 分解。"
    )
    add_equation(doc, "Bray-Curtis(i,j) = Σₖ |xᵢₖ - xⱼₖ| / Σₖ (xᵢₖ + xⱼₖ)")
    add_equation(doc, "β_sor = (b + c) / (2a + b + c)")
    add_equation(doc, "β_sim = min(b,c) / [a + min(b,c)]")
    add_equation(doc, "β_sne = β_sor - β_sim")
    doc.add_paragraph(
        "a 为两个样本共有分类单元数，b 和 c 为各自独有分类单元数。β_sim 表示 turnover，β_sne 表示 nestedness。"
        "底栖动物、鱼类、水生植物、浮游植物和浮游动物分别在 all、river、lake 三个范围计算；着生藻类只在 river"
        "范围计算。因此共有 5 × 3 + 1 = 16 个“类群 × 范围”组合，每个组合计算 4 个距离，共 64 个距离矩阵。"
    )

    doc.add_heading("3.2 Mean dissimilarity 与 LCBD", level=2)
    doc.add_paragraph("对每个距离矩阵，样本 i 相对同矩阵内其他样本的平均相异性为：")
    add_equation(doc, "MeanD(i) = [1 / (n - 1)] Σⱼ≠ᵢ d(i,j)")
    doc.add_paragraph("LCBD 由距离矩阵逐元素平方后的双中心化矩阵得到：")
    add_equation(doc, "H = Iₙ - (1/n)11ᵀ")
    add_equation(doc, "G = -(1/2) H D² H")
    add_equation(doc, "LCBD(i) = gᵢᵢ / Σᵢ gᵢᵢ")
    doc.add_paragraph(
        "每个矩阵内 LCBD 总和为 1。Sample Special 同时保留原始 LCBD 和可选变换版；当前 Site Special 主矩阵"
        "读取 input_feature_scaled_optional：Mean dissimilarity 保持原值，LCBD 在样本层面先做"
        "log10(LCBD + 10^-12)，再按对应 LCBD 特征的中位数和 IQR robust scaling。"
    )
    add_equation(doc, "LCBD* = [log₁₀(LCBD + 10⁻¹²) - median] / IQR")

    doc.add_heading("3.3 站点均值、覆盖范围与缺失", level=2)
    doc.add_paragraph(
        "Site Special 对 64 个 MeanD 列和 64 个 LCBD* 列分别按站点求非缺失均值。也就是说，最终 LCBD 特征是"
        "“样本级 LCBD 变换值的均值”，不是“原始 LCBD 站点均值再变换”。数据库中的 input_feature_raw_lcbd_mean"
        "额外保存 64 个原始 LCBD 的站点均值，便于敏感性分析，但不进入当前主矩阵。"
    )
    beta_coverage = (
        beta_site_long.groupby(["taxon_group", "beta_scope"], as_index=False)["site"]
        .nunique()
        .rename(columns={"site": "n_sites"})
        .sort_values(["taxon_group", "beta_scope"])
    )
    group_labels = {
        "benthos": "底栖动物",
        "fish": "鱼类",
        "macrophytes": "水生植物",
        "phytoplankton": "浮游植物",
        "periphyton": "着生藻类",
        "zooplankton": "浮游动物",
    }
    scope_labels = {"all": "All", "river": "River", "lake": "Lake"}
    beta_rows = [
        [group_labels[row["taxon_group"]], scope_labels[row["beta_scope"]], "Bray-Curtis; Sørensen; turnover; nestedness", int(row["n_sites"])]
        for _, row in beta_coverage.iterrows()
    ]
    doc.add_paragraph("表 5  β 特征的类群、比较范围与实际站点覆盖", style="Caption Small")
    add_table(doc, ["类群", "β 范围", "距离指标", "有值站点数"], beta_rows, [1500, 1100, 4800, 1960], font_size=8.5)
    doc.add_paragraph(
        "β 特征的缺失来自比较范围和原始类群覆盖，而不是合并失败。例如 lake-scope 列只对 Lake 有意义，"
        "river-scope 列只对 River 有意义，某些类群还存在未形成有效矩阵的样本。最终 β 模块共有 6360 个 NA："
        "MeanD 3180 个、LCBD* 3180 个。均值只使用该站点在该 β 特征上已有的样本记录；若整站均无适用值则保持 NA。"
    )

    doc.add_heading("4 OSM POI 人类活动综合压力指数（API）", level=1)
    doc.add_heading("4.1 原始计数、面积与统一标准化", level=2)
    doc.add_paragraph(
        "API 读取 OSM 点状和面状 POI，并按生态压力路径映射到 P1-P8。对每个站点和尺度，先用范围面积扣除"
        "OSM water_a 水面并集，得到分析陆域；点要素统计密度，面要素统计其裁剪面积占分析陆域的比例。"
    )
    add_equation(doc, "A_land(s,b) = A_range(s,b) - A_water(s,b)")
    add_equation(doc, "D(i,s,b) = N_point(i,s,b) / [A_land(s,b) / 10⁶]  (个/km²)")
    add_equation(doc, "R(i,s,b) = A_polygon(i,s,b) / A_land(s,b)")
    doc.add_paragraph(
        "在每个 buffer_m 内，以全部 130 个站点为总体分别对 D 和 R 做 z-score；River 与 Lake 不分开拟合。"
        "同一压力组的点密度 z 值和面比例 z 值等权合成 P_score，再在同一尺度内标准化为 P_z。"
    )
    add_equation(doc, "z_b(x_s) = [x_s - mean_b(x)] / sd_b(x)")
    add_equation(doc, "P_score(i,s,b) = 0.5 z_b[D(i,s,b)] + 0.5 z_b[R(i,s,b)]")
    add_equation(doc, "P_z(i,s,b) = z_b[P_score(i,s,b)]")

    doc.add_heading("4.2 P1-P8 专家权重", level=2)
    expert = (
        api_weights[api_weights["scheme"].eq("expert")][["pressure_group", "raw_weight", "weight"]]
        .drop_duplicates("pressure_group")
    )
    api_group_table = api_groups.merge(expert, on="pressure_group", how="left")
    pressure_path_zh = {
        "P1": "污染物输入、污水/废物处置及市政环境设施",
        "P2": "高客流、餐饮生活废物、噪声和夜间活动",
        "P3": "持续居民服务活动和消费服务渗透",
        "P4": "游憩、踩踏、景观利用及岸线/生境扰动",
        "P5": "稳定设施占地、公共服务使用及地表硬化",
        "P6": "访客流入、短期住宿和交通可达设施",
        "P7": "低频参访及文化遗产展示活动",
        "P8": "微型使用痕迹、公共便利和低单体压力设施",
    }
    api_rows = []
    for _, row in api_group_table.sort_values("pressure_group").iterrows():
        api_rows.append(
            [
                row["pressure_group"],
                row["group_name_zh"],
                f"{row['raw_weight']:.2f}",
                f"{row['weight']:.6f}",
                pressure_path_zh[row["pressure_group"]],
            ]
        )
    doc.add_paragraph("表 6  API 压力组和实际专家权重", style="Caption Small")
    add_table(doc, ["组", "压力名称", "原始权重", "归一化权重", "主要生态路径"], api_rows, [550, 2550, 950, 1200, 4110], font_size=8.3)
    add_equation(doc, "w(i) = a(i) / Σᵢ a(i)")
    add_equation(doc, "API_expert(s,b) = Σᵢ w(i) P_z(i,s,b)")
    expert_equal = api_sensitivity[
        api_sensitivity["api_a"].eq("API_expert_weighted")
        & api_sensitivity["api_b"].eq("API_equal_weighted")
    ]["correlation"]
    expert_entropy = api_sensitivity[
        api_sensitivity["api_a"].eq("API_expert_weighted")
        & api_sensitivity["api_b"].eq("API_entropy_weighted")
    ]["correlation"]
    doc.add_paragraph(
        f"权重敏感性中，10 个尺度的专家 API 与等权 API 的 Pearson r 为 {expert_equal.min():.3f}-{expert_equal.max():.3f}，"
        f"与熵权 API 的 r 为 {expert_entropy.min():.3f}-{expert_entropy.max():.3f}。最终主矩阵不同时保留三套综合指数，"
        "而是保留每尺度 API_expert_weighted 及 P1_z-P8_z，共 10 × 9 = 90 列。"
    )

    doc.add_heading("4.3 Site Special 的处理", level=2)
    doc.add_paragraph(
        "API 原始输出本来就是 130 个站点 × 10 个尺度的静态空间结果。Site Special 从 API_Features_Long 按"
        "water_body_type、site_id、经纬度和 buffer_m 透视为宽表，aggfunc=first；不对 577 个时间样本求均值，"
        "也不重复计算 POI。特征名为 API_expert_weighted_{buffer}m 和 P1_z_{buffer}m 至 P8_z_{buffer}m。"
    )

    doc.add_heading("5 ESA WorldCover 土地利用组合特征", level=1)
    doc.add_heading("5.1 栅格、几何与分母", level=2)
    doc.add_paragraph(
        "土地利用来源为 ESA WorldCover 2021 分类栅格，项目文件的空间分辨率约 10 m。River 和 Lake 使用第 1.4 节"
        "的统一几何范围；几何转换到栅格 CRS 后提取像元，并设置 all_touched=True，使与范围边界相交的像元也进入"
        "统计，尤其避免 50 m 小尺度因像元中心规则而漏取。"
    )
    doc.add_paragraph(
        "过程表中的详细非水类别比例使用非水像元分母，并把 WorldCover code 80 作为水体排除；但进入 ML 的四个"
        "组合特征包含 water_wetland_pct，因此统一使用范围内全部有效像元作为分母。"
    )
    add_equation(doc, "Pct(C,s,b) = N_pixels(C,s,b) / N_total(s,b) × 100%")

    doc.add_heading("5.2 四类组合与 Site Special 处理", level=2)
    wc_label = {
        "urban_pct": "urban_pct",
        "agriculture_pct": "agriculture_pct",
        "green_pct": "green_pct",
        "water_wetland_pct": "water_wetland_pct",
    }
    worldcover_rows = [
        [wc_label[row["combined_feature"]], row["worldcover_components"], row["calculation"]]
        for _, row in worldcover_rules.iterrows()
    ]
    doc.add_paragraph("表 7  WorldCover 最终组合规则", style="Caption Small")
    add_table(doc, ["最终特征", "WorldCover 组成", "计算"], worldcover_rows, [1800, 3800, 3760], font_size=8.5)
    doc.add_paragraph(
        "bare/sparse vegetation、snow/ice 和 moss/lichen 未进入四类组合，所以四类之和允许低于 100%；当前站点-尺度"
        "组合和范围为 70.796460%-100%。Site Special 直接复制 130 个点位级宽表，不做时间平均，形成"
        "10 个尺度 × 4 类 = 40 个无缺失特征，命名为 landuse_{buffer}m_{class}_pct。"
    )

    doc.add_heading("6 WorldPop 年度人口密度特征", level=1)
    doc.add_heading("6.1 Sample Special 的年度匹配", level=2)
    doc.add_paragraph(
        "WorldPop 使用 2021-2025 年约 100 m 人口密度栅格，按 sample 的 sampling_year 匹配同年栅格；同一站点"
        "同一年丰水期和枯水期使用同一组人口密度。River 使用点中心缓冲区，Lake 使用湖泊边界外扩环带；"
        "all_touched=True。范围内有效像元求平均，NoData 在当前流程中按 0 处理。"
    )
    add_equation(doc, "PopMean(s,y,b) = [1 / n_valid] Σₚ PopulationRaster(y,p)")

    doc.add_heading("6.2 先去重水期，再按年份等权平均", level=2)
    doc.add_paragraph(
        "Site Special 不直接对 577 行人口密度求均值，因为同一 site-year 的丰水期和枯水期会重复同一个年度栅格值。"
        "流程读取 WorldPop_SiteYear_Features 的唯一 site × sampling_year 记录，再对各站点已有年份等权求均值。"
    )
    add_equation(doc, "WorldPopSite(s,b) = [1 / m(s)] Σᵧ PopMean(s,y,b)")
    years_per_site = worldpop_siteyear.groupby(["water_body_type", "site"])["sampling_year"].nunique()
    year_distribution = years_per_site.value_counts().sort_index()
    doc.add_paragraph("表 8  WorldPop 每站点可用年份数", style="Caption Small")
    add_table(
        doc,
        ["可用年份数", "站点数"],
        [[int(years), int(count)] for years, count in year_distribution.items()],
        [3000, 6360],
    )
    year_counts = worldpop_siteyear["sampling_year"].value_counts().sort_index()
    year_count_text = "；".join(f"{int(year)} 年 {int(count)} 条" for year, count in year_counts.items())
    doc.add_paragraph(
        f"实际过程表共 316 个 site-year：{year_count_text}。最终输出 10 个站点多年平均人口密度特征，"
        "130 个站点均无缺失。该值表示采样年份覆盖期的平均人口背景，不等同于固定 2021 年或最新年份的人口密度。"
    )

    doc.add_heading("7 OSM 城市化面积指标", level=1)
    doc.add_heading("7.1 水体扣除和三个百分比", level=2)
    doc.add_paragraph(
        "OSM 城市化模块在 EPSG:32649 米制投影中计算。对每个 Ω(s,b)，先裁剪并精确合并 water_a 面，从范围中"
        "扣除水体得到分析陆域 L(s,b)。建筑、不透水和自然用地三个比例共用该陆地面积分母。"
    )
    add_equation(doc, "W(s,b) = Union({WaterPolygonⱼ ∩ Ω(s,b)})")
    add_equation(doc, "L(s,b) = Ω(s,b) \\ W(s,b)")
    add_equation(doc, "BuildingPct = Area[Union(Buildings ∩ L)] / Area(L) × 100%")
    add_equation(doc, "ImperviousPct = Area[Union(Buildings, RoadBuffers, Traffic, Transport) ∩ L] / Area(L) × 100%")
    add_equation(doc, "NaturalLandPct = Area[Union(Forest, Grass, Heath, Meadow, Scrub, Beach, Cliff) ∩ L] / Area(L) × 100%")
    doc.add_paragraph(
        "所有组成几何在求面积前做 union，重叠部分只计一次。NaturalLandPct 不是 100% - ImperviousPct；OSM 中仍有"
        "未分类陆地，不同源图层也可能局部重叠，因此两者不要求互补。"
    )

    doc.add_heading("7.2 OSM 图层与 fclass", level=2)
    fclass_rows = []
    for _, row in urban_fclass.iterrows():
        fclass_rows.append([row["component"], row["source_layer"], row["included_fclass"], row["role"]])
    doc.add_paragraph("表 9  OSM 城市化组成图层", style="Caption Small")
    add_table(doc, ["组成", "来源图层", "纳入 fclass", "作用"], fclass_rows, [1450, 2200, 3800, 1910], font_size=8.0)
    doc.add_paragraph(
        "道路源数据为中心线，先按 fclass 使用固定单侧半宽缓冲成面；解释性总宽为半宽的两倍，不另行参与计算。"
    )
    road_rows = [
        [row["road_fclass"], f"{row['half_width_m']:.1f}", f"{row['assumed_total_width_m']:.1f}"]
        for _, row in road_widths.iterrows()
    ]
    doc.add_paragraph("表 10  道路宽度假设", style="Caption Small")
    add_table(doc, ["road fclass", "单侧半宽 (m)", "解释性总宽 (m)"], road_rows, [3700, 2830, 2830], font_size=8.5)

    doc.add_heading("7.3 精确并集、分块和 Site Special 复用", level=2)
    doc.add_paragraph(
        "10 个尺度均采用 exact_tiled_vector_union。为控制 4000-6000 m 大范围的内存占用，范围被分成 1000 m"
        "互不重叠的计算瓦片；每个瓦片内仍用原始双精度坐标精确裁剪和 union，最后相加瓦片面积。流程不使用栅格化、"
        "坐标取整、snapping、precision grid、几何简化或按单要素面积直接累加。仅对无效拓扑进行 make_valid 或"
        "必要的 buffer(0) 修复。"
    )
    add_equation(doc, "Area(C(s,b)) = Σₖ Area{Union[Cⱼ ∩ L(s,b) ∩ Tileₖ]}")
    area_error = (urban_long["buffer_area_m2"] - urban_long["water_area_m2"] - urban_long["analysis_land_area_m2"]).abs().max()
    doc.add_paragraph(
        f"实际过程表有 1300 行：River 86 × 10 = 860，Lake 44 × 10 = 440；所有行计算方法均为"
        f" exact_tiled_vector_union。面积守恒最大绝对误差为 {area_error:.7g} m²。最大不透水、建筑和自然用地比例"
        "分别为 51.809776%、29.730442% 和 100%。Site Special 直接使用 130 点位宽表，共 30 个无缺失特征，"
        "不对时间样本求均值。"
    )

    doc.add_heading("8 最终 ML Input 合并、特征数量和质量控制", level=1)
    doc.add_heading("8.1 合并键、模块顺序和站点信息", level=2)
    doc.add_paragraph(
        "以 130 行 site metadata 为左表，各模块均按 water_body_type + site 左连接。经纬度只保留 metadata 中的一份，"
        "模块自身的坐标列在合并前移除。列顺序为站点信息、水生生物、API、β 多样性、WorldPop、WorldCover 和"
        "OSM 城市化。最终 site_special_id 无重复，各模块均覆盖 130 个站点。"
    )
    module_rows = [
        ["站点信息", 8, "site_special_id、水体类型、点位、坐标、采样次数、年份和水期"],
        ["水生生物", module_counts["aquatic"], "样本级指标及变换值的站点均值"],
        ["OSM API", module_counts["api"], "静态站点 × 尺度结果直接透视"],
        ["β 多样性", module_counts["beta"], "64 MeanD + 64 LCBD* 的站点均值"],
        ["WorldPop", module_counts["worldpop"], "唯一 site-year 的站点多年均值"],
        ["WorldCover", module_counts["landuse"], "静态点位级组合比例直接复用"],
        ["OSM 城市化", module_counts["urbanization"], "静态点位级面积比例直接复用"],
    ]
    doc.add_paragraph("表 11  最终矩阵模块与列数", style="Caption Small")
    add_table(doc, ["模块", "列数", "Site Special 处理"], module_rows, [1800, 900, 6660], font_size=8.8)

    doc.add_heading("8.2 缺失值结构", level=2)
    role_labels = {
        "site_info": "站点信息",
        "aquatic_biology_site_mean_feature": "水生生物",
        "api_feature": "OSM API",
        "beta_feature_mean_dissimilarity": "β MeanD",
        "beta_feature_lcbd": "β LCBD*",
        "worldpop_population_density_feature": "WorldPop",
        "landuse_worldcover_feature": "WorldCover",
        "urbanization_osm_feature": "OSM 城市化",
    }
    missing_notes = {
        "site_info": "无缺失",
        "aquatic_biology_site_mean_feature": "308；均为 44 个 Lake × 7 个着生藻类不适用特征",
        "api_feature": "无缺失",
        "beta_feature_mean_dissimilarity": "3180；比较范围/类群覆盖导致的结构性缺失",
        "beta_feature_lcbd": "3180；与对应 MeanD 的适用范围一致",
        "worldpop_population_density_feature": "无缺失",
        "landuse_worldcover_feature": "无缺失",
        "urbanization_osm_feature": "无缺失",
    }
    missing_rows = []
    for role, sub in feature_dictionary.groupby("role", sort=False):
        cols = [col for col in sub["column_name"] if col in matrix.columns]
        missing_rows.append(
            [
                role_labels[role],
                len(cols),
                int(matrix[cols].isna().sum().sum()),
                missing_notes[role],
            ]
        )
    doc.add_paragraph("表 12  各模块实际缺失值", style="Caption Small")
    add_table(doc, ["模块", "列数", "NA 单元格", "解释"], missing_rows, [1800, 900, 1200, 5460], font_size=8.6)
    doc.add_paragraph(
        f"最终矩阵共有 {int(matrix.isna().sum().sum())} 个 NA，全部来自水生生物不适用项或 β 特征的结构性范围，"
        "API、WorldPop、WorldCover、OSM 城市化和站点信息均无缺失。主矩阵没有重复 site_special_id。"
    )

    doc.add_heading("8.3 当前未纳入的 DEM", level=2)
    doc.add_paragraph(
        "DEM 尚未出现在当前 Sample Special 主矩阵及其 Feature_Database 中，因此本次 Site Special 不新增 DEM 列。"
        "这不是缺失值处理，而是当前特征集合的范围限制。后续只有在建立可追溯的 DEM Process 和 Feature Database、"
        "明确点位/范围统计口径并完成 QA 后，才应追加到两套矩阵。"
    )

    doc.add_heading("9 统计解释和建模边界", level=1)
    add_labeled_paragraph(
        doc,
        "响应变量含义：",
        "Site Special 水生生物值表示该点位已有采样事件的平均生态状态，不表示一个合并后的物种群落，也不代表某个特定年份或水期。",
    )
    add_labeled_paragraph(
        doc,
        "采样不均衡：",
        "每站点有 1-9 个样本，均值精度并不相同。建议保留 n_sample_records，在稳健性分析中比较是否剔除仅 1 次采样的 6 个站点，"
        "或根据研究问题采用精度权重；不要让 n_sample_records 无意中作为生态解释变量进入所有模型。",
    )
    add_labeled_paragraph(
        doc,
        "β 特征依赖总体：",
        "MeanD 和 LCBD 是相对于原 577 样本距离矩阵定义的组内相对指标。当前矩阵适合探索空间格局；若用于严格预测验证，"
        "应评估在训练折内部重算距离相关特征的方案，以避免验证样本参与全局 β 结构定义。",
    )
    add_labeled_paragraph(
        doc,
        "变换与交叉验证：",
        "水生生物总量和 LCBD 的 robust 参数来自完整 Sample Special 特征工程。若论文目标是外推预测性能，所有会从数据估计参数的"
        "标准化、缺失处理和特征筛选应放入训练折；若目标是对固定数据库进行解释，应明确这些参数属于预先构建的数据产品。",
    )
    add_labeled_paragraph(
        doc,
        "多尺度共线性：",
        "同一变量的 10 个累计尺度高度相关。建议逐尺度建模、嵌套特征选择或在训练折内进行正则化和共线性诊断，避免把 130 个站点"
        "同时暴露给过多相似预测因子。",
    )
    add_labeled_paragraph(
        doc,
        "适用场景：",
        "研究静态空间压力与多年平均生物状态时，Site Special 更直接；研究年际变化、丰枯水期差异或短期响应时，应使用 577 行"
        "Sample Special，并按 site 分组或进行空间阻塞交叉验证。",
    )

    doc.add_heading("10 文件与工作表追溯索引", level=1)
    output_rows = [
        ["Feature_Process_Aquatic_Biology_Site_Special.xlsx", "样本级复算长表/宽表、站点均值、标准化参数和 QA"],
        ["Feature_Database_Aquatic_Biology_Site_Special.xlsx", "130 × 48 水生生物站点均值输入"],
        ["Feature_Process_Beta_Site_Special.xlsx", "样本 β 长表、站点均值长表、raw/scaled 宽表和 QA"],
        ["Feature_Database_Beta_Site_Special.xlsx", "主矩阵使用的 scaled LCBD 均值及 raw LCBD 备选表"],
        ["Feature_Process_API_Site_Special.xlsx", "POI P1-P8、API 全尺度、权重、敏感性和 Site Special QA"],
        ["Feature_Database_API_Site_Special.xlsx", "130 点位 × 90 API/P1-P8 特征"],
        ["Feature_Process_LandUse_WorldCover_Site_Special.xlsx", "像元长表、类别字典、组合规则、几何规则和 QA"],
        ["Feature_Database_LandUse_WorldCover_Site_Special.xlsx", "130 点位 × 40 WorldCover 组合特征"],
        ["Feature_Process_WorldPop_Site_Special.xlsx", "316 个 site-year 记录、130 个站点均值和 QA"],
        ["Feature_Database_WorldPop_Site_Special.xlsx", "130 点位 × 10 多年平均人口密度"],
        ["Feature_Process_Urbanization_OSM_Site_Special.xlsx", "精确面积长表、fclass、道路宽度、假设和 QA"],
        ["Feature_Database_Urbanization_OSM_Site_Special.xlsx", "130 点位 × 30 城市化面积比例"],
        ["ML_Input_Site_Special_Feature_Matrix.xlsx", "最终 130 × 354 主矩阵、特征字典和合并 QA"],
    ]
    doc.add_paragraph("表 13  Site Special 输出文件", style="Caption Small")
    add_table(doc, ["文件", "用途"], output_rows, [4200, 5160], font_size=8.2)

    doc.add_heading("附录 A 特征命名规则", level=1)
    naming_rows = [
        ["水生生物多样性", "{taxon}_{richness|shannon|margalef|pielou|simpson}"],
        ["水生生物总量", "{taxon}_total_{density|biomass}_log10p1_robust"],
        ["β MeanD", "beta_{scope}_{taxon}_{distance}_mean"],
        ["β LCBD*", "beta_{scope}_{taxon}_{distance}_lcbd_log10p1e-12_robust"],
        ["API", "API_expert_weighted_{buffer}m；P1_z_{buffer}m 至 P8_z_{buffer}m"],
        ["WorldCover", "landuse_{buffer}m_{urban|agriculture|green|water_wetland}_pct"],
        ["WorldPop", "worldpop_{buffer}m_mean_density"],
        ["OSM 城市化", "urbanization_{buffer}m_{impervious|building|natural_land}_pct"],
    ]
    add_table(doc, ["模块", "命名模式"], naming_rows, [2200, 7160], font_size=8.5)

    doc.add_paragraph(
        "版本维护原则：任何原始生物数据库、点位坐标、外部栅格、OSM 图层、分类映射或特征算法变更，都应先更新对应"
        " Feature_Process，再生成 Feature_Database，最后重建 ML Input，并同步更新本说明中的样本数、列数、参数和 QA。"
    )

    path = SITE_DIR / "机器学习特征构建与数据处理方法综合说明_Site_Special.docx"
    temp_path = SITE_DIR / "机器学习特征构建与数据处理方法综合说明_Site_Special.tmp.docx"
    doc.save(temp_path)
    temp_path.replace(path)


def main() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    rename_log = safe_rename_sample_files()

    sample_matrix_path = p_sample("ML_Input_Sample_Feature_Matrix.xlsx")
    sample_input = pd.read_excel(sample_matrix_path, sheet_name="Model_Input_Matrix", engine=PY_ENGINE)
    site_meta = build_site_metadata(sample_input)

    aquatic, _, aquatic_qa = build_aquatic_site_features(rename_log)
    beta, beta_qa = beta_site_features()
    api, api_qa = flatten_api_features()

    landuse, landuse_qa = copy_site_feature_database(
        "Feature_Database_LandUse_WorldCover.xlsx",
        "feature_data",
        "LandUse_WorldCover",
        "sample-special WorldCover landuse feature database",
    )
    copy_process_with_site_note("Feature_Process_LandUse_WorldCover.xlsx", "LandUse_WorldCover", landuse_qa)

    urbanization, urban_qa = copy_site_feature_database(
        "Feature_Database_Urbanization_OSM.xlsx",
        "feature_data",
        "Urbanization_OSM",
        "sample-special OSM urbanization feature database",
    )
    copy_process_with_site_note("Feature_Process_Urbanization_OSM.xlsx", "Urbanization_OSM", urban_qa)

    worldpop, worldpop_qa = worldpop_site_features(site_meta)

    # API database/process files are derived from the sample-special API outputs.
    api_db = SITE_DIR / "Feature_Database_API_Site_Special.xlsx"
    with pd.ExcelWriter(api_db, engine=PY_ENGINE) as writer:
        api.to_excel(writer, sheet_name="feature_data", index=False)
        api_qa.to_excel(writer, sheet_name="QA_Checks", index=False)
    copy_process_with_site_note("Feature_Process_API.xlsx", "API", api_qa)

    module_qas = {
        "aquatic": aquatic_qa,
        "beta": beta_qa,
        "api": api_qa,
        "landuse": landuse_qa,
        "worldpop": worldpop_qa,
        "urbanization": urban_qa,
    }
    matrix = build_ml_input(site_meta, aquatic, beta, api, landuse, urbanization, worldpop, module_qas)

    module_counts = {
        "aquatic": aquatic.shape[1] - 2,
        "beta": beta.shape[1] - 2,
        "api": api.shape[1] - 4,
        "landuse": landuse.shape[1] - 4,
        "worldpop": worldpop.shape[1] - 4,
        "urbanization": urbanization.shape[1] - 4,
    }
    build_method_doc(matrix, module_counts)

    print("Sample special rename log:")
    print(rename_log.to_string(index=False))
    print(f"Site special matrix: {matrix.shape[0]} rows x {matrix.shape[1]} cols")
    print(f"Output folder: {SITE_DIR}")


if __name__ == "__main__":
    main()
