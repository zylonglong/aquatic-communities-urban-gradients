from __future__ import annotations

import bz2
import json
import math
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "processed" / "训练数据汇总"
SAMPLE_DIR = TRAIN_DIR / "Sample_Special"
SITE_DIR = TRAIN_DIR / "Site_Special"
SUMMARY_DIR = ROOT / "data" / "processed" / "Summary"
ANNOTATION_DIR = ROOT / "data" / "processed" / "Biological_Trait_Annotation"
CACHE_DIR = ANNOTATION_DIR / "Source_Cache"
BUILD_DIR = ANNOTATION_DIR / "Workbook_Build_CSV"

RAW_AQUATIC = SUMMARY_DIR / "Aquatic_Biological_Database.xlsx"
SAMPLE_MATRIX = SAMPLE_DIR / "ML_Input_Sample_Special_Feature_Matrix.xlsx"
SITE_MATRIX = SITE_DIR / "ML_Input_Site_Special_Feature_Matrix.xlsx"
METHOD_REFERENCE = TRAIN_DIR / "水生生物数据目标变量统计方法.docx"

TODAY = date.today().isoformat()
PY_ENGINE = "openpyxl"

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

GROUP_SHEETS = {
    "benthos": "Benthos",
    "benthos_qualitative": "Benthos_Qualitative",
    "fish": "Fish",
    "macrophytes": "Macrophytes",
    "phytoplankton": "Phytoplankton",
    "periphyton": "Periphyton",
    "zooplankton": "Zooplankton",
}

EXTERNAL_URLS = {
    "fishbase_species_fb_2104.tsv.bz2": "https://github.com/ropensci/rfishbase/releases/download/fb-21.04/fb.2fspecies.tsv.bz2",
    "fishbase_ecology_fb_2104.tsv.bz2": "https://github.com/ropensci/rfishbase/releases/download/fb-21.04/fb.2fecology.tsv.bz2",
    "fishbase_stocks_fb_2104.tsv.bz2": "https://github.com/ropensci/rfishbase/releases/download/fb-21.04/fb.2fstocks.tsv.bz2",
    "griis_china_v1_4_dwca.zip": "https://cloud.gbif.org/griis/archive.do?r=griis-china",
}

MACROPHYTE_GROWTH_FORMS = {
    "submerged": {
        "vallisneria", "hydrilla", "ceratophyllum", "potamogeton", "myriophyllum",
        "najas", "elodea", "egeria", "cabomba", "ottelia", "lymnophyton",
    },
    "floating": {
        "eichhornia", "pontederia", "pistia", "lemna", "spirodela", "azolla",
        "salvinia", "nymphoides", "nymphaea", "nelumbo", "trapa", "hydrocharis",
        "limnobium", "ludwigia",
    },
    "emergent": {
        "phragmites", "typha", "schoenoplectus", "scirpus", "acorus", "canna",
        "colocasia", "sagittaria", "zizania", "cyperus", "iris", "thalia",
        "eleocharis", "juncus", "alternanthera", "bolboschoenus", "monochoria",
    },
}

BLOOM_FORMING_GENERA = {
    "microcystis", "anabaena", "dolichospermum", "aphanizomenon", "planktothrix",
    "oscillatoria", "cylindrospermopsis", "raphidiopsis", "nodularia", "lyngbya",
    "pseudanabaena", "anabaenopsis", "radiocystis", "woronichinia", "phormidium",
}

DIATOM_GUILDS = {
    "low_profile": {
        "achnanthes", "achnanthidium", "cocconeis", "planothidium", "karayevia",
        "meridion", "eolimna", "diatoma",
    },
    "high_profile": {
        "gomphonema", "cymbella", "encyonema", "fragilaria", "synedra", "ulnaria",
        "licmophora", "tabellaria", "eunotia", "asterionella",
    },
    "motile": {
        "navicula", "nitzschia", "surirella", "gyrosigma", "pinnularia", "amphora",
        "craticula", "sellaphora", "luticola", "diploneis", "caloneis", "neidium",
        "pleurosigma", "stauroneis", "cymatopleura", "mastogloia",
    },
    "planktic": {
        "cyclotella", "stephanodiscus", "aulacoseira", "melosira", "coscinodiscus",
        "skeletonema", "thalassiosira",
    },
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\u00a0", " ").replace("\u3000", " ").strip()
    return re.sub(r"\s+", " ", text)


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_sampling_time(value: object) -> tuple[str, str]:
    text = normalize_text(value)
    year_match = re.search(r"(20\d{2})", text)
    year = year_match.group(1) if year_match else ""
    lower = text.lower()
    if any(token in lower for token in ["high-flow", "high flow", "hjgh-flow", "hjgh flow", "丰水"]):
        flow = "high-flow"
    elif any(token in lower for token in ["low-flow", "low flow", "枯水"]):
        flow = "low-flow"
    else:
        flow = ""
    return year, flow


def make_sample_id(water_body_type: object, site: object, sampling_time: object) -> str:
    year, flow = parse_sampling_time(sampling_time)
    flow_short = "high" if flow == "high-flow" else "low" if flow == "low-flow" else "unknown"
    return f"{normalize_text(water_body_type)}_{normalize_text(site)}_{year}_{flow_short}"


def add_sample_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["water_body_type"] = out["Water body type"].map(normalize_text)
    out["sampling_time"] = out["Sampling time"].map(normalize_text)
    out["site"] = out["Site"].map(normalize_text)
    parsed = out["sampling_time"].map(parse_sampling_time)
    out["sampling_year"] = parsed.map(lambda x: x[0])
    out["flow_period"] = parsed.map(lambda x: x[1])
    out["sample_id"] = [
        make_sample_id(wt, site, sampling_time)
        for wt, site, sampling_time in zip(out["water_body_type"], out["site"], out["sampling_time"])
    ]
    return out


def taxon_key(df: pd.DataFrame) -> pd.Series:
    scientific = df.get("Scientific name", pd.Series("", index=df.index)).map(normalize_text)
    species = df.get("Species", pd.Series("", index=df.index)).map(normalize_text)
    return scientific.mask(scientific.eq(""), species)


def latin_genus(value: object) -> str:
    match = re.search(r"[A-Za-z][A-Za-z-]+", normalize_text(value))
    return match.group(0).lower() if match else ""


def canonical_binomial(value: object) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z-]+", normalize_text(value))
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[1].lower() not in {"sp", "spp", "cf", "aff", "var"}:
        return f"{tokens[0]} {tokens[1]}".lower()
    return tokens[0].lower()


def mode_value(series: pd.Series) -> object:
    values = series.dropna().astype(str).map(normalize_text)
    values = values[values.ne("")]
    if values.empty:
        return np.nan
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def safe_percent(numerator: float, denominator: float, zero_value: float = 0.0) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return zero_value
    return float(numerator / denominator * 100.0)


def shannon_from_weights(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size <= 1:
        return 0.0
    proportions = array / array.sum()
    return float(-np.sum(proportions * np.log(proportions)))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = clean_numeric(values).to_numpy(dtype=float)
    weight_array = clean_numeric(weights).fillna(0).to_numpy(dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not valid.any() or weight_array[valid].sum() <= 0:
        return np.nan
    return float(np.sum(value_array[valid] * weight_array[valid]) / np.sum(weight_array[valid]))


def ensure_external_sources() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in EXTERNAL_URLS.items():
        path = CACHE_DIR / filename
        if not path.exists() or path.stat().st_size == 0:
            urllib.request.urlretrieve(url, path)


def load_sheets() -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(RAW_AQUATIC, sheet_name=None, engine=PY_ENGINE)
    return {name: add_sample_fields(df) for name, df in sheets.items() if name != "Water_Type_QA"}


def build_taxon_master(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group, sheet_name in GROUP_SHEETS.items():
        df = sheets[sheet_name].copy()
        df["original_scientific_name"] = taxon_key(df)
        df["chinese_name"] = df.get("Species", pd.Series("", index=df.index)).map(normalize_text)
        df = df[df["original_scientific_name"].ne("")]
        group_frame = (
            df.groupby("original_scientific_name", as_index=False)
            .agg(
                chinese_name=("chinese_name", mode_value),
                phylum=("Phylum", mode_value),
                class_name=("Class", mode_value),
                order=("Order", mode_value),
                family=("Family", mode_value),
                genus=("Genus", mode_value),
                source_record_count=("sample_id", "size"),
                source_sample_count=("sample_id", "nunique"),
            )
        )
        group_frame.insert(0, "taxon_group", group)
        group_frame["source_sheet"] = sheet_name
        group_frame["latin_genus"] = group_frame["original_scientific_name"].map(latin_genus)
        group_frame["original_name_key"] = group_frame["original_scientific_name"].map(canonical_binomial)
        group_frame["original_rank_inferred"] = np.where(
            group_frame["original_scientific_name"].str.contains(r"\bsp\.?\b|\bspp\.?\b", case=False, regex=True),
            "genus",
            np.where(group_frame["original_name_key"].str.contains(" "), "species", "unresolved"),
        )
        frames.append(group_frame)
    master = pd.concat(frames, ignore_index=True)
    master.insert(0, "taxon_record_id", [f"TX{i:05d}" for i in range(1, len(master) + 1)])
    return master.sort_values(["taxon_group", "original_scientific_name"]).reset_index(drop=True)


def gbif_query_name(name: str) -> str:
    normalized = normalize_text(name)
    match = re.match(r"^([A-Za-z][A-Za-z-]+)\s+(?:sp\.?|spp\.?)\b", normalized, flags=re.I)
    return match.group(1) if match else normalized


def fetch_gbif_match(name: str) -> dict[str, object]:
    query_name = gbif_query_name(name)
    if not query_name:
        return {"note": "blank_query"}
    params = urllib.parse.urlencode({"name": query_name})
    url = f"https://api.gbif.org/v1/species/match?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        result["query_name"] = query_name
        result["query_url"] = url
        return result
    except Exception as exc:
        return {"query_name": query_name, "query_url": url, "note": f"request_error:{type(exc).__name__}"}


def add_gbif_matches(master: pd.DataFrame) -> pd.DataFrame:
    cache_path = CACHE_DIR / "gbif_species_match_cache.json"
    cache: dict[str, dict[str, object]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    names = sorted(set(master["original_scientific_name"]) - set(cache))
    if names:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_gbif_match, name): name for name in names}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    cache[name] = future.result()
                except Exception as exc:
                    cache[name] = {"note": f"worker_error:{type(exc).__name__}"}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    records = []
    for name in master["original_scientific_name"]:
        result = cache.get(name, {})
        rank = normalize_text(result.get("rank", "")).lower()
        match_type = normalize_text(result.get("matchType", "")).lower()
        confidence = pd.to_numeric(result.get("confidence"), errors="coerce")
        if not result.get("usageKey"):
            match_status = "unmatched"
        elif rank in {"genus", "family", "order"}:
            match_status = f"{rank}-level"
        elif match_type == "exact":
            match_status = "exact"
        else:
            match_status = "fuzzy"
        records.append(
            {
                "gbif_query_name": result.get("query_name", ""),
                "gbif_usage_key": result.get("usageKey"),
                "accepted_name": result.get("canonicalName") or result.get("scientificName") or "",
                "accepted_rank": rank,
                "gbif_taxonomic_status": result.get("status", ""),
                "match_status": match_status,
                "gbif_match_type": match_type,
                "gbif_confidence": confidence,
                "gbif_note": result.get("note", ""),
                "source_database": "GBIF Species API v1",
                "source_url": result.get("query_url", ""),
                "manual_check": bool(match_status != "exact" or pd.isna(confidence) or confidence < 90),
            }
        )
    out = pd.concat([master.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    out["accepted_name_key"] = out["accepted_name"].map(canonical_binomial)
    out["accepted_name_key"] = out["accepted_name_key"].mask(out["accepted_name_key"].eq(""), out["original_name_key"])
    out["confidence"] = np.select(
        [out["match_status"].eq("exact") & out["gbif_confidence"].ge(90), out["match_status"].isin(["genus-level", "family-level"])],
        ["high", "medium"],
        default="low",
    )
    return out


def load_griis() -> pd.DataFrame:
    archive = CACHE_DIR / "griis_china_v1_4_dwca.zip"
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open("taxon.txt") as handle:
            taxon = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False)
        with bundle.open("distribution.txt") as handle:
            distribution = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False)
        with bundle.open("speciesprofile.txt") as handle:
            profile = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False)
    result = taxon.merge(distribution, on="id", how="left").merge(profile, on="id", how="left")
    result["griis_name_key"] = result["scientificName"].map(canonical_binomial)
    return result


def add_griis_status(annotation: pd.DataFrame, griis: pd.DataFrame) -> pd.DataFrame:
    lookup = (
        griis[griis["griis_name_key"].ne("")]
        .sort_values("id")
        .drop_duplicates("griis_name_key")
        .set_index("griis_name_key")
    )
    out = annotation.copy()
    keys = out["accepted_name_key"].mask(out["accepted_name_key"].eq(""), out["original_name_key"])
    out["griis_listed_alien_china"] = keys.map(lookup["establishmentMeans"]).eq("Alien")
    out["griis_invasive_china"] = keys.map(lookup["isInvasive"]).astype(str).str.lower().eq("invasive")
    out["griis_habitat"] = keys.map(lookup["habitat"])
    out["griis_match_name"] = keys.where(keys.isin(lookup.index), "")
    out["griis_source"] = "GRIIS-China v1.4 (2026-03-28), CC-BY 4.0"
    out["griis_interpretation"] = "listed=True is verified alien/invasive; not listed is not proof of native status"
    return out


def read_fishbase_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        CACHE_DIR / filename,
        sep="\t",
        compression="bz2",
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )


def build_fish_annotation(master: pd.DataFrame, griis: pd.DataFrame) -> pd.DataFrame:
    fish = master[master["taxon_group"].eq("fish")].copy()
    fish = add_griis_status(fish, griis)
    species = read_fishbase_table("fishbase_species_fb_2104.tsv.bz2")
    ecology = read_fishbase_table("fishbase_ecology_fb_2104.tsv.bz2")
    stocks = read_fishbase_table("fishbase_stocks_fb_2104.tsv.bz2")
    species["fishbase_name_key"] = (species["Genus"].fillna("") + " " + species["Species"].fillna("")).map(canonical_binomial)
    species = species.sort_values("SpecCode").drop_duplicates("fishbase_name_key")
    species_lookup = species.set_index("fishbase_name_key")

    ecology["DietTroph_num"] = clean_numeric(ecology["DietTroph"])
    ecology["FoodTroph_num"] = clean_numeric(ecology["FoodTroph"])
    ecology_summary = (
        ecology.groupby("SpecCode", as_index=False)
        .agg(
            diet_trophic_level=("DietTroph_num", "median"),
            food_trophic_level=("FoodTroph_num", "median"),
            fishbase_feeding_type=("FeedingType", mode_value),
            fishbase_herbivory_class=("Herbivory2", mode_value),
        )
    ).set_index("SpecCode")
    stock_summary = stocks.groupby("SpecCode", as_index=False).agg(fishbase_resilience=("Resilience", mode_value)).set_index("SpecCode")

    match_keys = fish["accepted_name_key"].where(fish["accepted_name_key"].isin(species_lookup.index), fish["original_name_key"])
    fish["fishbase_match_key"] = match_keys.where(match_keys.isin(species_lookup.index), "")
    fish["fishbase_spec_code"] = fish["fishbase_match_key"].map(species_lookup["SpecCode"])
    fish["fishbase_max_length"] = clean_numeric(fish["fishbase_match_key"].map(species_lookup["Length"]))
    fish["fishbase_max_length_type"] = fish["fishbase_match_key"].map(species_lookup["LTypeMaxM"])
    fish["fishbase_habitat_position"] = fish["fishbase_match_key"].map(species_lookup["DemersPelag"])
    fish["fishbase_vulnerability"] = clean_numeric(fish["fishbase_match_key"].map(species_lookup["Vulnerability"]))
    fish["fishbase_trophic_level_diet"] = fish["fishbase_spec_code"].map(ecology_summary["diet_trophic_level"])
    fish["fishbase_trophic_level_food"] = fish["fishbase_spec_code"].map(ecology_summary["food_trophic_level"])
    fish["fishbase_trophic_level"] = fish["fishbase_trophic_level_diet"].fillna(fish["fishbase_trophic_level_food"])
    fish["fishbase_feeding_type"] = fish["fishbase_spec_code"].map(ecology_summary["fishbase_feeding_type"])
    fish["fishbase_herbivory_class"] = fish["fishbase_spec_code"].map(ecology_summary["fishbase_herbivory_class"])
    fish["fishbase_resilience"] = fish["fishbase_spec_code"].map(stock_summary["fishbase_resilience"])
    resilience_text = fish["fishbase_resilience"].fillna("").str.lower()
    fish["fishbase_resilience_ordinal"] = np.select(
        [resilience_text.str.contains("very low"), resilience_text.str.contains("low"), resilience_text.str.contains("medium"), resilience_text.str.contains("high")],
        [0.0, 1.0, 2.0, 3.0],
        default=np.nan,
    )
    fish["trophic_guild_operational"] = np.select(
        [fish["fishbase_trophic_level"].lt(2.5), fish["fishbase_trophic_level"].le(3.5), fish["fishbase_trophic_level"].gt(3.5)],
        ["herbivore_detritivore", "omnivore", "carnivore"],
        default="unmatched",
    )
    fish["fishbase_source"] = "rfishbase FishBase snapshot 21.04; species/ecology/stocks tables"
    return fish


def build_benthos_annotation(master: pd.DataFrame) -> pd.DataFrame:
    benthos = master[master["taxon_group"].isin(["benthos", "benthos_qualitative"])].copy()
    benthos["ept_flag"] = benthos["order"].isin(["蜉蝣目", "襀翅目", "毛翅目"])
    benthos["chironomidae_flag"] = benthos["family"].eq("摇蚊科")
    benthos["oligochaeta_flag"] = benthos["class_name"].eq("寡毛纲") | benthos["order"].eq("单向蚓目")
    benthos["sensitive_indicator_proxy"] = benthos["ept_flag"]
    benthos["tolerant_indicator_proxy"] = benthos["chironomidae_flag"] | benthos["oligochaeta_flag"]
    benthos["indicator_rule_source"] = "local taxonomic hierarchy; EPT orders, Chironomidae family, Oligochaeta class/order"
    benthos["tolerance_score"] = np.nan
    benthos["functional_feeding_group"] = ""
    benthos["external_trait_status"] = "not populated: freshwaterecology.info bulk trait export requires registered access and regional validation"
    return benthos


def classify_macrophyte_growth_form(name: object) -> tuple[str, str, bool]:
    genus = latin_genus(name)
    for growth_form, genera in MACROPHYTE_GROWTH_FORMS.items():
        if genus in genera:
            return growth_form, "medium", True
    return "riparian_or_unresolved", "low", True


def build_macrophyte_annotation(master: pd.DataFrame, griis: pd.DataFrame) -> pd.DataFrame:
    macrophytes = master[master["taxon_group"].eq("macrophytes")].copy()
    macrophytes = add_griis_status(macrophytes, griis)
    classified = macrophytes["accepted_name"].mask(macrophytes["accepted_name"].eq(""), macrophytes["original_scientific_name"]).map(classify_macrophyte_growth_form)
    macrophytes["growth_form"] = classified.map(lambda x: x[0])
    macrophytes["growth_form_confidence"] = classified.map(lambda x: x[1])
    macrophytes["growth_form_manual_check"] = classified.map(lambda x: x[2])
    macrophytes["growth_form_source"] = "conservative genus-level operational rules; manual botanical review required"
    macrophytes["plant_height"] = np.nan
    macrophytes["height_trait_status"] = "not populated: TRY/Flora trait values require licensed or manual expert validation"
    return macrophytes


def build_phytoplankton_annotation(master: pd.DataFrame) -> pd.DataFrame:
    phyto = master[master["taxon_group"].eq("phytoplankton")].copy()
    phyto["cyanobacteria_flag"] = phyto["phylum"].eq("蓝藻门")
    phyto["diatom_flag"] = phyto["phylum"].eq("硅藻门")
    phyto["chlorophyta_flag"] = phyto["phylum"].eq("绿藻门")
    name_for_genus = phyto["accepted_name"].mask(phyto["accepted_name"].eq(""), phyto["original_scientific_name"])
    phyto["bloom_forming_flag"] = name_for_genus.map(latin_genus).isin(BLOOM_FORMING_GENERA)
    phyto["bloom_rule_source"] = "conservative genus list from freshwater cyanobacterial bloom literature"
    phyto["reynolds_functional_group"] = ""
    phyto["mbfg_group"] = ""
    phyto["external_trait_status"] = "RFG/MBFG not populated: morphology/ecology assignment requires curated taxon-level literature review"
    return phyto


def classify_diatom_guild(name: object) -> str:
    genus = latin_genus(name)
    for guild, genera in DIATOM_GUILDS.items():
        if genus in genera:
            return guild
    return ""


def build_periphyton_annotation(master: pd.DataFrame) -> pd.DataFrame:
    periphyton = master[master["taxon_group"].eq("periphyton")].copy()
    periphyton["cyanobacteria_flag"] = periphyton["phylum"].eq("蓝藻门")
    periphyton["diatom_flag"] = periphyton["phylum"].eq("硅藻门")
    name_for_genus = periphyton["accepted_name"].mask(periphyton["accepted_name"].eq(""), periphyton["original_scientific_name"])
    periphyton["diatom_guild"] = np.where(periphyton["diatom_flag"], name_for_genus.map(classify_diatom_guild), "")
    periphyton["diatom_guild_confidence"] = np.where(periphyton["diatom_guild"].ne(""), "medium", "unmatched")
    periphyton["diatom_guild_source"] = "conservative genus-level implementation of Passy (2007) and Rimet & Bouchez (2012); manual review required"
    periphyton["tolerant_diatom_flag"] = np.nan
    periphyton["tolerance_trait_status"] = "not populated: OMNIDIA/TDI taxon scores require licensed or regionally validated source"
    return periphyton


def build_zooplankton_annotation(master: pd.DataFrame) -> pd.DataFrame:
    zoo = master[master["taxon_group"].eq("zooplankton")].copy()
    zoo["major_group"] = np.select(
        [
            zoo["phylum"].eq("轮虫"),
            zoo["phylum"].eq("枝角类"),
            zoo["phylum"].eq("桡足类"),
            zoo["phylum"].eq("原生动物"),
        ],
        ["rotifer", "cladocera", "copepoda", "protozoa"],
        default="unresolved",
    )
    zoo["major_group_source"] = "local phylum classification"
    zoo["body_size_trait"] = np.nan
    zoo["feeding_type"] = ""
    zoo["external_trait_status"] = "body size and feeding traits not populated: curated zooplankton trait table required"
    return zoo


def annotation_lookup(annotation: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    use_columns = ["taxon_group", "original_scientific_name"] + columns
    return annotation[use_columns].drop_duplicates(["taxon_group", "original_scientific_name"])


def merge_annotation(df: pd.DataFrame, group: str, annotation: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["original_scientific_name"] = taxon_key(out)
    out["taxon_group"] = group
    return out.merge(annotation_lookup(annotation, columns), on=["taxon_group", "original_scientific_name"], how="left")


def base_sample_frame(sample_meta: pd.DataFrame) -> pd.DataFrame:
    return sample_meta[BASE_SAMPLE_COLUMNS].drop_duplicates("sample_id").sort_values(["water_body_type", "site", "sampling_year", "flow_period"]).reset_index(drop=True)


def summarize_fish(df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "griis_listed_alien_china", "griis_invasive_china", "fishbase_trophic_level",
        "fishbase_max_length", "fishbase_resilience_ordinal", "trophic_guild_operational",
        "fishbase_match_key", "class_name",
    ]
    data = merge_annotation(df, "fish", annotation, columns)
    data["weight_num"] = clean_numeric(data["Weight"]).fillna(0).clip(lower=0)
    data["total_length_num"] = clean_numeric(data["Total length"])
    data["standard_length_num"] = clean_numeric(data["Standard length"])
    rows: list[dict[str, object]] = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        total_count = len(sub)
        total_weight = float(sub["weight_num"].sum())
        counts = sub.groupby("original_scientific_name").size()
        alien = sub["griis_listed_alien_china"].fillna(False)
        invasive = sub["griis_invasive_china"].fillna(False)
        trophic_valid = sub["fishbase_trophic_level"].notna()
        max_length_valid = sub["fishbase_max_length"].notna()
        resilience_valid = sub["fishbase_resilience_ordinal"].notna()
        valid_total_length = sub["total_length_num"].where(sub["total_length_num"].gt(0))
        valid_standard_length = sub["standard_length_num"].where(sub["standard_length_num"].gt(0))
        guild = sub["trophic_guild_operational"].fillna("unmatched")
        rows.append(
            {
                **{column: meta[column] for column in BASE_SAMPLE_COLUMNS},
                "fish_observed_taxa_richness": int(counts.size),
                "fish_mean_total_length": float(valid_total_length.mean()) if valid_total_length.notna().any() else np.nan,
                "fish_median_total_length": float(valid_total_length.median()) if valid_total_length.notna().any() else np.nan,
                "fish_mean_standard_length": float(valid_standard_length.mean()) if valid_standard_length.notna().any() else np.nan,
                "fish_biomass_weighted_total_length": weighted_mean(sub["total_length_num"], sub["weight_num"]),
                "fish_small_individual_abundance_pct_le10cm": safe_percent(valid_total_length.le(10).sum(), valid_total_length.notna().sum()),
                "fish_dominant_species_abundance_pct": safe_percent(counts.max() if not counts.empty else 0, total_count),
                "fish_griis_alien_richness": int(sub.loc[alien, "original_scientific_name"].nunique()),
                "fish_griis_alien_abundance_pct": safe_percent(alien.sum(), total_count),
                "fish_griis_alien_biomass_pct": safe_percent(sub.loc[alien, "weight_num"].sum(), total_weight),
                "fish_griis_invasive_richness": int(sub.loc[invasive, "original_scientific_name"].nunique()),
                "fish_trophic_level_CWM_abundance": weighted_mean(sub["fishbase_trophic_level"], pd.Series(1.0, index=sub.index)),
                "fish_trophic_level_CWM_biomass": weighted_mean(sub["fishbase_trophic_level"], sub["weight_num"]),
                "fish_max_length_CWM_abundance": weighted_mean(sub["fishbase_max_length"], pd.Series(1.0, index=sub.index)),
                "fish_resilience_CWM_abundance": weighted_mean(sub["fishbase_resilience_ordinal"], pd.Series(1.0, index=sub.index)),
                "fish_omnivore_abundance_pct": safe_percent(guild.eq("omnivore").sum(), total_count),
                "fish_carnivore_abundance_pct": safe_percent(guild.eq("carnivore").sum(), total_count),
                "fish_carnivore_biomass_pct": safe_percent(sub.loc[guild.eq("carnivore"), "weight_num"].sum(), total_weight),
                "fish_trophic_trait_abundance_coverage_pct": safe_percent(trophic_valid.sum(), total_count),
                "fish_trophic_trait_biomass_coverage_pct": safe_percent(sub.loc[trophic_valid, "weight_num"].sum(), total_weight),
                "fish_max_length_trait_abundance_coverage_pct": safe_percent(max_length_valid.sum(), total_count),
                "fish_resilience_trait_abundance_coverage_pct": safe_percent(resilience_valid.sum(), total_count),
                "fish_non_actinopterygii_record_pct": safe_percent((~sub["class_name"].eq("辐鳍鱼纲")).sum(), total_count),
            }
        )
    return pd.DataFrame(rows)


def summarize_benthos_quantitative(df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    columns = ["ept_flag", "chironomidae_flag", "oligochaeta_flag", "sensitive_indicator_proxy", "tolerant_indicator_proxy"]
    data = merge_annotation(df, "benthos", annotation, columns)
    data["density_num"] = clean_numeric(data["Density"]).fillna(0).clip(lower=0)
    data["biomass_num"] = clean_numeric(data["Biomass"]).fillna(0).clip(lower=0)
    rows = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        total_density = float(sub["density_num"].sum())
        total_biomass = float(sub["biomass_num"].sum())
        positive = sub["density_num"].gt(0)
        ept = sub["ept_flag"].fillna(False)
        chir = sub["chironomidae_flag"].fillna(False)
        oligo = sub["oligochaeta_flag"].fillna(False)
        tolerant = sub["tolerant_indicator_proxy"].fillna(False)
        density_by_taxon = sub.groupby("original_scientific_name")["density_num"].sum()
        rows.append(
            {
                **{column: meta[column] for column in BASE_SAMPLE_COLUMNS},
                "benthos_quant_EPT_richness": int(sub.loc[positive & ept, "original_scientific_name"].nunique()),
                "benthos_quant_Chironomidae_richness": int(sub.loc[positive & chir, "original_scientific_name"].nunique()),
                "benthos_quant_Oligochaeta_richness": int(sub.loc[positive & oligo, "original_scientific_name"].nunique()),
                "benthos_EPT_density_pct": safe_percent(sub.loc[ept, "density_num"].sum(), total_density),
                "benthos_EPT_biomass_pct": safe_percent(sub.loc[ept, "biomass_num"].sum(), total_biomass),
                "benthos_Chironomidae_density_pct": safe_percent(sub.loc[chir, "density_num"].sum(), total_density),
                "benthos_Chironomidae_biomass_pct": safe_percent(sub.loc[chir, "biomass_num"].sum(), total_biomass),
                "benthos_Oligochaeta_density_pct": safe_percent(sub.loc[oligo, "density_num"].sum(), total_density),
                "benthos_Oligochaeta_biomass_pct": safe_percent(sub.loc[oligo, "biomass_num"].sum(), total_biomass),
                "benthos_tolerant_proxy_density_pct": safe_percent(sub.loc[tolerant, "density_num"].sum(), total_density),
                "benthos_sensitive_proxy_density_pct": safe_percent(sub.loc[ept, "density_num"].sum(), total_density),
                "benthos_indicator_balance_density_pct": safe_percent(sub.loc[ept, "density_num"].sum(), total_density) - safe_percent(sub.loc[tolerant, "density_num"].sum(), total_density),
                "benthos_dominant_taxon_density_pct": safe_percent(density_by_taxon.max() if not density_by_taxon.empty else 0, total_density),
            }
        )
    return pd.DataFrame(rows)


def summarize_benthos_qualitative(df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    columns = ["ept_flag", "chironomidae_flag", "oligochaeta_flag", "sensitive_indicator_proxy", "tolerant_indicator_proxy"]
    data = merge_annotation(df, "benthos_qualitative", annotation, columns)
    data = data.drop_duplicates(["sample_id", "original_scientific_name"])
    rows = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        ept = sub["ept_flag"].fillna(False)
        chir = sub["chironomidae_flag"].fillna(False)
        oligo = sub["oligochaeta_flag"].fillna(False)
        tolerant = sub["tolerant_indicator_proxy"].fillna(False)
        total_taxa = sub["original_scientific_name"].nunique()
        rows.append(
            {
                **{column: meta[column] for column in BASE_SAMPLE_COLUMNS},
                "benthos_qual_EPT_richness": int(sub.loc[ept, "original_scientific_name"].nunique()),
                "benthos_qual_Chironomidae_richness": int(sub.loc[chir, "original_scientific_name"].nunique()),
                "benthos_qual_Oligochaeta_richness": int(sub.loc[oligo, "original_scientific_name"].nunique()),
                "benthos_qual_sensitive_proxy_richness": int(sub.loc[ept, "original_scientific_name"].nunique()),
                "benthos_qual_tolerant_proxy_richness": int(sub.loc[tolerant, "original_scientific_name"].nunique()),
                "benthos_qual_EPT_taxa_pct": safe_percent(sub.loc[ept, "original_scientific_name"].nunique(), total_taxa),
                "benthos_qual_tolerant_proxy_taxa_pct": safe_percent(sub.loc[tolerant, "original_scientific_name"].nunique(), total_taxa),
            }
        )
    return pd.DataFrame(rows)


def summarize_macrophytes(df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    columns = ["growth_form", "growth_form_confidence", "griis_listed_alien_china", "griis_invasive_china"]
    data = merge_annotation(df, "macrophytes", annotation, columns)
    data["density_num"] = clean_numeric(data["Density"]).fillna(0).clip(lower=0)
    data["biomass_num"] = clean_numeric(data["Biomass"]).fillna(0).clip(lower=0)
    rows = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        total_biomass = float(sub["biomass_num"].sum())
        total_density = float(sub["density_num"].sum())
        covered = sub["growth_form_confidence"].isin(["high", "medium"])
        alien = sub["griis_listed_alien_china"].fillna(False)
        positive = sub["density_num"].gt(0) | sub["biomass_num"].gt(0)
        group_biomass = sub.loc[covered].groupby("growth_form")["biomass_num"].sum()
        dominant_group_pct = safe_percent(group_biomass.max() if not group_biomass.empty else 0, total_biomass)
        row = {
            **{column: meta[column] for column in BASE_SAMPLE_COLUMNS},
            "macrophyte_submerged_richness": int(sub.loc[positive & sub["growth_form"].eq("submerged"), "original_scientific_name"].nunique()),
            "macrophyte_emergent_richness": int(sub.loc[positive & sub["growth_form"].eq("emergent"), "original_scientific_name"].nunique()),
            "macrophyte_floating_richness": int(sub.loc[positive & sub["growth_form"].eq("floating"), "original_scientific_name"].nunique()),
            "macrophyte_submerged_biomass_pct": safe_percent(sub.loc[sub["growth_form"].eq("submerged"), "biomass_num"].sum(), total_biomass),
            "macrophyte_emergent_biomass_pct": safe_percent(sub.loc[sub["growth_form"].eq("emergent"), "biomass_num"].sum(), total_biomass),
            "macrophyte_floating_biomass_pct": safe_percent(sub.loc[sub["growth_form"].eq("floating"), "biomass_num"].sum(), total_biomass),
            "macrophyte_riparian_or_unresolved_biomass_pct": safe_percent(sub.loc[sub["growth_form"].eq("riparian_or_unresolved"), "biomass_num"].sum(), total_biomass),
            "macrophyte_growth_form_richness": int(group_biomass.gt(0).sum()),
            "macrophyte_growth_form_shannon_biomass": shannon_from_weights(group_biomass),
            "macrophyte_dominant_growth_form_biomass_pct": dominant_group_pct,
            "macrophyte_growth_form_biomass_coverage_pct": safe_percent(sub.loc[covered, "biomass_num"].sum(), total_biomass),
            "macrophyte_growth_form_density_coverage_pct": safe_percent(sub.loc[covered, "density_num"].sum(), total_density),
            "macrophyte_griis_alien_richness": int(sub.loc[alien & positive, "original_scientific_name"].nunique()),
            "macrophyte_griis_alien_biomass_pct": safe_percent(sub.loc[alien, "biomass_num"].sum(), total_biomass),
            "macrophyte_griis_alien_density_pct": safe_percent(sub.loc[alien, "density_num"].sum(), total_density),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_phytoplankton(df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    columns = ["cyanobacteria_flag", "diatom_flag", "chlorophyta_flag", "bloom_forming_flag", "phylum"]
    data = merge_annotation(df, "phytoplankton", annotation, columns)
    data["density_num"] = clean_numeric(data["Density"]).fillna(0).clip(lower=0)
    data["biomass_num"] = clean_numeric(data["Biomass"]).fillna(0).clip(lower=0)
    rows = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        total_density = float(sub["density_num"].sum())
        total_biomass = float(sub["biomass_num"].sum())
        positive = sub["density_num"].gt(0) | sub["biomass_num"].gt(0)
        flags = {
            "cyanobacteria": sub["cyanobacteria_flag"].fillna(False),
            "diatom": sub["diatom_flag"].fillna(False),
            "chlorophyta": sub["chlorophyta_flag"].fillna(False),
            "bloom_forming": sub["bloom_forming_flag"].fillna(False),
        }
        phylum_density = sub.groupby("phylum")["density_num"].sum()
        phylum_biomass = sub.groupby("phylum")["biomass_num"].sum()
        row = {**{column: meta[column] for column in BASE_SAMPLE_COLUMNS}}
        for label, flag in flags.items():
            row[f"phytoplankton_{label}_richness"] = int(sub.loc[positive & flag, "original_scientific_name"].nunique())
            row[f"phytoplankton_{label}_density_pct"] = safe_percent(sub.loc[flag, "density_num"].sum(), total_density)
            row[f"phytoplankton_{label}_biomass_pct"] = safe_percent(sub.loc[flag, "biomass_num"].sum(), total_biomass)
        row["phytoplankton_high_risk_algae_density_pct"] = row["phytoplankton_bloom_forming_density_pct"]
        row["phytoplankton_high_risk_algae_biomass_pct"] = row["phytoplankton_bloom_forming_biomass_pct"]
        row["phytoplankton_phylum_shannon_density"] = shannon_from_weights(phylum_density)
        row["phytoplankton_phylum_shannon_biomass"] = shannon_from_weights(phylum_biomass)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_periphyton(df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    columns = ["cyanobacteria_flag", "diatom_flag", "diatom_guild"]
    data = merge_annotation(df, "periphyton", annotation, columns)
    data["density_num"] = clean_numeric(data["Density"]).fillna(0).clip(lower=0)
    data["biomass_num"] = clean_numeric(data["Biomass"]).fillna(0).clip(lower=0)
    rows = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        total_density = float(sub["density_num"].sum())
        total_biomass = float(sub["biomass_num"].sum())
        positive = sub["density_num"].gt(0) | sub["biomass_num"].gt(0)
        diatom = sub["diatom_flag"].fillna(False)
        cyano = sub["cyanobacteria_flag"].fillna(False)
        guild_covered = diatom & sub["diatom_guild"].fillna("").ne("")
        guild_density = sub.loc[guild_covered].groupby("diatom_guild")["density_num"].sum()
        row = {
            **{column: meta[column] for column in BASE_SAMPLE_COLUMNS},
            "periphyton_diatom_richness": int(sub.loc[positive & diatom, "original_scientific_name"].nunique()),
            "periphyton_diatom_density_pct": safe_percent(sub.loc[diatom, "density_num"].sum(), total_density),
            "periphyton_diatom_biomass_pct": safe_percent(sub.loc[diatom, "biomass_num"].sum(), total_biomass),
            "periphyton_cyanobacteria_richness": int(sub.loc[positive & cyano, "original_scientific_name"].nunique()),
            "periphyton_cyanobacteria_density_pct": safe_percent(sub.loc[cyano, "density_num"].sum(), total_density),
            "periphyton_cyanobacteria_biomass_pct": safe_percent(sub.loc[cyano, "biomass_num"].sum(), total_biomass),
            "periphyton_diatom_guild_richness": int(guild_density.gt(0).sum()),
            "periphyton_diatom_guild_shannon_density": shannon_from_weights(guild_density),
            "periphyton_diatom_guild_density_coverage_pct": safe_percent(sub.loc[guild_covered, "density_num"].sum(), sub.loc[diatom, "density_num"].sum()),
        }
        for guild in ["low_profile", "high_profile", "motile", "planktic"]:
            flag = sub["diatom_guild"].eq(guild)
            row[f"periphyton_{guild}_diatom_density_pct"] = safe_percent(sub.loc[flag, "density_num"].sum(), total_density)
            row[f"periphyton_{guild}_diatom_biomass_pct"] = safe_percent(sub.loc[flag, "biomass_num"].sum(), total_biomass)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_zooplankton(df: pd.DataFrame, annotation: pd.DataFrame, phytoplankton_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["major_group"]
    data = merge_annotation(df, "zooplankton", annotation, columns)
    data["density_num"] = clean_numeric(data["Density"]).fillna(0).clip(lower=0)
    data["biomass_num"] = clean_numeric(data["Biomass"]).fillna(0).clip(lower=0)
    phyto = phytoplankton_df.copy()
    phyto["biomass_num"] = clean_numeric(phyto["Biomass"]).fillna(0).clip(lower=0)
    phyto_biomass = phyto.groupby("sample_id")["biomass_num"].sum()
    rows = []
    for sample_id, sub in data.groupby("sample_id", sort=False):
        meta = sub.iloc[0]
        total_density = float(sub["density_num"].sum())
        total_biomass = float(sub["biomass_num"].sum())
        positive = sub["density_num"].gt(0) | sub["biomass_num"].gt(0)
        group_density = sub.groupby("major_group")["density_num"].sum()
        group_biomass = sub.groupby("major_group")["biomass_num"].sum()
        row = {**{column: meta[column] for column in BASE_SAMPLE_COLUMNS}}
        for group in ["rotifer", "cladocera", "copepoda", "protozoa"]:
            flag = sub["major_group"].eq(group)
            row[f"zooplankton_{group}_richness"] = int(sub.loc[positive & flag, "original_scientific_name"].nunique())
            row[f"zooplankton_{group}_density_pct"] = safe_percent(sub.loc[flag, "density_num"].sum(), total_density)
            row[f"zooplankton_{group}_biomass_pct"] = safe_percent(sub.loc[flag, "biomass_num"].sum(), total_biomass)
        large_flag = sub["major_group"].isin(["cladocera", "copepoda"])
        row["zooplankton_large_crustacean_biomass_pct"] = safe_percent(sub.loc[large_flag, "biomass_num"].sum(), total_biomass)
        row["zooplankton_major_group_shannon_density"] = shannon_from_weights(group_density)
        row["zooplankton_major_group_shannon_biomass"] = shannon_from_weights(group_biomass)
        denominator = float(phyto_biomass.get(sample_id, np.nan))
        row["zooplankton_phytoplankton_biomass_ratio"] = total_biomass / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
        row["zooplankton_major_group_density_coverage_pct"] = safe_percent(sub.loc[sub["major_group"].ne("unresolved"), "density_num"].sum(), total_density)
        rows.append(row)
    return pd.DataFrame(rows)


def community_matrix(df: pd.DataFrame, value_mode: str) -> pd.DataFrame:
    data = df.copy()
    data["taxon"] = taxon_key(data)
    data = data[data["taxon"].ne("")]
    if value_mode == "count":
        long = data.groupby(["sample_id", "taxon"], as_index=False).size().rename(columns={"size": "value"})
    elif value_mode == "presence":
        long = data.drop_duplicates(["sample_id", "taxon"])[["sample_id", "taxon"]].assign(value=1.0)
    else:
        data["value"] = clean_numeric(data[value_mode]).fillna(0).clip(lower=0)
        long = data.groupby(["sample_id", "taxon"], as_index=False)["value"].sum()
    long = long[long["value"].gt(0)]
    if long.empty:
        return pd.DataFrame()
    return long.pivot(index="sample_id", columns="taxon", values="value").fillna(0).sort_index().sort_index(axis=1)


def bray_curtis_distance(matrix: np.ndarray) -> np.ndarray:
    n_samples = matrix.shape[0]
    result = np.zeros((n_samples, n_samples), dtype=float)
    row_sums = matrix.sum(axis=1)
    for start in range(0, n_samples, 32):
        stop = min(start + 32, n_samples)
        block = matrix[start:stop]
        numerator = np.abs(block[:, None, :] - matrix[None, :, :]).sum(axis=2)
        denominator = row_sums[start:stop, None] + row_sums[None, :]
        result[start:stop] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return result


def jaccard_distance(matrix: np.ndarray) -> np.ndarray:
    binary = matrix > 0
    intersection = binary.astype(np.int32) @ binary.astype(np.int32).T
    counts = binary.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union > 0)
    return 1.0 - similarity


def orient_coordinates(coordinates: np.ndarray) -> np.ndarray:
    oriented = coordinates.copy()
    for axis in range(oriented.shape[1]):
        column = oriented[:, axis]
        if column.size and column[np.argmax(np.abs(column))] < 0:
            oriented[:, axis] *= -1
    return oriented


def pcoa(distance: np.ndarray, n_axes: int = 3) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(1e-12, float(eigenvalues[0]) * 1e-12 if eigenvalues.size else 1e-12)
    positive_values = eigenvalues[positive]
    positive_vectors = eigenvectors[:, positive]
    axes_available = min(n_axes, len(positive_values))
    coordinates = np.full((n, n_axes), np.nan, dtype=float)
    if axes_available:
        coordinates[:, :axes_available] = positive_vectors[:, :axes_available] * np.sqrt(positive_values[:axes_available])
        coordinates[:, :axes_available] = orient_coordinates(coordinates[:, :axes_available])
    explained = np.full(n_axes, np.nan, dtype=float)
    if positive_values.sum() > 0:
        explained[:axes_available] = positive_values[:axes_available] / positive_values.sum()
    negative_abs = float(np.abs(eigenvalues[eigenvalues < 0]).sum())
    total_abs = float(np.abs(eigenvalues).sum())
    diagnostics = {
        "negative_eigenvalue_abs_sum": negative_abs,
        "negative_eigenvalue_abs_fraction": negative_abs / total_abs if total_abs > 0 else 0.0,
        "n_positive_axes": int(positive.sum()),
    }
    return coordinates, explained, diagnostics


def axis_taxon_associations(
    matrix: pd.DataFrame,
    coordinates: np.ndarray,
    matrix_id: str,
    transform: str,
    top_n: int = 10,
) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    transformed = np.log10(values + 1) if transform == "log10p1" else (values > 0).astype(float)
    rows: list[dict[str, object]] = []
    for axis in range(coordinates.shape[1]):
        axis_values = coordinates[:, axis]
        if not np.isfinite(axis_values).all() or np.nanstd(axis_values) == 0:
            continue
        correlations = []
        for taxon_index, taxon in enumerate(matrix.columns):
            taxon_values = transformed[:, taxon_index]
            if np.std(taxon_values) == 0:
                continue
            correlation = float(np.corrcoef(taxon_values, axis_values)[0, 1])
            if np.isfinite(correlation):
                correlations.append((taxon, correlation))
        correlations.sort(key=lambda item: item[1])
        selected = correlations[:top_n] + correlations[-top_n:]
        for taxon, correlation in selected:
            rows.append(
                {
                    "matrix_id": matrix_id,
                    "axis": axis + 1,
                    "direction": "negative" if correlation < 0 else "positive",
                    "taxon": taxon,
                    "axis_taxon_pearson_r": correlation,
                    "association_basis": transform,
                }
            )
    return pd.DataFrame(rows)


def run_ordination(
    matrix: pd.DataFrame,
    matrix_id: str,
    distance_name: str,
    feature_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if matrix.empty:
        summary = pd.DataFrame([{"matrix_id": matrix_id, "status": "skipped_empty"}])
        return pd.DataFrame(), summary, pd.DataFrame()
    positive = matrix.sum(axis=1).gt(0)
    used = matrix.loc[positive]
    varying_taxa = used.columns[used.nunique().gt(1)]
    used = used[varying_taxa]
    if used.shape[0] < 4 or used.shape[1] < 2:
        summary = pd.DataFrame([{"matrix_id": matrix_id, "status": "skipped_insufficient_samples_or_taxa", "n_samples": used.shape[0], "n_taxa": used.shape[1]}])
        return pd.DataFrame(), summary, pd.DataFrame()
    if distance_name == "bray_curtis":
        transformed = np.log10(used.to_numpy(dtype=float) + 1)
        distance = bray_curtis_distance(transformed)
        transform = "log10p1"
    elif distance_name == "jaccard":
        transformed = (used.to_numpy(dtype=float) > 0).astype(float)
        distance = jaccard_distance(transformed)
        transform = "presence_absence"
    else:
        raise ValueError(distance_name)
    coordinates, explained, diagnostics = pcoa(distance, n_axes=3)
    scores = pd.DataFrame({"sample_id": used.index})
    for axis in range(3):
        scores[f"{feature_prefix}_pcoa{axis + 1}"] = coordinates[:, axis]
    summary = pd.DataFrame(
        [
            {
                "matrix_id": matrix_id,
                "status": "computed",
                "distance": distance_name,
                "transform": transform,
                "n_samples": used.shape[0],
                "n_taxa": used.shape[1],
                "axis1_positive_eigenvalue_variance_pct": explained[0] * 100,
                "axis2_positive_eigenvalue_variance_pct": explained[1] * 100,
                "axis3_positive_eigenvalue_variance_pct": explained[2] * 100,
                **diagnostics,
                "axis_orientation_rule": "coordinate with largest absolute value is forced positive",
            }
        ]
    )
    associations = axis_taxon_associations(used, coordinates, matrix_id, transform)
    return scores, summary, associations


def build_ordination_features(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specifications = [
        ("fish_count", sheets["Fish"], "count"),
        ("benthos_density", sheets["Benthos"], "Density"),
        ("macrophytes_density", sheets["Macrophytes"], "Density"),
        ("phytoplankton_density", sheets["Phytoplankton"], "Density"),
        ("periphyton_density", sheets["Periphyton"], "Density"),
        ("zooplankton_density", sheets["Zooplankton"], "Density"),
    ]
    score_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    association_frames: list[pd.DataFrame] = []
    for prefix, data, value_mode in specifications:
        matrix = community_matrix(data, value_mode)
        for distance_name, suffix in [("bray_curtis", "bray"), ("jaccard", "jaccard")]:
            matrix_id = f"{prefix}_{suffix}"
            scores, summary, associations = run_ordination(matrix, matrix_id, distance_name, matrix_id)
            score_frames.append(scores)
            summary_frames.append(summary)
            association_frames.append(associations)
    qualitative = community_matrix(sheets["Benthos_Qualitative"], "presence")
    scores, summary, associations = run_ordination(
        qualitative,
        "benthos_qual_pa_jaccard",
        "jaccard",
        "benthos_qual_pa_jaccard",
    )
    score_frames.append(scores)
    summary_frames.append(summary)
    association_frames.append(associations)

    scores_wide: pd.DataFrame | None = None
    for frame in score_frames:
        if frame.empty:
            continue
        scores_wide = frame if scores_wide is None else scores_wide.merge(frame, on="sample_id", how="outer")
    return (
        scores_wide if scores_wide is not None else pd.DataFrame(columns=["sample_id"]),
        pd.concat(summary_frames, ignore_index=True, sort=False),
        pd.concat([frame for frame in association_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in association_frames) else pd.DataFrame(),
    )


def merge_sample_response_features(
    sample_meta: pd.DataFrame,
    component_frames: list[pd.DataFrame],
    ordination_scores: pd.DataFrame,
) -> pd.DataFrame:
    result = base_sample_frame(sample_meta)
    for frame in component_frames:
        feature_columns = [column for column in frame.columns if column not in BASE_SAMPLE_COLUMNS]
        result = result.merge(frame[["sample_id"] + feature_columns], on="sample_id", how="left")
    result = result.merge(ordination_scores, on="sample_id", how="left")

    feature_columns = [column for column in result.columns if column not in BASE_SAMPLE_COLUMNS]
    continuous_tokens = ["_mean_", "_median_", "_CWM_", "_biomass_weighted_", "_ratio", "_pcoa"]
    for column in feature_columns:
        if any(token in column for token in continuous_tokens):
            continue
        if column.startswith("periphyton_"):
            river = result["water_body_type"].eq("River")
            result.loc[river, column] = result.loc[river, column].fillna(0)
            result.loc[~river, column] = np.nan
        else:
            result[column] = result[column].fillna(0)
    pcoa_columns = [column for column in feature_columns if "_pcoa" in column]
    result.loc[result["water_body_type"].eq("Lake"), [column for column in pcoa_columns if column.startswith("periphyton_")]] = np.nan
    return result


def feature_category(column: str) -> str:
    lower = column.lower()
    if "pcoa" in lower:
        return "community_composition_axis"
    if "coverage" in lower or "non_actinopterygii" in lower:
        return "qa_coverage"
    if "cwm" in lower or "weighted" in lower or "mean_" in lower or "median_" in lower:
        return "continuous_trait_or_size"
    if "shannon" in lower:
        return "functional_or_taxonomic_group_diversity"
    if "richness" in lower:
        return "indicator_or_group_richness"
    if "pct" in lower:
        return "indicator_or_group_proportion"
    if "ratio" in lower:
        return "trophic_coupling_ratio"
    return "biological_response"


def taxon_group_from_feature(column: str) -> str:
    for prefix in ["fish", "benthos", "macrophyte", "phytoplankton", "periphyton", "zooplankton"]:
        if column.startswith(prefix):
            return prefix
    return "multiple"


def build_feature_dictionary(sample_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in sample_features.columns:
        if column in BASE_SAMPLE_COLUMNS:
            continue
        category = feature_category(column)
        model_include = category != "qa_coverage" and column != "fish_observed_taxa_richness"
        if "griis" in column.lower():
            source = "GRIIS-China listed alien/invasive taxa; lower-bound interpretation"
        elif column.startswith("fish_") and any(token in column for token in ["trophic", "max_length", "resilience", "omnivore", "carnivore"]):
            source = "FishBase snapshot 21.04"
        elif "diatom_guild" in column or any(token in column for token in ["low_profile", "high_profile", "motile", "planktic"]):
            source = "conservative Passy/Rimet genus guild rules"
        elif "pcoa" in column:
            source = "sample x taxon matrix; classical PCoA"
        else:
            source = "raw aquatic database taxonomy and measured abundance/biomass/length"
        rows.append(
            {
                "column_name": column,
                "role": "biological_response_extended_feature" if model_include else "biological_response_qa",
                "taxon_group": taxon_group_from_feature(column),
                "feature_category": category,
                "model_include": model_include,
                "source_basis": source,
                "sample_special_rule": "calculated independently for each sample_id",
                "site_special_rule": "arithmetic mean of nonmissing sample-level values by water_body_type + site",
                "missing_rule": "NA retained for undefined continuous traits/ordination; periphyton is NA for Lake; zero denotes no detected applicable indicator",
            }
        )
    return pd.DataFrame(rows)


def build_name_matching_qa(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in master.groupby("taxon_group"):
        total = len(sub)
        counts = sub["match_status"].value_counts()
        rows.append(
            {
                "taxon_group": group,
                "total_taxa": total,
                "species_level_match_pct": safe_percent(sub["accepted_rank"].eq("species").sum(), total),
                "genus_level_match_pct": safe_percent(sub["accepted_rank"].eq("genus").sum(), total),
                "family_level_match_pct": safe_percent(sub["accepted_rank"].eq("family").sum(), total),
                "exact_match_taxa": int(counts.get("exact", 0)),
                "fuzzy_match_taxa": int(counts.get("fuzzy", 0)),
                "unmatched_taxa": int(counts.get("unmatched", 0)),
                "manual_check_taxa": int(sub["manual_check"].sum()),
                "low_confidence_taxa": int(sub["confidence"].eq("low").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_trait_coverage_qa(
    master: pd.DataFrame,
    fish: pd.DataFrame,
    macrophytes: pd.DataFrame,
    phytoplankton: pd.DataFrame,
    periphyton: pd.DataFrame,
    zooplankton: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(scope: str, trait: str, matched: int, total: int, note: str) -> None:
        rows.append(
            {
                "scope": scope,
                "trait_or_annotation": trait,
                "matched_taxa": matched,
                "total_taxa": total,
                "taxon_coverage_pct": safe_percent(matched, total),
                "note": note,
            }
        )

    add("all", "GBIF species-level accepted rank", int(master["accepted_rank"].eq("species").sum()), len(master), "taxon-name normalization only")
    add("fish", "FishBase trophic level", int(fish["fishbase_trophic_level"].notna().sum()), len(fish), "snapshot 21.04")
    add("fish", "FishBase max length", int(fish["fishbase_max_length"].notna().sum()), len(fish), "snapshot 21.04")
    add("fish", "FishBase resilience", int(fish["fishbase_resilience_ordinal"].notna().sum()), len(fish), "ordinal 0-3")
    add("fish", "GRIIS-China listed alien", int(fish["griis_listed_alien_china"].sum()), len(fish), "listed count, not annotation coverage")
    add("macrophytes", "growth form high/medium confidence", int(macrophytes["growth_form_confidence"].isin(["high", "medium"]).sum()), len(macrophytes), "unresolved taxa excluded from functional-group Shannon")
    add("macrophytes", "GRIIS-China listed alien", int(macrophytes["griis_listed_alien_china"].sum()), len(macrophytes), "listed count, not annotation coverage")
    add("phytoplankton", "phylum indicator flags", len(phytoplankton), len(phytoplankton), "local taxonomy")
    add("phytoplankton", "conservative bloom-forming genus flag", int(phytoplankton["bloom_forming_flag"].sum()), len(phytoplankton), "positive flag count; absence is not proof of no bloom potential")
    diatoms = periphyton["diatom_flag"].fillna(False)
    add("periphyton", "diatom ecological guild", int((diatoms & periphyton["diatom_guild"].ne("")).sum()), int(diatoms.sum()), "genus-level conservative guild mapping")
    add("zooplankton", "major taxonomic group", int(zooplankton["major_group"].ne("unresolved").sum()), len(zooplankton), "local phylum classification")

    coverage_columns = [column for column in sample_features.columns if "coverage_pct" in column]
    for column in coverage_columns:
        values = clean_numeric(sample_features[column])
        rows.append(
            {
                "scope": "sample_weighted",
                "trait_or_annotation": column,
                "matched_taxa": np.nan,
                "total_taxa": np.nan,
                "taxon_coverage_pct": float(values.mean()) if values.notna().any() else np.nan,
                "note": f"mean sample-level coverage; min={values.min():.3f}, median={values.median():.3f}, max={values.max():.3f}" if values.notna().any() else "all missing",
            }
        )
    return pd.DataFrame(rows)


def build_planned_feature_status(feature_dictionary: pd.DataFrame) -> pd.DataFrame:
    computed = feature_dictionary[["column_name", "taxon_group", "feature_category", "model_include", "source_basis"]].copy()
    computed["status"] = "computed"
    computed["reason_or_requirement"] = "available from current raw data and documented annotation rules"
    unavailable = pd.DataFrame(
        [
            ["fish_native_submerged_or_native_status", "fish", "trait", False, "GRIIS absence does not prove native status", "not_computable_without_verified_native_range"],
            ["benthos_tolerance_CWM_density", "benthos", "continuous_trait", False, "requires validated regional tolerance scores", "external_trait_required"],
            ["benthos_functional_feeding_group_shannon_density", "benthos", "functional_group", False, "requires taxon-level functional feeding group table", "external_trait_required"],
            ["benthos_collector_gatherer_density_pct", "benthos", "functional_group", False, "requires taxon-level functional feeding group table", "external_trait_required"],
            ["benthos_scraper_density_pct", "benthos", "functional_group", False, "requires taxon-level functional feeding group table", "external_trait_required"],
            ["benthos_predator_density_pct", "benthos", "functional_group", False, "requires taxon-level functional feeding group table", "external_trait_required"],
            ["macrophyte_native_submerged_biomass_pct", "macrophyte", "trait", False, "absence from GRIIS cannot be treated as native", "not_computable_without_verified_native_range"],
            ["macrophyte_CWM_height", "macrophyte", "continuous_trait", False, "requires licensed/manual plant height trait data", "external_trait_required"],
            ["phytoplankton_RFG_richness", "phytoplankton", "functional_group", False, "requires curated Reynolds functional-group assignment", "external_trait_required"],
            ["phytoplankton_RFG_dominant_group", "phytoplankton", "functional_group", False, "requires curated Reynolds functional-group assignment", "external_trait_required"],
            ["phytoplankton_RFG_shannon", "phytoplankton", "functional_group", False, "requires curated Reynolds functional-group assignment", "external_trait_required"],
            ["phytoplankton_MBFG_richness", "phytoplankton", "functional_group", False, "requires taxon morphology/MBFG assignment", "external_trait_required"],
            ["periphyton_tolerant_diatom_density_pct", "periphyton", "indicator_trait", False, "requires licensed or regionally validated OMNIDIA/TDI scores", "external_trait_required"],
            ["zooplankton_body_size_CWM", "zooplankton", "continuous_trait", False, "requires taxon body-size trait table", "external_trait_required"],
            ["zooplankton_filter_feeder_density_pct", "zooplankton", "functional_group", False, "requires feeding-mode trait table", "external_trait_required"],
            ["zooplankton_raptorial_feeder_density_pct", "zooplankton", "functional_group", False, "requires feeding-mode trait table", "external_trait_required"],
            ["multidimensional_FRic_FEve_FDiv_FDis_RaoQ", "multiple", "functional_diversity", False, "second-stage only after sufficient multivariate trait coverage", "deferred_by_method_document"],
            ["NMDS1_NMDS2_stress", "multiple", "ordination", False, "visualization-oriented and requires a proven iterative NMDS engine; not used as ML response", "analysis_only_not_model_input"],
            ["dbRDA_constrained_axes", "multiple", "constrained_ordination", False, "requires an explicitly selected environmental formula and permutation design", "analysis_only_formula_not_specified"],
        ],
        columns=["column_name", "taxon_group", "feature_category", "model_include", "reason_or_requirement", "status"],
    )
    unavailable["source_basis"] = "method requirement audit"
    return pd.concat([computed, unavailable], ignore_index=True, sort=False)


def aggregate_site_features(sample_features: pd.DataFrame, site_meta: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in sample_features.columns if column not in BASE_SAMPLE_COLUMNS]
    means = sample_features.groupby(["water_body_type", "site"], as_index=False)[feature_columns].mean(numeric_only=True)
    return site_meta[SITE_INFO_COLUMNS].merge(means, on=["water_body_type", "site"], how="left")


def expand_ml_matrix(
    source_path: Path,
    scope: str,
    response_features: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excel = pd.ExcelFile(source_path, engine=PY_ENGINE)
    source_matrix = pd.read_excel(excel, sheet_name="Model_Input_Matrix")
    source_dictionary = pd.read_excel(excel, sheet_name="Feature_Dictionary")
    source_merge_qa = pd.read_excel(excel, sheet_name="Merge_QA")
    if "Merge_Module_QA" in excel.sheet_names:
        source_module_qa = pd.read_excel(excel, sheet_name="Merge_Module_QA")
    else:
        source_module_qa = pd.DataFrame(columns=["module", "rows_before", "rows_after", "unmatched_rows"])
    model_features = feature_dictionary.loc[feature_dictionary["model_include"], "column_name"].tolist()
    if scope == "sample":
        keys = ["sample_id"]
        response = response_features[["sample_id"] + model_features]
        duplicate_check = int(response["sample_id"].duplicated().sum())
    else:
        keys = ["water_body_type", "site"]
        response = response_features[keys + model_features]
        duplicate_check = int(response.duplicated(keys).sum())
    expanded = source_matrix.merge(response, on=keys, how="left", validate="one_to_one")
    if len(expanded) != len(source_matrix):
        raise RuntimeError(f"{scope} matrix row count changed during biological response merge")

    new_dictionary = feature_dictionary.loc[feature_dictionary["model_include"]].copy()
    for column in source_dictionary.columns:
        if column not in new_dictionary:
            new_dictionary[column] = np.nan
    new_dictionary = new_dictionary[source_dictionary.columns.tolist() + [column for column in new_dictionary.columns if column not in source_dictionary.columns]]
    expanded_dictionary = pd.concat([source_dictionary, new_dictionary], ignore_index=True, sort=False)
    merge_qa = pd.concat(
        [
            source_merge_qa,
            pd.DataFrame(
                [
                    ["biological_response_extended_scope", f"{scope}_special", "new response module"],
                    ["biological_response_extended_feature_columns", len(model_features), "model_include=True only"],
                    ["biological_response_extended_duplicate_keys", duplicate_check, "should be 0"],
                    ["expanded_matrix_rows", len(expanded), "unchanged from source"],
                    ["expanded_matrix_columns", expanded.shape[1], "source columns plus extended biological responses"],
                ],
                columns=source_merge_qa.columns,
            ),
        ],
        ignore_index=True,
    )
    module_qa = pd.concat(
        [
            source_module_qa,
            pd.DataFrame(
                [
                    {
                        "module": "biological_response_extended",
                        "rows_before": len(source_matrix),
                        "rows_after": len(expanded),
                        "unmatched_rows": int(expanded[model_features].isna().all(axis=1).sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return expanded, expanded_dictionary, merge_qa, module_qa


def source_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["raw_aquatic", str(RAW_AQUATIC), "local", "seven biological sheets; unchanged"],
            ["method_reference", str(METHOD_REFERENCE), "local", "requested target-variable specification"],
            ["GBIF Species API", "https://api.gbif.org/v1/species/match", "GBIF", "accepted-name and rank matching"],
            ["GRIIS-China v1.4", "https://cloud.gbif.org/griis/resource?r=griis-china&v=1.4", "ISSG/GBIF", "alien and invasive list for China, CC-BY 4.0"],
            ["FishBase snapshot 21.04", "https://github.com/ropensci/rfishbase/releases/tag/fb-21.04", "FishBase/rfishbase", "species, ecology and stocks tables"],
            ["Passy 2007", "https://doi.org/10.1016/j.aquabot.2006.09.018", "Aquatic Botany", "diatom guild framework"],
            ["Rimet & Bouchez 2012", "https://doi.org/10.1051/kmae/2012018", "Knowledge and Management of Aquatic Ecosystems", "diatom guild/life-form framework"],
        ],
        columns=["source_id", "path_or_url", "provider", "use"],
    )


def readme_method(scope: str, model_feature_count: int) -> pd.DataFrame:
    rows = [
        ["scope", scope, "Sample Special = one point x time sample; Site Special = one site"],
        ["source_samples", 577, "River 409; Lake 168"],
        ["model_feature_count", model_feature_count, "coverage/QA columns excluded from expanded ML input"],
        ["site_aggregation", "mean of sample-level features", "never pool multi-time raw taxon records"],
        ["indicator_denominator", "total measured density/biomass/records", "zero denominator yields 0 for applicable detection/proportion features"],
        ["continuous_missing", "NA", "undefined CWM, length, ratio or ordination scores remain missing"],
        ["periphyton_scope", "River only", "Lake remains structurally NA"],
        ["fish_small_threshold", "Total length <= 10", "operational threshold; source unit retained from raw data and recorded as configurable"],
        ["GRIIS_interpretation", "listed alien/invasive is positive evidence", "not listed is not proof of native status; estimates are lower bounds"],
        ["trait_policy", "no invented values", "licensed/manual traits are listed as external_trait_required"],
        ["PCoA", "log10(x+1) Bray-Curtis and presence/absence Jaccard", "three axes; deterministic sign orientation; positive-eigenvalue variance reported"],
        ["NMDS_dbRDA", "not added to ML input", "NMDS is visualization-oriented; dbRDA needs a prespecified environmental formula"],
        ["created_date", TODAY, ""],
    ]
    return pd.DataFrame(rows, columns=["item", "value", "note"])


def dataframe_qa(sample_features: pd.DataFrame, site_features: pd.DataFrame, feature_dictionary: pd.DataFrame, ordination_summary: pd.DataFrame) -> pd.DataFrame:
    model_features = feature_dictionary.loc[feature_dictionary["model_include"], "column_name"].tolist()
    all_na_sample = [column for column in model_features if sample_features[column].isna().all()]
    all_na_site = [column for column in model_features if site_features[column].isna().all()]
    return pd.DataFrame(
        [
            ["sample_rows", len(sample_features), "expected 577"],
            ["site_rows", len(site_features), "expected 130"],
            ["duplicate_sample_id", int(sample_features["sample_id"].duplicated().sum()), "expected 0"],
            ["duplicate_site", int(site_features.duplicated(["water_body_type", "site"]).sum()), "expected 0"],
            ["response_columns_total", sample_features.shape[1] - len(BASE_SAMPLE_COLUMNS), "includes QA/coverage"],
            ["model_feature_columns", len(model_features), "merged to expanded ML inputs"],
            ["all_na_model_features_sample", ";".join(all_na_sample), "expected blank"],
            ["all_na_model_features_site", ";".join(all_na_site), "expected blank"],
            ["ordination_matrices_computed", int(ordination_summary["status"].eq("computed").sum()), "expected 13"],
            ["sample_model_missing_cells", int(sample_features[model_features].isna().sum().sum()), "structural/undefined values retained"],
            ["site_model_missing_cells", int(site_features[model_features].isna().sum().sum()), "structural/undefined values retained"],
            ["created_date", TODAY, ""],
        ],
        columns=["check_item", "value", "note"],
    )


def write_csv(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def add_workbook_manifest(
    manifests: list[dict[str, object]],
    output_path: Path,
    sheets: list[tuple[str, pd.DataFrame]],
    csv_directory: Path,
) -> None:
    sheet_specs = []
    for sheet_name, frame in sheets:
        csv_path = write_csv(frame, csv_directory, f"{output_path.stem}__{sheet_name}")
        sheet_specs.append({"name": sheet_name[:31], "csv": str(csv_path)})
    manifests.append({"output": str(output_path), "sheets": sheet_specs})


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_external_sources()

    print("[1/8] Loading source matrices and aquatic biological records", flush=True)
    sample_meta = pd.read_excel(SAMPLE_MATRIX, sheet_name="Model_Input_Matrix", engine=PY_ENGINE)
    site_meta = pd.read_excel(SITE_MATRIX, sheet_name="Model_Input_Matrix", engine=PY_ENGINE)
    sheets = load_sheets()

    print("[2/8] Building taxon master, GBIF matches and external annotations", flush=True)
    master = add_gbif_matches(build_taxon_master(sheets))
    griis = load_griis()
    fish_annotation = build_fish_annotation(master, griis)
    benthos_annotation = build_benthos_annotation(master)
    macrophyte_annotation = build_macrophyte_annotation(master, griis)
    phytoplankton_annotation = build_phytoplankton_annotation(master)
    periphyton_annotation = build_periphyton_annotation(master)
    zooplankton_annotation = build_zooplankton_annotation(master)

    print("[3/8] Calculating sample-level indicator, trait and size responses", flush=True)
    component_frames = [
        summarize_fish(sheets["Fish"], fish_annotation),
        summarize_benthos_quantitative(sheets["Benthos"], benthos_annotation),
        summarize_benthos_qualitative(sheets["Benthos_Qualitative"], benthos_annotation),
        summarize_macrophytes(sheets["Macrophytes"], macrophyte_annotation),
        summarize_phytoplankton(sheets["Phytoplankton"], phytoplankton_annotation),
        summarize_periphyton(sheets["Periphyton"], periphyton_annotation),
        summarize_zooplankton(sheets["Zooplankton"], zooplankton_annotation, sheets["Phytoplankton"]),
    ]

    print("[4/8] Computing Bray-Curtis and Jaccard PCoA axes", flush=True)
    ordination_scores, ordination_summary, ordination_loadings = build_ordination_features(sheets)
    sample_features = merge_sample_response_features(sample_meta, component_frames, ordination_scores)
    feature_dictionary = build_feature_dictionary(sample_features)
    site_features = aggregate_site_features(sample_features, site_meta)

    print("[5/8] Expanding Sample Special and Site Special ML matrices", flush=True)
    sample_expanded, sample_expanded_dictionary, sample_merge_qa, sample_module_qa = expand_ml_matrix(
        SAMPLE_MATRIX, "sample", sample_features, feature_dictionary
    )
    site_expanded, site_expanded_dictionary, site_merge_qa, site_module_qa = expand_ml_matrix(
        SITE_MATRIX, "site", site_features, feature_dictionary
    )

    name_qa = build_name_matching_qa(master)
    coverage_qa = build_trait_coverage_qa(
        master,
        fish_annotation,
        macrophyte_annotation,
        phytoplankton_annotation,
        periphyton_annotation,
        zooplankton_annotation,
        sample_features,
    )
    planned_status = build_planned_feature_status(feature_dictionary)
    qa = dataframe_qa(sample_features, site_features, feature_dictionary, ordination_summary)
    sources = source_inventory()
    model_feature_count = int(feature_dictionary["model_include"].sum())

    annotation_tables = {
        "Taxon_Master": master,
        "Fish_Traits": fish_annotation,
        "Benthos_Traits": benthos_annotation,
        "Macrophyte_Traits": macrophyte_annotation,
        "Phytoplankton_Traits": phytoplankton_annotation,
        "Periphyton_Traits": periphyton_annotation,
        "Zooplankton_Traits": zooplankton_annotation,
    }

    print("[6/8] Preparing traceable workbook manifests", flush=True)
    manifests: list[dict[str, object]] = []
    for scope, output_dir, response_features in [
        ("Sample_Special", SAMPLE_DIR, sample_features),
        ("Site_Special", SITE_DIR, site_features),
    ]:
        annotation_output_dir = output_dir / "Biological_Trait_Annotation"
        annotation_output_dir.mkdir(parents=True, exist_ok=True)
        scope_readme = readme_method(scope, model_feature_count)
        scope_qa = qa.copy()
        scope_qa.insert(0, "matrix_scope", scope)

        process_sheets: list[tuple[str, pd.DataFrame]] = [
            ("README", scope_readme),
            ("Response_Features", response_features),
            ("Feature_Dictionary", feature_dictionary),
            ("Taxon_Master", master),
            *list(annotation_tables.items())[1:],
            ("Name_Matching_QA", name_qa),
            ("Trait_Coverage_QA", coverage_qa),
            ("Ordination_Summary", ordination_summary),
            ("Ordination_Loadings", ordination_loadings),
            ("Planned_Feature_Status", planned_status),
            ("Source_Inventory", sources),
            ("Data_QA", scope_qa),
        ]
        add_workbook_manifest(
            manifests,
            output_dir / f"Feature_Process_Biological_Response_Traits_{scope}.xlsx",
            process_sheets,
            BUILD_DIR / scope / "process",
        )
        add_workbook_manifest(
            manifests,
            output_dir / f"Feature_Database_Biological_Response_Traits_{scope}.xlsx",
            [
                ("README", scope_readme),
                ("Biological_Response", response_features),
                ("Feature_Dictionary", feature_dictionary),
                ("Trait_Coverage_QA", coverage_qa),
                ("Data_QA", scope_qa),
            ],
            BUILD_DIR / scope / "database",
        )

        add_workbook_manifest(
            manifests,
            annotation_output_dir / "Taxon_Master_List.xlsx",
            [("README", scope_readme), ("Taxon_Master", master), ("Source_Inventory", sources)],
            BUILD_DIR / scope / "annotation",
        )
        add_workbook_manifest(
            manifests,
            annotation_output_dir / "Taxon_Name_Matching_QA.xlsx",
            [("README", scope_readme), ("Matching_Summary", name_qa), ("Matching_Detail", master)],
            BUILD_DIR / scope / "annotation",
        )
        for sheet_name, table in list(annotation_tables.items())[1:]:
            file_stem = {
                "Fish_Traits": "Fish_Trait_Annotation",
                "Benthos_Traits": "Benthos_Trait_Annotation",
                "Macrophyte_Traits": "Macrophytes_Trait_Annotation",
                "Phytoplankton_Traits": "Phytoplankton_Trait_Annotation",
                "Periphyton_Traits": "Periphyton_Trait_Annotation",
                "Zooplankton_Traits": "Zooplankton_Trait_Annotation",
            }[sheet_name]
            add_workbook_manifest(
                manifests,
                annotation_output_dir / f"{file_stem}.xlsx",
                [("README", scope_readme), ("Trait_Annotation", table), ("Trait_Coverage_QA", coverage_qa)],
                BUILD_DIR / scope / "annotation",
            )
        add_workbook_manifest(
            manifests,
            annotation_output_dir / "Biological_Response_Features.xlsx",
            [
                ("README", scope_readme),
                ("Response_Features", response_features),
                ("Feature_Dictionary", feature_dictionary),
                ("Planned_Feature_Status", planned_status),
            ],
            BUILD_DIR / scope / "annotation",
        )

    add_workbook_manifest(
        manifests,
        SAMPLE_DIR / "ML_Input_Sample_Special_Feature_Matrix_Expanded_Biological_Response.xlsx",
        [
            ("Model_Input_Matrix", sample_expanded),
            ("Feature_Dictionary", sample_expanded_dictionary),
            ("Merge_QA", sample_merge_qa),
            ("Merge_Module_QA", sample_module_qa),
        ],
        BUILD_DIR / "Sample_Special" / "ml_input",
    )
    add_workbook_manifest(
        manifests,
        SITE_DIR / "ML_Input_Site_Special_Feature_Matrix_Expanded_Biological_Response.xlsx",
        [
            ("Model_Input_Matrix", site_expanded),
            ("Feature_Dictionary", site_expanded_dictionary),
            ("Merge_QA", site_merge_qa),
            ("Merge_Module_QA", site_module_qa),
        ],
        BUILD_DIR / "Site_Special" / "ml_input",
    )

    manifest_path = BUILD_DIR / "workbook_manifest.json"
    manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    calculation_manifest = {
        "created_date": TODAY,
        "sample_rows": len(sample_features),
        "site_rows": len(site_features),
        "total_response_columns": sample_features.shape[1] - len(BASE_SAMPLE_COLUMNS),
        "model_response_columns": model_feature_count,
        "ordination_matrices": len(ordination_summary),
        "workbooks": len(manifests),
        "workbook_manifest": str(manifest_path),
    }
    (ANNOTATION_DIR / "calculation_manifest.json").write_text(
        json.dumps(calculation_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for output_dir in [SAMPLE_DIR, SITE_DIR]:
        reference_copy = output_dir / METHOD_REFERENCE.name
        if not reference_copy.exists():
            shutil.copy2(METHOD_REFERENCE, reference_copy)

    print("[7/8] Calculation and CSV staging completed", flush=True)
    print(json.dumps(calculation_manifest, ensure_ascii=False, indent=2), flush=True)
    print("[8/8] Run the artifact-tool exporter to create formatted XLSX workbooks", flush=True)


if __name__ == "__main__":
    main()
