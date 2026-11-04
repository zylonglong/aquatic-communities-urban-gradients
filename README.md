# Analysis code for Aquatic communities and water quality encode urban environmental gradients at contrasting spatial scales

Custom data-processing, statistical-analysis and plotting sources for the associated manuscript. The regional study covers 564 site-events at 130 sites, six biological groups in rivers and five in lakes, and 13 spatial ranges from 200 to 6,000 m.

## Data

Scientific data are prepared as a separate Zenodo dataset, version 1.0.0. Its reserved DOI is [10.5281/zenodo.23942296](https://doi.org/10.5281/zenodo.23942296); the record remains a draft at this preparation stage, so the DOI does not yet establish public data availability. The dataset contains the four Supplementary Data workbooks, seven figure source-data workbooks, and Regional_Input_Data.zip (survey records, taxon annotations, native analysis inputs and the inferential water-quality monthly table). After publication, download the workbooks to one directory for the demo below and extract the regional ZIP for upstream input provenance. This software repository does not mirror the third-party raw spatial databases.

## Try the portable demo

Install Python 3 and openpyxl, then run:

```text
python -m pip install -r requirements-demo.txt
python code/demo_water_quality_correlations.py --data-dir /path/to/supplementary_data --output demo_results.json
```

The demo recalculates the 78 regional water-quality Spearman coefficients directly from analytical inputs, then compares them with the publication results. Its scope is described in docs/reproduction_status.md. Expected results are in examples/expected_demo_results.json.

For the separate 16-check publication-table verifier:

```text
python code/verify_submission_data.py --data-dir /path/to/supplementary_data --output verification_results.json
```

## Original analysis and preprocessing

- code/regional_source: final regional wrappers and their original dependencies. The original regional entry is 01_run_analysis.py. Configuration is in ch5_config.py; core computations are in ch5_primary_core.py and the related modules.
- code/global_source: global data preparation, adjusted models and trend fitting.
- code/upstream_features: sources for the regional site matrix, biological response traits and terrain extraction.
- code/upstream_water_quality: sources for spatial features, water-quality input preparation and associated diagnostics.

Archived scripts require path and input configuration. They are preserved source files, not a claim of a verified one-command reconstruction. The portable demo above works directly with the publication workbooks. See docs/reproduction_status.md before running archived stages.

## Figure numbering

The manuscript uses main Figs. 1-6 and Extended Data Fig. 1. Archived scripts may retain original names: original Fig. 1 became Extended Data Fig. 1, and original Figs. 2-7 became main Figs. 1-6. Supplementary Data indices use submission numbering.

## Citation and reuse

The archive creator is Yu Zhang (family name Zhang, given name Yu), as confirmed by the depositor. Creator metadata and version 1.0.0 are in CITATION.cff and .zenodo.json. No release DOI is claimed yet. Author-owned code is MIT licensed. Third-party data retain their original terms, documented in THIRD_PARTY_DATA.md, including GLORICH CC BY-NC-SA 4.0 and OpenStreetMap ODbL 1.0.

Repository preparation status: sources and portable entry points are being prepared for review. A GitHub Release and Zenodo software DOI will follow verification of the intended release contents.
