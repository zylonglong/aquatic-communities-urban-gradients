# Inputs and software environment

## Analytical input mapping

The supplied workbook indices and column dictionaries specify the processed inputs for each figure and model. Supplementary Data 1 contains regional survey/event and site matrices, definitions of the biological responses, environmental scores, and component weights. Supplementary Data 2 contains the global biological/economic model inputs and fitted outputs. Supplementary Data 3 contains the water-quality station-year archive used in the global comparison. Supplementary Data 4 contains the regional multiscale associations, comparison tables, resampling and sensitivity outputs. Figure-specific workbooks map panels to those sources.

Original field observations, taxonomic harmonisation, water-quality report transcription and spatial extraction steps precede the publication matrices. The archived configurations identify original file locations. Recreate those paths or edit them before running the archived pipeline. The original public raster/vector files and complete source database distributions are obtained from the sources in THIRD_PARTY_DATA.md and the manuscript Supplementary Notes; this release does not contain every upstream source file.

## Entry points

Regional analysis: code/regional_source/01_run_analysis.py, 08_rerun_taxon_family.py and the ch5_*.py modules. Global analysis: prepare_global_macro_data.py, fit_global_macro_models.py, prepare_recent_gdp_update.py, fit_recent_gdp_models.py and fit_density_balanced_trends.py under code/global_source/08_脚本. Plotting sources remain under their archived original directories. Earlier plotting dependencies are retained where imported by later figure scripts. Script and cache filenames may use the original figure numbers.

## Environment

The manuscript reports Python 3.12.0, pandas 3.0.3, NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2, statsmodels 0.14.4, Matplotlib 3.8.0 and seaborn 0.13.0. These are the author-reported versions, not a newly tested locked environment. Additional imports across the archive include geopandas, rasterio, h5py, patsy, Pillow, lxml, pypdf, pypdfium2, tifffile and python-docx. Those optional packages are needed only by the scripts that import them.

The portable data-verification entry requires Python 3 and openpyxl only. It was run successfully against the four publication workbooks. It is a check of supplied values, not a substitute for re-running the original analysis.
