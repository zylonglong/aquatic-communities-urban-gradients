# Reproduction status

## Portable entry points

The demo in code/demo_water_quality_correlations.py recomputes 78 regional Spearman correlations from Supplementary Data 1 and compares them with Supplementary Data 4. This is an input-based calculation of the reported coefficients. It does not rerun the 4,999 permutations, bootstrap intervals, biological family tests, global models or every figure.

The separate code/verify_submission_data.py utility checks 16 publication-workbook counts and reported values. It reads supplied outputs and is not a full model refit.

## Archived sources

The sources under regional_source and global_source preserve the original implementation. Their configurations contain original Windows paths and expected input layouts. The upstream feature and water-quality sources are also included because the older software archive omitted them. Adding their sources does not establish that their full upstream dependencies are satisfied.

Before a complete rerun: configure paths, place the original analytical inputs, resolve upstream dataset versions and licence terms, install the scientific environment, and run the relevant stages in a clean directory. The native analysis workbooks and three feature dictionaries are distributed with the dataset supplement, rather than the software tree. Historical comparison branches in ch5_primary_analysis.py also read original baseline CSV files, which are diagnostic comparisons rather than additional results claimed in the paper. Those branches must be separated from a public analysis entry or their dependencies explicitly supplied.

The environment inventory records packages found in the original local virtual environment on 4 November 2026. That environment inherits system packages, so this inventory is not a complete installation lockfile and does not establish that every dependency is present on a new computer.

The complete original project has not been rerun from a clean download as part of this archive preparation. Do not describe it as an end-to-end validated reproduction package until that check is completed.
