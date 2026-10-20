# T93_adni_full_end_to_end: ADNI Complete End-to-End Pipeline

## Objective
Run complete ADNI analysis pipeline from raw data to ROI timeseries

## Inputs
Raw ADNI NIfTI data (T1w and rs-fMRI)

## Outputs
Complete adni_output/ with all derivatives and QC reports

## Key Points
- T118: Stage raw ADNI data into BIDS format
- Validate BIDS compliance with bids-validator
- T119: Run fMRIPrep preprocessing on BIDS dataset
- T120: Extract DK68 ROI timeseries from preprocessed BOLD
- Generate integrated QC report spanning all processing stages
- Save final outputs in organized adni_output directory
- Include processing logs documenting all parameters
- Produce summary statistics for data quality assessment
- Final output ready for downstream connectome/network analysis

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, JSON, HTML where applicable)
- BIDS validation reports must show compliance or documented exceptions
- Timeseries CSV files must have proper dimensions (timepoints × ROIs)
- HTML QC reports must be readable and contain meaningful diagnostic information
- Processing logs must document all key parameters and software versions
- No critical errors during processing
