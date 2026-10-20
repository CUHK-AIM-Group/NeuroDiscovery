# T90_adni_bids_staging: ADNI BIDS Data Staging

## Objective
Automatically organize raw ADNI NIfTI data into BIDS-compliant format

## Inputs
Raw ADNI NIfTI data (T1w, rs-fMRI) in native directory structure

## Outputs
BIDS-compliant ADNI dataset with validation report

## Key Points
- Load ADNI raw NIfTI files (T1w and rs-fMRI)
- Map ADNI subject IDs to BIDS format (sub-ADNIXXXXX)
- Create proper BIDS directory structure: sub-*/ses-M00/anat/ and func/
- Rename files to BIDS standard: sub-*_ses-*_T1w.nii.gz, sub-*_ses-*_bold.nii.gz
- Generate JSON sidecars with MRI acquisition metadata
- Create dataset_description.json documenting ADNI dataset
- Generate README with ADNI-specific information
- Run bids-validator to ensure BIDS compliance
- Output validation report documenting any warnings/errors

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, JSON, HTML where applicable)
- BIDS validation reports must show compliance or documented exceptions
- Timeseries CSV files must have proper dimensions (timepoints × ROIs)
- HTML QC reports must be readable and contain meaningful diagnostic information
- Processing logs must document all key parameters and software versions
- No critical errors during processing
