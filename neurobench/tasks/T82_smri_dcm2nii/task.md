# T82_smri_dcm2nii: Structural MRI DICOM to NIfTI Conversion

## Objective
Batch convert structural MRI DICOM sequences to NIfTI format with metadata

## Inputs
DICOM files from structural MRI acquisitions

## Outputs
T1w/T2w/FLAIR .nii.gz files and JSON metadata sidecars

## Key Points
- Batch process all DICOM files
- Auto-detect sequence type (T1w, T2w, FLAIR, Proton Density)
- Extract and preserve MRI acquisition parameters in JSON
- Handle multi-echo and multi-series acquisitions
- Output organized directory with clear naming
- Validate converted NIfTI file integrity
- Generate conversion log with statistics

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, statistics files)
- Structural maps must have proper dimensions and valid data ranges
- CSV files must contain headers and valid numerical data
- No errors during processing, comprehensive logs generated
