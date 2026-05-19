# T63_dcm2nii_conversion: DICOM to NIfTI Conversion

## Objective
Batch convert DICOM folder to NIfTI format with JSON sidecars

## Inputs
DICOM files (organized by acquisition or flat structure)

## Outputs
.nii.gz files and BIDS-style JSON metadata files

## Key Points
- Batch process all DICOM files in input directory
- Preserve acquisition geometry and metadata
- Auto-detect sequence type (T1w, T2w, FLAIR, DWI, fMRI, etc.)
- Extract and save BIDS metadata in JSON sidecars
- Handle multi-echo and multi-series acquisitions
- Output organized directory with clear file naming
- Generate conversion log with statistics

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
