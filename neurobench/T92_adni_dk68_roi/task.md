# T92_adni_dk68_roi: ADNI DK68 ROI Timeseries Extraction

## Objective
Extract Desikan-Killiany 68 ROI timeseries from fMRIPrep output

## Inputs
fMRIPrep preprocessed BOLD data with brain mask

## Outputs
DK68 ROI timeseries CSV and QC metrics

## Key Points
- Load fMRIPrep preprocessed BOLD data
- Automatically detect TR from NIfTI header
- Apply denoising (optional: remove outlier frames)
- Apply bandpass filtering (0.01-0.08 Hz)
- Register DK68 atlas to functional space
- Extract mean BOLD timeseries for each of 68 ROIs
- Apply z-score standardization to timeseries
- Compute QC metrics (mean framewise displacement, ghost-to-signal ratio)
- Output CSV file: rows=timepoints, columns=68 ROIs
- Include ROI labels and metadata

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, JSON, HTML where applicable)
- BIDS validation reports must show compliance or documented exceptions
- Timeseries CSV files must have proper dimensions (timepoints × ROIs)
- HTML QC reports must be readable and contain meaningful diagnostic information
- Processing logs must document all key parameters and software versions
- No critical errors during processing
