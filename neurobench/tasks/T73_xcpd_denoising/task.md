# T73_xcpd_denoising: fMRI XCP-D Denoising

## Objective
Run XCP-D post-processing denoising on fMRIPrep derivatives

## Inputs
fMRIPrep output BOLD data with confound regressors

## Outputs
Denoised BOLD, ROI timeseries, and QC report

## Key Points
- Load fMRIPrep preprocessed BOLD and confound regressors
- 36-parameter nuisance regression (motion, CompCor, CSF, WM)
- Bandpass filtering (0.01-0.08 Hz)
- Despike procedures for outliers
- Motion scrubbing with FD threshold (>0.2 mm)
- Spatial smoothing (6mm FWHM)
- Generate denoised BOLD in MNI152 and native space
- Extract time series and compute network metrics
- Produce comprehensive QC report with motion traces

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TSV, NPZ where applicable)
- Statistical maps must contain valid numerical data with proper dimensions
- Connectivity matrices must be symmetric and valid (values between -1 and 1)
- No errors during processing, comprehensive logs generated
