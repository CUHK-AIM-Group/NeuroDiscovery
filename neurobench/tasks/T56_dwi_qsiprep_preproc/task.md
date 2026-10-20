# T56_dwi_qsiprep_preproc: DWI QSIPrep Preprocessing

## Objective
Run QSIPrep best-practice preprocessing on BIDS DWI dataset

## Inputs
BIDS-organized DWI dataset with anatomical T1w

## Outputs
preproc_dwi.nii.gz, QC HTML report, and preprocessing derivatives

## Key Points
- Execute QSIPrep with recommended parameters
- Perform head motion correction and distortion correction
- Output preprocessed DWI in MNI space (optionally native space)
- Generate HTML QC report for visual inspection
- Save JSON sidecars documenting preprocessing steps

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TCK, VTK where applicable)
- CSV files must contain headers and valid numerical data
- No errors or incomplete computations during processing
