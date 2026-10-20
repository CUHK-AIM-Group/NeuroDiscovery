# T88_smri_roi_features: Structural MRI ROI Morphological Features

## Objective
Extract ROI-wise morphological features from structural derivatives

## Inputs
FreeSurfer stats files or structural NIfTI maps (thickness, GM probability, etc.)

## Outputs
roi_stats.csv with morphological metrics per ROI

## Key Points
- Parse FreeSurfer .stats files or extract from volumetric maps
- Extract cortical thickness per ROI
- Extract cortical surface area per ROI
- Extract GM volume per ROI
- Extract GM probability maps per ROI
- Compute mean and standard deviation for each metric
- Support multiple atlases (aparc, aparc.a2009s, etc.)
- Output CSV with columns: roi_label, roi_name, thickness, area, gm_volume, gm_prob_mean
- Include metadata for atlas and subject identification

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, statistics files)
- Structural maps must have proper dimensions and valid data ranges
- CSV files must contain headers and valid numerical data
- No errors during processing, comprehensive logs generated
