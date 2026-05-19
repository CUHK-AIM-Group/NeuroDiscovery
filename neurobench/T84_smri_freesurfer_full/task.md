# T84_smri_freesurfer_full: Structural MRI FreeSurfer Cortical Reconstruction

## Objective
Run FreeSurfer recon-all complete pipeline on T1w (and optional T2w) NIfTI

## Inputs
T1w NIfTI file (and optional T2w for improved pial surface)

## Outputs
Cortical surface meshes, segmentation, thickness maps, and statistics tables

## Key Points
- Execute recon-all -all with parallel processing
- Optional: Include -T2pial flag if T2w available
- Generate cortical surface meshes (white matter and pial surfaces)
- Compute cortical thickness maps
- Perform automatic subcortical segmentation (aseg)
- Register to fsaverage template (spherical registration)
- Generate aparc atlases (Desikan-Killiany, aparc+aseg)
- Extract cortical and subcortical statistics tables (.stats files)
- Output QC images and summary stats

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, statistics files)
- Structural maps must have proper dimensions and valid data ranges
- CSV files must contain headers and valid numerical data
- No errors during processing, comprehensive logs generated
