# T66_freesurfer_reconall: FreeSurfer Cortical Reconstruction

## Objective
Run FreeSurfer recon-all full pipeline on T1w NIfTI

## Inputs
T1w anatomical NIfTI file

## Outputs
Surface reconstruction, segmentation, and thickness statistics

## Key Points
- Execute recon-all with parallel processing
- Generate cortical surface meshes (pial, white matter)
- Compute cortical thickness maps
- Perform automatic subcortical segmentation
- Register to fsaverage template
- Generate aparc, aparc+aseg atlases
- Extract cortical and subcortical statistics tables
- Output QC images for inspection

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
