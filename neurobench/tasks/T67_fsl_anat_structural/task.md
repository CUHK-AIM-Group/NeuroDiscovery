# T67_fsl_anat_structural: FSL Anatomical Structural Preprocessing

## Objective
Run FSL fsl_anat one-click structural preprocessing on T1w

## Inputs
T1w anatomical NIfTI file

## Outputs
Brain extraction, tissue segmentation, and MNI152 registration results

## Key Points
- Execute fsl_anat with standard settings
- Perform brain extraction using BET
- Run tissue segmentation (FAST)
- Compute bias field correction
- Register to MNI152 template (FLIRT + optional FNIRT)
- Generate brain mask and tissue masks
- Output transformation matrices
- Include QC report with visual checks

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
