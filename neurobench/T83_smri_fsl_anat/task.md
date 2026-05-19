# T83_smri_fsl_anat: Structural MRI FSL Anatomical Preprocessing

## Objective
Run FSL fsl_anat one-click structural preprocessing on T1w NIfTI

## Inputs
T1w anatomical NIfTI file

## Outputs
Brain extraction, tissue segmentation, MNI152 registration, and QC report

## Key Points
- Execute fsl_anat with standard settings
- Perform brain extraction using BET
- Run tissue segmentation (FAST): GM, WM, CSF probability maps
- Compute bias field correction
- Register to MNI152 template (FLIRT + optional FNIRT)
- Generate brain mask and tissue masks
- Output transformation matrices (native to MNI)
- Include QC report with visual checks and statistics

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, statistics files)
- Structural maps must have proper dimensions and valid data ranges
- CSV files must contain headers and valid numerical data
- No errors during processing, comprehensive logs generated
