# T81_smri_bids_organize: Structural MRI BIDS Organization

## Objective
Automatically organize local DICOM/NIfTI structural MRI data into BIDS-compliant format

## Inputs
Raw local structural MRI DICOM or NIfTI files (T1w, T2w, FLAIR) and writable output location

## Outputs
BIDS-compliant structural MRI dataset rooted at the requested output directory

If no structural MRI inputs are available, explicitly report `Missing required input` instead of switching to another modality or analysis task.

## Key Points
- Convert DICOM to NIfTI if needed
- Create BIDS directory structure (sub-*/ses-*/anat/)
- Rename files to BIDS standard: sub-*_ses-*_T1w.nii.gz, _T2w.nii.gz, _FLAIR.nii.gz
- Generate JSON sidecars with MRI acquisition parameters
- Create the required dataset-level metadata files needed for a valid BIDS dataset
- Handle multiple sequences (T1w, T2w, FLAIR) per subject
- Keep the task focused on structural MRI BIDS organization; do not introduce unrelated EEG or non-structural workflows
- Validation logs are helpful when available, but the required deliverable is the organized BIDS dataset itself
- Log conversion statistics and warnings when they are available from the chosen tooling

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for BIDS structural organization deliverables (for example NIfTI and JSON sidecars where applicable)
- No errors during processing, comprehensive logs generated
