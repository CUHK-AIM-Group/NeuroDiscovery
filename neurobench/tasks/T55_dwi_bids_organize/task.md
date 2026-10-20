# T55_dwi_bids_organize: DWI BIDS Organization

## Objective
Organize local DWI NIfTI/bval/bvec files into BIDS-compliant dataset structure

## Inputs
Local DWI NIfTI, bval, and bvec files (raw format) and writable output location

## Outputs
BIDS-compliant DWI structure with proper directory layout and file naming

If the required DWI NIfTI/bval/bvec inputs are missing, explicitly report `Missing required input` instead of changing to another task.

## Key Points
- Create BIDS directory structure (sub-*/ses-*/dwi/)
- Rename files to BIDS standard: sub-*_ses-*_dwi.nii.gz, .bval, .bvec
- Generate dataset_description.json and README
- Include JSON sidecars with DWI metadata (EchoTime, RepetitionTime, etc.)
- Keep the task focused on DWI BIDS organization rather than downstream preprocessing or tractography
- Validation logs are helpful when available, but the required deliverable is the organized BIDS dataset itself

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for DWI BIDS organization deliverables (for example NIfTI, bval, bvec, and JSON sidecars where applicable)
- No errors or incomplete computations during processing
