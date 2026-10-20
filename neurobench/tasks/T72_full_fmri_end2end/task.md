# T72_full_fmri_end2end: Complete End-to-End fMRI Analysis Pipeline

## Objective
Run complete fMRI analysis: BIDS organization → preprocessing → analysis

## Inputs
BIDS fMRI dataset with task design information or clearly identified missing analysis-critical metadata that must be requested from the user.

## Outputs
Complete fmri_output/ with all derivatives and QC reports, plus analysis outputs that are explicitly matched to the dataset type and available metadata.

## Key Points
- T90/T91: Organize data into BIDS format
- T92 or T93: Run fMRIPrep or HCP preprocessing
- T96 or T97: Perform task-based GLM or connectivity analysis, but first decide which branch is actually justified by the task contract and available inputs
- Generate summary statistics and activation maps
- Create group-level results (if multiple subjects)
- Production HTML QC report documenting all steps
- Save intermediate files for reproducibility
- Log all processing parameters and versions
- Final: fmri_output with organized final results
- If events/contrasts indicate a task-fMRI dataset, the default mainline should stay on the GLM/statistical-analysis path rather than switching to a connectome workflow just because a connectivity script is available
- If the necessary inputs for the chosen analysis branch are missing, explicitly identify the missing files or metadata and provide the most task-faithful fallback rather than changing the analysis goal silently
- The answer should make the branch decision explicit: task-GLM, resting-state connectivity, or both only when the task contract genuinely supports both

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
- Higher-quality answers keep the analysis branch aligned with the stated task contract instead of defaulting to whichever downstream example is easiest to reuse
