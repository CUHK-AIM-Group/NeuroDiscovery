# T79_fmri_adni_rsfmri_full_pipeline: fMRI ADNI-Style Resting-State Pipeline

## Objective
End-to-end ADNI-like resting-state fMRI analysis from raw data

## Inputs
Raw resting-state fMRI inputs (T1w + resting-state BOLD) in BIDS or local native format, plus writable output location

## Outputs
Complete fmri_output/ containing preprocessing derivatives and resting-state connectivity outputs matched to the provided inputs

If required resting-state inputs are missing, explicitly report `Missing required input` instead of switching to a different analysis goal.

## Key Points
- T90/T91: Organize data into BIDS format
- T92: Run fMRIPrep anatomical and functional preprocessing
- Keep the default benchmark mainline on single-dataset resting-state preprocessing, denoising, ROI time-series extraction, and connectivity estimation
- Use XCP-D, Nilearn, or equivalent denoising/connectivity tooling when appropriate, but do not expand the task into a broader multimodal or group-template workflow unless the inputs explicitly support it
- Do not change the goal to task-fMRI GLM, HCP-style multimodal analysis, or unrelated downstream branches
- Group-level templates and extended RSN analyses are optional and should not replace the required single-dataset resting-state outputs
- Produce QC/log outputs that document the chosen preprocessing and denoising steps
- Save final outputs in fmri_output directory

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for resting-state pipeline deliverables (for example NIfTI, CSV, TSV, or NPZ where applicable)
- Connectivity matrices must be symmetric and valid (values between -1 and 1)
- No errors during processing, comprehensive logs generated
