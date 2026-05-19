# T64_fmriprep_preprocessing: fMRI fMRIPrep Preprocessing

## Objective
Run standardized fMRIPrep preprocessing on BIDS fMRI dataset

## Inputs
BIDS fMRI dataset with anatomical T1w/T2w

## Outputs
Preprocessed BOLD, anatomical derivatives, and HTML QC report

## Key Points
- Execute fMRIPrep with recommended parameters
- Perform anatomical segmentation and surface reconstruction
- Perform functional distortion correction and realignment
- Output preprocessed BOLD in MNI152 and native space
- Generate comprehensive HTML report for visual QC
- Save confound regressors for downstream analysis
- The benchmark mainline is a standard participant-level fMRIPrep run on an already prepared BIDS dataset
- Keep the answer focused on the direct fMRIPrep invocation and expected derivatives rather than expanding into multiple container or platform deployment variants unless the task explicitly requires that distinction
- If the required BIDS dataset or required anatomical images are missing, report `Missing required input`
- Do not broaden this task into downstream first-level GLM, connectivity analysis, or unrelated post-processing

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Preprocessed derivatives and confound outputs must be internally consistent with a standard fMRIPrep run
- No errors during processing, proper logs generated
