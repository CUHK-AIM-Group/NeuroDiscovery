# T86_smri_fmriprep_anat: Structural MRI fMRIPrep Anatomical Processing

## Objective
Run fMRIPrep anatomical-only mode on BIDS structural dataset

## Inputs
BIDS structural MRI dataset

## Outputs
Standardized anatomical derivatives and HTML QC report

## Key Points
- Execute fMRIPrep with --anat-only flag
- Perform T1w/T2w preprocessing and segmentation
- Generate brain masks and tissue probability maps
- Perform registration to MNI152 template
- Generate surface reconstruction derivatives
- Compute individual template spaces
- Output normalized anatomical maps in MNI and native space
- Produce comprehensive HTML QC report
- The benchmark mainline is a standard participant-level fMRIPrep run restricted to anatomical processing only
- Keep the answer focused on the direct `--anat-only` workflow rather than expanding into multiple container/runtime variants unless the task explicitly requires environment-specific alternatives
- If the required BIDS structural dataset or required anatomical images are missing, report `Missing required input`
- Do not broaden this task into functional preprocessing, downstream morphometry analysis, or unrelated post-processing

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for anatomical fMRIPrep derivatives (for example NIfTI, transform files, segmentation outputs, and HTML reports)
- Structural maps must have proper dimensions and valid data ranges
- No errors during processing, comprehensive logs generated
