# T80_fmri_mni152_alignment: fMRI MNI152 Standard Space Alignment

## Objective
Complete atlas-based alignment of preprocessed BOLD to MNI152

## Inputs
Preprocessed BOLD data, anatomical reference, and writable output location

## Outputs
MNI152-normalized BOLD image and the transforms needed to reproduce the alignment

If the required BOLD or anatomical inputs are missing, explicitly report `Missing required input` instead of assuming existing transforms or changing the task goal.

## Key Points
- Register T1w anatomical to MNI152 template
- Generate and apply normalization transformation to BOLD
- Use a task-faithful registration chain such as BOLD-to-anatomical followed by anatomical-to-MNI when those transforms are not already provided
- If forward transforms already exist, they may be reused, but do not assume they exist without stating that prerequisite
- Output normalized BOLD in standard space
- Save the forward transforms used for the normalization
- Additional QC visualizations are optional and should not replace the core alignment deliverable

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for alignment deliverables (for example normalized NIfTI and transform files)
- Statistical maps must contain valid numerical data with proper dimensions
- No errors during processing, comprehensive logs generated
