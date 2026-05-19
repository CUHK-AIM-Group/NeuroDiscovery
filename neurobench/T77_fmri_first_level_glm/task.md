# T77_fmri_first_level_glm: fMRI First-Level Task GLM

## Objective
Run FSL FEAT first-level GLM analysis on preprocessed BOLD

## Inputs
Preprocessed BOLD data, task timing files in EV/3-column format, and writable output location

## Outputs
First-level FEAT outputs including the `.feat` directory, Z-stat maps, cope files, and contrast statistics

If the preprocessed BOLD input or EV timing files are missing, explicitly report `Missing required input` instead of changing the task to another analysis type.

## Key Points
- Design FEAT analysis with task conditions
- Keep the default benchmark mainline on first-level task GLM for the provided preprocessed BOLD data
- Specify contrasts and regressors from timing files
- Include temporal derivatives for better estimation
- Add motion confounds as nuisance regressors
- Set temporal filtering and smoothing parameters
- Run FILM with AUTOCORR prewhitening
- Generate Z-stat and F-stat maps for each contrast
- Output cope and varcope images
- Produce activation clusters and statistical summary
- Do not switch to connectivity analysis, resting-state workflows, or unrelated downstream summaries

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for FEAT/GLM deliverables (for example NIfTI maps and FEAT outputs)
- Statistical maps must contain valid numerical data with proper dimensions
- No errors during processing, comprehensive logs generated
