# T68_fsl_feat_glm: FSL FEAT Task-based GLM

## Objective
Run FEAT first-level task GLM analysis on preprocessed fMRI

## Inputs
Preprocessed BOLD data, task design file (EV/3col format)

## Outputs
Z-stat maps, cope files, and statistical results

## Key Points
- Set up FEAT analysis with proper design specifications
- Specify contrasts and regressors from task design
- Include confound regressors (motion, etc.)
- Apply temporal smoothing and high-pass filtering
- Run GLM using FILM with AUTOCORR prewhitening
- Generate Z-statistic maps for each contrast
- Produce cope and varcope files
- The benchmark mainline is a single first-level FEAT analysis on already preprocessed BOLD data
- Use the provided preprocessed BOLD image and task design inputs directly; do not expand backward into full preprocessing or unrelated workflow setup
- If required FEAT inputs such as the preprocessed BOLD image or EV/design file are missing, report `Missing required input`
- A concise FEAT setup is preferred over a broad template-management framework unless the task explicitly provides a reusable `.fsf` template as an input
- The expected result is the `.feat` output directory and its core first-level statistical maps, not a broad end-to-end fMRI pipeline

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format for FEAT first-level analysis (for example `.feat`, z-stat maps, cope/varcope, design files, and related reports)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
