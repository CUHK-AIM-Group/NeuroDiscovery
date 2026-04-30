# T69_conn_connectivity: CONN Functional/Effective Connectivity Analysis

## Objective
Run CONN Toolbox advanced connectivity analyses on preprocessed BOLD

## Inputs
Preprocessed BOLD data with ROI definitions

## Outputs
Connectivity matrices (ROI-to-ROI, seed-to-voxel) and statistical maps

## Key Points
- Required inputs are:
	- preprocessed BOLD
	- ROI or atlas definitions
	- writable output directory
- If required inputs are missing, return:
	- Missing required input
- Default benchmark-facing mainline:
	- load preprocessed BOLD into CONN
	- compute ROI-to-ROI correlation matrices
	- compute seed-to-voxel connectivity when seeds are provided
	- write outputs to a deterministic benchmark output directory
- Do not require DCM, gPPI, broad MATLAB project scaffolding, or multi-stage environment orchestration unless the task explicitly provides task-design inputs that require them.
- Reserve PPI/gPPI for tasks with explicit conditions/onsets.
- Generate connectivity matrices and statistical maps in standard formats (NIfTI, CSV, MAT)
- Keep the response focused on direct connectivity analysis from already preprocessed BOLD rather than broader preprocessing or multimodal expansion.

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
