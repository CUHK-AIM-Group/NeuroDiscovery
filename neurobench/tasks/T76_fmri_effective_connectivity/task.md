# T76_fmri_effective_connectivity: fMRI Effective Connectivity (PPI/DCM)

## Objective
Compute effective connectivity using PPI, gPPI, or DCM

## Inputs
Preprocessed BOLD and task design/seed ROI info

## Outputs
PPI maps, causal matrices, or DCM parameters

## Key Points
- Extract seed ROI timeseries
- Default mainline: use one method only, not multiple methods in the same answer.
- Preferred default:
	- use gPPI when multiple task conditions are present;
	- use standard PPI when there is a single psychological manipulation;
	- use DCM only when the task explicitly requires small-network causal model estimation.
- Do not provide parallel full implementations for PPI, gPPI, and DCM in one solution unless the prompt explicitly asks for comparison across methods.
- Generate statistical maps for connectivity changes
- Compute interaction effects between task and connectivity
- Save results in standard formats (NIfTI, CSV, MAT)
- Generate connectivity parameter estimates and t-statistics

## Missing Required Input

If required inputs are absent, return:

- `Missing required input`

Minimum required inputs for a concrete runnable solution:

- preprocessed BOLD data
- task timing/design information
- at least one seed ROI definition

## Output Requirement

Save artifacts to:

- `benchmark_results/T76_fmri_effective_connectivity/`

Expected outputs should match the selected single mainline method:

- PPI or gPPI: statistical maps and parameter files
- DCM: estimated DCM parameter/model files

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TSV, NPZ where applicable)
- Statistical maps must contain valid numerical data with proper dimensions
- If a connectivity matrix is produced, it must be numerically valid for the chosen method; do not assume symmetry or [-1, 1] bounds for all effective-connectivity methods.
- No errors during processing, comprehensive logs generated
