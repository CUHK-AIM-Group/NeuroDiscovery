# Benchmark Test Case 45: HCP Full Functional Pipeline

## Task Description

Load local BOLD data (task or resting-state) and run complete HCP functional pipeline with ICA-FIX.

## Input Requirement

Required input(s):

- Local BOLD image (task or resting-state, required)
- Either prior HCP structural outputs for the same subject, or local structural MRI required to generate them:
  - existing HCP structural outputs under the same subject directory, or
  - local T1w image and local T2w image
- Metadata required by HCP functional preprocessing:
  - TR
  - phase-encoding direction
  - dwell time / effective echo spacing
- ICA-FIX installation / classifier resources required to run FIX

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T45_hcppipeline_full_functional/
- Long-running processing is allowed to run as a background job.
- Use the canonical narrow mainline:
  - structural prerequisites if needed
  - `fMRIVolume`
  - `fMRISurface`
  - `ICA-FIX`
- Do not replace the HCP functional pipeline with an alternative non-HCP workflow.

## Expected Output

Expected output artifact(s):

- fMRIVolume + fMRISurface + ICA-FIX outputs
- logs for each completed stage

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
