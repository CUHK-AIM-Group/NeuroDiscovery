# Benchmark Test Case 47: HCP Full Multimodal Preprocessing

## Task Description

Load local T1w + T2w + BOLD + DWI (BIDS format) and run the complete HCP multimodal preprocessing pipeline.

## Input Requirement

Required input(s):

- Local T1w image in BIDS format (required)
- Local T2w image in BIDS format (required)
- Local BOLD image(s) in BIDS format (required)
- Local DWI image with bvecs/bvals in BIDS format (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T62_hcp_full_multimodal/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- PreFreeSurfer outputs
- FreeSurfer outputs
- PostFreeSurfer outputs
- fMRIVolume outputs
- fMRISurface outputs
- ICA-FIX outputs
- Diffusion outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
