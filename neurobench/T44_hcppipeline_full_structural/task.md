# Benchmark Test Case 44: HCP Full Structural Pipeline

## Task Description

Load local T1w + T2w (BIDS format) and run complete HCP structural pipeline.

## Input Requirement

Required input(s):

- Local T1w image in BIDS format (required)
- Local T2w image in BIDS format (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T44_hcppipeline_full_structural/
- Long-running processing is allowed to run as a background job.

- Assume this benchmark is structural-only.
- Use the canonical HCP structural sequence:
  - PreFreeSurferPipeline.sh
  - FreeSurferPipeline.sh
  - PostFreeSurferPipeline.sh
- Do not expand into functional, diffusion, multimodal, download, BIDS-conversion, or generic platform-orchestration work unless explicitly required by the provided inputs.
- If either required T1w or T2w image is missing, return:
  - Missing required input

## Expected Output

Expected output artifact(s):

- PreFreeSurfer + FreeSurfer + PostFreeSurfer outputs
- Structural outputs should remain under `benchmark_results/T44_hcppipeline_full_structural/` rather than a generic derivatives root.

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
