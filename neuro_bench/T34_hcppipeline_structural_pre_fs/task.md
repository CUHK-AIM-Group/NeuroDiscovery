# Benchmark Test Case 34: HCP Structural PreFreeSurfer

## Task Description

Load local T1w + T2w (BIDS format) and run HCP PreFreeSurfer structural preprocessing.

## Input Requirement

Required input(s):

- Local T1w image in BIDS format (required)
- Local T2w image in BIDS format (required)
- Enough BIDS context to resolve the target subject/session anatomy directory

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Prefer a BIDS-aware input discovery flow rather than hard-coding only one literal file path pattern with no validation.
- Explicitly validate required HCP resources before launch, including the PreFreeSurfer entry script and required template/config files.
- Make the no-fieldmap / no-gradient-distortion branch explicit when those auxiliary inputs are not available; do not silently assume unavailable correction assets exist.
- Save all generated artifacts to:
  - benchmark_results/T49_hcp_structural_pre_fs/
- Save logs, reports, and status files under the configured benchmark output folder rather than writing them into the input BIDS tree.
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- PreFreeSurfer processing outputs
- Output directory structure sufficient to inspect the subject-level PreFreeSurfer result tree

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- Higher-quality solutions explicitly handle BIDS subject/session discovery, required-resource validation, and missing-input decisions.
- Higher-quality solutions keep the narrow mainline on HCP PreFreeSurfer itself, without drifting into unrelated stages or helper skills that are not needed for this task.
- Higher-quality solutions justify fallback choices for sample spacing, fieldmap usage, and gradient-distortion handling when the required metadata or auxiliary inputs are absent.
- This test case is manually evaluated.
