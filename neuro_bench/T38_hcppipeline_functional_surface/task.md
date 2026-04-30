# Benchmark Test Case 38: HCP Functional Surface Projection

## Task Description

Load local fMRIVolume outputs and run HCP fMRISurface projection.

## Input Requirement

Required input(s):

- fMRIVolume output directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T53_hcp_functional_surface/
- Long-running processing is allowed to run as a background job.
- The benchmark mainline is the HCP `fMRISurface` stage applied to existing `fMRIVolume` outputs; do not expand backward into full HCP functional preprocessing when the volume-stage outputs are already provided.
- Prefer the canonical `fMRISurface` pipeline entry point over ad hoc parallel surface-projection recipes unless a required prerequisite is missing.

## Expected Output

Expected output artifact(s):

- Surface-projected functional outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
