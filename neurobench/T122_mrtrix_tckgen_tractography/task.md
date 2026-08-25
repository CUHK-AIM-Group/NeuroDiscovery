# T122_mrtrix_tckgen_tractography: MRtrix tckgen Streamline Tractography (stage split of T59)
## Task Description

Stage split of T59_dwi_mrtrix_tractography: run ONLY the tractography step.
Given a precomputed FOD image and a seed mask, generate a whole-brain
streamline tractogram with MRtrix3 `tckgen` using the iFOD2 algorithm.

## Input Requirement

Required input(s):

- FOD image (`wmfod.mif`, required)
- Seed mask (gray-matter/white-matter interface mask, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `tckgen -algorithm iFOD2` with 10M streamlines (`-select 10000000`).
- ACT is optional (`-act 5tt.mif`) and must be documented if used.
- Save all generated artifacts to:
  - benchmark_results/T122_mrtrix_tckgen_tractography/

## Expected Output

Expected output artifact(s):

- `tractogram_10M.tck`
- `tckgen_log.txt` (full command log with parameters)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
