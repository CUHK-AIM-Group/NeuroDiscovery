# T201_myelin_t1t2_ratio: HCP-Style T1w/T2w Myelin Map Pipeline
## Task Description

Compute HCP-style T1w/T2w-ratio myelin maps: bias-correct T1w and T2w, rigid-align, ratio on surfaces via FreeSurfer, and parcellate with a cortical atlas.

## Input Requirement

Required input(s):

- T1w + T2w NIfTIs (required)

- FreeSurfer recon-all output (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Follow the Glasser & Van Essen T1w/T2w methodology.

- Cortical ribbon masking required; report mean ratio per HCP-MMP parcel.

- Save all generated artifacts to:
  - benchmark_results/T201_myelin_t1t2_ratio/


## Expected Output

Expected output artifact(s):

- `t1t2_ratio.midthickness.func.gii` (or NIfTI equivalent)

- `parcel_myelin.csv`

- `myelin_render.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
