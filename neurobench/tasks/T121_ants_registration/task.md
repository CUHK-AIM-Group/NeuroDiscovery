# T121_ants_registration: ANTs SyN Registration to MNI152
## Task Description

Register a subject T1w image to the MNI152NLin2009cAsym template using ANTs
`antsRegistrationSyN.sh` (affine + SyN deformable), producing the warped brain
and the forward/inverse transform chain.

## Input Requirement

Required input(s):

- Subject T1w NIfTI file (required)
- MNI152NLin2009cAsym template path (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use ANTs `antsRegistrationSyN.sh` with `-d 3 -t s` (or the equivalent
  `antsRegistration` call with the same stages).
- Save all generated artifacts to:
  - benchmark_results/T121_ants_registration/
- Report the ANTs version (`antsRegistration --version`).

## Expected Output

Expected output artifact(s):

- `warped.nii.gz` (subject T1w in MNI space)
- `transform_0GenericAffine.mat`, `transform_1Warp.nii.gz`, `transform_1InverseWarp.nii.gz`
- Overlay QC image (template edges over warped brain, PNG)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
