# Benchmark Test Case 5: DICOM to NIfTI Conversion (dcm2nii)

## Task Description

Convert the DICOM files in `./201` to NIfTI files.

## Input Requirement

Required input folder:

- `./201`

If `./201` does not exist or contains no DICOM files, return:

- `Missing required input`

## Constraints

- Use the existing environment/tools (no interactive input).
- Keep conversion deterministic and reproducible.
- Save outputs to:
  - `benchmark_results/T05_dcm2nii/`

## Output Requirement

At least one NIfTI file must be generated:

- `*.nii` or `*.nii.gz`

Optional sidecar metadata files are allowed:

- `*.json`
- logs / command output files

## Success Criteria

- Input folder `./201` exists and contains DICOM files
- At least one valid NIfTI file is generated under `benchmark_results/T05_dcm2nii/`
- Generated NIfTI is consistent with source DICOM metadata (dimension/spacing level)

## Verification Suggestion

Use commands to self-check results, for example:

- `find ./201 -type f | head`
- `find benchmark_results/T05_dcm2nii -type f | sort`
- `find benchmark_results/T05_dcm2nii -type f | grep -E '\\.(nii|nii\\.gz)$'`

The grader validates consistency between generated NIfTI and the source DICOM series.
