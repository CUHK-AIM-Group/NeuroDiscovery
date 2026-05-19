# Benchmark Test Case 4: Functional Connectivity Extraction (conn-tool)

## Task Description

Extract functional connectivity (FC) from the preprocessed fMRI volume:

- `./hcp-subset/sub-100206/func/sub-100206_task-rest_bold.nii.gz`

and save the results.

## Requirements

- Use the **existing conda environment** (do not create a new one).
- The fMRI is already preprocessed, so compute FC directly.
- TR is fixed at **0.8 s**.
- Use `./aal3.nii.gz` as atlas if needed.
- Skip additional masking; rely on atlas overlap.

## Input Requirement

Required input files:

- `./hcp-subset/sub-100206/func/sub-100206_task-rest_bold.nii.gz`
- `./aal3.nii.gz` (if using atlas-based extraction)

If required input is missing, return:

- `Missing required input`

## Output

Save results to:

- `benchmark_results/T04_conn_tool/`

At least one FC matrix file must be generated. Recommended file names:

- `fc_matrix.npy`
- `fc_matrix.npz`
- `fc_matrix.csv`
- `fc_matrix.json`

## Success Criteria

- FC result file is generated under `benchmark_results/T04_conn_tool/`
- FC matrix is 2D and square (N x N)
- N is reasonable for atlas-based FC (N >= 10)
- Matrix values are numeric and finite

## Verification Suggestion

Use commands to verify output files by yourself, for example:

- `find benchmark_results/T04_conn_tool -type f | sort`
- `python -c "import numpy as np; x=np.load('benchmark_results/T04_conn_tool/fc_matrix.npy'); print(x.shape)"`
- `python -c "import json; print(json.load(open('benchmark_results/T04_conn_tool/result.json')).keys())"`
