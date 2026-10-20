# Benchmark Test Case 2: BIDS Organizer (HCP Subset)

## Task Description

Please store the **T1** and **fMRI** data in `./hcp-subset` using the **BIDS** format.

## Input Requirement

- Required input folder: `./hcp-subset`
- If `./hcp-subset` does not exist, return:
  - `Missing required input`

## Constraints

- No interactive input.
- Organize data in-place under `./hcp-subset`.
- Include both modalities:
  - T1 structural MRI (anat)
  - fMRI BOLD (func)
- Use BIDS-compatible naming and directory layout.

## Required BIDS Structure (minimum)

For each subject `sub-<label>`:

- `sub-<label>/anat/sub-<label>_T1w.nii` or `sub-<label>_T1w.nii.gz`
- `sub-<label>/anat/sub-<label>_T1w.json`
- `sub-<label>/func/sub-<label>_task-<task>_bold.nii` or `sub-<label>_task-<task>_bold.nii.gz`
- `sub-<label>/func/sub-<label>_task-<task>_bold.json`

Dataset-level required files:

- `dataset_description.json`

## Verification Requirement

You must use shell commands to verify directory structure and file extensions by yourself, e.g.:

- `find ./hcp-subset -maxdepth 4 -type d | sort`
- `find ./hcp-subset -type f | sort`
- `find ./hcp-subset -type f | grep -E '\\.(nii|nii\\.gz|json)$'`

## Success Criteria

- `./hcp-subset` exists
- Contains at least one BIDS subject folder: `sub-*`
- Each detected subject has valid `anat` T1w file pair (`.nii/.nii.gz` + `.json`)
- Each detected subject has valid `func` BOLD file pair (`.nii/.nii.gz` + `.json`)
- File names match BIDS modality suffixes (`_T1w`, `_bold`)
- `dataset_description.json` exists
