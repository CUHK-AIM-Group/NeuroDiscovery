# T96_hcp_bids_staging: HCP BIDS Data Staging and Organization

## Objective
Organize and standardize raw HCP data into BIDS-style directory structure

## Inputs
Raw HCP data from hcp_output/raw/

## Outputs
Standardized BIDS staging directory with validation report

## Key Points
- Parse raw HCP directory structure and identify modalities
- Organize structural MRI (T1w, T2w)
- Organize functional MRI (fMRI resting-state and task)
- Organize diffusion MRI (DWI, bvals, bvecs)
- Create BIDS-compliant directory layout: sub-*/ses-01/anat|func|dwi/
- Generate JSON sidecars with HCP parameters
- Create dataset_description.json for HCP
- The benchmark focus is staging and naming, not a full downstream processing or conversion pipeline
- A lightweight staging script that maps the HCP directory tree into a BIDS-style layout is the expected mainline
- Prefer re-linking or copying existing HCP NIfTI/bval/bvec assets directly rather than expanding into unrelated conversion workflows
- If a modality or metadata file is absent in the provided HCP source tree, stage the available data and clearly record the missing items instead of inventing extra recovery branches
- Include a concise staging report describing what was staged and what was skipped or missing

## Evaluation Criteria
- Task completion verified by presence of required output files
- BIDS-style staging layout must be correct and internally consistent for the files that were provided
- Modality-specific outputs must match expected file formats
- The staging report must clearly describe mapped modalities and any skipped or missing inputs
- Processing logs should document the mapping choices that were applied
- No critical errors during staging
