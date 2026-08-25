# T167_ciftify_fmri: ciftify fMRI CIFTI Workflow
## Task Description

Run ciftify `ciftify_recon_all` + `ciftify_subject_fmri` to map one subject's fMRI to CIFTI grayordinates from FreeSurfer recon-all output.

## Input Requirement

Required input(s):

- BIDS dataset with fMRI (required)

- FreeSurfer recon-all output for the subject (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use ciftify (HCP-derived surfaces, 32k fsLR).

- Document the HCP templates version.

- Save all generated artifacts to:
  - benchmark_results/T167_ciftify_fmri/


## Expected Output

Expected output artifact(s):

- `*.dtseries.nii` CIFTI time series

- Surface QC scene/PNG

- `ciftify_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
