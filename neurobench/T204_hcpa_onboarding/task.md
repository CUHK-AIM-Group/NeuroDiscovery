# T204_hcpa_onboarding: HCP-A Onboarding: Download, Stage, Validate
## Task Description

Download a subject subset of HCP Aging (HCP-A (Lifespan Aging) via ConnectomeDB/NDA), stage it into a BIDS-compliant directory, and produce a validation + inventory report.

## Input Requirement

Required input(s):

- Subject/session list file (required)

- Access credentials or pre-downloaded archive path (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Organize as BIDS (sub-*/ses-*/<modality>/) with `dataset_description.json` and `participants.tsv`.

- Staged dataset must pass `bids-validator` with no errors.

- Downloads must be resumable; keep a per-file status log.

- Save all generated artifacts to:
  - benchmark_results/T204_hcpa_onboarding/


## Expected Output

Expected output artifact(s):

- BIDS dataset tree

- `bids_validation_report.txt`

- `data_inventory.csv` (file, size, checksum)

- `download_log.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
