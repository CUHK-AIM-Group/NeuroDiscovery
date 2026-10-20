# T383_dataset_citation_audit: Dataset Citation Audit
## Task Description

Audit whether papers using public datasets (HCP/ABIDE/ADNI/UKB) cite them correctly: required citations + acknowledgments present.

## Input Requirement

Required input(s):

- Paper list or corpus (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Required-citation lists per dataset compiled first.

- Verdict per paper: compliant / partial / missing.

- Save all generated artifacts to:
  - benchmark_results/T383_dataset_citation_audit/


## Expected Output

Expected output artifact(s):

- `dataset_citation_audit.csv`

- `compliance_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
