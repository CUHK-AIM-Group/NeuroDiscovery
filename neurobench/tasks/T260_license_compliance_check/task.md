# T260_license_compliance_check: License Compliance Check
## Task Description

Audit a dataset + derivatives for license compliance: license files present, redistribution terms compatible with planned release, and attribution strings collected.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Contributors/criteria files where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Check each source component (atlases, templates) license too.

- Report conflicts as blockers, not suggestions.

- Save all generated artifacts to:
  - benchmark_results/T260_license_compliance_check/


## Expected Output

Expected output artifact(s):

- `license_audit.md`

- `attribution_strings.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
