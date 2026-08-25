# T323_credential_audit_repo: Credential Handling Audit
## Task Description

Audit the repository for credential risks: scan history for secrets (gitleaks), inventory where credentials are used, and write handling rules.

## Input Requirement


- No interactive input.


## Constraints

- Use gitleaks or equivalent on full history.

- Rules doc covers env vars, .env files, CI secrets.

- Save all generated artifacts to:
  - benchmark_results/T323_credential_audit_repo/


## Expected Output

Expected output artifact(s):

- `gitleaks_report.json`

- `CREDENTIAL_POLICY.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
