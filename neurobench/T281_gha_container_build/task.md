# T281_gha_container_build: GitHub Actions: Container Build + Push
## Task Description

CI workflow that builds the project Dockerfile on tag pushes and publishes to a registry with the tag as image version.

## Input Requirement


- No interactive input.


## Constraints

- Build only on version tags.

- Push logins via secrets; document required secrets.

- Save all generated artifacts to:
  - benchmark_results/T281_gha_container_build/


## Expected Output

Expected output artifact(s):

- Workflow YAML

- `secrets_checklist.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
