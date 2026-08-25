# T240_release_diff_changelog: Release Diff Changelog
## Task Description

Diff two dataset releases and produce a human-readable changelog: added/removed/modified files grouped by subject and modality.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Config/metadata inputs as stated (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Modified = same path, different checksum.

- Summary counts at top; detail tables below.

- Save all generated artifacts to:
  - benchmark_results/T240_release_diff_changelog/


## Expected Output

Expected output artifact(s):

- `CHANGELOG_vA_vB.md`

- `diff_detail.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
