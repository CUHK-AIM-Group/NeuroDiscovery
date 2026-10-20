# T285_release_changelog_automation: Release Changelog Automation
## Task Description

Automate changelog generation from conventional commits (git-cliff or similar), wired into the tag workflow.

## Input Requirement


- No interactive input.


## Constraints

- Config committed; changelog groups by type (feat/fix/docs).

- Dry-run on existing history included.

- Save all generated artifacts to:
  - benchmark_results/T285_release_changelog_automation/


## Expected Output

Expected output artifact(s):

- `cliff.toml` (or equivalent)

- `CHANGELOG.md` sample


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
