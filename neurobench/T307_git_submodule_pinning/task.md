# T307_git_submodule_pinning: Git Submodule Pinning Audit
## Task Description

Audit submodules in the analysis monorepo: pin each to a commit, document the upstream URL + pinned SHA, and add update instructions.

## Input Requirement


- No interactive input.


## Constraints

- Table of submodule -> SHA -> purpose.

- Include the update workflow commands.

- Save all generated artifacts to:
  - benchmark_results/T307_git_submodule_pinning/


## Expected Output

Expected output artifact(s):

- `submodules.md`

- `.gitmodules` updates


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
