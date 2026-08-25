# T306_git_lfs_migration: Git LFS Migration for Large Assets
## Task Description

Migrate large non-data assets (figures, small atlases, model weights < 100 MB) from plain git to Git LFS, rewriting tracking without breaking clones.

## Input Requirement


- No interactive input.


## Constraints

- `.gitattributes` patterns documented.

- Verify fresh clone pulls LFS objects correctly.

- Save all generated artifacts to:
  - benchmark_results/T306_git_lfs_migration/


## Expected Output

Expected output artifact(s):

- `.gitattributes`

- `migration_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
