# Benchmark Test Case 9: Overleaf Upload (Cookie-based)

## Task Description

After the user provides a valid Overleaf cookie, upload files from local `./cvpr_paper` directory to Overleaf project `cvpr_paper`.

Required workflow:

1. Ask user for Overleaf authentication cookie.
2. Attempt to locate project named `cvpr_paper`.
3. If the project does not exist, attempt to create a new project named `cvpr_paper`.
4. If project creation fails, exit.
5. Upload local files from `./cvpr_paper` to the project.

## Input Requirement

Required inputs:

- User-provided Overleaf cookie (interactive input)
- Local directory: `./cvpr_paper`

If `./cvpr_paper` does not exist, return:

- `Missing required input`

If cookie is missing/invalid, return a clear authentication error and exit.

## Constraints

- Must rely on user-provided cookie for authentication.
- Must try project creation only when project is not found.
- Must stop if project creation fails.
- Save execution artifact(s) to:
  - `benchmark_results/T09_overleaf/`

## Expected Output

At least one execution result file should be written, recommended:

- `result_YYYYMMDD_HHMMSS.json`

Recommended JSON structure:

```json
{
  "metadata": {
    "task": "T09_overleaf",
    "timestamp": "ISO-8601",
    "status": "success or fail"
  },
  "input": {
    "local_dir": "./cvpr_paper",
    "project_name": "cvpr_paper",
    "cookie_provided": true
  },
  "actions": {
    "project_found": true,
    "project_created": false,
    "upload_attempted": true,
    "upload_succeeded": true
  },
  "notes": "optional string"
}
```

## Evaluation

- This test case is **manually evaluated**.
- Manual reviewer checks whether:
  - cookie interaction exists,
  - project lookup/create logic is correct,
  - upload behavior matches task requirements.
