# Benchmark Test Case 8: Git Workflow (Interactive Config + Amend + Force Push)

## Task Description

Complete the following Git workflow tasks:

1. Interactively ask the user to configure Git identity:
   - `user.name`
   - `user.email`
2. Create a new Git repository.
3. Create a `README.md` with any initial content and push it.
4. Amend the previous commit:
   - Change `README.md` content to exactly:
     - `NeuroBench Task 8`
5. Force push the amended commit.

## Input Requirement

- No fixed local input files.
- The agent should collect required Git identity values from the user interaction.

## Constraints

- Must include explicit user interaction step for Git identity configuration.
- Must perform `git commit --amend` on the last commit.
- Must perform force push after amend.
- Save execution result to:
  - `benchmark_results/T08_git/`

## Expected Output (JSON)

Save one JSON file, for example:
- `result_YYYYMMDD_HHMMSS.json`

Recommended structure:

```json
{
  "metadata": {
    "task": "T08_git",
    "timestamp": "ISO-8601",
    "status": "success or fail"
  },
  "git": {
    "user_name": "string",
    "user_email": "string",
    "repo_url": "string",
    "branch": "main"
  },
  "verification": {
    "amend_used": true,
    "force_push_used": true,
    "final_readme": "NeuroBench Task 8"
  },
  "notes": "optional string"
}
```

## Success Criteria

- Result JSON exists in `benchmark_results/T08_git/`
- Grader can clone/pull the target repository
- Repository `README.md` content is exactly `NeuroBench Task 8`

## Verification Suggestion

Use commands to self-check, for example:

- `git config --global user.name`
- `git config --global user.email`
- `git log --oneline -n 3`
- `cat README.md`
- `git push --force-with-lease`
