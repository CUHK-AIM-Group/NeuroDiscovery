---
name: conda-env-manager
description: "Create, inspect, export, update or clone a Conda environment with explicit package targets and execution receipts. Use for environment management, not merely because a task uses Python."
license: MIT License (NeuroClaw custom skill)
layer: subagent
skill_type: tool
dependencies:
  - dependency-planner
---
# Conda Environment Manager

Honor the requested environment operation and existing authorization. Use the
given name/prefix; for a new environment choose an unused name/prefix, and stop
on a collision instead of silently reusing it. Keep base unchanged unless the
user explicitly selected it. Loading this skill plans methods, not execution.

## Create and run in a known environment

Resolve an actual Conda executable, for example from a valid `CONDA_EXE` or the
current PATH, and check `conda info --json`. Capture missing-executable and
nonzero-exit errors. For a new isolated prefix:

```bash
# PREFIX is a new, unused location; PACKAGE is the requested package specification.
conda create --prefix "$PREFIX" --channel conda-forge --yes "$PACKAGE"
conda run --prefix "$PREFIX" --no-capture-output "$PROGRAM" --version
conda list --prefix "$PREFIX" --json
```

Choose requested channels and versions rather than applying a universal order.
Use `--yes` only within the authorized operation; an optional dry-run belongs on
the relevant subcommand. Stop if creation or installation fails; do not continue
to a system executable and report a successful environment test.

An executable name may still resolve outside the selected environment. Record
its resolved location inside that environment and confirm it belongs to the
prefix, along with package/version metadata. For Python record `sys.executable`
and `sys.prefix`; for a program such as htop use the installed prefix's binary
and report its version/exit status. Prefer `conda run -p/-n` in scripts to
activation in another shell that cannot modify the caller's environment.
[Create reference](https://docs.conda.io/projects/conda/en/stable/commands/create.html),
[run reference](https://docs.conda.io/projects/conda/en/22.11.x/commands/run.html).

## Record failures as well as successes

Initialize the output summary before dispatch. For each actual command retain
argv, start/end or elapsed time, return code, stdout and stderr. Catch OS errors
such as `FileNotFoundError` in addition to failed commands, serialize the failure
summary, and return nonzero. A summary marked successful must follow successful
creation/install and the requested tests; printing a command is plan-only.
When giving executable code, implement these paths rather than commenting out
the execution call or leaving a pseudo-delegate.

## Reproducibility matched to the task

Export an existing environment with `conda env export -p "$PREFIX" --from-history`
for human-readable intent, or `conda list -p "$PREFIX" --explicit` for a concrete
platform-specific package list. Record actual Conda/package/platform versions
when available. A history export is not an exact cross-platform lock file.
Only report hashes, GPU information, sizes or verification scores actually
measured; do not populate plausible demonstration metadata as real observations.

For clone/update/remove, select the explicit target, preserve useful exports and
respect the user's scope. Dependency-planner may help select packages when
available, but an unavailable delegate is not required to install an already
specified package. No environment archival or checkpoint hook runs merely from
reading this document; write it in the actual execution workflow if needed.
