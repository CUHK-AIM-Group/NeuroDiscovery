# Benchmark Test Case T00: Installer and Environment Validation

## Task Description

Validate that the NeuroClaw self-contained installer correctly generates
`neuroclaw_environment.json` and `core/config/features.json`, and that the
resulting environment satisfies all runtime prerequisites.

### Motivation

NeuroClaw is now self-contained (no OpenClaw prerequisite). The installer
(`installer/setup.py` / `installer/config_wizard.py`) is the critical path
for every new deployment. This task verifies the installer's correctness
end-to-end.

---

## Constraints

- **No manual configuration**: all inputs must be either auto-detected or
  supplied via the non-interactive mode (`--non-interactive` flag).
- **No OpenClaw installation required**: the test must pass on a clean system
  with only Python ≥ 3.10 and Git.

---

## Input

- Fresh repository clone (no `neuroclaw_environment.json` present).
- Command: `python installer/setup.py --non-interactive`

---

## Expected Outputs

Saved to `benchmark_results/T00_installer_validation/`:

```
T00_installer_validation/
├── neuroclaw_environment.json   (copy of generated file)
├── features.json                (copy of core/config/features.json)
├── validation_report.json       (structured test results)
└── install_log.txt              (copy of installer/install_log.txt if generated)
```

### `validation_report.json` structure

```json
{
  "task_id": "T00",
  "timestamp": "<ISO-8601>",
  "overall_status": "PASS | FAIL",
  "checks": [
    {
      "name": "env_file_exists",
      "status": "PASS | FAIL",
      "detail": "neuroclaw_environment.json created at <path>"
    },
    {
      "name": "env_file_schema",
      "status": "PASS | FAIL",
      "detail": "All required keys present: setup_type, python_path, cuda, toolchain, llm_backend, neuro_defaults"
    },
    {
      "name": "python_path_valid",
      "status": "PASS | FAIL | SKIP",
      "detail": "python_path points to a valid Python executable"
    },
    {
      "name": "features_file_exists",
      "status": "PASS | FAIL",
      "detail": "core/config/features.json present"
    },
    {
      "name": "connectors_disabled",
      "status": "PASS | FAIL",
      "detail": "WhatsApp, Telegram, Slack, calendar, ecommerce connectors all have enabled=false"
    },
    {
      "name": "core_features_enabled",
      "status": "PASS | FAIL",
      "detail": "agent_loop, skill_loader, tool_runtime, session_manager, tmux_shell all have enabled=true"
    },
    {
      "name": "check_flag_exits_zero",
      "status": "PASS | FAIL",
      "detail": "python installer/setup.py --check returns exit code 0"
    }
  ]
}
```

---

## Key Steps

1. Remove any existing `neuroclaw_environment.json` from the repo root
   (or run in a clean checkout).
2. Run: `python installer/setup.py --non-interactive`
3. Verify `neuroclaw_environment.json` was created.
4. Parse the JSON and confirm all top-level keys exist.
5. Confirm `python_path` resolves to a real Python executable.
6. Parse `core/config/features.json` and confirm:
   - `connectors.whatsapp.enabled == false`
   - `connectors.telegram.enabled == false`
   - `connectors.slack.enabled == false`
   - `core.agent_loop.enabled == true`
7. Run `python installer/setup.py --check` and confirm exit code 0.
8. Write `validation_report.json` with results of each check.
9. Copy generated files into `benchmark_results/T00_installer_validation/`.

---

## Important Checks

| Check | Criterion | Severity |
|-------|-----------|----------|
| `neuroclaw_environment.json` created | File exists | FAIL if missing |
| All required JSON keys present | `setup_type`, `python_path`, `cuda`, `toolchain`, `llm_backend`, `neuro_defaults` | FAIL if any missing |
| `python_path` is executable | `os.path.isfile(python_path)` | WARN if missing (Docker setup acceptable) |
| Non-neuroscience connectors disabled | All listed connectors `enabled=false` | FAIL if any enabled |
| Core features enabled | All core features `enabled=true` | FAIL if any disabled |
| `--check` flag exits cleanly | Exit code 0 | FAIL otherwise |
