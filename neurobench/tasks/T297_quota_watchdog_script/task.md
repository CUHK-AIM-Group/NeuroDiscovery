# T297_quota_watchdog_script: Disk Quota Watchdog
## Task Description

Write a watchdog script that checks user/group quota, warns at configurable thresholds, and logs to a rotating file; wire it to cron or a systemd timer.

## Input Requirement


- No interactive input.


## Constraints

- Read-only on the filesystem.

- Thresholds configurable at the top of the script.

- Save all generated artifacts to:
  - benchmark_results/T297_quota_watchdog_script/


## Expected Output

Expected output artifact(s):

- `quota_watchdog.sh`

- `crontab_snippet.txt`

- `sample_alert.log`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
