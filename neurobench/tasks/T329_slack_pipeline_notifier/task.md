# T329_slack_pipeline_notifier: Pipeline Slack/Webhook Notifier
## Task Description

Add a notifier that posts pipeline completion/failure digests to a webhook (Slack/Teams/ntfy), callable from SLURM scripts and local runs.

## Input Requirement


- No interactive input.


## Constraints

- Webhook URL via env/config, never committed.

- Message format: job, status, duration, failed subjects.

- Save all generated artifacts to:
  - benchmark_results/T329_slack_pipeline_notifier/


## Expected Output

Expected output artifact(s):

- `notify.sh`

- `sample_message.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
