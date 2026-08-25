# T175_mne_erp_full_pipeline: MNE ERP Full Pipeline
## Task Description

Event-related pipeline with MNE-Python: preprocessing, epoching, artifact rejection, evoked responses per condition, and a between-condition contrast with cluster-corrected sensor statistics.

## Input Requirement

Required input(s):

- Task EEG recording (required)

- Event definitions per condition (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Document rejection thresholds and trial counts kept per condition.

- Statistics: spatio-temporal cluster permutation test.

- Save all generated artifacts to:
  - benchmark_results/T175_mne_erp_full_pipeline/


## Expected Output

Expected output artifact(s):

- `evoked_conditions.fif` + butterfly plots

- `contrast_stats.png` (significant clusters)

- `trial_counts.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
