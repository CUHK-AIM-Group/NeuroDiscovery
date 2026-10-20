# T257_inventory_dashboard_html: Dataset Inventory Dashboard
## Task Description

Generate a self-contained HTML dashboard summarizing a dataset: counts per modality/site/scanner, acquisition-date timeline, and storage footprint.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Single HTML file, no external JS/CSS dependencies.

- All numbers reproducible from the dataset alone.

- Save all generated artifacts to:
  - benchmark_results/T257_inventory_dashboard_html/


## Expected Output

Expected output artifact(s):

- `inventory_dashboard.html`

- `inventory_stats.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
