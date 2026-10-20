# T370_topic_trend_by_year: Topic Trend by Year
## Task Description

Quantify publication trends for a topic: per-year counts via PubMed/OpenAlex queries, growth rate, and a text-rendered or PNG trend chart.

## Input Requirement

Required input(s):

- Corpus file (paper list or query) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Queries identical across years (only date filter varies).

- Report CAGR over the window.

- Save all generated artifacts to:
  - benchmark_results/T370_topic_trend_by_year/


## Expected Output

Expected output artifact(s):

- `trend_data.csv`

- `trend_chart.png`

- `trend_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
