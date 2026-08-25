# T395_funding_ack_mining: Funding Acknowledgment Mining
## Task Description

Mine funding acknowledgments from the corpus: funding agencies, grant numbers, and co-funding patterns relevant to our grant application.

## Input Requirement

Required input(s):

- Bibliography/corpus file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Agency normalization (name variants merged).

- Grant-number validation per agency format.

- Save all generated artifacts to:
  - benchmark_results/T395_funding_ack_mining/


## Expected Output

Expected output artifact(s):

- `funding_mentions.csv`

- `funding_landscape.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
