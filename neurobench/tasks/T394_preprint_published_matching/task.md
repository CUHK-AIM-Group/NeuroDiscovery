# T394_preprint_published_matching: Preprint-to-Published Matching
## Task Description

Match preprints in the corpus to their published versions (Crossref/OpenAlex relation), updating citation records to the version of record.

## Input Requirement

Required input(s):

- Bibliography/corpus file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Match evidence documented (relation field or title+authors).

- Unmatched preprints listed.

- Save all generated artifacts to:
  - benchmark_results/T394_preprint_published_matching/


## Expected Output

Expected output artifact(s):

- `version_matches.csv`

- `updated_references.bib`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
