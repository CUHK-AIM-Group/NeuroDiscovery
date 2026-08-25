# T239_datacite_metadata_generation: DataCite Metadata Generation
## Task Description

Generate a DataCite-compatible metadata record (JSON + YAML) for a dataset release: creators, ORCIDs, license, funding, related identifiers.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Config/metadata inputs as stated (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Validate against the DataCite schema (jsonschema).

- Pull author ORCIDs from the provided contributors file only.

- Save all generated artifacts to:
  - benchmark_results/T239_datacite_metadata_generation/


## Expected Output

Expected output artifact(s):

- `datacite.json`

- `datacite.yaml`

- `schema_validation.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
