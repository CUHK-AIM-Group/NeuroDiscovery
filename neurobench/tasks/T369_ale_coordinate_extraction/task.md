# T369_ale_coordinate_extraction: ALE Coordinate Extraction
## Task Description

Extract activation foci coordinates from papers for an ALE-style meta-analysis: normalize to MNI (document conversions from Talairach), format for GingerALE.

## Input Requirement

Required input(s):

- Included-paper list from screening (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Conversion tool/params documented (icbm2tal or reverse).

- GingerALE input format validated.

- Save all generated artifacts to:
  - benchmark_results/T369_ale_coordinate_extraction/


## Expected Output

Expected output artifact(s):

- `foci_ale.txt`

- `coordinate_log.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
