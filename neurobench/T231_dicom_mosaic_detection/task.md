# T231_dicom_mosaic_detection: DICOM Mosaic Format Detection
## Task Description

Detect Siemens mosaic-format DWI/fMRI DICOMs in a dump and produce a conversion-readiness report (which series need mosaic-aware conversion).

## Input Requirement

Required input(s):

- DICOM directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Detect via ImageType/SIEMENS CSA headers; do not rely on filenames.

- Recommend the converter (dcm2niix version) per series.

- Save all generated artifacts to:
  - benchmark_results/T231_dicom_mosaic_detection/


## Expected Output

Expected output artifact(s):

- `mosaic_series.csv`

- `conversion_readiness.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
