# T202_mrs_fitting_pipeline: MRS Fitting Pipeline (FSL-MRS)
## Task Description

Process a single-voxel MRS spectrum: format conversion (spec2nii), eddy-current correction, and basis-set fitting with FSL-MRS, reporting major metabolite concentrations with CRLBs.

## Input Requirement

Required input(s):

- Raw MRS data (Twix/DICOM) + water reference (required)

- Basis set and sequence parameters (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use spec2nii + fsl_mrs; document versions.

- Report SNR and FWHM QC metrics; state exclusion criteria.

- Save all generated artifacts to:
  - benchmark_results/T202_mrs_fitting_pipeline/


## Expected Output

Expected output artifact(s):

- `metabolite_concentrations.csv` (with CRLB)

- `fit_report.pdf` (per-metabolite fit plots)

- `mrs_qc.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
