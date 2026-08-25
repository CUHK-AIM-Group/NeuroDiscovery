# T149_xcpd_denoise_36p: XCP-D Denoising: 36P
## Task Description

Post-process fMRIPrep derivatives with XCP-D using the 36-parameter model (24 motion + WM/CSF + global signal with derivatives) strategy. Produce denoised BOLD, confound-filtering QC, and framewise-displacement statistics.

## Input Requirement

Required input(s):

- fMRIPrep derivatives directory for the subject(s) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use the `xcp_d` BIDS app with the matching confound strategy documented in the command log.

- Band-pass filter 0.01-0.1 Hz; FD threshold 0.5 mm unless the strategy dictates otherwise.

- Report percent of volumes censored.

- Save all generated artifacts to:
  - benchmark_results/T149_xcpd_denoise_36p/


## Expected Output

Expected output artifact(s):

- Denoised BOLD (NIfTI + CIFTI if available)

- `qc_report.html` (pre/post denoising carpet plots)

- `fd_summary.json` (mean FD, censored fraction)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
