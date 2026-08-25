# T162_fmriprep_multiecho_tedana: fMRIPrep Pipeline: multi-echo + tedana
## Task Description

Run the full fMRIPrep anatomical + functional workflow for a multi-echo fMRI subject with multi-echo ICA (tedana) enabled. Produce standard fMRIPrep derivatives and the HTML QC report.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)

- FreeSurfer license file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- fMRIPrep LTS version pinned in the log; output spaces MNI152NLin2009cAsym + T1w (add fsnative for the multi-echo variant).

- Document any non-default flags and justify them.

- Save all generated artifacts to:
  - benchmark_results/T162_fmriprep_multiecho_tedana/


## Expected Output

Expected output artifact(s):

- fMRIPrep derivatives tree (anat + func)

- `sub-*.html` QC report

- `run_summary.json` (version, flags, wall time)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
