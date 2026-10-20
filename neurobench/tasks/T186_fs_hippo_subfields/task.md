# T186_fs_hippo_subfields: FreeSurfer Hippocampal Subfields
## Task Description

Segment hippocampal subfields (and amygdala nuclei if T2 is available) with FreeSurfer `segmentHA_T1.sh`/`segmentHA_T2.sh`, extract volumes, and QC against expected ranges.

## Input Requirement

Required input(s):

- FreeSurfer recon-all output for the subject (required)

- T2w NIfTI (optional, improves subfield accuracy)


If any required input is missing, return:

- Missing required input


## Constraints

- Document FreeSurfer version (subfield labels differ across versions).

- Volumes in mm^3, corrected for eTIV in the report table.

- Save all generated artifacts to:
  - benchmark_results/T186_fs_hippo_subfields/


## Expected Output

Expected output artifact(s):

- `[lr]h.hippoSfVolumes.txt` parsed to CSV

- `subfield_qc.png` overlays

- `volume_range_flags.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
