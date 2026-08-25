# T123_qsiprep_qsirecon_pipeline: QSIPrep + QSIRecon Diffusion Pipeline
## Task Description

Run an end-to-end diffusion workflow for one subject: preprocess raw DWI with
QSIPrep, then reconstruct with QSIRecon (DSI Studio GQI + deterministic
tracking), producing a connectome-ready tractogram and scalar maps.

## Input Requirement

Required input(s):

- BIDS dataset directory with DWI + reverse phase-encoding data (required)
- FreeSurfer license file (required by QSIPrep)

If any required input is missing, return:

- Missing required input

## Constraints

- QSIPrep: default preprocessing with eddy motion/eddy-current correction.
- QSIRecon: `--recon-spec dsi_studio_gqi` tractography with 1M streamlines.
- Container execution (Docker or Singularity) with versions pinned in the log.
- Save all generated artifacts to:
  - benchmark_results/T123_qsiprep_qsirecon_pipeline/

## Expected Output

Expected output artifact(s):

- QSIPrep derivatives tree (preprocessed DWI, bval/bvec, QC report HTML)
- QSIRecon outputs (`streamlines.tck`, FA/MD/RD scalar maps)
- Pipeline runtime log (per-stage wall time)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
