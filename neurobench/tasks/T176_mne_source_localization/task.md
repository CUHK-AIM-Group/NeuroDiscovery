# T176_mne_source_localization: MNE EEG Source Localization (stage beyond T124)
## Task Description

Source-localize cleaned epoched EEG: BEM forward model from the subject's FreeSurfer recon, dSPM inverse solution, and cortical activation maps per condition.

## Input Requirement

Required input(s):

- Cleaned epochs (FIF, e.g. from T124) (required)

- FreeSurfer subject directory (required)

- Coregistration fiducials (required)


If any required input is missing, return:

- Missing required input


## Constraints

- 3-layer BEM; document conductivity values.

- Inverse: dSPM with loose=0.2, depth=0.8 unless justified.

- Save all generated artifacts to:
  - benchmark_results/T176_mne_source_localization/


## Expected Output

Expected output artifact(s):

- `fwd.fif`, `inv.fif`, per-condition `stc` files

- `source_activation.png` (inflated brain, per condition)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
