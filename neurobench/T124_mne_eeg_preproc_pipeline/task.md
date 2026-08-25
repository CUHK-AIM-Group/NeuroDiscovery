# T124_mne_eeg_preproc_pipeline: MNE-Python EEG Preprocessing Chain (stage split of T70)
## Task Description

Stage split of T70_eeg_full_pipeline: preprocessing only. Load raw EEG, apply
band-pass filtering, run ICA-based artifact removal, epoch around events, and
compute the evoked response. No source localization or statistics.

## Input Requirement

Required input(s):

- Raw EEG recording (FIF/EDF/BrainVision, required)
- Event annotations or stim channel (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use MNE-Python throughout.
- Band-pass 1-40 Hz, notch at line frequency (50/60 Hz documented).
- ICA with the extended infomax method; document which components were
  rejected and why (EOG/ECG correlation criteria).
- Save all generated artifacts to:
  - benchmark_results/T124_mne_eeg_preproc_pipeline/

## Expected Output

Expected output artifact(s):

- `cleaned_epo.fif` (epoched, artifact-corrected data)
- `evoked-ave.fif` plus a butterfly-plot PNG
- `ica_report.html` (MNE Report with topomaps of rejected components)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
