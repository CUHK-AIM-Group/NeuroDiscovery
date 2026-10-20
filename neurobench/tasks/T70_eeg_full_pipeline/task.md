# T70_eeg_full_pipeline: EEG Full Preprocessing and Feature Extraction

## Objective
Run MNE-Python complete EEG pipeline with feature extraction

## Inputs
EEG data files (.set/.edf/.bdf/.fif formats)

## Outputs
Processed EEG and feature matrices (power, CSP, connectivity, microstates)

## Key Points
- Load and parse raw EEG data with proper montage
- Apply bandpass filtering (1-100 Hz recommended)
- Run automatic ICA decomposition for artifact removal
- Identify and remove ICA components (eye artifacts, muscle, etc.)
- Segment into epochs based on experimental conditions
- Compute frequency band power (Delta, Theta, Alpha, Beta, Gamma)
- Calculate Common Spatial Patterns (CSP) if multiple conditions
- Compute connectivity metrics (coherence, PLI, phase-locking)
- Extract microstate features and transitions
- Output: eeg_output/ with processed data, features, and visualizations

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
