# T98_hcp_fmri_processing: HCP Functional MRI Processing

## Objective
Run functional MRI preprocessing and connectivity analysis on HCP data

## Inputs
HCP functional MRI (resting-state BOLD) from BIDS staging

## Outputs
Functional derivatives in fmri_output/

## Key Points
- Delegate to fmri-skill for preprocessing and analysis
- Run fMRIPrep: motion correction, distortion correction, registration
- Run XCP-D: denoising with motion scrubbing, bandpass filtering
- Extract ROI timeseries from multiple atlases
- Compute functional connectivity matrices
- Generate resting-state network maps
- Compute network metrics (degree, strength, betweenness)
- Output preprocessed BOLD in MNI and native space
- Include confound regressors and QC metrics

## Evaluation Criteria
- Task completion verified by presence of required output files
- BIDS staging must pass bids-validator with minimal warnings
- Modality-specific outputs must match expected file formats
- QC reports must be comprehensive and readable
- Processing logs must document parameters, versions, and execution time
- Data integrity verified by file count and checksum validation
- No critical errors during processing
