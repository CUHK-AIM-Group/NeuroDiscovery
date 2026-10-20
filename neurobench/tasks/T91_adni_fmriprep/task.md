# T91_adni_fmriprep: ADNI fMRIPrep Preprocessing

## Objective
Run complete fMRIPrep preprocessing on ADNI BIDS dataset

## Inputs
BIDS-formatted ADNI dataset with T1w and rs-fMRI

## Outputs
fMRIPrep derivatives (preprocessed BOLD, confounds, anatomy) and HTML QC

## Key Points
- Execute fMRIPrep on ADNI BIDS data
- Run FreeSurfer surface reconstruction for anatomical
- Perform functional preprocessing (motion correction, distortion correction)
- Output preprocessed BOLD in MNI152 and native space
- Generate confound regressors (motion, CompCor, ICA-AROMA)
- Produce anatomical derivatives (brain masks, tissue probability maps)
- Generate comprehensive HTML QC report
- Save processing derivatives in BIDS derivatives format

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, JSON, HTML where applicable)
- BIDS validation reports must show compliance or documented exceptions
- Timeseries CSV files must have proper dimensions (timepoints × ROIs)
- HTML QC reports must be readable and contain meaningful diagnostic information
- Processing logs must document all key parameters and software versions
- No critical errors during processing
