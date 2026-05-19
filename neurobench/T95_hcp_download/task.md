# T95_hcp_download: HCP Data Download and Acquisition

## Objective
Download HCP Young Adult (HCP1200) dataset subsets using NeuroSTORM

## Inputs
HCP download request specifications (subset type, subject IDs)

## Outputs
Raw HCP data in hcp_output/raw/ directory

## Key Points
- Initialize HCP dataset connection (ConnectomeDB credentials)
- Support download subsets: all, rfMRI, tfMRI, t1t2, dwi
- Use NeuroSTORM scripts for parallelized downloading
- Handle large-scale downloads with resume capability
- Validate downloaded files (check MD5 checksums)
- Organize raw data by subject in hcp_output/raw/sub-*/ses-01/
- Store download logs and error reports
- Generate data inventory report

## Evaluation Criteria
- Task completion verified by presence of required output files
- BIDS staging must pass bids-validator with minimal warnings
- Modality-specific outputs must match expected file formats
- QC reports must be comprehensive and readable
- Processing logs must document parameters, versions, and execution time
- Data integrity verified by file count and checksum validation
- No critical errors during processing
