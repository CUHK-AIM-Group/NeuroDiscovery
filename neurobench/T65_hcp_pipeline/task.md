# T65_hcp_pipeline: HCP-Style High-Quality Pipeline

## Objective
Run HCP-style preprocessing on multimodal BIDS MRI data

## Inputs
BIDS multimodal dataset (T1w, T2w, fMRI, DWI with fmap)

## Outputs
HCP-style derivatives (structural, functional, diffusion processed)

## Key Points
- PreFreeSurfer: Anatomical preprocessing and alignment
- FreeSurfer: Surface reconstruction and segmentation
- PostFreeSurfer: Surface registration and refinement
- fMRIVolume: Functional preprocessing to volume space
- fMRISurface: Functional mapping to surface
- ICA-FIX: Automated denoising of fMRI
- DiffusionPreprocessing: Eddy and topup corrections
- Save high-quality intermediate and final derivatives

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
