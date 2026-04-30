# T85_smri_hcp_structural: Structural MRI HCP-Style Processing

## Objective
Run HCP-style structural preprocessing on BIDS anatomical data

## Inputs
- BIDS anatomical dataset containing one subject's structural images under `sub-*/[ses-*/]anat/`
- Required structural inputs for the canonical HCP structural path:
	- one T1w NIfTI file
	- one matching T2w NIfTI file
- Required installed prerequisites:
	- HCP Pipelines with `PreFreeSurferPipeline.sh`, `FreeSurferPipeline.sh`, and `PostFreeSurferPipeline.sh`
	- FSL
	- FreeSurfer with readable license file

If T1w or T2w is missing, report `Missing required input` instead of downgrading to a non-HCP structural workflow.

## Outputs
- HCP structural outputs for the selected subject written under `benchmark_results/T85_smri_hcp_structural/`
- Required benchmark outputs:
	- subject `T1w/` directory
	- subject `MNINonLinear/` directory
	- subject `MNINonLinear/fsaverage_LR32k/` directory
	- structural log files for each stage

## Key Points
- PreFreeSurfer: Anatomical preprocessing and alignment
- Perform brain extraction and normalization
- Bias field correction and tissue segmentation
- FreeSurfer: Surface reconstruction and segmentation
- PostFreeSurfer: Surface registration and refinement
- Use one canonical mainline only:
	- `PreFreeSurferPipeline.sh`
	- `FreeSurferPipeline.sh`
	- `PostFreeSurferPipeline.sh`
- This benchmark is structural-only; do not expand to functional, diffusion, or multimodal processing
- Do not require or invoke `MSMAll`; that belongs to a broader multimodal HCP workflow, not this structural benchmark
- For BIDS anatomical inputs without fieldmap or gradient-distortion metadata, use the structural-only no-fieldmap / no-gdcoeffs path
- Output high-quality structural surfaces and standard HCP structural derivatives for the selected subject

## Evaluation Criteria
- Task completion verified by presence of required output files
- Required benchmark outputs are the presence of:
	- `benchmark_results/T85_smri_hcp_structural/<subject>/T1w/`
	- `benchmark_results/T85_smri_hcp_structural/<subject>/MNINonLinear/`
	- `benchmark_results/T85_smri_hcp_structural/<subject>/MNINonLinear/fsaverage_LR32k/`
	- `benchmark_results/T85_smri_hcp_structural/logs/`
- Output files must be in correct HCP structural format rather than generic CSV/statistics artifacts
- Structural maps must have proper dimensions and valid data ranges
- No errors during processing, comprehensive logs generated
