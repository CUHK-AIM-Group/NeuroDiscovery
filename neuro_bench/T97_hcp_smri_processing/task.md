# T97_hcp_smri_processing: HCP Structural MRI Processing

## Objective
Run complete structural MRI processing pipeline on HCP data

## Inputs
HCP structural MRI (T1w, T2w) from local staged inputs

## Outputs
Structural derivatives in smri_output/

## Key Points
- Required inputs are:
	- local T1w image
	- local T2w image
	- installed HCP Pipelines, FSL, and FreeSurfer license/environment
- If T1w or T2w is missing, return:
	- Missing required input
- Use the canonical HCP structural-only mainline:
	- PreFreeSurferPipeline.sh
	- FreeSurferPipeline.sh
	- PostFreeSurferPipeline.sh
- Do not expand into generic `smri-skill` multi-method processing, fMRIPrep, functional MRI, diffusion MRI, download workflows, or multimodal orchestration.
- Keep outputs under a deterministic benchmark structural output directory and focus on HCP structural derivatives, surfaces, and QC.
- Generate cortical thickness maps and surface atlases
- Compute cortical and subcortical statistics
- Output HCP-grade surfaces (white, pial, inflated)
- Include QC metrics and visualization

## Evaluation Criteria
- Task completion verified by presence of required output files
- BIDS staging must pass bids-validator with minimal warnings
- Modality-specific outputs must match expected file formats
- QC reports must be comprehensive and readable
- Processing logs must document parameters, versions, and execution time
- Data integrity verified by file count and checksum validation
- No critical errors during processing
