# T99_hcp_dwi_processing: HCP Diffusion MRI Processing

## Objective
Run diffusion MRI preprocessing, tensor metrics, and tractography on HCP data

## Inputs
Locally available HCP diffusion MRI inputs (DWI, bvals, bvecs, and required masks/metadata for the selected pipeline stage) plus writable output location

## Outputs
Diffusion derivatives in dwi_output/

If required HCP diffusion inputs are missing, explicitly report `Missing required input` instead of broadening the task into unrelated dataset staging or non-HCP workflows.

## Key Points
- Keep the benchmark mainline on HCP diffusion processing for the provided local inputs
- Run preprocessing and tensor/tractography steps only as required by the task; do not expand into generic BIDS organization, cross-dataset harmonization, or unrelated downstream analytics unless explicitly requested
- Required deliverables are the core diffusion derivatives produced by the chosen HCP-faithful pipeline stage
- Additional connectome or topology analyses are optional extras and should not replace the required diffusion outputs

## Evaluation Criteria
- Task completion verified by presence of required output files
- Modality-specific outputs must match expected file formats
- QC reports must be comprehensive and readable
- Processing logs must document parameters, versions, and execution time
- No critical errors during processing
