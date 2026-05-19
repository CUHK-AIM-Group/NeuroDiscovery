# T57_dwi_tensor_metrics: DWI DTI Tensor Metrics

## Objective
Fit DTI tensor model and compute scalar maps (FA, MD, AD, RD)

## Inputs
Preprocessed DWI data with valid b-values and gradients

## Outputs
FA.nii.gz, MD.nii.gz, AD.nii.gz, RD.nii.gz scalar maps

## Key Points
- Estimate diffusion tensor using least-squares fitting
- Compute Fractional Anisotropy (FA) map
- Compute Mean Diffusivity (MD) map
- Compute Axial Diffusivity (AD) map
- Compute Radial Diffusivity (RD) map
- Include numerical safety checks (NaN handling, range validation)

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TCK, VTK where applicable)
- CSV files must contain headers and valid numerical data
- No errors or incomplete computations during processing
