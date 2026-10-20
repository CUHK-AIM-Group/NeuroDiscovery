# T59_dwi_mrtrix_tractography: DWI MRtrix3 Fiber Tractography

## Objective
Perform anatomically-constrained tractography using MRtrix3 (ACT + SIFT)

## Inputs
Preprocessed DWI + T1w + aparc+aseg segmentation

## Outputs
SIFT-filtered tractogram (TCK format) and VTK visualization file

## Key Points
- Compute response function from preprocessed DWI
- Perform constrained spherical deconvolution (CSD)
- Generate fiber orientation distribution follows
- Run anatomically-constrained tractography using 5-tissue model
- Apply SIFT filter to reduce spurious tracks
- Output tractogram.tck and visualization.vtk for QC

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TCK, VTK where applicable)
- CSV files must contain headers and valid numerical data
- No errors or incomplete computations during processing
