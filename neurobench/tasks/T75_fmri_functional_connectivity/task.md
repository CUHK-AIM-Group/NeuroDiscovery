# T75_fmri_functional_connectivity: fMRI Functional Connectivity Matrices

## Objective
Compute Pearson correlation functional connectivity matrices

## Inputs
ROI timeseries from multiple atlases

## Outputs
Whole-brain connectivity matrices and network graphs

## Key Points
- Compute Pearson correlation between all ROI pairs
- Generate connectivity matrix for each atlas
- Apply Fisher z-transformation for statistics
- Generate network visualization (adjacency networks)
- Compute nodal metrics (degree, strength, betweenness)
- Identify network modules (if applicable)
- Output matrices in CSV and NPZ formats
- Generate connectivity visualizations (chord diagrams, matrices)

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TSV, NPZ where applicable)
- Statistical maps must contain valid numerical data with proper dimensions
- Connectivity matrices must be symmetric and valid (values between -1 and 1)
- No errors during processing, comprehensive logs generated
