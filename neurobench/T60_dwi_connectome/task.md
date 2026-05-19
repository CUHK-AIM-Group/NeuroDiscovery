# T60_dwi_connectome: DWI Structural Connectome

## Objective
Generate ROI-to-ROI structural connectivity matrix from tractography

## Inputs
Filtered tractogram (TCK), parcellation atlas in subject space, and writable output location

## Outputs
connectome.csv containing the ROI-to-ROI connectivity matrix

If the tractogram or atlas is missing, explicitly report `Missing required input` instead of switching to an upstream diffusion-modeling or tractography workflow.

## Key Points
- Use tck2connectome to compute connection matrix
- Treat the provided tractogram as the starting point for this benchmark task
- Do not rerun DWI preprocessing, tensor fitting, CSD modeling, or whole-brain tractography unless the task is explicitly re-scoped
- Count number of tracts between each ROI pair by default
- Optional weighted variants are allowed, but the required deliverable is the ROI-to-ROI CSV matrix
- Ensure symmetric/undirected connectivity matrix
- Output CSV with shape (N_ROIs × N_ROIs)
- Include labels and metadata for interpretation

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (CSV where required)
- CSV files must contain headers and valid numerical data
- No errors or incomplete computations during processing
