# T58_dwi_roi_features: DWI ROI-wise Diffusion Features

## Objective
Extract ROI-wise diffusion metrics using DTI scalar maps and parcellation

## Inputs
Either existing FA/MD/AD/RD maps plus atlas/parcellation inputs, or raw DWI inputs (`dwi.nii.gz`, `bval`, `bvec`, mask) sufficient to generate missing scalar maps; atlas/parcellation labels/LUT may be provided separately

## Outputs
- `roi_stats.csv` with ROI-wise diffusion metrics
- provenance record stating for each reported metric whether it came from pre-existing scalar maps or from scalar maps newly fitted in this workflow
- explicit manifest of atlas/parcellation assets used, including label/LUT source and any alignment/resampling decisions

## Key Points
- Explicitly validate spatial consistency between atlas/parcellation, scalar maps, mask, and DWI space before extracting ROI statistics
- Explicitly handle atlas/parcellation alignment: if the atlas is not already in the scalar-map space, state the required registration/resampling step and preserve label integrity with nearest-neighbor handling for labels
- Use labels/LUT information when available so the output identifies both ROI IDs and ROI names; if labels/LUT are missing, surface that as a missing-input decision rather than silently fabricating atlas metadata
- If scalar maps already exist and pass spatial-consistency checks, skip recomputation and extract ROI metrics directly from those existing maps
- If required scalar maps do not exist, use a DIPY/QSIPrep-compatible path to derive FA/MD/AD/RD from raw DWI inputs before ROI aggregation
- Make missing-input decisions explicit: if required DWI, atlas, labels/LUT, mask, or alignment assets are absent, state what can proceed, what must be requested, and what should not be silently assumed
- Extract mean and standard deviation of FA within each ROI
- Extract mean and standard deviation of MD within each ROI
- Extract mean and standard deviation of AD within each ROI
- Extract mean and standard deviation of RD within each ROI
- Output CSV with columns: roi_label, roi_name, fa_mean, fa_std, md_mean, md_std, ad_mean, ad_std, rd_mean, rd_std
- Output provenance must distinguish metrics summarized from pre-existing maps versus metrics produced from newly fitted scalar maps in the same workflow

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TCK, VTK where applicable)
- CSV files must contain headers and valid numerical data
- Solutions should correctly validate and report atlas/parcellation alignment, labels/LUT use, and spatial-consistency checks instead of assuming them away
- Solutions should correctly choose between reusing existing scalar maps and deriving them from raw DWI, with the decision justified by available inputs
- Provenance should clearly record which ROI metrics came from reused maps and which depended on newly generated scalar maps
- No errors or incomplete computations during processing
