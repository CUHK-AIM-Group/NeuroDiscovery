# T74_fmri_roi_extraction: fMRI ROI Timeseries Extraction

## Objective
Extract standard atlas ROI timeseries from denoised BOLD

## Benchmark Mode Contract
In benchmark mode, treat this as a narrow ROI-only extraction task. The main deliverable must be a direct executable workflow that takes denoised BOLD and writes atlas ROI timeseries outputs.

Do not widen the task into connectivity, connectome, seed-correlation, graph analysis, HCP pipeline orchestration, BIDS organization, or higher-level fMRI orchestration unless the prompt explicitly asks for those products.

Planning is allowed only to justify atlas routing, resampling, masker choice, or fallback behavior. The answer should still center on one direct ROI extraction implementation rather than a multi-skill orchestration or delegation plan.

## Inputs
Denoised BOLD data (XCP-D output)

## Outputs
Multi-atlas timeseries files (.tsv format)

## Key Points
- Load multiple standard atlases (Schaefer 100/200/400, Glasser 360, Gordon 333, Tian 96)
- Register atlases to subject space if needed
- Extract mean BOLD timeseries for each ROI
- Standardize and quality control timeseries
- Output separate .tsv files for each atlas
- Include ROI labels and coordinates in headers
- Save metadata with atlas descriptions
- Keep the workflow ROI-timeseries-only; do not add connectivity or connectome products unless explicitly requested
- Prefer a single direct Nilearn-style extraction script over HCP/BIDS/fMRI orchestration-layer wrappers or delegation-only plans

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TSV, NPZ where applicable)
- Statistical maps must contain valid numerical data with proper dimensions
- Stronger answers stay narrowly focused on ROI timeseries extraction and do not spend the main deliverable on connectome, connectivity, or unrelated orchestration outputs
- No errors during processing, comprehensive logs generated
