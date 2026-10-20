# T61_dwi_full_pipeline: DWI Full Analysis Pipeline

## Objective
End-to-end DWI analysis: BIDS organization → QSIPrep → tensor → ROI → tractography → connectome

## Inputs
Local raw DWI NIfTI/bval/bvec files, plus structural/anatomical support needed for downstream tractography/connectome steps.

## Outputs
Complete dwi_output/ with all derivatives: BIDS structure, preproc_dwi, scalar maps, roi_stats.csv, tractogram, streamline weighting, connectome, and an explicit manifest of any required user-supplied atlas/alignment assets.

## Key Points
- T83: Organize input files into BIDS format
- T84: Run QSIPrep preprocessing
- T85: Compute DTI scalar maps (FA, MD, AD, RD)
- T86: Extract ROI-wise diffusion features
- T87: Perform MRtrix3 tractography with ACT and prefer SIFT2 weighting when the task is a full tractography/connectome pipeline
- T88: Generate structural connectome matrix with an explicit atlas/parcellation choice, label/LUT handling, and alignment path between diffusion and atlas space
- Save all intermediate and final outputs in organized dwi_output directory
- Generate comprehensive processing log documenting all steps
- If atlas/parcellation files, label tables, or diffusion-to-structural alignment prerequisites are missing, the answer must explicitly say which required inputs are missing and ask the user to choose the key analysis-shaping options rather than silently falling back to a weaker default pipeline
- The preferred default solution is a QSIPrep -> MRtrix3 multi-step route for full tractography/connectome analysis; tensor-only outputs are required but are not sufficient by themselves

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TCK, VTK where applicable)
- CSV files must contain headers and valid numerical data
- No errors or incomplete computations during processing
- Higher-quality answers explicitly surface the required atlas/labels/alignment decisions and produce a connectome path that is consistent with the chosen tractography workflow
