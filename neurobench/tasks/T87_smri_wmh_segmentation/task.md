# T87_smri_wmh_segmentation: Structural MRI WMH Segmentation

## Objective
Automatic white matter hyperintensity segmentation using MARS-WMH nnU-Net

## Inputs
- One subject FLAIR NIfTI file in native space
- The matching T1w anatomical NIfTI file for the same subject
- Writable output directory for benchmark artifacts

If either FLAIR or T1w is not provided, report `Missing required input` instead of inventing placeholder paths or switching to another pipeline.

## Outputs
- A WMH segmentation mask NIfTI file
- A WMH statistics CSV file with header row
- Optional probability/QC files if the chosen MARS-WMH invocation produces them

Write benchmark outputs under `benchmark_results/T87_smri_wmh_segmentation/`.

## Key Points
- Use one narrow mainline only: run the pre-trained MARS-WMH nnU-Net segmentation workflow on the provided FLAIR + T1w pair
- Do not broaden the solution into generic model training, alternative WMH frameworks, or unrelated preprocessing branches
- Keep images in native space unless the selected MARS-WMH invocation requires internal registration
- Produce a binary WMH segmentation mask as the required imaging output
- Compute WMH volume or lesion statistics and save them as CSV
- If probability maps or QC overlays are supported, treat them as optional extras rather than required substitutes for the mask and CSV
- Save outputs to the benchmark result directory instead of a generic derivatives or ad hoc subject folder

## Evaluation Criteria
- Task completion verified by presence of required output files
- Required output files for this benchmark are:
	- `benchmark_results/T87_smri_wmh_segmentation/wmh_mask.nii.gz`
	- `benchmark_results/T87_smri_wmh_segmentation/wmh_stats.csv`
- Output files must be in correct format (NIfTI and CSV)
- Structural maps must have proper dimensions and valid data ranges
- CSV files must contain headers and valid numerical data
- No errors during processing, comprehensive logs generated
