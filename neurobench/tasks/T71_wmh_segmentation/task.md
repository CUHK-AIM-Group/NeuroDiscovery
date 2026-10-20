# T71_wmh_segmentation: White Matter Hyperintensity Segmentation

## Objective
Run MARS-WMH nnU-Net automatic WMH segmentation

## Inputs
FLAIR and T1w anatomical NIfTI files

## Outputs
WMH segmentation mask in NIfTI format

## Key Points
- Prepare FLAIR and T1w images (registration if needed)
- Register to MNI152 template for consistency
- Run pre-trained nnU-Net model for WMH detection
- Generate binary segmentation mask
- Compute WMH volume and spatial statistics
- Output probability map for visual inspection
- Generate QC report with lesion maps overlaid
- Save results in BIDS derivatives format

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
