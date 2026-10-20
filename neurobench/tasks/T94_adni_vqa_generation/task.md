# T94_adni_vqa_generation: ADNI VQA Question-Answer Generation

## Objective
Generate Visual Question Answering (VQA) datasets from ADNI data

## Inputs
Processed ADNI data or raw imaging files

## Outputs
VQA task labels and question-answer pairs in vqa_outputs/

## Key Points
- Reorganize ADNI data using reorganize_adni utility
- Generate ADNI task files with generate_adni_task_files
- Create visual question templates from anatomical/functional images
- Generate multiple VQA task types:
-   - Task 1: Brain region identification
-   - Task 2: ROI localization and anatomy
-   - Task 3: Functional activation patterns
-   - Task 4: Connectivity/network visualization
-   - Task 5: Quality assessment and artifact detection
- Run generate_vqa_from_tasks to produce Q&A pairs
- Output organized vqa_outputs/ with task labels and QA datasets
- Include metadata for VQA model training and evaluation

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, JSON, HTML where applicable)
- BIDS validation reports must show compliance or documented exceptions
- Timeseries CSV files must have proper dimensions (timepoints × ROIs)
- HTML QC reports must be readable and contain meaningful diagnostic information
- Processing logs must document all key parameters and software versions
- No critical errors during processing
