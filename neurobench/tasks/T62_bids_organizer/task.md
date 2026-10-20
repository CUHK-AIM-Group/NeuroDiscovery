# T62_bids_organizer: BIDS Dataset Organization

## Objective
Automatically organize local DICOM/NIfTI/EEG raw data into BIDS-compliant format

## Inputs
Raw DICOM, NIfTI, and/or EEG files (mixed formats)

## Outputs
BIDS-compliant dataset structure with validation report

## Benchmark Mode Contract
- In benchmark mode, do not stop at a scan/plan/confirm workflow and do not wait for interactive approval before organization.
- The expected answer is a direct end-to-end organizer or staging solution that can inventory mixed raw inputs, make deterministic subject/session/modality decisions, and produce the final BIDS tree in one executable script or command sequence.
- Use planning only to justify mapping rules or fallback behavior; the mainline deliverable should still be the concrete organizer itself.

## Key Points
- Convert DICOM to NIfTI if needed
- Create proper BIDS directory hierarchy (sub-*/ses-*/anat/func/dwi/fmap/eeg/)
- Rename files according to BIDS naming convention
- Generate dataset_description.json, README, and CHANGES files
- Create JSON sidecars with metadata
- Run bids-validator and generate validation report
- Log any conversion or organization warnings
- Prefer a dataset-level generic organizer over single-subject examples, interactive confirmation flows, or wrapper/delegation-only plans
- Make subject/session inference, modality routing, and fallback handling explicit when raw inputs are heterogeneous or partially labeled

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, JSON, CSV, HTML where applicable)
- Statistical maps must contain valid numerical data
- No errors during processing, proper logs generated
- Stronger answers provide one direct, executable end-to-end organizer rather than a confirmation-first workflow, partial scaffold, or delegation-only plan
