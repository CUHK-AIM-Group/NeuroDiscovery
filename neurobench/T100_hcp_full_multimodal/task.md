# T100_hcp_full_multimodal: HCP Complete Multimodal End-to-End Pipeline

## Objective
Run complete multimodal HCP processing from download to final derivatives

## Benchmark Mode Contract
In benchmark mode, do not stop at a confirmation-first orchestration layer and do not make the main deliverable a multi-skill delegation plan. The answer should provide one unified end-to-end canonical HCP shell plan that covers download or input acquisition, staging, stage invocation, and final output checks in an executable mainline.

That mainline should make missing-input gates explicit before execution begins, including credentials or tokens, subject scope, destination paths, required HCP pipeline resources, and any required T1w, T2w, BOLD, or DWI inputs for the requested subset. If the needed inputs are missing, the plan should fail clearly or skip stages according to an explicit rule rather than leaving the decision implicit.

Stage order and per-stage validation should be written into the mainline itself. The benchmark-preferred pattern is to validate prerequisites, run the canonical stages in order, and verify expected outputs after each major stage before continuing.

## Inputs
HCP dataset specifications (download parameters and processing options)

## Outputs
Complete hcp_output/ with all multimodal derivatives and QC

## Key Points
- T123: Download specified HCP subset from ConnectomeDB
- T124: Organize raw data into BIDS staging directory
- Run T125, T126, T127 in parallel (structural, functional, diffusion)
- In benchmark mode, present one direct task-level shell workflow rather than a confirmation-first multi-skill orchestration wrapper
- Make missing-input gates explicit for credentials, subset selection, subject IDs, paths, HCP software roots, and modality-specific required files
- Write canonical stage order into the mainline and include concrete per-stage validation checks before advancing
- Coordinate across three processing streams:
-   - sMRI: surfaces, segmentation, cortical metrics
-   - fMRI: preprocessing, connectivity, network analysis
-   - DWI: tensor fitting, tractography, connectome
- Generate individual subject reports for each modality
- Create integrated multimodal QC report
- Produce summary statistics across all modalities
- Save organized hcp_output with consistent naming and metadata
- Ready for group-level multimodal analysis

## Evaluation Criteria
- Task completion verified by presence of required output files
- BIDS staging must pass bids-validator with minimal warnings
- Modality-specific outputs must match expected file formats
- QC reports must be comprehensive and readable
- Processing logs must document parameters, versions, and execution time
- Data integrity verified by file count and checksum validation
- Stronger answers provide a unified canonical HCP shell mainline with explicit missing-input gates, fixed stage order, and per-stage validation rather than a generic confirmation/delegation plan
- No critical errors during processing
