# T119_cross_dataset_generalization: Cross-Dataset Generalisation with Harmonization

## Task Description

Evaluate model generalisation across datasets / scanners. Train on one dataset, test on another, with and without harmonization (ComBat / ComBat-GAM / CovBat / site-as-covariate). Tests the joint value of mega-analysis tooling and modelling.

Two settings:
- HCP -> ABIDE (age regression on subset where age range overlaps)
- ABIDE -> ADHD-200 (control vs case classification with shared subjects)

## Input Requirement

Required input(s):

- ROI-level FC matrices for both source and target dataset
- Site / scanner metadata (required for harmonization)
- Subject list and labels CSV per dataset
- Choice of harmonization method (or `none` for baseline)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `skills/harmonization-tool/` for ComBat / ComBat-GAM / CovBat.
- Fit harmoniser on source train split + applicable target split per chosen splitter.
- Split protocol: site-stratified within source; full target as test.
- Save artefacts under `models/benchmark_results/T119_cross_dataset/<source>_to_<target>/<harmonizer>/`.

## Expected Output

- Per-method (4 harmonizers + none) test metric on target
- Train-source vs test-target metric gap (generalisation drop)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- At least one harmonization method must reduce the source->target gap vs `none`.
- ComBat/ComBat-GAM cannot be combined with leave-site-out (confirmed via [[feedback_harmonization_loso_combat]]) - task must use site-stratified split.
- Document failure modes if no method helps.
