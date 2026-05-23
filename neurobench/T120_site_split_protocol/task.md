# T120_site_split_protocol: Site-Stratified vs Leave-Site-Out Evaluation

## Task Description

Compare the same model under two site-aware evaluation protocols on a multi-site dataset (ABIDE or ADHD-200): (a) site-stratified 80/10/10 split, (b) leave-site-out CV. Quantifies the optimism of within-site evaluation.

## Input Requirement

Required input(s):

- ROI-level FC matrices with site metadata
- Subject list and labels CSV
- Choice of model (default BNT)

If any required input is missing, return:

- Missing required input

## Constraints

- Site-stratified protocol uses 80/10/10 train/val/test, single split per [[feedback-cv-protocol]].
- Leave-site-out: each site is held out once; harmonization restricted to site-covariate (not ComBat) per [[feedback_harmonization_loso_combat]].
- Use `skills/harmonization-tool/` splitter utilities.
- Save artefacts under `models/benchmark_results/T120_site_protocol/<protocol>/`.

## Expected Output

- Per-protocol test metric (mean +/- std across sites for LOSO)
- Optimism gap = site-stratified - leave-site-out
- Per-site failure analysis (which site is hardest?)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Both protocols completed end-to-end.
- Optimism gap must be quantified, even if small.
- Code must enforce no leakage: harmoniser fit only on train sites in LOSO.
