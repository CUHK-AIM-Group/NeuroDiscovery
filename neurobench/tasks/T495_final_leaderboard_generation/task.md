# T495_final_leaderboard_generation: Final Leaderboard Generation
## Task Description

Generate the definitive benchmark leaderboard: all models x settings, mean +/- std, significance marks vs. the top model, in CSV + Markdown + LaTeX.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Significance from the corrected tests (T447-T454).

- Formats: CSV, MD, LaTeX booktabs.

- Save artefacts under `models/benchmark_results/T495_final_leaderboard_generation/`.


## Expected Output

Expected output artifact(s):

- `leaderboard.csv`

- `leaderboard.md`

- `leaderboard.tex`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
