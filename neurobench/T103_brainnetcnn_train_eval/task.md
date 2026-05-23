# T103_brainnetcnn_train_eval: BrainNetCNN Training and Evaluation

## Task Description

Train BrainNetCNN (Kawahara et al.) on FC matrices. The model uses Edge-to-Edge (E2E), Edge-to-Node (E2N), and Node-to-Graph (N2G) filters specialised for symmetric connectivity matrices.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ) for a chosen atlas
- Subject list and labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Implement E2E/E2N/N2G blocks; input FC must be symmetric NxN.
- 5-fold deterministic split shared with T101-T102 for paired comparison.
- Save artefacts under `models/benchmark_results/T103_brainnetcnn/<setting>/`.

## Expected Output

- Per-fold test metrics CSV (regression + classification)
- Edge-importance map (gradient x input) for one subject
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- HCP age MAE within ~10% of BrainGNN baseline.
- ABIDE AUC >= 0.60 on at least one atlas.
- Verify input symmetry assumption in code.
