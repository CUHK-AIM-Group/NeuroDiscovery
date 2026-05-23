# T113_spacenet_train_eval: SpaceNet (Spatially Regularised) Training and Evaluation

## Task Description

Train SpaceNet (Dohmatob et al.) - TV-L1 / graph-net spatially regularised decoder - on FC upper-triangle features (or ROI maps) for HCP age and ABIDE dx.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ) OR voxel/ROI feature maps
- Subject list and labels CSV

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn's `SpaceNetRegressor` / `SpaceNetClassifier` interface.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T113_spacenet/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Coefficient map (thresholded) saved as TSV
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Beats ridge (T107) when penalty is properly tuned, otherwise within 10%.
- Coefficient map spatially smooth (visual or quantitative TV check).
