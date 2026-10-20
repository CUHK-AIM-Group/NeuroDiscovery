# T112_hierarchical_gnn_train_eval: Hierarchical GNN Training and Evaluation

## Task Description

Train a hierarchical GNN (multi-resolution pooling across atlas levels - e.g. ROI -> community -> whole-brain) on FC, using the `skills/hierarchical/` skill and `models/` infrastructure.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Atlas community assignment (Yeo-7 / Yeo-17)
- Subject list and labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Pooling hierarchy: ROI -> Yeo network -> graph.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T112_hierarchical/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Per-network pooled embedding visualised (UMAP/PCA)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Outperforms flat GIN baseline on at least one metric.
- Network-level embeddings cluster meaningfully (silhouette > 0).
