# T104_ibgnn_train_eval: IBGNN (Interpretable Brain GNN) Training and Evaluation

## Task Description

Train and evaluate IBGNN - an interpretable brain GNN with explanation masks on edges and nodes - on FC graphs. The model learns an explainable mask jointly with the predictive task.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Subject list and labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `models/ibgnn/` net + training script.
- Loss must include explanation sparsity regulariser (mask L1).
- 5-fold deterministic split shared with T101-T103.
- Save artefacts under `models/benchmark_results/T104_ibgnn/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- Per-fold explanation mask (top-k edges) saved as TSV
- Aggregate consensus mask (edges selected in >= 3 / 5 folds)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Predictive metric within ~10% of BrainGNN baseline.
- Consensus mask must be non-empty and interpretable (top edges traceable to known networks).
- Mask sparsity <= 5% of total edges.
