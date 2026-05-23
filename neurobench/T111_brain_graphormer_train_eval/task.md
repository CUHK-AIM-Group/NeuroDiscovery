# T111_brain_graphormer_train_eval: Brain Graph Transformer (Graphormer-Brain) Training and Evaluation

## Task Description

Train a Graphormer-style transformer adapted for brain FC graphs - uses centrality encoding, spatial encoding via shortest-path on the FC backbone, and edge-weight bias in the attention scores.

## Input Requirement

Required input(s):

- ROI-level FC matrices (NPZ)
- Subject list and labels CSV (HCP age + ABIDE dx)

If any required input is missing, return:

- Missing required input

## Constraints

- Backbone graph from top-k thresholded FC (k=20 default).
- Centrality + spatial + edge-bias all enabled; ablate one for the report.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T111_brain_graphormer/<setting>/`.

## Expected Output

- Per-fold test metrics CSV (full + ablated)
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Full model beats ablation on at least one metric.
- Comparable to BNT (T102) on HCP age.
