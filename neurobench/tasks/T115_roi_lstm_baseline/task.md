# T115_roi_lstm_baseline: ROI Time-Series LSTM/GRU Baseline

## Task Description

Train a bi-directional LSTM (or GRU) over ROI BOLD time-series with mean-pooling readout. Strong temporal-but-not-graph baseline against STAGIN / BolT.

## Input Requirement

Required input(s):

- ROI BOLD time-series (NPZ)
- Subject list and labels CSV

If any required input is missing, return:

- Missing required input

## Constraints

- BiLSTM hidden 64-128; 1-2 layers; dropout 0.2.
- Mean-pool over time -> linear head.
- 5-fold deterministic split shared with T101.
- Save artefacts under `models/benchmark_results/T115_roi_lstm/<setting>/`.

## Expected Output

- Per-fold test metrics CSV
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- Compared against STAGIN (T108) and BolT (T109) to isolate the contribution of explicit temporal-attention.
