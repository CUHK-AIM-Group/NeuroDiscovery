---
name: temporal-models
description: "Use this skill whenever the input contains ordered repeated measurements, longitudinal visits, ROI time series, dynamic connectivity features, or variable-length sequences. It supports LSTM, GRU, temporal convolutional networks, and temporal Transformers for classification and regression. Triggers include 'longitudinal model', 'sequence model', 'time series', 'repeated visits', 'LSTM', 'GRU', 'TCN', 'temporal Transformer', 'dynamic connectivity', and 'variable-length sequence'."
license: MIT
layer: base
skill_type: model
dependencies:
  - fmri-skill
  - run_models
---
# Temporal Models Skill

## Overview

`temporal-models` trains sequence encoders on ordered neuroimaging or clinical
measurements. It supports variable sequence lengths and estimates normalization
statistics from training subjects only.

**Supported models**

| Model | Encoder | Typical use |
|---|---|---|
| `lstm` | long short-term memory | longitudinal visits |
| `gru` | gated recurrent unit | compact recurrent baseline |
| `tcn` | temporal convolutional network | local temporal patterns |
| `transformer` | masked temporal self-attention | longer dependencies |

Both `classification` and `regression` are supported.

---

## Installation

```bash
pip install numpy torch scikit-learn pandas
```

Verify:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Workflows

### 1. Prepare an NPZ sequence bundle

```text
X:          float array [subjects, time, features]
y:          array [subjects]
lengths:    integer array [subjects] (optional)
subject_id: string array [subjects] (optional)
```

Padded frames must occur after each valid sequence. If `lengths` is absent,
every sequence is treated as fully valid.

```python
import numpy as np

np.savez(
    "sequences.npz",
    X=X.astype("float32"),
    y=y,
    lengths=lengths,
    subject_id=subject_ids,
)
```

### 2. Classification with GRU

```bash
python skills/temporal-models/scripts/train_reference.py \
  --input sequences.npz \
  --model gru \
  --task classification \
  --hidden-dim 64 \
  --layers 2 \
  --epochs 100 \
  --batch-size 32 \
  --folds 5 \
  --device cuda \
  --output-dir run_models_output/gru
```

### 3. Regression with temporal Transformer

```bash
python skills/temporal-models/scripts/train_reference.py \
  --input sequences.npz \
  --model transformer \
  --task regression \
  --hidden-dim 128 \
  --layers 3 \
  --dropout 0.2 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --output-dir run_models_output/temporal_transformer
```

Use subject-level folds; never split frames or visits from one subject across
training and test sets.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | `.npz` with `X`, `y`, optional `lengths`, `subject_id` |
| Predictions | `predictions.csv` |
| Fold membership | `fold_assignments.csv` |
| Metrics | `metrics.json` |
| Fold checkpoints | `checkpoint.pt` |
| Provenance | `config.json`, `run_manifest.json` |

Each fold checkpoint stores the model state plus fold-local feature mean and
standard deviation.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/temporal-models/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/temporal_models/
├── net.py              LSTM, GRU, TCN, and Transformer encoders
└── train.py            cross-validated sequence trainer

skills/temporal-models/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Sequence masking is driven by the optional `lengths` array.
- Scaling is fitted from valid training frames and padding is restored to zero.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
