---
name: cpm
description: "Use this model skill whenever the user wants Connectome Predictive Modeling with fold-local functional-connectivity edge selection for classification or regression. Triggers include 'CPM', 'connectome predictive modeling', 'functional connectivity prediction', 'positive network', 'negative network', and 'edge selection'."
license: MIT
layer: base
skill_type: model
dependencies:
  - fmri-skill
  - run_models
---
# CPM Skill

## Overview

`cpm` is the canonical NeuroClaw implementation of Connectome Predictive
Modeling. Edge selection is repeated independently inside every training fold.

| Task | Input | Output |
|---|---|---|
| Classification | subject FC matrices/vectors | class and probability |
| Regression | subject FC matrices/vectors | continuous prediction |

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn joblib
```

---

## Workflows

### 1. Prepare data

`connectomes.npz`:

```text
X:          [subjects, nodes, nodes] or [subjects, edges]
subject_id: [subjects]
```

`labels.csv` contains the same subject IDs and a target column.

### 2. Regression

```bash
python skills/cpm/scripts/train_reference.py \
  --connectomes connectomes.npz \
  --labels labels.csv \
  --target cognitive_score \
  --subject-col subject_id \
  --task regression \
  --p-threshold 0.01 \
  --folds 5 \
  --output-dir run_models_output/cpm
```

### 3. Classification

```bash
python skills/cpm/scripts/train_reference.py \
  --connectomes connectomes.npz \
  --labels labels.csv \
  --target diagnosis \
  --task classification \
  --p-threshold 0.01 \
  --output-dir run_models_output/cpm_classification
```

If `p-threshold` is tuned, use nested validation or training-only selection.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Connectomes | `.npz` with `X`, `subject_id` |
| Labels | CSV keyed by subject ID |
| Predictions | `predictions.csv` |
| Fold membership | `fold_assignments.csv` |
| Metrics | `metrics.json` |
| Fold models | `checkpoint.joblib` |
| Provenance | `config.json`, `run_manifest.json` |

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/cpm/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/cpm/
├── cpm.py              fold-local CPM estimator
└── train.py            cross-validated CLI

skills/cpm/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Finn et al. functional connectome fingerprinting and connectome-based
  prediction framework, Nature Neuroscience (2015).

---

Created At: 2026-07-29 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
