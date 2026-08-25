---
name: brainnetcnn
description: "Use this model skill whenever the user wants to train, evaluate, or apply BrainNetCNN to dense ROI-by-ROI structural or functional connectivity matrices for neuroimaging classification or regression. Triggers include 'BrainNetCNN', 'edge-to-edge convolution', 'connectome CNN', 'FC matrix CNN', 'brain network classification', and 'connectivity regression'."
license: MIT
layer: base
skill_type: model
dependencies:
  - fmri-skill
  - run_models
---
# BrainNetCNN Model Skill

## Overview

BrainNetCNN applies convolutional operators designed for adjacency matrices:
edge-to-edge (E2E), edge-to-node (E2N), and node-to-graph (N2G). Use it when
each subject is represented by a dense, consistently ordered ROI connectivity
matrix and the target is categorical or continuous.

- Paper: Kawahara et al., 2017, *BrainNetCNN: Convolutional neural networks for
  brain networks; towards predicting neurodevelopment*
- NeuroClaw implementation: `models/brainnetcnn/`
- Input: dense FC matrix `[subjects, ROI, ROI]`
- Tasks: classification and regression
- Data adapter: shared with BNT

**Research use only.**

---

## Input Contract

Prepare one file per subject:

```text
data/braingnn_input/<atlas>/sub-<subject_id>.pt
```

Each file must contain:

```python
{
    "subject_id": str,
    "atlas": str,
    "fc_matrix": Tensor[n_roi, n_roi],  # Fisher-z values
    "node_features": Tensor[n_roi, n_roi],  # accepted fallback
}
```

The shared BNT adapter applies `tanh` to recover Pearson correlations and
zeros the diagonal. All subjects in one run must use the same atlas, ROI
ordering, and matrix size.

Labels use CSV format:

```text
subject_id,label
100001,0
100002,1
```

Change the columns with `--subject-col` and `--label-col`.

---

## Quick Start

### Validate data loading

```bash
python skills/brainnetcnn/scripts/train_reference.py \
  --atlas schaefer_100_7net \
  --labels-csv data/hcp_gender_labels.csv \
  --dry-run
```

### Classification

```bash
python skills/brainnetcnn/scripts/train_reference.py \
  --atlas schaefer_100_7net \
  --labels-csv data/hcp_gender_labels.csv \
  --task classification \
  --nclass 2 \
  --fold 0 \
  --kfold 5 \
  --n-epochs 100 \
  --batch-size 16 \
  --device cuda
```

### Regression

```bash
python skills/brainnetcnn/scripts/train_reference.py \
  --atlas aal_116 \
  --labels-csv data/hcp_age_labels.csv \
  --label-col age \
  --task regression \
  --fold 0 \
  --kfold 5 \
  --n-epochs 100 \
  --batch-size 16 \
  --device cuda
```

Use subject-level folds. If data come from multiple sites, families, or
repeated visits, construct group-aware splits before interpreting results.

---

## Architecture

```text
Dense connectivity matrix [B, 1, N, N]
  -> E2E convolution blocks
  -> E2N convolution
  -> N2G convolution
  -> fully connected prediction head
  -> class logits or one regression value
```

| Parameter | Default | Meaning |
|---|---:|---|
| `--e2e-channels` | 32 | E2E feature maps |
| `--e2n-channels` | 64 | E2N feature maps |
| `--n2g-channels` | 256 | graph-level representation |
| `--dropout` | 0.5 | prediction-head dropout |
| `--lr` | 0.001 | Adam learning rate |
| `--weight-decay` | 0.0005 | L2 regularization |
| `--kfold` | 5 | subject-level folds |

---

## Outputs

The reference trainer writes:

```text
models/brainnetcnn/checkpoints/<atlas>/fold<fold>.pt
```

The checkpoint contains the model state, resolved arguments, ROI count, and
best fold metric. Keep checkpoints and experiment logs ignored by Git.

---

## Delegation Rules

- Delegate ROI extraction and FC computation to `fmri-skill`.
- Delegate model comparison and routing to `run_models`.
- Use `bnt` when attention and DEC assignments are required.
- Use `brain_gnn`, `ibgnn`, or `lggnn` when sparse/PyG graph operations or
  graph-specific explanations are required.
- Use `cpm` for a transparent, low-parameter connectome baseline.

---

## Testing

```bash
python skills/brainnetcnn/scripts/train_reference.py --help
pytest models/tests/test_extended_models.py -q
```

---

## Directory Reference

```text
models/brainnetcnn/
├── net/brainnetcnn.py
└── scripts/
    ├── data_adapter.py
    └── train.py

skills/brainnetcnn/
├── SKILL.md
├── agents/openai.yaml
└── scripts/train_reference.py
```

---

## Reference

- Kawahara J, Brown CJ, Miller SP, et al. BrainNetCNN: Convolutional neural
  networks for brain networks; towards predicting neurodevelopment.
  *NeuroImage*. 2017;146:1038-1049.
- Official implementation: https://github.com/jeremykawahara/brainnetcnn

Created At: 2026-07-31 14:24:19 HKT
Last Updated At: 2026-07-31 14:24:19 HKT
Author: chengwang96
