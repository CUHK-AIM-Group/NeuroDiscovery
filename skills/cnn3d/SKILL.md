---
name: cnn3d
description: "Use this model skill whenever the user wants a compact residual 3D convolutional neural network for voxel-level classification or regression from structural MRI, functional MRI summaries, statistical maps, or other aligned volumetric neuroimaging data. Triggers include 'CNN3D', '3D CNN', 'voxel model', 'volumetric MRI', 'whole-brain volume classification', 'sMRI deep learning', and 'voxel regression'."
license: MIT
layer: base
skill_type: model
dependencies:
  - fmri-skill
  - smri-skill
  - run_models
---
# CNN3D Skill

## Overview

`cnn3d` is NeuroClaw's canonical compact residual 3D CNN. It owns one model,
one NPZ input contract, and one checkpoint format. NeuroSTORM remains a
separate model skill with its own external repository and runtime.

| Model | Input | Tasks |
|---|---|---|
| `VoxelCNN3D` | whole-volume tensor | classification, regression |

---

## Installation

```bash
pip install numpy torch scikit-learn pandas
```

Verify:

```bash
python -c "from models.cnn3d import VoxelCNN3D; print('CNN3D OK')"
```

---

## Workflows

### 1. Prepare a volume NPZ

```text
X:          float array [subjects, channels, depth, height, width]
y:          array [subjects]
subject_id: string array [subjects] (optional)
```

All subjects must use the same orientation, voxel size, grid, crop, and
intensity-normalization protocol.

### 2. Classification

```bash
python skills/cnn3d/scripts/train_reference.py \
  --input volumes.npz \
  --task classification \
  --base-channels 16 \
  --dropout 0.1 \
  --epochs 50 \
  --batch-size 4 \
  --folds 5 \
  --device cuda \
  --output-dir run_models_output/cnn3d
```

### 3. Regression

```bash
python skills/cnn3d/scripts/train_reference.py \
  --input volumes.npz \
  --task regression \
  --base-channels 32 \
  --epochs 100 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --output-dir run_models_output/cnn3d_regression
```

Preprocessing must be frozen before cross-validation. Site harmonization,
augmentation, and intensity transforms must not use held-out subjects.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | `.npz` with `X`, `y`, optional `subject_id` |
| Predictions | `predictions.csv` |
| Fold membership | `fold_assignments.csv` |
| Metrics | `metrics.json` |
| Fold checkpoints | `checkpoint.pt` |
| Provenance | `config.json`, `run_manifest.json` |

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/cnn3d/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/cnn3d/
├── net.py              residual 3D CNN
└── train.py            cross-validated trainer

skills/cnn3d/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Use `neurostorm` instead when the request explicitly targets NeuroSTORM,
  SwiFT, or the upstream multi-model fMRI platform.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
