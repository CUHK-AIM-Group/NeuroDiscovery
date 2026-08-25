---
name: neuroimaging-decoding
description: "Use this skill whenever the user needs multivariate neuroimaging decoding or spatial statistical maps from ROI or voxel data. It supports ROI MVPA, mass-univariate ROI GLM, and voxel-wise Nilearn SearchLight analysis. Triggers include 'MVPA', 'decoding', 'ROI classifier', 'ROI GLM', 'mass univariate', 'searchlight', 'voxel-wise decoding', 'task fMRI decoding', and 'brain activation classification'."
license: MIT
layer: base
skill_type: workflow
dependencies:
  - glm
  - nilearn-tool
  - statistical-ml
---
# Neuroimaging Decoding Workflow

## Overview

`neuroimaging-decoding` coordinates three complementary analyses:

| Mode | Input | Scientific output |
|---|---|---|
| `mvpa` | ROI/parcel feature CSV | cross-validated prediction |
| `roi-glm` | ROI feature CSV + design CSV | ROI-wise effect and FDR table |
| `searchlight` | aligned NIfTI images + mask | voxel-wise decoding map |

Use `nilearn-tool` for full first-level and second-level task-fMRI GLM design.
This skill handles the downstream ROI or SearchLight analysis.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn statsmodels nilearn nibabel
```

---

## Workflows

### 1. ROI MVPA

```bash
python skills/neuroimaging-decoding/scripts/train_reference.py \
  --mode mvpa \
  --features roi_features.csv \
  --target diagnosis \
  --subject-col subject_id \
  --task classification \
  --model svm \
  --folds 5 \
  --output-dir run_models_output/mvpa
```

The tabular estimator choices are inherited from `statistical-ml`. Scaling and
feature selection must remain inside cross-validation.

### 2. ROI-wise GLM

`roi_features.csv` contains subject ID plus ROI columns. `design.csv` contains
the same subject ID plus intercept/covariate/contrast columns.

```bash
python skills/neuroimaging-decoding/scripts/train_reference.py \
  --mode roi-glm \
  --features roi_features.csv \
  --design design.csv \
  --subject-col subject_id \
  --contrast-index 1 \
  --output-dir run_models_output/roi_glm
```

The output includes effect, standard error, P value, and FDR-corrected Q value
for every ROI.

### 3. Voxel-wise SearchLight

Create `images.txt` with one aligned NIfTI path per line. The row order must
match the labels CSV.

```bash
python skills/neuroimaging-decoding/scripts/train_reference.py \
  --mode searchlight \
  --images-list images.txt \
  --features labels.csv \
  --target condition \
  --mask group_mask.nii.gz \
  --folds 5 \
  --output-dir run_models_output/searchlight
```

All images and the mask must share the same space, affine, and voxel grid.

---

## Input / Output Summary

| Mode | Output |
|---|---|
| MVPA | standard prediction, fold, metric, checkpoint artifacts |
| ROI GLM | `roi_glm_results.csv`, `metrics.json` |
| SearchLight | `searchlight_scores.nii.gz`, `metrics.json` |
| All modes | `config.json`, `run_manifest.json` |

Report atlas/space metadata for ROI analyses and mask/voxel resolution for
SearchLight analyses.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/neuroimaging-decoding/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/neuroimaging_decoding/
├── roi_glm.py          ROI-wise statistical tests and FDR
├── searchlight.py      Nilearn SearchLight adapter
└── train.py            unified decoding CLI

skills/neuroimaging-decoding/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Nilearn provides the voxel-wise SearchLight implementation.
- The ROI GLM uses explicit design matrices and Benjamini-Hochberg correction.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
