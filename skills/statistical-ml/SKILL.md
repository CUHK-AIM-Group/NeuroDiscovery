---
name: statistical-ml
description: "Use this skill whenever the user needs classical statistical inference or tabular machine learning for neuroimaging-derived features: OLS/GLM, Cohen's d, logistic regression, Ridge, Elastic Net, SVM/SVR, XGBoost, mixed-effects models, site-aware cross-validation, or imaging-by-treatment interactions. Triggers include 'statistical model', 'GLM', 'OLS', 'effect size', 'Cohen d', 'logistic regression', 'Ridge', 'Elastic Net', 'SVM', 'SVR', 'XGBoost', 'mixed effects', 'site split', 'tabular neuroimaging features', and 'imaging treatment interaction'."
license: MIT
layer: base
skill_type: model
dependencies:
  - run_models
  - harmonization-tool
---
# Statistical ML Skill

## Overview

`statistical-ml` is the shared NeuroClaw implementation for classical prediction
and inference on subject-level tabular features. It keeps imputation, scaling,
feature selection, and model fitting inside each training fold.

**Supported estimators**

| Task | Models |
|---|---|
| Classification | `logistic`, `ridge`, `elastic_net`, `svm`, `xgboost` |
| Regression | `ols`, `ridge`, `elastic_net`, `svr`, `xgboost` |
| Inference | Cohen's d, robust formula OLS/GLM, linear mixed-effects models |
| Longitudinal treatment analysis | dose/time/treatment mixed-effects formulas |

Use this skill after imaging data have been converted into one row per subject
or observation. Use `temporal-models` for sequence tensors and
`cnn3d` for full 3D volumes.

---

## Installation

Core dependencies are installed with NeuroClaw:

```bash
pip install numpy pandas scipy scikit-learn statsmodels joblib
```

XGBoost is optional:

```bash
pip install xgboost
```

Verify imports:

```bash
python -c "import sklearn, statsmodels; print('Statistical ML OK')"
```

---

## Workflows

### 1. Prepare a tabular CSV

The CSV must contain a subject identifier, target, and numeric features:

```text
subject_id,site,diagnosis,age,roi_001,roi_002,network_fc
sub-001,A,0,24,0.12,-0.04,0.31
sub-002,B,1,31,0.08,-0.09,0.27
```

Identifier, target, and optional group columns are excluded from predictors.

### 2. Classification

```bash
python skills/statistical-ml/scripts/train_reference.py \
  --features features.csv \
  --target diagnosis \
  --subject-col subject_id \
  --group-col site \
  --model logistic \
  --task classification \
  --folds 5 \
  --output-dir run_models_output/logistic
```

Use `--group-col site` or another cohort column when sites must not be split
between training and test folds.

### 3. Regression

```bash
python skills/statistical-ml/scripts/train_reference.py \
  --features features.csv \
  --target cognitive_score \
  --model ridge \
  --task regression \
  --folds 5 \
  --output-dir run_models_output/ridge
```

### 4. Dry-run schema validation

```bash
python skills/statistical-ml/scripts/train_reference.py \
  --features features.csv \
  --target diagnosis \
  --model svm \
  --task classification \
  --output-dir run_models_output/check \
  --dry-run
```

### 5. Formula-based inference

Use the Python APIs in `models/statistical_ml/mixed_effects.py` for robust OLS,
mixed-effects, and dose-response analyses. Report the formula, grouping
variable, effect estimate, confidence interval, and multiplicity correction.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | CSV; one row per subject or observation |
| Required columns | subject ID, target, numeric features |
| Optional column | grouping/site column |
| Predictions | `predictions.csv` |
| Fold membership | `fold_assignments.csv` |
| Metrics and config | `metrics.json`, `config.json` |
| Fitted estimators | `checkpoint.joblib` |
| Provenance | `run_manifest.json` |

Classification reports AUROC, AUPRC, accuracy, and balanced accuracy where
defined. Regression reports MAE, RMSE, and correlation metrics.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/statistical-ml/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/statistical_ml/
├── estimators.py       estimator factory and preprocessing pipelines
├── mixed_effects.py    OLS, effect-size, and mixed-effects utilities
└── train.py            cross-validated CLI

skills/statistical-ml/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Scikit-learn pipelines are used to prevent preprocessing leakage.
- Statsmodels provides formula-based OLS/GLM and mixed-effects inference.
- XGBoost remains optional and is loaded only when requested.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
