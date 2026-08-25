---
name: brain-age-modeling
description: "Use this skill whenever the user wants brain-age prediction, predicted-age bias correction, Brain-PAD/brain-age gap export, cross-validated age modeling, or downstream group analysis of accelerated or delayed brain aging. Triggers include 'brain age', 'Brain-PAD', 'brain age gap', 'predicted age', 'age bias correction', 'accelerated aging', and 'neuroimaging age model'."
license: MIT
layer: base
skill_type: workflow
dependencies:
  - statistical-ml
  - cnn3d
  - run_models
---
# Brain-Age Modeling Workflow

## Overview

`brain-age-modeling` is a leakage-safe task workflow over NeuroClaw regression
estimators. It trains predicted-age models, fits age-bias correction on each
training fold, and exports held-out raw age, corrected age, and Brain-PAD.

```text
Brain-PAD = bias-corrected predicted age - chronological age
```

Positive Brain-PAD indicates an older-appearing brain relative to chronological
age under the fitted model; it is not by itself a diagnosis or causal effect.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn joblib
```

Optional feature generators such as FreeSurfer, NeuroSTORM, or a 3D CNN are
handled by their own skills before this tabular brain-age workflow.

---

## Workflows

### 1. Prepare brain features

```text
subject_id,site,age,cortical_thickness,hippocampal_volume,fc_001
sub-001,A,64,2.51,3810,0.12
sub-002,B,59,2.63,4022,0.08
```

Use a healthy training reference when the scientific interpretation requires
deviation from normative aging. Do not include downstream disease outcomes as
predictors.

### 2. Ridge brain-age model

```bash
python skills/brain-age-modeling/scripts/train_reference.py \
  --features brain_features.csv \
  --age-col age \
  --subject-col subject_id \
  --group-col site \
  --model ridge \
  --folds 5 \
  --seed 123 \
  --output-dir run_models_output/brain_age
```

### 3. Alternative regressors

The workflow reuses regression estimators from `statistical-ml`, including
`ols`, `ridge`, `elastic_net`, `svr`, and optional `xgboost`. Keep site,
family, or cohort groups intact where appropriate.

### 4. Downstream analysis

After held-out Brain-PAD has been generated, analyze group differences or
clinical associations with explicit age, sex, site, intracranial-volume, and
other prespecified covariates. Use only held-out Brain-PAD values.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | CSV with subject, chronological age, and numeric brain features |
| Optional grouping | site/cohort/family column |
| Predictions | `predictions.csv` |
| Prediction columns | raw age, corrected age, Brain-PAD |
| Fold membership | `fold_assignments.csv` |
| Metrics | raw and bias-corrected metrics in `metrics.json` |
| Checkpoint | predictor and corrector per fold in `checkpoint.joblib` |
| Provenance | `config.json`, `run_manifest.json` |

The bias corrector is fitted from chronological age and predictions in the
training fold only, then applied to the held-out fold.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/brain-age-modeling/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/brain_age/
├── correction.py       fold-local predicted-age bias correction
└── train.py            cross-validated brain-age workflow

skills/brain-age-modeling/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Brain-PAD should always be reported together with the training population,
  feature family, validation design, and bias-correction procedure.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
