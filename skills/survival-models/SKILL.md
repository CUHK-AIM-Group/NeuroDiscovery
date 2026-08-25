---
name: survival-models
description: "Use this skill whenever the outcome is right-censored time-to-event data, including progression, conversion, relapse, hospitalization, or mortality prediction. It supports Cox proportional hazards, Random Survival Forest, DeepSurv, and XGBoost Cox survival with censor-aware cross-validation. Triggers include 'survival analysis', 'time to event', 'censored outcome', 'Cox model', 'hazard ratio', 'Random Survival Forest', 'DeepSurv', 'XGBoost survival', 'progression', and 'conversion risk'."
license: MIT
layer: base
skill_type: model
dependencies:
  - statistical-ml
  - run_models
---
# Survival Models Skill

## Overview

`survival-models` trains censor-aware prognosis models from subject-level
features. Every sample requires a follow-up duration and an event indicator;
censored observations must not be converted into ordinary regression labels.

**Supported models**

| Model | Implementation | Dependency |
|---|---|---|
| `cox` | proportional hazards baseline | core |
| `rsf` | Random Survival Forest | `scikit-survival` |
| `deepsurv` | neural Cox risk model | PyTorch |
| `xgboost_survival` | XGBoost `survival:cox` | `xgboost` |

The primary metric is Harrell's concordance index. The exported prediction is
a relative risk score, not an absolute probability unless separately
calibrated at a specified time horizon.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn joblib torch
```

Optional estimators:

```bash
pip install scikit-survival
pip install xgboost
```

---

## Workflows

### 1. Prepare survival data

```text
subject_id,site,followup_days,progressed,roi_001,roi_002,age
sub-001,A,730,1,0.12,-0.04,64
sub-002,B,910,0,0.08,-0.09,59
```

`event-col` must be binary (`1` observed event, `0` censored). Duration must be
positive and use one consistent unit.

### 2. Cox proportional hazards

```bash
python skills/survival-models/scripts/train_reference.py \
  --features prognosis.csv \
  --duration-col followup_days \
  --event-col progressed \
  --subject-col subject_id \
  --group-col site \
  --model cox \
  --folds 5 \
  --output-dir run_models_output/cox
```

### 3. DeepSurv

```bash
python skills/survival-models/scripts/train_reference.py \
  --features prognosis.csv \
  --duration-col followup_days \
  --event-col progressed \
  --model deepsurv \
  --epochs 200 \
  --device cuda \
  --output-dir run_models_output/deepsurv
```

### 4. Tree-based survival models

Set `--model rsf` or `--model xgboost_survival`. Use grouped folds for
multi-site cohorts and report event counts per fold in addition to sample
counts.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | CSV with subject, duration, event, and features |
| Optional grouping | site/cohort/family column |
| Predictions | `predictions.csv` with risk score |
| Fold membership | `fold_assignments.csv` |
| Metrics | `metrics.json` with concordance |
| Checkpoint | `checkpoint.joblib` |
| Provenance | `config.json`, `run_manifest.json` |

Before interpreting risk, check proportional-hazards assumptions for Cox,
event prevalence, follow-up distribution, and calibration at clinically
meaningful horizons.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/survival-models/scripts/train_reference.py --help
```

Optional dependency tests are skipped when the corresponding package is not
installed.

---

## Directory Reference

```text
models/survival_models/
├── estimators.py       Cox, RSF, DeepSurv, and XGBoost adapters
├── metrics.py          censor-aware concordance utilities
└── train.py            cross-validated CLI

skills/survival-models/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Katzman et al. DeepSurv, BMC Medical Research Methodology (2018).
- Random Survival Forest and XGBoost are optional third-party backends.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
