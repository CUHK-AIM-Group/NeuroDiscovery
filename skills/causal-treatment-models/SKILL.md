---
name: causal-treatment-models
description: "Use this skill whenever the scientific target is a treatment effect rather than ordinary outcome prediction: propensity weighting, S/T/X learners, doubly robust learning, policy learning, causal forests, TARNet, DragonNet, CATE estimation, heterogeneous treatment effects, and individualized treatment selection. Triggers include 'causal inference', 'treatment effect', 'CATE', 'ATE', 'propensity score', 'IPW', 'doubly robust', 'causal forest', 'TARNet', 'DragonNet', and 'treatment policy'."
license: MIT
layer: base
skill_type: model
dependencies:
  - statistical-ml
  - run_models
---
# Causal Treatment Models Skill

## Overview

`causal-treatment-models` estimates average or conditional treatment effects
from observational subject-level features. It is for the contrast
`Y(1) - Y(0)`, not for predicting the observed outcome alone.

**Supported estimators**

| Model | Output |
|---|---|
| `ipw` | propensity-weighted ATE as constant CATE |
| `s_learner` | single outcome model treatment contrast |
| `t_learner` | separate treated/control outcome models |
| `x_learner` | imputed effects blended by propensity |
| `doubly_robust` | doubly robust pseudo-outcome CATE |
| `policy_learner` | interpretable treatment assignment policy |
| `causal_forest` | `econml` CausalForestDML |
| `tarnet` | shared representation with two outcome heads |
| `dragonnet` | TARNet plus propensity head |

The CLI uses cross-fitted held-out predictions. Causal interpretation still
requires consistency, positivity, no unmeasured confounding, and a defensible
temporal ordering.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn joblib torch
```

For Causal Forest:

```bash
pip install econml
```

---

## Workflows

### 1. Prepare treatment data

```text
subject_id,treatment,response,age,sex,baseline_score,roi_001
sub-001,1,4.2,64,0,18.0,0.12
sub-002,0,1.7,59,1,17.5,0.08
```

Treatment must be binary for the current CLI. Include only pretreatment
covariates in the feature matrix.

### 2. Doubly robust CATE

```bash
python skills/causal-treatment-models/scripts/train_reference.py \
  --features treatment.csv \
  --treatment-col treatment \
  --outcome-col response \
  --subject-col subject_id \
  --model doubly_robust \
  --folds 5 \
  --output-dir run_models_output/treatment_dr
```

### 3. Neural treatment-effect model

```bash
python skills/causal-treatment-models/scripts/train_reference.py \
  --features treatment.csv \
  --treatment-col treatment \
  --outcome-col response \
  --model dragonnet \
  --epochs 200 \
  --device cuda \
  --output-dir run_models_output/dragonnet
```

### 4. Causal Forest or policy learning

Use `--model causal_forest` for nonlinear heterogeneous effects and
`--model policy_learner` for a compact treatment rule. Report overlap,
propensity distributions, standardized mean differences, ATE/CATE uncertainty,
and policy value under an explicit evaluation design.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | CSV with subject, treatment, outcome, pretreatment covariates |
| Treatment | binary `0/1` |
| Predictions | `predictions.csv` with held-out CATE and policy |
| Fold membership | `fold_assignments.csv` |
| Metrics | `metrics.json` with effect/policy summaries |
| Checkpoint | `checkpoint.joblib` |
| Provenance | `config.json`, `run_manifest.json` |

Do not describe a high predictive score as causal evidence. Negative controls,
sensitivity analyses, and randomized validation remain separate requirements.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/causal-treatment-models/scripts/train_reference.py --help
```

`causal_forest` tests require the optional `econml` installation.

---

## Directory Reference

```text
models/causal_treatment/
├── estimators.py       IPW, meta-learners, DR, policy, forest, neural models
└── train.py            cross-fitted CLI

skills/causal-treatment-models/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Shalit et al. representation learning for individual treatment effects (2017).
- Shi et al. DragonNet for targeted regularization (2019).
- `econml` provides the optional CausalForestDML backend.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
