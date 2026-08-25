---
name: imaging-genetics-models
description: "Use this skill whenever the user needs imaging-genetics analysis: variant-imaging association scans, kinship-aware linear mixed models, polygenic or pathway scores, PLS/CCA links between genotype and imaging phenotypes, or audited PLINK2 command construction. Triggers include 'imaging genetics', 'GWAS', 'PLINK2', 'variant association', 'kinship', 'LMM', 'PRS', 'polygenic score', 'PLS', 'CCA', 'genotype imaging phenotype', and 'pathway score'."
license: MIT
layer: base
skill_type: model
dependencies:
  - statistical-ml
  - smri-skill
  - fmri-skill
---
# Imaging Genetics Models Skill

## Overview

`imaging-genetics-models` provides matrix-based association, polygenic scoring,
and multivariate genotype-imaging analysis. It also builds explicit PLINK2
commands without embedding or redistributing the external executable.

**Supported modes**

| Mode | Required arrays | Output |
|---|---|---|
| `association` | `genotype`, `phenotype` | covariate-adjusted variant tests |
| `lmm` | above plus `kinship` | kinship-aware variant tests |
| `prs` | `genotype`, `weights` | subject polygenic score |
| `pls` | `X`, `Y` | paired latent components |
| `cca` | `X`, `Y` | canonical variates |

Population structure, ancestry, batch, age, sex, site, and relatedness must be
handled before genetic effects are interpreted.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn statsmodels joblib
```

For genome-wide command-line analyses, install PLINK2 separately and execute
the generated command through NeuroClaw's audited shell workflow.

---

## Workflows

### 1. Variant-imaging association

Create an NPZ bundle:

```text
genotype:   [subjects, variants]
phenotype:  [subjects] or [subjects, phenotypes]
variant_id: [variants] (optional)
covariates: [subjects, covariates] (optional)
kinship:    [subjects, subjects] (required only for `lmm`)
```

```bash
python skills/imaging-genetics-models/scripts/train_reference.py \
  --input imaging_genetics.npz \
  --model association \
  --output-dir run_models_output/imaging_gwas
```

Use `--model lmm` when the bundle contains a kinship matrix.

### 2. Polygenic score

```text
genotype:   [subjects, variants]
weights:    [variants]
subject_id: [subjects] (optional)
```

```bash
python skills/imaging-genetics-models/scripts/train_reference.py \
  --input prs_bundle.npz \
  --model prs \
  --output-dir run_models_output/prs
```

### 3. PLS or CCA

```text
X: [subjects, genetic features]
Y: [subjects, imaging phenotypes]
```

```bash
python skills/imaging-genetics-models/scripts/train_reference.py \
  --input imaging_genetics.npz \
  --model cca \
  --components 3 \
  --output-dir run_models_output/cca
```

Fit dimensionality reduction and covariate residualization inside the training
data when the analysis is evaluated predictively.

---

## Input / Output Summary

| Mode | Main output |
|---|---|
| Association/LMM | `association_results.csv` |
| PRS | `predictions.csv` with `polygenic_score` |
| PLS/CCA | `predictions.csv` with paired component scores |
| PLS/CCA model | `checkpoint.joblib` |
| All modes | `metrics.json`, `config.json`, `run_manifest.json` |

Never split related individuals across train and test folds. Record genome
build, allele orientation, QC thresholds, ancestry definition, and phenotype
construction with every analysis.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/imaging-genetics-models/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/imaging_genetics/
├── association.py      association, LMM, and polygenic score utilities
├── multivariate.py     PLS and CCA
├── plink.py            audited PLINK2 command builder
└── train.py            matrix-analysis CLI

skills/imaging-genetics-models/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- PLINK2 remains an external executable with independent installation terms.
- Association outputs include effect estimates and multiplicity-ready P values.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
