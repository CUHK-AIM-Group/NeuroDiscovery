---
name: subject-subtyping
description: "Use this skill whenever the user wants unsupervised disease subtyping, patient stratification, latent phenotype discovery, cluster stability analysis, or low-dimensional embeddings from neuroimaging features. It supports K-means, Gaussian mixture models, spectral clustering, NMF, consensus clustering, PCA embeddings, and autoencoder embeddings. Triggers include 'subtype', 'patient stratification', 'clustering', 'latent phenotype', 'consensus clustering', 'GMM', 'NMF', 'PCA embedding', and 'autoencoder clustering'."
license: MIT
layer: base
skill_type: model
dependencies:
  - statistical-ml
  - run_models
---
# Subject Subtyping Skill

## Overview

`subject-subtyping` discovers unsupervised subject groups from tabular imaging or
multimodal features. It exports subtype assignments, latent embeddings,
silhouette diagnostics, and a reusable checkpoint.

**Supported models**

| Model | Method | Typical use |
|---|---|---|
| `kmeans` | Euclidean partitioning | compact baseline |
| `gmm` | Gaussian mixture | soft distributional subtypes |
| `spectral` | graph spectral clustering | non-convex structure |
| `nmf` | non-negative embedding + K-means | parts-based phenotypes |
| `consensus` | bootstrap co-clustering | stability-focused analysis |
| `pca` | PCA embedding + K-means | linear latent subtypes |
| `autoencoder` | neural embedding + K-means | nonlinear latent subtypes |

Outcome labels must not be used to choose the number of clusters. Clinical
outcomes may be tested only after subtype definitions are frozen.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn joblib torch
```

Verify:

```bash
python -c "import sklearn, torch; print('Subtyping models OK')"
```

---

## Workflows

### 1. Prepare features

Input is a CSV with `subject_id` and numeric features. Do not include diagnosis,
survival, or treatment outcome columns among the clustering features.

```text
subject_id,roi_001,roi_002,network_fc,brain_age_gap
sub-001,0.12,-0.04,0.31,2.1
sub-002,0.08,-0.09,0.27,-1.4
```

### 2. Consensus clustering

```bash
python skills/subject-subtyping/scripts/train_reference.py \
  --features features.csv \
  --subject-col subject_id \
  --model consensus \
  --n-clusters 3 \
  --seed 123 \
  --output-dir run_models_output/subtyping_consensus
```

### 3. PCA or autoencoder embeddings

```bash
python skills/subject-subtyping/scripts/train_reference.py \
  --features features.csv \
  --model pca \
  --n-clusters 4 \
  --latent-dim 8 \
  --output-dir run_models_output/subtyping_pca
```

```bash
python skills/subject-subtyping/scripts/train_reference.py \
  --features features.csv \
  --model autoencoder \
  --n-clusters 4 \
  --latent-dim 8 \
  --epochs 200 \
  --output-dir run_models_output/subtyping_ae
```

### 4. Select the number of clusters

Run a prespecified range such as `k=2..8`, compare silhouette and bootstrap
stability, then freeze `k` before association with clinical endpoints. Repeat
the final model across seeds when cluster stability is central to the claim.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | CSV; one row per subject |
| Required | numeric feature columns |
| Optional | configurable subject ID column |
| Assignments | `predictions.csv` with subject and subtype |
| Embedding | latent dimensions in `predictions.csv` |
| Metrics | `metrics.json` including silhouette and cluster count |
| Model | `checkpoint.joblib` |
| Provenance | `config.json`, `run_manifest.json` |

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/subject-subtyping/scripts/train_reference.py --help
```

---

## Directory Reference

```text
models/subtyping/
├── estimators.py       clustering and embedding implementations
└── train.py            artifact-producing CLI

skills/subject-subtyping/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Consensus clustering uses bootstrap co-assignment frequencies.
- NMF inputs are transformed to a non-negative scale before decomposition.
- PCA and autoencoder modes cluster the learned embedding rather than raw data.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
