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
silhouette diagnostics, and a fitted estimator checkpoint. The reporting route
also exports patient-level QA, feature profiles, group sizes and figures.

Research use only: clusters are not clinical diagnoses. The legacy training
checkpoint does not bundle the training imputer/scaler and cannot by itself
assign new patients from raw features. Preserve the fitted preprocessing and
feature order in a separately validated deployment pipeline before future-patient
inference; spectral/consensus training does not define an out-of-sample rule.

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

### 5. Export the requested patient outputs

Use frozen assignments from the training route above, or an existing study; do
not retrain just to produce a report. Provide `feature_columns.json`, an explicit
JSON array of the prespecified imaging/multimodal features. Exclude IDs, site
identifiers and outcome/diagnosis variables from the feature set. Input feature
IDs must be unique, and assignments must have exactly the same subject set;
the report joins by ID, never by assumed row order.

```bash
python skills/subject-subtyping/scripts/report_subtypes.py \
  --features features.csv --feature-columns feature_columns.json \
  --assignments run_models_output/subtyping_consensus/predictions.csv \
  --output-dir run_models_output/subtyping_report
```

Outputs include `subtype_assignments.csv`, `subtype_profiles.csv`,
`subtype_counts.csv`, `embedding.csv`, `subtype_report.png`, `report.md`,
`qc.json`, descriptive preprocessing parameters and a verified manifest. Profiles
retain observed-value missingness. Standardization and fallback PCA are
**descriptive**, not a new trained subtype definition or evidence of replication.
There are no fabricated outcome associations or subtype confidence probabilities.

If completed resampling/seed runs exist, repeat `--replicate predictions_seed2.csv`
to add permutation-invariant adjusted Rand and optimally matched cluster Jaccard
scores in `stability.csv`. These runs must cover the same subjects and genuinely
represent the specified perturbation. Without them, stability is explicitly
`not_evaluated`; request/scopingly plan resampling if the user requires it.

```bash
python -m models.common.research_outputs run_models_output/subtyping_report/run_manifest.json \
  --require subtype_assignments subtype_profiles subtype_counts subtype_figure report
```

Use a new output directory for every run. With requested stability, add
`--require stability` to the final check. An absent requested output is incomplete
work, not an optional artifact silently omitted after training.

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
pytest models/tests/test_research_output_skills.py -q
```

---

## Directory Reference

```text
models/subtyping/
├── estimators.py       clustering and embedding implementations
└── train.py            artifact-producing CLI

skills/subject-subtyping/
├── SKILL.md
└── scripts/
    ├── train_reference.py
    └── report_subtypes.py
```

---

## Reference

- Consensus clustering uses bootstrap co-assignment frequencies.
- NMF inputs are transformed to a non-negative scale before decomposition.
- PCA and autoencoder modes cluster the learned embedding rather than raw data.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-09-14 15:05:31.978 HKT
Author: chengwang96
