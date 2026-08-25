---
name: connectome-discovery
description: "Use this workflow skill whenever the user wants to turn connectome-model outputs into network discoveries: compare aligned brain maps, compute permutation significance, rank neuromodulation targets, summarize atlas-level effects, or combine CPM with spatial interpretation. Triggers include 'connectome discovery', 'network map similarity', 'permutation P value', 'target ranking', 'neuromodulation target', and 'spatial connectome interpretation'."
license: MIT
layer: base
skill_type: workflow
dependencies:
  - cpm
  - fmri-skill
  - brain-visualization
---
# Connectome Discovery Workflow

## Overview

`connectome-discovery` is a scientific interpretation workflow, not a second
CPM implementation. It consumes fitted-model outputs or aligned network maps
and produces map similarities, empirical significance, and candidate target
rankings.

| Stage | Canonical owner |
|---|---|
| Connectome prediction | `cpm`, BrainGNN, BNT, or another model skill |
| Atlas/space validation | `fmri-skill`, `nibabel-skill` |
| Map similarity and permutation | `models/connectome_discovery/mapping.py` |
| Surface/network rendering | `brain-visualization` |

---

## Installation

```bash
pip install numpy scipy pandas
```

---

## Workflows

### 1. Generate model evidence

Run `cpm` or another connectome model and freeze its held-out predictions,
selected edges, atlas, and node ordering.

### 2. Align maps

Reference and candidate maps must use the same atlas, node order, hemisphere
convention, and value orientation. Resampling or atlas mapping must be recorded.

### 3. Score and rank targets

Use `models/connectome_discovery/mapping.py` for:

- `cosine_similarity_map`
- `permutation_pvalue`
- `rank_targets`

Save the observed score, null distribution settings, permutation count, random
seed, atlas, and coordinate space.

### 4. Visualize

Route final ROI/network values to `brain-visualization`. Do not infer an
anatomical target from an unlabeled edge vector.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | aligned ROI/network maps or model-derived connectome signatures |
| Statistics | cosine similarity and empirical permutation P value |
| Ranking | target identifier, similarity, rank, atlas/space metadata |
| Visualization | publication-ready network or surface map |

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
```

---

## Directory Reference

```text
models/connectome_discovery/
├── __init__.py
└── mapping.py          map similarity, permutation, and ranking

skills/connectome-discovery/
└── SKILL.md
```

---

## Reference

- Use `skills/cpm/SKILL.md` for CPM training.
- Use `skills/brain-visualization/SKILL.md` for spatial rendering.

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
