---
name: kg-link-prediction
description: "Use this skill whenever the user wants knowledge-graph embeddings or graph neural link prediction over NeuroOracle: ComplEx, R-GCN, GraphSAGE, or GAT encoders; relation-aware triple scoring; filtered MRR/Hits evaluation; or hypothesis plausibility features. Triggers include 'KG embedding', 'link prediction', 'ComplEx', 'R-GCN', 'GraphSAGE', 'GAT', 'triple scoring', 'filtered ranking', 'MRR', 'Hits@K', and 'NeuroOracle GNN'."
license: MIT
layer: base
skill_type: model
dependencies:
  - knowledge-graph-builder
  - run_models
---
# KG Link Prediction Skill

## Overview

`kg-link-prediction` trains embeddings or graph neural encoders on NeuroOracle
triples and scores candidate relations.

**Supported models**

| Model | Encoder | Decoder/evaluation |
|---|---|---|
| `complex` | complex-valued entity/relation embeddings | ComplEx score |
| `rgcn` | relation-specific graph convolution | relation-aware DistMult |
| `graphsage` | neighborhood aggregation | relation-aware DistMult |
| `gat` | graph attention | relation-aware DistMult |

GNN message passing uses training edges only. Validation and test edges are
excluded from encoder adjacency, and negative samples are checked against all
known positive triples.

---

## Installation

```bash
pip install numpy torch scikit-learn
```

The implementation uses native PyTorch operations and does not require
PyTorch Geometric.

---

## Workflows

### 1. Prepare a NeuroOracle graph

Input is a NeuroOracle `knowledge_graph.json` containing typed nodes and
confidence-bearing edges. Low-confidence edges can be excluded with
`--min-confidence`.

### 2. Train R-GCN

```bash
python skills/kg-link-prediction/scripts/train_reference.py \
  --kg neurooracle/data/full_v2/knowledge_graph.json \
  --model rgcn \
  --embedding-dim 128 \
  --layers 2 \
  --dropout 0.1 \
  --epochs 100 \
  --negatives 10 \
  --min-confidence 0.2 \
  --device cuda \
  --output-dir run_models_output/kg_rgcn
```

### 3. Train GraphSAGE or GAT

Change `--model` to `graphsage` or `gat`. Keep the same split seed when
comparing encoders.

### 4. Train the existing ComplEx route

```bash
python skills/kg-link-prediction/scripts/train_reference.py \
  --kg neurooracle/data/full_v2/knowledge_graph.json \
  --model complex \
  --embedding-dim 128 \
  --epochs 100 \
  --output-dir run_models_output/kg_complex
```

Retrain the model whenever the graph snapshot changes materially. Record the
graph hash and freeze year when the embeddings are used for hindcasting.

---

## Input / Output Summary

| Item | Format |
|---|---|
| Input | NeuroOracle knowledge graph JSON |
| Filtering | edge confidence threshold |
| Split | train/validation/test triples |
| Checkpoint | `checkpoint.pt` |
| Metrics | `metrics.json` |
| Provenance | `config.json`, `run_manifest.json` with split counts |

GNN metrics include AUROC, AUPRC, filtered MRR, and filtered Hits metrics.
ComplEx currently exports test AUROC and training summary metrics through this
unified entry point.

---

## Testing

```bash
pytest models/tests/test_extended_models.py -q
python skills/kg-link-prediction/scripts/train_reference.py --help
```

For temporal experiments, additionally test that no post-freeze edge enters
training or message passing.

---

## Directory Reference

```text
models/kg_link_prediction/
├── gnn.py              R-GCN, GraphSAGE, GAT, decoder, graph indexing
└── train.py            split, train, rank, and artifact CLI

skills/kg-link-prediction/
├── SKILL.md
└── scripts/train_reference.py
```

---

## Reference

- Trouillon et al. ComplEx (2016).
- Schlichtkrull et al. R-GCN (2018).
- Hamilton et al. GraphSAGE (2017).
- Velickovic et al. Graph Attention Networks (2018).

---

Created At: 2026-07-26 HKT
Last Updated At: 2026-07-29 HKT
Author: chengwang96
