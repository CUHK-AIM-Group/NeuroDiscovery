---
title: NeuroOracle
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Dual-source knowledge graph for NeuroClaw
tags:
  - neuroscience
  - knowledge-graph
  - neuroimaging
  - hypothesis-generation
  - autoresearch
---

<div align="center">

# NeuroOracle

**A dual-source knowledge graph and hypothesis engine — part of [NeuroClaw](https://github.com/CUHK-AIM-Group/NeuroClaw).**

[![NeuroClaw](https://img.shields.io/badge/Part%20of-NeuroClaw-blueviolet)](https://github.com/CUHK-AIM-Group/NeuroClaw)
[![Project Page](https://img.shields.io/badge/Project-Homepage-orange)](https://cuhk-aim-group.github.io/NeuroClaw/)
[![Paper](https://img.shields.io/badge/arXiv-2604.24696-b31b1b)](https://arxiv.org/abs/2604.24696)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/CUHK-AIM-Group/NeuroClaw/blob/main/LICENSE)

</div>

---

## What is NeuroOracle?

NeuroOracle is the knowledge-graph component of **NeuroClaw**, an autonomous research framework for neuroimaging. It combines two complementary information sources to provide a quality-grounded foundation for hypothesis generation:

1. **Curated structured databases** — concepts and relations imported from NeuroNames, MeSH, DisGeNET, BrainMap, and Cognitive Atlas, all aligned to UMLS semantic types.
2. **PubMed-derived scientific claims** — evidence-weighted edges extracted from neuroimaging literature using LLM-based claim extractors, with provenance (PMID, p-value, sample size, study type) preserved on every edge.

Together they form a graph of approximately **89K concept nodes** and **174K edges**, covering brain anatomy, diseases, genes, neurotransmitters, drugs, cognitive functions, imaging features, connectivity, visual stimuli, and emotion/vigilance labels.

This Space provides an interactive explorer for browsing the graph, inspecting evidence chains behind individual claims, and visualising multi-hop hypothesis paths.

## Why dual-source matters

Existing autoresearch systems either rely on free-form LLM ideation (no quality anchor) or on a single curated KG (limited coverage and stale evidence). NeuroOracle's dual-source design is what enables NeuroClaw to:

- Generate hypotheses with **traceable evidence chains** back to specific PubMed papers
- Filter or re-rank hypotheses using **evidence weights** (effect size, sample size, replicability)
- Iterate the graph itself in response to new findings, rather than treating the KG as a static asset

## NeuroClaw ecosystem

NeuroOracle is one of three modules within the broader NeuroClaw system:

| Module | Role |
|--------|------|
| **NeuroClaw** | Top-level system: data processing, model execution, skill library (85 skills across 29 datasets) |
| **NeuroOracle** | Knowledge graph and hypothesis engine (this Space) |
| **NeuroBench** | Multi-agent neuroimaging workflow benchmark |

NeuroClaw is the umbrella framework; NeuroOracle is its scientific memory; NeuroBench measures how effectively the agent can use that memory to do real research work.

## What you can do here

- **Browse concepts** across 13 domain tags (neuroanatomy, disease, gene, drug, imaging_feature, connectivity, cognitive_function, visual_stimulus, emotion, vigilance, paradigm, dataset, ml_model)
- **Inspect claims** — every PubMed-derived edge carries the source paper, predicate (`is_biomarker_of`, `predicts`, `correlates_with`, etc.), and structured evidence fields
- **Trace hypothesis paths** — multi-hop reasoning examples such as `visual stimulus → functional ROI → anatomical region`, or `imaging feature → gene → disease`
- **Filter subgraphs** by domain, dataset, or relation type for focused exploration

## Links

- 🏠 **Project homepage**: <https://cuhk-aim-group.github.io/NeuroClaw/>
- 💻 **Source code (GitHub)**: <https://github.com/CUHK-AIM-Group/NeuroClaw>
- 📄 **Technical report (arXiv)**: <https://arxiv.org/abs/2604.24696>
- 🧠 **NeuroOracle docs page**: <https://cuhk-aim-group.github.io/NeuroClaw/neuro-oracle.html>

## Citation

If NeuroOracle or NeuroClaw is useful for your research, please cite the NeuroClaw technical report:

```bibtex
@article{neuroclaw2026,
  title   = {NeuroClaw: Closed-Loop Agentic AI for Executable and Reproducible Neuroimaging Research},
  author  = {NeuroClaw Team},
  journal = {arXiv preprint arXiv:2604.24696},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.24696}
}
```

## License

MIT — same as the NeuroClaw repository. See <https://github.com/CUHK-AIM-Group/NeuroClaw/blob/main/LICENSE> for full terms.

## Contact

For questions or issues, please open an issue on the [NeuroClaw GitHub repository](https://github.com/CUHK-AIM-Group/NeuroClaw/issues).
