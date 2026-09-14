---
title: NeuroOracle
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Dual-source knowledge graph for NeuroDiscovery
tags:
  - neuroscience
  - knowledge-graph
  - neuroimaging
  - hypothesis-generation
  - autoresearch
---

<div align="center">

# NeuroOracle

**A dual-source knowledge graph and hypothesis engine — part of [NeuroDiscovery](https://github.com/CUHK-AIM-Group/NeuroDiscovery).**

[![NeuroDiscovery](https://img.shields.io/badge/Part%20of-NeuroDiscovery-blueviolet)](https://github.com/CUHK-AIM-Group/NeuroDiscovery)
[![Project Page](https://img.shields.io/badge/Project-Homepage-orange)](https://cuhk-aim-group.github.io/NeuroDiscovery/)
[![Paper](https://img.shields.io/badge/arXiv-2604.24696-b31b1b)](https://arxiv.org/abs/2604.24696)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/CUHK-AIM-Group/NeuroDiscovery/blob/main/LICENSE)

</div>

---

## What is NeuroOracle?

NeuroOracle is the existing knowledge-graph module and explorer name within **NeuroDiscovery**, a closed-loop framework for evidence-grounded neuroimaging autoresearch. It combines two complementary information sources to support hypothesis generation:

1. **Curated structured databases** — concepts and relations imported from NeuroNames, MeSH, DisGeNET, BrainMap, and Cognitive Atlas, with UMLS alignment where supported.
2. **PubMed-derived scientific claims** — source-linked evidence records extracted from neuroimaging literature using LLM-based claim extractors, preserving supporting text, provenance, publication dates, evidence polarity and reported study metadata where available.

Together they form a graph covering brain anatomy, diseases, genes, neurotransmitters, drugs, cognitive functions, imaging features, connectivity, visual stimuli, and emotion/vigilance labels. Graph size depends on the loaded snapshot; see the version information below.

This Space provides an interactive explorer for browsing the graph, inspecting evidence chains behind individual claims, and visualising multi-hop hypothesis paths.

## Graph versions

The earlier demo documentation reported approximately **89K concept nodes** and **174K edges**. These are historical snapshot statistics, not live counts for this Space or statistics for the expanded NeuroDiscovery research graph.

**NeuroDiscovery** is the research framework that combines a neuroscience knowledge graph, a hypothesis generator and **NeuroRuntime** for neuroimaging autoresearch. Its manuscript uses a separately versioned research graph. The paper-specific frozen graph and supporting source tables are being prepared for release; a release identifier and manifest will be provided through the [main repository](https://github.com/CUHK-AIM-Group/NeuroDiscovery#neurodiscovery-and-research-materials).

For a reproducible graph-size comparison, identify the exact snapshot and distinguish concept nodes, biomedical relations, evidence records and links from evidence records to concepts. An aggregate edge count can include different link types and should not be compared directly with a count of biomedical relations alone.

## Why dual-source matters

Curated resources provide standardized concepts and relations, while literature-derived records retain the source and context of individual findings. NeuroOracle's dual-source design enables NeuroDiscovery to:

- Generate hypotheses with **traceable evidence chains** back to specific PubMed papers
- Filter or re-rank hypotheses using **evidence weights** (effect size, sample size, replicability)
- Use run-specific experimental feedback to refine hypotheses while keeping the formal graph frozen; formal graph updates use a separate evidence-ingestion process

## NeuroClaw and NeuroDiscovery

The project uses the following names:

| Name | Role |
|--------|------|
| **NeuroClaw** | Former project name, retained in historical releases, citations and compatibility identifiers |
| **NeuroDiscovery** | Overall project and closed-loop framework for evidence-grounded neuroimaging autoresearch |
| **NeuroRuntime** | Agent-based execution platform, including GUI/CLI workflows and the skill library |
| **NeuroOracle** | Existing knowledge-graph module and explorer name (this Space) |
| **NeuroBench** | Existing interface name for the neuroimaging execution-task collection |

The graph explorer, execution-task collection and manuscript experiments are distinct resources. Their sizes and results should be read using the corresponding snapshot or run metadata.

## What you can do here

- **Browse concepts** across 13 domain tags (neuroanatomy, disease, gene, drug, imaging_feature, connectivity, cognitive_function, visual_stimulus, emotion, vigilance, paradigm, dataset, ml_model)
- **Inspect claims** — source-linked evidence records retain the source paper, predicate (`is_biomarker_of`, `predicts`, `correlates_with`, etc.), and available evidence metadata
- **Trace hypothesis paths** — multi-hop reasoning examples such as `visual stimulus → functional ROI → anatomical region`, or `imaging feature → gene → disease`
- **Filter subgraphs** by domain, dataset, or relation type for focused exploration

## Links

- 🏠 **Project homepage**: <https://cuhk-aim-group.github.io/NeuroDiscovery/>
- 💻 **Source code (GitHub)**: <https://github.com/CUHK-AIM-Group/NeuroDiscovery>
- 📄 **Technical report (arXiv)**: <https://arxiv.org/abs/2604.24696>
- 🧠 **NeuroOracle docs page**: <https://cuhk-aim-group.github.io/NeuroDiscovery/neuro-oracle.html>

## Citation

For the earlier NeuroOracle/NeuroClaw release, cite the NeuroClaw technical report below. It is a separate report from the NeuroDiscovery manuscript; the manuscript citation will be added when available.

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

MIT — same as the NeuroDiscovery repository. See <https://github.com/CUHK-AIM-Group/NeuroDiscovery/blob/main/LICENSE> for full terms.

## Contact

For questions or issues, please open an issue on the [NeuroDiscovery GitHub repository](https://github.com/CUHK-AIM-Group/NeuroDiscovery/issues).
