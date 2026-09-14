---
name: knowledge-graph-builder
description: "Use this skill when users need to build, populate, or extend a domain-specific knowledge graph from literature and structured databases. Triggers include: 'build knowledge graph', 'extract claims from papers', 'ingest data into graph', 'batch extract claims', 'knowledge graph construction', 'populate graph from PubMed', 'extract structured claims', 'ingest atlas data', or any request involving knowledge graph population from scientific literature or biomedical databases. Covers both structured data ingestion (Phase 1) and LLM-based claim extraction from papers (Phase 2)."
license: MIT License (NeuroClaw custom skill – freely modifiable within the project)
layer: base
skill_type: tool
dependencies:
  - multi-search-engine
  - academic-research-hub
---
# Knowledge Graph Builder

## Overview

This skill provides a reusable framework for constructing domain-specific knowledge graphs by combining two complementary data pipelines:

- **Phase 1 — Structured Ingestion**: Import concepts and relations from curated databases, ontologies, and brain atlases (e.g., NeuroNames, MeSH, DisGeNET, Cognitive Atlas, Nilearn atlases).
- **Phase 2 — Literature Claim Extraction**: Use LLMs to extract structured scientific claims from PubMed paper abstracts, then resolve entities and ingest into the graph.
- **Phase 3 — Hypothesis Engine**: Traverse the graph to find novel connections, contradictions, and unexplored gaps — turning raw claims into testable research hypotheses.

The output is a directed knowledge graph (NetworkX DiGraph + JSON serialization) where nodes represent domain concepts and claims, and edges represent typed relationships with confidence scores and provenance.

**Primary implementation**: `neurooracle/` in the NeuroClaw project.

## Architecture

```
                    ┌─────────────────────┐
                    │  Knowledge Graph     │
                    │  (NetworkX DiGraph)  │
                    └──────┬──────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼────────┐ ┌─────▼──────────┐ ┌────▼─────────────┐
│  Phase 1:       │ │  Phase 2:      │ │  Phase 3:        │
│  Structured     │ │  Literature    │ │  Hypothesis      │
│  Data Ingestion │ │  Claim Extract │ │  Engine          │
└────────┬────────┘ └──────┬─────────┘ └────┬─────────────┘
         │                 │                 │
┌────────▼────────┐ ┌──────▼─────────┐ ┌────▼─────────────┐
│ - NeuroNames    │ │ - PubMed search│ │ - Path finding   │
│ - MeSH          │ │ - LLM extract  │ │ - Bridge discover│
│ - DisGeNET      │ │ - Entity resol │ │ - Contradictions │
│ - Cognitive Atl │ │ - Claim ingest │ │ - Gap detection  │
│ - Nilearn atlas │ │                │ │ - Ranking        │
└─────────────────┘ └────────────────┘ └──────────────────┘
```

## Key Design Decisions (Lessons Learned)

### 1. Schema Design: Three-Tier Nodes

The graph uses three types of nodes, all stored in the same DiGraph:

| Node Type | ID Format | Purpose |
|-----------|-----------|---------|
| **ConceptNode** | `NN:1234`, `CUI:xxx`, `MESH:D0001` | Domain concepts (brain regions, diseases, genes, drugs) |
| **Claim** | `CLM:abc123def456` | Structured scientific claims extracted from papers |
| **Edge** | (implicit) | Typed relationships between any two nodes |

**Why this matters**: Claims are stored as nodes (not just edges) so they can carry full metadata (evidence, p-value, sample size, conditions, population). Simplified edges are also generated for fast traversal.

### 2. Entity Resolution: 5-Level Matching

When ingesting claims, entity names must be resolved to existing concept IDs. Use a cascading strategy:

1. **Exact match** on preferred_name
2. **Case-insensitive** match
3. **Alias match** (check synonyms)
4. **Substring match** (entity contained in name or vice versa, prefer shortest name)
5. **Create new** concept if no match found

**Why not just use embeddings?** For small-to-medium graphs (<100K concepts), string matching is fast and predictable. SapBERT/FAISS alignment is recommended when UMLS is available and the graph exceeds 100K concepts.

### 3. LLM Extraction: Keep Prompts Short

LLMs (especially via proxy endpoints) return empty responses when prompts + expected output exceed the token window. Hard-won rules:

- **Send the complete abstract** to the LLM. Do not apply a character cap to
  abstracts. Conference/proceedings compilations must be excluded during paper
  curation instead of being hidden by truncation. A separate bounded policy may
  still be used for full-text bodies.
- **Keep extraction prompts concise** — list field names and allowed values, not verbose descriptions
- **Use `max_tokens=8192`** — 4096 is too small for papers with many claims
- **Fix `[[` double brackets** — a common LLM error when outputting JSON arrays
- **Temperature=0.0** for v3 combined extraction/scope audit; legacy extraction
  may retain 0.1. Final reproducibility comes from seal reuse, not sampling.

### 4. Contextualized Triplets (MDKG-style)

Beyond simple (subject, predicate, object), extract:

- **Conditions**: list of conditions under which the claim holds (e.g., `["female only", "age > 65"]`)
- **Population**: study demographics (mean_age, gender distribution, sample size, cohort name)

This enables more nuanced graph queries and downstream hypothesis generation.

### 5. Checkpoint/Resume for Batch Jobs

Large-scale extraction (10 diseases x 27 years x 20 papers = 5400 papers) takes 10-60 hours. Always implement:

- Save checkpoint after each batch (disease+year)
- Track completed_diseases and completed_years
- Save graph periodically (every 5 years or after each disease)
- CSV export of paper metadata for audit trail

## Quick Reference

| Task | Command |
|------|---------|
| Ingest atlas data (Phase 1) | `python -m neurooracle.ingest_pipeline` |
| Generate brain atlas TSV | `python neurooracle/data/raw/generate_brain_atlas_nilearn.py` |
| Run single-disease extraction | `python -m neurooracle.batch_extract --diseases "Alzheimer's disease" --year-start 2024 --year-end 2024 --papers-per-year 5` |
| Run full batch extraction | `python -m neurooracle.batch_extract` |
| Resume from checkpoint | `python -m neurooracle.batch_extract` (auto-resumes) |
| Start fresh (ignore checkpoint) | `python -m neurooracle.batch_extract --no-resume` |
| Verbose logging | `python -m neurooracle.batch_extract -v` |
| Query graph stats | `python -c "from neurooracle import load_graph; g = load_graph(); print(g.stats())"` |
| **Batch generate hypotheses** | `python -m neurooracle.hypothesis_cli batch --output data/hypotheses.json` |
| **Rank saved hypotheses** | `python -m neurooracle.hypothesis_cli rank --input data/hypotheses.json --top 20` |
| Find hypothesis paths | `python -m neurooracle.hypothesis_cli paths "hippocampus" "Alzheimer Disease"` |
| Bridge discovery | `python -m neurooracle.hypothesis_cli bridge "hippocampus" --target-domain disease` |
| **Discover from concept** | `python -m neurooracle.hypothesis_cli discover "Alzheimer" --max-hops 3` |
| **Find trending evidence** | `python -m neurooracle.hypothesis_cli trending --since 2020 --direction strengthening` |
| Find contradictions | `python -m neurooracle.hypothesis_cli contradictions` |
| Detect gaps | `python -m neurooracle.hypothesis_cli gaps --domain-a neuroanatomy --domain-b disease` |
| Explore a concept | `python -m neurooracle.hypothesis_cli explore "hippocampus"` |

## Agent Reference Rule

When the agent needs knowledge graph implementation code, it should first consult the curated snippets in `skills/knowledge-graph-builder/scripts/` instead of writing from scratch.

Reference snippets available:
- `scripts/entity_resolution.py` → EntityResolver class with 5-level matching
- `scripts/graph_query.py` → CLI tool for graph queries (stats, search, neighbors, paths, domain)
- `scripts/hypothesis_cli_reference.py` → Hypothesis engine usage patterns (executable code in `neurooracle/hypothesis_cli.py`)
- `scripts/extraction_prompt_template.txt` → LLM extraction prompt template
- `scripts/new_data_source_template.py` → Template for adding new data sources

## Installation

```bash
# Core dependencies
pip install networkx requests openai

# Optional: for atlas generation
pip install nilearn nibabel

# Optional: for Biopython Entrez (PubMed)
pip install biopython

# Use the neuroclaw conda environment
conda activate neuroclaw
```

## Phase 1: Structured Data Ingestion

### Supported Data Sources

| Source | Data Type | Entity Type | Edge Type |
|--------|-----------|-------------|-----------|
| NeuroNames / Nilearn atlases | Brain region hierarchy | neuroanatomy | part_of |
| MeSH (desc*.xml) | Medical subject headings | disease, anatomy | is_a |
| DisGeNET (TSV) | Gene-disease associations | gene | gene_associated_with_disease |
| Cognitive Atlas (API) | Tasks, concepts, disorders | cognitive_function, paradigm | — |
| UMLS (pending) | Unified medical language system | all types | various |

### Adding a New Data Source

Create a new file in `neurooracle/ingestion/` following this pattern:

```python
"""Ingest data from [SOURCE] into the knowledge graph."""

from ..schema import ConceptNode, Edge, DomainTag
from ..graph_manager import KnowledgeGraph

def ingest_source(kg: KnowledgeGraph, data_path: str) -> dict:
    """Parse source data and add to graph.

    Returns summary dict with counts.
    """
    concepts_added = 0
    edges_added = 0

    # 1. Parse raw data
    records = parse_data(data_path)

    # 2. Create ConceptNodes
    for record in records:
        node = ConceptNode(
            id=record["id"],
            preferred_name=record["name"],
            domain_tags=[DomainTag.DISEASE.value],
            source_vocab="my_source",
            aliases=record.get("synonyms", []),
        )
        kg.add_concept(node)
        concepts_added += 1

    # 3. Create Edges (if hierarchical)
    for record in records:
        if record.get("parent_id"):
            edge = Edge(
                source_id=record["id"],
                target_id=record["parent_id"],
                relation_type="is_a",
                source="my_source",
            )
            kg.add_edge(edge)
            edges_added += 1

    return {"concepts_added": concepts_added, "edges_added": edges_added}
```

### Atlas Generation (Nilearn)

Use `scripts/generate_atlas.py` as a template for generating brain region hierarchies from Nilearn built-in atlases. Key points:

- Start with a manual hierarchy of core brain regions (~100-200)
- Augment with atlas labels (Talairach, Harvard-Oxford, AAL, Dosenbach, Pauli, Seitzman)
- Handle SSL issues by patching `requests.Session.verify` before Nilearn calls
- Output: TSV with columns: NN_ID, Name, Latin_Name, Synonyms, Parent_ID, Brodmann_area

## Phase 2: Literature Claim Extraction

All Case Study literature searches use the single frozen publication window
`1980-2026`. Do not narrow or widen this range for an individual Case Study,
source, preset, or retry campaign.

### Pipeline: Search → KG Deduplication → LLM Extraction → Ingestion

```
Literature search → candidate papers → candidate/KG/staging deduplication
                                              │
                                      abstract retrieval
                                              │
                                        LLM extraction
                                              │
                                       [Claim objects]
                                              │
                                       Entity resolution
                                              │
                                   Graph ingestion (nodes + edges)
```

### PubMed Search Strategy

For each disease+year combination, search with neuroimaging focus:

```
({disease}[Title/Abstract])
AND ("brain imaging"[Title/Abstract] OR "neuroimaging"[Title/Abstract]
     OR "MRI"[Title/Abstract] OR "fMRI"[Title/Abstract] OR "PET"[Title/Abstract])
AND {year}:{year}[pdat]
```

Rate limit: 0.4s between NCBI API calls (3 req/sec without API key).

### Mandatory Pre-Extraction Paper Deduplication

Search results must never go directly into abstract retrieval or claim
extraction. After search and before any LLM/API extraction:

1. Deduplicate the candidate list by PMID, DOI, PMCID, arXiv ID, OpenAlex ID,
   and normalized title/year.
2. Stream the formal claim store and compare every candidate with the
   `source_paper` identities already represented in the KG.
3. Compare candidates with all staged, not-yet-injected claim files as well.
4. Exclude matched papers from the extraction queue and write a separate audit
   file containing the match evidence and exclusion reason.
5. Report candidate counts before and after both deduplication gates.

Only papers absent from both the formal KG and staged claim sets may proceed to
abstract retrieval and claim extraction. This gate is required even when the
search script already attempted deduplication.

### LLM Extraction Prompt Design

See `scripts/extraction_prompt_template.txt` for the recommended prompt structure. Key fields to extract:

| Field | Description |
|-------|-------------|
| subject / object | Entity names |
| subject_type / object_type | Entity category (brain_region, disease, gene, ...) |
| predicate | Relationship type (reduces, increases, correlates_with, ...) |
| negated | Whether the claim states NO relationship |
| effect_metric / effect_size | Statistical effect (Cohen's d, r, OR, ...) |
| p_value | Statistical significance |
| sample_size | Study sample size |
| study_type | fMRI, PET, GWAS, meta_analysis, ... |
| conditions | List of contextual conditions |
| population | Study demographics |
| raw_sentence | Source sentence from abstract |

Each extracted claim must also carry the canonical, non-exclusive routing
fields `paper_case_study_ids` and `claim_case_study_ids`. The paper field is the
union of all claim-level assignments for that paper; the claim field remains
specific to the individual assertion. The 17 IDs come from
`neurooracle.src.case_studies`. `general` is implicit shared-corpus membership,
and `hindcasting` is a validation protocol—neither is a Case Study ID.

For v3 expansion, Case Study routing must use
`neurooracle.src.case_study_membership_policy`, which parses the frozen
`2026-08-10.peer17.v2` rubric. Do not
copy or summarize the Case Study definitions into another prompt. Automated
extraction must return all nine `case_study_gates`, evidence spans, confidence,
and a decision basis; the resulting claim must carry a validated
`scope_reaudit` seal from `case_study_membership_contract.v4`. Formal KG
ingestion is fail-closed by default (`require_final_scope_audit=True`) and must
fail the whole batch before mutation if any extraction failed or any claim is
missing/fails that seal. Historical or manual repair paths must opt out
explicitly. Search provenance is never membership evidence: search remains
high-recall, while claim routing is decided only from extracted evidence under
the frozen policy.

`case2_pathway_mediation` follows the same contribution rule as Case 1. A direct
genetic/pathway-to-neural claim or a direct baseline-neural-to-later-outcome
claim is sufficient; the paper need not contain the complete mediation chain.
Membership is canonicalized deterministically from `imaging_genetics`,
`progression_prediction`, or `prognosis`, and a standalone Case 2 label is
invalid. The historical same-paper `case2_paper_chain_validation.v1` record is
retained only as optional analysis metadata and for validating immutable v3
seals; it is not a v4 membership gate.

### Entity Resolution During Ingestion

When a claim references "hippocampus" and the graph already has `NN:11` (preferred_name="Hippocampus"), the entity resolver matches them. If no match is found, a new concept node is created with prefix `CLM_CONCEPT:`.

This means the graph grows organically: atlas data provides the backbone, and claim extraction fills in relationships and discovers new entities.

### Claim Node vs. Simplified Edge

Each claim generates **three** graph elements:

1. **Claim node** (`CLM:abc123`): full metadata (evidence, conditions, population, raw text)
2. **Simplified edge** (subject → object): for fast multi-hop traversal
3. **About edges** (claim → subject, claim → object): for provenance queries

## Output Files

| File | Description |
|------|-------------|
| `data/full_v2/knowledge_graph.json` | Authoritative current graph (concepts + edges + metadata) |
| `data/full_v2/extracted_claims.jsonl` | Canonical synchronized extraction/audit store |
| `data/full_v2/CURRENT_STATE.json` | Validated taxonomy and Case Study coverage snapshot |
| `data/papers_metadata.csv` | Paper records: pmid, doi, title, authors, year, journal, disease, abstract_length, n_claims, timestamp |
| `data/batch_checkpoint.json` | Resume checkpoint: completed_diseases, completed_years, totals |

## Complementary / Related Skills

- `academic-research-hub` → paper search (arXiv, PubMed, Semantic Scholar)
- `research-idea` → consumes knowledge graph for hypothesis generation
- `method-design` → uses graph structure for method comparison

## Reference

- MDKG paper: Gao et al., "Large language model powered knowledge graph construction for mental health exploration." Nature Communications (2025). PMID: 40804250
- NeuroNames: Brain region hierarchy
- MeSH: Medical Subject Headings (NLM)
- DisGeNET: Gene-disease association database
- Cognitive Atlas: Cognitive paradigm ontology
- Nilearn: Python brain atlas library

## Phase 3: Hypothesis Engine

The hypothesis engine **batch-generates** hypotheses across the entire graph, **persists** them to JSON, and **ranks** by novelty, evidence, testability, and confidence. See `neurooracle/hypothesis_engine.py` for the implementation.

### Workflow

```
batch_generate() → save_hypotheses() → rank_hypotheses() → (Phase 5: convert to analysis tasks)
```

### Capabilities

| Function | Description |
|----------|-------------|
| `batch_generate()` | Traverse entire graph, generate hypotheses across all domain pairs |
| `save_hypotheses()` / `load_hypotheses()` | Persist to JSON for iterative re-ranking |
| `rank_hypotheses()` | Sort by composite score (4 dimensions) |
| `find_paths(src, tgt)` | Interactive: multi-hop path finding |
| `bridge_discovery(concept, domain)` | Interactive: cross-domain connection discovery |
| `discover_hypotheses(concept)` | Find hypotheses radiating from a single concept to all reachable domains |
| `find_trending(since_year, direction)` | Find concept pairs with strengthening/weakening evidence over time |
| `contradiction_detection()` | Find opposing claims on same concept pair |
| `gap_detection(domain_a, domain_b)` | Find 2-hop concept pairs with no direct edge |

### Scoring (4 Dimensions)

Each hypothesis is scored on four dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| **Confidence** | 0.25 | Edge confidence × study type quality × replicability |
| **Novelty** | 0.25 | Cross-domain paths, rare relations, few supporting papers |
| **Evidence** | 0.25 | p-value strength, sample size, effect size presence |
| **Testability** | 0.25 | Can NeuroClaw execute this? Modality detection (sMRI, EEG, fMRI, PET, DTI), brain region specificity |

Composite ranking: `confidence^0.25 * evidence^0.25 * novelty^0.25 * testability^0.25`

### Default Domain Pairs

The batch generator explores these cross-domain pairs:
- neuroanatomy ↔ disease
- neuroanatomy ↔ cognitive_function
- disease ↔ gene
- disease ↔ drug
- disease ↔ biomarker
- gene ↔ disease
- drug ↔ disease
- cognitive_function ↔ disease
- neurotransmitter ↔ disease

## Future Work

- **SapBERT entity alignment** with UMLS (cosine similarity > 0.9)
- **LLM-based hypothesis summarization** — use LLM to generate natural language hypothesis descriptions
- **Result feedback loop**: validated hypotheses write back to graph
- ~~**Temporal analysis**~~: implemented as `find_trending()` — tracks strengthening/weakening evidence trends across publication years

---
Created At: 2026-05-04 20:28 HKT
Last Updated At: 2026-05-06 14:46 HKT
Author: chengwang96
