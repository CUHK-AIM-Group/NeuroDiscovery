# Wiring a real GNN/KGE score into `balanced` hypothesis selection

> **Current status:** offline repair only; no qualified model or live wiring.
> `docs/HYPOTHESIS_GNN_OFFLINE_REPAIR_20260927.md` supersedes earlier evidence
> claims and adapter behaviour: OOV/unknown graphs now fail closed, not at 0.5.
> Historical experiments exceeded their run budget; no new training is enabled.

`balanced` ranks candidates as novelty 40%, structural evidence 20%, GNN/graph-path
20%, scientific review 20%. In AutoResearch `idea` mode the last two *graph*
numbers were never measured: candidates are natural-language hypotheses and the
gate passed `structural_score = 0.0`, `gnn_path_score = 0.0` and disclosed them
under `unmeasured_scores`. This document describes how to make those two terms
real, using assets that already exist in this repository, and how to stay honest
when they still cannot be measured.

It is a design + partial implementation note. Only the mapping/scoring bridge
(`neurooracle/src/kge/hypothesis_path.py`) is implemented; the service-side
wiring described in Stage 3 is a proposal, not something this note authorizes.

## What already exists

- **Trained ComplEx checkpoint** `neurooracle/data/full_snapshot_v2/kge_complex.pt`
  (`complex_d128_lr1e3`, dim 128, 12,890 entities, 19 relations). Loads and scores
  with `ComplExScorer.load(...)`; out-of-vocabulary triples return a neutral `0.5`.
- **The current graph** `neurooracle/data/knowledge_graph.json.gz` (372,928
  concepts, 439,249 edges). 12,889 of the checkpoint's 12,890 entities appear in
  it, so the KGE was trained on this lineage.
- **A relation-aware GNN** `models/kg_link_prediction/gnn.py`
  (`GNNLinkPredictor`, R-GCN / GraphSAGE / GAT encoders + DistMult decoder) with a
  trainer (`models/kg_link_prediction/train.py`) and a documented skill
    (`skills/kg-link-prediction/SKILL.md`). No GNN checkpoint is wired into the
    live idea-mode gate; historical experiment checkpoints are not deployment assets.
- **Path-level scoring** `neurooracle/src/kge/plausibility.py`
  (`local_plausibility` = geometric mean of per-edge scores, 0.7× weak-link penalty
  below 0.3) and `neurooracle/src/kge/specificity.py` (`path_specificity`, hub and
  vague-node rejection).
- **The frozen development semantics** for `structural_score` and
  `gnn_path_score`, in `.codex_tmp/supplemental_two_case_strict_luna_20260915_v2/src/full_nd.py`
  (`structural_score`, `score_candidates`). Those are the scales reproduced here.
- **The KG service** `core/web/server.py` `/api/kg/*` already loads the graph
  lazily via `_load_kg_blocking`, builds `name_index` / `clean_name_index`, and can
  resolve paths (`/api/kg/paths`), neighbourhoods and node payloads.

## The gap

`novelty_policy.select_reviewed` needs a `structural_score` and `gnn_path_score`
for every candidate. `core/novelty_gate.py` builds those candidate records from
`candidate_payload`, whose structured field is `source_ids` — a list of PMIDs, not
KG triples. So the pipeline is:

    hypothesis prose + PMIDs  ──?──►  KG entities + relations  ──►  per-edge scores

Two things are missing: a **hypothesis → triples mapping** (the `?`), and a
**scoring call** that reaches the loaded model without reading the whole graph
again. The environment to run either is the local conda env
`C:\Users\45846\anaconda3\envs\neuroclaw\python.exe`
(torch 2.5.1+cu121 with CUDA, torch_geometric 2.7.0, numpy 2.3.5).

### Local runtime verification — 2026-09-27

The environment name is **`neuroclaw`**, confirmed by `conda env list --json`
and actual imports with its Python executable, not the WindowsApps `python`
launcher. No installation or environment recreation is needed on this machine:

```powershell
conda activate neuroclaw
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__, torch.cuda.is_available())"
```

The direct executable above works without activation. Verification found Python
3.11.15, the package versions listed above, and an available NVIDIA GeForce RTX
4060 Laptop GPU. The existing ComplEx checkpoint loaded on CPU with 12,890
entities and 19 relations and returned a finite score for a synthetic in-vocabulary
triple. The existing R-GCN completed a finite CUDA forward pass on a synthetic
three-node graph. These are runtime checks, **not** scientific validation, GNN
training, graph compatibility acceptance, or a live selection integration.

The earlier graph-overlap observation is not sufficient to establish compatibility
with the paper-layer reconstruction. Deployment needs a version/hash-bound graph,
vocabulary, and checkpoint contract; neither the historical checkpoint nor a live
graph may silently become the reconstruction's accepted baseline.

## Stage 1 — `hypothesis_path.py` (implemented)

`neurooracle/src/kge/hypothesis_path.py` is the fail-closed bridge. It:

1. reads only the **explicitly declared** `kg_triples` field of a candidate
   (`declared_triples`); it never parses prose into edges, because guessing edges
   from text would manufacture the evidence the score is meant to weigh;
2. resolves each endpoint through an optional `resolve_entity` callback, then
   checks it against the predictor's vocabulary;
3. scores the resolved edges with any object exposing
   `score_batch`/`ent2idx`/`rel2idx` (the `Scorer` surface — so ComplEx today, an
   R-GCN later, same call);
4. reduces them to the frozen shapes:
   - `gnn_path_score` = geometric mean of per-edge sigmoid scores, ×0.7 if the
     weakest edge is below 0.3;
   - `structural_score` = `confidence^0.20 · traceability^0.20 ·
     graph_only_novelty^0.25 · testability^0.35`, every component floored at 0.01;
5. returns `measured: False` with `None` scores and a plain-language `disclosure`
   whenever **any** input is missing (no triples, out-of-vocabulary endpoint or
   relation, no model, no edge confidence, no provenance fraction, no testability).

A partial reading never rounds up to a number: if the GNN term is measured but the
structural inputs are not, `gnn_path_score` is a float and `structural_score` is
`None` — which is exactly when `choosing` should fall back to the disclosed floor.

Verified against the real checkpoint: score a two-edge chain of real graph entity
ids and it returns `measured: True` with `gnn_path_score ≈ 0.50` and
`structural_score ≈ 0.79`; give it one non-existent entity id and it returns
`measured: False` citing that endpoint. 16 synthetic tests in
`neurooracle/tests/test_kge_hypothesis_path.py` cover the mapping, the weak-link
penalty, the floor, and every fail-closed branch.

## Stage 2 — hypothesis → triples (the real work, not yet implemented)

This is the part that decides whether the GNN term means anything. Options, in
increasing cost and honesty:

1. **Declared triples from the generator.** Extend the `idea`-mode candidate
   contract so a submitted hypothesis *may* include `kg_triples`
   (`source`/`relation`/`target`), and have the gate score them when present.
   Cheap and honest, but only fires when the model both knows the graph ids and
   chooses to declare them — likely rare without help.
2. **Entity linking with the existing index.** Reuse the service's `name_index`
   (preferred names + aliases, built once and cached) to map each endpoint name to
   a candidate node id, then require a unique best match. Feed the mapper in as
   `resolve_entity`. Ambiguous or absent names must stay unresolved — never pick
   the top-1 by default.
3. **Relation typing by the trained vocabulary.** Restrict declared relations to
   the 19 trained relation types; anything else is disclosed, not coerced into the
   nearest known relation.
4. **Path search as a fallback.** When a hypothesis names two endpoints but no
   intermediate hops, `/api/kg/paths` can supply candidate paths. This is the most
   generative step and the easiest to over-trust: a retrieved path is graph
   structure, not evidence that the hypothesis is true. It may seed a *scored*
   triple set only if each returned hop is a real, confidence-bearing edge.

The rule for all four: a mapping is a claim about text-to-graph identity, so it
must be recorded and auditable (`edge_scores` already carries the resolved
(`source`, `relation`, `target`) triples) and must not silently drop or invent
edges.

## Stage 3 — service wiring (proposal)

To make Stage 2 reachable without reloading the 64 MB graph per call:

1. Load the ComplEx checkpoint **once**, lazily, beside `_load_kg_blocking`, and
   cache the predictor on the existing KG state object.
2. Expose the `name_index` / node payloads as a `resolve_entity` callable and an
   `edge_confidence(s, r, t)` callable that reads the graph edge's `confidence`
   attribute; expose `pair_support(s, t)` as a cached degree-like count (the
   frozen scorer used a `Counter` over the evidence pool).
3. When the gate builds each reviewed candidate, call
   `hypothesis_path.score_candidate(...)` with those callables. Where
   `measured` is true, write the two numbers into the candidate record and leave
   `unmeasured_scores` empty; where false, keep the `0.0` floor and put the
   `disclosure` into `unmeasured_scores` (today's behaviour is preserved exactly).
4. Keep the whole stage **optional and off by default** until a shadow comparison
   shows the measured numbers change selection in the intended direction. This is
   a change to an evidence path, not a cosmetic one; it should be gated the same
   way a model change is.

## Stage 4 — prefer a GNN path scorer, without requiring KGE

GNN is an architecture; KG embedding/link prediction is one possible task.
ComplEx is an embedding model, not a message-passing GNN. The existing
`GNNLinkPredictor` really uses message passing, but still trains for edge
reconstruction. Neither is automatically a hypothesis-quality model.

The recommended target is **a relation-aware GNN encoder plus an ordered
path/subgraph ranking head**. KGE remains an optional comparison baseline,
not a prerequisite or a score that should be labelled as a measured GNN result.

1. **Input contract.** Require explicitly mapped entity IDs, typed and directed
   relations, and the candidate path. Retrieve a bounded read-only neighbourhood
   with provenance and scope metadata. Preserve species, disease/stage, evidence
   role, signs, null results and unresolved scope. Missing or ambiguous mappings
   remain unmeasured; a model may not fill them in silently.
2. **Encoder.** Start with a two-layer R-GCN and bounded neighbour sampling for
   the local GPU. Treat reverse relations as distinct types. Evaluate memory and
   latency on a bounded subset before fixing dimensions or batch sizes. Frozen
   node features plus message passing can replace a pure learned entity-ID
   lookup; feature quality and unseen-node behaviour need separate evaluation.
3. **Ranking head.** Feed ordered source/relation/target representations and
   path/subgraph context into a small directional MLP or ordered path encoder.
   Score structural compatibility, not truth, literature novelty or first-report
   status. The current DistMult decoder is symmetric under endpoint reversal;
   the current `message_graph` also reuses the forward relation for reverse
   edges. Both must change for asymmetric scientific relations. Merely selecting
   `--model rgcn` does not address these limitations.
4. **Training and validation.** Link prediction can pretrain the encoder but is
   not sufficient validation for path ranking. A path head needs a separately
   referenced set of relevant/irrelevant or preferred/non-preferred paths,
   hard negatives and unresolved cases. Missing edges are unknown, not known
   falsehoods. Split by work/source family (and time when available) before
   constructing message graphs; exclude validation/test target edges, duplicates,
   reverse-edge proxies and source-derived answer features. Do not consume the
   protected 40 paper-reconstruction heldout assignments. Unresolved examples
   test abstention rather than becoming invented negative labels.
5. **Runtime integration.** Run the worker with the verified `neuroclaw` Python;
   keep the graph/model resident and cache only against matching graph, feature,
   vocabulary and checkpoint versions. An edge-scoring adapter can implement
   `score_batch` for the existing bridge; a genuine path-ranking head needs a
   separate `score_path` interface instead of pretending its output is a
   geometric mean of edge probabilities. Record backend/model kind, graph hash,
   checkpoint hash, resolved path, coverage and missing-score reasons.
6. **Acceptance.** Compare structure-only, optional ComplEx, GNN edge scoring
   and GNN path scoring on the same new source-bound validation set. Report
   mapping coverage, directional hard-negative errors, ranking quality,
   abstention, latency and peak memory. Check both selection modes in shadow:
   a GNN score cannot override science/novelty gates in `novelty_first`, and
   `balanced` must disclose missing graph terms rather than report a complete
   measured composite. Only validated outputs should enter the existing 20%
   graph term; do not change policy weights as part of the rename.

The smallest implementation sequence is: mapping/provenance contract →
direction-aware GNN adapter and offline tests → bounded training/validation →
optional shadow wiring. The path-ranking head follows only when suitable labels
exist. A new training run, full-corpus dispatch and production activation are
**not started or authorized by this design note**. Existing historical experiment
checkpoints, fixed datasets and reconstruction artifacts remain unchanged.

### Implementation status — 2026-09-27 (step 2 of the sequence)

The direction-aware adapter and its offline tests are implemented. All changes
are additive and the historical defaults are unchanged:

- `models/kg_link_prediction/gnn.py` adds an optional asymmetric
  `DirectionalDecoder` (`--decoder directional`) and `directed_message_graph`,
  which gives reversed edges their own relation slots instead of reusing the
  forward relation. `GNNLinkPredictor` still defaults to `distmult` with the
  original `message_graph`, so existing callers and checkpoints are unaffected.
- `models/kg_link_prediction/scorer.py` adds `GNNEdgeScorer`, a scored wrapper
  exposing `score_batch`/`ent2idx`/`rel2idx` — the exact surface
  `hypothesis_path.score_candidate` already accepts. It encodes the adjacency
  once and caches the node embeddings; out-of-vocabulary triples return the same
  neutral `0.5` prior ComplEx uses. It fails closed if no graph was supplied.
- `models/kg_link_prediction/train.py` exposes `--decoder` and
  `--reverse-relations` (rejected for non-R-GCN encoders).
- `models/tests/test_gnn_direction.py` (7 tests) and
  `models/tests/test_gnn_edge_scorer.py` (6 tests) cover the symmetric-decoder
  defect, direction separation, distinct reverse relation slots, the historical
  default path, trainability on an asymmetric task, and end-to-end use through
  `hypothesis_path`.

Verified with `models/tests/test_extended_models.py` (30 passed, 1 skipped) and
the 13 new tests. This is an implementation and unit-test milestone, **not** a
trained model, a graph-compatibility acceptance, or a live selection change. No
training run, corpus dispatch or shadow wiring has been started; the next step
remains bounded training/validation against a source-bound split.

The design for that next step is `docs/HYPOTHESIS_GNN_VALIDATION_PLAN_20260927.md`.
It specifies the source-bound item buckets (including the directional hard
negatives this adapter exists to separate), the work-family split and leakage
rules, the structural/ComplEx/GNN baselines, the separate direction/ranking/
abstention metrics, and the proposed acceptance gates. Like this note, it is a
plan only: it starts no training, makes no model calls and writes to no fixed,
production or heldout asset. Concrete spend and training-size caps still need
their own explicit authorization.

## What this does and does not establish

- It makes the two graph terms **measurable** and keeps them **honest when
  absent**. It does not make `balanced` better until Stage 2 produces reliable
  mappings and Stage 3 is validated.
- A high `gnn_path_score` means the trained model finds the declared edges
  plausible under the graph distribution. It is **not** truth, novelty, or
  first-report evidence, and `asserts_truth_or_novelty` stays `False`.
- Nothing here reads or writes the fixed neuroscience layer or the frozen
  reconstruction artifacts; it scores candidate text against the same read-only
  graph the KG service already serves.
# 当前实施入口

GNN 辅助排序后续实施以 `HYPOTHESIS_GNN_CONSOLIDATED_PLAN.md` 为准；本文件保留历史设计与接口说明，
不是平行实验计划。路径排序头、KGE 对照与线上接入均不在当前最小实验范围。
