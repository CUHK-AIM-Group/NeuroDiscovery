# Source-bound bucket census — results (2026-09-27)

> **Historical / scientific claims withdrawn.** See
> `docs/HYPOTHESIS_GNN_OFFLINE_REPAIR_20260927.md` and the hash-bound
> `tmp/gnn_validation_20260927/OFFLINE_CORRECTION_V1.json`.
> The labels, hard negatives, graph-wide extrapolations and reported accuracy
> below are not accepted validation. They remain here as a historical record.

Read-only availability census for step 3 of
`docs/HYPOTHESIS_GNN_VALIDATION_PLAN_20260927.md`. It answers one question:
**from assets that already exist, how many auditable items can each of the six
buckets actually produce?** It builds no labels, trains nothing, calls no
provider, and writes only inside `tmp/gnn_validation_20260927/`.

This is a capacity measurement, not validation. Nothing here is production or
publication credit.

## What was measured

Inputs reused, never modified: the 19,985-node / 31,507-edge read-only subgraph
from `tmp/gnn_validation_20260927/SUBGRAPH.json`, and 6,000 READY abstracts from
`BATCH_MODEL/INPUTS.sqlite`. The source graph was not re-read for this census.

Artifacts:

| File | Contents |
| --- | --- |
| `CENSUS.json` | structural counts per bucket + a 400-edge source-bound sample |
| `RELATION_DIAGNOSIS.json` | per-relation co-occurrence rates |
| `CANDIDATES.json` | 80 anchored direction positives + 80 swapped hard negatives |
| `CANDIDATES_PATHS.json` | path buckets and the two unlabeled buckets |
| `census_buckets.py`, `build_candidates.py`, `build_path_candidates.py`, `summarize_census.py` | reproducible, read-only |

## Defect found and fixed: an always-false decode test

The first census reported **zero** source-bound matches for every relation.
That was not a fact about the data. The abstract column stores zlib bytes, and
the check was `blob[:2] == b"\x78"` — a one-byte literal compared against a
**two**-byte slice, which is always `False`. Every abstract was therefore
decoded as raw compressed bytes, and every name lookup silently matched nothing.

Positive controls made this visible: with the bug, `hippocampus`,
`schizophrenia`, `brain` and the rest matched `0/1500` abstracts. After
decompressing unconditionally, `brain` matched `753/1500`. All numbers below are
post-fix. The bug is recorded rather than quietly corrected because "the graph
has no provenance" was a plausible-looking wrong conclusion — the graph really
does have empty `evidence_ref`, but that was not the cause of the zero.

## Availability

Structural counts are exact over the reused subgraph. They say an item can be
*formed*, not that it is correct.

| Bucket | Structural availability | Anchored candidates built | Status |
| --- | --- | --- | --- |
| `direction_positive` | 31,011 clean directional edges | **80** (cap) | unadjudicated |
| `direction_hard_negative` | same pool, endpoint order swapped | **80** (cap) | unadjudicated; adversarial only if the text rejects the swap |
| `path_positive` | large raw chain count | **5** | unadjudicated; hubs and symmetric hops excluded |
| `path_hard_negative` | same chains, all hops reversed | **5** | unadjudicated |
| `path_unresolved` | 9,051 endpoint pairs | 0 | **unlabeled** — abstention targets, not negatives |
| `edge_unrelated_negative` | 25,407 cross-family edges | 0 | **unlabeled** — graph silence is unknown, not absence |

## What the census changed about the plan

- **The direction buckets are producible at the agreed size.** Both reached the
  80-item cap with a literal sentence anchor, spread over 13 relations
  (`activates` 25, `distinguishes` 17, `gene_associated_with_disease` 8, …).
  27 of 80 anchors carry an explicit relation cue; the rest are name
  co-occurrences that a root reader must judge.
- **The path buckets are not.** Only 5 chains survive hub rejection and a
  relation-cue requirement on both hops. The plan asked for 30–50. Reaching that
  would require either accepting hub-mediated chains (scientifically empty) or a
  much larger anchored graph. Report the 5 and treat path ranking as
  **infeasible at this budget**, rather than padding the bucket.
- **Symmetric relations must be excluded from the direction task.**
  `correlates_with` (co-occurrence 0.32), `is_associated_with` (0.44) and
  `links` (0.36) are the *best-anchored* relations, but "A correlates with B" is
  true in both orders, so swapping endpoints is the same proposition, not a hard
  negative. The first candidate build used them and produced a direction task any
  model passes trivially. Only genuinely directional relations are used now.
- **Two buckets cannot be auto-labelled at all.** A missing edge is unknown, not
  false, and an absent two-hop path is not evidence against a relation. These are
  left explicitly `unlabeled` with sample rows, to be labelled by root source
  reading or dropped — never converted into negatives by the builder.

## Honest limitations

- The anchor is a **name co-occurrence**, not a citation link. 23% of sampled
  edges (92/400) have both endpoint names somewhere in one abstract; that is a
  recall ceiling, and it does not establish that the abstract asserts this
  relation in this direction. Only root source reading can do that.
- Candidate selection stops at the first matching abstract per
  `(work_key, relation)`, so the pool is not a random sample and its composition
  will drift if the scan order changes.
- Hub rejection uses a single degree threshold (100, the ~99.5th percentile here)
  applied globally; per-relation thresholds would be more defensible but were not
  fitted at this budget.
- `path_positive` chains are pairs of separately anchored hops. That two hops are
  each anchored does not make the chain a stated finding — the artifact records
  this explicitly as `anchors_are_separate_facts_not_yet_a_chain`.
- The census covers 6,000 of 321,874 ready abstracts. A larger scan would raise
  the counts; the *rates* are the more stable number.

## Next step

The 160 direction candidates are ready for root source-first adjudication, which
is the only step that turns a candidate into a label. Path ranking needs a
separate decision about scope, since 5 chains cannot support a metric. Nothing
is trained, wired or published until that adjudication exists.

## Root adjudication and frozen-label evaluation

Executed in the same turn, in the order the plan requires: labels were read and
frozen **before** any model was scored.

`adjudicate_anchors.py` records one verdict per anchor, read from the sentence.
`ROOT_ADJUDICATION.json` holds the result: **24 `positive_supported`, 25
`not_asserted`, 26 `spurious_entity`, 5 `sign_or_relation_contradicted`**.

The two failure modes are different and both matter:

- `spurious_entity` (26/80, 32.5%) — the endpoint is a generic word (`action`,
  `activation`, `association`, `logic`, `knowledge`) matched inside a longer
  phrase. "EMG **activation**", "mode of **action**", "allelic **association**"
  are not the graph concepts. These edges are vocabulary artifacts, not claims.
- `not_asserted` (25/80, 31.2%) — both endpoints are genuinely named, but the sentence
  asserts no such relation.

So of 31,011 structurally available "clean directional edges", the anchored
sample suggests about **30% (24/80) are source-supported** and **about 64%
(51/80) are unusable** artefacts or non-assertions. The structural count was
never an availability estimate for *evidence*.

Each `positive_supported` item yields a matched pair, because the anchored
sentence supports `A -> B` and does not support `B -> A`. No positive's swapped
twin is itself a positive, so the 24 hard negatives are genuine.

**Deduplication matters here.** The 24 positives reduce to **19 distinct facts
across 19 works**: five are the same fact recorded in both directions
(`Stroke is_assessed_by Barthel` and `Barthel measures Stroke`, and four more
like it). An `n` of 24 therefore overstates the independent evidence by about a
quarter; the honest count is 19.

### Evaluation on the frozen labels

`evaluate_frozen_labels.py`, same 20k-node subgraph and matched configuration,
3 seeds. The labelled edge is in the training split for 17 of 24 items, so the
leaked and unleaked subsets are reported separately.

| Decoder | Accuracy | Unleaked accuracy | Ties | Mean leaked |
| --- | --- | --- | --- | --- |
| DistMult | 0.500 | 0.500 | 23.0 | 17.0 |
| Directional | **0.841** | **0.889** | 0.0 | 17.0 |

DistMult is exactly 0.500 **by construction**, confirmed as 23 ties per run, so
it cannot separate a direction at all. The directional decoder separates 84% of
adjudicated positives (0.826 / 0.826 / 0.870 per seed, 0 ties). But this is a
**leakage-dominated number**: 17 of 24 labelled edges are in the training split,
so only **6 items** are unleaked, where the directional decoder is 5.33/6
averaged (0.889). Six items cannot support a rate; the 0.84 figure should be
read as "the decoder is not at chance", not as a validated accuracy.

Two defects were found and fixed while getting these numbers, both of which had
produced a *plausible* wrong answer first:

1. Scoring initially returned `0/0`. The labels carried entity display names
   while the model vocabulary is keyed by node id (`CUI:…`), so nothing matched
   and it looked like total model failure.
2. DistMult first appeared to win a few rows at 0.52 accuracy. Those "wins" were
   float32 reduction-order noise; with the same tolerance
   `run_experiment.reversal_probe` uses, it is 0.500 with ties, as it must be.

### What this establishes

- On source-verified cases, a relation-aware GNN decoder separates a directed
  assertion from its reversed twin far better than a symmetric one. The earlier
  random-corruption probe understated this, because random corruptions are much
  easier than the endpoint swaps that actually matter.
- The graph's usable signal for this task is roughly a quarter of its
  directional edges, and the rest are vocabulary artifacts or mere co-occurrence.
  Any recall or coverage figure computed off raw edge counts would be inflated
  by about 4×.
- Nothing here is whole-corpus accuracy, independent validation, or production
  credit. 24 positives across 7 relations from one 20k-node subgraph is a
  diagnostic, not a benchmark. Path ranking remains unmeasured: only 5 chains
  survived, which cannot support a metric.
