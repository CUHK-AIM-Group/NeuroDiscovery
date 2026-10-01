# Bounded GNN training/validation set — protocol (2026-09-27)

> Current implementation plan: `HYPOTHESIS_GNN_CONSOLIDATED_PLAN.md`.
> This older direction/path protocol is historical, not a parallel execution route.

> **Superseded where inconsistent:** `docs/HYPOTHESIS_GNN_OFFLINE_REPAIR_20260927.md`.
> Reversal, unrelated topics and missing edges are challenges, not automatic
> negative labels. DistMult reversal ties are not measured 50% accuracy.
> Historical runs did occur; none qualifies this plan as successfully executed.

This is the plan for **step 3** of the sequence in `docs/HYPOTHESIS_GNN_SCORING.md`:
gather and freeze a source-bound validation set so a graph model can be trained
and judged. Steps 1–2 (the mapping bridge and the direction-aware adapter) are
implemented and unit-tested. Nothing in this document has been run.

It authorizes **no** training run, no corpus dispatch, no model calls, no
production/shadow wiring, and no writes to the fixed neuroscience layer, the
reconstruction artifacts or the protected heldout set. It is a design to be
approved and bounded before execution.

## Why a new set is required

Every existing asset is unfit as an independent test:

- The historical ComplEx checkpoint and its training graph predate the paper
  layer reconstruction; they are a **comparison baseline**, not ground truth.
- The 28 root-reviewed staging abstracts and the 25 development cases are
  development material. Root extraction re-checked by the same root is not
  independent validation.
- The 40 frozen paper-reconstruction heldout assignments belong to the
  extraction/merge acceptance, must stay **unconsumed**, and are far too small
  and off-domain for a link/path-ranking model.
- Missing KG edges are *unknown*, not *false*. Negatives must be constructed
  deliberately and defensibly, or the model learns the graph's silence.

## What the model is judged on

Two separate questions, never merged into one number:

1. **Directional edge scoring** — does the relation-aware encoder separate a
   true, directed `(s, r, o)` from a plausible **reversed or swapped-endpoint**
   variant of it, and from a same-shape but unrelated triple?
2. **Path ranking** — given two endpoints, does the model rank a source-supported
   chain above a structurally plausible but source-unsupported chain, and can it
   **abstain** when the graph has no business answering?

The output is a structural compatibility signal. It is never truth, novelty, or
first-report evidence; `asserts_truth_or_novelty` stays `False` in every record.

## Set composition

Target size, deliberately small and hand-adjudicable, not a sampling of the
whole corpus. Counts are ceilings until the source search shows the cases exist.

| Bucket | Contents | Target | Source of truth |
| --- | --- | --- | --- |
| `direction_positive` | Directed relations a paper states explicitly, with literal anchor | 60–100 | own-abstract/own-result sentence |
| `direction_hard_negative` | Same endpoints/relation type, order reversed, or endpoint swapped within the same sentence family | 60–100 | the same source, read for the intended direction |
| `edge_unrelated_negative` | Same relation type, unrelated endpoints from a different topic family | 60–100 | topic cohort membership, not the model |
| `path_positive` | Two-endpoint chains where each hop is a real, confidence-bearing edge and the whole chain is stated or entailable from sources | 30–50 | root source-first reading |
| `path_unresolved` | Two-endpoint questions whose intermediate hop is genuinely absent or unread | 20–40 | absence of a source-bound hop |
| `path_hard_negative` | Chains that are graph-embeddable but whose direction, scope or sign contradicts the sources | 30–50 | root source-first reading |

Every item carries: `item_id`, `bucket`, the resolved KG ids, the source
`work_key`/PMID with `source_sha256`, a **literal anchor** (verbatim sentence or
value), the evidence role, the scope (species, disease/stage, measurement,
anatomy, cohort), and a `label_rationale`. An item without a root-readable
literal anchor is not admitted.

Hard negatives are the point of the exercise. A set that only contains easy
unrelated triples will report a flattering AUROC that means nothing.

## Splitting and leakage control

- Split by **work/source family** (and by time when a date is available) so no
  paper's other claims train a model that is then tested on its neighbours.
  Random triple-level splits leak and are not used.
- Build message graphs per split: validation/test target edges, their duplicate
  or reverse-edge proxies, and any source-derived answer feature are **excluded**
  from message passing.
- Entities that appear only in validation/test are reported as transductive
  versus inductive; do not silently move them into train without recording it.
- The frozen 40 heldout paper assignments, the fixed layer and the reconstruction
  artifacts are never read as features, labels or answers.

## Baselines to compare

Reported on the identical frozen split, in increasing cost:

1. **Structural-only** — degree/path existence, no learned model.
2. **ComplEx** — the existing checkpoint, as the KGE reference.
3. **GNN edge** — `GNNLinkPredictor` with the directional decoder and reverse
   relation slots.
4. **GNN path** — the same encoder with a path-ranking head, only if enough
   source-bound path labels exist.

Every row names its checkpoint hash, graph hash, vocabulary hash and coverage.

## Metrics

- Mapping/coverage: fraction of candidate endpoints and relations that resolve
  to the graph vocabulary, with the unresolved reasons counted separately.
- Direction: accuracy on `direction_positive` vs `direction_hard_negative`, and
  the specific **reversal-error rate** (score of `A→B` minus score of `B→A`).
- Ranking: MRR / Hits@K on `path_positive` vs negatives, micro and per-bucket.
- Abstention: how often the model correctly declines the `path_unresolved`
  items instead of emitting a confident order.
- Honesty: fraction of items left explicitly unmeasured, never imputed.
- Cost: latency per batch and peak GPU/CPU memory on the bounded subset.

No single aggregate score is used to accept a model. A model that improves
ranking while raising reversal errors is rejected.

## Acceptance gates (proposed, to be confirmed)

- Zero severe errors: no direction-inverted positive silently accepted; no
  scope/species/disease cross-over in any admitted item.
- Reversal-error rate on the direction buckets materially below the
  symmetric-decoder baseline (chance is 0.5 for a coin flip; the symmetric
  DistMult model is structurally at chance).
- Ranking above the structural-only baseline on `path_positive` with negatives
  fixed, and no worse on abstention.
- All labels carry a root-readable literal anchor; a model's self-reported
  confidence is never the acceptance authority.
- Development material and the protected heldout set are not counted as
  independent evidence in the report.

Passing a small hand-adjudicated set is **not** a whole-corpus accuracy claim
and grants no production or publication credit.

## Bounded execution proposal

After explicit approval, and only for this set:

- Build the item pools from the already-present source corpus and cached
  sources; do not rebuild the graph or the queue.
- Root reads each item source-first and freezes labels **before** any model
  prediction is produced (predictions must not precede or inform labels).
- Train on a bounded subset with the verified `neuroclaw` environment
  (`C:\Users\45846\anaconda3\envs\neuroclaw\python.exe`), fixed seed, fixed
  epochs, and a recorded metrics receipt.
- Report every bucket separately, including the ones that failed.

The concrete spend/request cap, training-size cap and the go/no-go for step 4
(optional shadow wiring) require separate, explicit user authorization; nothing
here should be read as that authorization.

## What this plan does not do

- It does not train, dispatch or wire anything, and starts no background work.
- It does not modify the fixed layer, the frozen reconstruction artifacts, the
  source corpus or the protected heldout assignments.
- It does not turn a structural score into a truth, novelty or first-report claim.
