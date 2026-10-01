# GNN offline repair and withdrawal — 2026-09-27

## Scope and current decision

User approved correcting the experiment, without increasing training budget.
This turn repairs offline contracts and code; it does not run an experiment,
call providers, launch agents, or read new source abstracts. Fixed datasets,
corpus, heldout assignments and production routing are unchanged.

**Current status: HELD_NO_TRAINING.** The previous run cap was six. Retained
JSON alone contains 24 training records (6 initial + 6 matched + 2 smoke +
4 tuning + 6 retained label evaluation). This is a lower bound, not a total:
overwritten evaluation attempts and an unpersisted measurement rerun are not
recoverable from those files. No budget remains for another experimental run.

The immutable, source-hash-bound correction is
`tmp/gnn_validation_20260927/OFFLINE_CORRECTION_V1.json`.
`models/kg_link_prediction/audit_validation.py` generated it without changing
any of the nine bound input artifacts, and verified their hashes afterward.
It refuses to overwrite an existing output. All 80 old labels are held for
re-review, **not relabelled as negative**; no new scientific label is supplied.

## Claims withdrawn

- Forward support does not establish a contradicted reverse. The old 24
  source-verified hard negatives are not established.
- Sentence co-occurrence and a hand-written `positive_supported` flag do not
  verify measurement, scope, entity identity, evidence role, or scientific truth.
  Counting endpoint pairs does not establish 19 independent facts/cohorts.
- The 0.841 and 0.889 numbers are development diagnostics, not accepted accuracy.
  Seventeen directly trained edges and six without direct-edge overlap do not
  constitute a source-family-isolated test. One of 24 labels was unmeasured.
- The 64% and 4x claims cannot be extrapolated from selected noisy candidates
  to the graph. Name co-occurrence is not a semantic recall ceiling either.
- Five candidate paths do not prove budget infeasibility. Inspection also
  identifies `PP:000` and `PP:001` as repeated-vertex paths; inverse measurement
  round trips are not new endpoint chains. The remaining three are not thereby
  accepted or scientifically adjudicated.
- The prior candidate SQL did not filter `fold='corpus'`. No claim that the
  previous scan preserved heldout exposure is accepted without a separate
  exclusion audit. This turn does not consume any heldout content.
- Experimental training did occur. The matched result reports 7.11 GiB allocated,
  above the 6 GiB cap. Time being under two hours did not authorize more runs.

Old reports and JSON remain historical; this correction supersedes their
scientific interpretation. No retrospective blind-validation claim is made.

## Implemented offline safeguards

### Sources, mappings and labels

`source_packets.py` opens SQLite read-only, selects corpus records in stable
order, excludes supplied protected/exposed work IDs before fetching payloads,
checks the compiled input hash/length, decompresses strictly and parses the
abstract rather than matching compressed bytes or a JSON envelope. It returns
at most 60 records per call, preserves raw text and distinguishes document,
compiled-input and abstract-text hashes. It leaves source family and labels
unset. This loader was tested on synthetic databases only; it is not a new scan.
The caller must maintain an aggregate exposure/reading ledger across calls.

`validation.py` implements word-boundary alias recall, preserves ambiguous and
nested matches, and never calls them resolved entities. Exact mappings still
require review. Label checks require raw-text offsets, matching source hashes,
work/family identity, scope, measurement and role. A contradicted label requires
an explicit exact-scoped contradiction decision; reversal, absence and nulls
cannot substitute. These checks validate the recorded contract, not the truth
of a human review. New labels need source-first review before any predictions.

### Split and message graph

Connected components join work identities, source families, fact families and
endpoint-pair proxies before deterministic 60/20/20 assignment. This conservative
split keeps inverse/synonym endpoint pairs together. It does not move unseen
validation/test entities into train or guarantee populated folds on tiny sets.
Canonical IDs and source/version-family review remain prerequisites.

Preflight refuses missing folds, unknown/exposed evaluation cases, missing
supported/contradicted/unresolved evaluation classes, unknown provenance and
cross-fold leakage. The graph filter removes source-family, fact, endpoint and
source-feature leakage, including protected works. Unknown provenance is held,
not admitted. The old subgraph lacks these bindings and is **not qualified**.
Node feature matrices require their own provenance audit before use; these
utilities do not certify arbitrary external features.

### Scoring and resource control

- Generic trainer ranking now calls the selected decoder rather than always
  applying DistMult. Ties use average rank; empty metrics are null.
- The directional decoder computes by relation group instead of expanding a
  matrix per triple. Synthetic forward values and gradients match the reference;
  this is not a measured real-run GPU memory guarantee.
- The GNN adapter rejects unknown graph edges, OOV scoring and nonfinite logits.
  A failed graph refresh clears old embeddings rather than silently retaining
  stale scores or returning a measured 0.5.
- `ValidationStopper` accepts only finite validation metrics and immutable
  checkpoint bytes; test/training metrics cannot choose its checkpoint.
- `RunGuard` persists immutable authorization and reservations in SQLite,
  serializes reservation transactions, counts failed attempts, blocks unknown
  attempts and duplicate keys, and charges reserved rather than guessed time.
  Prior overruns can be imported without resetting the budget.
- Its worker supervisor terminates an owned child at its wall-time/RSS bound;
  CUDA allocator fraction must be installed **inside that worker before model
  allocation**. The PyTorch allocator cap is not a device-wide cap on other
  programs or third-party allocations. Such usage requires separate monitoring.

## What is deliberately not wired or claimed

These are tested offline building blocks, **not a qualified experiment runner**.
The legacy `tmp/gnn_validation_20260927` training scripts remain historical and
must not be rerun. The generic trainer is still a legacy triple-split baseline;
its new ranking fix does not make its splitting/early stopping scientifically
qualified. No live route or experimental launcher is enabled by these changes.

Before another run: assemble reviewed source/version families and exact labels,
freeze hashes and split before predictions, audit all message/feature inputs,
connect preflight + durable reservation + supervised worker + in-worker CUDA
cap + validation checkpoint selection into one entrypoint, and obtain a new
bounded training allocation. A new directory/tag is not a budget reset. No
request to expand budget is being made in this offline repair turn.

## Verification receipt

In the installed `neuroclaw` environment, the focused repair, decoder, source
loader, scorer, direction, hypothesis-path and novelty-policy suites pass:
**121 passed, 1 deselected**. The existing 300-step trainability test was
deliberately deselected; this turn performs no optimizer training. Synthetic
forward/gradient equivalence checks do run on CPU. One upstream PyG deprecation
warning remains. `git diff --check` passes for the touched model/doc paths.

Tests include concurrent reservations, restart/failed-attempt holds, resource
overflow, hash tampering, pre-decode heldout exclusion, substring false matches,
family/proxy leakage, empty/tied metrics, invalid reference withdrawal and
immutable correction output. Test success is technical, not scientific accuracy.
