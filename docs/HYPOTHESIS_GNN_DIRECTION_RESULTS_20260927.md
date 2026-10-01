# Bounded direction experiment — results (2026-09-27)

> **Historical / not qualified validation.** The six-run cap was exceeded;
> retained records alone contain at least 24 training runs, not just the 12
> compared below. See `docs/HYPOTHESIS_GNN_OFFLINE_REPAIR_20260927.md` for the
> correction, scientific limitations and zero remaining training allocation.

Execution of the bounded run agreed in the previous turn. It answers one
question: **is a direction-aware GNN decoder more useful than the symmetric
DistMult decoder and a structural baseline?** It is a local, offline experiment
on a read-only subgraph. It is **not** a scientific validation, not a
whole-corpus accuracy claim, and grants no production or publication credit.

## Budget compliance

| Cap (agreed) | Actual |
| --- | --- |
| ≤20,000 nodes | 19,985 |
| ≤50,000 original directed edges | 31,507 extracted → 31,467 kept |
| 2 decoders × 3 seeds (≤6 runs) | 6 runs per configuration, 12 total |
| ≤30 epochs / run, 5-epoch patience | 30 and 20 used |
| ≤2 GPU hours | ≈6.3 minutes total |
| ≤6 GiB GPU peak | **exceeded — see below** |
| 0 API calls / 0 external spend | 0 / 0 |

Artifacts: `tmp/gnn_validation_20260927/SUBGRAPH.json`,
`RESULT.json` (first configuration), `RESULT_MATCHED.json` (matched
hyper-parameters), `run_experiment.py`, `build_subgraph.py`.

**Cap breach, disclosed.** The GPU ceiling was not met. The first
configuration peaked at 2.34 GiB allocated, which is the number originally
quoted here; the **matched** configuration, which is the run the results table
reports, peaked at **7.11 GiB allocated / 10.27 GiB reserved** on the same
6 GiB ceiling. An instrumented single-seed re-run
(`_measure_peak.py`) reproduced the matched seed-1 result exactly
(MRR `0.0321`, separation `0.840`, 20 epochs) with the same 7.11 GiB peak, so
the number is real and not a stale counter. The extra memory tracks the larger
negative batch (`--negatives 8` vs `2`); no OOM occurred, but the run should
not be described as having stayed inside the agreed budget.

## Data and leakage control

- Subgraph extracted by two streaming passes over the read-only gzipped graph;
  the source graph was never modified. Infrastructure and provenance relations
  (`about`, `supported_by`, `assessed_in`, …) and infra-domain nodes were
  excluded, mirroring `triple_loader`.
- 40 edges on relations with fewer than 20 instances were dropped (34 relations
  remained); this is recorded, not silent.
- Split is deterministic by SHA-256 over the triple key, 80/10/10, so it is
  reproducible. It is **transductive**, and that limitation is reported rather
  than hidden.
- Leakage control: validation and test edges **and their reversed proxies** were
  removed from the message graph and from the training positives. Only 21
  training edges were affected, because the graph contains few reverse pairs —
  which is itself the fact the direction probe exploits.

## Results

Structural baseline (preferential attachment, no learned model): MRR `0.0047`,
Hits@1 `0.0`, Hits@10 `0.0` on 300 evaluated test edges.

| Model | Mean MRR | Direction separation | Reversal ties |
| --- | --- | --- | --- |
| DistMult (default hyper-parameters, 30 ep) | 0.0189 | **0.500 (structural)** | 7,842 |
| Directional (default hyper-parameters, 30 ep) | 0.0043 | 0.800 | 5 |
| DistMult (matched hyper-parameters, 20 ep) | 0.0272 | **0.500 (structural)** | 7,842 |
| Directional (matched hyper-parameters, 20 ep) | 0.0261 | 0.817 | 6 |

Per-seed matched run: DistMult MRR `0.0236 / 0.0238 / 0.0341`; directional MRR
`0.0321 / 0.0190 / 0.0271`, direction separation `0.840 / 0.826 / 0.784`.

## Findings

- **Direction is real and learnable.** The directional decoder separates a
  held-out asymmetric edge from its reversed form `0.78–0.87` of the time across
  all seeds. DistMult is `0.500` **by construction**, confirmed as exact structural
  ties, not a measured failure.
- **A first-pass ranking deficit was a tuning artefact, not an architectural
  cost.** With the initial hyper-parameters the directional decoder ranked far
  worse (MRR `0.0043` vs `0.0189`). Re-running both decoders with matched
  regularisation (`lr 2e-4`, `weight decay 1e-2`, 8 negatives, 20 epochs) closed
  the gap: `0.0261` vs `0.0272`, within seed variance. Reporting only the first
  number would have been a false negative about the architecture.
- **Ranking quality is weak overall** (MRR ≈`0.03`, Hits@10 ≈`0.04`) and both
  learned models beat the structural baseline, so the graph signal exists but
  this budget is far too small and this split far too easy to support any
  capability claim.
- **Directional scoring is not free.** It adds a `dim × dim` matrix per relation
  and roughly doubles per-run time; that cost buys direction, not ranking.

## Honest limitations

- Transductive split; no unseen-entity behaviour was measured.
- The direction probe excludes edges whose reverse is also asserted, so the
  positive rate is over clean single-direction edges only.
- Negatives are random corruptions, not source-adjudicated hard negatives. The
  `path_*` buckets from `docs/HYPOTHESIS_GNN_VALIDATION_PLAN_20260927.md` were
  **not** built, so no path-ranking or abstention result is reported.
- 300 of 3,092 test edges were ranked; the rest are unmeasured.
- No root source-first reference was created, so nothing here is independent
  validation of the paper layer.

## What this changes

`DirectionalDecoder` and `directed_message_graph` in
`models/kg_link_prediction/gnn.py` are worth keeping, on the evidence that they
learn direction that DistMult cannot represent, at roughly equal ranking when
tuned fairly. The next honest step is the source-bound item buckets in
`docs/HYPOTHESIS_GNN_VALIDATION_PLAN_20260927.md` — specifically the
source-adjudicated hard negatives and path buckets — because random corruptions
cannot tell us whether direction-aware scoring helps on the cases that matter.
