# Case Study 2 v3 execution protocol

Case Study 2 tests pathway-PRS -> imaging-marker -> longitudinal clinical
outcome mediation in ADNI. Version 3 is a cohort-phase-held-out protocol made
after the ADNI3 pilot exposed the planned endpoints.

## Freeze semantics

Three distinct concepts must not be conflated:

1. **KG snapshot pinning** fixes the NeuroOracle graph and claim-store hashes
   used to build outcome-blind priors.
2. **Initial ranking freeze** commits every generator's complete 210-candidate
   order before ADNI1/GO/2 association results are accessed.
3. **Cohort-phase holdout** reserves ADNI1/GO/2 for formal validation after
   model development on ADNI3.

This protocol is not temporal hindcasting. It has no freeze year and does not
claim that the graph predates the held-out cohort.

## Frozen inputs

The protocol root contains:

- `public_candidate_registry.csv`: the 210 executable hypotheses visible to
  every generator. It excludes coefficients, P values, FDR values, expected
  directions, KG ranks, NeuroDiscovery scores, and experimental labels.
- `PRIVATE_REPLICATION_REGISTRY.csv`: four ADNI3 nominal chains and their
  expected directions. Generators must never read this file.
- `protocol_freeze_manifest.json`: registry, KG, claim-store, and protocol
  hashes plus the phase-held-out freeze ID.

## Ranking stage

NeuroDiscovery and six native autoresearch frameworks each produce ten initial
rankings. The baselines are AI Scientist-v2, Open Co-Scientist, SciAgents,
Virtual Lab, BrainPilot, and Biomni. Native invalid, duplicate, or missing
slots consume their requested slot and are not repaired after generation.

BrainPilot must be launched against the local OpenAI Responses endpoint rather
than its default Anthropic Messages route:

```powershell
$env:CASE_STUDY_LOCAL_API_KEY = "<local API key>"
& core/scripts/start_brainpilot_local_service.ps1 `
  -DataDir "<brainpilot run directory>" `
  -Port 18080 `
  -Model "gpt-5.5" `
  -BaseUrl "http://localhost:8080/v1" `
  -ReasoningEffort high `
  -Restart
```

The launcher reads the key only from the process environment. Its model file
contains an environment-variable reference and its runtime manifest contains
no credential.

The baseline campaign is resumable: existing `final.txt` batch artifacts and
`search_policy.json` trial artifacts are reused unless `--force` is supplied.
BrainPilot and Biomni each use one serial worker, while the two frameworks run
concurrently.

Before writing `BASELINE_RANKINGS.lock.json`, the campaign must pass
`RANKING_AUDIT.json`. The audit checks the full method-by-trial matrix, exact
candidate IDs, complete 210-candidate permutations, forbidden private fields,
and cross-method Top-K collapse. The ranking lock includes the audit hash.

## Held-out validation

Only after both NeuroDiscovery and baseline ranking locks exist may the
evaluation plan be frozen and ADNI1/GO/2 associations be computed. The primary
scientific endpoint asks whether each of the four registered ADNI3 chains
replicates directionally with one-sided `a`, `b`, and indirect-effect P values
below 0.05 and Holm-adjusted indirect P below 0.05 across the four chains.
The full 210-candidate family/global FDR screen is secondary.

Generator performance is assessed by normalized replication-recovery AUC and
held-out chain-evidence nDCG. A clear SOTA claim requires NeuroDiscovery to beat
all six baselines on both metrics with a positive paired mean difference, a
positive lower 95% bootstrap bound, and Holm-adjusted one-sided exact sign-flip
P below 0.05 for all twelve metric-baseline comparisons. If no registered chain
replicates, clear SOTA is non-identifiable rather than negative by fiat.

## Order of operations

1. Seal and preserve the v2 negative result.
2. Freeze the v3 protocol and private replication registry.
3. Build and lock the outcome-blind NeuroDiscovery policy.
4. Run, audit, and lock all six baseline methods across ten trials.
5. Freeze the evaluation plan.
6. Access ADNI1/GO/2 associations and run the held-out mediation analysis.
7. Evaluate generators, seal the final result, and stop local services.

Last updated: 2026-08-16 15:33 HKT

## Design language

Case Study 2 v3 is a **cohort-phase-held-out replication**. ADNI3 is the
development phase and ADNI1/GO/2 is the untouched validation phase. The target
endpoint definitions were already seen in ADNI3, so v3 is not endpoint-held-out
and is not temporal hindcasting.

The canonical KG and claim store are hash-pinned for reproducibility. Snapshot
pinning does not imply a historical publication cutoff.

## Frozen data boundary

All seven generators receive the public 210-candidate registry. They do not
receive ADNI3-selected candidate IDs, expected directions, ADNI effect sizes,
P values, FDR labels, continuous held-out evidence, or NeuroDiscovery scores.

The four ADNI3-selected replication candidates and their expected component
directions remain in `PRIVATE_REPLICATION_REGISTRY.csv`. This file is hashed at
protocol verification but is parsed only by the evaluator after all initial
rankings and the evaluation plan are locked.

## Execution order

1. Verify the protocol freeze and pinned input hashes.
2. Generate 10 native runs for each of AI Scientist-v2, Open Co-Scientist,
   SciAgents, Virtual Lab, BrainPilot, and Biomni.
3. Compile each native output into a complete deterministic SearchPolicy and
   create `BASELINE_RANKINGS.lock.json`.
4. Verify the already locked NeuroDiscovery initial ranking.
5. Freeze `case2_adni_generator_evaluation_v3.json` as
   `EVALUATION_PLAN.lock.json`.
6. Create the held-out access-start record, then and only then fit the unchanged
   mediation model in ADNI1/GO/2.
7. Run 10 NeuroDiscovery closed-loop trials. Each next batch is hash-committed
   before its selected outcomes are looked up.
8. Recompute all labels, metrics, paired tests, and hashes independently and
   create `FINAL_RESULT_SEAL.json`.

## Scientific endpoint

The primary scientific family contains the four ADNI3 chains that met the
sealed v2 nominal rule. A chain replicates only when all three effects match
their frozen directions, all three one-sided component P values are below
0.05, and the one-sided indirect-effect P value also passes Holm correction
across the four candidates.

The 210-candidate pathway-outcome family FDR and global FDR endpoints are
secondary.

## Generator endpoints

The two primary generator metrics are:

- area under cumulative recovery of independently replicated registered chains;
- full-ranking nDCG using the frozen weakest-link evidence score
  `min(50, -log10(max(P_a, P_b, P_indirect)))`.

Non-estimable candidates receive zero evidence and retain their consumed rank.
NeuroDiscovery receives only categorical nominal-chain feedback from candidates
already executed. Raw statistics, continuous relevance, and private replication
labels remain evaluator-only.

## Clear-SOTA rule

NeuroDiscovery is clear SOTA only if it exceeds every baseline on both primary
metrics. For each of the 12 metric-baseline comparisons, the paired mean
difference and its 95% bootstrap lower bound must be positive, and the
Holm-adjusted one-sided exact sign-flip P value must be below 0.05. If no
registered candidate independently replicates, the recovery metric is not
identifiable and clear SOTA cannot be claimed.

Last updated: 2026-08-16 16:56 HKT
