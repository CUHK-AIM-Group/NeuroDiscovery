# Case Study 2 v3 phase-held-out results

## Scope and status

Case Study 2 v3 is a cohort-phase-held-out validation of 210 prespecified
pathway-PRS -> imaging-marker -> longitudinal-outcome association chains in
ADNI. ADNI3 was used for method development. ADNI1, ADNIGO, and ADNI2 were
reserved for validation. This is not endpoint holdout, temporal hindcasting,
or a causal mediation claim.

The protocol, public candidate registry, NeuroDiscovery initial ranking, six
baseline ranking sets, evaluation metrics, thresholds, and clear-SOTA rule
were hash-locked before the phase-held-out association files were opened.
The final run is sealed with status `sealed_without_clear_sota`.

## Held-out scientific results

| Quantity | Result |
|---|---:|
| Subjects in locked ADNI1/GO/2 cohort | 350 |
| Subjects represented in analysis rows | 336 |
| Prespecified candidates | 210 |
| Estimable candidates | 210 |
| Nominal weakest-link chain hits | 2 |
| Pathway-outcome family-FDR hits | 0 |
| Global-FDR hits | 0 |
| Registered ADNI3-selected chains | 4 |
| Strict directional Holm replications | 0 |

The two nominal chains both used the curated AD-risk GWAS pathway PRS and
amyloid PET Centiloids. Their outcomes were LDELTOTAL and mPACCdigit. Neither
survived the prespecified pathway-outcome family FDR.

None of the four ADNI3-selected chains passed the complete directional
replication rule in ADNI1/GO/2. The closest registered chain was the myelin
pathway PRS -> hippocampal volume -> LDELTOTAL chain: all three one-sided
component P values were below 0.05, but the indirect-effect P value was 0.0328
before and 0.1310 after the frozen four-chain Holm correction.

## Generator comparison

Each method contributed ten independent runs. The six baselines used their
native frameworks and frozen static rankings. NeuroDiscovery used one frozen
outcome-blind initial ranking and ten prespecified random seeds for the
sequential closed loop. Every NeuroDiscovery run contains 42 ranking commits,
42 matching feedback traces, and complete execution of all 210 candidates.

| Method | Runs | Held-out evidence nDCG, mean | s.d. |
|---|---:|---:|---:|
| NeuroDiscovery | 10 | 0.863988 | 0.000489 |
| Open Co-Scientist | 10 | 0.853665 | 0.033676 |
| SciAgents | 10 | 0.827307 | 0.026943 |
| Biomni | 10 | 0.815322 | 0.017846 |
| BrainPilot | 10 | 0.808843 | 0.013513 |
| AI Scientist-v2 | 10 | 0.805772 | 0.021343 |
| Virtual Lab | 10 | 0.798631 | 0.013367 |

NeuroDiscovery had the highest mean held-out-chain evidence nDCG. It passed
the frozen paired superiority test against five baselines, but not against
Open Co-Scientist.

| Baseline | Mean difference | 95% bootstrap CI | Exact one-sided P | Holm P | Superior |
|---|---:|---:|---:|---:|---:|
| AI Scientist-v2 | 0.058217 | [0.046083, 0.071213] | 0.000977 | 0.005859 | Yes |
| Open Co-Scientist | 0.010324 | [-0.006504, 0.032236] | 0.218750 | 0.218750 | No |
| SciAgents | 0.036681 | [0.021335, 0.052648] | 0.001953 | 0.005859 | Yes |
| Virtual Lab | 0.065358 | [0.056373, 0.072670] | 0.000977 | 0.005859 | Yes |
| BrainPilot | 0.055145 | [0.047778, 0.063621] | 0.000977 | 0.005859 | Yes |
| Biomni | 0.048666 | [0.038443, 0.059240] | 0.000977 | 0.005859 | Yes |

## Formal interpretation

The normalized replication-recovery AUC is non-identifiable because its
frozen positive set is empty. The clear-SOTA rule required NeuroDiscovery to
pass all six comparisons on both replication-recovery AUC and held-out
evidence nDCG. It therefore cannot be satisfied in this run. Thresholds were
not relaxed, candidates were not replaced, and the v3 result must not be
reported as clear SOTA.

The defensible conclusion is narrower: NeuroDiscovery ranked the continuous
held-out weakest-link evidence most effectively on average and was superior to
five of six native autoresearch baselines, but it was not significantly better
than Open Co-Scientist and the registered replication endpoint contained no
strict positives.

Any further optimization requires a new protocol identifier and a genuinely
untouched cohort. The sealed v3 outcomes cannot be used to retune and then
retest v3.

## Authoritative artifacts

- Final seal: `FINAL_RESULT_SEAL.json`
- Held-out result lock: `hidden_phase_heldout_results/PHASE_HELDOUT_RESULTS.lock.json`
- Generator verdict: `generator_evaluation/case2_sota_verdict.json`
- Performance table: `generator_evaluation/case2_generator_performance_table.csv`
- Paired comparisons: `generator_evaluation/case2_primary_paired_comparisons.csv`

The artifact root is:

`\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc\case2_adni_genetics_v1\experiments\case2_adni_phase_holdout_v3\56cdd4b227ce`

Last updated: 2026-08-16 17:04 HKT
