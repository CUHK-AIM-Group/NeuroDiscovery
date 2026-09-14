# KG v3 Case Study membership policy

## Status

- Canonical rubric: `neurooracle/src/CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md`
- Rubric epoch: `2026-08-10.peer17.v2`
- Formal Case Study registry: 17 ordered, non-exclusive IDs from
  `neurooracle.src.case_study_scope.CASE_STUDY_IDS`
- New-claim seal: `case_study_membership_contract.v4`
- Prior extraction seal: `case_study_membership_contract.v3` (preserved and
  still verifiable)
- Historical full-v2 seal: `case_study_reaudit_contract.v1` (preserved)

`general` is implicit shared-corpus membership. `hindcasting` is a validation
protocol. Neither value is a Case Study ID.

## One policy, two consumers

`neurooracle.src.case_study_membership_policy` parses the canonical rubric and
provides its version, SHA-256, ordered definitions, mandatory gates, prompt
block, and strict decision validator. Both standalone re-audit and new claim
extraction import this module. Do not copy the definitions into another prompt,
script, notebook, or manual top-up builder.

Changing the rubric file or the ordered registry changes its hash and therefore
starts a new, explicitly versioned audit epoch. Never overwrite a finalized
decision in place.

Case 2 follows the same evidence-contribution principle as Case 1. A Case 1
paper may contribute one disease-specific neural result without itself being
transdiagnostic. Likewise, a Case 2 claim may contribute either the
genetic/pathway-to-neural link or the baseline-neural-to-later-outcome link;
the paper need not contain the complete mediation chain.

## v3 expansion lifecycle

All Case Study search campaigns use the frozen publication window 1980–2026.

1. Search broadly for relevant papers. Query provenance is candidate-generation
   evidence only and never assigns Case Study membership.
2. Deduplicate candidates against both the formal KG and every staged queue by
   PMID, DOI, PMCID, arXiv/OpenAlex ID, then normalized title and year.
3. Retrieve the abstract/full text and retain its source identity.
4. Extract source-linked claims. Every returned claim must also contain:
   `case_study_ids`, all nine `case_study_gates`, verbatim
   `scope_evidence_spans`, `scope_confidence`, and
   `scope_decision_basis`.
5. Validate all IDs and gates against the frozen policy. Unknown IDs, duplicate
   IDs, missing gates, string booleans, labels with a false mandatory gate,
   ungrounded evidence spans, or missing decision reasons fail the response and
   trigger extraction retry.
   Case 2 is then canonicalized deterministically: every validated
   `imaging_genetics`, `progression_prediction`, or `prognosis` claim also
   receives `case2_pathway_mediation`; a standalone Case 2 label is invalid.
6. Seal the accepted decision with the claim evidence hash, source-context
   hash, rubric hash, ordered-registry hash, labels, gates, confidence, and
   decision basis. The serialized record is stored in `scope_reaudit` with
   `review_status=final_complete`.
7. Stage claims and run entity resolution/deduplication. At actual KG mutation,
   call `ingest_claims(..., require_final_scope_audit=True)`. A missing or
   mismatched seal refuses the whole batch before graph mutation.
8. Recompute paper membership deterministically as the ordered union of all
   retained claim memberships from that paper.

## Reuse and correction

A finalized seal is canonical for its evidence and policy hashes. A model,
reasoning-effort, concurrency, or session change is not a reason to review it
again. Identical evidence reuses the stored decision.

Model inference before sealing is not mathematically deterministic, and a
sealed decision may still contain a semantic error. A correction must be an
explicit audited replacement with its reason and superseded decision hash; it
must not silently mutate the old decision.

## Compatibility boundary

The completed full-v2 re-audit remains valid under its historical v1 contract,
and claims extracted under the strict same-paper Case 2 rule retain their v3
seals. New v3-expansion claims use the component-based v4 contract. Existing
claims are not silently relabelled during collection; after KG v3 collection is
complete, one planned full-graph re-audit will project all retained claims to
the current rubric epoch.
