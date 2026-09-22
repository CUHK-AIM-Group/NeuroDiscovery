# Case Study 2 readiness

Case Study 2 has one public identity. Historical directory names and protocol
hashes remain only as immutable provenance; they are not manuscript versions.

## Locked design

The outcome-blind master registry contains 5,265 hypotheses:

- 27 gene or pathway PRS exposures;
- 13 continuous imaging markers across structural MRI, amyloid PET, tau PET,
  tau PVC PET, and FDG PET;
- 15 longitudinal clinical outcomes.

Candidate construction prefers the `p1em03` PRS threshold when available and
otherwise uses `p5em02`. No association estimate, mediation estimate, P value,
or future validation label participates in candidate inclusion.

The primary generator endpoint is held-out graded-evidence `nDCG@100`.
NeuroDiscovery must be significantly better than all six native autoresearch
baselines after Holm correction, exceed the best baseline by at least 0.02
absolute nDCG or 10% cumulative evidence, win at least five of seven registered
experiment counts, and have at least five strict positive labels available.

## Current data status

The available development table contains 691 genotyped participants from
ADNI1, ADNIGO, ADNI2, and ADNI3. Of the 5,265 candidates, 3,861 have at least
400 complete development cases. These data have already contributed to Case
Study 2 method development and cannot serve as the final unseen cohort.

No ADNI4 pathway-genetics input or independent AIBL/OASIS multimodal genetics
cohort is currently present on the registered storage roots. The protocol is
therefore locked with status `design_locked_pending_unseen_cohort`, and final
outcome access is fail-closed.

The ranking freeze in this protocol means committing all method policies,
seeds, code hashes, metrics, and thresholds before computing effects in the
unseen cohort. It is distinct from the publication-year KG freeze used in
temporal hindcasting.

## Generated audit

The current readiness artifacts are stored at:

`\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc\case2_adni_genetics_v1\experiments\case2\readiness\20260816_1746`

The directory contains the candidate registry, dataset inventory, power
planning table, protocol lock, and readiness audit. A final evaluation remains
prohibited until a validated independent-cohort manifest passes every gate in
the protocol lock.

# Last Updated At: 2026-08-16 17:53 HKT
