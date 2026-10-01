"""Versioned prompt amendment; not installed into any live sender by import."""

EXTRACTION_AMENDMENT_V3 = """
RESULT CONTRACT V3 supersedes conflicting extraction instructions above.
Return structured_result for each primary_result observation, alongside existing
fields. Unknowns remain null with a reason, never guessed from typical patients.
Population is not exposure. Association endpoints are predictor/exposure and
outcome; diagnosis belongs in population unless it is the tested predictor.
Within-group genotype association does not compare patients against controls.
baseline/follow-up are timepoints, never disease stages. PRS as exposure is not
automatically an adjustment. controls does not imply healthy controls.

One result is one population/exposure/outcome/comparison/time combination.
Split explicitly different diseases and cell-type risk scores; do not collapse
them into psychotic disorders or a combined OPC/RG exposure. Do not split a
single result just because its significance and signed direction occur in
adjacent sentences. Bind both sentences to the same observation. Do not split
an indivisible composite measure into invented component results. If uncertain,
record atomicity unresolved, proposition null and the reason in unresolved.

Source-bound slots use exact source phrases; canonical labels remain separate
in proposition. Provide enough quotes to bind conditions, anatomy and comparison,
not just a detached percentage. Offsets are zero-based Python Unicode character
offsets within the corresponding quote, end exclusive, after input normalization.
Each binding is {"quote_index":0,"start":0,"end":12,"text":"literal span"}.
Slots are {"value":"literal span","bindings":[binding],"unknown_reason":null}
or {"value":null,"bindings":[],"unknown_reason":"not reported"}.

structured_result shape:
{"schema":"source_result_v3",
 "kind":"between_groups|within_group_change|association|interaction|other",
 "population":slot,"exposure":slot,"outcome":slot,"comparator":slot,
 "timepoint":slot,"disease_stage":slot,
 "contrast":binding,
 "atomicity":{"status":"single|unresolved","reason":"explain result unit"},
 "test":{"status":"reported_significant|reported_not_significant|not_reported|derived|unresolved",
         "contrast":binding_or_null,"evidence":binding_or_null}}

contrast binds the source clause describing THIS comparison, not model prose.
For reported tests, test.contrast must bind that same comparison and test.evidence
must contain the source's explicit test result. A p value elsewhere in a long
quote does not establish significance for every outcome. Separate within-group
significant/non-significant activation is NOT a significant between-group effect.
Use not_reported for 'unchanged', 'no association' or failure to replicate unless
statistical significance is explicitly stated. Preserve null wording; do not
upgrade it to proven absence. Signed numeric r supports correlation direction;
positive/negative correlation is not higher/lower group volume. A confidence
interval-derived decision is derived, never reported_significant. Preserve raw
numeric text, including decimal whitespace, next to the parsed value.
Unknown scope and unresolved tests prevent release; matching text does not prove
semantic equivalence, study independence or scientific validity.
"""


def build_prompt(base_prompt):
    return base_prompt.rstrip() + "\n\n" + EXTRACTION_AMENDMENT_V3
