"""Specialized compact Case Study review groups and deterministic merger."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from neurooracle.scripts.benchmark_compact_case_study_reaudit import (
    expand_compact_response,
)
from neurooracle.scripts.run_full_graph_case_study_reaudit import (
    GATE_NAMES,
    request_json,
    validate_response,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


GROUP_LABELS: dict[str, tuple[int, ...]] = {
    "disease": (0, 2, 3, 6),
    "longitudinal": (4, 15, 16),
    "molecular": (1, 5),
    "treatment": (7, 8, 9, 10, 11),
    "cognition": (12, 13, 14),
}

COMMON = """You are one specialist in an immutable neuroscience claim audit. Judge only the target claim assertion (raw text plus subject-predicate-object). Paper title/abstract/context may clarify terms or verify the assertion but must not donate a missing relation, modality, diagnosis, outcome, comparison, or time direction from a parallel finding. Evaluate every allowed label independently and use only supplied evidence. Do not infer unreported results.

Return JSON only as {"b":"exact supplied batch token","r":[[i,[global_label_indexes],confidence_0_to_100,gate_bitmask,secondary_boolean,reason_code],...]}. Preserve every i in order. Labels are sorted unique global registry indexes and may only come from this specialist's allowed set. Gate bits 0..9 are neural disease change, complete case2 chain, longitudinal, genetic/molecular-to-neural, disease-vs-disease, decoding, brain-age construct, pretreatment drug-response prediction, treatment selection, concrete stimulation target. Set secondary true for material ambiguity or confidence below 80. reason_code is evidence-specific and at most 12 words."""

GROUP_PROMPTS = {
    "disease": COMMON
    + """
Allowed labels:
0 Case1: target claim directly reports a brain-region/network/neural-system/neuroimaging change in at least one disease or disease group; one diagnosis is enough. Disease background, genetic risk alone, peripheral-only evidence, or a healthy-only neural phenotype fails. Gate bit 0 required.
2 Biomarker: target neural/imaging/pathology measure is directly associated with disease state, diagnosis, symptoms, clinically relevant outcome/risk, or classification/prediction. It need not be a validated clinical assay. A healthy-only gene-brain association fails.
3 Subtyping: target claim compares or identifies within-disease genotype, phenotype, pathology, or imaging-defined subgroups with differing neural/imaging features. Ordinary case-control work fails.
6 Differential diagnosis: target claim directly distinguishes disease A from disease B using neural/imaging evidence. Healthy-control comparison alone fails. Gate bit 4 required.""",
    "longitudinal": COMMON
    + """
Allowed labels:
4 Progression: a baseline neural/imaging feature predicts a later conversion or disease progression event. Gate bit 2 required.
15 Brain age: target claim produces, validates, or relates a brain-age estimate or brain-age gap to an outcome. Ordinary chronological-age effects fail. Gate bit 6 required. Set gate bit 2 additionally only when the target claim itself is longitudinal.
16 Prognosis: baseline/acute disease plus neural/imaging state predicts a later clinical or functional outcome. Gate bit 2 required. Do not assign for later imaging change alone, cross-sectional association, or a contemporaneous outcome.
Set longitudinal gate bit 2 without labels only when the target claim explicitly reports repeated follow-up or later measurement.""",
    "molecular": COMMON
    + """
Allowed labels:
1 Case2: target paper evidence supports the complete genetic/pathway -> brain imaging/physiology -> longitudinal clinical/cognitive outcome mediation or causal chain. All stages, time direction, and mediation/causal linkage are mandatory. Gate bit 1 required; also set bits 2 and 3.
5 Imaging genetics: a genetic, genomic, transcriptomic, protein, molecular-pathology, or pathway factor is directly linked to a neural/imaging phenotype. Null tests still qualify when the tested link is explicit. Genetic-to-clinical outcome without a neural phenotype fails. Gate bit 3 required.""",
    "treatment": COMMON
    + """
Allowed labels:
7 Drug response: a pretreatment patient/neural profile plus a named pharmacological treatment predicts later response. Treatment effect alone fails. Gate bit 7 required.
8 Personalised treatment: patient features inform selection between alternative treatments. Response association alone fails. Gate bit 8 required.
9 Drug repurposing: an existing drug is supported for a novel indication.
10 Adverse event: a drug/intervention predicts or causes a defined adverse event.
11 Neuromodulation target: a concrete stimulation site, circuit, or target is identified or validated. Generic brain mechanism or therapy mention fails. Gate bit 9 required.""",
    "cognition": COMMON
    + """
Allowed labels:
12 Functional localization: a target task, stimulus, or cognitive operation maps to a neural region/network/activation/connectivity pattern. A structural measure merely correlated with test score does not automatically localize function.
13 Cognitive decoding: neural patterns predict/decode a stimulus or mental state. Association or localization without held-out prediction/decoding fails. Gate bit 5 required.
14 Connectome-behavior: connectivity/network organization directly relates to or predicts behavior, cognition, symptom, or trait. Regional activation, volume, pathology, or generic imaging without a connectivity/network measure fails.""",
}


def execute_group_batch(
    group: str,
    rows: list[tuple[str, str, str]],
    payload: dict[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    if group not in GROUP_PROMPTS:
        raise ValueError(f"unknown specialist group: {group}")
    try:
        result, metadata = request_json(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            wire_api="responses",
            payload={**payload, "specialist_group": group},
            timeout=args.timeout,
            retries=args.transport_retries,
            responses_plain_json=True,
            transport=args.transport,
            instructions=GROUP_PROMPTS[group],
            max_output_tokens=args.max_output_tokens,
        )
        if metadata.get("status") not in {None, "", "completed"}:
            raise ValueError(f"Responses status is not completed: {metadata.get('status')}")
        if metadata.get("incomplete_details"):
            raise ValueError("Responses API reported incomplete output")
        reviews = expand_compact_response(rows, result, batch_token=payload["b"])
        allowed = {CASE_STUDY_IDS[index] for index in GROUP_LABELS[group]}
        for review in reviews:
            unexpected = set(review["claim_case_study_ids"]) - allowed
            if unexpected:
                raise ValueError(f"{group} specialist returned disallowed labels: {sorted(unexpected)}")
        return {"ok": True, "group": group, "reviews": reviews, "metadata": metadata}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "group": group,
            "claim_ids": [row[0] for row in rows],
            "error": f"{type(exc).__name__}: {exc}",
        }


def merge_group_reviews(
    rows: list[tuple[str, str, str]], group_results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    missing = set(GROUP_LABELS) - set(group_results)
    failed = [name for name, result in group_results.items() if not result.get("ok")]
    if missing or failed:
        raise ValueError(f"specialist results incomplete: missing={sorted(missing)} failed={sorted(failed)}")
    by_group = {
        group: {review["claim_id"]: review for review in result["reviews"]}
        for group, result in group_results.items()
    }
    merged: list[dict[str, Any]] = []
    for claim_id, _, _ in rows:
        labels: set[str] = set()
        gates = {name: False for name in GATE_NAMES}
        confidence = 1.0
        secondary = False
        reasons: list[str] = []
        for group in GROUP_LABELS:
            review = by_group[group][claim_id]
            labels.update(review["claim_case_study_ids"])
            for gate, value in review["gates"].items():
                gates[gate] = gates[gate] or value
            confidence = min(confidence, review["confidence"])
            secondary = secondary or review["needs_secondary_review"]
            if review["claim_case_study_ids"] or review["needs_secondary_review"]:
                reasons.append(f"{group}: {review['reason'].removeprefix('Compact primary evidence: ')}")
        ordered_labels = [label for label in CASE_STUDY_IDS if label in labels]
        merged.append(
            {
                "claim_id": claim_id,
                "claim_case_study_ids": ordered_labels,
                "confidence": confidence,
                "reason": "; ".join(reasons) if reasons else "All five specialists found no direct Case Study evidence.",
                "needs_secondary_review": secondary,
                "gates": gates,
            }
        )
    return validate_response(rows, {"reviews": merged})
