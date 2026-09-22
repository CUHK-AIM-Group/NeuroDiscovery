"""Build an auditable pre-imputation sample manifest for Case Study 2."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


COUNT_SUFFIX = "_count"


def as_bool(value: object) -> bool:
    """Interpret common CSV boolean representations."""

    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def phenotype_completeness(row: Mapping[str, object]) -> int:
    """Score longitudinal clinical and imaging coverage for tie-breaking."""

    visits = int(float(str(row.get("visits") or 0)))
    repeated_measures = sum(
        int(float(str(value or 0)))
        for key, value in row.items()
        if key.endswith(COUNT_SUFFIX)
    )
    return visits * 10 + repeated_measures


def build_preimputation_manifest(
    qc_rows: Iterable[Mapping[str, object]],
    eligible_rows: Iterable[Mapping[str, object]],
    exact_matches_by_batch: Mapping[str, int],
    related_pairs: Iterable[tuple[str, str]],
) -> list[dict[str, object]]:
    """Apply sample QC, collapse duplicate arrays, and prune related subjects.

    Relatedness is resolved after duplicate-array selection. Within each
    related component, the participant with the most complete longitudinal
    phenotype is retained; exact reference-matched variant count is the next
    tie-breaker.
    """

    eligible = {
        str(row["subject_id"]): dict(row)
        for row in eligible_rows
        if as_bool(row.get("case2_core_eligible"))
    }
    manifest: list[dict[str, object]] = []
    for source in qc_rows:
        row = dict(source)
        iid = str(row["IID"])
        batch = str(row["batch"])
        phenotype = eligible.get(iid)
        reasons: list[str] = []
        if phenotype is None:
            reasons.append("not_case2_core_eligible")
        if not as_bool(row.get("EUR_compatible")):
            reasons.append("non_eur_or_eur_outlier")
        if as_bool(row.get("heterozygosity_outlier")):
            reasons.append("heterozygosity_outlier")
        if as_bool(row.get("sexcheck_problem")):
            reasons.append("sexcheck_problem")
        manifest.append(
            {
                **row,
                "subject_id": iid,
                "exact_reference_matches": int(exact_matches_by_batch.get(batch, 0)),
                "phenotype_completeness": phenotype_completeness(phenotype or {}),
                "base_qc_pass": not reasons,
                "base_qc_fail_reasons": ";".join(reasons),
                "selected_array": False,
                "duplicate_array_removed": False,
                "relatedness_removed": False,
                "final_preimputation_keep": False,
            }
        )

    passing_by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest:
        if row["base_qc_pass"]:
            passing_by_subject[str(row["subject_id"])].append(row)
    selected_by_subject: dict[str, dict[str, object]] = {}
    for iid, candidates in passing_by_subject.items():
        candidates.sort(
            key=lambda row: (
                -int(row["exact_reference_matches"]),
                -int(row["phenotype_completeness"]),
                str(row["batch"]),
            )
        )
        selected_by_subject[iid] = candidates[0]
        candidates[0]["selected_array"] = True
        for row in candidates[1:]:
            row["duplicate_array_removed"] = True

    adjacency: dict[str, set[str]] = defaultdict(set)
    selected_ids = set(selected_by_subject)
    for left, right in related_pairs:
        if left == right or left not in selected_ids or right not in selected_ids:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        component: list[str] = []
        while stack:
            iid = stack.pop()
            if iid in visited:
                continue
            visited.add(iid)
            component.append(iid)
            stack.extend(sorted(adjacency[iid] - visited))
        component.sort(
            key=lambda iid: (
                -int(selected_by_subject[iid]["phenotype_completeness"]),
                -int(selected_by_subject[iid]["exact_reference_matches"]),
                iid,
            )
        )
        for iid in component[1:]:
            selected_by_subject[iid]["relatedness_removed"] = True

    for row in manifest:
        row["final_preimputation_keep"] = bool(
            row["selected_array"] and not row["relatedness_removed"]
        )
    return manifest
