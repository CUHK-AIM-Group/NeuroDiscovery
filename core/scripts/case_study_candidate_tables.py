"""Build outcome-blind candidate registries for closed-loop case studies.

Task-specific experiment code supplies candidate factors and hidden validation
statistics.  This module is deliberately responsible only for stable identity,
KG-derived priors, physical outcome separation, and provenance manifests.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import (
    PUBLIC_REGISTRY_SCHEMA,
    OUTCOME_SCHEMA,
    sha256_file,
    validate_hidden_outcomes,
    validate_public_registry,
)
from core.scripts.case1_kg_stream import load_case1_kg_index_payload
from core.scripts.case_study_feedback_adapters import adapter_for


TABLE_BUNDLE_SCHEMA = "case-study-table-bundle.v1"


# Runtime factor names are stable experiment identifiers, while the KG stores
# scientific terminology. Keep this bridge explicit and outcome-independent.
SEMANTIC_VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "adhd": ("attention deficit hyperactivity disorder",),
    "mdd depression": ("major depressive disorder", "depression"),
    "ocd oc related": ("obsessive compulsive disorder",),
    "ptsd trauma": ("post traumatic stress disorder",),
    "psychosis sz sza": (
        "psychosis",
        "schizophrenia",
        "schizoaffective disorder",
    ),
    "bipolar": ("bipolar disorder",),
    "substance use": ("substance use disorder",),
    "correlation fc": (
        "functional connectivity",
        "resting state functional connectivity",
    ),
    "partial fc": (
        "partial correlation functional connectivity",
        "functional connectivity",
    ),
    "amplitude": (
        "amplitude of low frequency fluctuation",
        "fractional amplitude of low frequency fluctuation",
    ),
    "structure": ("structural mri", "regional brain volume"),
    "temporal": ("regional brain activity", "temporal variability"),
    "roi alff proxy": ("amplitude of low frequency fluctuation",),
    "roi falff proxy": ("fractional amplitude of low frequency fluctuation",),
    "corr mean": ("functional connectivity",),
    "corr mean abs": ("functional connectivity strength",),
    "corr negative mean": ("negative functional connectivity",),
    "corr positive mean": ("positive functional connectivity",),
    "corr node degree abs top10": (
        "functional connectivity",
        "node degree",
        "network centrality",
    ),
    "partial mean": ("partial correlation functional connectivity",),
    "partial mean abs": ("partial correlation functional connectivity",),
    "partial negative mean": ("negative functional connectivity",),
    "partial positive mean": ("positive functional connectivity",),
    "normalized volume fraction": (
        "regional brain volume",
        "gray matter volume",
        "brain volume",
    ),
    "roi temporal mean": ("regional brain activity",),
    "roi temporal mean abs": ("regional brain activity",),
    "roi temporal std": ("temporal variability",),
    "roi temporal variance": ("temporal variability",),
    "dmn": ("default mode network",),
    "default": ("default mode network",),
    "cont": ("frontoparietal control network", "cognitive control network"),
    "dorsattn": ("dorsal attention network",),
    "salventattn": ("salience network", "ventral attention network"),
    "sommot": ("somatomotor network",),
    "motor": ("motor network", "somatomotor network"),
    "vis": ("visual network",),
    "limbic": ("limbic network",),
    "cereb": ("cerebellum", "cerebellar network"),
    "acc": ("anterior cingulate cortex",),
    "aud": ("auditory network",),
    "dlpfc": ("dorsolateral prefrontal cortex",),
    "ips": ("intraparietal sulcus",),
    "ins": ("insula",),
    "tpj": ("temporoparietal junction",),
    "non affective vs affective psychosis": (
        "affective psychosis",
        "psychosis",
        "schizophrenia",
        "bipolar disorder",
    ),
    "psychosis spectrum": ("psychosis", "schizophrenia", "bipolar disorder"),
    "correlation topology": ("functional connectivity", "brain network topology"),
    "partial correlation topology": ("functional connectivity", "brain network topology"),
    "roi dynamics": (
        "functional connectivity",
        "regional homogeneity",
        "amplitude of low frequency fluctuation",
    ),
    "age adjusted total cognition": (
        "cognitive performance",
        "general cognition",
        "cognition",
    ),
    "fc edge projection": (
        "functional connectivity",
        "resting state functional connectivity",
        "connectome wide functional connectivity",
    ),
    "node connectivity profile": (
        "functional connectivity",
        "regional connectivity",
        "nodal functional connectivity",
        "functional connectivity strength",
    ),
    "graph topology summary": (
        "brain network topology",
        "graph theory metrics",
        "network topology",
        "global efficiency",
        "small worldness",
    ),
    "resting state fmri": (
        "resting state fmri",
        "resting state functional connectivity",
        "functional magnetic resonance imaging",
    ),
    "mci to dementia": (
        "mild cognitive impairment",
        "conversion to dementia",
        "dementia",
        "cognitive decline",
        "disease progression",
    ),
    "clinical plus structural mri": (
        "structural mri",
        "hippocampal volume",
        "cortical thickness",
        "brain volume",
    ),
    "structural mri only": (
        "structural mri",
        "hippocampal volume",
        "cortical thickness",
        "brain volume",
    ),
    "clinical plus genetic risk": (
        "polygenic risk score",
        "alzheimer disease polygenic risk score",
        "apoe e4",
    ),
    "clinical plus structural and genetic": (
        "structural mri",
        "hippocampal volume",
        "cortical thickness",
        "polygenic risk score",
        "apoe e4",
    ),
    "adni structural mri": ("structural mri",),
    "entorhinal icv": ("entorhinal cortex", "entorhinal volume"),
    "fusiform icv": ("fusiform gyrus", "fusiform volume"),
    "hippocampus icv": ("hippocampus", "hippocampal volume"),
    "midtemp icv": ("middle temporal gyrus", "middle temporal volume"),
    "ventricles icv": ("ventricular volume", "lateral ventricular volume"),
    "wholebrain icv": ("whole brain volume", "brain volume"),
    "ad prs p1em03 avg": (
        "alzheimer disease polygenic risk score",
        "polygenic risk score",
    ),
    "ad prs p5em08 avg": (
        "alzheimer disease polygenic risk score",
        "polygenic risk score",
    ),
    "ad prs p1em05 avg": (
        "alzheimer disease polygenic risk score",
        "polygenic risk score",
    ),
    "ad prs p5em02 avg": (
        "alzheimer disease polygenic risk score",
        "polygenic risk score",
    ),
    "structural mri": ("structural mri", "regional brain volume"),
    "polygenic risk score": (
        "alzheimer disease polygenic risk score",
        "polygenic risk score",
    ),
    "pathway polygenic risk score": (
        "pathway polygenic risk score",
        "polygenic risk score",
        "gene pathway",
    ),
    "cox adjusted": ("cox proportional hazards model", "survival analysis"),
    "apoe e4 dosage": ("apoe", "apoe e4"),
    "pathway prs curated ad risk gwas p5em02": (
        "alzheimer disease polygenic risk score",
        "genome wide association study",
    ),
    "pathway prs curated cholinergic p5em02": ("cholinergic",),
    "pathway prs curated gaba glutamate p5em02": ("gaba glutamate", "gaba", "glutamate"),
    "pathway prs curated mendelian ad p5em02": ("alzheimer disease",),
    "pathway prs curated microglia immune p5em02": ("microglia immune", "microglia"),
    "pathway prs curated myelin p5em02": ("myelin",),
    "pathway prs curated synaptic p5em02": ("synaptic",),
}


# These columns are deterministic descriptions of an anatomy factor, not new
# experimental degrees of freedom. They improve KG grounding for numeric ROI
# labels while preserving the registered candidate identity and feedback atoms.
SEMANTIC_FIELD_AUXILIARIES: dict[str, tuple[str, ...]] = {
    "anatomy": ("network", "anatomy_full", "roi_name", "structure_class"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_term(value: object) -> str:
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_candidate_id(task: str, fields: Mapping[str, object]) -> str:
    """Return a content-addressed ID independent of row order or outcomes."""

    payload = {
        "task": str(task),
        "fields": {
            str(key): str(value if value is not None else "")
            for key, value in sorted(fields.items())
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"{task}:{digest}"


def attach_candidate_ids(
    frame: pd.DataFrame,
    *,
    task: str,
    identity_fields: Sequence[str],
) -> pd.DataFrame:
    missing = sorted(set(identity_fields) - set(frame.columns))
    if missing:
        raise ValueError(f"candidate identity fields are missing: {missing}")
    out = frame.copy()
    out["candidate_id"] = [
        stable_candidate_id(task, {field: row[field] for field in identity_fields})
        for _, row in out.iterrows()
    ]
    if out["candidate_id"].duplicated().any():
        duplicated = out.loc[out["candidate_id"].duplicated(), identity_fields]
        raise ValueError(
            "candidate identity is not unique; first duplicate: "
            f"{duplicated.iloc[0].to_dict()}"
        )
    return out


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=float)
    low = float(np.nanmin(values[finite]))
    high = float(np.nanmax(values[finite]))
    out = values.copy()
    out[~finite] = low
    if math.isclose(low, high):
        return np.zeros(len(values), dtype=float)
    return (out - low) / (high - low)


def _terms_for_value(value: object) -> tuple[str, ...]:
    normalized = normalize_term(value)
    if not normalized:
        return ()
    terms = {normalized}
    terms.update(SEMANTIC_VALUE_ALIASES.get(normalized, ()))
    # Atlas labels often retain a useful suffix after the atlas prefix.
    tokens = normalized.split()
    if len(tokens) >= 3 and tokens[0] in {
        "aal",
        "basc",
        "cc200",
        "cc400",
        "destrieux",
        "dk",
        "dosenbach",
        "glasser",
        "harvard",
        "msdl",
        "power",
        "schaefer",
        "talairach",
    }:
        terms.add(" ".join(tokens[1:]))
    return tuple(sorted(terms))


def _ids_for_terms(
    terms: Iterable[str],
    *,
    name_to_ids: Mapping[str, Sequence[str]],
    degrees: Mapping[str, int],
    limit: int = 16,
) -> tuple[str, ...]:
    concept_ids: set[str] = set()
    for term in terms:
        concept_ids.update(str(item) for item in name_to_ids.get(term, ()))
    return tuple(
        sorted(concept_ids, key=lambda item: degrees.get(item, 0), reverse=True)[
            :limit
        ]
    )


def _relation_field_structure(
    case_study_id: str,
    semantic_fields: Sequence[str],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    tuple[str, ...],
]:
    """Return scientific atom groups and relation pairs, excluding qualifiers."""

    field_set = set(semantic_fields)
    try:
        # Auxiliary semantic columns describe registered atoms but are not
        # independent experiment factors, so they must not trigger the adapter's
        # factor-field validation or the generic all-pairs fallback.
        adapter = adapter_for(case_study_id)
    except KeyError:
        groups = tuple((field,) for field in semantic_fields)
        pairs = tuple(
            ((left,), (right,))
            for index, left in enumerate(semantic_fields)
            for right in semantic_fields[index + 1 :]
        )
        return groups, pairs, ()

    def expanded_fields(fields: Sequence[str]) -> tuple[str, ...]:
        expanded: list[str] = []
        for field in fields:
            if field in field_set and field not in expanded:
                expanded.append(field)
            for auxiliary in SEMANTIC_FIELD_AUXILIARIES.get(field, ()):
                if auxiliary in field_set and auxiliary not in expanded:
                    expanded.append(auxiliary)
        return tuple(expanded)

    groups: list[tuple[str, ...]] = []
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for relation in adapter.relations:
        left = expanded_fields(relation.subject_fields)
        right = expanded_fields(relation.object_fields)
        if left and left not in groups:
            groups.append(left)
        if right and right not in groups:
            groups.append(right)
        if left and right:
            pairs.append((left, right))
    if not groups:
        groups = [(field,) for field in semantic_fields]
    return tuple(groups), tuple(pairs), tuple(adapter.qualifier_fields)


def _pair_support(
    left: Sequence[str],
    right: Sequence[str],
    *,
    adjacency: Mapping[str, set[str]],
    directed: Mapping[tuple[str, str], float],
) -> float:
    best = 0.0
    for left_id in left:
        left_neighbors = adjacency.get(left_id, set())
        for right_id in right:
            right_neighbors = adjacency.get(right_id, set())
            direct = max(
                float(directed.get((left_id, right_id), 0.0)),
                0.85 * float(directed.get((right_id, left_id), 0.0)),
            )
            shared = len(left_neighbors & right_neighbors)
            best = max(best, direct + min(1.0, math.log1p(shared) / 4.0))
    return best


def score_public_candidates(
    frame: pd.DataFrame,
    *,
    semantic_fields: Sequence[str],
    case_study_id: str,
    kg_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach an outcome-blind NeuroDiscovery score derived only from the KG."""

    requested_semantic_fields = tuple(dict.fromkeys(semantic_fields))
    missing = sorted(set(requested_semantic_fields) - set(frame.columns))
    if missing:
        raise ValueError(f"semantic score fields are missing: {missing}")
    semantic_fields = tuple(
        dict.fromkeys(
            (
                *requested_semantic_fields,
                *(
                    auxiliary
                    for field in requested_semantic_fields
                    for auxiliary in SEMANTIC_FIELD_AUXILIARIES.get(field, ())
                    if auxiliary in frame.columns
                ),
            )
        )
    )
    out = frame.copy()
    unique_terms = sorted(
        {
            term
            for field in semantic_fields
            for value in out[field].fillna("").astype(str).unique()
            for term in _terms_for_value(value)
        }
    )
    payload = load_case1_kg_index_payload(
        kg_path,
        unique_terms,
        case_study_id=case_study_id,
    )
    name_to_ids = payload["name_to_ids"]
    degrees = payload["degrees"]
    scoped_degrees = payload["scoped_degrees"]
    atom_groups, relation_pairs, qualifier_fields = _relation_field_structure(
        case_study_id, semantic_fields
    )

    value_terms: dict[tuple[str, str], tuple[str, ...]] = {}
    value_ids: dict[tuple[str, str], tuple[str, ...]] = {}
    global_degree: dict[tuple[str, str], float] = {}
    scoped_degree: dict[tuple[str, str], float] = {}
    global_degree_max: dict[tuple[str, str], float] = {}
    scoped_degree_max: dict[tuple[str, str], float] = {}
    for field in semantic_fields:
        for raw in out[field].fillna("").astype(str).unique():
            key = (field, raw)
            terms = _terms_for_value(raw)
            ids = _ids_for_terms(
                terms,
                name_to_ids=name_to_ids,
                degrees=degrees,
            )
            value_terms[key] = terms
            value_ids[key] = ids
            global_degree[key] = sum(
                float(degrees.get(concept_id, 0)) for concept_id in ids
            )
            scoped_degree[key] = sum(
                float(scoped_degrees.get(concept_id, 0)) for concept_id in ids
            )
            global_degree_max[key] = max(
                (float(degrees.get(concept_id, 0)) for concept_id in ids),
                default=0.0,
            )
            scoped_degree_max[key] = max(
                (float(scoped_degrees.get(concept_id, 0)) for concept_id in ids),
                default=0.0,
            )

    legacy_node_global: list[float] = []
    legacy_node_scoped: list[float] = []
    legacy_pair_global: list[float] = []
    legacy_pair_scoped: list[float] = []
    relation_node_global: list[float] = []
    relation_node_scoped: list[float] = []
    relation_pair_global: list[float] = []
    relation_pair_scoped: list[float] = []
    for _, row in out.iterrows():
        keys = {
            field: (
                field,
                "" if pd.isna(row[field]) else str(row[field]),
            )
            for field in semantic_fields
        }
        ordered_keys = [keys[field] for field in semantic_fields]
        legacy_node_global.append(
            float(
                np.mean(
                    [math.log1p(global_degree_max[key]) for key in ordered_keys]
                )
            )
        )
        legacy_node_scoped.append(
            float(
                np.mean(
                    [math.log1p(scoped_degree_max[key]) for key in ordered_keys]
                )
            )
        )
        all_global_pairs: list[float] = []
        all_scoped_pairs: list[float] = []
        for left_index, left_key in enumerate(ordered_keys):
            for right_key in ordered_keys[left_index + 1 :]:
                left_ids = value_ids[left_key]
                right_ids = value_ids[right_key]
                if not left_ids or not right_ids:
                    continue
                all_global_pairs.append(
                    _pair_support(
                        left_ids,
                        right_ids,
                        adjacency=payload["adjacency"],
                        directed=payload["directed_support"],
                    )
                )
                all_scoped_pairs.append(
                    _pair_support(
                        left_ids,
                        right_ids,
                        adjacency=payload["scoped_adjacency"],
                        directed=payload["scoped_directed_support"],
                    )
                )
        legacy_pair_global.append(max(all_global_pairs, default=0.0))
        legacy_pair_scoped.append(max(all_scoped_pairs, default=0.0))

        def grouped_node_support(
            support: Mapping[tuple[str, str], float],
        ) -> float:
            values = []
            for group in atom_groups:
                matched = [
                    math.log1p(support[keys[field]])
                    for field in group
                    if support[keys[field]] > 0
                ]
                if matched:
                    values.append(max(matched))
            return float(np.mean(values)) if values else 0.0

        relation_node_global.append(grouped_node_support(global_degree))
        relation_node_scoped.append(grouped_node_support(scoped_degree))
        global_pairs: list[float] = []
        scoped_pairs: list[float] = []
        active_pairs = relation_pairs
        if not active_pairs:
            active_pairs = tuple(
                ((left,), (right,))
                for index, left in enumerate(semantic_fields)
                for right in semantic_fields[index + 1 :]
            )
        for left_fields, right_fields in active_pairs:
            left_ids = tuple(
                dict.fromkeys(
                    concept_id
                    for field in left_fields
                    for concept_id in value_ids[keys[field]]
                )
            )
            right_ids = tuple(
                dict.fromkeys(
                    concept_id
                    for field in right_fields
                    for concept_id in value_ids[keys[field]]
                )
            )
            if not left_ids or not right_ids:
                continue
            global_pairs.append(
                _pair_support(
                    left_ids,
                    right_ids,
                    adjacency=payload["adjacency"],
                    directed=payload["directed_support"],
                )
            )
            scoped_pairs.append(
                _pair_support(
                    left_ids,
                    right_ids,
                    adjacency=payload["scoped_adjacency"],
                    directed=payload["scoped_directed_support"],
                )
            )
        relation_pair_global.append(max(global_pairs, default=0.0))
        relation_pair_scoped.append(max(scoped_pairs, default=0.0))

    out["kg_global_node_support"] = _minmax(np.asarray(legacy_node_global))
    out["kg_scoped_node_support"] = _minmax(np.asarray(legacy_node_scoped))
    out["kg_global_pair_support"] = _minmax(np.asarray(legacy_pair_global))
    out["kg_scoped_pair_support"] = _minmax(np.asarray(legacy_pair_scoped))
    out["kg_relation_global_node_support"] = _minmax(
        np.asarray(relation_node_global)
    )
    out["kg_relation_scoped_node_support"] = _minmax(
        np.asarray(relation_node_scoped)
    )
    out["kg_relation_global_pair_support"] = _minmax(
        np.asarray(relation_pair_global)
    )
    out["kg_relation_scoped_pair_support"] = _minmax(
        np.asarray(relation_pair_scoped)
    )
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(out))
    out["score_neurodiscovery"] = (
        0.20 * out["kg_global_node_support"]
        + 0.20 * out["kg_global_pair_support"]
        + 0.20 * out["kg_scoped_node_support"]
        + 0.40 * out["kg_scoped_pair_support"]
        + 1e-9 * tie_break
    )
    field_value_coverage: dict[str, dict[str, float | int]] = {}
    for field in semantic_fields:
        raw_values = out[field].fillna("").astype(str).unique().tolist()
        matched_values = sum(bool(value_ids[(field, raw)]) for raw in raw_values)
        field_value_coverage[field] = {
            "values": len(raw_values),
            "matched_values": matched_values,
            "coverage": float(matched_values / len(raw_values)) if raw_values else 0.0,
        }
    matched_semantic_fields = sum(
        int(details["matched_values"] > 0) for details in field_value_coverage.values()
    )
    audit = {
        "case_study_id": case_study_id,
        "kg_path": str(kg_path.resolve()),
        "kg_sha256": sha256_file(kg_path.resolve()),
        "requested_semantic_fields": list(requested_semantic_fields),
        "semantic_fields": list(semantic_fields),
        "auxiliary_semantic_fields": [
            field for field in semantic_fields if field not in requested_semantic_fields
        ],
        "scientific_atom_groups": [list(group) for group in atom_groups],
        "relation_field_pairs": [
            {"subject_fields": list(left), "object_fields": list(right)}
            for left, right in relation_pairs
        ],
        "qualifier_fields_excluded_from_node_score": [
            field for field in qualifier_fields if field in semantic_fields
        ],
        "score_semantics_version": "legacy-plus-relation-aware.v2",
        "query_term_count": len(unique_terms),
        "matched_concepts": int(payload["stats"].get("matched_concepts", 0)),
        "matched_query_terms": len(name_to_ids),
        "query_term_coverage": (
            float(len(name_to_ids) / len(unique_terms)) if unique_terms else 0.0
        ),
        "field_value_coverage": field_value_coverage,
        "matched_semantic_fields": matched_semantic_fields,
        "semantic_field_coverage": (
            float(matched_semantic_fields / len(semantic_fields))
            if semantic_fields
            else 0.0
        ),
        "global_semantic_edges": int(payload["stats"].get("semantic_edges_seen", 0)),
        "scoped_semantic_edges": int(
            payload["stats"].get("case_study_semantic_edges_seen", 0)
        ),
        "uses_experimental_outcomes": False,
    }
    return out, audit


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "bytes": path.stat().st_size,
    }


def export_table_bundle(
    *,
    task: str,
    public: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame | None,
    factor_fields: Sequence[str],
    output_dir: Path,
    provenance: Mapping[str, Any],
    kg_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and persist a leak-resistant three-table task bundle."""

    public = validate_public_registry(public, factor_fields=factor_fields)
    internal = validate_hidden_outcomes(public, internal)
    if external is not None:
        external = validate_hidden_outcomes(public, external, external=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "public_candidates": _write_csv(
            public, output_dir / "public_candidates.csv"
        ),
        "internal_outcomes": _write_csv(
            internal, output_dir / "internal_outcomes.csv"
        ),
    }
    if external is not None:
        files["external_outcomes"] = _write_csv(
            external, output_dir / "external_outcomes.csv"
        )

    manifest = {
        "schema_version": TABLE_BUNDLE_SCHEMA,
        "created_at": utc_now(),
        "task": task,
        "status": "complete",
        "public_schema": PUBLIC_REGISTRY_SCHEMA,
        "outcome_schema": OUTCOME_SCHEMA,
        "candidate_count": int(len(public)),
        "internal_validated": int(internal["validated"].astype(bool).sum()),
        "external_executable": (
            int(external["executable"].astype(bool).sum())
            if external is not None
            else None
        ),
        "external_validated": (
            int(external["validated"].astype(bool).sum())
            if external is not None
            else None
        ),
        "factor_fields": list(factor_fields),
        "outcome_isolation": {
            "public_contains_validation_columns": False,
            "external_read_required_during_generation": False,
            "public_internal_external_are_distinct_files": True,
        },
        "kg_scoring": dict(kg_audit),
        "provenance": dict(provenance),
        "files": files,
    }
    manifest_path = output_dir / "table_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = sha256_file(manifest_path.resolve())
    return manifest


__all__ = [
    "TABLE_BUNDLE_SCHEMA",
    "attach_candidate_ids",
    "export_table_bundle",
    "normalize_term",
    "score_public_candidates",
    "stable_candidate_id",
]
