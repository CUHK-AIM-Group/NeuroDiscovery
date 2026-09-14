"""Conservative repair of measurement mentions incorrectly linked to genes.

A literal node represents the complete original endpoint, not an asserted UMLS
equivalence. Case, laterality, conjunctions and qualifiers are identity-bearing.
No source, predicate, scientific context or role is inferred or rewritten.
"""
from copy import deepcopy
import hashlib
import re

from .claim_semantics import (concept_atom_roles, declared_type_atoms,
    looks_like_concrete_imaging_measurement, looks_like_method_or_procedure_entity,
    looks_like_non_imaging_assay_entity)
from .kg_bulk_identity import change_claim
from .kg_identity_pilot import digest, nonidentity_claim
from .relation_evidence import name_key
from .schema import ConceptNode

VERSION = "kg.literal_endpoint_repair.v1"
GENES = {"CUI:C1414531": "FANCE", "CUI:C1421437": "VCP"}
MOLECULAR = re.compile(r"\b(?:genes?|genetic|genomic|genotypes?|proteins?|alleles?|snps?|"
    r"variants?|polygenic|expression|methylation|plasma|serum|microbiota|enzymes?|"
    r"receptors?|glutamate|gaba|dopamine|serotonin|fance|vcp|p97)\b", re.I)


def endpoint_gate(md, side):
    field = side + "_id"
    if md.get(field) not in GENES:
        return "not_selected_gene"
    inner = md.get("metadata") or {}
    if field in inner and inner[field] != md[field]:
        return "nested_id_conflict"
    declared = md.get(side + "_type") or inner.get(side + "_type")
    outer_type, inner_type = md.get(side + "_type"), inner.get(side + "_type")
    if outer_type and inner_type and declared_type_atoms(outer_type) != declared_type_atoms(inner_type):
        return "nested_type_conflict"
    if {a.value for a in declared_type_atoms(declared)} != {"imaging_marker"}:
        return "not_explicit_imaging_type"
    text = md.get(side + "_name")
    if not isinstance(text, str) or not name_key(text):
        return "missing_complete_name"
    if MOLECULAR.search(text):
        return "molecular_or_genetic_context"
    if looks_like_non_imaging_assay_entity(text) or looks_like_method_or_procedure_entity(text):
        return "assay_or_method_context"
    if not looks_like_concrete_imaging_measurement(text):
        return "not_unambiguous_measurement_surface"
    return None


def literal_node(name):
    # This is an ordinary existing KG concept class, not a new ontology class.
    # Full SHA avoids slug/case/truncation collisions; no per-claim or paper salt.
    name = name_key(name)
    key = hashlib.sha256((VERSION + "|imaging_marker|" + name).encode()).hexdigest()
    return ConceptNode(id="CLM_CONCEPT:imaging_mention_" + key,
        preferred_name=name, domain_tags=["imaging_feature"],
        source_vocab="claim_extraction").to_dict()


def reuse_gate(node, name, incidents):
    if not node["id"].startswith("CLM_CONCEPT:"):
        return "not_literal_concept"
    if name_key(node.get("preferred_name")) != name_key(name):
        return "not_same_complete_name"
    if {a.value for a in concept_atom_roles(node)} != {"imaging_marker"}:
        return "existing_type_not_confirmed"
    if node.get("semantic_types") or node.get("external_ids"):
        return "external_identity_requires_separate_proof"
    for incident in incidents:
        if name_key(incident["name"]) != name_key(name):
            return "existing_incident_scope_differs"
        if set(incident["declared_roles"]) not in (set(), {"imaging_marker"}):
            return "existing_incident_type_conflict"
    return None


def reviewed_claim(record, changes):
    event = dict(claim_id=record["id"], claim_sha256=digest(record), changes=changes)
    for change in changes:
        side = change["side"]
        reason = endpoint_gate(record["metadata"], side)
        if reason:
            raise ValueError(reason)
        if change["old_id"] != record["metadata"][side + "_id"]:
            raise ValueError("wrong source endpoint")
        if change["name"] != name_key(record["metadata"][side + "_name"]):
            raise ValueError("full name changed")
    out = change_claim(record, event)
    event["current_node_sha256"] = digest(out)
    event["nonidentity_sha256"] = digest(nonidentity_claim(record))
    return event, out


def edge_owner(row):
    return row["source_id"] if row["relation_type"] == "about" else (row.get("metadata") or {}).get("claim_id")


def reviewed_edges(cid, original, current, edge_rows):
    """Require complete agreeing owned references; do not guess a mixed closure."""
    before, after = original["metadata"], current["metadata"]
    if not edge_rows:
        raise ValueError("missing owned edges")
    seen_about, science, events = set(), 0, []
    for ordinal, row in edge_rows:
        if edge_owner(row) != cid:
            raise ValueError("edge owner differs")
        out = deepcopy(row)
        if row["relation_type"] == "about":
            if (row.get("metadata") or {}).get("claim_id") not in (None, "", cid):
                raise ValueError("conflicting about owner")
            sides = [s for s in ("subject", "object") if before[s + "_id"] == row["target_id"]]
            if len(sides) != 1:
                raise ValueError("ambiguous about endpoint")
            side = sides[0]
            if side in seen_about:
                raise ValueError("duplicate about reference")
            seen_about.add(side); out["target_id"] = after[side + "_id"]
        else:
            if (row["source_id"], row["target_id"], row["relation_type"]) != (
                    before["subject_id"], before["object_id"], before["predicate"]):
                raise ValueError("scientific edge disagrees with claim")
            science += 1
            out["source_id"], out["target_id"] = after["subject_id"], after["object_id"]
        if row != out:
            if out["source_id"] == out["target_id"]:
                raise ValueError("new self loop")
            events.append(dict(ordinal=ordinal, claim_id=cid, edge_sha256=digest(row),
                current_edge_sha256=digest(out),
                changes={f:dict(old=row[f], new=out[f]) for f in ("source_id", "target_id") if row[f] != out[f]}))
    # The current KG legitimately stores many claims with two about edges only.
    # Keep that representation: correct any existing scientific edge but never
    # manufacture one, nor require a duplicate materialized relation to exist.
    if seen_about != {"subject", "object"} or science not in (0, 1):
        raise ValueError("reference closure is not two about and zero/one science edges")
    return events


def apply_edge(row, event, *, reverse=False):
    expected = event["current_edge_sha256" if reverse else "edge_sha256"]
    if digest(row) != expected:
        raise ValueError("edge changed after review")
    out = deepcopy(row)
    for field, change in event["changes"].items():
        if field not in {"source_id", "target_id"} or row[field] != change["new" if reverse else "old"]:
            raise ValueError("edge endpoint change differs")
        out[field] = change["old" if reverse else "new"]
    if digest(out) != event["edge_sha256" if reverse else "current_edge_sha256"]:
        raise ValueError("non-approved edge fields changed")
    return out


def reverse_claim(row, event):
    if digest(row) != event["current_node_sha256"]:
        raise ValueError("claim output changed")
    original = deepcopy(row)
    for change in event["changes"]:
        field = change["side"] + "_id"
        if original["metadata"][field] != change["target_id"]:
            raise ValueError("wrong repaired endpoint")
        original["metadata"][field] = change["old_id"]
        inner = original["metadata"].get("metadata") or {}
        if field in inner:
            if inner[field] != change["target_id"]:
                raise ValueError("wrong repaired inner endpoint")
            inner[field] = change["old_id"]
    if digest(original) != event["claim_sha256"] or change_claim(original, event) != row:
        raise ValueError("original claim hash / identity-only proof differs")
    return original
