"""Whole-surface, previously accepted UMLS identity reuse, never fuzzy merging."""
from collections import Counter
from copy import deepcopy
import json

from .claim_semantics import concept_atom_roles, declared_type_atoms, looks_like_imaging_measurement
from .kg_identity_pilot import digest, nonidentity_claim
from .relation_evidence import name_key
from .umls_mention_mapping import normalize_term
from .umls_existing_alignment_review import source_trace, semantic_types

VERSION = "kg.whole_surface_identity.v1"

# Conservative TUI gates supplement (never replace) the accepted mapping.
# Accumulated source domains on a CUI can be polluted: a Finding is not thereby
# a disease, and a Diagnostic Procedure is not a quantitative imaging marker.
ROLE_TUIS = {
    "disease": {"T019", "T020", "T037", "T047", "T048", "T191"},
    "drug": {"T121", "T200"},
    "outcome": {"T032", "T033", "T034", "T040", "T041", "T046", "T080", "T081", "T184", "T201"},
    "individual_data": {"T032", "T033", "T034", "T041", "T055", "T080", "T081", "T098", "T100", "T201"},
    "imaging_marker": {"T023", "T024", "T029", "T030", "T033", "T034", "T040", "T042", "T081", "T201"},
}


def mapping_proof(atom, mappings):
    md = atom.get("metadata") or {}
    if md.get("mapping_status") != "auto_accepted_exact" or md.get("mapping_count") != 1 or len(mappings) != 1:
        return None, "not_unique_accepted"
    edge = mappings[0]
    em = edge.get("metadata") or {}
    variant = em.get("lookup_variant") or {}
    span = md.get("evidence_span") or {}
    text = md.get("source_text")
    if not (md.get("atomization_rule") == "full_mention" and isinstance(text, str)
            and span.get("field") == "preferred_name" and span.get("start") == 0
            and span.get("end") == len(text) and span.get("text") == text == atom.get("preferred_name")):
        return None, "not_verbatim_whole_mention"
    if not (em.get("review_status") == "auto_accepted_exact" and em.get("semantic_compatibility") == "compatible"
            and em.get("candidate_overflow") is False and em.get("ambiguous_best_cui_count") == 1
            and variant.get("rule") == "normalized_exact" and variant.get("source_field") == "preferred_name"
            and normalize_term(text) == variant.get("normalized") == normalize_term(em.get("matched_term", ""))
            and edge.get("source_id") == atom["id"] and str(edge.get("target_id", "")).startswith("CUI:")):
        return None, "not_unmodified_compatible_exact"
    # Acronyms, gene/protein symbols and case-sensitive molecular identities
    # need species/expansion context beyond a normalized dictionary lookup.
    if len(text.split()) == 1 and (len(text) <= 3 or (len(text) <= 10 and sum(c.isupper() for c in text) >= 2)):
        return None, "short_or_acronym"
    return dict(parent_id=md["source_mention_id"], atom_id=atom["id"], target_id=edge["target_id"],
                name=text, atom_sha256=digest(atom), mapping_sha256=digest(edge)), None


def load_proofs(connection):
    result, counts = {}, Counter()
    for payload, in connection.execute("SELECT payload_json FROM atoms WHERE in_core=1 AND mapping_status='auto_accepted_exact' ORDER BY ordinal"):
        atom = json.loads(payload)
        mappings = [json.loads(row[0]) for row in connection.execute(
            "SELECT payload_json FROM mappings WHERE source_id=? ORDER BY ordinal", (atom["id"],))]
        proof, reason = mapping_proof(atom, mappings)
        counts[reason or "eligible_mapping"] += 1
        if proof:
            pid = proof["parent_id"]
            if pid in result:
                raise ValueError("multiple full-mention proofs for one parent")
            result[pid] = proof
    return result, dict(counts)


def validate_entities(proof, parent, target, atom):
    if not source_trace(atom, parent)["valid"] or not source_trace(atom, parent)["full_parent_surface"]:
        return "source_trace_mismatch"
    if target["id"] != proof["target_id"] or digest(atom) != proof["atom_sha256"]:
        return "identity_record_mismatch"
    if not semantic_types(atom) or not semantic_types(atom) & semantic_types(target):
        return "semantic_type_mismatch"
    recorded_roles = {a.value for a in concept_atom_roles(target)}
    if not recorded_roles or "gene_target" in recorded_roles:
        return "unknown_or_molecular_role"
    tuis = semantic_types(target)
    roles = {role for role in recorded_roles if tuis & ROLE_TUIS.get(role,set())}
    if "imaging_marker" in roles and not (looks_like_imaging_measurement(proof["name"]) or tuis & {"T023","T024","T029","T030"}):
        roles.remove("imaging_marker")
    if not roles:
        return "target_TUI_does_not_confirm_role"
    # The full surface must still be an explicit label on this actual target.
    if normalize_term(proof["name"]) not in {normalize_term(n) for n in [target["preferred_name"], *target.get("aliases", [])]}:
        return "target_label_mismatch"
    proof.update(target_roles=sorted(roles), parent_sha256=digest(parent), target_sha256=digest(target))
    return None


def endpoint_decision(claim, side, proof):
    inner = claim.get("metadata") or {}
    field = side + "_id"
    if name_key(claim.get(side + "_name")) != name_key(proof["name"]):
        return "claim_full_name_differs"
    declared = claim.get(side + "_type") or inner.get(side + "_type")
    if not declared:
        return "missing_declared_type"
    roles = {a.value for a in declared_type_atoms(declared, claim.get(side + "_name"))}
    if not roles or not roles <= set(proof["target_roles"]):
        return "claim_role_not_confirmed"
    if field in inner and inner[field] != claim[field]:
        return "conflicting_nested_endpoint"
    return None


def change_claim(record, event):
    if digest(record) != event["claim_sha256"]:
        raise ValueError("claim changed after full-graph review")
    result = deepcopy(record)
    for change in event["changes"]:
        field = change["side"] + "_id"
        if result["metadata"][field] != change["old_id"]:
            raise ValueError("stale endpoint")
        result["metadata"][field] = change["target_id"]
        inner = result["metadata"].get("metadata") or {}
        if field in inner:
            if inner[field] != change["old_id"]:
                raise ValueError("conflicting nested endpoint")
            inner[field] = change["target_id"]
    if result["metadata"]["subject_id"] == result["metadata"]["object_id"]:
        raise ValueError("identity correction creates self relation")
    if digest(nonidentity_claim(record)) != digest(nonidentity_claim(result)):
        raise ValueError("scientific claim changed")
    return result


def change_edge(record, events):
    about = record["relation_type"] == "about"
    owner = record["source_id"] if about else (record.get("metadata") or {}).get("claim_id")
    event = events.get(owner)
    if not event:
        return record
    result = deepcopy(record)
    for change in event["changes"]:
        field = "target_id" if about else "source_id" if change["side"] == "subject" else "target_id"
        if about and record[field] != change["old_id"]:
            continue
        if record[field] != change["old_id"]:
            raise ValueError("owned edge disagrees with claim")
        if about and (record.get("metadata") or {}).get("claim_id") not in (None, "", owner):
            raise ValueError("conflicting about owner")
        result[field] = change["target_id"]
    if result == record:
        return record
    if result["source_id"] == result["target_id"]:
        raise ValueError("new self loop")
    return result
