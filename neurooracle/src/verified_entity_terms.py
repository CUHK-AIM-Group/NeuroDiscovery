"""Exact whole-endpoint identity from live, independently checked UMLS terms.

A term may originate inside a compound mention. It can resolve another claim
only when that claim's COMPLETE endpoint surface equals the term; never strip
its modifiers or replace a compound by one component. No model is involved.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import re

from .claim_semantics import declared_type_atoms, looks_like_imaging_measurement
from .kg_bulk_identity import ROLE_TUIS
from .kg_identity_pilot import digest
from .relation_evidence import name_key
from .umls_audit_store import compact_record
from .umls_existing_alignment_review import source_trace, semantic_types
from .umls_mention_mapping import normalize_term

VERSION = "kg.verified_entity_terms.v1"
GENERIC = {"disease","disorder","syndrome","drug","medication","treatment","brain","age","control","risk","response"}


def atomic_term_proof(atom, mapping_rows):
    md=atom.get("metadata") or {}
    if md.get("mapping_status")!="auto_accepted_exact" or md.get("mapping_count")!=1 or len(mapping_rows)!=1:
        return None,"not_unique_accepted"
    record_id,edge=mapping_rows[0]; em=edge.get("metadata") or {}; variant=em.get("lookup_variant") or {}
    text=atom.get("preferred_name") or ""
    parent=dict(id=md.get("source_mention_id"),preferred_name=md.get("source_text"))
    if md.get("atomization_rule") not in {"full_mention","top_level_composite_split"} or not source_trace(atom,parent)["valid"]:
        return None,"bad_source_span"
    if not (em.get("review_status")=="auto_accepted_exact" and em.get("semantic_compatibility")=="compatible"
            and em.get("candidate_overflow") is False and em.get("ambiguous_best_cui_count")==1
            and variant.get("rule")=="normalized_exact" and variant.get("source_field")=="preferred_name"
            and variant.get("normalized")==normalize_term(text)==normalize_term(em.get("matched_term",""))
            and edge.get("source_id")==atom["id"] and str(edge.get("target_id","")).startswith("CUI:")):
        return None,"not_unmodified_exact"
    letters=[c for c in text if c.isalpha()]
    if name_key(text).casefold() in GENERIC or len(letters)<4 or (len(letters)<=10 and sum(c.isupper() for c in letters)>=2 and len(text.split())==1):
        return None,"generic_or_short_symbol"
    return dict(name=name_key(text),target_id=edge["target_id"],atom_id=atom["id"],parent_id=parent["id"],
        source_text=md["source_text"],atom_sha256=digest(compact_record(atom,"atoms",atom["id"])),
        mapping_ref="mappings/"+record_id,mapping_sha256=digest(compact_record(edge,"mappings",record_id)),
        atom_tuis=sorted(semantic_types(atom))),None


def load_candidates(connection):
    candidates,targets,counts=defaultdict(list),defaultdict(set),Counter()
    for payload, in connection.execute("SELECT payload_json FROM atoms WHERE in_core=1 ORDER BY ordinal"):
        atom=json.loads(payload)
        mappings=[(rid,json.loads(value)) for rid,value in connection.execute(
            "SELECT record_id,payload_json FROM mappings WHERE source_id=? ORDER BY ordinal",(atom["id"],))]
        key=name_key(atom["preferred_name"])
        for _,mapping in mappings:
            variant=(mapping.get("metadata") or {}).get("lookup_variant") or {}
            if variant.get("rule")=="normalized_exact" and variant.get("normalized")==normalize_term(key):
                targets[key].add(mapping["target_id"])
        proof,reason=atomic_term_proof(atom,mappings)
        counts[reason or "accepted_atomic_proof"]+=1
        if proof: candidates[key].append(proof)
    for key in list(candidates):
        if len(targets[key])!=1:
            counts["globally_ambiguous_surface"]+=1; del candidates[key]
    counts["unique_candidate_surfaces"]=len(candidates)
    return dict(candidates),dict(counts)


def target_contract(target, name):
    tuis=semantic_types(target)
    if not tuis or tuis & {"T028","T085","T086","T087","T114"}:
        return None
    roles={role for role,allowed in ROLE_TUIS.items() if tuis & allowed}
    anatomy=bool(tuis & {"T023","T024","T029","T030"})
    if "imaging_marker" in roles and not (anatomy or looks_like_imaging_measurement(name)):
        roles.remove("imaging_marker")
    if not roles: return None
    # Missing extraction types do not require inventing a type annotation.
    # Identity can still be established for a named drug, disease or anatomical
    # structure with its own concrete ontology type; broad findings cannot.
    missing_safe=(roles=={"drug"} or roles=={"disease"} or roles=={"imaging_marker"} and anatomy
                  or bool(tuis & {"T081","T201"}) and roles<={"outcome","individual_data"})
    if re.search(r"\b(?:risk|susceptibility|response|score|level|change)s?\b",name,re.I): missing_safe=False
    return dict(target_roles=sorted(roles),missing_type_safe=missing_safe,
                canonical_name=name_key(target["preferred_name"]),target_sha256=digest(target))


def approve_live_terms(candidates,nodes,live_mapping_refs):
    approved,held={},Counter()
    for name,proofs in candidates.items():
        for proof in proofs:
            target=nodes.get(proof["target_id"]); atom=nodes.get(proof["atom_id"]); parent=nodes.get(proof["parent_id"])
            if not target or not atom or not parent or proof["mapping_ref"] not in live_mapping_refs:
                held["missing_live_proof"]+=1; continue
            if digest(atom)!=proof["atom_sha256"] or live_mapping_refs[proof["mapping_ref"]]!=proof["mapping_sha256"]:
                raise ValueError("live UMLS atom/mapping differs from detail store")
            if parent["preferred_name"]!=proof["source_text"]:
                held["parent_source_text_changed"]+=1; continue
            if not set(proof["atom_tuis"]) & semantic_types(target):
                held["TUI_disagreement"]+=1; continue
            contract=target_contract(target,name)
            if not contract:
                held["unknown_or_molecular_target"]+=1; continue
            # Exact accepted MRCONSO term membership is the label proof; a
            # canonical node's bounded alias list need not contain every term.
            approved[name]={**proof,**contract,"parent_sha256":digest(parent)}
            break
        if name not in approved: held["unapproved_surface"]+=1
    return approved,dict(held)


def endpoint_eligibility(claim,side,term):
    if name_key(claim.get(side+"_name"))!=term["name"]: return "not_whole_endpoint"
    inner=claim.get("metadata") or {}
    outer=claim.get(side+"_type"); nested=inner.get(side+"_type")
    declared_values=[v for v in (outer,nested) if v not in (None,"")]
    if not declared_values:
        if not term["missing_type_safe"]: return "missing_type_for_contextual_concept"
    for declared in declared_values:
        roles={a.value for a in declared_type_atoms(declared,"")}
        if str(declared).strip().upper() in {"FINDING","CLINICAL_FINDING"}:
            roles={"outcome"}  # observation role, not an assertion of disease/drug identity
        target_roles=set(term["target_roles"])
        # An explicitly named disease may be the clinical outcome of a study.
        # This changes only identity routing, not that study-specific role.
        disease_as_outcome=bool(roles) and roles<={"disease","outcome"} and "disease" in target_roles
        if not roles or not (roles<=target_roles or disease_as_outcome): return "declared_type_conflict_or_unknown"
    field=side+"_id"
    if field in inner and inner[field]!=claim.get(field): return "conflicting_nested_endpoint"
    return None


class VerifiedEntityTerms:
    def __init__(self,payload):
        if payload.get("version")!=VERSION: raise ValueError("unsupported verified identity registry")
        self.entries={}
        for term in payload["terms"]:
            if term["name"] in self.entries: raise ValueError("duplicate verified term")
            if not term["target_id"].startswith("CUI:") or not term["canonical_name"] or not term["target_roles"]:
                raise ValueError("incomplete verified term")
            self.entries[term["name"]]=deepcopy(term)
        self._node_ids=frozenset(term[f] for term in self.entries.values() for f in ("target_id","atom_id","parent_id"))
        self._atom_ids=frozenset(term["atom_id"] for term in self.entries.values())

    def term_for(self,claim,side):
        term=self.entries.get(name_key(claim.get(side+"_name")))
        return term if term and endpoint_eligibility(claim,side,term) is None else None

    def relation_key(self,claim):
        from .relation_evidence import relation_key
        key=list(relation_key(claim))
        for side,offset in (("subject",0),("object",3)):
            term=self.term_for(claim,side)
            if term and term.get("canonicalize",True) and claim.get(side+"_id")==term["target_id"]:
                key[offset+1]=term["canonical_name"]
        return tuple(key)

    def validate_nodes(self,nodes):
        expected={}
        for term in self.entries.values():
            for field,h in (("target_id","target_sha256"),("atom_id","atom_sha256"),("parent_id","parent_sha256")):
                if term[field] in expected and expected[term[field]]!=term[h]: raise ValueError("conflicting node seal")
                expected[term[field]]=term[h]
        if any(nid not in nodes or digest(nodes[nid])!=sig for nid,sig in expected.items()):
            raise ValueError("identity registry no longer matches graph nodes")

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def atom_ids(self):
        return self._atom_ids

    def validate_mappings(self,edges):
        expected={term["mapping_ref"]:term["mapping_sha256"] for term in self.entries.values()}
        seen={}
        for edge in edges:
            ref=(edge.get("metadata") or {}).get("audit_ref")
            if ref in expected:
                sig=digest(edge)
                if sig!=expected[ref] or ref in seen and seen[ref]!=sig:
                    raise ValueError("identity mapping edge no longer matches registry")
                seen[ref]=sig
        if seen!=expected: raise ValueError("identity mapping proof missing")

    def bind_runtime(self,kg):
        self._runtime_nodes={nid:digest(kg.get_concept(nid).to_dict()) for nid in self.node_ids}
        refs={term["mapping_ref"] for term in self.entries.values()}
        self._runtime_mappings={(e.get("metadata") or {}).get("audit_ref"):digest(e) for e in kg.iter_edge_records()
            if (e.get("metadata") or {}).get("audit_ref") in refs}

    def export_payload(self,kg,nodes,edges):
        if not hasattr(self,"_runtime_nodes"):
            raise ValueError("identity registry was not bound to this runtime graph")
        if any(kg.get_concept(nid) is None or digest(kg.get_concept(nid).to_dict())!=sig for nid,sig in self._runtime_nodes.items()):
            raise ValueError("identity proof nodes changed outside supported graph API")
        refs=set(self._runtime_mappings)
        current={(e.get("metadata") or {}).get("audit_ref"):digest(e) for e in kg.iter_edge_records()
                 if (e.get("metadata") or {}).get("audit_ref") in refs}
        if current!=self._runtime_mappings: raise ValueError("identity mapping proof changed")
        serialized={(e.get("metadata") or {}).get("audit_ref"):digest(e) for e in edges
                    if (e.get("metadata") or {}).get("audit_ref") in refs}
        terms=deepcopy(list(self.entries.values()))
        for term in terms:
            for field,h in (("target_id","target_sha256"),("atom_id","atom_sha256"),("parent_id","parent_sha256")):
                term[h]=digest(nodes[term[field]])
            term["mapping_sha256"]=serialized[term["mapping_ref"]]
        return dict(version=VERSION,terms=terms,scope="exact complete endpoint identities; original scientific roles and evidence preserved")


def load_graph_terms(graph_path,data):
    import hashlib
    from pathlib import Path
    declaration=(data.get("metadata") or {}).get("entity_identity")
    if not declaration: return None
    if declaration.get("version")!=VERSION: raise ValueError("unsupported graph entity identity version")
    base=Path(graph_path).resolve().parent
    path=(base/declaration["registry"]).resolve()
    if not path.is_relative_to(base.parent) or path.stat().st_size>64*1024**2:
        raise ValueError("identity registry outside local graph bundle or too large")
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=declaration["sha256"]: raise ValueError("identity registry SHA mismatch")
    registry=VerifiedEntityTerms(json.loads(payload))
    registry.validate_nodes(data["concepts"]); registry.validate_mappings(data["edges"])
    return registry
