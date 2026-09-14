"""Model-independent shared relation evidence; grouping is not fact merging.

Full endpoint surface names guard against over-broad canonical IDs. Negation,
population, method and direction remain per-claim evidence. No synonyms, case
study scopes or scientific conclusions are inferred by this module.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import unicodedata

from .case_study_membership_contract import strongest_paper_key

VERSION = "kg.relation_evidence.v1"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def name_key(value):
    # Keep case: e.g. species-specific gene case must not be folded blindly.
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def relation_key(claim):
    fields = ("subject_id", "subject_name", "predicate", "object_id", "object_name")
    values = tuple(name_key(claim.get(key)) for key in fields)
    if not all(values):
        raise ValueError("relation requires both endpoint IDs, complete names and predicate")
    return values


def relation_id(key):
    return "REL:" + hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()


def evidence_context(claim):
    inner = claim.get("metadata") or {}
    def value(key):
        outer = claim.get(key)
        return outer if outer not in (None, "", [], {}) else inner.get(key)
    return {"negated": claim.get("negated"),
        **{key: value(key) for key in ("subject_type", "object_type", "conditions", "population")},
        # Preserve the complete evidence object; no averaging or best-paper pick.
        "evidence": deepcopy(claim.get("evidence")), "polarity": claim.get("polarity")}


def summarize_relation(key, claims, *, identities=None, papers=None):
    members = []
    seen = {}
    for claim in claims:
        cid = claim.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("claim identity is required")
        signature = canonical_json(claim)
        if cid in seen:
            if signature != seen[cid]:
                raise ValueError("conflicting records for the same claim ID")
            continue
        if (identities.relation_key(claim) if identities else relation_key(claim)) != tuple(key):
            raise ValueError("claim does not belong to this fine-grained relation")
        seen[cid] = signature
        members.append(evidence_member(claim, papers=papers))
    return summarize_members(key, members)


def evidence_member(claim, *, papers=None):
    result = {"claim_id": claim["id"], "paper_key": strongest_paper_key(claim), "negated": claim.get("negated"),
        "context_signature": hashlib.sha256(canonical_json(evidence_context(claim)).encode("utf-8")).hexdigest()}
    if papers is not None:
        identity = papers.resolve(claim)
        result.update(paper_key=identity["paper_key"], paper_status=identity["status"])
    return result


def summarize_members(key, evidence_members):
    """Compact index, not copied evidence. paper_count counts source keys.

    With authority witnesses, verified_paper_count counts confirmed different
    articles, not independent cohorts, agreement, study quality or consensus.
    Unverified sources remain explicit and never inflate that verified count.
    """
    contexts = defaultdict(list)
    members = []
    seen = set()
    for item in evidence_members:
        if item["claim_id"] in seen:
            raise ValueError("duplicate indexed claim identity")
        seen.add(item["claim_id"])
        contexts[item["context_signature"]].append(item["claim_id"])
        member = {k: item[k] for k in ("claim_id", "paper_key", "negated")}
        if "paper_status" in item:
            if item["paper_status"] not in {"verified", "unverified", "unresolved", "conflict"}:
                raise ValueError("invalid paper identity status")
            member["paper_status"] = item["paper_status"]
        members.append(member)
    members.sort(key=lambda item: item["claim_id"])
    result = {"id": relation_id(key), "subject_id": key[0], "subject_name": key[1], "predicate": key[2],
        "object_id": key[3], "object_name": key[4], "claim_count": len(members),
        "paper_count": len({item["paper_key"] for item in members}),
        "members": members, "evidence_variant_count": len(contexts),
        "evidence_variants": [{"signature": k, "claim_ids": sorted(v)} for k,v in sorted(contexts.items())],
        "interpretation": "shared_relation_evidence_not_consensus"}
    if any("paper_status" in item for item in members):
        if not all("paper_status" in item for item in members):
            raise ValueError("mixed legacy and authority paper identities")
        result.update(verified_paper_count=len({item["paper_key"] for item in members if item["paper_status"] == "verified"}),
            unverified_claim_count=sum(item["paper_status"] != "verified" for item in members))
    return result


def group_claim_evidence(claims, *, minimum_claims=1, identities=None, papers=None):
    if type(minimum_claims) is not int or minimum_claims < 1:
        raise ValueError("minimum_claims must be a positive integer")
    groups = defaultdict(list)
    seen = {}
    for claim in claims:
        cid = claim.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("claim identity is required")
        if cid in seen and canonical_json(seen[cid]) != canonical_json(claim):
            raise ValueError("conflicting records for the same claim ID")
        seen[cid] = claim
        groups[identities.relation_key(claim) if identities else relation_key(claim)].append(claim)
    for key in sorted(groups):
        if len(groups[key]) >= minimum_claims:
            result = summarize_relation(key, groups[key], identities=identities, papers=papers)
            if result["claim_count"] >= minimum_claims:
                yield result
