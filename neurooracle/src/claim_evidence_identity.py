"""Conservative evidence-level idempotency; a relation is not an observation."""
from copy import deepcopy
import hashlib

from .case_study_membership_contract import strongest_paper_key
from .kg_metadata_compaction import compact_record
from .relation_evidence import canonical_json
from .schema import Evidence

# Only known bookkeeping is excluded. Unknown scientific fields remain in the
# fingerprint, so adding an unrecognised observation cannot silently discard it.
BOOKKEEPING = frozenset({"confidence", "scope_reaudit", "paper_case_study_ids", "claim_case_study_ids",
    "case_study_membership_schema_version", "kg_injected", "staged_only"})


def evidence_payload(claim):
    payload = deepcopy(compact_record("node", {"id": "CLM:identity", "metadata": claim})["metadata"])
    payload.pop("id", None)
    payload.pop("claim_id", None)
    for holder in (payload, payload.get("metadata") or {}):
        for key in BOOKKEEPING:
            holder.pop(key, None)
    # Storage may omit declared empty defaults, but structured evidence extras
    # and typed values (False versus zero) must survive.
    payload["evidence"] = Evidence.from_dict(payload.get("evidence", {})).to_dict()
    if not payload.get("metadata"):
        payload.pop("metadata", None)
    return payload


def evidence_signature(claim):
    return hashlib.sha256(canonical_json(evidence_payload(claim)).encode("utf-8")).hexdigest()


def evidence_dedup_key(claim, *, papers=None):
    paper_key = strongest_paper_key(claim)
    if papers is not None:
        identity = papers.resolve(claim)
        if identity["status"] in {"conflict", "unresolved"}: return None
        if identity["status"] == "verified":
            paper_key = identity["paper_key"]
        else:
            # Incomplete authority coverage must not disable existing exact-ID
            # idempotency. Only VERIFIED witnesses bridge different ID systems.
            from .kg_paper_identity import identifiers
            ids = identifiers(claim.get("source_paper"))
            paper_key = next((kind + ":" + ids[kind] for kind in ("pmid", "doi", "pmcid") if ids[kind]), paper_key)
    # Titles/year alone and missing papers are not authoritative paper identity.
    if paper_key.startswith(("title_year:", "claim_fallback:")):
        return None
    payload = evidence_payload(claim)
    paper = payload.get("source_paper") or {}
    bibliography = {"pmid", "doi", "pmcid", "arxiv_id", "openalex_id", "title", "authors", "year", "publication_year", "journal"}
    payload["source_paper"] = {"identity": paper_key,
        "extra": {k: v for k,v in paper.items() if k not in bibliography}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
