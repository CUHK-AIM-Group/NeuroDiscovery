"""Read the current, validated shared-evidence index without loading the KG.

Ordinary queries use native graph fingerprints and the trusted acceptance;
the small receipt/catalog are SHA-checked. A changed graph must be reindexed.
"""
import hashlib
import json
from pathlib import Path


def check_file(fingerprint, *, full_hash=False):
    path = Path(fingerprint["path"])
    stat = path.stat()
    if stat.st_size != fingerprint["bytes"] or stat.st_mtime_ns != fingerprint["mtime_ns"]:
        raise ValueError("stale KG relation index: file fingerprint changed")
    if full_hash:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != fingerprint["sha256"]:
            raise ValueError("KG relation index integrity check failed")
    return path


def current_shared_relations(campaign_path):
    """Yield validated shared groups; singleton claims remain in the KG."""
    campaign = json.loads(Path(campaign_path).read_text(encoding="utf-8"))
    if campaign.get("status") != "COMPLETED" or campaign.get("active_process") is not None:
        raise ValueError("current KG change is in progress; relation index is not ready")
    receipt_path = check_file(campaign["current_acceptance"], full_hash=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("graph") != campaign["current_graph"] or
        receipt.get("shared_relations") != campaign.get("current_shared_relations") or
        not receipt.get("checks", {}).get("shared_relation_index_complete")):
        raise ValueError("current KG lacks a matching validated shared relation index")
    check_file(campaign["current_graph"])
    identity_fp=campaign.get("current_entity_terms")
    if identity_fp:
        if receipt.get("entity_terms")!=identity_fp or not receipt.get("checks",{}).get("verified_identity_proofs_complete"):
            raise ValueError("unvalidated current entity identity registry")
        check_file(identity_fp,full_hash=True)
    paper_fp = campaign.get("current_paper_identities")
    if paper_fp:
        if receipt.get("paper_identities") != paper_fp or not receipt.get("checks", {}).get("paper_identity_witnesses_validated"):
            raise ValueError("unvalidated current paper identity registry")
        check_file(paper_fp, full_hash=True)
    catalog = check_file(campaign["current_shared_relations"], full_hash=True)
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
    # Fail closed if a writer advanced either the graph or control while reading.
    check_file(campaign["current_graph"])
    check_file(campaign["current_shared_relations"])
    if identity_fp: check_file(identity_fp)
    if paper_fp: check_file(paper_fp)
    if json.loads(Path(campaign_path).read_text(encoding="utf-8")) != campaign:
        raise ValueError("current KG changed during relation lookup")


def find_shared_relations(campaign_path, *, subject=None, predicate=None, object_name=None, minimum_papers=1, verified_papers_only=None, include_retracted=False):
    """Filter source counts; do not delete original members or change raw counts.

    With reviewed publication evidence, confirmed retracted source keys do not
    count toward minimum_papers by default. Corrected versions remain separate
    identities and are not blanket-excluded. counted_paper_count and excluded
    keys are query-only annotations, not new per-node/edge or index metadata.
    No retraction flag is NOT a guarantee of scientific validity.
    """
    if type(minimum_papers) is not int or minimum_papers < 1:
        raise ValueError("minimum_papers must be a positive integer")
    if verified_papers_only is not None and type(verified_papers_only) is not bool:
        raise ValueError("verified_papers_only must be boolean or None")
    if type(include_retracted) is not bool: raise ValueError("include_retracted must be boolean")
    # Consume the validated snapshot before using its optional alias dictionary.
    campaign=json.loads(Path(campaign_path).read_text(encoding="utf-8"))
    groups=list(current_shared_relations(campaign_path))
    from .correlation_grouping import enabled as symmetric_grouping_enabled
    current_receipt=json.loads(check_file(campaign['current_acceptance'],full_hash=True).read_text(encoding='utf-8'))
    symmetric=symmetric_grouping_enabled(current_receipt)
    if verified_papers_only is None:
        verified_papers_only = bool(campaign.get("current_paper_identities"))
    count_field = "verified_paper_count" if verified_papers_only else "paper_count"
    papers=None
    if campaign.get("current_paper_identities"):
        from .kg_paper_identity import VerifiedPaperIdentities
        papers=VerifiedPaperIdentities(json.loads(check_file(campaign["current_paper_identities"],full_hash=True).read_text(encoding="utf-8")))
        if papers.has_publication_reviews:
            receipt=json.loads(check_file(campaign["current_acceptance"],full_hash=True).read_text(encoding="utf-8"))
            if not receipt.get("checks",{}).get("publication_status_witnesses_validated"):
                raise ValueError("unvalidated publication status witnesses")
    aliases={}
    if campaign.get("current_entity_terms"):
        registry=json.loads(check_file(campaign["current_entity_terms"],full_hash=True).read_text(encoding="utf-8"))
        for term in registry["terms"]:
            if term.get("canonicalize",True):
                aliases.setdefault(term["name"].casefold(),set()).add((term["target_id"],term["canonical_name"]))
    # Case-insensitive *search* only. Distinct indexed identities are not merged.
    def equal(row,side,query):
        return query is None or str(row[side+"_name"]).casefold()==query.casefold() or (
            row.get(side+"_id"),row[side+"_name"]) in aliases.get(query.casefold(),set())
    result=[]
    for row in groups:
        forward=equal(row,"subject",subject) and equal(row,"object",object_name)
        reverse=symmetric and row['predicate']=='correlates_with' and equal(row,'object',subject) and equal(row,'subject',object_name)
        if not ((forward or reverse) and (predicate is None or row["predicate"]==predicate)): continue
        count=row.get(count_field,0)
        if papers and papers.has_publication_reviews:
            keys={m["paper_key"] for m in row["members"] if not verified_papers_only or m.get("paper_status")=="verified"}
            excluded={k for k in keys if papers.publication_review(k)["status"]=="retracted"} if not include_retracted else set()
            count=len(keys-excluded)
            row={**row,"counted_paper_count":count,"excluded_retracted_paper_keys":sorted(excluded)}
        if count>=minimum_papers: result.append(row)
    if json.loads(Path(campaign_path).read_text(encoding="utf-8"))!=campaign:
        raise ValueError("current KG changed during relation lookup")
    return result
