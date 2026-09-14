"""Authority-linked publication status, separate from identity and claim truth."""
from copy import deepcopy
import re

VERSION = "kg.publication_review.v1"
RETRACTION_NOTICE_TYPES = frozenset({"Retraction Notice", "Retraction of Publication"})
LINKS = frozenset({"RetractionIn", "RetractionOf", "CorrectedandRepublishedIn", "CorrectedandRepublishedFrom",
    "RetractedandRepublishedIn", "RetractedandRepublishedFrom", "RepublishedIn", "RepublishedFrom",
    "ExpressionOfConcernIn", "ExpressionOfConcernFor"})


def owning_status_projection(article, witness):
    from .kg_paper_identity import normalize_pmid
    if article.tag != "PubmedArticle": raise ValueError("publication status requires owning PubmedArticle")
    p = article.find("./MedlineCitation/PMID")
    pmid = normalize_pmid(p.text if p is not None else None)
    if not pmid: raise ValueError("missing publication status PMID")
    links = []
    for el in article.findall("./MedlineCitation/CommentsCorrectionsList/CommentsCorrections"):
        kind = el.get("RefType")
        if kind not in LINKS: continue
        p = el.find("PMID"); other = normalize_pmid(p.text if p is not None else None)
        # Unlinked text is not promoted to a PMID by parsing RefSource/title.
        links.append(dict(relation=kind, pmid=other))
    row = dict(pmid=pmid, publication_types=sorted({e.text for e in article.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType") if e.text}),
        links=sorted(links,key=lambda r:(r["relation"],r["pmid"])), witness=deepcopy(witness))
    validate_projection(row)
    return row


def validate_projection(row):
    from .kg_paper_identity import normalize_pmid
    if not isinstance(row,dict) or not row.get("pmid") or normalize_pmid(row["pmid"]) != row["pmid"]:
        raise ValueError("invalid publication status PMID")
    types = row.get("publication_types")
    if not isinstance(types,list) or any(not isinstance(t,str) or not t for t in types) or len(types)!=len(set(types)):
        raise ValueError("invalid publication types")
    links = row.get("links")
    if not isinstance(links,list): raise ValueError("invalid publication links")
    seen=set()
    for item in links:
        if not isinstance(item,dict) or item.get("relation") not in LINKS:
            raise ValueError("invalid publication link type")
        other=item.get("pmid")
        if not isinstance(other,str) or (other and normalize_pmid(other)!=other): raise ValueError("invalid linked PMID")
        key=(item["relation"],other)
        if key in seen or other==row["pmid"]: raise ValueError("duplicate or self publication link")
        seen.add(key)
    witness=row.get("witness") or {}
    if not re.fullmatch("[a-f0-9]{64}",str(witness.get("response_sha256",""))): raise ValueError("missing publication response hash")
    if not str(witness.get("url","")).startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"):
        raise ValueError("invalid publication authority URL")


def adjudicate_publication_status(source, linked):
    validate_projection(source)
    peers={}
    for row in linked:
        validate_projection(row)
        if row["pmid"] in peers or row["pmid"]==source["pmid"]: raise ValueError("duplicate publication peer")
        peers[row["pmid"]]=row
    def reciprocal(forward,reverse,notice_types=None):
        for item in source["links"]:
            if item["relation"]!=forward:continue
            peer=peers.get(item["pmid"])
            if peer and dict(relation=reverse,pmid=source["pmid"]) in peer["links"]:
                if notice_types is None or set(peer["publication_types"]) & notice_types: return True
        return False
    types=set(source["publication_types"])
    if "Retracted Publication" in types:
        if not reciprocal("RetractionIn","RetractionOf",RETRACTION_NOTICE_TYPES):
            raise ValueError("retraction notice lacks reciprocal owning PMID/type")
        return "retracted"
    if "Corrected and Republished Article" in types:
        # Old PubMed records use RepublishedFrom/In even for corrected versions;
        # the explicit owning publication type prevents confusing a plain reprint.
        if any(reciprocal(f,r) for f,r in (("CorrectedandRepublishedFrom","CorrectedandRepublishedIn"),
            ("RepublishedFrom","RepublishedIn"), ("RetractedandRepublishedFrom","RetractedandRepublishedIn"))):
            return "corrected_republished"
        raise ValueError("corrected republication lacks reciprocal origin link")
    raise ValueError("no supported reviewed publication status")


def make_publication_review(source, linked):
    return dict(version=VERSION,pmid=source["pmid"],status=adjudicate_publication_status(source,linked),
        source=deepcopy(source),linked=deepcopy(linked))


def validate_publication_review(pmid, review):
    if not isinstance(review,dict) or review.get("version")!=VERSION or review.get("pmid")!=pmid:
        raise ValueError("invalid publication review identity")
    source=review.get("source")
    if not isinstance(source,dict) or source.get("pmid")!=pmid: raise ValueError("publication review owner mismatch")
    if not isinstance(review.get("linked"),list): raise ValueError("missing linked publication witnesses")
    status=adjudicate_publication_status(source,review["linked"])
    if review.get("status")!=status: raise ValueError("publication status not reproduced from witnesses")
    return status
