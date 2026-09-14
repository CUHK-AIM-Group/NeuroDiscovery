"""Strict KG bibliographic identity primitives. No title-only or fuzzy merging."""
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path

VERSION="kg.paper_identity.v1"


def normalize_pmid(value):
    if isinstance(value,bool) or not isinstance(value,(str,int)): return ""
    text=str(value).strip()
    match=re.fullmatch(r"(?:https?://pubmed\.ncbi\.nlm\.nih\.gov/|PMID\s*:\s*)?([1-9][0-9]{0,8})/?",text,re.I)
    return match[1] if match else ""


def normalize_doi(value):
    if not isinstance(value,str): return ""
    text=value.strip()
    text=re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)","",text,flags=re.I)
    # Never remove trailing punctuation: parentheses and punctuation may be
    # part of an authentic DOI. Malformed values are held, not repaired by guess.
    # DOI suffixes are opaque. Legacy SICI/Wiley DOIs legitimately contain
    # angle brackets and other punctuation; a modern-only regex rejects them.
    # Syntax alone never counts as a registry or article-identity witness.
    if not re.fullmatch(r"10\.[0-9]{4,9}/[^\s]+",text,re.I): return ""
    if any(unicodedata.category(c) in {"Cc", "Cs"} for c in text): return ""
    return text.casefold()


def normalize_pmcid(value):
    if not isinstance(value,str): return ""
    text=re.sub(r"^PMCID\s*:\s*", "", value.strip(), flags=re.I).upper()
    return text if re.fullmatch(r"PMC[1-9][0-9]*",text) else ""


def title_key(value):
    if not isinstance(value,str): return ""
    # Complete text, case and whitespace normalization only; do not drop words,
    # punctuation, Greek letters, digits or negation to manufacture a match.
    return " ".join(unicodedata.normalize("NFKC",value).casefold().split()).rstrip(".")


def bibliography(paper):
    if not isinstance(paper,dict): return {}
    return {k:v for k,v in paper.items() if k in {"pmid","doi","pmcid","arxiv_id","openalex_id","title","year","publication_year","journal","authors"}}


def identifiers(paper):
    if not isinstance(paper,dict): return {}
    return {"pmid":normalize_pmid(paper.get("pmid")),"doi":normalize_doi(paper.get("doi")),"pmcid":normalize_pmcid(paper.get("pmcid"))}


def authority_title(value):
    # PubMed titles may contain presentation-only HTML. Never shorten the title.
    return title_key(html.unescape(re.sub(r"</?(?:i|b|sup|sub)\b[^>]*>", "", str(value or ""), flags=re.I)))


def pubmed_record(pmid, record, witness):
    """Keep bibliographic identity evidence only; no abstracts or model output."""
    pmid = normalize_pmid(pmid)
    if not pmid or str(record.get("uid")) != pmid:
        raise ValueError("PubMed identity mismatch")
    ids = {"doi": set(), "pmcid": set()}
    for item in record.get("articleids", []):
        kind = item.get("idtype")
        if kind == "pubmed" and normalize_pmid(item.get("value")) != pmid:
            raise ValueError("PubMed articleids mismatch")
        if kind == "doi" and (value := normalize_doi(item.get("value"))): ids["doi"].add(value)
        if kind == "pmc" and (value := normalize_pmcid(item.get("value"))): ids["pmcid"].add(value)
    years = sorted({year for name in ("pubdate", "epubdate")
        for year in re.findall(r"\b(?:18|19|20|21)[0-9]{2}\b", str(record.get(name) or ""))})
    preprint = "Preprint" in record.get("pubtype", []) or bool(re.search(r"\b(?:biorxiv|medrxiv|arxiv|research square|preprints?)\b", str(record.get("fulljournalname") or ""), re.I))
    return dict(pmid=pmid, title=authority_title(record.get("title")), years=years, preprint=preprint,
        **{k: sorted(v) for k,v in ids.items()}, witness=deepcopy(witness))


class VerifiedPaperIdentities:
    """Exact public-authority crosswalk; conflicting or unverified claims survive.

    This class does not use title similarity, co-occurrence or citation counts
    as an identity witness. Complete titles/years are compatibility guards.
    """

    def __init__(self, payload):
        if payload.get("version") != VERSION or not isinstance(payload.get("records"), dict):
            raise ValueError("unsupported paper identity registry")
        self._payload = deepcopy(payload)
        self._records = self._payload["records"]
        self._aliases = {"doi": defaultdict(set), "pmcid": defaultdict(set)}
        self._complete_titles = {}
        for pmid, row in self._records.items():
            if normalize_pmid(pmid) != pmid or row.get("pmid") != pmid:
                raise ValueError("invalid authority PMID")
            if not isinstance(row.get("title"), str) or not isinstance(row.get("years"), list):
                raise ValueError("invalid authority bibliography")
            witness = row.get("witness") or {}
            if not re.fullmatch("[a-f0-9]{64}", str(witness.get("response_sha256", ""))):
                raise ValueError("missing authority response hash")
            for kind, normalizer in (("doi", normalize_doi), ("pmcid", normalize_pmcid)):
                values = row.get(kind)
                if not isinstance(values, list) or len(set(values)) != len(values):
                    raise ValueError("invalid authority aliases")
                for value in values:
                    if not value or normalizer(value) != value: raise ValueError("invalid authority alias")
                    self._aliases[kind][value].add(pmid)
            if "complete_title_evidence" in row:
                from .pubmed_title_evidence import validated_title_keys
                self._complete_titles[pmid] = validated_title_keys(row)
        self._observed_collisions = {kind: set(payload.get("observed_collisions", {}).get(kind, []))
            for kind in ("doi", "pmcid")}
        self._publication_reviews = self._payload.get("publication_reviews", {})
        if not isinstance(self._publication_reviews, dict): raise ValueError("invalid publication review registry")
        for pmid, review in self._publication_reviews.items():
            from .kg_publication_status import validate_publication_review
            if pmid not in self._records: raise ValueError("publication review lacks verified identity authority")
            validate_publication_review(pmid, review)

    def export_payload(self):
        return deepcopy(self._payload)

    @property
    def has_publication_reviews(self):
        return bool(self._publication_reviews)

    def publication_review(self, paper_key):
        """Known status only: not_reviewed never means healthy or valid science.

        Canonical PMID source keys only; this method does not resolve identities,
        bridge versions, remove claims, or change a claim's observed conclusion.
        """
        pmid = paper_key[5:] if isinstance(paper_key, str) and paper_key.startswith("pmid:") else ""
        return deepcopy(self._publication_reviews.get(pmid, {"status":"not_reviewed"}))

    def resolve(self, claim):
        from .case_study_membership_contract import strongest_paper_key
        paper = claim.get("source_paper") or {}
        if not isinstance(paper, dict):
            return dict(paper_key=strongest_paper_key(claim), status="unresolved", reasons=["invalid_source_paper"])
        ids = identifiers(paper)
        reasons = {"invalid_" + kind for kind, value in ids.items() if paper.get(kind) and not value}
        if paper.get("arxiv_id"):
            reasons.add("unverified_arxiv_version")  # an accompanying journal DOI is not version-equivalence proof
        inner = claim.get("metadata") or {}
        for holder in (claim, inner if isinstance(inner, dict) else {}):
            if holder.get("pmid") and normalize_pmid(holder["pmid"]) != ids["pmid"]:
                reasons.add("alternate_pmid_conflict")
        candidates = {ids["pmid"]} if ids["pmid"] in self._records else set()
        for kind in ("doi", "pmcid"):
            found = self._aliases[kind].get(ids[kind], set())
            if len(found) > 1 or (ids[kind] in self._observed_collisions[kind] and not ids["pmid"]):
                reasons.add("ambiguous_" + kind)
            candidates.update(found)
        if len(candidates) > 1: reasons.add("identifier_disagreement")
        if len(candidates) == 1:
            pmid = next(iter(candidates)); record = self._records[pmid]
            if re.search(r"\b(?:biorxiv|medrxiv|arxiv|research square|preprints?)\b", str(paper.get("journal") or ""), re.I) and not record.get("preprint"):
                reasons.add("unverified_preprint_publication_version")
            if ids["pmid"] and ids["pmid"] != pmid: reasons.add("pmid_disagreement")
            for kind in ("doi", "pmcid"):
                if ids[kind] and ids[kind] not in record[kind]: reasons.add("authority_does_not_confirm_" + kind)
            if paper.get("title") and authority_title(paper["title"]) != record["title"]:
                from .pubmed_title_evidence import presentation_key
                if pmid not in self._complete_titles or presentation_key(paper["title"]) not in self._complete_titles[pmid]:
                    reasons.add("complete_title_disagreement")
            for field in ("year", "publication_year"):
                if paper.get(field) not in (None, "") and str(paper[field]) not in record["years"]:
                    reasons.add("publication_year_disagreement")
            if not reasons:
                return dict(paper_key="pmid:" + pmid, status="verified", reasons=[])
        key = strongest_paper_key(claim)
        if reasons:
            # Do not group conflicted identifiers into an apparent single paper.
            key = "unresolved:" + str(claim.get("id") or hashlib.sha256(
                json.dumps(paper, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
            return dict(paper_key=key, status="conflict", reasons=sorted(reasons))
        return dict(paper_key=key, status="unverified", reasons=["no_authority_witness"])


def load_graph_papers(path, metadata):
    declaration = metadata.get("paper_identity")
    if declaration is None: return None
    if not isinstance(declaration, dict) or declaration.get("version") != VERSION:
        raise ValueError("unsupported graph paper identity declaration")
    bundle = Path(path).resolve().parent
    relative = Path(declaration.get("registry", ""))
    target = (bundle / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(bundle.parent):
        raise ValueError("paper identity registry escapes graph bundle")
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != declaration.get("sha256"):
        raise ValueError("paper identity registry hash mismatch")
    return VerifiedPaperIdentities(json.loads(raw))


def census_identifier_projection(db):
    """Re-evaluate stored raw bibliography, not outdated normalized SQL columns."""
    aliases = {kind:defaultdict(set) for kind in ("doi", "pmcid")}
    changed = []
    for sig, encoded, pmid, doi, pmcid in db.execute("SELECT sig,payload,pmid,doi,pmcid FROM papers"):
        current = identifiers(json.loads(encoded))
        if current != dict(pmid=pmid, doi=doi, pmcid=pmcid):
            changed.append(dict(paper_signature=sig, normalized_identifiers=current))
        if current["pmid"]:
            for kind in aliases:
                if current[kind]: aliases[kind][current[kind]].add(current["pmid"])
    return dict(observed_collisions={kind:sorted(k for k,v in values.items() if len(v)>1) for kind,values in aliases.items()},
        normalized_column_overrides=changed,
        interpretation="read-only projection of immutable raw bibliographies; old normalized columns are not current authority")
