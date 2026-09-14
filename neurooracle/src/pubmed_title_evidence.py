"""Complete, owning PubMed XML title witnesses; never fuzzy title identity.

Sup/sub parentheses and markup-boundary spacing are rendered from the actual
authority element, never removed from arbitrary scientific text. All variants
retain every text node, including nested text, qualifiers, negation and tails.
"""
import html
import re
from xml.etree import ElementTree as ET

VERSION = "pubmed.complete_title.v1"
INLINE = frozenset({"i", "b", "sup", "sub", "u"})


def presentation_key(value):
    from .kg_paper_identity import authority_title
    # Bounded decoding handles observed doubly encoded HTML. No punctuation
    # deletion, word expansion, edit distance, accent/Greek or digit stripping.
    value = str(value or "")
    for _ in range(2): value = html.unescape(value)
    value = value.translate(str.maketrans({"‘":"'", "’":"'", "“":'"', "”":'"', "‐":"-", "‑":"-"}))
    # Hyphenation between letters only. Numeric ranges, minus, primes and
    # standalone dashes are deliberately NOT equivalent to ASCII hyphens.
    value = re.sub(r"(?<=[^\W\d_])–(?=[^\W\d_])", "-", value)
    return authority_title(value)


def title_renderings(title_xml):
    if not isinstance(title_xml, str) or len(title_xml) > 40000:
        raise ValueError("invalid complete title XML")
    root = ET.fromstring(title_xml)
    if root.tag != "ArticleTitle" or any(e.tag not in INLINE for e in list(root.iter())[1:]):
        raise ValueError("unsupported title markup; retain hold")
    if root.tail and root.tail.strip(): raise ValueError("unexpected title tail")

    def render(element, scripts):
        pieces = [element.text or ""]
        for child in element:
            part = render(child, scripts)
            if scripts and child.tag in {"sup", "sub"}: part = "(" + part + ")"
            pieces.extend((part, child.tail or ""))
        return "".join(pieces)

    # Plain itertext and space-joined itertext reproduce common complete XML
    # readers; the latter adds spaces only at real markup boundaries.
    variants = {presentation_key(render(root, False)), presentation_key(render(root, True)),
        presentation_key(" ".join(root.itertext()))}
    variants.discard("")
    if not variants: raise ValueError("empty complete title")
    return variants


def validated_title_keys(record):
    evidence = record.get("complete_title_evidence")
    if evidence is None: return None
    if not isinstance(evidence, dict) or evidence.get("version") != VERSION or evidence.get("pmid") != record["pmid"]:
        raise ValueError("invalid complete-title identity")
    if evidence.get("document_kind") not in {"journal_article", "book_document"}:
        raise ValueError("invalid owning document kind")
    witness = evidence.get("witness") or {}
    if not re.fullmatch("[a-f0-9]{64}", str(witness.get("response_sha256", ""))):
        raise ValueError("missing complete-title response hash")
    if not str(witness.get("url", "")).startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"):
        raise ValueError("invalid complete-title authority URL")
    if evidence.get("doi") != record["doi"] or evidence.get("pmcid") != record["pmcid"]:
        raise ValueError("XML and ESummary owning identifiers disagree")
    keys = title_renderings(evidence.get("title_xml"))
    if presentation_key(record["title"]) not in keys:
        raise ValueError("XML and ESummary complete titles disagree")
    return keys


def complete_title_evidence(article, record, witness):
    from .kg_paper_identity import normalize_pmid, normalize_doi, normalize_pmcid
    if article.tag == "PubmedArticle":
        base = "./MedlineCitation"; title_path = base + "/Article/ArticleTitle"
        ids_path = "./PubmedData/ArticleIdList/ArticleId"; kind = "journal_article"
    elif article.tag == "PubmedBookArticle":
        base = "./BookDocument"; title_path = base + "/ArticleTitle"
        ids_path = base + "/ArticleIdList/ArticleId"; kind = "book_document"
    else: raise ValueError("unsupported owning article")
    pmid_el = article.find(base + "/PMID")
    pmid = normalize_pmid(pmid_el.text if pmid_el is not None else None)
    if pmid != record["pmid"]: raise ValueError("XML owning PMID mismatch")
    ids = {"doi": set(), "pmcid": set()}
    for el in article.findall(ids_path):
        idtype = el.get("IdType")
        if idtype == "pubmed" and normalize_pmid(el.text) != pmid:
            raise ValueError("XML own articleids PMID mismatch")
        if idtype == "doi":
            value = normalize_doi(el.text)
            if not value: raise ValueError("invalid owning XML DOI")
            ids["doi"].add(value)
        if idtype == "pmc":
            value = normalize_pmcid(el.text)
            if not value: raise ValueError("invalid owning XML PMCID")
            ids["pmcid"].add(value)
    title = article.find(title_path)
    if title is None: raise ValueError("missing owning complete title")
    evidence = dict(version=VERSION, pmid=pmid, document_kind=kind,
        title_xml=ET.tostring(title, encoding="unicode"),
        **{k:sorted(v) for k,v in ids.items()}, witness=dict(witness))
    validated_title_keys({**record, "complete_title_evidence": evidence})
    return evidence
