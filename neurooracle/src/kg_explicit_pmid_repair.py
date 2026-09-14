"""Exact source-provenance repair helpers, never a fuzzy identity fallback."""
from copy import deepcopy
from collections import Counter
import hashlib

from .kg_bibliography_repair import primary_abstract, quote_key
from .kg_identity_pilot import digest
from .metadata_field_audit import Coverage


def require(ok, message):
    if not ok:
        raise ValueError(message)


def abstract_witness(article, original, pmid):
    """Match a complete owning abstract, optionally its own partner abstract.

    NLM OtherAbstract may be a collaborator's summary, not the author's text.
    Never take text from referenced articles or count it as another source.
    """
    require(article.tag == "PubmedArticle" and article.findtext("./MedlineCitation/PMID") == pmid,
            "wrong owning PMID")
    value = quote_key(original)
    primary = primary_abstract(article)
    other = article.findall("./MedlineCitation/OtherAbstract")
    combined = " ".join([primary, *[" ".join("".join(x.itertext()) for x in a.findall("./AbstractText")) for a in other]])
    if len(value) < 100:
        return None
    if value == quote_key(primary):
        mode, parts = "complete_primary_abstract", []
    elif other and value == quote_key(combined):
        mode = "complete_primary_plus_own_other_abstract"
        parts = [dict(type=a.get("Type"), language=a.get("Language", "eng")) for a in other]
    else:
        return None
    return dict(mode=mode, other_abstracts=parts,
                normalized_full_abstract_sha256=hashlib.sha256(value.encode()).hexdigest())


def reference_changes(edge, claim, pmid_event=None, title_event=None):
    """Only exact owner-linked bibliography copies; extraction tags stay tags."""
    require((edge.get("metadata") or {}).get("claim_id") == claim["id"], "different reference owner")
    md = claim["metadata"]; p = md["source_paper"]
    changed, reasons, held = {}, {}, []
    if pmid_event:
        target = "claim:" + pmid_event["set_source_paper_pmid"]
        allowed = {"claim:" + (p.get("pmid") or claim["id"]): "schema_source"}
        if p.get("doi"):
            allowed["claim:" + p["doi"]] = "same_proven_explicit_doi"
        if edge.get("source") == target:
            pass
        elif edge["relation_type"] == "about" and edge.get("source") == "claim_extraction":
            pass  # This is extraction provenance, not an article identifier.
        elif edge.get("source") in allowed:
            changed["source"] = target
            reasons["source"] = allowed[edge["source"]]
        else:
            held.append(("source", "unproven_source_reference"))
    if title_event and edge.get("evidence_ref") != p["title"]:
        predecessor = deepcopy(claim)
        predecessor["metadata"]["source_paper"]["title"] = edge.get("evidence_ref")
        if digest(predecessor) == title_event["source_node_sha256"]:
            changed["evidence_ref"] = p["title"]
            reasons["evidence_ref"] = "exact_accepted_predecessor_title_hash"
        else:
            held.append(("evidence_ref", "not_the_exact_verified_prior_title"))
    return changed, reasons, held


def apply_reference(edge, event):
    require(digest(edge) == event["source_edge_sha256"], "edge changed after review")
    require(set(event["set_fields"]) <= {"source", "evidence_ref"} and event["set_fields"], "unapproved edge fields")
    out = deepcopy(edge); out.update(event["set_fields"])
    require(digest(out) == event["current_edge_sha256"], "edge result differs")
    return out


def masked_record(kind, row, event):
    if event is None:
        return row
    out = deepcopy(row)
    if kind == "node":
        out["metadata"].pop("pmid", None)
        del out["metadata"]["source_paper"]["pmid"]
    else:
        require(set(event["set_fields"]) <= {"source", "evidence_ref"}, "unapproved edge mask")
        for field in event["set_fields"]:
            del out[field]
    return out


def project_coverage(baseline_rows, removed, added):
    """Exact same-record-count delta; zero-occurrence fields disappear."""
    require(removed.denominators == added.denominators, "coverage record counts changed")
    out = Coverage()
    for row in baseline_rows:
        key = (row["scope"], row["field"])
        require(key not in out.fields, "duplicate coverage field")
        out.denominators[row["scope"]] = row["denominator"]
        out.fields[key] = Counter({k: row[k] for k in ("present", "nonempty", "placeholder_like")})
        out.types[key] = Counter(row["types"])
    for key in set(removed.fields) | set(added.fields):
        out.fields[key].subtract(removed.fields[key]); out.fields[key].update(added.fields[key])
        out.types[key].subtract(removed.types[key]); out.types[key].update(added.types[key])
        require(all(n >= 0 for n in (*out.fields[key].values(), *out.types[key].values())), "negative coverage")
        out.types[key] = +out.types[key]
        require(sum(out.types[key].values()) == out.fields[key]["present"], "coverage type counts differ")
        if not out.fields[key]["present"]:
            require(not any(out.fields[key].values()), "nonzero empty field")
            del out.fields[key]; del out.types[key]
    return out.rows()
