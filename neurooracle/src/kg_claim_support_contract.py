"""Source-bound semantic decisions for the paper-centered KG pipeline.

This module checks a structured assessment; it does not infer scientific
entailment or authorize graph publication. Old binary labels stay diagnostic.
No network, graph scan, model invocation, or production mutation occurs here.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json

VERSION = "kg.claim_support_contract.v1"
DIMENSIONS = (
    "subject", "relation", "object", "polarity", "modality", "quantifier",
    "species", "population", "intervention", "comparator", "outcome",
    "time", "setting",
)
CORE = {"subject", "relation", "object", "polarity", "modality", "quantifier"}
MATCHES = {"entailed", "narrower", "partial", "different", "missing",
           "contradicts", "not_asserted"}
ROLES = {"original_result", "background_statement", "review_synthesis",
         "hypothesis_only", "context_only"}
DECISIONS = {"full_support", "narrower_scope", "partial_support",
             "context_only", "contradicts", "uncertain"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contract(proposition, *, modality="asserted", quantifier="as worded; no universal inference",
             scope=None, atomicity="atomic", needs_clarification=False):
    """Construct a contract from an explicitly adjudicated reading, not NLP.

    Unspecified dimensions are not universal claims. Reviewers must still
    report material restrictions found in the source for those dimensions.
    """
    require(modality in {"asserted", "proposed", "historical_report"}, "invalid modality")
    require(atomicity in {"atomic", "all_of", "any_of", "ambiguous"}, "invalid atomicity")
    require(set(proposition) == {"subject_name", "predicate", "object_name"}, "proposition fields")
    require(all(isinstance(v, str) and v.strip() for v in proposition.values()), "empty proposition")
    values = {"subject": proposition["subject_name"], "relation": proposition["predicate"],
              "object": proposition["object_name"], "polarity": "affirmative",
              "modality": modality, "quantifier": quantifier}
    require(not (set(scope or {}) - set(DIMENSIONS)), "unknown scope dimension")
    require(not (set(scope or {}) & {"subject", "relation", "object", "modality", "quantifier"}),
            "scope cannot replace the proposition")
    values.update(scope or {})
    result = {"version": VERSION, "proposition": deepcopy(proposition),
              "atomicity": atomicity, "needs_clarification": needs_clarification,
              "dimensions": {d: {"target": values.get(d),
                                  "requirement": "asserted" if d in values else "not_asserted"}
                             for d in DIMENSIONS}}
    validate_contract(result)
    return result


def validate_contract(value):
    require(value.get("version") == VERSION, "contract version")
    require(value.get("atomicity") in {"atomic", "all_of", "any_of", "ambiguous"}, "atomicity")
    require(type(value.get("needs_clarification")) is bool, "clarification flag")
    require(set(value.get("dimensions", {})) == set(DIMENSIONS), "contract dimension coverage")
    for key, row in value["dimensions"].items():
        require(row.get("requirement") in {"asserted", "not_asserted"}, "scope requirement")
        if row["requirement"] == "asserted":
            require(isinstance(row.get("target"), str) and row["target"].strip(), "empty target")
        else:
            require(key not in CORE and row.get("target") is None, "core scope cannot be omitted")
    for d, p in (("subject", "subject_name"), ("relation", "predicate"), ("object", "object_name")):
        require(value["dimensions"][d]["target"] == value["proposition"][p], "proposition drift")
    require(value["dimensions"]["modality"]["target"] in
            {"asserted", "proposed", "historical_report"}, "invalid modality")


def span(source, paragraph, start=0, end=None):
    """Use Python Unicode character offsets into the unmodified abstract."""
    require(type(paragraph) is int and 0 <= paragraph < len(source["abstract"]), "paragraph range")
    text = source["abstract"][paragraph]["text"]
    end = len(text) if end is None else end
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "span range")
    return {"id": f"p{paragraph}:{start}:{end}", "paragraph": paragraph,
            "start": start, "end": end, "text": text[start:end]}


def locate_quote(source, quote):
    """Return all exact matches; never repair or case-fold a model quotation."""
    if not isinstance(quote, str) or not quote:
        return []
    found = []
    for p, row in enumerate(source.get("abstract", [])):
        start = row["text"].find(quote)
        while start >= 0:
            found.append(span(source, p, start, start + len(quote)))
            start = row["text"].find(quote, start + 1)
    return found


def evaluate(claim, source, assessment):
    """Derive a decision from source-bound fields, never a reported boolean.

    An internally consistent answer can still be scientifically wrong. Host
    review or an independently validated classifier remains a separate step.
    """
    validate_contract(claim)
    require(assessment.get("version") == VERSION, "assessment version")
    require(assessment.get("contract_sha256") == digest(claim), "contract changed")
    require(assessment.get("source_sha256") == digest(source), "source changed")
    require(str(assessment.get("pmid")) == str(source.get("pmid")) and bool(source.get("pmid")), "source identity")
    require(assessment.get("source_role") in ROLES, "source role")
    require(type(assessment.get("uncertainty")) is bool, "uncertainty must be boolean")
    require(set(assessment.get("dimensions", {})) == set(DIMENSIONS), "assessment dimension coverage")
    spans = assessment.get("spans")
    require(isinstance(spans, list) and bool(spans), "source spans required")
    by_id = {}
    for item in spans:
        require(item == span(source, item["paragraph"], item["start"], item["end"]), "source quote differs")
        require(item["id"] not in by_id, "duplicate span id")
        by_id[item["id"]] = item
    flags = defaultdict(list)
    for dim in DIMENSIONS:
        row = assessment["dimensions"][dim]
        match = row.get("match")
        require(match in MATCHES, "unknown dimension match")
        require(isinstance(row.get("reason"), str) and row["reason"].strip(), "dimension explanation required")
        refs = row.get("span_ids")
        require(isinstance(refs, list) and all(isinstance(r, str) for r in refs), "span references")
        require(len(refs) == len(set(refs)) and all(r in by_id for r in refs), "unbound span")
        if match == "not_asserted":
            require(claim["dimensions"][dim]["requirement"] == "not_asserted", "required scope ignored")
        elif match != "missing":
            require(bool(refs), "positive or conflicting match needs a source span")
        flags[match].append(dim)
    has_counterevidence = bool(flags["contradicts"])
    if flags["different"] or assessment["source_role"] == "context_only":
        decision = "context_only"
    elif claim["atomicity"] != "atomic" or claim["needs_clarification"]:
        decision = "uncertain"
    elif assessment["uncertainty"]:
        decision = "uncertain"
    elif has_counterevidence:
        # A negative result in another population or with a missing comparator
        # is not a direct refutation of the complete target proposition.
        decision = "uncertain" if (flags["narrower"] or flags["partial"] or flags["missing"]) else "contradicts"
    elif flags["partial"]:
        decision = "partial_support"
    elif flags["missing"]:
        decision = "uncertain"
    elif flags["narrower"]:
        decision = "narrower_scope"
    elif assessment["source_role"] == "hypothesis_only" and claim["dimensions"]["modality"]["target"] != "proposed":
        decision = "context_only"
    else:
        decision = "full_support"
    reported = assessment.get("reported_decision")
    require(reported is None or reported in DECISIONS, "invalid reported decision")
    agrees = reported is None or reported == decision
    return {"version": VERSION, "decision": decision,
            "contract_sha256": digest(claim), "source_sha256": digest(source),
            "assessment_sha256": digest(assessment), "pmid": str(source["pmid"]),
            "dimension_flags": dict(flags), "reported_decision_consistent": agrees,
            "source_role": assessment["source_role"],
            "semantic_candidate": decision == "full_support" and agrees,
            "scientific_entailment_verified_by_this_validator": False,
            "graph_write_authorized": False, "independent_replication_established": False}


def adapt_legacy(case, result):
    """Preserve an old result and diagnostics, without fabricating new fields."""
    require(case["case_id"] == result["case_id"], "legacy case mismatch")
    require(str(case["source"]["pmid"]) == str(result["pmid"]), "legacy source mismatch")
    quotes = []
    source_text = [p["text"] for p in case["source"].get("abstract", [])]
    for q in result["exact_source_quotes"]:
        matches = locate_quote(case["source"], q)
        issue = None
        if not matches:
            # Diagnostic classification only: neither normalization creates
            # an exact quote or a valid source offset.
            if any(" ".join(q.split()) in " ".join(t.split()) for t in source_text):
                issue = "whitespace_only"
            elif any(q.casefold() in t.casefold() for t in source_text):
                issue = "case_only"
            else:
                issue = "not_exact_in_abstract"
        quotes.append({"quote": q, "exact_matches": matches, "mismatch_kind": issue})
    return {"version": VERSION, "case_id": case["case_id"], "pmid": str(result["pmid"]),
            "legacy_result": deepcopy(result), "legacy_result_sha256": digest(result),
            "source_sha256": digest(case["source"]), "quote_diagnostics": quotes,
            "unmatched_quote_count": sum(not q["exact_matches"] for q in quotes),
            "decision": "uncertain", "needs_structured_reassessment": True,
            "semantic_candidate": False, "graph_write_authorized": False}


def pack_by_paper(cases, contracts=None):
    """One unchanged abstract per source document, any number of target claims.

    A PMID is a retrieval key, not proof that papers or cohorts are distinct.
    Two different documents for the same PMID are flagged, never overwritten.
    """
    grouped, seen, pmids = {}, set(), defaultdict(set)
    for case in cases:
        require(set(case) == {"case_id", "source", "proposition"}, "unhandled case fields would be lost")
        cid, source = case["case_id"], case["source"]
        require(cid not in seen, "duplicate case id")
        seen.add(cid)
        require(bool(source.get("pmid")), "missing paper retrieval identity")
        key = digest(source)
        pmids[str(source["pmid"])].add(key)
        unit = grouped.setdefault(key, {"source_sha256": key, "source": deepcopy(source), "claims": []})
        item = {"case_id": cid, "proposition": deepcopy(case["proposition"])}
        if contracts is not None:
            validate_contract(contracts[cid])
            require(contracts[cid]["proposition"] == case["proposition"], "packed proposition drift")
            item["contract"] = deepcopy(contracts[cid])
        unit["claims"].append(item)
    return {"version": VERSION, "case_count": len(seen), "source_document_count": len(grouped),
            "conflicting_pmid_documents": {p: sorted(v) for p, v in pmids.items() if len(v) > 1},
            "units": list(grouped.values())}


def unpack_cases(packed):
    require(packed.get("version") == VERSION, "packed version")
    result, seen = [], set()
    for unit in packed["units"]:
        require(digest(unit["source"]) == unit["source_sha256"], "packed source changed")
        for item in unit["claims"]:
            require(item["case_id"] not in seen, "duplicate packed case")
            seen.add(item["case_id"])
            if "contract" in item:
                validate_contract(item["contract"])
                require(item["contract"]["proposition"] == item["proposition"], "packed contract changed")
            result.append({"case_id": item["case_id"], "proposition": deepcopy(item["proposition"]),
                           "source": deepcopy(unit["source"])})
    require(len(result) == packed["case_count"], "packed case count")
    require(len(packed["units"]) == packed["source_document_count"], "packed document count")
    return result
