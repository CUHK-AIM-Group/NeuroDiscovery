"""Adversarial contract checks; synthetic fixtures are not accuracy estimates."""
from copy import deepcopy
import pytest
from neurooracle.src import kg_claim_support_contract as k


@pytest.fixture
def example():
    source = {"pmid": "synthetic-1", "abstract": [{"label": "RESULTS", "text":
              "In adult rats, compound α increased sleep after 30 days. We observed this in 12 rats.\u2028No human study was performed."}]}
    claim = k.contract({"subject_name": "compound α", "predicate": "increased",
                        "object_name": "sleep in adult rats after 30 days"},
                       scope={"species": "rats", "population": "adult", "time": "30 days"})
    quote = k.span(source, 0)
    assessment = {"version": k.VERSION, "pmid": source["pmid"], "contract_sha256": k.digest(claim),
                  "source_sha256": k.digest(source), "source_role": "original_result", "uncertainty": False,
                  "spans": [quote], "dimensions": {d: {
                      "match": "entailed" if v["requirement"] == "asserted" else "not_asserted",
                      "reason": "Synthetic matched statement" if v["requirement"] == "asserted" else "No claim constraint",
                      "span_ids": [quote["id"]] if v["requirement"] == "asserted" else []}
                      for d, v in claim["dimensions"].items()}}
    return claim, source, assessment


def set_match(assessment, dim, match):
    assessment["dimensions"][dim] = {"match": match, "reason": "Material test difference",
                                     "span_ids": [assessment["spans"][0]["id"]]}


def rebind(claim, assessment):
    assessment["contract_sha256"] = k.digest(claim)


def test_full_is_not_publication_or_replication(example):
    result = k.evaluate(*example)
    assert result["decision"] == "full_support"
    assert result["semantic_candidate"]
    assert not result["graph_write_authorized"]
    assert not result["independent_replication_established"]
    assert not result["scientific_entailment_verified_by_this_validator"]


@pytest.mark.parametrize("dimension", k.DIMENSIONS)
def test_material_mismatch_overrides_reported_support(example, dimension):
    claim, source, assessment = example
    set_match(assessment, dimension, "different")
    assessment["reported_decision"] = "full_support"
    result = k.evaluate(claim, source, assessment)
    assert result["decision"] == "context_only"
    assert not result["reported_decision_consistent"]
    assert not result["semantic_candidate"]


@pytest.mark.parametrize("match,expected", [("narrower", "narrower_scope"),
    ("partial", "partial_support"), ("missing", "uncertain"), ("contradicts", "contradicts")])
def test_scoped_and_negative_evidence_retained(example, match, expected):
    claim, source, assessment = example
    set_match(assessment, "outcome", match)
    assert k.evaluate(claim, source, assessment)["decision"] == expected


def test_actual_material_limit_on_unspecified_scope_still_blocks(example):
    claim, source, assessment = example
    assert claim["dimensions"]["setting"]["requirement"] == "not_asserted"
    set_match(assessment, "setting", "narrower")
    assert k.evaluate(*example)["decision"] == "narrower_scope"


def test_unasserted_cohort_size_is_not_automatic_failure(example):
    claim, source, assessment = example
    assessment["dimensions"]["population"]["reason"] = "Adult rats match; cohort size 12 remains evidence metadata."
    assert k.evaluate(*example)["decision"] == "full_support"


@pytest.mark.parametrize("atomicity", ["all_of", "any_of", "ambiguous"])
def test_composite_needs_explicit_atomic_children(example, atomicity):
    claim, source, assessment = example
    claim["atomicity"] = atomicity
    rebind(claim, assessment)
    assert k.evaluate(*example)["decision"] == "uncertain"


def test_clarification_cannot_be_overridden(example):
    claim, source, assessment = example
    claim["needs_clarification"] = True
    rebind(claim, assessment)
    assert k.evaluate(*example)["decision"] == "uncertain"


def test_hypothesis_not_established_mechanism(example):
    claim, source, assessment = example
    assessment["source_role"] = "hypothesis_only"
    assert k.evaluate(*example)["decision"] == "context_only"
    claim["dimensions"]["modality"]["target"] = "proposed"
    rebind(claim, assessment)
    assert k.evaluate(*example)["decision"] == "full_support"


@pytest.mark.parametrize("role", ["background_statement", "review_synthesis"])
def test_background_is_semantic_support_not_replication(example, role):
    example[2]["source_role"] = role
    result = k.evaluate(*example)
    assert result["semantic_candidate"] and not result["independent_replication_established"]


@pytest.mark.parametrize("mutation", ["case", "offset", "source", "contract", "pmid", "span_ref", "missing_dimension", "ignored_dimension", "non_bool"])
def test_binding_and_schema_attacks_fail(example, mutation):
    claim, source, assessment = example
    if mutation == "case": assessment["spans"][0]["text"] = assessment["spans"][0]["text"].upper()
    elif mutation == "offset": assessment["spans"][0]["start"] = True
    elif mutation == "source": source["abstract"][0]["text"] += "!"
    elif mutation == "contract": claim["dimensions"]["species"]["target"] = "humans"
    elif mutation == "pmid": assessment["pmid"] = "other"
    elif mutation == "span_ref": assessment["dimensions"]["subject"]["span_ids"] = ["p9:0:5"]
    elif mutation == "missing_dimension": del assessment["dimensions"]["species"]
    elif mutation == "ignored_dimension": set_match(assessment, "species", "not_asserted")
    elif mutation == "non_bool": assessment["uncertainty"] = "false"
    with pytest.raises(ValueError): k.evaluate(*example)


def test_legacy_preserved_and_case_mismatch_not_silently_fixed(example):
    claim, source, _ = example
    case = {"case_id": "a", "source": source, "proposition": claim["proposition"]}
    legacy = {"case_id": "a", "pmid": source["pmid"], "supports_proposition": True,
              "exact_source_quotes": ["we observed", "compound α"], "scope_differences": "No concerns"}
    before = deepcopy(legacy)
    result = k.adapt_legacy(case, legacy)
    assert legacy == before == result["legacy_result"]
    assert result["unmatched_quote_count"] == 1
    assert result["needs_structured_reassessment"] and not result["semantic_candidate"]


def test_lossless_paper_reuse_and_conflicting_documents(example):
    claim, source, _ = example
    cases = [{"case_id": str(i), "source": deepcopy(source), "proposition": claim["proposition"]} for i in range(3)]
    packed = k.pack_by_paper(cases)
    assert packed["source_document_count"] == 1
    assert k.unpack_cases(packed) == cases
    cases[2]["source"]["abstract"][0]["text"] += " Different record."
    packed = k.pack_by_paper(cases)
    assert packed["source_document_count"] == 2 and len(packed["conflicting_pmid_documents"]) == 1
    assert sorted(k.unpack_cases(packed), key=lambda r:r["case_id"]) == cases


def test_duplicate_and_corrupt_packing_fails(example):
    claim, source, _ = example
    case = {"case_id": "a", "source": source, "proposition": claim["proposition"]}
    with pytest.raises(ValueError): k.pack_by_paper([case, case])
    packed = k.pack_by_paper([case])
    packed["units"][0]["source"]["pmid"] = "different"
    with pytest.raises(ValueError): k.unpack_cases(packed)


def test_required_quantifier_cannot_be_deleted(example):
    claim, source, assessment = example
    claim["dimensions"]["quantifier"] = {"target": None, "requirement": "not_asserted"}
    rebind(claim, assessment)
    with pytest.raises(ValueError): k.evaluate(*example)


@pytest.mark.parametrize("scope_match,expected", [("different", "context_only"),
    ("narrower", "uncertain"), ("missing", "uncertain")])
def test_different_or_incomplete_scope_is_not_a_global_refutation(example, scope_match, expected):
    set_match(example[2], "outcome", "contradicts")
    set_match(example[2], "species", scope_match)
    assert k.evaluate(*example)["decision"] == expected


def test_quote_diagnostics_do_not_repair_nbsp_or_case(example):
    claim, source, _ = example
    source["abstract"][0]["text"] = "More than 80\u00a0% carry the mutation. In this study we tested a hypothesis."
    case = {"case_id": "a", "proposition": claim["proposition"], "source": source}
    legacy = {"case_id": "a", "pmid": source["pmid"],
              "exact_source_quotes": ["More than 80 % carry the mutation.", "We tested a hypothesis."]}
    result = k.adapt_legacy(case, legacy)
    assert [q["mismatch_kind"] for q in result["quote_diagnostics"]] == ["whitespace_only", "case_only"]
    assert all(not q["exact_matches"] for q in result["quote_diagnostics"])


def test_packing_cannot_silently_drop_qualifiers(example):
    claim, source, _ = example
    case = {"case_id": "a", "proposition": claim["proposition"], "source": source, "extra_scope": "rats"}
    with pytest.raises(ValueError): k.pack_by_paper([case])
