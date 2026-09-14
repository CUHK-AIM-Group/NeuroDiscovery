import pytest

from neurooracle.src.kg_paper_identity import normalize_pmid,normalize_doi,normalize_pmcid,title_key,bibliography


@pytest.mark.parametrize("value",["12345678",12345678,"PMID: 12345678","https://pubmed.ncbi.nlm.nih.gov/12345678/"])
def test_exact_pmid_formats(value):
    assert normalize_pmid(value)=="12345678"


@pytest.mark.parametrize("value",[True,False,0,"0","123e5","1.0","0123","123,456","PMID:123:456","N/A"])
def test_invalid_pmid_not_coerced(value):
    assert normalize_pmid(value)==""


@pytest.mark.parametrize("value",["10.1234/ABC(5).", "doi:10.1234/ABC(5).", "https://doi.org/10.1234/ABC(5).","http://dx.doi.org/10.1234/ABC(5)."])
def test_doi_only_unwraps_known_prefix_and_case(value):
    assert normalize_doi(value)=="10.1234/abc(5)."


@pytest.mark.parametrize("value",[None,True,"10.123/x","10.1234/has space","10.1234/control\x00", "doi:bad","https://untrusted.test/10.1234/ABC"])
def test_invalid_doi_held(value):
    assert normalize_doi(value)==""


def test_complete_title_never_discards_negation_qualifiers_or_greek():
    assert title_key("A study.")==title_key("a   study")
    assert title_key("Treatment of α-disease")!=title_key("Treatment of disease")
    assert title_key("Treatment works")!=title_key("Treatment does not work")
    assert title_key("Left hippocampal volume")!=title_key("hippocampal volume")


def test_bibliography_does_not_copy_abstract_and_pmc_version_not_coerced():
    assert bibliography({"pmid":"123","abstract":"private/full content"})=={"pmid":"123"}
    assert normalize_pmcid("pmc123")=="PMC123" and normalize_pmcid("PMC123.2")==""
    assert normalize_pmcid("PMCID:PMC2719003") == "PMC2719003"


@pytest.mark.parametrize("value", [
    "10.1002/1098-2779(2000)6:3<180::AID-MRDD5>3.0.CO;2-I",
    "10.1002/(SICI)1521-3951(199911)216:1<135::AID-PSSB135>3.0.CO;2-#",
    "10.1234/opaque&suffix",
])
def test_legacy_opaque_doi_punctuation_preserved_not_misclassified(value):
    assert normalize_doi(value) == value.casefold()
