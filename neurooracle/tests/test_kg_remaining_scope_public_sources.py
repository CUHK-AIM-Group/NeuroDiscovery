from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fetch_kg_remaining_scope_sources import owning_fulltext_ids


def test_fulltext_ids_only_own_article_not_reference_list():
    xml='<pmc-articleset><article><front><article-meta><article-id pub-id-type="pmid">123</article-id><article-id pub-id-type="doi">10.1/own</article-id></article-meta></front><back><article-id pub-id-type="pmid">999</article-id></back></article></pmc-articleset>'
    assert owning_fulltext_ids(xml)=={'pmid':'123','doi':'10.1/own'}


def test_non_fulltext_response_does_not_become_source_evidence():
    with pytest.raises(ValueError,match='no fulltext'):owning_fulltext_ids('<error>Unavailable</error>')
