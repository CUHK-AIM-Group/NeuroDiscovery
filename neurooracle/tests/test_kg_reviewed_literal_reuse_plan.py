from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_reviewed_literal_reuse import adjusted_ordinal,protein_identity_candidates,inspection_bridge


@pytest.mark.parametrize('ordinal,expected',[(1,1),(4,3),(7,4),(100,97)])
def test_only_preceding_removed_edges_shift_ordinal(ordinal,expected):
    assert adjusted_ordinal(ordinal,[3,5,6])==expected


def test_removed_ordinal_has_no_new_index():
    with pytest.raises(ValueError):adjusted_ordinal(5,[3,5,6])


@pytest.mark.parametrize('node,expected',[
    (dict(id='UNIPROT:Q01853'),True),
    (dict(id='x',external_ids=dict(UniProt='Q01853')),True),
    (dict(id='x',preferred_name='Vcp protein, mouse'),True),
    (dict(id='x',preferred_name='VCP',external_ids=dict(NCBI_Gene='7415')),False),
    (dict(id='x',external_ids=dict(other='Q018530')),False),
])
def test_species_accession_search_is_not_a_generic_acronym_match(node,expected):
    assert protein_identity_candidates(node) is expected
