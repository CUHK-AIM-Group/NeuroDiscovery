"""Real resolver/save/reload tests on tiny synthetic graphs, no model calls."""
from types import SimpleNamespace
import pytest
from neurooracle.src import claim_ingestion as ingestion
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode
from neurooracle.src.storage import save_graph,load_graph
from neurooracle.src.claim_semantics import concept_atom_roles
from neurooracle.src import kg_scoped_structure as r

NAMES=[r.ACTIVATION,r.INTERACTION,*[s[1] for s in r.MRI_SPECS.values()]]


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('roundtrip',[False,True])
def test_actual_resolver_reuses_complete_literal_without_new_node(name,roundtrip,tmp_path,monkeypatch):
    monkeypatch.setattr(ingestion,'_resolution_idx',ingestion._ResolutionIndex())
    monkeypatch.setattr(ingestion,'_NOISE_FILTER_ENABLED',False)
    monkeypatch.setattr(ingestion,'_STRICT_PHASE1',False)
    monkeypatch.setattr(ingestion,'_DROP_LOG',SimpleNamespace(record=lambda *a,**k:None))
    node=r.literal_node(name);kg=KnowledgeGraph()
    kg.add_concept(ConceptNode(**node))
    kg.add_concept(ConceptNode(id='CUI:C1414531',preferred_name='FANCE',domain_tags=['gene'],aliases=['FACE']))
    kg.add_concept(ConceptNode(id='CUI:C1421437',preferred_name='VCP',domain_tags=['gene'],aliases=['TERA']))
    if roundtrip:kg=load_graph(save_graph(kg,tmp_path/'synthetic_literals.json'))
    expected='individual_data' if name==r.INTERACTION else 'imaging_marker'
    # The existing dataset_variable domain intentionally supports both roles;
    # the claim's explicit individual_data role is not changed by this repair.
    expected_roles={'individual_data','outcome'} if name==r.INTERACTION else {'imaging_marker'}
    assert {a.value for a in concept_atom_roles(kg.get_concept(node['id']))}==expected_roles
    before={key:n.to_dict() for key,n in kg._index.items()}
    for _ in range(2):assert ingestion.resolve_entity(kg,name,expected.upper())==node['id']
    assert {key:n.to_dict() for key,n in kg._index.items()}==before
