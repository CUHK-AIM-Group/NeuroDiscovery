from copy import deepcopy
import pytest
from neurooracle.src import kg_expanded_literal_repair as repair
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges


def fixture(name='gray matter asymmetry',typ='IMAGING_MARKER'):
    row=claim(name,typ);row['metadata']['confidence']=0.6
    gene=dict(id='CUI:C1823340',preferred_name='MATT',semantic_types=['T028'],aliases=['MATT'],metadata={})
    row['metadata']['subject_id']=gene['id'];row['metadata']['metadata']['subject_id']=gene['id']
    witness=dict(node_id=gene['id'],name='MATT',labels=['MATT'],semantic_types=['T028'],node_sha256=digest(gene))
    from neurooracle.src.claim_semantics import declared_type_atoms
    review=dict(claim_id=row['id'],claim_sha256=digest(row),side='subject',name=name,current_node_id=gene['id'],
        literal_candidate_gate=None,structural_holds=[],existing_name_or_alias_candidates=[],declared_roles=sorted(a.value for a in declared_type_atoms(typ)),
        outer_type=typ,inner_type=None,source_gene_node_sha256=digest(gene),word_interior_alias_hits=['matt'])
    change=dict(side='subject',old_id=gene['id'],name=name,target_id=repair.literal_node(name)['id'])
    return row,gene,witness,review,change


def event_for(row,witness,review,change):
    return repair.reviewed_claim(row,[change],{witness['node_id']:witness},{repair.review_key(row['id'],'subject'):review})


def test_complete_literal_preserves_all_scientific_values_and_seal():
    row,gene,witness,review,change=fixture();before=deepcopy(row)
    event,current=event_for(row,witness,review,change)
    assert row==before and nonidentity_claim(current)==nonidentity_claim(before)
    assert repair.reverse_claim(current,event)==before
    node=repair.literal_node(change['name'])
    assert node['metadata']=={} and node['aliases']==[] and node['semantic_types']==[] and node['external_ids']=={}


@pytest.mark.parametrize('kind',['source_hash','gene_hash','whole_alias','role','nested','collision','surface','target','scope','no_review'])
def test_no_unreviewed_scope_is_generalized(kind):
    row,gene,witness,review,change=fixture()
    if kind=='source_hash':review['claim_sha256']='0'*64
    elif kind=='gene_hash':witness['node_sha256']='0'*64
    elif kind=='whole_alias':witness['labels'].append('gray')
    elif kind=='role':review['declared_roles']=['gene_target']
    elif kind=='nested':
        row['metadata']['metadata']['subject_id']='different';review['claim_sha256']=digest(row)
    elif kind=='collision':review['existing_name_or_alias_candidates']=['existing']
    elif kind=='surface':row['metadata']['subject_name']='left gray matter asymmetry'
    elif kind=='target':change['target_id']='CLM_CONCEPT:paper_specific_target'
    elif kind=='scope':review['structural_holds']=['unsafe']
    else:review={}
    with pytest.raises(ValueError):event_for(row,witness,review,change)


@pytest.mark.parametrize('name',['gene expression in gray matter','gray matter protein levels','APOE-related gray matter asymmetry'])
def test_gene_molecular_phrases_stay_held(name):
    row,gene,witness,review,change=fixture(name)
    with pytest.raises(ValueError,match='nonmolecular'):event_for(row,witness,review,change)


def test_owned_reference_transform_remains_complete_and_invertible():
    row,gene,witness,review,change=fixture();event,current=event_for(row,witness,review,change)
    owned=edges(row);changes=repair.reviewed_edges(row['id'],row,current,owned)
    assert len(changes)==2
    for e in changes:
        original=dict(owned)[e['ordinal']]
        assert repair.apply_edge(repair.apply_edge(original,e),e,reverse=True)==original
    with pytest.raises(ValueError):repair.reviewed_edges(row['id'],row,current,owned+[(4,deepcopy(owned[0][1]))])
