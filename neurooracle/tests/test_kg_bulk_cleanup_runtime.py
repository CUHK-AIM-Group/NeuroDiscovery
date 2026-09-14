from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

from neurooracle.src.kg_bulk_cleanup import simplify_record
from neurooracle.src.kg_metadata_compaction import LAYOUT,compact_record
from neurooracle.src.claim_evidence_identity import evidence_signature,evidence_dedup_key
from neurooracle.src.correlation_grouping import POLICY,IndexTerms,enabled
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode,Edge
from neurooracle.src.storage import load_graph,save_graph
from neurooracle.tests.test_kg_bulk_cleanup import row

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_paired_luna_claim_campaign import relation_signature,scope_signature
from manage_paired_luna_postreview import is_low_scope_confidence


def test_scientific_idempotency_and_legacy_readers_before_after_compaction():
    source=row();out=simplify_record('node',source)
    assert evidence_signature(source['metadata'])==evidence_signature(out['metadata'])
    assert evidence_dedup_key(source['metadata'])==evidence_dedup_key(out['metadata'])
    assert relation_signature(source['metadata'])==relation_signature(out['metadata'])
    assert scope_signature(source['metadata'])==scope_signature(out['metadata'])
    assert is_low_scope_confidence([source['metadata']])==is_low_scope_confidence([out['metadata']])
    changed=deepcopy(out);changed['metadata']['evidence']['p_value']=0
    assert evidence_dedup_key(source['metadata'])!=evidence_dedup_key(changed['metadata'])


@pytest.mark.parametrize('policy',[False,True])
def test_real_storage_and_relation_query_preserve_evidence_and_direction(tmp_path,policy):
    kg=KnowledgeGraph();kg.serialization_metadata={'metadata_layout':deepcopy(LAYOUT)}
    if policy:kg.serialization_metadata['relation_grouping']=deepcopy(POLICY)
    for nid in ('A','B'):kg.add_concept(ConceptNode(id=nid,preferred_name=nid))
    first=row();second=deepcopy(first)
    second['id']=second['metadata']['id']='CLM:second'
    sm=second['metadata'];sm['subject_id'],sm['object_id']=sm['object_id'],sm['subject_id']
    sm['subject_name'],sm['object_name']=sm['object_name'],sm['subject_name']
    sm['source_paper']['pmid']='456';sm['negated']=True
    for r in (first,second):
        kg.add_concept(ConceptNode.from_dict(r))
        m=r['metadata'];kg.add_edge(Edge(source_id=m['subject_id'],target_id=m['object_id'],relation_type=m['predicate'],metadata={'claim_id':r['id'],'negated':m['negated']}))
    original_edges=list(kg.iter_edge_records())
    for i in range(2):
        path=save_graph(kg,tmp_path/f'round{i}.json');parsed=json.loads(path.read_text(encoding='utf8'))
        md=parsed['concepts']['CLM:test']['metadata']
        assert 'scope_confidence' not in md['metadata'] and 'subject_canonical_hint' not in md['metadata']
        assert md['evidence']['p_value'] is None and md['evidence']['effect_size']==0
        assert md['scope_reaudit']==first['metadata']['scope_reaudit']
        kg=load_graph(path)
        groups=list(kg.iter_relation_evidence())
        assert len(groups)==(1 if policy else 2)
        if policy:
            assert groups[0]['paper_count']==2 and groups[0]['claim_count']==2
            assert {m['negated'] for m in groups[0]['members']}=={False,True}
        assert list(kg.iter_edge_records())==original_edges
        claim=kg.get_claim('CLM:test')
        kg._index['CLM:test'].metadata=claim.to_dict()


def test_unknown_policy_fails_closed():
    with pytest.raises(ValueError):enabled({'relation_grouping':{'version':'future'}})


def test_whole_name_alias_canonicalization_precedes_orientation():
    class ExactTerms:
        def relation_key(self,claim):return ('Z','complete verified alias','correlates_with','A','outcome')
    assert IndexTerms(ExactTerms()).relation_key({})==('A','outcome','correlates_with','Z','complete verified alias')


def test_storage_compactor_is_idempotent():
    result=compact_record('node',row())
    assert compact_record('node',result)==result
