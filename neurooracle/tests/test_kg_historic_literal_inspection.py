from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_kg_historic_literal_candidates import exact_label,node_projection
from neurooracle.src.kg_identity_pilot import digest


def test_current_literal_candidate_projection_is_not_a_node_backup():
    row=dict(id='CLM_CONCEPT:a',preferred_name='complete scope',aliases=[],metadata=dict(legacy_field='preserved',audit={'x':1}),
        semantic_types=[],external_ids={},domain_tags=['claim_concept'],source_vocab='manual')
    out=node_projection(row)
    assert out['node_sha256']==digest(row) and out['metadata_sha256']==digest(row['metadata'])
    assert out['metadata_keys']==['audit','legacy_field'] and out['complete_record_not_saved']
    assert 'metadata' not in out and 'legacy_field' not in out


def test_full_name_scope_and_case_are_not_merged():
    assert exact_label(' bilateral   volume ')==exact_label('bilateral volume')
    assert exact_label('left volume')!=exact_label('bilateral volume')
    assert exact_label('Vcp')!=exact_label('VCP')
    assert exact_label('brain-structural change')!=exact_label('brain structural change')
