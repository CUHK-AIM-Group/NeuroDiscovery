from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import apply_kg_literal_endpoint_repair as engine
import apply_kg_measurement_reuse as adapter
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from neurooracle.src import kg_measurement_reuse as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import literal_node, reviewed_edges, apply_edge
from neurooracle.tests.test_kg_measurement_reuse import claim, proof, changes


def fixture(monkeypatch, cid=repair.COP):
    p = proof(); row = claim(cid); node = literal_node(row['metadata']['subject_name'])
    event, out = repair.reviewed_claim(row, changes(row), p)
    refs = [(1, dict(source_id=cid, target_id=row['metadata']['subject_id'], relation_type='about', metadata={})),
            (2, dict(source_id=cid, target_id=row['metadata']['object_id'], relation_type='about', metadata={}))]
    plan = dict(events=[event], edge_events=reviewed_edges(cid, row, out, refs),
        new_literals=[dict(id=node['id'], name=node['preferred_name'], node_sha256=digest(node))], existing_targets={})
    source = [('metadata', None, {}), ('node', 'CUI:C1414531', dict(id='CUI:C1414531', preferred_name='FANCE')),
        ('node', 'CUI:outcome', dict(id='CUI:outcome', preferred_name='psychotic experience severity')),
        ('node', cid, row), *[('edge', n, e) for n, e in refs]]
    monkeypatch.setattr(adapter, '_proof', p, raising=False)
    monkeypatch.setattr(engine, 'reviewed_claim', adapter.reviewed_claim)
    monkeypatch.setattr(engine, 'change_claim', repair.change_claim)
    monkeypatch.setattr(engine, 'reverse_claim', adapter.verified_reverse)
    return source, plan


def test_adapter_configuration_is_new_round_without_import_time_mutation():
    assert engine.OUTPUT.name == 'round37_relation_scope'
    conf = adapter.configuration()
    assert conf['OUTPUT'].name == 'round38_measurement_reuse'
    assert conf['TEMP'].name.endswith('.measurement.tmp')
    assert conf['SOURCE'] == engine.SOURCE and conf['CATALOG'] != engine.CATALOG
    assert conf['reverse_claim'] is adapter.verified_reverse


@pytest.mark.parametrize('cid', sorted(repair.BOUNDS))
def test_full_stream_serialization_catalog_and_inverse_source_digest(tmp_path, monkeypatch, cid):
    source, plan = fixture(monkeypatch, cid)
    transform = engine.LiteralTransform(plan); candidate = list(transform.records(iter(source)))
    output = BytesIO(); catalog = tmp_path / 'catalog.jsonl'
    result = stream_patch(iter(candidate), output, [], {}, catalog)
    data = json.loads(output.getvalue())
    reread = [('metadata', None, data['metadata']), *[('node', k, v) for k, v in data['concepts'].items()],
              *[('edge', i, e) for i, e in enumerate(data['edges'], 1)]]
    checks = verify_catalog_and_graph(iter(reread), result, catalog)
    assert checks['shared_relation_index_complete'] and result['counts'] == dict(nodes=4, claims=1, edges=2)
    inverse = {k: hashlib.sha256() for k in ('nodes', 'edges')}
    by_edge = {e['ordinal']: e for e in plan['edge_events']}
    for kind, key, row in candidate:
        if kind == 'metadata' or key == plan['new_literals'][0]['id']: continue
        if kind == 'node' and key == cid: row = adapter.verified_reverse(row, plan['events'][0])
        if kind == 'edge' and key in by_edge: row = apply_edge(row, by_edge[key], reverse=True)
        inverse[kind + 's'].update(compact(row).encode() + b'\n')
    assert {k: v.hexdigest() for k, v in inverse.items()} == {k: v.hexdigest() for k, v in transform.digests.items()}


def test_independent_reverse_reproduces_source_decision_not_just_plan_hash(monkeypatch):
    source, plan = fixture(monkeypatch)
    row = source[3][2]; event = deepcopy(plan['events'][0])
    event['science_changes'][0]['new'] = .088
    event.pop('current_node_sha256')
    out = repair.change_claim(row, event); event['current_node_sha256'] = digest(out)
    assert repair.reverse_claim(out, event) == row
    with pytest.raises(Exception, match='review differs'):
        adapter.verified_reverse(out, event)


def test_plan_numeric_tampering_rejected_in_forward_stream(monkeypatch):
    source, plan = fixture(monkeypatch)
    plan['events'][0]['science_changes'][0]['new'] = .088
    with pytest.raises(ValueError, match='output hash'):
        list(engine.LiteralTransform(plan).records(iter(source)))


def test_scientific_fields_do_not_masquerade_as_identity_only_validation(tmp_path, monkeypatch):
    path = tmp_path / 'VALIDATED.json'
    payload = dict(checks=dict(all_nonidentity_claim_fields_preserved=True,
        records_match_approved_identity_only_transform=True, shared_relation_index_complete=True))
    monkeypatch.setattr(adapter, 'OUTPUT', tmp_path)
    monkeypatch.setattr(adapter, '_original_validate', lambda: adapter.journal.atomic_json(path, payload))
    adapter.validate(); result = adapter.journal.read_json(path)['checks']
    assert 'all_nonidentity_claim_fields_preserved' not in result
    assert 'records_match_approved_identity_only_transform' not in result
    for key in ('verified_identity_proofs_complete', 'paper_identity_witnesses_validated',
                'publication_status_witnesses_validated', 'only_approved_three_numeric_and_method_repairs',
                'all_other_scientific_fields_preserved', 'regression_not_replaced_by_univariate_correlation'):
        assert result[key] is True

