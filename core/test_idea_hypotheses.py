"""Offline typed-chain contract, real mini graph and runtime handoff tests."""
from copy import deepcopy
from dataclasses import replace
import json
import sqlite3
import threading

import pytest

from core.idea_hypotheses import (TEMPLATES, _candidate, bind_candidate_chain, generate_hypotheses,
                                  validate_candidate_shape, run_chain_tool)
from core.web.claim_layer_v8 import AcceptedClaimLayer
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import relation_id, relation_key
from neurooracle.tests.test_claim_evidence_query import fingerprint


@pytest.fixture(autouse=True)
def synthetic_sources(monkeypatch):
    from core import idea_source_review
    monkeypatch.setattr(idea_source_review, 'load_source_packets', lambda ids: {
        s: dict(source_id=s, status='available', text='Synthetic source abstract for ' + s + ', not real science.') for s in ids})


def synthetic_rows():
    result = []
    for i, (source, stype, predicate, target, ttype) in enumerate([
        ('Gene', 'GENE_TARGET', 'is_associated_with', 'Marker', 'IMAGING_MARKER'),
        ('Marker', 'IMAGING_MARKER', 'is_biomarker_of', 'Disease', 'DISEASE'),
    ], 1):
        md = dict(id=f'CLM:{i}', subject_id=source, subject_name=source, subject_type=stype,
                  predicate=predicate, object_id=target, object_name=target, object_type=ttype,
                  negated=False, population='adults', conditions=['adjusted'], measurement='MRI', time='baseline',
                  raw_text=f'Synthetic evidence {i}, not real science.', source_paper=dict(pmid=str(i), title='Synthetic source '+str(i), year=2020),
                  evidence=dict(p_value=0.02, sample_size=100, direction='positive'), metadata={'qualifier': 'retained'})
        review = dict(source_anchor=md['raw_text'], scope_note='Synthetic adult baseline observations only.')
        obs = dict(claim_id=md['id'], original_claim=md, negated=False, raw_text=md['raw_text'],
                   source_review=review, proposition_support='supports', observation_role='own_result')
        result.append(dict(metadata=md, observation=obs, bibliography=md['source_paper'],
                           paper_key='pmid:'+str(i), work_key='pmid:'+str(i), verified=True, publication_review=None))
    return result


def synthetic_candidate():
    return _candidate(TEMPLATES['genetic_imaging_disease'], ['Gene', 'Marker', 'Disease'],
                      synthetic_rows(), {'graph_revision': 'test-revision'}, 'Marker Disease')


def test_shared_node_does_not_overwrite_one_papers_measurement_with_the_other():
    rows=synthetic_rows()
    original=_candidate(TEMPLATES['genetic_imaging_disease'],['Gene','Marker','Disease'],
                        rows,{'graph_revision':'test-revision'},'Marker Disease')
    rows[0]['metadata']['object_name']='hippocampal volume'
    rows[1]['metadata']['subject_name']='cortical thickness'
    got=_candidate(TEMPLATES['genetic_imaging_disease'],['Gene','Marker','Disease'],
                   rows,{'graph_revision':'test-revision'},'Marker Disease')
    assert 'hippocampal volume' in got['hypothesis'] and 'cortical thickness' in got['hypothesis']
    assert 'equivalence remains unverified' in got['hypothesis']
    assert got['chain']==original['chain'] and got['candidate_id']==original['candidate_id']
    assert got['status']=='provisional'  # A label difference alone is not a veto.


def synthetic_batch(queries):
    rows = {r['metadata']['id']: r for r in synthetic_rows()}
    return dict(graph_revision='test-revision', endpoint_nodes={n:dict(id=n, preferred_name=n) for n in ('Gene','Marker','Disease')},
                results=[dict(graph_revision='test-revision', requested_claim_id=q['claim_id'],
                original_claim_ids=[q['claim_id']], papers=[dict(bibliography=rows[q['claim_id']]['bibliography'],
                paper_key=rows[q['claim_id']]['paper_key'], work_key=rows[q['claim_id']]['work_key'], verified=True,
                publication_review=None, observations=[deepcopy(rows[q['claim_id']]['observation'])])]) for q in queries])


def setup_chain_graph(tmp_path, change=None, rows=None):
    custom_rows = rows is not None
    rows = synthetic_rows() if rows is None else rows
    if change:
        change(rows)
    records = [dict(id=row['metadata']['id'], metadata=row['metadata']) for row in rows]
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        return fingerprint(path)
    concepts = {n:dict(id=n, preferred_name=n, domain_tags=[d]) for n, d in [('Gene','gene'),('Marker','imaging_feature'),('Disease','disease')]}
    if custom_rows:
        for row in rows:
            for side in ('subject', 'object'):
                nid = row['metadata'][side + '_id']
                concepts[nid] = dict(id=nid, preferred_name=row['metadata'][side + '_name'], domain_tags=[])
    graph = save('graph.json', dict(metadata={}, concepts={**concepts, **{r['id']: r for r in records}}, edges=[]))
    graph_bytes = (tmp_path / 'graph.json').read_bytes()
    census_path, index_path = tmp_path / 'census.sqlite', tmp_path / 'index.sqlite'
    with sqlite3.connect(census_path) as census, sqlite3.connect(index_path) as index:
        census.execute('CREATE TABLE claims(cid TEXT PRIMARY KEY,node_sha TEXT,relation_id TEXT,shared INTEGER)')
        index.execute('CREATE TABLE observations(number INTEGER PRIMARY KEY,cid TEXT,node_sha TEXT,byte_offset INTEGER,rid TEXT,'
                      'subject_id TEXT,subject_name TEXT,predicate TEXT,object_id TEXT,object_name TEXT,pmid TEXT,doi TEXT,source_key TEXT,paper_sig TEXT,subject_type TEXT,object_type TEXT)')
        for i, r in enumerate(records):
            md = r['metadata']; rid = relation_id(relation_key(md)); seal = digest(r)
            census.execute('INSERT INTO claims VALUES(?,?,?,0)', (r['id'], seal, rid))
            key = json.dumps(r['id']).encode() + b':'
            index.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                          (i, r['id'], seal, graph_bytes.index(key)+len(key), rid,
                           *(md[k] for k in ('subject_id', 'subject_name', 'predicate', 'object_id', 'object_name')),
                           md['source_paper']['pmid'], '', rows[i]['paper_key'], str(i), json.dumps(md.get('subject_type')), json.dumps(md.get('object_type'))))
    census = save('census.json', dict(graph=graph, database=fingerprint(census_path)))
    for name in ('catalog.jsonl', 'dossiers.jsonl'):
        (tmp_path / name).write_text('', encoding='utf-8')
    catalog, dossiers = fingerprint(tmp_path/'catalog.jsonl'), fingerprint(tmp_path/'dossiers.jsonl')
    papers = save('papers.json', dict(version='kg.paper_identity.v1', records={str(i):dict(pmid=str(i),
                  title='Synthetic source '+str(i), years=['2020'], doi=[], pmcid=[], witness={'response_sha256':'a'*64}) for i in (1, 2)}))
    terms = save('terms.json', dict(version='kg.verified_entity_terms.v1', terms=[]))
    reviews = save('reviews.json', {r['id']:dict(claim_sha256=digest(r), source_role='primary_research_article',
                  observation_role='own_result', proposition_support='supports', source_anchor=r['metadata']['raw_text'],
                  scope_note='Synthetic.') for r in records})
    checks = {key: True for key in ('shared_relation_index_complete', 'verified_identity_proofs_complete',
              'paper_identity_witnesses_validated', 'all_current_census_rows_independently_verified', 'shared_observation_dossiers_complete')}
    acceptance = save('acceptance.json', dict(graph=graph, current_paper_census=census, shared_relations=catalog,
                      entity_terms=terms, paper_identities=papers, evidence_dossiers=dossiers, source_role_reviews=reviews, checks=checks))
    manifest = save('index.json', dict(schema='kg.global_claim_index.acceptance.v1', status='COMPLETE_RETRIEVAL_INDEX_NOT_SCIENTIFIC_APPROVAL',
                    graph=graph, database=fingerprint(index_path), all_claim_hashes_verified=True, all_current_claims_indexed=len(records)))
    projection = save('projection.json', dict(groups=[], source_reviews={}, original_relation_extension=dict(global_index_acceptance=manifest, members=[])))
    layer = save('layer.json', dict(schema='kg.accepted_claim_layer.v6', status='ACCEPTED', base_graph=graph, base_acceptance=acceptance,
                 base_dossiers=dossiers, projection=projection, checks=dict(source_anchors_and_whole_scopes_validated=True, original_observations_unchanged=True),
                 counts=dict(reviewed_multipaper_claims_after=0)))
    campaign = save('campaign.json', dict(status='COMPLETED', active_process=None, current_graph=graph, current_acceptance=acceptance,
                    current_paper_census=census, current_shared_relations=catalog, current_evidence_dossiers=dossiers,
                    current_entity_terms=terms, current_paper_identities=papers, current_source_role_reviews=reviews, current_claim_layer=layer))
    return AcceptedClaimLayer(tmp_path/'campaign.json')


def test_real_mini_graph_generates_traceable_postprocessed_chains_without_writes(tmp_path):
    layer = setup_chain_graph(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = generate_hypotheses(layer, 'Marker', limit=2)
    assert len(result['candidates']) == 1 and result['gap'] is None
    c = result['candidates'][0]
    assert c['chain']['node_ids'] == ['Gene', 'Marker', 'Disease']
    assert c['chain']['links'][0]['relation'] == 'is_associated_with'
    assert c['chain_context']['edges'][0]['context']['population'] == 'adults'
    assert c['chain_context']['edges'][0]['metadata']['metadata']['qualifier'] == 'retained'
    assert 'Recorded evidence' in c['rationale'] and 'adjusted' in c['rationale']
    assert 'out-of-sample' in c['prediction'] and not c['proposed_inference']['observed_links_establish_full_chain']
    batch = layer.query_batch(queries=c['evidence_queries'])
    rebound = bind_candidate_chain(c, batch['results'], {k: batch[k] for k in ('graph_revision', 'claim_layer_revision', 'original_index_revision')},
                                   endpoint_nodes=layer.graph_nodes(c['chain']['node_ids']))
    assert rebound['chain_context'] == c['chain_context']
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('change', [
    lambda r: r[0]['metadata'].update(subject_type=None),
    lambda r: r[0]['metadata'].update(predicate='about'),
    lambda r: r[1]['metadata'].update(subject_id='different'),
    lambda r: r[0]['metadata'].update(subject_id='Marker', object_id='Gene'),
])
def test_wrong_type_relation_disconnected_or_reversed_edges_do_not_generate(tmp_path, change):
    layer = setup_chain_graph(tmp_path, change)
    result = generate_hypotheses(layer, 'Marker')
    assert result['candidates'] == [] and result['gap']


@pytest.mark.parametrize('change', [lambda r: r[1]['metadata'].update(population='children'),
                                   lambda r: r[1]['metadata'].update(negated=True)])
def test_scope_difference_or_negative_edge_is_held_not_silently_composed(tmp_path, change):
    result = generate_hypotheses(setup_chain_graph(tmp_path, change), 'Marker')
    assert result['candidates'] == [] and result['held_chains'][0]['status'] == 'scope_review_required'


@pytest.mark.parametrize('population', [None, {'mean_age': None, 'cohort_name': None}])
def test_unknown_scope_is_kept_unknown(population):
    rows = synthetic_rows()
    for r in rows:
        r['metadata']['population'] = population
        r['verified'] = False
        r['observation']['source_review'] = None
    c = _candidate(TEMPLATES['genetic_imaging_disease'], ['Gene', 'Marker', 'Disease'], rows, {'graph_revision':'r'}, 'Marker')
    assert c['chain_context']['edges'][0]['context']['population'] is None
    assert c['chain_context']['edges'][0]['context']['direction'] == 'positive'
    assert c['chain_context']['edges'][0]['metadata']['population'] == population
    assert 'to be specified' in c['hypothesis'] and 'not been reviewed' in c['limitations']


def test_source_supplements_remain_evidence_not_traversable_graph_edges(tmp_path):
    from core.idea_hypotheses import records_from_documents
    result = generate_hypotheses(setup_chain_graph(tmp_path), 'Marker')
    assert len(result['candidates']) == 1
    assert result['retrieval']['typed_index_records'] == 2
    docs = synthetic_batch([{'claim_id':'CLM:1'}])['results']
    obs = docs[0]['papers'][0]['observations'][0]
    obs['source_derived_claim'] = obs.pop('original_claim')
    assert records_from_documents(docs) == {}


def test_legacy_chain_walker_cannot_ignore_strict_relation_rules():
    engine = HypothesisEngine.__new__(HypothesisEngine)
    with pytest.raises(ValueError, match='enumerate_typed_paths'):
        engine.batch_generate_for_chain(TEMPLATES['genetic_imaging_disease'])


@pytest.mark.parametrize('field,value', [('relation','causes'), ('source','fake'), ('claim_id','CLM:fake')])
def test_candidate_cannot_change_graph_triples(field, value):
    c = synthetic_candidate(); c['chain']['links'][0][field] = value
    with pytest.raises(ValueError):
        batch = synthetic_batch(c['evidence_queries'])
        bind_candidate_chain(c, batch['results'], {'graph_revision':'test-revision'}, endpoint_nodes=batch['endpoint_nodes'])


def test_rebinding_discards_candidate_authored_evidence_and_rejects_stale_revision():
    c = synthetic_candidate(); c['chain_context'] = {'fabricated': 'evidence'}
    docs = synthetic_batch(c['evidence_queries'])['results']
    nodes = synthetic_batch(c['evidence_queries'])['endpoint_nodes']
    bound = bind_candidate_chain(c, docs, {'graph_revision':'test-revision'}, endpoint_nodes=nodes)
    assert 'fabricated' not in bound['chain_context'] and bound['chain_context']['edges']
    with pytest.raises(ValueError, match='revision'):
        bind_candidate_chain(c, docs, {'graph_revision':'different'}, endpoint_nodes=nodes)
    with pytest.raises(ValueError, match='actual graph'):
        bind_candidate_chain(c, docs, {'graph_revision':'test-revision'}, endpoint_nodes={})


def test_symmetric_walk_keeps_actual_edge_direction_and_no_arbitrary_reversal():
    template = TEMPLATES['disease_biomarker_prognosis']
    rows = synthetic_rows()
    rows[0]['metadata'].update(subject_id='Marker', subject_type='IMAGING_MARKER', object_id='Disease', object_type='DISEASE')
    rows[1]['metadata'].update(object_id='Outcome', object_type='OUTCOME', predicate='predicts')
    paths = HypothesisEngine.enumerate_typed_paths(template, [r['metadata'] for r in rows], anchor_claim_ids={'CLM:1'})
    assert paths[0][0] == ['Disease', 'Marker', 'Outcome']
    assert paths[0][1][0]['subject_id'] == 'Marker'
    with pytest.raises(ValueError, match='symmetric'):
        replace(template, relations=(('causes',), ('predicts',)))


def test_http_and_readonly_tool(tmp_path):
    from fastapi.testclient import TestClient
    from core.web.server import create_app
    from core.permissions import tool_permission
    layer = setup_chain_graph(tmp_path)
    app = create_app(); app.state.accepted_claim_evidence = layer
    with TestClient(app) as client:
        result = client.get('/api/kg/idea-hypotheses', params={'topic':'Marker'})
        assert result.status_code == 200 and len(result.json()['candidates']) == 1
        assert client.get('/api/kg/idea-hypotheses', params={'topic':'Marker','template':'invented'}).status_code == 422
    assert tool_permission('read_only', 'generate_idea_hypotheses', {}, tmp_path) == 'allow'
    cancelled = threading.Event(); cancelled.set()
    assert run_chain_tool(layer, {'topic':'Marker'}, cancelled)['error_type'] == 'cancelled'


def test_runtime_writer_requires_chain(tmp_path):
    from core.research_writer import write_research_file
    c = synthetic_candidate()
    args = dict(path='candidate.json', kind='hypotheses', content=json.dumps([c]))
    c.pop('chain')
    bad = {**args, 'content':json.dumps([c])}
    assert not write_research_file(tmp_path, bad, threading.Event(), require_chain=True)['success']
    assert write_research_file(tmp_path, args, threading.Event(), require_chain=True)['success']


def test_all_reviews_receive_rebound_chain_and_context(monkeypatch):
    from core import novelty_gate
    from core.agent.test_novelty_gate import reviewer
    monkeypatch.setattr(novelty_gate, 'resolve_graph_evidence', synthetic_batch)
    seen = []
    def review(system, user, label):
        packet = json.loads(user); seen.append(packet)
        assert packet['chain']['node_ids'] == ['Gene','Marker','Disease']
        assert packet['chain_context']['edges'][0]['context']['measurement'] == 'MRI'
        return reviewer({'H1':'substantive_extension'})(system, user, label)
    result = novelty_gate.gate_candidates([synthetic_candidate()], 'balanced', review, topic='Marker', require_chain=True)
    assert len(seen) == 4 and result['candidates'][0]['chain'] == seen[0]['chain']


@pytest.mark.parametrize('novelty_required', [False, True])
def test_final_chain_check_cannot_be_waived_by_score_switch_or_retry_cap(tmp_path, monkeypatch, novelty_required):
    from core import novelty_gate
    from core.agent.test_autoresearch_execution import session
    from core.autoresearch_runtime import AutoResearchRun
    agent, calls = session(tmp_path, [], mode='idea')
    run = AutoResearchRun(tmp_path, 'idea')
    path = tmp_path / 'candidates.json'
    candidate = synthetic_candidate()
    # Still a legal template predicate, but different from the actual observation.
    candidate['chain']['links'][0]['relation'] = 'causes'
    candidate['kg_triples'][0]['relation'] = 'causes'
    path.write_text(json.dumps([candidate]), encoding='utf-8')
    run.state.update(status='verifying', artifacts=[str(path)], novelty_gate_required=novelty_required,
                     novelty_gate_rejections=3, novelty_gate_attempts=3)
    monkeypatch.setattr(novelty_gate, 'resolve_graph_evidence', synthetic_batch)
    result = agent._chain_validation_rejection(run)
    assert result['error_type'] == 'invalid_hypothesis_chain'
    assert run.state['status'] == 'running' and not run.state['chain_validation']['passed']
    assert not calls
    path.write_text(json.dumps([synthetic_candidate()]), encoding='utf-8')
    run.state['status'] = 'verifying'
    assert agent._chain_validation_rejection(run) is None
    assert run.state['chain_validation']['passed']


@pytest.mark.parametrize('payload', [lambda c: {'candidates': 'malformed'}, lambda c: [c, {'chain': {}}]])
def test_malformed_candidate_submission_is_not_treated_as_evidence_gap(tmp_path, payload):
    from core.agent.test_autoresearch_execution import session
    from core.autoresearch_runtime import AutoResearchRun
    agent, _ = session(tmp_path, [], mode='idea')
    run = AutoResearchRun(tmp_path, 'idea')
    path = tmp_path / 'candidates.json'
    path.write_text(json.dumps(payload(synthetic_candidate())), encoding='utf-8')
    run.state.update(status='verifying', artifacts=[str(path)])
    assert agent._chain_validation_rejection(run)['error_type'] == 'invalid_hypothesis_chain'


def test_idea_runtime_calls_real_mini_graph_tool_and_rebinds_submission(tmp_path):
    from core.agent.test_autoresearch_execution import session, reply, tool, completion
    graph_dir = tmp_path / 'graph'
    graph_dir.mkdir()
    layer = setup_chain_graph(graph_dir)
    candidate = generate_hypotheses(layer, 'Marker')['candidates'][0]
    agent, calls = session(tmp_path, [
        reply(calls=[tool('generate_idea_hypotheses', topic='Marker')]),
        reply(calls=[tool('write_research_file', path='candidates.json', kind='hypotheses', content=json.dumps([candidate]))]),
        reply(calls=[tool('finish_autoresearch', **completion(artifacts=['candidates.json'], evidence_ids=[1, 2]))]),
    ], mode='idea')
    agent._idea_claim_layer = layer
    agent.env.update(autoresearch_novelty_gate=False, autoresearch_independent_review=False)
    agent._chat()
    assert agent.autoresearch_state['status'] == 'completed'
    assert agent.autoresearch_state['chain_validation']['passed']
    assert any(t['function']['name'] == 'generate_idea_hypotheses' for t in calls[0]['tools'])
    assert len(calls) == 3  # Scripted main replies only; no model review/score call.


def test_large_pool_preserves_same_paper_edges_and_pages_without_retraversal(tmp_path, monkeypatch):
    rows = []
    for side, count in ((0, 17), (1, 18)):
        for i in range(count):
            row = deepcopy(synthetic_rows()[side])
            md = row['metadata']; md['id'] = f'CLM:{side}:{i}'
            endpoint = 'subject' if side == 0 else 'object'
            md[endpoint + '_id'] += str(i)
            md[endpoint + '_name'] += ' ' + str(i)
            rows.append(row)
    layer = setup_chain_graph(tmp_path, rows=rows)
    before = {p:p.read_bytes() for p in tmp_path.iterdir()}
    result = generate_hypotheses(layer, 'Marker')
    assert result['requested_limit'] == 500
    assert len(result['candidates']) == 306  # 17 * 18 distinct graph paths, not paper permutations.
    assert result['retrieval']['anchor_source_count'] == 2
    assert len({tuple(c['chain']['node_ids']) for c in result['candidates']}) == 306
    assert before == {p:p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(layer, 'chain_index', lambda **kw: pytest.fail('cached pages must not rescan the index'))
    page = run_chain_tool(layer, {'topic':'Marker','offset':20,'page_size':20}, threading.Event())
    assert page['candidate_count'] == 306 and len(page['candidates']) == 20 and page['next_offset'] == 40
    assert page['candidates'] == result['candidates'][20:40]


def test_paper_variants_do_not_inflate_hypothesis_count():
    records = [row['metadata'] for row in synthetic_rows()]
    for i in range(100):
        records.append({**records[0], 'id':f'CLM:duplicate:{i}'})
    paths = HypothesisEngine.enumerate_typed_paths(TEMPLATES['genetic_imaging_disease'], records,
                                                  anchor_claim_ids={r['id'] for r in records}, limit=500)
    assert len(paths) == 1


def test_shared_topic_match_must_survive_each_paths_own_evidence(tmp_path, monkeypatch):
    layer = setup_chain_graph(tmp_path)
    monkeypatch.setattr(layer, 'claim_index', lambda: [(dict(original_claim_ids=['CLM:1', 'CLM:2']), 'Marker MCI')])
    result = generate_hypotheses(layer, 'Marker MCI')
    assert not result['candidates'] and result['retrieval']['topic_mismatch_paths'] == 1


def test_pool_cache_cannot_hide_changed_evidence(tmp_path):
    layer = setup_chain_graph(tmp_path)
    assert generate_hypotheses(layer, 'Marker')['candidates']
    (tmp_path/'index.sqlite').write_bytes(b'changed evidence')
    with pytest.raises((ValueError, RuntimeError, OSError)):
        generate_hypotheses(layer, 'Marker')
