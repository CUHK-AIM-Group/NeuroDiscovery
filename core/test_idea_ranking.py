"""Synthetic scoring/reviewer fixtures; no provider calls or scientific accuracy claims."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest

from core import idea_ranking as ranking
from core.idea_hypotheses import generate_hypotheses
from core.novelty_gate import gate_candidates, resolve_graph_evidence, NoveltyGateError
from core.test_idea_hypotheses import setup_chain_graph, synthetic_rows, synthetic_sources
from core.agent.test_novelty_gate import reviewer
from neurooracle.src.kge import hypothesis_path


class SyntheticScorer:
    def __init__(self, candidates):
        triples = [e for c in candidates for e in c['kg_triples']]
        self.ent2idx = {v: i for i, v in enumerate(sorted({e[k] for e in triples for k in ('source', 'target')}))}
        self.rel2idx = {v: i for i, v in enumerate(sorted({e['relation'] for e in triples}))}

    def score_batch(self, triples):
        return [0.9 if s.endswith('3') else 0.6 for s, r, t in triples]


def make_pool(tmp_path):
    rows = []
    for side, count in ((0, 4), (1, 3)):
        for i in range(count):
            row = deepcopy(synthetic_rows()[side])
            md = row['metadata']
            md.update(id=f'CLM:{side}:{i}', confidence=0.8)
            endpoint = 'subject' if side == 0 else 'object'
            md[endpoint + '_id'] += str(i)
            md[endpoint + '_name'] += ' ' + str(i)
            rows.append(row)
    layer = setup_chain_graph(tmp_path, rows=rows)
    return layer, generate_hypotheses(layer, 'Marker')


def test_full_pool_scores_and_pagination_preserve_unreviewed_rows_and_files(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROCLAW_IDEA_GNN_MANIFEST', raising=False)
    layer, pool = make_pool(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result, same = ranking.prepare_ranking(layer, 'Marker')
    assert len(result['ranking']) == 12 and same == pool
    assert all(r['final_score'] is None and r['review_status'] == 'not_reviewed' for r in result['ranking'])
    assert result['gnn_covered_count'] == 0
    assert all(r['measurement']['gnn_path_score'] is None for r in result['ranking'])
    page = ranking.run_ranking_tool(layer, dict(topic='Marker', offset=4, page_size=4), threading.Event())
    assert page['pool_count'] == 12 and page['next_offset'] == 8
    assert [c['candidate_id'] for c in page['candidates']] == [r['candidate_id'] for r in result['ranking'][4:8]]
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}


def test_measured_gnn_can_move_a_late_candidate_without_inventing_novelty(tmp_path):
    layer, pool = make_pool(tmp_path)
    scorer = SyntheticScorer(pool['candidates'])
    result = ranking.screen_pool(pool, scorer=scorer, pair_counts={})
    best = next(c for c in pool['candidates'] if c['candidate_id'] == result['ranking'][0]['candidate_id'])
    assert best['chain']['node_ids'][0] == 'Gene3'
    assert result['gnn_covered_count'] == 12 and result['gnn_used_for_screening']
    assert result['ranking'][0]['measurement']['edge_scores'][0]['score'] == 0.9
    assert result['ranking'][0]['final_score'] is None
    scorer.ent2idx.pop('Gene3')
    partial = ranking.screen_pool(pool, scorer=scorer)
    assert partial['gnn_covered_count'] == 9 and not partial['gnn_used_for_screening']
    assert len(partial['ranking']) == 12  # OOV candidates were not discarded.


@pytest.mark.parametrize('stage', ['expert:prior_art', 'response:operationalization'])
def test_peer_round_retains_initial_and_new_prior_veto(tmp_path, stage):
    layer, pool = make_pool(tmp_path)
    candidate = pool['candidates'][0]
    calls = []
    def call(system, user, label):
        packet = json.loads(user)
        calls.append((label, packet))
        if 'response:' in label:
            assert len(packet['initial_reviews']) == 3
            assert all('critic_score' in r for r in packet['initial_reviews'])
        classification = 'exact_prior' if stage in label else 'substantive_extension'
        return reviewer({'H1': classification})(system, user, label)
    result = gate_candidates([candidate], 'novelty_first', call, require_chain=True, peer_response=True,
                             resolver=lambda q: resolve_graph_evidence(q, layer=layer))
    row = result['candidates'][0]
    assert len(calls) == 7 and result['selected_ids'] == []
    assert row['known_prior_veto'] and row['candidate_id'] == candidate['candidate_id']
    assert len(row['initial_reviews']) == len(row['peer_responses']) == 3
    assert len(calls[-1][1]['peer_responses']) == 3


def test_scores_flow_into_same_review_and_final_policy_then_merge_back_into_pool(tmp_path):
    layer, pool = make_pool(tmp_path)
    scorer = SyntheticScorer(pool['candidates'])
    pairs = layer.pair_support_counts([(c['chain']['node_ids'][0], c['chain']['node_ids'][-1]) for c in pool['candidates']])
    screened = ranking.screen_pool(pool, scorer=scorer, pair_counts=pairs)
    by_id = {c['candidate_id']: c for c in pool['candidates']}
    shortlist = [by_id[r['candidate_id']] for r in screened['ranking'][:3]]
    result = gate_candidates(shortlist, 'balanced', reviewer({'H1': 'substantive_extension'}),
        require_chain=True, peer_response=True, max_model_calls=7,
        resolver=lambda q: resolve_graph_evidence(q, layer=layer),
        score_provider=lambda payload: ranking.measure_candidate(payload, scorer=scorer, pair_counts=pairs))
    assert result['model_calls'] == 7 and len(result['incomplete_candidates']) == 2
    row = result['candidates'][0]
    assert row['score_complete'] and row['gnn_path_score'] > 0 and row['structural_score'] > 0
    assert row['final_score'] == pytest.approx(0.4 + 0.2 * row['gnn_path_score'] + 0.2 * row['structural_score'] + 0.16)
    merged = ranking.attach_pool_ranking(result, screened)['pool_ranking']
    assert merged['reviewed_in_pool'] == 1 and len(merged['ranking']) == 12
    assert sum(r['final_score'] is None for r in merged['ranking']) == 11


def test_later_review_failure_keeps_completed_candidate_and_no_fabricated_consensus(tmp_path):
    layer, pool = make_pool(tmp_path)
    def call(system, user, label):
        if label == 'H2 response:scientific_delta':
            raise RuntimeError('Synthetic transport unavailable')
        return reviewer({'H1': 'substantive_extension', 'H2': 'substantive_extension'})(system, user, label)
    result = gate_candidates(pool['candidates'][:3], 'balanced', call,
        require_chain=True, peer_response=True, max_model_calls=21,
        resolver=lambda q: resolve_graph_evidence(q, layer=layer))
    assert result['selected_ids'] == ['H1'] and result['reviewed_candidates'] == 1
    unfinished = result['incomplete_candidates'][0]
    assert unfinished['hypothesis_id'] == 'H2' and len(unfinished['initial_reviews']) == 3
    assert len(unfinished['peer_responses']) == 1
    assert result['incomplete_candidates'][1]['review_status'] == 'not_reviewed'
    assert result['model_calls'] == 12  # No blind retry of a transport failure.


def test_no_budget_no_model_calls_and_forged_score_is_ignored(tmp_path):
    layer, pool = make_pool(tmp_path)
    c = {**pool['candidates'][0], 'gnn_path_score': 1, 'structural_score': 1}
    result = gate_candidates([c], 'balanced', lambda *a: pytest.fail('No review authorized'),
        peer_response=True, max_model_calls=6, require_chain=True,
        resolver=lambda q: resolve_graph_evidence(q, layer=layer))
    assert result['selected_ids'] == [] and result['model_calls'] == 0
    reviewed = gate_candidates([c], 'balanced', reviewer({'H1': 'substantive_extension'}), require_chain=True,
        resolver=lambda q: resolve_graph_evidence(q, layer=layer))
    assert reviewed['candidates'][0]['gnn_path_score'] is None


def test_peer_reply_cannot_introduce_unsupported_citation(tmp_path):
    layer, pool = make_pool(tmp_path)
    def call(system, user, label):
        refs = ('PMID:invented',) if 'response:' in label else ('PMID:1',)
        return reviewer({'H1': 'substantive_extension'}, references=refs)(system, user, label)
    with pytest.raises(NoveltyGateError, match='outside'):
        gate_candidates(pool['candidates'][:1], 'balanced', call, require_chain=True, peer_response=True,
                        resolver=lambda q: resolve_graph_evidence(q, layer=layer))


def test_stale_checkpoint_binding_is_unavailable_before_loading(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(graph_revision='old')), encoding='utf-8')
    monkeypatch.setenv('NEUROCLAW_IDEA_GNN_MANIFEST', str(manifest))
    scorer, receipt = ranking.configured_scorer(SimpleNamespace(), {'graph_revision': 'current'})
    assert scorer is None and 'different graph' in receipt['reason']


@pytest.mark.parametrize('returned', [[], [0.8, 0.8]])
def test_incomplete_scorer_output_cannot_produce_a_score(returned):
    scorer = SimpleNamespace(ent2idx={'A': 0, 'B': 1}, rel2idx={'r': 0}, score_batch=lambda _: returned)
    result = hypothesis_path.score_candidate({'kg_triples': [dict(source='A', relation='r', target='B')]}, scorer=scorer)
    assert result['gnn_path_score'] is None and not result['measured']


def test_rank_tool_rejects_changed_sealed_index_even_with_cached_pool(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROCLAW_IDEA_GNN_MANIFEST', raising=False)
    layer, pool = make_pool(tmp_path)
    ranking.prepare_ranking(layer, 'Marker')
    # Use the same existing mini-graph seal negative control as problem 2.
    db = next(tmp_path.glob('*.sqlite'))
    with db.open('ab') as stream:
        stream.write(b'changed')
    result = ranking.run_ranking_tool(layer, {'topic': 'Marker'}, threading.Event())
    assert not result['success']


def test_runtime_ranks_pool_reviews_shortlist_and_saves_all_rows(tmp_path, monkeypatch):
    from core.agent.test_autoresearch_execution import session, reply, tool, completion
    from core.agent.test_novelty_gate import _ReviewerClient
    monkeypatch.delenv('NEUROCLAW_IDEA_GNN_MANIFEST', raising=False)
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS', '40')
    graph_dir = tmp_path / 'graph'
    graph_dir.mkdir()
    layer, pool = make_pool(graph_dir)
    c = pool['candidates'][0]
    agent, calls = session(tmp_path, [
        reply(calls=[tool('rank_idea_hypotheses', topic='Marker')]),
        reply(calls=[tool('write_research_file', path='candidates.json', kind='hypotheses', content=json.dumps([c]))]),
        reply(calls=[tool('finish_autoresearch', **completion(artifacts=['candidates.json'], evidence_ids=[1, 2]))]),
    ], mode='idea')
    agent._idea_claim_layer = layer
    agent.env.update(autoresearch_novelty_gate=True, autoresearch_independent_review=False)
    reviews = []
    def create(**kwargs):
        reviews.append(kwargs)
        return reply(reviewer({'H1': 'substantive_extension'})(kwargs['messages'][0]['content'],
                      kwargs['messages'][1]['content'], 'H1 synthetic'))
    agent._review_llm = _ReviewerClient(create)
    agent._chat()
    state = agent.autoresearch_state
    assert state['status'] == 'completed' and state['chain_validation']['passed']
    assert len(reviews) == 5 and len(calls) == 3
    assert len(state['idea_pool_ranking']['ranking']) == 12
    assert state['novelty_selection']['pool_ranking']['reviewed_in_pool'] == 1
    row = state['novelty_selection']['candidates'][0]
    assert row['gnn_path_score'] is None and row['structural_score'] is not None
    assert not row['score_complete']
    assert any(t['function']['name'] == 'rank_idea_hypotheses' for t in calls[0]['tools'])
