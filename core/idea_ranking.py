"""Read-only pool screening and measured score inputs for the existing idea gate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

from core.idea_hypotheses import (IDEA_CHAIN_TOOL, REVISION_KEYS, generate_hypotheses,
                                  validate_candidate_shape)
from neurooracle.src.kge.hypothesis_path import score_candidate, structural_score

RANK_TOOL_NAME = 'rank_idea_hypotheses'
RANK_TOOL = deepcopy(IDEA_CHAIN_TOOL)
RANK_TOOL['function'].update(name=RANK_TOOL_NAME, description=(
    'Screen the entire generated topic pool using lexical coverage, measured GNN edge scores '
    'when configured, and graph-only endpoint exposure. Returns a ranked page; keeps all '
    'candidates, missing scores and held chains. No model calls. Submit a small shortlist '
    'for the runtime three-reviewer discussion and final novelty ranking (7 calls per candidate).'))


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configured_scorer(layer, revision):
    """Explicit, hash-bound GNN inference only; no training or artifact auto-discovery.

    The operator's manifest binds a checkpoint and its message graph to the current
    KG revision. This verifies identity, not scientific usefulness or training quality.
    """
    setting = os.environ.get('NEUROCLAW_IDEA_GNN_MANIFEST', '').strip()
    if not setting:
        default = Path(__file__).resolve().parents[1] / 'neurooracle/configs/idea_gnn.json'
        setting = str(default) if default.is_file() else ''
    if not setting:
        return None, dict(status='unavailable', reason='No GNN manifest configured for the current graph.')
    try:
        path = Path(setting).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get('graph_revision') != revision['graph_revision']:
            raise ValueError('GNN manifest belongs to a different graph revision')
        if not isinstance(manifest.get('training_scope'), str) or not manifest['training_scope'].strip():
            raise ValueError('GNN manifest must disclose its training scope')
        binding = hashlib.sha256(raw).hexdigest()
        cached = getattr(layer, '_idea_gnn_cache', None)
        kind = manifest.get('kind', 'gnn_checkpoint')
        if kind in {'graphsage_export', 'rgcn_export'}:
            from models.kg_link_prediction.scorer import FrozenGraphSAGEScorer, FrozenRGCNScorer
            record = manifest['export']
            export = (path.parent / record['path']).resolve()
            if cached and cached[0] == binding:
                cached[1].verify_files()
                return cached[1:]
            loader = FrozenGraphSAGEScorer if kind == 'graphsage_export' else FrozenRGCNScorer
            scorer = loader.load(export, record['sha256'])
            if path.read_bytes() != raw:
                raise ValueError('GNN configuration changed while loading')
            receipt = dict(status='available', manifest_sha256=binding, graph_revision=revision['graph_revision'],
                           training_scope=manifest['training_scope'], name=scorer.name, **scorer.provenance)
            receipt['trained_on_current_graph'] = receipt['training_graph_revision'] == revision['graph_revision']
            layer._idea_gnn_cache = (binding, scorer, receipt)
            return scorer, receipt
        if kind != 'gnn_checkpoint':
            raise ValueError('Unknown GNN checkpoint/export kind')
        files = {}
        for key in ('checkpoint', 'message_graph'):
            record = manifest[key]
            source = (path.parent / record['path']).resolve()
            if _sha(source) != record['sha256']:
                raise ValueError(f'{key} fingerprint changed')
            files[key] = source
        if cached and cached[0] == binding:
            return cached[1:]
        from models.kg_link_prediction.scorer import GNNEdgeScorer
        graph = json.loads(files['message_graph'].read_text(encoding='utf-8'))
        if (not isinstance(graph, list) or not graph or any(
                not isinstance(row, list) or len(row) != 3 or
                any(not isinstance(value, str) or not value for value in row) for row in graph)):
            raise ValueError('message_graph must be a nonempty JSON list of ID triples')
        scorer = GNNEdgeScorer.load(files['checkpoint'], [tuple(row) for row in graph], device='cpu')
        # Recheck after loading, before caching any inference result.
        if path.read_bytes() != raw or any(_sha(files[k]) != manifest[k]['sha256'] for k in files):
            raise ValueError('GNN inputs changed while loading')
        receipt = dict(status='available', manifest_sha256=binding,
                       checkpoint_sha256=manifest['checkpoint']['sha256'],
                       message_graph_sha256=manifest['message_graph']['sha256'],
                       graph_revision=revision['graph_revision'], training_scope=manifest['training_scope'],
                       name=scorer.name, architecture=scorer.model.model_name, decoder=scorer.model.decoder_name,
                       direction_sensitive=scorer.model.decoder is not None, scientific_usefulness_validated=False)
        layer._idea_gnn_cache = (binding, scorer, receipt)
        return scorer, receipt
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        return None, dict(status='unavailable', reason=f'{type(exc).__name__}: {exc}')


def _unit(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1 else None)


def measure_candidate(candidate, *, scorer=None, pair_counts=None, scorer_receipt=None):
    """Use only a generated or independently rebound chain, never candidate self-scores."""
    validate_candidate_shape(candidate)
    edges = candidate['chain_context']['edges']
    confidences = [_unit(edge['metadata'].get('confidence')) for edge in edges]
    confidence = (math.prod(confidences) ** (1 / len(edges))
                  if all(value is not None for value in confidences) else None)
    traceability = sum(bool(e.get('paper_key') and (e.get('source_anchor') or
                         e['observation'].get('raw_text'))) for e in edges) / len(edges)
    pair = tuple(sorted((candidate['chain']['node_ids'][0], candidate['chain']['node_ids'][-1])))
    count = pair_counts.get(pair) if pair_counts is not None else None
    graph_novelty = (0.5 * (1 - 1 / len(edges)) + 0.5 / (1 + math.log1p(count))
                     if type(count) is int and count >= 0 else None)
    # Reuse the existing geometric edge-score reducer; never guess entity mappings.
    scores = score_candidate(candidate, scorer=scorer)
    return dict(gnn_path_score=scores['gnn_path_score'], structural_score=None,
                components=dict(confidence=confidence, traceability=traceability,
                                graph_only_novelty=graph_novelty, testability=None),
                edge_scores=scores['edge_scores'], endpoint_pair_source_count=count,
                gnn=scorer_receipt or dict(status='unavailable' if scorer is None else 'provided'),
                gnn_missing_reasons=[s for s in scores['unresolved'] if
                                     'vocabulary' in s or 'predictor' in s or 'score' in s],
                scope='GNN measures existing edges, not the untested whole chain. Endpoint exposure is graph-only, '
                      'not literature novelty. Traceability is a retained source anchor, not source verification.')


def finish_measurement(measurement, adjudication):
    result = deepcopy(measurement)
    components = result['components']
    components['testability'] = float(adjudication['executable_test_covers_delta'] and
                                      adjudication['registered_test_matches_hypothesis'])
    result['structural_score'], result['structural_missing_components'] = structural_score(components)
    return result


def screen_pool(pool, *, scorer=None, pair_counts=None, scorer_receipt=None):
    """Rank all drafts for review, retaining a distinct unreviewed status and stable IDs."""
    rows = []
    seen = set()
    for candidate in pool['candidates']:
        hid = candidate['candidate_id']
        if hid in seen:
            raise ValueError('Duplicate candidate ID in the hypothesis pool')
        seen.add(hid)
        if any(candidate.get(key) != pool.get(key) for key in REVISION_KEYS):
            raise ValueError('Mixed graph revisions in the hypothesis pool')
        measured = measure_candidate(candidate, scorer=scorer, pair_counts=pair_counts, scorer_receipt=scorer_receipt)
        rows.append(dict(candidate_id=hid, template=candidate['chain']['template'],
                         topic_coverage=candidate.get('topic_match', {}).get('coverage', 0.0),
                         measurement=measured, review_status='not_reviewed', final_score=None, selected=False))
    # Compare the GNN signal only if the WHOLE pool is covered; OOV must not be
    # interpreted as a bad score or automatically excluded from review.
    use_gnn = bool(rows) and all(row['measurement']['gnn_path_score'] is not None for row in rows)
    def order(row):
        m = row['measurement']
        return (-row['topic_coverage'], -(m['gnn_path_score'] if use_gnn else 0),
                -(m['components']['graph_only_novelty'] or 0),
                -(m['components']['confidence'] or 0), row['candidate_id'])
    rows.sort(key=order)
    for index, row in enumerate(rows, 1):
        row['screen_rank'] = index
    return dict(topic=pool['topic'], **{k: pool.get(k) for k in REVISION_KEYS},
                pool_count=len(rows), held_count=pool['held_count'], ranking=rows,
                held_ids=[c['candidate_id'] for c in pool['held_chains']],
                gnn_covered_count=sum(row['measurement']['gnn_path_score'] is not None for row in rows),
                gnn_used_for_screening=use_gnn, gnn=scorer_receipt,
                scope='Pre-review ordering only: lexical topic coverage, GNN if full coverage, graph endpoint exposure, '
                      'recorded edge confidence. Unreviewed rows have no literature novelty or final score.')


def prepare_ranking(layer, topic, *, templates=None, limit=500):
    with layer.read_snapshot() as revision:
        pool = generate_hypotheses(layer, topic, templates=templates, limit=limit)
        scorer, receipt = configured_scorer(layer, revision)
        pairs = [(c['chain']['node_ids'][0], c['chain']['node_ids'][-1]) for c in pool['candidates']]
        pair_counts = layer.pair_support_counts(pairs)
        result = screen_pool(pool, scorer=scorer, pair_counts=pair_counts, scorer_receipt=receipt)
    layer._idea_ranking = deepcopy(result)
    return result, pool


def run_ranking_tool(layer, arguments, cancel):
    if cancel.is_set():
        return dict(success=False, executed=False, error_type='cancelled')
    try:
        offset, size = arguments.get('offset', 0), arguments.get('page_size', 20)
        if type(offset) is not int or offset < 0 or type(size) is not int or not 1 <= size <= 50:
            raise ValueError('Use a nonnegative offset and page_size from 1 to 50')
        result, pool = prepare_ranking(layer, arguments.get('topic'), templates=arguments.get('templates'),
                                       limit=arguments.get('limit', 500))
        if cancel.is_set():
            return dict(success=False, executed=False, error_type='cancelled')
        page = result['ranking'][offset:offset + size]
        candidates = {c['candidate_id']: c for c in pool['candidates']}
        return dict(success=True, **{k: v for k, v in result.items() if k not in {'ranking', 'held_ids'}},
                    ranking=page, candidates=[candidates[row['candidate_id']] for row in page],
                    offset=offset, next_offset=offset + size if offset + size < len(result['ranking']) else None)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        return dict(success=False, executed=False, error_type='idea_ranking_failed', error=str(exc))


def attach_pool_ranking(result, pool):
    """Merge actual reviews by stable ID; never label unreviewed drafts rejected."""
    if not pool:
        return result
    if any(any(row.get(k) != pool.get(k) for k in REVISION_KEYS) for row in result['candidates']):
        raise ValueError('Ranking belongs to a different graph revision')
    rows = deepcopy(pool['ranking'])
    reviewed = {c['candidate_id']: c for c in result['candidates'] if c.get('candidate_id')}
    for row in rows:
        review = reviewed.get(row['candidate_id'])
        if review:
            row.update(review_status='reviewed', final_score=review['final_score'], selected=review['selected'],
                       novelty_priority_points=review['novelty_priority_points'], critic_score=review['critic_score'],
                       selection_reason=review['selection_reason'])
    result['pool_ranking'] = {**{k: v for k, v in pool.items() if k != 'ranking'}, 'ranking': rows,
                              'reviewed_in_pool': sum(r['review_status'] == 'reviewed' for r in rows)}
    return result
