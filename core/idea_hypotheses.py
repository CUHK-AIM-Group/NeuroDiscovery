"""Topic evidence -> exact typed graph chains -> conditional hypothesis drafts.

No model calls, graph changes or scores. The existing evidence reader owns
provenance, TaskChain owns shape and HypothesisEngine owns path validation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import sys
from pathlib import Path

from neurooracle.src.atoms import CANONICAL_CHAINS
from neurooracle.src.hypothesis_engine import HypothesisEngine

CONTRACT = 'idea.typed_chain.v1'
ASSOCIATIONS = ('is_associated_with', 'associated_with', 'correlates_with')
CHANGES = (*ASSOCIATIONS, 'increases', 'reduces', 'modulates', 'causes', 'activates', 'inhibits')
PREDICTS = (*ASSOCIATIONS, 'predicts', 'is_biomarker_of', 'is_risk_factor_for')
# Explicit allowed predicates; graph observations never define their own rules.
STEP_RULES = {
    'genetic_imaging_disease': ((CHANGES + ('gene_associated_with_anatomy',), PREDICTS), ('forward', 'forward')),
    'drug_imaging_outcome': ((CHANGES, PREDICTS + ('mediates',)), ('forward', 'forward')),
    'task_brain_behavior': ((CHANGES + ('evokes', 'elicits'), PREDICTS), ('forward', 'forward')),
    'disease_biomarker_prognosis': ((ASSOCIATIONS, PREDICTS), ('either', 'forward')),
    'pathway_polygenic_mediation': ((CHANGES, PREDICTS), ('forward', 'forward')),
}
TEMPLATES = {c.name: replace(c, relations=STEP_RULES[c.name][0], directions=STEP_RULES[c.name][1])
             for c in CANONICAL_CHAINS if c.name in STEP_RULES}
REVISION_KEYS = ('graph_revision', 'claim_layer_revision', 'original_index_revision')
DEFAULT_POOL_SIZE = 500
MAX_POOL_SIZE = 2000

IDEA_CHAIN_TOOL_NAME = 'generate_idea_hypotheses'
IDEA_CHAIN_TOOL = {'type': 'function', 'function': {
    'name': IDEA_CHAIN_TOOL_NAME,
    'description': 'Read the current graph for a topic and return bounded, validated typed-chain hypothesis drafts '
                   'with per-edge evidence, conditions, predictions and limitations. No graph writes or scores. '
                   'Keep chain and graph_revision unchanged when refining prose; empty candidates explain evidence gaps.',
    'parameters': {'type': 'object', 'properties': {
        'topic': {'type': 'string'},
        'templates': {'type': 'array', 'items': {'type': 'string', 'enum': list(TEMPLATES)}},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': MAX_POOL_SIZE, 'default': DEFAULT_POOL_SIZE,
                  'description': 'Candidate pool size, separate from the final review shortlist.'},
        'offset': {'type': 'integer', 'minimum': 0},
        'page_size': {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 20},
    }, 'required': ['topic']},
}}


def template_for(candidate):
    chain = candidate.get('chain')
    if not isinstance(chain, dict) or chain.get('contract') != CONTRACT:
        raise ValueError('Idea hypotheses require a validated typed graph chain')
    template = TEMPLATES.get(chain.get('template'))
    if template is None or chain.get('template_spec') != template.to_dict():
        raise ValueError('Unknown or altered chain template')
    return template


def validate_candidate_shape(candidate):
    """Draft validation only; graph binding is separately required before review."""
    template = template_for(candidate)
    chain = candidate['chain']
    nodes, links = chain.get('node_ids'), chain.get('links')
    if (not isinstance(nodes, list) or len(nodes) != len(template.chain) or
            any(not isinstance(n, str) or not n for n in nodes) or len(set(nodes)) != len(nodes) or
            not isinstance(links, list) or len(links) != len(nodes) - 1):
        raise ValueError('Incomplete or repeated-node chain')
    if chain.get('node_types') != [a.value for a in template.chain]:
        raise ValueError('Chain node types differ from template')
    for i, link in enumerate(links):
        if not isinstance(link, dict) or not str(link.get('claim_id', '')).startswith('CLM:'):
            raise ValueError('Each chain edge needs a graph claim ID')
        if link.get('relation') not in template.relations[i]:
            raise ValueError('Chain predicate differs from template')
        pair = [link.get('source'), link.get('target')]
        allowed = []
        if template.directions[i] in {'forward', 'either'}:
            allowed.append(nodes[i:i + 2])
        if template.directions[i] in {'reverse', 'either'}:
            allowed.append(nodes[i:i + 2][::-1])
        if pair not in allowed:
            raise ValueError('Chain edge direction or continuity differs from template')
    if len({e['claim_id'] for e in links}) != len(links):
        raise ValueError('Repeated chain observation')
    if not candidate.get('graph_revision'):
        raise ValueError('A graph revision is required for typed chains')


def records_from_documents(documents):
    """Keep each complete, resolved observation and its own paper identity."""
    records = {}
    for document in documents:
        for paper in document.get('papers', []):
            for obs in paper.get('observations', []):
                # Source-derived supplements support a dossier but are not
                # original graph edges and cannot become traversable CLM IDs.
                md = obs.get('original_claim') or {}
                cid = obs.get('claim_id')
                if not cid or md.get('id') != cid:
                    continue
                row = dict(metadata=deepcopy(md), observation=deepcopy(obs),
                           bibliography=deepcopy(paper.get('bibliography') or {}),
                           paper_key=paper.get('paper_key'), work_key=paper.get('work_key'),
                           verified=paper.get('verified'), publication_review=deepcopy(paper.get('publication_review')))
                if cid in records and records[cid] != row:
                    raise ValueError('Conflicting resolved versions of a chain observation')
                records[cid] = row
    return records


def _value(record, key):
    md = record['metadata']
    def known(value):
        if isinstance(value, dict):
            return any(known(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(known(v) for v in value)
        return value is not None and value != ''
    for container in (md, md.get('metadata') or {}, record['observation']):
        if known(container.get(key)):
            return container[key]
    if key == 'direction':
        # Effect direction is commonly retained inside evidence, independently
        # of the directed predicate. Read it without inferring causality/sign.
        return (md.get('evidence') or record['observation'].get('evidence') or {}).get(key)
    return None


def chain_context(rows):
    keys = ('population', 'conditions', 'species', 'disease_stage', 'measurement', 'time',
            'comparator', 'direction', 'qualifiers', 'interaction', 'moderator')
    edges, limits, conflicts = [], [], []
    for row in rows:
        obs, md = row['observation'], row['metadata']
        context = {key: _value(row, key) for key in keys}
        review = obs.get('source_review') or {}
        edges.append(dict(claim_id=md['id'], source=md['subject_id'], relation=md['predicate'], target=md['object_id'],
                          context=context, negated=obs.get('negated', md.get('negated')),
                          proposition_support=obs.get('proposition_support'), source_reviewed=bool(review),
                          source_anchor=review.get('source_anchor'), scope_note=review.get('scope_note'),
                          **deepcopy(row)))
        if not row['verified'] or not review:
            limits.append(md['id'] + ': source identity or observation has not been reviewed.')
        missing = [key for key in ('population', 'conditions', 'measurement', 'time') if not context[key]]
        if missing:
            limits.append(md['id'] + ': unknown ' + ', '.join(missing) + '.')
    for key in ('population', 'species', 'disease_stage', 'measurement', 'time', 'conditions'):
        values = [edge['context'][key] for edge in edges]
        known = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values if value not in (None, '', [], {})}
        if len(known) > 1:
            conflicts.append(dict(field=key, values=values, meaning='Different recorded scopes; compatibility not established.'))
    counterevidence = [e['claim_id'] for e in edges if e['negated'] is True or e['proposition_support'] in {'does_not_support', 'contradicts'}]
    limits.append('Separate graph observations do not establish the whole-chain mechanism, causality, independence or novelty.')
    return dict(edges=edges, scope_differences=conflicts, counterevidence=counterevidence, limitations=limits)


def _candidate(template, nodes, rows, revision, topic):
    mds = [row['metadata'] for row in rows]
    HypothesisEngine.validate_typed_path(template, nodes, mds)
    names = {md[side + '_id']: md[side + '_name'] for md in mds for side in ('subject', 'object')}
    names = [names[n] for n in nodes]
    links = [dict(source=md['subject_id'], relation=md['predicate'], target=md['object_id'], claim_id=md['id']) for md in mds]
    context = chain_context(rows)
    context['bridge_bindings'] = [dict(claim_id=md['id'], node_id=nodes[1],
        label=md['subject_name'] if md['subject_id'] == nodes[1] else md['object_name'],
        measurement=edge['context']['measurement'], population=edge['context']['population'],
        time=edge['context']['time'], relation=md['predicate']) for md, edge in zip(mds, context['edges'])]
    bridge_labels = list(dict.fromkeys(b['label'] for b in context['bridge_bindings']))
    if len({label.strip().casefold() for label in bridge_labels}) > 1:
        # One concept ID does not license replacing the first paper's measured
        # variable with the last paper's wording. A name difference is a review
        # question, not by itself evidence that the two measures are incompatible.
        names[1] = ('a proposed common measurement described separately as ' +
                    ' and '.join(json.dumps(label, ensure_ascii=False) for label in bridge_labels) +
                    ' (measurement equivalence remains unverified)')
        context['limitations'].append('The two source labels differ; preserve both and verify the actual measured variable before composing their relations.')
    context['review_requirements'] = [
        'Verify that both uses of the middle node refer to the same specific measurement, not just the same ID.',
        'Check each claimed relation against its source, including null results and main-effect versus moderation.',
        'Specify population, comparator and time; any cross-scope transfer remains a proposed test.',
    ]
    sources = []
    for row in rows:
        b = row['bibliography']
        source = ('PMID:' + str(b['pmid']) if b.get('pmid') else 'DOI:' + b['doi'] if b.get('doi') else row['paper_key'])
        if source and source not in sources:
            sources.append(source)
    population = [_value(row, 'population') for row in rows]
    scope = str(population[0]) if population[0] and all(v == population[0] for v in population) else 'a population and measurement setting to be specified from the source evidence'
    first, mediator, last = names
    rationale_parts = []
    for md, edge in zip(mds, context['edges']):
        excerpt = edge['source_anchor'] or edge['observation'].get('raw_text') or 'No source text retained; metadata only.'
        rationale_parts.append(f"{md['subject_name']} --[{md['predicate']}]--> {md['object_name']} ({md['id']}). "
                               f"Recorded evidence: {excerpt} Context: " + json.dumps(edge['context'], ensure_ascii=False))
    rationale = '\n'.join(rationale_parts)
    result = dict(
        candidate_id='HYP:CHAIN:' + hashlib.sha256(json.dumps([template.name, nodes, [md['id'] for md in mds]], sort_keys=True).encode()).hexdigest()[:16],
        topic=topic, **revision,
        chain=dict(contract=CONTRACT, template=template.name, template_spec=template.to_dict(),
                   node_ids=nodes, node_types=[a.value for a in template.chain], links=links),
        kg_triples=[{k: edge[k] for k in ('source', 'relation', 'target')} for edge in links],
        hypothesis=f'In {scope}, the relationship between {first} and {last} may involve {mediator}; this joint relationship remains to be tested.',
        rationale=rationale,
        prediction=f'In one independent dataset with a prespecified population, measurements and time window, test both recorded links and whether {mediator} adds out-of-sample information about {last} beyond {first} alone. Failure to reproduce either link or the prespecified improvement does not support this joint predictive hypothesis; this test does not establish causal mediation.',
        proposed_inference=dict(status='untested_joint_prediction', source=nodes[0], mediator=nodes[1], target=nodes[-1],
                                observed_links_establish_full_chain=False),
        source_ids=sources, evidence_queries=[{'claim_id': md['id']} for md in mds],
        chain_context=context, limitations='\n'.join(context['limitations']), counterevidence=context['counterevidence'],
        structural_validation='passed', scientific_validation='not_established',
        status='scope_review_required' if context['scope_differences'] or context['counterevidence'] else 'provisional')
    validate_candidate_shape(result)
    return result


def bind_candidate_chain(candidate, documents, revision, *, endpoint_nodes):
    """Independently rebind shape and context before sending any reviewer calls."""
    validate_candidate_shape(candidate)
    if any(endpoint_nodes.get(nid, {}).get('id') != nid for nid in candidate['chain']['node_ids']):
        raise ValueError('Chain endpoint is absent from the actual graph concepts')
    for key in REVISION_KEYS:
        if candidate.get(key) != revision.get(key):
            raise ValueError('Typed chain belongs to a different graph/evidence revision')
    records = records_from_documents(documents)
    rows = []
    for link in candidate['chain']['links']:
        row = records.get(link['claim_id'])
        if row is None:
            raise ValueError('Chain observation absent from resolved evidence')
        md = row['metadata']
        if (link['source'], link['relation'], link['target']) != (md.get('subject_id'), md.get('predicate'), md.get('object_id')):
            raise ValueError('Chain triple differs from its actual graph observation')
        rows.append(row)
    template = template_for(candidate)
    HypothesisEngine.validate_typed_path(template, candidate['chain']['node_ids'], [r['metadata'] for r in rows])
    bound = _candidate(template, candidate['chain']['node_ids'], rows, revision, candidate.get('topic', ''))
    if candidate.get('candidate_id', bound['candidate_id']) != bound['candidate_id']:
        raise ValueError('Candidate ID differs from its validated chain')
    if candidate.get('kg_triples', bound['kg_triples']) != bound['kg_triples']:
        raise ValueError('Declared triples differ from the validated chain')
    return {key: bound[key] for key in ('chain', 'kg_triples', 'chain_context', 'proposed_inference',
                                        'structural_validation', 'scientific_validation', 'status', *REVISION_KEYS) if key in bound}


def generate_hypotheses(layer, topic, *, templates=None, limit=DEFAULT_POOL_SIZE):
    from core.topic_evidence import query_groups, _matchers, _expand
    if type(limit) is not int or not 1 <= limit <= MAX_POOL_SIZE:
        raise ValueError(f'Generate 1 to {MAX_POOL_SIZE} hypothesis drafts')
    if not isinstance(topic, str) or not topic.strip() or len(topic) > 500:
        raise ValueError('Provide a topic of 1 to 500 characters')
    matchers = _matchers(query_groups(topic))
    if not matchers:
        raise ValueError('Provide at least one searchable topic term')
    names = list(TEMPLATES) if templates is None else templates
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or name not in TEMPLATES for name in names):
        raise ValueError('Choose existing typed-chain templates')
    selected = [TEMPLATES[name] for name in dict.fromkeys(names)]
    with layer.read_snapshot() as revision:
        cache_key = (topic, tuple(names), limit, json.dumps(revision, sort_keys=True))
        cached = getattr(layer, '_idea_pool_cache', None)
        if cached and cached[0] == cache_key:
            return deepcopy(cached[1])
        index = layer.chain_index(predicates={p for t in selected for step in t.relations for p in step})
        if not index['available']:
            raise ValueError('The accepted original-claim index is required for candidate-pool generation')
        minimum = max(1, math.ceil(len(matchers) * 0.5))
        diagnoses = {term for term, _ in matchers} & {'mci', 'alzheimer', 'adhd', 'schizophrenia', 'epilepsy', 'parkinson'}
        def hits(text):
            return {term for term, pattern in matchers if pattern.search(text)}
        # Reuse the shared catalog's bibliography/alias matches as additional
        # anchors, without turning supplemental source IDs into graph edges.
        shared_hits = {}
        for summary, text in layer.claim_index():
            terms = hits(text)
            if len(terms) >= minimum and diagnoses <= terms:
                for cid in summary['original_claim_ids']:
                    shared_hits[cid] = terms
        matched = {}
        for md in index['records']:
            terms = hits(' '.join(str(md.get(k) or '') for k in ('subject_name', 'object_name', 'predicate', 'pmid', 'doi')))
            terms |= shared_hits.get(md['id'], set())
            if terms:
                matched[md['id']] = terms
        anchors = {cid for cid, terms in matched.items() if len(terms) >= minimum and diagnoses <= terms}
        anchor_sources = {md.get('source_key') or md.get('pmid') or md['id'] for md in index['records'] if md['id'] in anchors}
        # Two-hop templates need only anchors and their direct neighbors, but
        # every indexed matching neighbor participates; there is no first-64 cap.
        anchor_nodes = {md[side + '_id'] for md in index['records'] if md['id'] in anchors for side in ('subject', 'object')}
        eligible = [md for md in index['records'] if md['subject_id'] in anchor_nodes or md['object_id'] in anchor_nodes]
        paths, counts, seen, truncated = [], {}, set(), False
        path_budget = min(10000, max(1000, limit * 2))
        for template in selected:
            found = HypothesisEngine.enumerate_typed_paths(template, eligible, anchor_claim_ids=anchors, limit=path_budget + 1)
            counts[template.name] = min(len(found), path_budget)
            truncated |= len(found) > path_budget
            for nodes, mds in found[:path_budget]:
                key = (tuple(nodes), tuple((md['subject_id'], md['predicate'], md['object_id']) for md in mds))
                if key in seen:
                    continue
                seen.add(key)
                terms = set().union(*(matched.get(md['id'], set()) for md in mds))
                paths.append((template, nodes, mds, terms))
        paths.sort(key=lambda row: (-len(row[3]), row[0].name, row[1], [md['id'] for md in row[2]]))
        candidates, held, resolved, missing_nodes, examined, irrelevant, held_seen = [], [], {}, 0, 0, 0, 0
        endpoint_nodes, looked_up = {}, set()
        for start in range(0, len(paths), 64):
            if len(candidates) >= limit:
                break
            batch_paths = paths[start:start + 64]
            ids = list(dict.fromkeys(md['id'] for _, _, mds, _ in batch_paths for md in mds if md['id'] not in resolved))
            for pos in range(0, len(ids), 128):
                resolved.update(records_from_documents(_expand(layer, ids[pos:pos + 128], revision).values()))
            node_ids = sorted({nid for _, nodes, _, _ in batch_paths for nid in nodes} - looked_up)
            for pos in range(0, len(node_ids), 128):
                endpoint_nodes.update(layer.graph_nodes(node_ids[pos:pos + 128]))
            looked_up.update(node_ids)
            for template, nodes, mds, terms in batch_paths:
                examined += 1
                if any(endpoint_nodes.get(nid, {}).get('id') != nid for nid in nodes):
                    missing_nodes += 1
                    continue
                rows = [resolved[md['id']] for md in mds]
                # A shared dossier's bibliography may mention a different
                # member's condition. Require topic hits in these own records.
                terms = hits(json.dumps([[row['metadata'], row['bibliography']] for row in rows], ensure_ascii=False))
                if len(terms) < minimum or not diagnoses <= terms:
                    irrelevant += 1
                    continue
                candidate = _candidate(template, nodes, rows, revision, topic)
                candidate['topic_match'] = dict(matched_terms=sorted(terms), missing_terms=[t for t, _ in matchers if t not in terms],
                                               coverage=round(len(terms) / len(matchers), 4), semantic_relevance_verified=False)
                if candidate['status'] == 'scope_review_required':
                    held_seen += 1
                    if len(held) < limit:
                        held.append(candidate)
                elif len(candidates) < limit:
                    candidates.append(candidate)
        result = dict(contract=CONTRACT, topic=topic, **revision, templates=[t.to_dict() for t in selected],
                    candidates=candidates, held_chains=held, template_path_counts=counts,
                    requested_limit=limit, candidate_count=len(candidates), held_count=len(held),
                    retrieval=dict(indexed_claims=index['indexed'], typed_index_records=len(index['records']),
                                   anchor_claims=len(anchors), anchor_source_count=len(anchor_sources), original_observations=len(eligible),
                                   unique_paths_found=len(paths), paths_examined=examined, missing_node_paths=missing_nodes,
                                   topic_mismatch_paths=irrelevant, held_paths_examined=held_seen,
                                   path_search_truncated=truncated, pool_truncated=examined < len(paths), held_truncated=held_seen > len(held)),
                    limitations=['Only declared endpoint types and exact claim-backed edges are traversed.',
                                 'All matching indexed neighbors participate; template/path/output bounds still apply.',
                                 'Repeated paper combinations for the same nodes and predicates are not counted as new hypotheses.',
                                 'Lexical topic coverage and template counts are not semantic relevance or scientific quality scores.',
                                 'Postprocessing preserves recorded context; missing facts are not supplied by a model.',
                                 'Drafts propose a joint predictive test; causal mediation and novelty remain unestablished.'],
                    gap=None if candidates else 'No scope-compatible, fully typed chain in this indexed template search; do not invent nodes or relations.')
        layer._idea_pool_cache = (cache_key, deepcopy(result))
        return result


def run_chain_tool(layer, arguments, cancel):
    if cancel.is_set():
        return dict(success=False, executed=False, error_type='cancelled')
    try:
        offset, page_size = arguments.get('offset', 0), arguments.get('page_size', 20)
        if type(offset) is not int or offset < 0 or type(page_size) is not int or not 1 <= page_size <= 50:
            raise ValueError('Use a nonnegative offset and page_size from 1 to 50')
        result = generate_hypotheses(layer, arguments.get('topic'), templates=arguments.get('templates'), limit=arguments.get('limit', DEFAULT_POOL_SIZE))
        if cancel.is_set():
            return dict(success=False, executed=False, error_type='cancelled')
        page = result['candidates'][offset:offset + page_size]
        return dict(success=True, scientifically_validated=False, **{k: v for k, v in result.items() if k not in {'candidates', 'held_chains'}},
                    candidates=page, held_chains=result['held_chains'][offset:offset + page_size], offset=offset, page_size=page_size,
                    next_offset=offset + page_size if offset + page_size < max(result['candidate_count'], result['held_count']) else None)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        return dict(success=False, executed=False, error_type='chain_generation_failed', error=str(exc))


def main(argv=None):
    from core.topic_evidence import default_layer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--template', action='append', choices=list(TEMPLATES))
    parser.add_argument('--limit', type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    result = generate_hypotheses(default_layer(), args.topic, templates=args.template, limit=args.limit)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(content + '\n')
    try:
        print(content)
    except UnicodeEncodeError:
        # The payload is UTF-8 JSON; a non-UTF-8 console (e.g. GBK) must not
        # fail the run after the result has already been written.
        sys.stdout.buffer.write(content.encode('utf-8', 'replace') + b'\n')


if __name__ == '__main__':
    main()
