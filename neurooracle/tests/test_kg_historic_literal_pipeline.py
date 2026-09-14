from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from apply_kg_historic_literal_repair import LiteralTransform
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from neurooracle.src import kg_historic_literal_repair as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import literal_node as imaging_literal_node
from neurooracle.tests.test_kg_historic_literal_repair import fixture, event_for, edges


def stream_fixture(science=True, reuse=False):
    row, gene, witness, review = fixture('frontal cortical surface area' if reuse else 'brain structural alterations')
    if reuse:
        gene.update(id='CUI:C1414531', preferred_name='FANCE', aliases=['FACE'])
        row['metadata']['subject_id'] = gene['id']; row['metadata']['metadata']['subject_id'] = gene['id']
        witness.update(node_id=gene['id'], name='FANCE', labels=['FANCE','FACE'], node_sha256=digest(gene))
        review.update(current_node_id=gene['id'], source_gene_node_sha256=digest(gene), word_interior_alias_hits=['face'], claim_sha256=digest(row))
    node = imaging_literal_node(review['name']) if reuse else repair.literal_node(review['name'])
    event, current = event_for(row, witness, review, node)
    refs = edges(row) if science else edges(row)[:2]
    plan = dict(events=[event], edge_events=repair.reviewed_edges(row['id'], row, current, refs),
        gene_witnesses={gene['id']: witness}, endpoint_reviews={repair.review_key(row['id'], 'subject'): review},
        new_literals=[] if reuse else [dict(id=node['id'], name=node['preferred_name'], node_sha256=digest(node))],
        existing_targets={gene['id']: digest(gene), **({node['id']: digest(node)} if reuse else {})})
    source = [('metadata', None, {}), ('node', gene['id'], gene),
        ('node', 'CUI:disease', dict(id='CUI:disease', preferred_name='anxiety')), ('node', row['id'], row)]
    if reuse: source.append(('node', node['id'], node))
    source.extend(('edge', i, edge) for i, edge in refs)
    return source, plan


@pytest.mark.parametrize('science', [True, False])
@pytest.mark.parametrize('reuse', [True, False])
def test_stream_identity_inverse_and_independent_graph(tmp_path, science, reuse):
    source, plan = stream_fixture(science, reuse); before = deepcopy(source)
    forward = LiteralTransform(plan); candidate = list(forward.records(iter(source)))
    assert source == before
    inverse = {k: hashlib.sha256() for k in ('nodes','edges')}
    new = {r['id'] for r in plan['new_literals']}; edge_events = {r['ordinal']: r for r in plan['edge_events']}
    for kind, key, row in candidate:
        if kind == 'metadata' or key in new: continue
        if key == 'CLM:test': row = repair.reverse_claim(row, plan['events'][0])
        if kind == 'edge' and key in edge_events: row = repair.apply_edge(row, edge_events[key], reverse=True)
        inverse[kind+'s'].update(compact(row).encode()+b'\n')
    assert {k:h.hexdigest() for k,h in inverse.items()} == {k:h.hexdigest() for k,h in forward.digests.items()}
    out = BytesIO(); catalog = tmp_path/'catalog.jsonl'
    result = stream_patch(iter(candidate), out, [], {}, catalog)
    graph = json.loads(out.getvalue())
    reread = [('metadata', None, graph['metadata']), *[('node', key, row) for key,row in graph['concepts'].items()],
        *[('edge',i,row) for i,row in enumerate(graph['edges'],1)]]
    assert verify_catalog_and_graph(iter(reread), result, catalog)['shared_relation_index_complete']
    assert result['counts'] == dict(nodes=4, claims=1, edges=3 if science else 2)


def test_no_reviewed_claim_can_be_added_by_adjusting_plan_only():
    source, plan = stream_fixture(); plan['endpoint_reviews'] = {}
    with pytest.raises(ValueError, match='finite_reviewed_scope'): list(LiteralTransform(plan).records(iter(source)))


def test_extra_owned_reference_and_changed_target_are_rejected():
    source, plan = stream_fixture(); source.append(('edge',4,deepcopy(source[-1][2])))
    with pytest.raises(ValueError): list(LiteralTransform(plan).records(iter(source)))
    source, plan = stream_fixture(True, True); source[-4][2]['metadata']['unexpected'] = True
    with pytest.raises(ValueError, match='existing target changed'): list(LiteralTransform(plan).records(iter(source)))


def test_missing_gene_witness_or_literal_collision_is_not_silently_accepted():
    source, plan = stream_fixture(); source = [r for r in source if r[1] != 'CUI:C1421437']
    with pytest.raises(ValueError, match='target witnesses incomplete'): list(LiteralTransform(plan).records(iter(source)))
    source, plan = stream_fixture(); node = repair.literal_node(plan['new_literals'][0]['name'])
    source.insert(1, ('node', node['id'], node))
    with pytest.raises(ValueError, match='collides'): list(LiteralTransform(plan).records(iter(source)))
