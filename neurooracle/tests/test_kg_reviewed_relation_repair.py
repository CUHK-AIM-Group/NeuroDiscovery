from copy import deepcopy
import pytest

from neurooracle.src import kg_reviewed_relation_repair as repair
from neurooracle.src.claim_ingestion import resolve_claim_entities
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_root_application import reverse_event
from neurooracle.src.kg_root_repairs import apply_event
from neurooracle.src.schema import Claim, ConceptNode


def record():
    md = dict(id='CLM:reviewed', subject_id='CUI:S', subject_name='drug', predicate='has_adverse_effect',
        object_id='CUI:wrong', object_name='complete adverse outcome', negated=False, confidence=.7,
        source_paper=dict(pmid='12345', title='Own source', publication_types=['Review']),
        raw_text='The complete adverse outcome is a potential risk, without an estimated incidence.',
        evidence=dict(sample_size=0, raw_stats='', study_type='review', unknown_stat={'denominator': None}),
        scope_reaudit=dict(decision='include', source_note='Original audit'),
        subject_type='drug', object_type='adverse_event', unknown_observation={'nested': [None, False, 0]},
        metadata=dict(subject_type='drug', object_type='adverse_event', object_id='CUI:wrong',
            conditions=['cycle 1, day 8'], unknown_inner='retain'))
    return dict(id=md['id'], preferred_name='drug has_adverse_effect complete adverse outcome', metadata=md,
                unknown_node={'value': 0})


def owned(before, with_science=True):
    cid = before['id']; md = before['metadata']
    out = [(7, dict(source_id=cid, target_id=md['subject_id'], relation_type='about', metadata={'anchor_role': 'subject'})),
           (11, dict(source_id=cid, target_id=md['object_id'], relation_type='about', metadata={'unknown_edge': False}))]
    if with_science:
        out.append((16, dict(source_id=md['subject_id'], target_id=md['object_id'], relation_type=md['predicate'],
            metadata=dict(claim_id=cid, negated=False, raw_stats=None, unknown_edge=['keep']))))
    return out


def policy_fixture(monkeypatch):
    before = record(); event, after = repair.revise_claim(before, {'object_id': 'CLM_CONCEPT:complete'}, review_id='review1')
    kg = KnowledgeGraph()
    for node in (ConceptNode(id='CUI:S', preferred_name='drug'),
                 ConceptNode(id='CLM_CONCEPT:complete', preferred_name='complete adverse outcome')): kg.add_concept(node)
    target_nodes = {node.id: node.to_dict() for node in (kg.get_concept('CUI:S'), kg.get_concept('CLM_CONCEPT:complete'))}
    rule = repair.reimport_rule(before, after, target_nodes, 'review1')
    policy = dict(version=repair.VERSION, rules=[rule])
    monkeypatch.setattr(repair, 'APPROVED_RULESET_DIGEST', digest(policy['rules']))
    kg.serialization_metadata['reviewed_relation_repairs'] = policy
    return kg, before, after, event


def test_exact_revision_and_inverse_preserve_unknown_and_scientific_fields():
    before = record(); snapshot = deepcopy(before)
    event, after = repair.revise_claim(before, {'object_id': 'CLM_CONCEPT:complete'}, review_id='review1')
    assert before == snapshot and apply_event(before, event) == after and reverse_event(after, event) == before
    expected = deepcopy(before)
    expected['metadata']['object_id'] = expected['metadata']['metadata']['object_id'] = 'CLM_CONCEPT:complete'
    assert after == expected


@pytest.mark.parametrize('replacement', [{'raw_text': 'New result'}, {'negated': True}, {'object_type': 'disease'}, {}])
def test_evidence_and_type_edits_are_outside_finite_module_scope(replacement):
    with pytest.raises(ValueError, match='unapproved claim field'):
        repair.revise_claim(record(), replacement, review_id='review1')


@pytest.mark.parametrize('problem', ['nested', 'label', 'self_loop'])
def test_claim_conflicts_fail_before_edit(problem):
    before = record(); replacement = {'object_id': 'CLM_CONCEPT:complete'}
    if problem == 'nested': before['metadata']['metadata']['object_id'] = 'CUI:other'
    if problem == 'label': before['preferred_name'] += ' extra unreviewed qualification'
    if problem == 'self_loop': replacement['object_id'] = before['metadata']['subject_id']
    snapshot = deepcopy(before)
    with pytest.raises(ValueError): repair.revise_claim(before, replacement, review_id='review1')
    assert before == snapshot


@pytest.mark.parametrize('with_science', [True, False])
def test_roleless_about_edge_and_existing_science_are_repaired_together(with_science):
    before = record(); _, after = repair.revise_claim(before, {'object_id': 'CLM_CONCEPT:complete'}, review_id='review1')
    edges = owned(before, with_science); snapshot = deepcopy(edges)
    events = repair.revise_edges(before, after, edges)
    assert {e['ordinal'] for e in events} == ({11, 16} if with_science else {11})
    for event in events:
        original = dict(edges)[event['ordinal']]; changed = apply_event(original, event)
        assert changed['target_id'] == 'CLM_CONCEPT:complete'
        assert changed['metadata'] == original['metadata'] and reverse_event(changed, event) == original
    assert edges == snapshot


def test_noncausal_predicate_changes_only_science_edge_and_derived_label():
    before = record(); before['metadata']['predicate'] = 'reduces'
    before['preferred_name'] = 'drug reduces complete adverse outcome'
    event, after = repair.revise_claim(before, {'predicate': 'correlates_with'}, review_id='ALPS')
    edges = owned(before); changed = repair.revise_edges(before, after, edges)
    assert len(changed) == 1 and changed[0]['ordinal'] == 16
    assert apply_event(dict(edges)[16], changed[0])['relation_type'] == 'correlates_with'
    assert after['metadata']['raw_text'] == before['metadata']['raw_text']
    assert after['metadata']['evidence'] == before['metadata']['evidence']
    assert reverse_event(after, event) == before


@pytest.mark.parametrize('problem', ['missing_about', 'foreign_owner', 'duplicate_about', 'science_disagrees'])
def test_incomplete_or_conflicting_owned_edges_are_rejected(problem):
    before = record(); _, after = repair.revise_claim(before, {'object_id': 'CLM_CONCEPT:complete'}, review_id='review1')
    edges = owned(before)
    if problem == 'missing_about': edges.pop(0)
    if problem == 'foreign_owner': edges[-1][1]['metadata']['claim_id'] = 'CLM:someone_else'
    if problem == 'duplicate_about': edges.append((23, deepcopy(edges[0][1])))
    if problem == 'science_disagrees': edges[-1][1]['relation_type'] = 'treats'
    with pytest.raises(ValueError): repair.revise_edges(before, after, edges)


def test_reimport_hook_covers_new_claim_id_and_keeps_all_evidence(monkeypatch):
    kg, before, after, _ = policy_fixture(monkeypatch)
    incoming = Claim.from_dict(before['metadata']); incoming.id = 'CLM:new_import_id'
    incoming.metadata['object_canonical_hint'] = 'a conflicting fuzzy hint must not override reviewed identity'
    original = incoming.to_dict()
    assert resolve_claim_entities(kg, incoming) is incoming
    expected = deepcopy(original); expected['object_id'] = expected['metadata']['object_id'] = after['metadata']['object_id']
    assert incoming.to_dict() == expected
    assert repair.apply_reviewed_reimport(kg, incoming) is incoming and incoming.to_dict() == expected


@pytest.mark.parametrize('field,value', [('pmid', '99999'), ('raw_text', 'A different sentence.'),
    ('negated', True), ('object_name', 'complete adverse outcome and CRS'), ('predicate', 'treats'),
    ('subject_name', 'Drug'), ('object_type', 'disease')])
def test_other_sources_qualifiers_polarity_and_roles_are_not_canonicalized(monkeypatch, field, value):
    kg, before, _, _ = policy_fixture(monkeypatch); md = before['metadata']
    if field == 'pmid': md['source_paper']['pmid'] = value
    elif field == 'object_type': md['metadata']['object_type'] = md['object_type'] = value
    else: md[field] = value
    incoming = Claim.from_dict(md); original = incoming.to_dict()
    assert repair.apply_reviewed_reimport(kg, incoming) is None and incoming.to_dict() == original


@pytest.mark.parametrize('problem', ['stale_target', 'missing_target', 'conflicting_id', 'nested_conflict', 'policy_tampered'])
def test_applicable_reimport_fails_without_partial_mutation(monkeypatch, problem):
    kg, before, _, _ = policy_fixture(monkeypatch)
    incoming = Claim.from_dict(before['metadata'])
    if problem == 'stale_target': kg.get_concept('CLM_CONCEPT:complete').metadata['changed'] = True
    if problem == 'missing_target':
        original_get = kg.get_concept
        monkeypatch.setattr(kg, 'get_concept', lambda nid: None if nid == 'CLM_CONCEPT:complete' else original_get(nid))
    if problem == 'conflicting_id': incoming.object_id = 'CUI:unreviewed'
    if problem == 'nested_conflict': incoming.metadata['object_id'] = 'CUI:unreviewed'
    if problem == 'policy_tampered': kg.serialization_metadata['reviewed_relation_repairs']['rules'][0]['pmid'] = '99999'
    original = incoming.to_dict()
    with pytest.raises(ValueError): repair.apply_reviewed_reimport(kg, incoming)
    assert incoming.to_dict() == original


def test_no_optin_leaves_normal_graphs_and_claims_untouched():
    kg = KnowledgeGraph(); incoming = Claim.from_dict(record()['metadata']); original = incoming.to_dict()
    assert repair.apply_reviewed_reimport(kg, incoming) is None and incoming.to_dict() == original


def test_ambiguous_exact_source_rules_fail_closed(monkeypatch):
    kg, before, _, _ = policy_fixture(monkeypatch)
    rules = kg.serialization_metadata['reviewed_relation_repairs']['rules']; rules.append(deepcopy(rules[0]))
    monkeypatch.setattr(repair, 'APPROVED_RULESET_DIGEST', digest(rules))
    incoming = Claim.from_dict(before['metadata']); original = incoming.to_dict()
    with pytest.raises(ValueError, match='ambiguous'): repair.apply_reviewed_reimport(kg, incoming)
    assert incoming.to_dict() == original
