from copy import deepcopy
import pytest

from neurooracle.src import kg_multipaper_predicates_structure as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_root_repairs import apply_event
from neurooracle.src.kg_root_application import reverse_event


def fixture(monkeypatch):
    before = dict(id='CLM:test', metadata=dict(subject_id='S', object_id='O', predicate='is_biomarker_of',
                  raw_text='Measured marker supports diagnosis.', source_paper=dict(pmid='123')))
    after = deepcopy(before); after['metadata'].update(subject_id='S2', object_id='O2')
    def edge(source, target, kind, role=None):
        md = dict(claim_id=before['id'], unknown=dict(values=[3, 7]))
        if role: md['anchor_role'] = role
        return dict(source_id=source, target_id=target, relation_type=kind, source='claim_extraction',
                    confidence=0.74, metadata=md)
    owned = [(1, edge(before['id'], 'S', 'about', 'subject')),
             (2, edge(before['id'], repair.EMPTY_ANCHOR, 'about', 'subject')),
             (3, edge('S', 'O', 'is_biomarker_of'))]
    monkeypatch.setattr(repair, 'REVIEWED_ANCHORS', {before['id']:dict(claim_sha256=digest(before),
                         edges={i:digest(e) for i,e in owned})})
    return before, after, owned


def test_reviewed_placeholder_rejoins_object_and_preserves_inverse(monkeypatch):
    before, after, owned = fixture(monkeypatch)
    original = deepcopy((before, after, owned))
    events = {e['ordinal']:e for e in repair.revise_owned_edges(before, after, owned)}
    final = {i:apply_event(e, events[i]) for i,e in owned}
    assert final[1]['target_id'] == 'S2'
    assert final[2]['target_id'] == 'O2' and final[2]['metadata']['anchor_role'] == 'object'
    assert (final[3]['source_id'], final[3]['target_id']) == ('S2','O2')
    for i,e in owned:
        assert reverse_event(final[i],events[i]) == e
        assert final[i]['metadata']['unknown'] == e['metadata']['unknown']
    assert (before, after, owned) == original


@pytest.mark.parametrize('damage', ['claim','edge','missing','extra','duplicate','unlisted'])
def test_placeholder_exception_is_finite_and_sealed(monkeypatch, damage):
    before, after, owned = fixture(monkeypatch)
    if damage == 'claim': before['metadata']['raw_text'] += ' Changed.'
    if damage == 'edge': owned[1][1]['confidence'] = 0.75
    if damage == 'missing': owned.pop()
    if damage == 'extra': owned.append((4, deepcopy(owned[0][1])))
    if damage == 'duplicate': owned[2] = owned[1]
    if damage == 'unlisted': monkeypatch.setattr(repair, 'REVIEWED_ANCHORS', {})
    with pytest.raises(ValueError): repair.revise_owned_edges(before,after,owned)


def test_placeholder_cannot_hide_disagreeing_scientific_edge(monkeypatch):
    before,after,owned=fixture(monkeypatch)
    owned[2][1]['target_id']='Different disease'
    repair.REVIEWED_ANCHORS[before['id']]['edges']={i:digest(e) for i,e in owned}
    with pytest.raises(ValueError,match='scientific edge differs'):
        repair.revise_owned_edges(before,after,owned)
