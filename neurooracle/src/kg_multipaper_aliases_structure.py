"""Repair one reviewed empty object anchor while consolidating their claims.

Each exception is bound to the complete current claim and all three owned
edges. The scientific edge and owning-source review agree on the object;
the second about edge incorrectly repeats the subject role at an empty node.
No unlisted observation receives this exception.
"""
from copy import deepcopy

from .kg_identity_pilot import digest
from .kg_literal_endpoint_repair import edge_owner
from .kg_root_repairs import apply_event, event
from .kg_systematic_consolidation import revise_owned_edges as standard_edges

EMPTY_ANCHOR = 'CLM_CONCEPT:unnamed_da39a3ee5e6b'
REVIEWED_ANCHORS = {'CLM:5ed8293e6f247360': {'claim_sha256': 'b278c968fd5bbdd0e385f565cc49cb856576bf5aebbf3f55c51e27430c28622d',
                          'edges': {224831: 'e5ab13a562a3ef6aeac2f5e64032f1022b2e72afff9c23a20cfdf574dd091162',
                                    224832: '99fa0a9e8d8c9594fb98b6f5af73e60b6717436669e7d3845f833f9bcdb238c3',
                                    362482: '3e80146001d9c4917488c0b4145d381ecfdc799d650d5c5c7b13f7431b40d00b'}}}


def revise_owned_edges(before, after, owned):
    cid = before['id']
    review = REVIEWED_ANCHORS.get(cid)
    if review is None:
        return standard_edges(before, after, owned)
    owned = list(owned)
    if (digest(before) != review['claim_sha256'] or len(owned) != 3
            or len({i for i, _ in owned}) != 3
            or {i: digest(e) for i, e in owned} != review['edges']):
        raise ValueError('reviewed placeholder binding changed')
    md = before['metadata']
    normalized = []
    corrected = 0
    for ordinal, edge in owned:
        if edge_owner(edge) != cid:
            raise ValueError('reviewed placeholder owner changed')
        revised = deepcopy(edge)
        if edge['relation_type'] == 'about' and edge['target_id'] == EMPTY_ANCHOR:
            if (edge['source_id'] != cid or edge['metadata'].get('anchor_role') != 'subject'
                    or md['subject_id'] == md['object_id'] or EMPTY_ANCHOR in (md['subject_id'], md['object_id'])):
                raise ValueError('reviewed placeholder pattern changed')
            revised['target_id'] = md['object_id']
            revised['metadata']['anchor_role'] = 'object'
            corrected += 1
        normalized.append((ordinal, revised))
    if corrected != 1 or sum(e['relation_type'] != 'about' for _, e in owned) != 1:
        raise ValueError('reviewed placeholder closure changed')
    changes = {e['ordinal']: e for e in standard_edges(before, after, normalized)}
    final = {i: apply_event(e, changes[i]) if i in changes else e for i, e in normalized}
    return [event(e, final[i], ordinal=i, claim_id=cid)
            for i, e in owned if e != final[i]]
