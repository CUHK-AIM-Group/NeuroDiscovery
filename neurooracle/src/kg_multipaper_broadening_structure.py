"""Repair five reviewed empty object anchors while consolidating their claims.

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
REVIEWED_ANCHORS = {'CLM:031854fa5e746b5c': {'claim_sha256': '946cdcaf60631726c76ee6656ce2c33f7bd07642e2b9d7ff4e61016fc7f47c31',
                          'edges': {221495: '9bababa1373bd413af392e11623c3d6d37454a5985a91bf9e561fe055fed40e7',
                                    221496: 'a9ccb0873eb8396acfdf7b556a7010caecad05709b4a03a02368447a09fbd23b',
                                    366785: '72bd638dc0f459adc21eb8e465be0c45586b798f7edf8a728196a74aca146f40'}},
 'CLM:71a4fd6249c76c52': {'claim_sha256': 'd92f54203563cbc5d56c6185b8dada4724315fe6d1caca55672594507a348b0b',
                          'edges': {221417: '15cde9301e4f8d4e74754badd5cc2bf27d2e91039c137982612385cf3fbdbd46',
                                    221418: '9d734523192c0ad28549f94e4a7a5c8f61b5f7c0a10a629b75f608ba34f537b0',
                                    366750: '8cc37e36aa709e44593c26504be2498ba39ec55221db2b0067bfd8bf03142b1c'}},
 'CLM:99a325201fc370cb': {'claim_sha256': '4236eabc06cafda2d20729fac132fbd4cedcd8b34c1dc4df240f3f5f0c3fe6b2',
                          'edges': {221889: 'c6cf0d9925afd78fc3a839005f1d465a2cecf00437aeccac2f69285388f12ec8',
                                    221890: 'b77a97ab9478b3ae8c3ee9f363f5eba6de183bc326ea5dd24d00fa5dba81d120',
                                    366964: '04ef89d2c4ea378326ce8d87db3b5efb80723fa559650d82e6a034aab55dbaaf'}},
 'CLM:a547641dbdd75abe': {'claim_sha256': '9f4684b8623362e80325be6a37977f7b74dacd14654e25a85ecae439754844cc',
                          'edges': {221545: 'a0fc80018eaf28183ab42cb1caa182de5b3406b80960efdc9a094dbe2a09c98b',
                                    221546: 'b052be1fff76a1f1fb1c78d8c098d5427503ee1829ee39ae448371d747a9354c',
                                    355268: '47b15a15ce925577e6b1b0fa6e2a4533552281bf0daeec8a1324db3bd3da6305'}},
 'CLM:d975febb6bda7720': {'claim_sha256': '66a7d16844c14a87576cce70b00e2c7aa9d9c557176e6216f3ff81a7fbeb238e',
                          'edges': {227621: 'ca628285c69898d4a1013d814e4b4acc073e61e767ac3010a9df5d98e88edd90',
                                    227622: 'e9a9dc0199ead42663aa0f6c1aa86ff3fffa65339cf4e0d56b94a2ca4a825ff8',
                                    369621: 'b2b727f31bd2968d2c66fd07b18413d08a4190b18b0de8bdcf6bfbf7fce7f376'}}}


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
