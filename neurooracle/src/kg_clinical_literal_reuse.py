"""Exact complete measurement reuse; preserve all detail-backed source anchors."""
from .claim_semantics import concept_atom_roles, declared_type_atoms
from .kg_bulk_identity import change_claim
from .kg_identity_pilot import digest, nonidentity_claim
from .kg_literal_endpoint_repair import endpoint_gate, GENES, reverse_claim
from .relation_evidence import name_key

FAMILIES = {
    'voxel-mirrored homotopic connectivity alterations': {
        'target': 'CLM_CONCEPT:bc498e1c54527121',
        'members': {'CLM_CONCEPT:bc498e1c54527121', 'CLM_CONCEPT:voxel_mirrored_homotopic_connectivity_alterations'}},
    'gray matter volume alterations': {
        'target': 'CLM_CONCEPT:0a55ca2778fecaaf',
        'members': {'CLM_CONCEPT:0a55ca2778fecaaf', 'CLM_CONCEPT:gray_matter_volume_alterations_416455921a5b'}},
}


def roles(value): return {a.value for a in declared_type_atoms(value)}


def validate_family(name, nodes, incidents, aliases):
    group = FAMILIES[name]
    if set(nodes) != group['members']: raise ValueError('complete name family differs')
    if aliases: raise ValueError('unreviewed exact alias')
    target = nodes[group['target']]
    if {a.value for a in concept_atom_roles(target)} != {'imaging_marker'}:
        raise ValueError('canonical target type not confirmed')
    for nid, node in nodes.items():
        if name_key(node['preferred_name']) != name: raise ValueError('complete case-sensitive name differs')
        if node.get('semantic_types') or node.get('external_ids') or node.get('definition') or node.get('spatial_mapping') or node.get('aliases'):
            raise ValueError('additional semantic identity requires review')
        if {a.value for a in concept_atom_roles(node)} not in (set(), {'imaging_marker'}):
            raise ValueError('conflicting concept type')
        if not incidents[nid]: raise ValueError('missing current claim witness')
        for incident in incidents[nid]:
            if name_key(incident['name']) != name or set(incident['roles']) != {'imaging_marker'}:
                raise ValueError('incident name/type scope differs')
    return dict(name=name, target_id=group['target'], member_hashes={nid:digest(n) for nid,n in sorted(nodes.items())},
        incident_claim_ids=sorted({r['claim_id'] for refs in incidents.values() for r in refs}),
        scope='identical complete literal measurement expression; not new UMLS equivalence',
        source_anchor_nodes_and_detail_rows_preserved=True)


def reviewed_claim(record, changes):
    md = record['metadata']; inner = md.get('metadata') or {}
    if len({c['side'] for c in changes}) != len(changes): raise ValueError('duplicate side')
    for ch in changes:
        side = ch['side']; name = name_key(md.get(side+'_name'))
        if name not in FAMILIES or ch['name'] != name: raise ValueError('unreviewed complete name')
        group = FAMILIES[name]; old = md.get(side+'_id')
        if ch['target_id'] != group['target'] or ch['old_id'] != old or old == ch['target_id']:
            raise ValueError('wrong canonical endpoint')
        if inner.get(side+'_id',old) != old: raise ValueError('nested endpoint conflict')
        outer_type, inner_type = md.get(side+'_type'), inner.get(side+'_type')
        if outer_type and inner_type and roles(outer_type) != roles(inner_type): raise ValueError('nested type conflict')
        if roles(outer_type or inner_type) != {'imaging_marker'}: raise ValueError('claim measurement type not explicit')
        if old in GENES:
            if endpoint_gate(md,side): raise ValueError('gene measurement gate not confirmed')
        elif old not in group['members']: raise ValueError('unknown identity source')
    event = dict(claim_id=record['id'], claim_sha256=digest(record), changes=changes,
        nonidentity_sha256=digest(nonidentity_claim(record)))
    out = change_claim(record,event); event['current_node_sha256'] = digest(out)
    return event,out


def verified_reverse(record,event):
    original = reverse_claim(record,event)
    reproduced,_ = reviewed_claim(original,event['changes'])
    if not all(event[k] == v for k,v in reproduced.items()): raise ValueError('independent literal review differs')
    return original

