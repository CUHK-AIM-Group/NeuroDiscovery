"""Repair two exact, source-reviewed empty object anchors in the R81 batch."""
from copy import deepcopy
from .kg_identity_pilot import digest
from .kg_literal_endpoint_repair import edge_owner
from .kg_root_repairs import apply_event,event
from .kg_systematic_consolidation import revise_owned_edges as standard_edges

EMPTY_ANCHOR='CLM_CONCEPT:unnamed_da39a3ee5e6b'
REVIEWED_ANCHORS={
 'CLM:086ab921b5e31c72':dict(claim_sha256='e238a30487cd0f3aca4b46dc5a662476168e18ff45e8e748086261e70b708631',
    edges={223219:'2866252976bd60e52015546a6a95e72d0851945069f3595542c4d5fc60d1f237',
           223220:'6139bfaeae54c12b13303e2f156f6ef16ea86cfb8cf27a523f0cf17191c217ad',
           367597:'c531519f5bf49a83dc32249244d4df48650e0188e5f3318ba392ff7264957050'}),
 'CLM:9a04a72e8016fe07':dict(claim_sha256='9c318bf546ad0bf6110624dd6b8566544d6a948f11657f2fae1b7623a59957f0',
    edges={223469:'05e9d6b7b498014964f87c10bc17f56ddcd2436f31c0b0425c0d00193ce82979',
           223470:'30ca3ff781828b1e8f19e8075cfd70fb96945c763127e5be5d0d76edfba56791',
           367710:'3af5a5ec637617d5a9fe0e6b4a2d6118d90adf3ef5d166401dad46f40daec770'}),
}

def revise_owned_edges(before,after,owned):
    cid=before['id'];review=REVIEWED_ANCHORS.get(cid)
    if review is None:return standard_edges(before,after,owned)
    owned=list(owned)
    if (digest(before)!=review['claim_sha256'] or len(owned)!=3 or len({i for i,_ in owned})!=3
            or {i:digest(e) for i,e in owned}!=review['edges']):
        raise ValueError('reviewed placeholder binding changed')
    md=before['metadata'];normalized=[];corrected=0
    for ordinal,edge in owned:
        if edge_owner(edge)!=cid:raise ValueError('reviewed placeholder owner changed')
        revised=deepcopy(edge)
        if edge['relation_type']=='about' and edge['target_id']==EMPTY_ANCHOR:
            if (edge['source_id']!=cid or edge['metadata'].get('anchor_role')!='subject'
                    or md['subject_id']==md['object_id'] or EMPTY_ANCHOR in (md['subject_id'],md['object_id'])):
                raise ValueError('reviewed placeholder pattern changed')
            revised['target_id']=md['object_id'];revised['metadata']['anchor_role']='object';corrected+=1
        normalized.append((ordinal,revised))
    if corrected!=1 or sum(e['relation_type']!='about' for _,e in owned)!=1:
        raise ValueError('reviewed placeholder closure changed')
    changes={e['ordinal']:e for e in standard_edges(before,after,normalized)}
    final={i:apply_event(e,changes[i]) if i in changes else e for i,e in normalized}
    return [event(e,final[i],ordinal=i,claim_id=cid) for i,e in owned if e!=final[i]]
