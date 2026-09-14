from copy import deepcopy
import json
import sqlite3

import pytest

from core.web.claim_layer_extension_v2 import extend_base
from core.web.claim_layer_v2 import AcceptedClaimLayer, EvidenceUnavailable
from core.web.test_claim_layer_v1 import release, attach
from neurooracle.tests.test_claim_evidence_query import setup, fingerprint
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import relation_id, relation_key


def fixture(tmp_path):
    path,records,dossier=setup(tmp_path)
    campaign=json.loads(path.read_text())
    raw=(tmp_path/'graph.json').read_bytes()
    dbpath=tmp_path/'global_index.sqlite'
    entries=[]
    with sqlite3.connect(dbpath) as db:
        db.execute('CREATE TABLE observations(cid TEXT PRIMARY KEY,node_sha TEXT,byte_offset INTEGER,rid TEXT)')
        for record in records:
            encoded=json.dumps(record,ensure_ascii=False,separators=(',',':')).encode()
            offset=raw.find(encoded);assert offset>=0
            rid=relation_id(relation_key(record['metadata']))
            db.execute('INSERT INTO observations VALUES(?,?,?,?)',(record['id'],digest(record),offset,rid))
            entries.append(dict(claim_id=record['id'],node_sha=digest(record),byte_offset=offset,original_relation_id=rid))
    manifest=tmp_path/'index_acceptance.json'
    manifest.write_text(json.dumps(dict(graph=campaign['current_graph'],database=fingerprint(dbpath))))
    payload=dict(original_relation_extension=dict(global_index_acceptance=fingerprint(manifest),members=[entries[3]]))
    relations=[json.loads(l) for l in (tmp_path/'catalog.jsonl').read_text().splitlines()]
    return campaign,records,relations,[dossier],payload,entries


def test_extension_reads_the_unchanged_complete_original_relation(tmp_path):
    campaign,records,relations,dossiers,payload,_=fixture(tmp_path)
    before=deepcopy((relations,dossiers))
    rs,ds,inputs=extend_base(relations,dossiers,payload,campaign)
    assert (relations,dossiers)==before
    assert len(rs)==len(relations)+1
    assert ds[-1]['observations'][0]['claim']==records[3]['metadata']
    assert ds[-1]['observations'][0]['source_identity']['status']=='verified'
    assert ds[-1]['observations'][0]['source_review']['claim_sha256']==digest(records[3])
    assert inputs


@pytest.mark.parametrize('fault',['hash','offset','duplicate','old_member','incomplete','graph','altered_identity'])
def test_extension_rejects_false_bindings_and_incomplete_scope(tmp_path,fault):
    campaign,records,relations,dossiers,payload,entries=fixture(tmp_path)
    ext=payload['original_relation_extension']
    if fault=='hash':ext['members'][0]['node_sha']='a'*64
    elif fault=='offset':ext['members'][0]['byte_offset']+=1
    elif fault=='duplicate':ext['members'].append(deepcopy(ext['members'][0]))
    elif fault=='old_member':ext['members']=[entries[0]]
    elif fault=='graph':campaign['current_graph']=dict(campaign['current_graph'],sha256='a'*64)
    elif fault=='altered_identity':
        identity=json.loads((tmp_path/'papers.json').read_text());identity['records']['111']['title']='changed'
        fp=tmp_path/'supplemental.json';fp.write_text(json.dumps(identity));ext['supplemental_identity_registry']=fingerprint(fp)
    else:
        # Claim IDs belonging to a relation must be complete even if not all
        # member records would otherwise be materialized by the projection.
        dbpath=tmp_path/'global_index.sqlite'
        with sqlite3.connect(dbpath) as db:db.execute('INSERT INTO observations VALUES(?,?,?,?)',('CLM:omitted','a'*64,0,entries[3]['original_relation_id']))
        mp=tmp_path/'index_acceptance.json';m=json.loads(mp.read_text());m['database']=fingerprint(dbpath);mp.write_text(json.dumps(m));ext['global_index_acceptance']=fingerprint(mp)
    with pytest.raises((EvidenceUnavailable,ValueError)):
        extend_base(relations,dossiers,payload,campaign)


def test_v2_adapter_preserves_all_v1_aliases_and_source_evidence(tmp_path):
    path,campaign,records,_,dossier,payload=release(tmp_path)
    attach(tmp_path,path,campaign,payload)
    service=AcceptedClaimLayer(path)
    expected=service.query(claim_id='CLM:0')
    assert service.status()['reviewed_multipaper_claim_count']==1
    assert service.query(relation_id=dossier['relation_id'])['papers']==expected['papers']
    assert service.query(claim_id='CLM:3')['article_count']==1
