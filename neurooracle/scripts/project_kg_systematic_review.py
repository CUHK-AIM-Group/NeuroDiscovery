"""Source-bound reduced R73 review fields and byte locations, never preimages.

The full discovery has already parsed and sealed the JSON. Byte locations are
only an acceleration for rereading selected records from that exact source.
Every node location is checked against its independent full-record census hash;
the complete edge section is enumerated and every selected owner is decoded.
"""
from collections import defaultdict
import codecs
import json
import mmap
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.shared_relation_catalog import check_file

OUTPUT=j.OUTPUT/'round73_systematic_relation_consolidation'
DATABASE=OUTPUT/'WORKING_PROJECTIONS.sqlite'
LOCATIONS=OUTPUT/'REVIEW_LOCATIONS.sqlite'
NODE=re.compile(rb'"([^"\\]+)":\{')
OWNER=re.compile(rb'"(?:claim_id|source_id)":"(CLM:[^"\\]+)"')


def record_at(mm,start):
    size=16384
    while size<=4194304:
        text=codecs.getincrementaldecoder('utf-8')().decode(mm[start:start+size],final=False)
        try:return json.JSONDecoder().raw_decode(text)[0]
        except json.JSONDecodeError:size*=2
    raise ValueError('review record exceeds bounded decoder')


def read_bound(mm,entry):
    record=record_at(mm,entry['offset'])
    require(digest(record)==entry['sha'],'bound source record changed')
    return record


def science(record,rid):
    md=record['metadata'];inner=md.get('metadata') or {}
    fields=('subject_id','subject_name','subject_type','predicate','object_id','object_name','object_type',
        'raw_text','source_paper','evidence','negated','conditions','population','scope_reaudit','confidence')
    return dict(claim_id=record['id'],claim_sha256=digest(record),relation_id=rid,
        scientific_fields={k:md[k] for k in fields if k in md},
        nested_endpoint_fields={k:v for k,v in inner.items() if k in fields[:7]},
        metadata_field_hashes={k:digest(v) for k,v in md.items()},
        unknown_fields_read_but_not_copied=True,record_preimage_saved=False)


def main():
    require(not LOCATIONS.exists() and not (OUTPUT/'REVIEW_PROJECTION.json').exists(),'projection exists')
    analysis=j.read_json(OUTPUT/'ANALYSIS.json');baseline=j.read_json(OUTPUT/'SOURCE_BASELINE.json')
    require(baseline==j.read_json(j.OUTPUT/'CAMPAIGN.json'),'baseline advanced')
    check_file(analysis['database'],full_hash=True);check_file(baseline['current_graph'])
    db=sqlite3.connect(DATABASE.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    selected={r[0] for r in db.execute('SELECT cid FROM candidate_claims')}
    selected.update(r['claim_id'] for r in rows(OUTPUT/'CURRENT_CANDIDATE_CLAIM_BINDINGS.jsonl'))
    names=set();root_nodes=set()
    for field in ('current_gene_holds','current_root_holds','current_closure_review_remainders','current_source_identifier_holds'):
        for r in rows(baseline[field]['path']):
            selected.add(r['claim_id'])
            if r.get('name'):names.add(r['name'])
    previous=Path(baseline['current_acceptance']['path']).parent
    for r in j.read_json(previous/'source_review/ROOT_TYPE_REVIEW.json'):
        names.add(r['complete_surface'])
        root_nodes.update(x['node_id'] for x in r['claims'])
    for r in rows(previous/'DEFERRED_GROUPS.jsonl'):selected.update(r['complete_current_member_ids']);names.add(r['subject_name']);names.add(r['object_name'])
    for r in rows(previous/'CLAIM_EVENTS.jsonl'):selected.add(r['claim_id'])
    db.execute('CREATE TEMP TABLE wanted_names(name TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO wanted_names VALUES (?)',[(n,) for n in names])
    selected.update(r[0] for r in db.execute('SELECT e.cid FROM endpoint_incidents e JOIN wanted_names w ON w.name=e.name'))
    # Every incident of a source mention whose type might change is reviewed.
    for nid in root_nodes:selected.update(r[0] for r in db.execute('SELECT cid FROM endpoint_incidents WHERE nid=?',(nid,)))
    db.execute('CREATE TEMP TABLE wanted_claims(cid TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO wanted_claims VALUES (?)',[(cid,) for cid in selected])
    claims={r['cid']:dict(r) for r in db.execute('SELECT c.* FROM claims c JOIN wanted_claims w ON c.cid=w.cid')}
    require(set(claims)==selected,'selected claim missing')
    endpoint_ids={r[k] for r in claims.values() for k in ('sid','tid')}|root_nodes
    # Include all literal full-name targets and canonical registry candidates.
    db.execute('CREATE TEMP TABLE wanted_surfaces(name TEXT PRIMARY KEY)')
    surfaces={r[k] for r in claims.values() for k in ('sn','tn')}
    db.executemany('INSERT INTO wanted_surfaces VALUES (?)',[(n,) for n in surfaces])
    endpoint_ids.update(r[0] for r in db.execute("SELECT n.nid FROM nodes n JOIN wanted_surfaces w ON n.name=w.name COLLATE NOCASE WHERE n.kind<>'umls_atom'"))
    for field in ('current_entity_terms',):
        reg=j.read_json(baseline[field]['path'])
        from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
        from neurooracle.src.relation_evidence import name_key
        terms=VerifiedEntityTerms(reg)
        for name in surfaces:
            term=terms.entries.get(name_key(name))
            if term:endpoint_ids.add(term['target_id'])
    db.execute('CREATE TEMP TABLE wanted_nodes(nid TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO wanted_nodes VALUES (?)',[(nid,) for nid in endpoint_ids])
    endpoints={r['nid']:dict(r) for r in db.execute('SELECT n.* FROM nodes n JOIN wanted_nodes w ON w.nid=n.nid')}
    require(set(endpoints)==endpoint_ids,'target node absent')
    expected={cid:r['node_sha'] for cid,r in claims.items()}|{nid:r['node_sha'] for nid,r in endpoints.items()}
    j.atomic_json(OUTPUT/'REVIEW_SELECTION.json',dict(graph=baseline['current_graph'],claims=sorted(claims),nodes=sorted(endpoints)))
    loc=sqlite3.connect(LOCATIONS)
    loc.executescript('CREATE TABLE nodes(nid TEXT PRIMARY KEY,offset INTEGER,sha TEXT);CREATE TABLE edges(ordinal INTEGER PRIMARY KEY,owner TEXT,offset INTEGER,sha TEXT);CREATE INDEX edges_owner ON edges(owner);')
    scientific=[];witnesses=[];edge_rows=[];found=set();node_count=0
    print(json.dumps(dict(phase='SELECTED_CURRENT_SOURCE_PROJECTIONS',claims=len(claims),entity_nodes=len(endpoints))),flush=True)
    with open(baseline['current_graph']['path'],'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        edge_marker=mm.find(b'},"edges":[');require(edge_marker>0,'edge section absent')
        for m in NODE.finditer(mm,0,edge_marker):
            node_count+=1;nid=m[1].decode()
            if nid not in expected:continue
            offset=m.end()-1;require(mm[offset:offset+1]==b'{','node layout differs')
            record=record_at(mm,offset);seal=digest(record)
            require(record['id']==nid and seal==expected[nid] and nid not in found,'node/census location differs')
            found.add(nid);loc.execute('INSERT INTO nodes VALUES (?,?,?)',(nid,offset,seal))
            if nid in claims:scientific.append(science(record,claims[nid]['rid']))
            else:
                witnesses.append(dict(node_id=nid,node_sha256=seal,preferred_name=record['preferred_name'],
                    aliases=record.get('aliases') or [],semantic_types=record.get('semantic_types') or [],
                    domain_tags=record.get('domain_tags') or [],source_vocab=record.get('source_vocab'),
                    metadata_identity_fields={k:v for k,v in (record.get('metadata') or {}).items() if k in ('audit_ref','atom_type','anchor_role','canonical_id','cui')},
                    external_ids=record.get('external_ids') or {},definition=record.get('definition') or '',
                    metadata_field_hashes={k:digest(v) for k,v in (record.get('metadata') or {}).items()}))
        require(found==set(expected),'selected node location proof differs; missing '+repr(sorted(set(expected)-found)[:10]))
        print(json.dumps(dict(phase='COMPLETE_OWNED_EDGE_ENUMERATION',nodes=node_count,selected=len(found))),flush=True)
        start=edge_marker+len(b'},"edges":[');ordinal=0
        while start>=0:
            nxt=mm.find(b'{"source_id":',start+1);end=nxt if nxt>=0 else len(mm)
            ordinal+=1
            owners={m[1].decode() for m in OWNER.finditer(mm,start,end)} & selected
            if owners:
                record=record_at(mm,start);owner=edge_owner(record)
                require(owner in selected and owners=={owner},'ambiguous selected edge owner')
                seal=digest(record);loc.execute('INSERT INTO edges VALUES (?,?,?,?)',(ordinal,owner,start,seal))
                edge_rows.append(dict(ordinal=ordinal,claim_id=owner,edge_sha256=seal,
                    source_id=record['source_id'],target_id=record['target_id'],relation_type=record['relation_type'],
                    anchor_role=(record.get('metadata') or {}).get('anchor_role')))
            start=nxt
        require(ordinal==baseline['counts']['edges'],'complete edge layout count differs')
    loc.commit();require(loc.execute('PRAGMA integrity_check').fetchone()[0]=='ok','locations integrity')
    loc.close();db.close()
    check_file(baseline['current_graph'],full_hash=True)
    require(baseline==j.read_json(j.OUTPUT/'CAMPAIGN.json'),'baseline advanced during review projection')
    write_rows(OUTPUT/'SCIENTIFIC_PROJECTIONS.jsonl',sorted(scientific,key=lambda r:r['claim_id']))
    write_rows(OUTPUT/'TARGET_WITNESSES.jsonl',sorted(witnesses,key=lambda r:r['node_id']))
    write_rows(OUTPUT/'OWNED_EDGE_BINDINGS.jsonl',edge_rows)
    j.atomic_json(OUTPUT/'REVIEW_PROJECTION.json',dict(status='CURRENT_REVIEW_PROJECTION_COMPLETE',at=j.utc_now(),graph=baseline['current_graph'],
        discovery=j.fingerprint(OUTPUT/'ANALYSIS.json'),claims=len(claims),nodes=len(witnesses),owned_edges=len(edge_rows),
        selected_nodes_independently_hash_bound=len(found),visited_object_keys=node_count,
        complete_edge_layout_count=ordinal,full_source_sha_verified=True,source_mutations=0,
        artifacts={n:j.fingerprint(OUTPUT/n) for n in ('SCIENTIFIC_PROJECTIONS.jsonl','TARGET_WITNESSES.jsonl','OWNED_EDGE_BINDINGS.jsonl','REVIEW_LOCATIONS.sqlite')},
        code=j.fingerprint(Path(__file__)),full_record_preimages_saved=False))


if __name__=='__main__':main()
