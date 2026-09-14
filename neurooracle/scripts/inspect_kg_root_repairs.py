"""R69 Batch 2: bounded repair workspace, never write a KG or a full preimage.

The disposable SQLite holds selected records only. Durable outputs contain
hashes, finite field changes and evidence decisions, not old graph copies.
"""
from collections import Counter
from pathlib import Path
import json
import os
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.relation_evidence import name_key
from neurooracle.src.shared_relation_catalog import check_file
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = j.OUTPUT / 'round69_root_cause_repairs'
R68 = j.OUTPUT / 'round68_fragmentation_root_census'
DATABASE = OUTPUT / 'WORKING_REPAIR_SELECTION.sqlite'
INPUT_NAMES = ('BATCH_ACCEPTANCE.json', 'CANDIDATE_MEMBERS.jsonl', 'CANDIDATE_GROUPS.jsonl',
               'CLAIM_AS_ENTITY_ENDPOINTS.jsonl', 'EXACT_EVIDENCE_CANDIDATES.jsonl',
               'CROSS_SOURCE_TEXT_REVIEW.jsonl', 'SCIENTIFIC_IDENTITY_CASES.jsonl',
               'CURRENT_CASE_PROJECTIONS.jsonl', 'SCIENTIFIC_IDENTITY_CONTRACT.md')


def progress(phase, **values):
    state = dict(status='READ_ONLY_RUNNING', at=j.utc_now(), pid=os.getpid(), phase=phase, **values)
    j.atomic_json(OUTPUT / 'RUN_STATE.json', state)
    print(compact(state), flush=True)


def main():
    require(not DATABASE.exists() and not (OUTPUT / 'SCAN.json').exists(), 'inspect existing selection before rerun')
    c = j.read_json(j.OUTPUT / 'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(Path(c['current_acceptance']['path']).parent.name == 'round67_relation_semantic_consolidation', 'R67 required')
    census = j.read_json(c['current_paper_census']['path'])
    require(census['graph'] == c['current_graph'], 'census/graph binding')
    for fp in [c[k] for k in ('current_acceptance', 'current_entity_terms', 'current_paper_identities', 'current_paper_census')]:
        check_file(fp, full_hash=True)
    check_file(census['database'], full_hash=True)
    inputs = [j.fingerprint(R68 / name) for name in INPUT_NAMES]
    code = [j.fingerprint(Path(__file__)), *[j.fingerprint(j.REPO / ('neurooracle/src/' + n + '.py'))
            for n in ('verified_entity_terms', 'claim_ingestion', 'claim_extractor', 'claim_evidence_identity', 'graph_manager')]]
    j.atomic_json(OUTPUT / 'BASELINE.json', dict(at=j.utc_now(), campaign=c, census=census,
        input_artifacts=inputs, runtime_before=code, scope='Batch 2 repairs and frozen plan; Batch 3 builds and switches KG',
        graph_writes_authorized_in_this_batch=False, model_calls=0, full_record_backups=False))
    members = rows(R68 / 'CANDIDATE_MEMBERS.jsonl')
    broken = rows(R68 / 'CLAIM_AS_ENTITY_ENDPOINTS.jsonl')
    dup = rows(R68 / 'EXACT_EVIDENCE_CANDIDATES.jsonl')
    duplicate_ids = {r['cid'] for g in dup for r in g['members']}
    selected = {r['cid'] for r in members + broken} | duplicate_ids
    selected.update(r['claim_id'] for r in rows(R68 / 'CURRENT_CASE_PROJECTIONS.jsonl'))
    required = set(selected)
    terms = VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path']))
    wanted_ids = set(terms.node_ids) | {r[k] for r in members + broken for k in ('sid', 'tid')}
    wanted_names = {name_key(r[k]) for r in members + broken for k in ('sn', 'tn')}
    wanted_names.update(('emotional dysregulation', 'treatment-resistant schizophrenia'))
    db = sqlite3.connect(DATABASE, uri=True)
    db.executescript('''
        PRAGMA journal_mode=DELETE;
        CREATE TABLE claims(cid TEXT PRIMARY KEY,sha TEXT,sid TEXT,sn TEXT,st TEXT,tid TEXT,tn TEXT,tt TEXT);
        CREATE TABLE selected(cid TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE nodes(nid TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE owned(ordinal INTEGER PRIMARY KEY,cid TEXT,payload TEXT);
        CREATE TABLE refs(kind TEXT,key TEXT,payload TEXT);
        CREATE TABLE routes(cid TEXT,side TEXT,term TEXT);
        CREATE TABLE study_types(value TEXT PRIMARY KEY,n INTEGER);
    ''')
    db.execute('ATTACH DATABASE ? AS frozen', (Path(census['database']['path']).as_uri() + '?mode=ro',))
    counts, stypes = Counter(), Counter()
    claim_batch, select_batch, node_batch, edge_batch, ref_batch, route_batch = [], [], [], [], [], []
    proof_nodes, proof_mappings = {}, {}
    refs_expected = {t['mapping_ref']: t['mapping_sha256'] for t in terms.entries.values()}
    seen_ids = set()
    def flush():
        for sql, batch in [('INSERT INTO claims VALUES (?,?,?,?,?,?,?,?)', claim_batch),
                           ('INSERT INTO selected VALUES (?,?)', select_batch), ('INSERT INTO nodes VALUES (?,?)', node_batch),
                           ('INSERT INTO owned VALUES (?,?,?)', edge_batch), ('INSERT INTO refs VALUES (?,?,?)', ref_batch),
                           ('INSERT INTO routes VALUES (?,?,?)', route_batch)]:
            if batch: db.executemany(sql, batch); batch.clear()
        db.commit()
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    progress('FULL_SOURCE_REPAIR_SELECTION')
    try:
        with hashed_reader(Path(c['current_graph']['path'])) as (reader, h):
            for kind, key, row in walk_graph(reader):
                counts[kind] += 1
                md = row.get('metadata') or {}
                if kind == 'node':
                    seen_ids.add(key)
                    if key.startswith('CLM:'):
                        counts['claims'] += 1
                        inner = md.get('metadata') or {}
                        claim_batch.append((key, digest(row), md['subject_id'], md['subject_name'],
                            compact([md.get('subject_type'), inner.get('subject_type')]), md['object_id'], md['object_name'],
                            compact([md.get('object_type'), inner.get('object_type')])))
                        stypes[compact((md.get('evidence') or {}).get('study_type'))] += 1
                        for side in ('subject', 'object'):
                            term = terms.term_for(md, side)
                            if term and md[side + '_id'] != term['target_id']:
                                selected.add(key); route_batch.append((key, side, compact(term)))
                        if key in selected: select_batch.append((key, compact(row)))
                    else:
                        labels = {name_key(x) for x in [row.get('preferred_name'), *(row.get('aliases') or [])]}
                        if key in wanted_ids or labels & wanted_names: node_batch.append((key, compact(row)))
                    if key in terms.node_ids: proof_nodes[key] = digest(row)
                elif kind == 'edge':
                    owner = edge_owner(row)
                    if owner in selected: edge_batch.append((int(key), owner, compact(row)))
                    ref = md.get('audit_ref')
                    if ref in refs_expected:
                        require(ref not in proof_mappings or proof_mappings[ref] == digest(row), 'ambiguous mapping witness')
                        proof_mappings[ref] = digest(row)
                # Only potential evidence-removal IDs need complete external reference closure.
                refs = list(exact_references(row, duplicate_ids))
                if refs: ref_batch.append((kind, str(key), compact(dict(sha=digest(row), refs=refs))))
                if counts[kind] % 10000 == 0: flush()
                if counts[kind] % 250000 == 0: progress('SCAN_' + kind, counts=dict(counts), selected=len(selected))
            require(h.hexdigest() == c['current_graph']['sha256'], 'full source SHA differs')
        flush()
        require(tuple(counts[k] for k in ('node', 'claims', 'edge')) == tuple(c['counts'][k] for k in ('nodes', 'claims', 'edges')), 'counts changed')
        require(db.execute('''SELECT COUNT(*) FROM claims c LEFT JOIN frozen.claims f ON c.cid=f.cid
            WHERE f.cid IS NULL OR c.sha<>f.node_sha''').fetchone()[0] == 0, 'all-claim census hash mismatch')
        require(db.execute('SELECT COUNT(*) FROM frozen.claims').fetchone()[0] == counts['claims'], 'census count differs')
        seen_selected = {r[0] for r in db.execute('SELECT cid FROM selected')}
        require(seen_selected == selected and required <= seen_selected, 'selected closure missing')
        for term in terms.entries.values():
            for f, s in (('target_id', 'target_sha256'), ('atom_id', 'atom_sha256'), ('parent_id', 'parent_sha256')):
                require(proof_nodes.get(term[f]) == term[s], 'registry live node proof differs')
        require(proof_mappings == refs_expected, 'registry mapping closure differs')
        endpoint_ids = {v for r in db.execute('SELECT sid,tid FROM claims') for v in r}
        require(not endpoint_ids - seen_ids, 'dangling endpoint')
        db.executemany('INSERT INTO study_types VALUES (?,?)', sorted(stypes.items()))
        db.executescript('CREATE INDEX claims_sid ON claims(sid); CREATE INDEX claims_tid ON claims(tid); CREATE INDEX owned_cid ON owned(cid);')
        db.commit()
        require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'working DB invalid')
        require(j.read_json(j.OUTPUT / 'CAMPAIGN.json') == c and j.fingerprint(Path(__file__)) == code[0], 'campaign/collector changed')
        j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
        j.atomic_json(OUTPUT / 'SCAN.json', dict(status='FULL_READ_ONLY_SELECTION_COMPLETE', at=j.utc_now(),
            graph=c['current_graph'], counts=dict(counts), selected_claims=len(selected),
            selected_nodes=db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0],
            owned_edges=db.execute('SELECT COUNT(*) FROM owned').fetchone()[0],
            verified_whole_name_routes=db.execute('SELECT COUNT(*) FROM routes').fetchone()[0],
            study_type_values=len(stypes), all_claim_hashes_compared=counts['claims'],
            registry_live_node_proofs=len(proof_nodes), registry_live_mapping_proofs=len(proof_mappings),
            graph_modified=False, protected_guards_unchanged=True, temporary_selection_not_a_graph=True))
    finally:
        db.close()
    progress('SELECTION_COMPLETE')


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        j.atomic_json(OUTPUT / 'RUN_STATE.json', dict(status='FAILED', at=j.utc_now(), error=repr(error), graph_modified=False))
        raise
