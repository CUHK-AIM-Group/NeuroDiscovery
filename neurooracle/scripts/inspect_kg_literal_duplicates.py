"""R39 complete current reference closure; console-only records, hash-only files."""
from collections import Counter, defaultdict
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.relation_evidence import name_key
from neurooracle.src.claim_semantics import concept_atom_roles, declared_type_atoms

OUTPUT = journal.OUTPUT/'round39_literal_duplicates'
NAMES = {'voxel-mirrored homotopic connectivity alterations', 'bilateral hippocampal volume', 'gray matter volume alterations'}


def progress(phase, **values):
    data = dict(status='READ_ONLY_INSPECTION', phase=phase, at=journal.utc_now(), **values)
    journal.atomic_json(OUTPUT/'INSPECTION_STATE.json', data); print(compact(data), flush=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(), 'inspection exists; reuse current proof')
    c = journal.read_json(journal.OUTPUT/'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    for key in ('current_acceptance','current_runtime_acceptance','current_gene_holds','current_entity_terms','current_shared_relations','current_paper_census'):
        require(journal.fingerprint(c[key]['path']) == c[key], 'current evidence changed')
    receipt = journal.read_json(c['current_acceptance']['path'])
    for fp in receipt['code']: require(journal.fingerprint(fp['path']) == fp, 'frozen current code changed')
    journal.guards([c['current_graph'], c['current_detail_store'], c['formal_sources']])
    prior = {r['node_id']:r for r in rows(journal.OUTPUT/'round38_measurement_reuse/CURRENT_NAME_CANDIDATES.jsonl') if r['preferred_name'] in NAMES}
    node_ids = set(prior)
    held = [r for r in rows(c['current_gene_holds']['path']) if r['reason'].startswith('reference_closure:') or r['reason']=='multiple_complete_name_candidates']
    claim_ids = {r['claim_id'] for r in held} | {i['claim_id'] for r in prior.values() for i in r['all_incident_claim_endpoints']} | {'CLM:593b77e75c8f89d8'}
    wanted = node_ids | claim_ids
    needle = re.compile('|'.join(re.escape(i) for i in sorted(wanted)))
    nodes, claims, refs, refs_by_owner, incidents, aliases = {}, {}, [], defaultdict(list), defaultdict(list), []
    counts = Counter(); code = journal.fingerprint(Path(__file__))
    progress('FULL_CURRENT_REFERENCE_BOUNDARY', node_candidates=len(node_ids), claims=len(claim_ids))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader, hashed):
        for kind, key, row in walk_graph(reader):
            counts[kind] += 1
            if kind == 'node':
                if key in node_ids:
                    require(digest(row) == prior[key]['node_sha256'], 'prior candidate changed')
                    nodes[key] = row
                if key.startswith('CLM:'):
                    md = row['metadata']
                    if key in claim_ids: claims[key] = row
                    for side in ('subject','object'):
                        if md.get(side+'_id') in node_ids:
                            require(key in claim_ids, 'incident closure expanded')
                            typ = md.get(side+'_type') or (md.get('metadata') or {}).get(side+'_type')
                            incidents[md[side+'_id']].append(dict(claim_id=key, side=side, name=md.get(side+'_name'),
                                roles=sorted(a.value for a in declared_type_atoms(typ)), claim_sha256=digest(row)))
                elif name_key(row.get('preferred_name')) in NAMES:
                    require(key in node_ids, 'same-name candidate expanded')
                if not key.startswith('CLM:'):
                    for alias in row.get('aliases') or []:
                        if name_key(alias) in NAMES: aliases.append(dict(node_id=key, alias=alias, node_sha256=digest(row)))
            payload = compact(row)
            if needle.search(payload):
                links = list(exact_references(row, wanted))
                if links: refs.append(dict(kind=kind,key=key,sha256=digest(row),references=links))
            if kind == 'edge' and edge_owner(row) in claim_ids:
                refs_by_owner[edge_owner(row)].append((int(key), row))
            if counts[kind] % 1000000 == 0: progress(kind,count=counts[kind])
        require(hashed.hexdigest() == c['current_graph']['sha256'], 'source full SHA differs')
    require(set(nodes) == node_ids and set(claims) == claim_ids, 'selected scope incomplete')
    detail = Path(c['current_detail_store']['path'])
    db = sqlite3.connect(detail.as_uri()+'?mode=ro', uri=True)
    dependency = {}
    for nid in sorted(node_ids):
        dependency[nid] = dict(atoms=db.execute('SELECT COUNT(*) FROM atoms WHERE source_mention_id=?',(nid,)).fetchone()[0],
            mappings_as_source=db.execute('SELECT COUNT(*) FROM mappings WHERE source_id=?',(nid,)).fetchone()[0],
            mappings_as_target=db.execute('SELECT COUNT(*) FROM mappings WHERE target_id=?',(nid,)).fetchone()[0],
            cui_records=db.execute('SELECT COUNT(*) FROM cuis WHERE record_id=?',(nid,)).fetchone()[0])
    db.close()
    require(sha256(detail) == c['current_detail_store']['sha256'], 'detail full source SHA differs')
    terms = journal.read_json(c['current_entity_terms']['path'])
    term_refs = list(exact_references(terms, node_ids))
    write_rows(OUTPUT/'EXACT_REFERENCES.jsonl', refs)
    write_rows(OUTPUT/'NODE_SCOPE.jsonl', [dict(node_id=nid, node_sha256=digest(row), preferred_name=row['preferred_name'],
        roles=sorted(a.value for a in concept_atom_roles(row)), incidents=incidents[nid], field_names=sorted(row),
        detail_dependencies=dependency[nid]) for nid,row in sorted(nodes.items())])
    write_rows(OUTPUT/'CLAIM_SCOPE.jsonl', [dict(claim_id=cid, claim_sha256=digest(row),
        raw_text_sha256=digest(row['metadata'].get('raw_text')), source_paper=row['metadata']['source_paper'],
        references=[dict(ordinal=n,edge_sha256=digest(e)) for n,e in refs_by_owner[cid]]) for cid,row in sorted(claims.items())])
    write_rows(OUTPUT/'FULL_ALIAS_CANDIDATES.jsonl', aliases)
    for nid,row in sorted(nodes.items()): print('CONSOLE_NODE='+compact(row), flush=True)
    for cid,row in sorted(claims.items()):
        print('CONSOLE_CLAIM='+compact(row), flush=True)
        for ordinal,edge in refs_by_owner[cid]: print('CONSOLE_EDGE='+compact(dict(ordinal=ordinal,row=edge)), flush=True)
    require(journal.read_json(journal.OUTPUT/'CAMPAIGN.json') == c and journal.fingerprint(Path(__file__)) == code, 'current input/code advanced')
    journal.guards([c['current_graph'], c['current_detail_store'], c['formal_sources']])
    summary = dict(status='CURRENT_DUPLICATE_SCOPE_INSPECTED_NOT_APPLIED', at=journal.utc_now(), graph=c['current_graph'],
        source_acceptance=c['current_acceptance'], detail_store=c['current_detail_store'], counts=dict(counts),
        source_full_sha_verified=True, detail_full_sha_verified=True, entity_registry_references=term_refs,
        code=code, graph_modified=False, record_preimages_saved=False,
        artifacts={name:journal.fingerprint(OUTPUT/name) for name in ('EXACT_REFERENCES.jsonl','NODE_SCOPE.jsonl','CLAIM_SCOPE.jsonl','FULL_ALIAS_CANDIDATES.jsonl')})
    journal.atomic_json(OUTPUT/'SOURCE_INSPECTION.json', summary)
    progress('COMPLETE', nodes=len(nodes), claims=len(claims), detail_dependencies=dependency, entity_registry_references=len(term_refs))


if __name__=='__main__': main()

