"""Current-source projections for qualified claims and the historic gene watchlist.

Read-only, one full accepted graph boundary. Saves review projections/hashes, not
complete node/edge records, rollback images, new science, or new audit seals.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.kg_scoped_structure import source_documents

OUTPUT = j.OUTPUT / 'round45_semantic_sources'
HISTORIC = j.OUTPUT / 'round42_source_scope/ENDPOINT_REVIEW.jsonl'
FIELDS = ('subject_id', 'object_id', 'subject_name', 'object_name', 'predicate',
          'negated', 'subject_type', 'object_type', 'conditions', 'population',
          'raw_text', 'evidence', 'original_predicate', 'confidence', 'study_type')
INNER = ('original_predicate', 'subject_type', 'object_type', 'conditions',
         'population', 'polarity', 'evidence_direction', 'direction')


def label_key(value):
    return ' '.join(str(value or '').split()).casefold()


def paper_identity_matches(paper, public):
    # The census identifier column lowercases DOI; the source bibliography
    # deliberately preserves spelling. No suffix inference or fuzzy title match.
    return (str(paper.get('pmid')) == public['pmid'] and
            str(paper.get('doi') or '').strip().casefold() == public['doi'].strip().casefold())


def projection(row):
    md = row['metadata']; inner = md.get('metadata') or {}; audit = md.get('scope_reaudit') or {}
    return dict(claim_id=row['id'], claim_sha256=digest(row),
                science={k: md[k] for k in FIELDS if k in md},
                inner_science={k: inner[k] for k in INNER if k in inner},
                source_paper=md.get('source_paper'),
                prior_audit_sha256=digest(audit),
                prior_audit_decision=audit.get('decision'),
                audit_not_revalidated=True, complete_record_not_saved=True)


def progress(phase, **values):
    state = dict(status='READ_ONLY_RUNNING', phase=phase, at=j.utc_now(), **values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json', state)
    print(compact(state), flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(), 'inspection exists; reuse evidence')
    c = j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'accepted idle source required')
    require(Path(c['current_acceptance']['path']).parent.name == 'round44_source_resolution', 'expected accepted R44')
    require((j.OUTPUT/'round44_source_resolution/REPORT_VALIDATION.json').exists(), 'publish R44 first')
    code = j.fingerprint(Path(__file__))
    public_fp = j.fingerprint(OUTPUT/'PUBLIC_SOURCE_FETCH.json'); public = j.read_json(public_fp['path'])
    require(j.fingerprint(public['response']['path']) == public['response'], 'public XML changed')
    docs = source_documents([Path(public['response']['path']).read_text(encoding='utf8')])
    issues = rows(c['current_issues']['path']); issue_map = {r['claim_id']: r for r in issues}
    require(len(issue_map) == len(issues) == 21, 'selected semantic hold set differs')
    public_map = {r['claim_id']: r for r in public['selected']}
    require(set(public_map) == set(issue_map), 'public/hold set differs')
    scopes = j.read_json(c['current_scope_findings']['path'])
    current_gene = {(r['claim_id'], r['side']): r for r in rows(c['current_gene_holds']['path'])}
    old = rows(HISTORIC)
    require(len(old) == 449 and len({(r['claim_id'], r['side']) for r in old}) == 449, 'historic watchlist differs')
    watch = [dict(r, current_hold=current_gene[(r['claim_id'], r['side'])]) for r in old
             if (r['claim_id'], r['side']) in current_gene]
    watched_claims = {r['claim_id'] for r in watch}
    selected = set(issue_map) | watched_claims | set(scopes['current_claim_hashes'])
    pmids = set(docs)
    needle = re.compile('|'.join(re.escape(k) for k in sorted(issue_map)))
    names = {label_key(r['name']) for r in watch}
    target_ids = {r['current_hold']['current_node_id'] for r in watch}
    found = {}; same_paper = {}; target_nodes = {}; name_matches = []
    owned = defaultdict(list); exact = []; counts = Counter()
    census = j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    progress('FULL_CURRENT_SOURCE_CLAIMS_ENDPOINTS_AND_EXACT_REFERENCES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader, h):
        for kind, key, row in walk_graph(reader):
            counts[kind] += 1
            if kind == 'node':
                if key.startswith('CLM:'):
                    counts['claims'] += 1
                    if key in selected:
                        found[key] = projection(row)
                    if str((row['metadata'].get('source_paper') or {}).get('pmid') or '') in pmids:
                        same_paper[key] = projection(row)
                else:
                    if key in target_ids:
                        target_nodes[key] = dict(node_id=key, node_sha256=digest(row),
                            name=row.get('preferred_name'), aliases=row.get('aliases'),
                            external_ids=row.get('external_ids'), domain_tags=row.get('domain_tags'))
                    labels = [row.get('preferred_name'), *(row.get('aliases') or [])]
                    matched = [v for v in labels if label_key(v) in names]
                    if matched:
                        name_matches.append(dict(node_id=key, node_sha256=digest(row), name=row.get('preferred_name'),
                            matched_complete_labels=matched, domain_tags=row.get('domain_tags'),
                            external_ids=row.get('external_ids'), not_automatically_reusable=True))
            elif kind == 'edge' and edge_owner(row) in selected:
                owned[edge_owner(row)].append(dict(ordinal=int(key), edge_sha256=digest(row),
                    source_id=row['source_id'], target_id=row['target_id'], relation_type=row['relation_type'],
                    metadata_sha256=digest(row.get('metadata') or {})))
            if needle.search(compact(row)):
                hits = list(exact_references(row, set(issue_map)))
                if hits:
                    exact.append(dict(kind=kind, key=key, record_sha256=digest(row), references=hits))
            if counts[kind] % 1000000 == 0:
                progress(kind, counts=dict(counts))
        require(h.hexdigest() == c['current_graph']['sha256'], 'full source SHA differs')
    require(set(found) == selected and set(target_nodes) == target_ids, 'selected source coverage incomplete')
    require((counts['node'], counts['claims'], counts['edge']) == tuple(c['counts'][k] for k in ('nodes', 'claims', 'edges')), 'source census differs')
    for cid, issue in issue_map.items():
        r = found[cid]; p = public_map[cid]; paper = r['source_paper']
        require(r['claim_sha256'] == issue['current_node_sha256'] == p['claim_sha256'], 'source claim hash changed')
        require(paper_identity_matches(paper, p), 'own PMID/DOI differs: '+cid)
        require(digest(docs[p['pmid']]['abstract']) == p['public_abstract_sha256'], 'public abstract differs')
    for r in watch:
        require(found[r['claim_id']]['claim_sha256'] == r['current_hold']['claim_sha256'], 'current watch hash changed')
    for cid, sha in scopes['current_claim_hashes'].items():
        require(found[cid]['claim_sha256'] == sha, 'scope hash changed')
    require(sha256(Path(census['database']['path'])) == census['database']['sha256'], 'census full SHA differs')
    db = sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro', uri=True)
    paper_groups = {}
    for cid in issue_map:
        got = db.execute('SELECT node_sha,paper_sig,legacy_key FROM claims WHERE cid=?', (cid,)).fetchone()
        require(got and got[0] == found[cid]['claim_sha256'], 'claim/current census mismatch')
        paper_groups[cid] = dict(same_bibliography_claims=db.execute('SELECT COUNT(*) FROM claims WHERE paper_sig=?', (got[1],)).fetchone()[0],
                                same_source_key_claims=db.execute('SELECT COUNT(*) FROM claims WHERE legacy_key=?', (got[2],)).fetchone()[0])
    db.close()
    detail = sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro', uri=True)
    dependencies = {}
    for cid in issue_map:
        dependencies[cid] = dict(atoms=detail.execute('SELECT COUNT(*) FROM atoms WHERE source_mention_id=?', (cid,)).fetchone()[0],
            mappings_source=detail.execute('SELECT COUNT(*) FROM mappings WHERE source_id=?', (cid,)).fetchone()[0],
            mappings_target=detail.execute('SELECT COUNT(*) FROM mappings WHERE target_id=?', (cid,)).fetchone()[0])
    detail.close()
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json') == c and j.fingerprint(Path(__file__)) == code, 'source/code advanced')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    artifacts = {
        'CURRENT_CLAIM_PROJECTIONS.jsonl': [found[k] for k in sorted(found)],
        'SAME_PAPER_PROJECTIONS.jsonl': [same_paper[k] for k in sorted(same_paper)],
        'CURRENT_OWNED_EDGE_PROJECTIONS.jsonl': [dict(claim_id=k, edges=owned[k]) for k in sorted(owned)],
        'EXACT_SEMANTIC_REFERENCES.jsonl': exact,
        'CURRENT_HISTORIC_WATCHLIST.jsonl': watch,
        'CURRENT_COMPLETE_NAME_CANDIDATES.jsonl': name_matches,
        'CURRENT_GENE_NODE_WITNESSES.jsonl': [target_nodes[k] for k in sorted(target_nodes)],
    }
    for name, records in artifacts.items():
        j.atomic_text(OUTPUT/name, ''.join(compact(r)+'\n' for r in records))
    result = dict(status='INSPECTED_NOT_APPLIED', at=j.utc_now(), graph=c['current_graph'], source_acceptance=c['current_acceptance'],
        source_census=c['current_paper_census'], public_manifest=public_fp, code=code, full_source_sha_verified=True,
        full_census_sha_verified=True, counts=dict(counts), semantic_holds=21, original_watchlist=449,
        current_original_watchlist=len(watch), same_paper_claims=len(same_paper), selected_claims=len(found),
        paper_groups=paper_groups, detail_dependencies=dependencies,
        artifacts={name: j.fingerprint(OUTPUT/name) for name in artifacts},
        graph_modified=False, record_preimages_saved=False, current_audits_not_revalidated=True)
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json', result)
    progress('COMPLETED', claims=len(found), old_watchlist_remaining=len(watch), same_paper_claims=len(same_paper))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json', dict(status='FAILED', at=j.utc_now(),
            error=repr(error), graph_modified=False, record_preimages_saved=False))
        raise
