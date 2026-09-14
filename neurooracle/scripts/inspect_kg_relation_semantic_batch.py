"""R66 complete current incidence review for thirteen finite names and two claims.

Read-only source scan. Persist bounded science projections and hashes, never
complete record preimages or a second KG. Historical matches are search leads,
not inherited scientific approval or automatic equivalence decisions.
"""
from collections import Counter, defaultdict
from pathlib import Path
import os
import re
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from inspect_kg_remaining_scoped_literals import scope_projection, detail_references, owned_projection
from inspect_kg_semantic_hold_sources import projection
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.relation_evidence import name_key

OUTPUT = j.OUTPUT / 'round66_relation_semantic_review'
R60 = j.OUTPUT / 'round60_multiple_name_scope_review'
NAMES = frozenset(('bilateral hippocampal volume', 'dementia diagnosis',
    'hippocampal subfield volume', 'hippocampal volume', 'left hippocampal volume',
    'lifetime major depressive disorder', 'persistent cognitive impairment',
    'reduced hippocampal volume', 'smaller hippocampal volume',
    'structural-functional connectivity coupling', 'treatment-resistant depression',
    'white matter fractional anisotropy', 'white matter microstructure'))
SEMANTIC = frozenset(('CLM:61f3088497c0', 'CLM:a0340c2ca3ca'))
EXTRA_IDS = frozenset(('CLM_CONCEPT:supplementary_motor_area_activation',
    'COGAT_CONCEPT:trm_4a3fd79d0a17f', 'CUI:C1420009',
    'CLM_CONCEPT:resting_state_functional_connectivity_patterns_74fa96109660'))


def selected_groups(values):
    groups = [g for g in values if g['name'] in NAMES]
    require(len(groups) == len(NAMES) and {g['name'] for g in groups} == NAMES,
            'finite thirteen-name search scope differs')
    require(all(set(g['candidate_ids']) == set(g['current_candidate_ids']) for g in groups),
            'historical search candidate set inconsistent')
    return groups


def incidence_rows(key, row, targets):
    if not key.startswith('CLM:'): return []
    md = row['metadata']; inner = md.get('metadata') or {}
    return [dict(node_id=md[s + '_id'], claim_id=key, side=s,
        name=md.get(s + '_name'), claim_sha256=digest(row),
        outer_type=md.get(s + '_type'), inner_type=inner.get(s + '_type'))
        for s in ('subject', 'object') if md.get(s + '_id') in targets]


def progress(phase, **values):
    state = dict(status='READ_ONLY_INSPECTING', pid=os.getpid(), at=j.utc_now(), phase=phase, **values)
    j.atomic_json(OUTPUT / 'STATE.json', state); print(compact(state), flush=True)


def main():
    require(not (OUTPUT / 'SOURCE_INSPECTION.json').exists(), 'frozen R66 inspection exists')
    tests_fp = j.fingerprint(OUTPUT / 'TEST_RESULTS.xml')
    suite = ET.parse(tests_fp['path']).getroot().find('testsuite')
    require(int(suite.get('tests')) >= 10 and all(int(suite.get(k, 0)) == 0 for k in ('errors', 'failures', 'skipped')),
            'inspection tests not complete')
    c = j.read_json(j.OUTPUT / 'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(Path(c['current_acceptance']['path']).parent.name == 'round65_nominal_consolidation', 'requires accepted R65')
    required = ('current_acceptance', 'current_paper_census', 'current_gene_holds', 'current_issues',
        'current_structure_holds', 'current_scope_findings', 'current_nominal_semantic_followups')
    for key in required: small_check(c[key])
    acceptance = j.read_json(c['current_acceptance']['path'])
    for fp in acceptance['code']: small_check(fp)
    historical = j.fingerprint(R60 / 'MULTIPLE_NAME_GROUP_REVIEW.jsonl')
    groups = selected_groups(rows(historical['path']))
    target_ids = {nid for g in groups for nid in g['candidate_ids']}
    queue = rows(c['current_gene_holds']['path'])
    queued = [r for r in queue if r['name'] in NAMES]
    historic_queued = [r for r in queued if any(r['claim_id'] in g['queued_claim_ids'] for g in groups)]
    require(len(queued) == 76 and len(historic_queued) == 47 and len(target_ids) == 33,
            'finite current/historical candidate scope differs')
    selected = {r['claim_id'] for r in queued} | set(SEMANTIC)
    genes = {r['current_node_id'] for r in queued}
    node_ids = target_ids | genes | EXTRA_IDS
    followups = rows(c['current_nominal_semantic_followups']['path'])
    require({r['claim_id'] for r in followups} == SEMANTIC, 'new followup set differs')
    census = j.read_json(c['current_paper_census']['path'])
    code = [j.fingerprint(p) for p in (Path(__file__), j.REPO / 'neurooracle/tests/test_kg_relation_semantic_inspection.py')]
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    claims = {}; nodes = {}; incidences = []; owned = []; references = []
    matches = defaultdict(set); counts = Counter(); other = []
    needle = re.compile('|'.join(re.escape(n) for n in sorted(target_ids)))
    search_names = {name_key(n).casefold() for n in NAMES}
    progress('FULL_CURRENT_THIRTEEN_GROUP_INCIDENCES_AND_SEMANTICS', groups=13, candidate_nodes=33, queued_endpoints=76)
    with hashed_reader(Path(c['current_graph']['path'])) as (reader, h):
        for kind, key, row in walk_graph(reader):
            counts[kind] += 1
            if kind == 'node':
                if key in node_ids: nodes[key] = scope_projection(row)
                if key.startswith('CLM:'):
                    counts['claims'] += 1
                    inc = incidence_rows(key, row, target_ids)
                    if inc: selected.add(key); incidences.extend(inc)
                    if key in selected:
                        p = projection(row)
                        p['preferred_name'] = row.get('preferred_name')
                        p['outer_metadata_keys'] = sorted(row['metadata'])
                        p['inner_metadata_keys'] = sorted((row['metadata'].get('metadata') or {}))
                        claims[key] = p
                else:
                    for label in {name_key(v).casefold() for v in [row.get('preferred_name'), *(row.get('aliases') or [])]} & search_names:
                        matches[label].add(key)
                        nodes[key] = scope_projection(row)
            elif kind == 'edge':
                owner = edge_owner(row)
                if owner in selected:
                    item = owned_projection(int(key), row)
                    item['scientific_metadata'] = {k:v for k,v in (row.get('metadata') or {}).items()
                        if k in {'claim_id', 'negated', 'original_predicate'}}
                    owned.append(item)
                if not owner and {row.get('source_id'), row.get('target_id')} & target_ids:
                    other.append(owned_projection(int(key), row))
            if needle.search(compact(row)):
                for ref in exact_references(row, target_ids):
                    references.append(dict(kind=kind, key=key, node_id=ref['claim_id'], json_path=ref['json_path'],
                        record_sha256=digest(row), **(dict(edge_owner=edge_owner(row)) if kind == 'edge' else {})))
            if counts[kind] % 1000000 == 0: progress(kind, counts=dict(counts))
        require(h.hexdigest() == c['current_graph']['sha256'], 'full current graph SHA differs')
    require(set(claims) == selected and node_ids <= set(nodes), 'current selected scope incomplete')
    require((counts['node'], counts['claims'], counts['edge']) == tuple(c['counts'][k] for k in ('nodes', 'claims', 'edges')),
            'current complete source counts differ')
    for r in queued:
        p = claims[r['claim_id']]; s = r['side']
        require(p['claim_sha256'] == r['claim_sha256'] and p['science'][s+'_id'] == r['current_node_id']
                and p['science'][s+'_name'] == r['name'], 'queued source differs')
    for r in followups:
        require(r['claim_sha256'] == claims[r['claim_id']]['claim_sha256'], 'semantic followup binding differs')
    current_groups = []
    for g in groups:
        ids = matches[name_key(g['name']).casefold()]
        require(ids == set(g['candidate_ids']), 'current label candidates changed: ' + g['name'])
        inc = [r for r in incidences if r['node_id'] in ids]
        current_groups.append(dict(name=g['name'], candidate_ids=sorted(ids),
            queued_endpoints=sum(r['name'] == g['name'] for r in queued), current_incidences=len(inc),
            distinct_incident_names=sorted({r['name'] for r in inc}),
            historical_search_reasons=g['review_reasons'], scientific_approval_inherited=False))
    db = sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro', uri=True)
    try:
        for cid, p in claims.items():
            require(db.execute('SELECT node_sha FROM claims WHERE cid=?', (cid,)).fetchone() == (p['claim_sha256'],), 'current census binding differs')
    finally: db.close()
    require(sha256(Path(census['database']['path'])) == census['database']['sha256'], 'complete census SHA differs')
    db = sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro', uri=True)
    try: details = detail_references(db, target_ids)
    finally: db.close()
    require(sha256(Path(c['current_detail_store']['path'])) == c['current_detail_store']['sha256'], 'complete details SHA differs')
    data = {'CURRENT_GROUPS.jsonl': current_groups, 'CURRENT_CLAIM_PROJECTIONS.jsonl': [claims[cid] for cid in sorted(claims)],
        'CURRENT_NODE_WITNESSES.jsonl': [nodes[nid] for nid in sorted(nodes)], 'CURRENT_TARGET_INCIDENCES.jsonl': incidences,
        'CURRENT_OWNED_EDGE_PROJECTIONS.jsonl': owned, 'CURRENT_EXACT_REFERENCES.jsonl': references,
        'CURRENT_NONCLAIM_RELATIONS.jsonl': other, 'SOURCE_QUEUED_ENDPOINTS.jsonl': queued}
    for name, values in data.items(): write_rows(OUTPUT / name, values)
    require(j.read_json(j.OUTPUT / 'CAMPAIGN.json') == c and all(j.fingerprint(fp['path']) == fp for fp in code), 'source/code changed during inspection')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    receipt = dict(status='INSPECTED_NOT_APPLIED', at=j.utc_now(), graph=c['current_graph'], source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'], census=c['current_paper_census'], code=code, tests=tests_fp,
        historical_search_leads=historical, semantic_followups=c['current_nominal_semantic_followups'],
        source_bindings={k:c[k] for k in required}, full_source_sha_verified=True, full_census_sha_verified=True,
        full_detail_sha_verified=True, groups=len(groups), queued_endpoints=len(queued), selected_claims=len(claims),
        candidate_nodes=len(target_ids), historical_queued_endpoints=len(historic_queued),
        additional_same_name_queued_endpoints=len(queued)-len(historic_queued),
        incidences=len(incidences), owned_edges=len(owned), detail_references=details,
        graph_modified=False, record_preimages_saved=False, artifacts={n:j.fingerprint(OUTPUT/n) for n in data})
    j.atomic_json(OUTPUT / 'SOURCE_INSPECTION.json', receipt)
    progress('COMPLETED', selected_claims=len(claims), incidences=len(incidences), owned_edges=len(owned))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        j.atomic_json(OUTPUT / 'STATE.json', dict(status='FAILED', at=j.utc_now(), pid=os.getpid(), error=repr(error), graph_modified=False)); raise
