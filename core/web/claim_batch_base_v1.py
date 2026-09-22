"""Load accepted shared base evidence once for a revision-bound batch reader.

These are the same catalog, census, dossier summary and node-seal checks used
by query_claim_evidence, applied to all shared dossiers in one bounded pass.
Singletons keep the original graph lookup path. No scientific review changes.
"""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from neurooracle.src.claim_evidence_query import paper_evidence, require
from neurooracle.src.relation_evidence_dossier import summarize
from neurooracle.src.shared_relation_catalog import check_file, current_shared_relations


def shared_snapshot(campaign_path, campaign):
    path = Path(campaign_path)
    require(json.loads(path.read_text(encoding='utf8')) == campaign, 'Campaign changed before shared batch load')
    catalog = {group['id']: group for group in current_shared_relations(path)}
    receipt = json.loads(check_file(campaign['current_acceptance'], full_hash=True).read_text(encoding='utf8'))
    census_fp = campaign['current_paper_census']
    require(receipt.get('current_paper_census') == census_fp and
            receipt['checks'].get('all_current_census_rows_independently_verified'), 'Unvalidated claim census')
    census = json.loads(check_file(census_fp, full_hash=True).read_text(encoding='utf8'))
    require(census['graph'] == campaign['current_graph'], 'Census graph differs from current snapshot')
    database = check_file(census['database'])
    expected = defaultdict(dict)
    con = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        # Include every member, including any incorrect shared=0 member of an
        # indexed relation, so a corrupt census cannot evade closure checks.
        for rid, cid, seal, shared in con.execute('SELECT relation_id,cid,node_sha,shared FROM claims'):
            if rid in catalog:
                require(bool(shared), 'Catalog relation contains an unshared census member')
                expected[rid][cid] = seal
    finally:
        con.close()
    fp = campaign.get('current_evidence_dossiers')
    require(fp and receipt.get('evidence_dossiers') == fp and
            receipt['checks'].get('shared_observation_dossiers_complete'), 'Unvalidated evidence dossier')
    result = {}
    with check_file(fp, full_hash=True).open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip():
                continue
            dossier = json.loads(line)
            rid = dossier['relation_id']
            require(rid in catalog and rid not in result, 'Unknown or duplicate current dossier')
            group, seals = catalog[rid], expected[rid]
            require({member['claim_id'] for member in group['members']} == set(seals),
                    'Catalog and complete census membership differ')
            require(summarize(group, dossier['observations']) == dossier, 'Dossier summary differs')
            require({o['claim_id']: o['claim_sha256'] for o in dossier['observations']} == seals,
                    'Dossier census seals differ')
            result[rid] = dict(shared_claim_id=rid,
                claim={k: group[k] for k in ('subject_id', 'subject_name', 'predicate', 'object_id', 'object_name')},
                original_claim_ids=sorted(seals), observation_count=len(seals), **paper_evidence(dossier),
                scope='complete current graph membership of this reviewed fine relation; not complete literature recall')
    require(set(result) == set(catalog), 'Missing shared dossier in accepted batch snapshot')
    for name in ('current_graph', 'current_acceptance', 'current_paper_census', 'current_shared_relations',
                 'current_evidence_dossiers', 'current_source_role_reviews', 'current_entity_terms', 'current_paper_identities'):
        if campaign.get(name):
            check_file(campaign[name])
    check_file(census['database'])
    require(json.loads(path.read_text(encoding='utf8')) == campaign, 'Campaign changed during shared batch load')
    return result
