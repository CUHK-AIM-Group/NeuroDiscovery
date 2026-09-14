"""Bind new claim-layer members to complete original relations, without edits."""
from collections import defaultdict
from copy import deepcopy
import json
import mmap
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

from core.web.claim_layer_v1 import FIELDS, require
from neurooracle.src.claim_evidence_query import digest
from neurooracle.src.correlation_grouping import IndexTerms
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.relation_evidence import relation_id
from neurooracle.src.relation_evidence_dossier import observation
from neurooracle.src.shared_relation_catalog import check_file
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
from neurooracle.scripts.project_kg_systematic_review import record_at
from neurooracle.scripts.extend_cached_kg_paper_authority_20260913 import authority_record


def extend_base(base_relations, base_dossiers, payload, campaign):
    extension = payload.get('original_relation_extension')
    if not extension:
        return base_relations, base_dossiers, []
    manifest_fp = extension['global_index_acceptance']
    manifest = json.loads(check_file(manifest_fp, full_hash=True).read_text(encoding='utf-8'))
    require(manifest['graph'] == campaign['current_graph'], 'Extension index belongs to a different graph')
    database = check_file(manifest['database'])
    inputs = [manifest_fp, manifest['database']]
    original_identities = json.loads(check_file(campaign['current_paper_identities']).read_text(encoding='utf-8'))
    identities = original_identities
    if extension.get('supplemental_identity_registry'):
        fp = extension['supplemental_identity_registry']
        identities = json.loads(check_file(fp, full_hash=True).read_text(encoding='utf-8'))
        require(all(identities['records'].get(p) == row for p, row in original_identities['records'].items()), 'An accepted identity was changed')
        require(identities.get('publication_reviews') == original_identities.get('publication_reviews'), 'Accepted publication status changed')
        inputs.append(fp)
    registry = VerifiedPaperIdentities(identities)
    source_reviews = json.loads(check_file(campaign['current_source_role_reviews']).read_text(encoding='utf-8'))
    terms = IndexTerms(VerifiedEntityTerms(json.loads(check_file(campaign['current_entity_terms']).read_text(encoding='utf-8'))))
    relations, dossiers = deepcopy(base_relations), deepcopy(base_dossiers)
    old_relations = {r['id'] for r in relations}
    old_claims = {o['claim_id'] for d in dossiers for o in d['observations']}
    additions, relation_rows, seen, new_pmids = defaultdict(list), {}, set(), set()
    db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        with check_file(campaign['current_graph']).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for entry in extension['members']:
                cid, rid = entry['claim_id'], entry['original_relation_id']
                require(cid not in seen and cid not in old_claims and rid not in old_relations, 'Duplicate or existing relation extension')
                seen.add(cid)
                witness = db.execute('SELECT node_sha,byte_offset,rid FROM observations WHERE cid=?', (cid,)).fetchone()
                require(witness == (entry['node_sha'], entry['byte_offset'], rid), 'Original member differs from accepted index')
                node = record_at(mm, entry['byte_offset'])
                require(node['id'] == cid and digest(node) == entry['node_sha'], 'Original source observation changed')
                key = terms.relation_key(node['metadata'])
                require(relation_id(key) == rid, 'Original relation identity changed')
                row = dict(id=rid, **dict(zip(FIELDS, key)))
                require(rid not in relation_rows or relation_rows[rid] == row, 'Inconsistent complete relation')
                relation_rows[rid] = row
                obs = observation(node, registry, source_reviews)
                additions[rid].append(obs)
                pmid = obs['source_identity']['paper_key'].removeprefix('pmid:') if obs['source_identity']['status'] == 'verified' else None
                if pmid and pmid not in original_identities['records'] and pmid in identities['records']:
                    new_pmids.add(pmid)
        for rid, observations in additions.items():
            expected = {row[0] for row in db.execute('SELECT cid FROM observations WHERE rid=?', (rid,))}
            require(expected == {o['claim_id'] for o in observations}, 'Incomplete original relation extension')
            relations.append(relation_rows[rid])
            dossiers.append(dict(relation_id=rid, observations=observations))
    finally:
        db.close()
    files = {fp['path']: fp for fp in extension.get('raw_authority_files', [])}
    checked = {}
    for pmid in sorted(new_pmids):
        expected = identities['records'][pmid]
        witness = expected['witness'];fp = files.get(witness.get('response_path'))
        require(fp is not None and fp['sha256'] == witness['response_sha256'], 'Supplemental identity lacks its own authority file')
        if fp['path'] not in checked:
            tree = ET.parse(check_file(fp, full_hash=True))
            checked[fp['path']] = {a.findtext('./MedlineCitation/PMID'): a for a in tree.getroot().findall('./PubmedArticle')}
            inputs.append(fp)
        article = checked[fp['path']].get(pmid)
        require(article is not None and authority_record(article, fp) == expected, 'Supplemental identity differs from owning PubMed record')
    return relations, dossiers, inputs
