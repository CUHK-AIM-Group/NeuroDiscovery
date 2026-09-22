"""Bind whole original relations with additive, own-XML title evidence.

Original bibliography and previous reviews are never rewritten. The existing
completion guard is reused; only title proof added to an accepted record is new.
"""
from collections import defaultdict
from copy import deepcopy
import json
import mmap
from pathlib import Path
import sqlite3

from core.web.claim_layer_v1 import FIELDS, require
from neurooracle.src.claim_evidence_query import digest
from neurooracle.src.correlation_grouping import IndexTerms
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.relation_evidence import relation_id
from neurooracle.src.relation_evidence_dossier import observation
from neurooracle.src.shared_relation_catalog import check_file
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
from neurooracle.scripts.project_kg_systematic_review import record_at
from neurooracle.src.kg_title_identity_extension_v1 import OwningTitleWitnesses, require_monotonic_registry
from core.web.claim_layer_extension_v5 import complete_identity


def extend_original_relations(base_relations, base_dossiers, payload, campaign, witnesses):
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
        require_monotonic_registry(original_identities, identities)
        inputs.append(fp)
    registry = VerifiedPaperIdentities(identities)
    completion_ids = {e['claim_id'] for e in extension.get('existing_identity_completions', [])}
    historical_registries, historical_by_claim = {}, {}
    for cid, fp in extension.get('original_identity_registries', {}).items():
        require(cid in completion_ids, 'Historical identity registry has no explicit completion')
        if fp['path'] not in historical_registries:
            previous = json.loads(check_file(fp, full_hash=True).read_text(encoding='utf-8'))
            require_monotonic_registry(previous, identities)
            historical_registries[fp['path']] = VerifiedPaperIdentities(previous)
            inputs.append(fp)
        historical_by_claim[cid] = historical_registries[fp['path']]
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
                pmid_record = registry.resolve(node['metadata'])
                if pmid_record['status'] == 'verified':
                    record = identities['records'][pmid_record['paper_key'].removeprefix('pmid:')]
                    if (record.get('complete_title_evidence') is not None and
                            record != original_identities['records'].get(record['pmid'])):
                        witnesses.verify(record)
                obs = observation(node, historical_by_claim.get(cid, registry), source_reviews)
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
    require(set(historical_by_claim) <= seen, 'Historical identity registry does not refer to an original extension member')
    for pmid in sorted(new_pmids, key=lambda p: (identities['records'][p]['witness'].get('response_path', ''), p)):
        witnesses.require_authority(identities['records'][pmid], original_identities['records'])
    return relations, dossiers, inputs


def extend_base(base_relations, base_dossiers, payload, campaign):
    extension = payload.get('original_relation_extension') or {}
    if not extension:
        return base_relations, base_dossiers, []
    witnesses = OwningTitleWitnesses(extension.get('raw_authority_files', []), check_file)
    relations, dossiers, inputs = extend_original_relations(
        base_relations, base_dossiers, payload, campaign, witnesses)
    entries = extension.get('existing_identity_completions', [])
    if entries:
        identity_fp = extension['supplemental_identity_registry']
        identities = json.loads(check_file(identity_fp).read_text(encoding='utf-8'))
        original = json.loads(check_file(campaign['current_paper_identities']).read_text(encoding='utf-8'))
        registry = VerifiedPaperIdentities(identities)
        by_id = {}
        for entry in entries:
            cid, pmid = entry['claim_id'], entry['pmid']
            require(cid not in by_id, 'Repeated existing identity completion')
            require(cid in payload['source_reviews'], 'Identity completion has no bound scientific review')
            witnesses.require_authority(identities['records'][pmid], original['records'])
            by_id[cid] = entry
        seen = set()
        for dossier in dossiers:
            for i, obs in enumerate(dossier['observations']):
                cid = obs['claim_id']
                if cid in by_id:
                    require(cid not in seen, 'Existing completion duplicates an observation')
                    dossier['observations'][i] = complete_identity(obs, by_id[cid], registry)
                    seen.add(cid)
        require(seen == set(by_id), 'Existing completion refers to an absent observation')
    return relations, dossiers, list({fp['path']:fp for fp in [*inputs,*witnesses.inputs]}.values())
