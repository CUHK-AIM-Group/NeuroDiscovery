"""Versioned own-authority identity completion for previously unverified claims.

Original bibliography, observations, accepted identities and scientific reviews
are immutable. An explicit completion retains the previous identity projection.
"""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

from core.web.claim_layer_extension_v2 import extend_base as extend_base_v2
from core.web.claim_layer_v1 import require
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.shared_relation_catalog import check_file
from neurooracle.scripts.extend_cached_kg_paper_authority_20260913 import authority_record


def complete_identity(observation, entry, registry):
    require(observation['claim_id'] == entry['claim_id'] and observation['claim_sha256'] == entry['claim_sha256'], 'Stale existing identity completion')
    require(observation['source_identity'] == entry['previous_source_identity'], 'Previous source identity changed')
    require(observation['source_identity']['status'] != 'verified', 'Accepted source identity cannot be replaced')
    require((observation.get('source_review') or {}).get('proposition_support') in {None, 'not_reviewed'}, 'Already adjudicated observation cannot change identity')
    identity = registry.resolve(observation['claim'])
    require(identity == entry['source_identity'] and identity['status'] == 'verified', 'Owning authority does not verify original bibliography')
    require(identity['paper_key'] == 'pmid:' + entry['pmid'], 'Identity completion belongs to a different paper')
    result = deepcopy(observation)
    result['previous_source_identity'] = deepcopy(result['source_identity'])
    result['source_identity'] = identity
    publication = registry.publication_review(identity['paper_key'])
    if publication != result.get('publication_review'):
        result['previous_publication_review'] = deepcopy(result.get('publication_review'))
        result['publication_review'] = publication
    return result


def extend_base(base_relations, base_dossiers, payload, campaign):
    relations, dossiers, inputs = extend_base_v2(base_relations, base_dossiers, payload, campaign)
    extension = payload.get('original_relation_extension') or {}
    entries = extension.get('existing_identity_completions', [])
    if not entries:
        return relations, dossiers, inputs
    identity_fp = extension['supplemental_identity_registry']
    identities = json.loads(check_file(identity_fp, full_hash=True).read_text(encoding='utf-8'))
    registry = VerifiedPaperIdentities(identities)
    files = {fp['path']:fp for fp in extension['raw_authority_files']}
    checked, by_id = {}, {}
    for entry in entries:
        cid, pmid = entry['claim_id'], entry['pmid']
        require(cid not in by_id, 'Repeated existing identity completion')
        require(cid in payload['source_reviews'], 'Identity completion has no bound scientific review')
        expected = identities['records'][pmid]
        witness = expected['witness']; fp = files.get(witness.get('response_path'))
        require(fp is not None and fp['sha256'] == witness['response_sha256'], 'Existing identity completion lacks owning raw authority')
        if fp['path'] not in checked:
            tree = ET.parse(check_file(fp, full_hash=True))
            checked[fp['path']] = {a.findtext('./MedlineCitation/PMID'):a for a in tree.getroot().findall('./PubmedArticle')}
            inputs.append(fp)
        article = checked[fp['path']].get(pmid)
        require(article is not None and authority_record(article, fp) == expected, 'Existing identity completion differs from its own PubMed article')
        by_id[cid] = entry
    seen = set()
    for dossier in dossiers:
        for i, observation in enumerate(dossier['observations']):
            cid = observation['claim_id']
            if cid in by_id:
                require(cid not in seen, 'Existing completion duplicates an observation')
                dossier['observations'][i] = complete_identity(observation, by_id[cid], registry)
                seen.add(cid)
    require(seen == set(by_id), 'Existing completion refers to an absent observation')
    return relations, dossiers, inputs
