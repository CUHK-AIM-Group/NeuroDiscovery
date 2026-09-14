"""Accepted scientific claim projection over immutable original observations.

The optional layer is an independently sealed annotation/identity release. It
never edits the source graph, changes a source observation, or counts retrieval
candidates as reviewed claims. Old shared IDs remain query aliases.
"""
from collections import OrderedDict
from copy import deepcopy
import json

from core.web.claim_evidence import AcceptedClaimEvidence, EvidenceUnavailable
from neurooracle.src.claim_evidence_query import paper_evidence
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import relation_id
from neurooracle.src.shared_relation_catalog import check_file

FIELDS = ('subject_id', 'subject_name', 'predicate', 'object_id', 'object_name')
VERSION = 'kg.accepted_claim_layer.v2'
VERSION_FIELDS = ('verified_work_key', 'verified_version_pmids', 'preferred_version_pmid', 'version_identity_witness')


def require(condition, message):
    if not condition:
        raise EvidenceUnavailable(message)


def merge_review(observation, review):
    """Scientific re-review cannot silently discard accepted work deduplication."""
    result = deepcopy(review)
    previous = observation.get('source_review') or {}
    for key in VERSION_FIELDS:
        if key in previous:
            require(key not in result or result[key] == previous[key], 'Conflicting accepted publication version identity')
            result[key] = deepcopy(previous[key])
    if previous:
        result['previous_source_review'] = deepcopy(previous)
    return result


def project(base_relations, base_dossiers, layer):
    """Validate complete relation ownership and project source-reviewed changes."""
    relations = {r['id']: deepcopy(r) for r in base_relations}
    dossiers = {d['relation_id']: deepcopy(d) for d in base_dossiers}
    affected, aliases, details = set(), {}, {}
    source_reviews = layer['source_reviews']
    applied_reviews = set()
    for group in layer['groups']:
        previous = group['source_relation_ids']
        require(previous and len(set(previous)) == len(previous), 'Empty or repeated source relation')
        require(not affected.intersection(previous), 'Overlapping approved claim assignments')
        require(set(previous) <= relations.keys() and set(previous) <= dossiers.keys(), 'Unknown source relation')
        affected.update(previous)
        target = group['claim']
        require(set(target) == set(FIELDS) and all(isinstance(v, str) and v for v in target.values()), 'Incomplete canonical proposition')
        target_id = group['shared_claim_id']
        require(target_id == relation_id(tuple(target[k] for k in FIELDS)), 'Canonical claim identity does not match its proposition')
        require(target_id not in relations or target_id in previous, 'Canonical target requires complete existing target scope')
        observations = [deepcopy(o) for rid in previous for o in dossiers[rid]['observations']]
        ids = {o['claim_id'] for o in observations}
        require(len(ids) == len(observations) and ids == set(group['original_claim_ids']), 'Incomplete or duplicate original observation closure')
        require(set(group['reviewed_claim_ids']) <= ids, 'Review outside canonical claim scope')
        for obs in observations:
            cid = obs['claim_id']
            if cid not in group['reviewed_claim_ids']:
                continue
            review = source_reviews.get(cid)
            require(review is not None and review['claim_sha256'] == obs['claim_sha256'], 'Stale or absent original observation review')
            require(obs['source_identity']['status'] == 'verified', 'New scientific support requires confirmed source identity')
            require(str(obs['claim']['source_paper'].get('pmid')) == review['pmid'], 'Review belongs to another paper')
            require(review.get('scope_note') and review.get('source_anchor'), 'Source-grounded scientific decision missing')
            obs['source_review'] = merge_review(obs, review)
            applied_reviews.add(cid)
        evidence = paper_evidence(dict(observations=observations))
        require(evidence['reviewed_supporting_article_count'] == group['reviewed_supporting_article_count'], 'Supporting paper count is inconsistent')
        for rid in previous:
            relations.pop(rid)
            dossiers.pop(rid)
            aliases[rid] = target_id
        relations[target_id] = dict(id=target_id, **target)
        dossiers[target_id] = dict(relation_id=target_id, observations=observations)
        details[target_id] = dict(shared_claim_id=target_id, claim=deepcopy(target), original_claim_ids=sorted(ids),
                                 observation_count=len(observations), original_relation_ids=sorted(previous),
                                 canonical_scope_note=group['scope_note'], **evidence)
    require(applied_reviews == set(source_reviews), 'Unused or unassigned source review')
    return relations, dossiers, aliases, details


class AcceptedClaimLayer(AcceptedClaimEvidence):
    def __init__(self, campaign_path):
        super().__init__(campaign_path)
        self._layer_fp = None
        self._layer_inputs = []
        self._layer_aliases = {}
        self._layer_details = {}
        self._layer_loaded_campaign = None

    def _check_current(self, campaign):
        super()._check_current(campaign)
        if campaign.get('current_claim_layer'):
            check_file(campaign['current_claim_layer'])
            if self._layer_loaded_campaign == campaign:
                for fp in self._layer_inputs:
                    check_file(fp)

    def _ensure_current(self):
        campaign = super()._ensure_current()
        if campaign == self._layer_loaded_campaign:
            return campaign
        self._layer_loaded_campaign = None
        self._layer_fp = None
        self._layer_inputs = []
        self._layer_aliases = {}
        self._layer_details = {}
        fp = campaign.get('current_claim_layer')
        if fp:
            layer = json.loads(check_file(fp, full_hash=True).read_text(encoding='utf-8'))
            require(layer.get('schema') in {VERSION, 'kg.accepted_claim_layer.v1'} and layer.get('status') == 'ACCEPTED', 'Claim layer is not accepted')
            require(layer.get('base_graph') == campaign['current_graph'] and
                    layer.get('base_acceptance') == campaign['current_acceptance'] and
                    layer.get('base_dossiers') == campaign['current_evidence_dossiers'], 'Claim layer belongs to another accepted baseline')
            payload = json.loads(check_file(layer['projection'], full_hash=True).read_text(encoding='utf-8'))
            require(layer.get('checks', {}).get('source_anchors_and_whole_scopes_validated') is True and
                    layer.get('checks', {}).get('original_observations_unchanged') is True, 'Claim layer validation is incomplete')
            with check_file(campaign['current_shared_relations']).open(encoding='utf-8') as stream:
                base_relations = [json.loads(line) for line in stream if line.strip()]
            with check_file(campaign['current_evidence_dossiers']).open(encoding='utf-8') as stream:
                base_dossiers = [json.loads(line) for line in stream if line.strip()]
            from core.web.claim_layer_extension_v2 import extend_base
            base_relations, base_dossiers, extra_inputs = extend_base(base_relations, base_dossiers, payload, campaign)
            relations, dossiers, aliases, details = project(base_relations, base_dossiers, payload)
            summaries, searches = [], {}
            for rid, relation in relations.items():
                dossier = dossiers[rid]
                evidence = paper_evidence(dossier)
                ids = sorted(o['claim_id'] for o in dossier['observations'])
                row = dict(shared_claim_id=rid, claim={k: relation[k] for k in FIELDS},
                           original_claim_ids=ids, observation_count=len(ids),
                           **{k: evidence[k] for k in self.COUNTS})
                terms = [rid, *row['claim'].values(), *ids]
                terms.extend(old for old, new in aliases.items() if new == rid)
                for paper in evidence['papers']:
                    for version in paper['publication_versions']:
                        terms.extend(str(version['bibliography'].get(k) or '') for k in ('pmid', 'doi', 'title'))
                searches[rid] = ' '.join(terms).casefold()
                summaries.append(row)
            require(sum(s['reviewed_supporting_article_count'] >= 2 for s in summaries) ==
                    layer['counts']['reviewed_multipaper_claims_after'], 'Claim layer total is inconsistent')
            self._summaries = sorted(summaries, key=lambda s: (-s['reviewed_supporting_article_count'], -s['article_count'], s['claim']['subject_name'].casefold(), s['shared_claim_id']))
            self._search = searches
            self._member_relations = {cid: s['shared_claim_id'] for s in summaries for cid in s['original_claim_ids']}
            self._layer_aliases, self._layer_details = aliases, details
            self._layer_fp = fp
            self._layer_inputs = [layer['projection'], *layer.get('review_inputs', []), *extra_inputs]
            self._details = OrderedDict()
        self._layer_loaded_campaign = campaign
        self._check_current(campaign)
        return campaign

    def _revision(self, campaign):
        result = super()._revision(campaign)
        if self._layer_fp:
            result.update(source='accepted_claim_layer', claim_layer_revision=self._layer_fp['sha256'])
        return result

    def query(self, *, claim_id=None, relation_id=None):
        if bool(claim_id) == bool(relation_id):
            raise ValueError('Provide exactly one original claim ID or shared claim ID')
        value = claim_id or relation_id
        if not isinstance(value, str) or len(value) > 300 or not value.startswith('CLM:' if claim_id else 'REL:'):
            raise ValueError('Invalid claim ID')
        with self._lock:
            campaign = self._ensure_current()
            rid = relation_id or self._member_relations.get(claim_id)
            rid = self._layer_aliases.get(rid, rid)
            if rid in self._layer_details:
                result = deepcopy(self._layer_details[rid])
                require(claim_id is None or claim_id in result['original_claim_ids'], 'Original claim not in projected relation')
                result['requested_claim_id'] = claim_id
                self._check_current(campaign)
                return dict(result, **self._revision(campaign))
            return super().query(claim_id=claim_id, relation_id=relation_id)
