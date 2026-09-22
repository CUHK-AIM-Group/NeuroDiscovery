"""Read a bounded batch of accepted evidence under one revision boundary.

The v7 loader adds own-XML title evidence; projection and scientific gates remain unchanged.
Projected responses share a before/after dependency check within one locked
request. Unprojected originals retain the existing source-bound lookup path.
"""
from copy import deepcopy

from core.web.claim_layer_v7 import AcceptedClaimLayer as PreviousLayer, require
from core.web.claim_batch_base_v1 import shared_snapshot

MAX_BATCH_QUERIES = 128


def validate_queries(queries):
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_BATCH_QUERIES:
        raise ValueError(f"Provide 1 to {MAX_BATCH_QUERIES} evidence queries")
    result = []
    for query in queries:
        if not isinstance(query, dict) or set(query) not in ({'claim_id'}, {'relation_id'}):
            raise ValueError('Each query requires exactly one claim_id or relation_id')
        name, value = next(iter(query.items()))
        prefix = 'CLM:' if name == 'claim_id' else 'REL:'
        if not isinstance(value, str) or len(value) > 300 or not value.startswith(prefix):
            raise ValueError('Invalid evidence ID')
        result.append({name: value})
    return result


class AcceptedClaimLayer(PreviousLayer):
    def __init__(self, campaign_path):
        super().__init__(campaign_path)
        self._batch_base_campaign = None
        self._batch_base_details = None

    def query_batch(self, *, queries):
        queries = validate_queries(queries)
        with self._lock:
            campaign = self._ensure_current()
            if campaign != self._batch_base_campaign:
                self._batch_base_campaign = None
                self._batch_base_details = None
            revision = self._revision(campaign)
            results = []
            for query in queries:
                cid = query.get('claim_id')
                rid = query.get('relation_id') or self._member_relations.get(cid)
                rid = self._layer_aliases.get(rid, rid)
                if rid in self._layer_details:
                    result = deepcopy(self._layer_details[rid])
                    require(cid is None or cid in result['original_claim_ids'],
                            'Original claim not in projected relation')
                    result['requested_claim_id'] = cid
                    result.update(revision)
                else:
                    if self._batch_base_details is None:
                        snapshot = shared_snapshot(self.path, campaign)
                        self._batch_base_details = snapshot
                        self._batch_base_campaign = campaign
                    if rid in self._batch_base_details:
                        result = deepcopy(self._batch_base_details[rid])
                        require(cid is None or cid in result['original_claim_ids'],
                                'Original claim not in shared relation')
                        result['requested_claim_id'] = cid
                        result.update(revision)
                    else:
                        result = super().query(**query)
                        require(all(result.get(key) == value for key, value in revision.items()),
                                'Graph revision changed during the evidence batch')
                results.append(result)
            # Return no partial response if any dependency or revision changed.
            self._check_current(campaign)
            return dict(results=results, query_count=len(results), **revision)
