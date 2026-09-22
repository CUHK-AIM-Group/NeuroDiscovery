"""Moving a warm reader to the adopted identical campaign keeps actual evidence."""
from copy import deepcopy
import json

import pytest

from core.web.claim_layer_v8 import AcceptedClaimLayer
from core.web.claim_evidence import EvidenceUnavailable
from core.web.test_claim_layer_v1 import release, attach


def test_warm_reader_rebind_preserves_cache_and_exact_query_results(tmp_path):
    path,campaign,_,_,dossier,payload=release(tmp_path)
    attach(tmp_path,path,campaign,payload)
    reader=AcceptedClaimLayer(path)
    queries=[{'claim_id':'CLM:0'},{'relation_id':dossier['relation_id']},{'claim_id':'CLM:3'}]
    expected=reader.query_batch(queries=queries)
    loaded=reader._layer_loaded_campaign
    details=reader._layer_details
    official=tmp_path/'adopted_campaign.json'
    official.write_bytes(path.read_bytes())
    reader.path=official
    assert reader.query_batch(queries=queries)==expected
    assert reader._layer_loaded_campaign is loaded and reader._layer_details is details
    # A different adopted layer cannot reuse the old cached evidence.
    changed=deepcopy(loaded)
    changed['current_claim_layer']['sha256']='0'*64
    official.write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises((ValueError,EvidenceUnavailable)):
        reader.query_batch(queries=queries)
