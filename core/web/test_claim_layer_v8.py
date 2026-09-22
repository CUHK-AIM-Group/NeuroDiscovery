"""Run real batch/revision/HTTP regression cases with the additive-title reader."""
import pytest

from core.web import test_claim_layer_v6 as legacy
from core.web.claim_layer_v8 import AcceptedClaimLayer, validate_queries
from core.web.test_claim_layer_v6 import (
    prepared,
    test_mixed_original_alias_canonical_and_unprojected_batch_is_exact,
    test_shared_base_snapshot_matches_single_queries_and_reloads_on_revision_change,
    test_shared_base_batch_does_not_repeat_single_lookup,
    test_dependency_change_before_or_during_batch_returns_no_evidence,
    test_campaign_change_during_batch_fails_closed,
    test_projected_batch_checks_every_dependency_at_both_boundaries_only,
    test_invalid_batch_rejected_before_reading,
    test_http_batch_matches_single_queries_and_rejects_partial_results,
)


@pytest.fixture(autouse=True)
def additive_reader(monkeypatch):
    monkeypatch.setattr(legacy, 'AcceptedClaimLayer', AcceptedClaimLayer)
    monkeypatch.setattr(legacy, 'validate_queries', validate_queries)
