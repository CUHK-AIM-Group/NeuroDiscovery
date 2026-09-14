from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plan_kg_historic_literal_repair import candidate_target, inspection_bridge
from neurooracle.src.kg_literal_endpoint_repair import literal_node as imaging_literal_node


def test_no_existing_name_gets_generic_shared_literal():
    node, reason, new = candidate_target('structural alterations', {}, {}, {})
    assert new and reason is None and node['domain_tags'] == ['claim_concept']


def test_casefold_only_blocks_new_duplicates_not_merges():
    node = dict(id='CLM_CONCEPT:old', preferred_name='Structural alterations')
    out, reason, new = candidate_target('structural alterations', {'structural alterations': {node['id']}}, {node['id']: node}, {})
    assert out is None and not new and reason == 'existing_case_variant_or_alias_requires_separate_review'


def test_multiple_existing_names_and_unreviewed_metadata_are_held():
    assert candidate_target('brain structural alterations', {'brain structural alterations': {'a','b'}}, {}, {})[1] == 'multiple_existing_complete_names_or_aliases'
    node = dict(id='CLM_CONCEPT:old', preferred_name='structural alterations', metadata={})
    assert candidate_target('structural alterations', {'structural alterations': {node['id']}}, {node['id']: node}, {})[1] == 'existing_literal_needs_separate_scope_review'


def test_safe_existing_imaging_literal_reused_without_new_metadata():
    name = 'parietal cortical surface area'; node = imaging_literal_node(name)
    result, reason, new = candidate_target(name, {name: {node['id']}}, {node['id']: node}, {node['id']: [dict(name=name, declared_roles=['imaging_marker'])]})
    assert result == node and reason is None and not new


def bridge_fixture():
    review = dict(full_source_sha_verified=True, reviewed_endpoints=447, graph={'sha256': 'old'})
    baseline = dict(current_graph={'sha256': 'new'})
    accepted = dict(status='CURRENT_RESEARCH_STATEMENT_RETIREMENT_APPLIED', graph=baseline['current_graph'], source_graph=review['graph'],
        checks=dict(all_kept_source_node_and_edge_record_digests_reproduced=True, all_existing_concepts_and_detail_store_unchanged=True),
        changed=dict(deleted_claims=4, deleted_owned_edges=12, identity_claim_nodes=0))
    return review, baseline, accepted


def test_exact_kept_record_bridge_required():
    review, baseline, accepted = bridge_fixture(); inspection_bridge(review, baseline, accepted)
    accepted['source_graph'] = {'sha256': 'unrelated'}
    with pytest.raises(ValueError, match='source advancement'): inspection_bridge(review, baseline, accepted)


@pytest.mark.parametrize('what', ['kept_proof', 'concept_proof', 'identity_changes', 'wrong_count'])
def test_bridge_cannot_hide_mutation(what):
    review, baseline, accepted = bridge_fixture()
    if what == 'kept_proof': accepted['checks']['all_kept_source_node_and_edge_record_digests_reproduced'] = False
    elif what == 'concept_proof': accepted['checks']['all_existing_concepts_and_detail_store_unchanged'] = False
    elif what == 'identity_changes': accepted['changed']['identity_claim_nodes'] = 1
    else: accepted['changed']['deleted_claims'] = 5
    with pytest.raises(ValueError): inspection_bridge(review, baseline, accepted)
