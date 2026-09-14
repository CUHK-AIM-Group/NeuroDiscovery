"""Bind a single manually resumed batch to R64 full-source nominal evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from inspect_kg_nominal_consolidation import OUTPUT as REVIEW, TARGETS
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from neurooracle.src import kg_nominal_consolidation as repair
from neurooracle.src.kg_gene_boundary_repair import word_interior_hits
from neurooracle.src.correlation_grouping import POLICY

OUTPUT = j.OUTPUT / 'round65_nominal_consolidation'
NEW_FILES = [j.REPO / p for p in (
    'neurooracle/src/kg_nominal_consolidation.py',
    'neurooracle/scripts/plan_kg_nominal_consolidation.py',
    'neurooracle/scripts/apply_kg_nominal_consolidation.py',
    'neurooracle/tests/test_kg_nominal_consolidation.py')]


def load_plan(baseline):
    fp = j.fingerprint(OUTPUT / 'PLAN.json'); plan = j.read_json(fp['path'])
    require(plan['status'] == 'REVIEWED_NOT_APPLIED' and plan['version'] == repair.VERSION
            and plan['relation_grouping'] == POLICY, 'wrong finite nominal policy')
    require(plan['graph'] == baseline['current_graph'] and plan['source_acceptance'] == baseline['current_acceptance']
            and plan['detail_store'] == baseline['current_detail_store'], 'source advancement')
    for item in [*plan['code'], plan['source_review'], plan['remaining_queue'], plan['closed_night_window']]: small_check(item)
    inspection = j.read_json(plan['source_review']['path'])
    for item in [*inspection['code'], inspection['tests'], *inspection['artifacts'].values()]: small_check(item)
    require(inspection['graph'] == plan['graph'] and inspection['source_acceptance'] == plan['source_acceptance']
            and all(inspection[k] for k in ('full_source_sha_verified', 'full_detail_sha_verified', 'full_census_sha_verified')),
            'R64 full-source proof missing')
    require(plan['events'] == rows(inspection['artifacts']['PROPOSED_CLAIM_EVENTS.jsonl']['path'])
            and plan['edge_events'] == rows(inspection['artifacts']['PROPOSED_EDGE_EVENTS.jsonl']['path']), 'event scope differs')
    require((plan['changed_claims'], plan['changed_endpoints'], plan['changed_edges'], len(plan['removed_nodes']),
             len(plan['preserved_source_nodes']), plan['held_endpoints']) == (36, 36, 64, 2, 6, 2190), 'finite batch counts differ')
    require(plan['canonical_targets'] == TARGETS and not plan['new_literals'] and plan['manual_authorization']['one_batch_only'],
            'manual scope differs')
    require(plan['manual_authorization']['no_new_goal_or_automation'] and plan['original_watchlist_449_remaining_after'] == 1,
            'authorization or historic hold differs')
    return fp, plan


def main():
    require(not (OUTPUT / 'PLAN.json').exists(), 'frozen plan exists')
    c = j.read_json(j.OUTPUT / 'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None and not c['rollback_retention'], 'active writer or rollback retention')
    fp = j.fingerprint(REVIEW / 'SOURCE_INSPECTION.json'); inspection = j.read_json(fp['path'])
    require(inspection['graph'] == c['current_graph'] and inspection['source_acceptance'] == c['current_acceptance'], 'inspection not current')
    for item in [c['current_acceptance'], c['current_gene_holds'], c['current_gene_review_summary'], *inspection['code'],
                 inspection['tests'], *inspection['artifacts'].values()]: small_check(item)
    old = j.read_json(c['current_acceptance']['path'])
    for item in old['code']: small_check(item)
    window_fp = j.fingerprint(c['night_window']); window = j.read_json(window_fp['path'])
    require(window['status'] == 'COMPLETED_WITH_DOCUMENTED_REMAINDERS' and window['heartbeat_stopped'], 'old night must stay closed')
    read = lambda name: rows(inspection['artifacts'][name + '.jsonl']['path'])
    witnesses = {r['node_id']: r for r in read('CURRENT_TARGET_WITNESSES')}
    genes = {r['node_id']: r for r in read('CURRENT_GENE_WITNESSES')}
    incidences = read('CURRENT_TARGET_INCIDENCES'); projections = {r['claim_id']: r for r in read('CURRENT_CLAIM_PROJECTIONS')}
    events = read('PROPOSED_CLAIM_EVENTS'); edges = read('PROPOSED_EDGE_EVENTS'); queued = read('SOURCE_QUEUED_ENDPOINTS')
    for name, target in TARGETS.items(): repair.check_node(witnesses[target], name)
    for row in witnesses.values(): repair.check_node(row, row['name'])
    for row in incidences: repair.check_types(row['name'], row['outer_type'], row['inner_type'])
    reviews = {}
    for event in events:
        projection = projections[event['claim_id']]
        require(event['claim_sha256'] == projection['claim_sha256'], 'projection hash differs')
        for ch in event['changes']:
            side = ch['side']; outer = projection['science']; inner = projection['inner_science']
            repair.check_types(ch['name'], outer.get(side + '_type'), inner.get(side + '_type'))
            require(outer[side + '_name'] == ch['name'] and outer[side + '_id'] == ch['old_id'], 'projection endpoint differs')
            review = dict(claim_sha256=event['claim_sha256'], change=ch,
                          outer_type=outer.get(side + '_type'), inner_type=inner.get(side + '_type'))
            if ch['reason'] == 'gene_endpoint_to_existing_nominal_concept':
                gene = genes[ch['old_id']]
                review['word_interior_alias_hits'] = word_interior_hits(ch['name'], [gene['name'], *gene['aliases']])
                require(review['word_interior_alias_hits'] and 'T028' in gene['semantic_types'], 'gene boundary proof missing')
            reviews[event['claim_id'] + '|' + side] = review
    fixed = {(r['claim_id'], r['side']) for r in queued}
    queue = rows(c['current_gene_holds']['path'])
    require(len(queue) == 2204 and len(fixed) == 14 and all(r in queue for r in queued), 'fresh review queue differs')
    remaining = [r for r in queue if (r['claim_id'], r['side']) not in fixed]
    protected = {r['claim_id'] for field in ('current_issues', 'current_structure_holds') for r in rows(c[field]['path'])}
    protected.update(j.read_json(c['current_scope_findings']['path'])['current_claim_hashes'])
    require(not protected & {e['claim_id'] for e in events}, 'separate scientific hold overlaps')
    previous_summary = j.read_json(c['current_gene_review_summary']['path'])
    require(previous_summary['original_watchlist_remaining'] == 1, 'historic hold changed')
    removed = read('REMOVABLE_SOURCE_NODES'); preserved = read('PRESERVED_SOURCE_NODES')
    require(len(removed) == 2 and len(preserved) == 6 and all(not r['reasons'] and not any(r['detail_references'].values()) for r in removed),
            'source detail retirement boundary differs')
    write_rows(OUTPUT / 'REMAINING_GENE_ENDPOINTS.jsonl', remaining)
    plan = dict(version=repair.VERSION, status='REVIEWED_NOT_APPLIED', at=j.utc_now(), graph=c['current_graph'],
        source_acceptance=c['current_acceptance'], detail_store=c['current_detail_store'], source_review=fp,
        code=[j.fingerprint(p) for p in NEW_FILES], events=events, edge_events=edges, endpoint_reviews=reviews,
        target_witnesses=witnesses, gene_witnesses=genes, source_incidences=incidences,
        existing_targets={nid: r['node_sha256'] for nid, r in {**witnesses, **genes}.items()},
        canonical_targets=TARGETS, redirects=inspection['redirects'], removed_nodes=removed, preserved_source_nodes=preserved,
        expected_shared_claim_ids=inspection['expected_shared_claim_ids'], relation_grouping=POLICY,
        changed_claims=36, changed_endpoints=36, changed_edges=64, reused_existing_nodes=7, new_literals=[], added_literal_nodes=0,
        held_endpoints=len(remaining), expanded_review_endpoints=2204, repaired_queued_endpoints=14,
        original_watchlist_449_remaining_after=1, original_watchlist_449_repaired=0,
        remaining_queue=j.fingerprint(OUTPUT / 'REMAINING_GENE_ENDPOINTS.jsonl'), closed_night_window=window_fp,
        manual_authorization=dict(user_request='继续', resumed_on='2026-09-10', one_batch_only=True, no_new_goal_or_automation=True),
        source_full_sha_verified=True, source_census_full_sha_verified=True, record_preimages_saved=False, graph_backups=0,
        scientific_reaudit_performed=False, scientific_followup_candidates=[
            dict(claim_id='CLM:61f3088497c0', issue='activates versus stored hypoactivation/impairment wording; source review required'),
            dict(claim_id='CLM:a0340c2ca3ca', issue='distinguishes versus stored similarities wording; source review required')],
        boundaries=['Seven finite identical full names, not global name-based equivalence or identical measurement definitions.',
            'Preserve all raw text, types, conditions, source evidence and original audit; no scientific revalidation.',
            'Two reference-free concepts removed; six original detail-store anchors remain unchanged.',
            'No claim or edge deletion, new concept, model, training, formal full_v2 change, goal or automation.'])
    j.atomic_json(OUTPUT / 'PLAN.json', plan)
    load_plan(c)
    print('REVIEWED_NOT_APPLIED: 36 claims, 36 endpoints, 64 edge references, 2 retired concepts, 6 retained source anchors', flush=True)


if __name__ == '__main__': main()
