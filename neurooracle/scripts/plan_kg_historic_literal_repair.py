"""R48 finite historic-queue identity plan rebound to the actual R46 graph."""
from collections import Counter, defaultdict
from pathlib import Path
import os
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_historic_literal_candidates import node_projection
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_historic_literal_repair as repair
from neurooracle.src.claim_semantics import declared_type_atoms
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import name_key, relation_id
from neurooracle.src.correlation_grouping import POLICY, IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = j.OUTPUT / 'round48_historic_literal_repair'
REVIEW = j.OUTPUT / 'round47_historic_mentions'


def progress(phase, **values):
    state = dict(status='READ_ONLY_PLANNING', pid=os.getpid(), at=j.utc_now(), phase=phase, **values)
    j.atomic_json(OUTPUT / 'PLAN_STATE.json', state); print(compact(state), flush=True)


def inspection_bridge(review, baseline, accepted):
    """Old projections are usable only across the exact keep-everything-else batch."""
    require(review['full_source_sha_verified'] and review['reviewed_endpoints'] == 447, 'historic review incomplete')
    require(accepted['status'] == 'CURRENT_RESEARCH_STATEMENT_RETIREMENT_APPLIED'
        and accepted['graph'] == baseline['current_graph']
        and accepted['source_graph'] == review['graph'], 'unreviewed source advancement')
    require(accepted['checks']['all_kept_source_node_and_edge_record_digests_reproduced']
        and accepted['checks']['all_existing_concepts_and_detail_store_unchanged'], 'kept-record bridge missing')
    require(accepted['changed']['deleted_claims'] == 4
        and accepted['changed']['deleted_owned_edges'] == 12
        and accepted['changed']['identity_claim_nodes'] == 0, 'bridge mutation differs')


def candidate_target(name, matches, nodes, incidents):
    """Case-folding only blocks ambiguous duplicates; it never merges names."""
    choices = matches.get(name_key(name).casefold(), set())
    if not choices:
        return repair.literal_node(name), None, True
    if len(choices) != 1:
        return None, 'multiple_existing_complete_names_or_aliases', False
    node = nodes[next(iter(choices))]
    if name_key(node.get('preferred_name')) != name_key(name):
        return None, 'existing_case_variant_or_alias_requires_separate_review', False
    reason = repair.reuse_gate(node, name, incidents.get(node['id'], []))
    return (None if reason else node), reason, False


def main():
    require(not (OUTPUT / 'PLAN.json').exists(), 'plan exists; inspect/resume')
    c = j.read_json(j.OUTPUT / 'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None and not c['rollback_retention'], 'source boundary')
    receipt = j.read_json(c['current_acceptance']['path'])
    inspection_fp = j.fingerprint(REVIEW / 'SOURCE_INSPECTION.json')
    inspection = j.read_json(inspection_fp['path'])
    inspection_bridge(inspection, c, receipt)
    for fp in [inspection['code'], *inspection['artifacts'].values(), *receipt['code']]: small_check(fp)
    reviews = {repair.review_key(r['claim_id'], r['side']): r
        for r in rows(inspection['artifacts']['CURRENT_ENDPOINT_ADJUDICATION.jsonl']['path'])}
    require(len(reviews) == 447, 'historic allowlist differs')
    witnesses = {r['node_id']: r for r in rows(inspection['artifacts']['GENE_WITNESSES.jsonl']['path'])}
    prior_nodes = {r['node_id']: r for r in rows(inspection['artifacts']['EXISTING_NODE_WITNESSES.jsonl']['path'])}
    prior_incidents = rows(inspection['artifacts']['EXISTING_INCIDENTS.jsonl']['path'])
    held = rows(c['current_gene_holds']['path']); queue = {(r['claim_id'], r['side']): r for r in held}
    require(len(queue) == len(held) == 5934, 'current queue differs')
    protected = {r['claim_id'] for f in ('current_issues', 'current_structure_holds') for r in rows(c[f]['path'])}
    protected.update(j.read_json(c['current_scope_findings']['path'])['current_claim_hashes'])
    for r in reviews.values():
        q = queue.get((r['claim_id'], r['side']))
        require(q and q['claim_sha256'] == r['claim_sha256'] and q['current_node_id'] == r['current_node_id'], 'reviewed current hold changed')
    proposals = {k: r for k, r in reviews.items() if not r['current_gate'] and r['claim_id'] not in protected}
    selected = {r['claim_id'] for r in reviews.values()}
    names = {name_key(r['name']).casefold() for r in proposals.values()}
    generated = {repair.literal_node(r['name'])['id'] for r in proposals.values()}
    code = [j.fingerprint(p) for p in (Path(__file__), j.REPO / 'neurooracle/src/kg_historic_literal_repair.py')]
    census = j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    claims = {}; nodes = {}; matches = defaultdict(set); owned = defaultdict(list)
    incidents = []; found_prior = set(); found_genes = set(); counts = Counter()
    progress('FULL_CURRENT_SOURCE_NAMES_CASE_VARIANTS_ALIASES_447_CLAIMS_AND_CURRENT_EDGES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader, h):
        for kind, key, row in walk_graph(reader):
            counts[kind] += 1
            if kind == 'node':
                if key in witnesses:
                    require(digest(row) == witnesses[key]['node_sha256'], 'gene witness changed'); found_genes.add(key)
                if key in prior_nodes:
                    require(node_projection(row) == prior_nodes[key], 'reviewed existing node changed'); found_prior.add(key)
                if key in selected:
                    require(all(digest(row) == r['claim_sha256'] for r in reviews.values() if r['claim_id'] == key), 'historic claim changed')
                    claims[key] = row
                if key in generated:
                    raise ValueError('generated identity already exists; requires separate reuse review')
                if key.startswith('CLM:'):
                    counts['claims'] += 1; md = row['metadata']; inner = md.get('metadata') or {}
                    for side in ('subject', 'object'):
                        if md.get(side + '_id') in prior_nodes:
                            paper = md.get('source_paper') or {}
                            incidents.append(dict(node_id=md[side + '_id'], claim_id=key, claim_sha256=digest(row), side=side, name=md.get(side + '_name'),
                                outer_type=md.get(side + '_type'), inner_type=inner.get(side + '_type'),
                                declared_roles=sorted(a.value for a in declared_type_atoms(md.get(side + '_type') or inner.get(side + '_type'))),
                                pmid=str(paper.get('pmid') or ''), doi=paper.get('doi'), raw_text_sha256=digest(md.get('raw_text')),
                                scope_is_not_a_new_scientific_validation=True))
                else:
                    labels = {name_key(v).casefold() for v in [row.get('preferred_name'), *(row.get('aliases') or [])]}
                    for label in names & labels:
                        matches[label].add(key); nodes[key] = row
            elif kind == 'edge' and repair.edge_owner(row) in selected:
                owned[repair.edge_owner(row)].append((int(key), row))
            if counts[kind] % 1000000 == 0: progress(kind, counts=dict(counts))
        require(h.hexdigest() == c['current_graph']['sha256'], 'actual current full graph SHA differs')
    require(set(claims) == selected and found_genes == set(witnesses) and found_prior == set(prior_nodes), 'complete current source scope missing')
    require(incidents == prior_incidents, 'existing node incident source scope changed across R46')
    require((counts['node'], counts['claims'], counts['edge']) == tuple(c['counts'][k] for k in ('nodes', 'claims', 'edges')), 'source counts differ')
    by_target = defaultdict(list)
    for r in incidents: by_target[r['node_id']].append(r)
    terms = IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    events = []; edge_events = []; fixed = set(); new = {}; reused = {}; declines = {}
    for k, r in reviews.items():
        if r['current_gate']: declines[(r['claim_id'], r['side'])] = r['current_gate']
        elif r['claim_id'] in protected: declines[(r['claim_id'], r['side'])] = 'separate_scientific_or_structure_scope_hold'
    for cid, row in sorted(claims.items()):
        changes = []; local_new = {}; local_reused = {}
        for side in ('subject', 'object'):
            k = repair.review_key(cid, side)
            if k not in proposals: continue
            r = proposals[k]; name = name_key(r['name'])
            reason = repair.endpoint_gate(row, side, witnesses[r['current_node_id']], r)
            require(reason is None, 'frozen endpoint review differs: ' + str(reason))
            target, reason, is_new = candidate_target(name, matches, nodes, by_target)
            if reason:
                declines[cid, side] = reason; continue
            if is_new: local_new[target['id']] = dict(id=target['id'], name=name, node_sha256=digest(target))
            else: local_reused[target['id']] = digest(target)
            changes.append(dict(side=side, name=name, old_id=r['current_node_id'], target_id=target['id']))
        if not changes: continue
        try:
            event, out = repair.reviewed_claim(row, changes, witnesses, reviews)
            edges = repair.reviewed_edges(cid, row, out, owned[cid])
        except ValueError as error:
            for ch in changes: declines[cid, ch['side']] = 'current_reference_closure:' + str(error)
            continue
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])), new_relation_id=relation_id(terms.relation_key(out['metadata'])))
        events.append(event); edge_events.extend(edges); new.update(local_new); reused.update(local_reused)
        fixed.update((cid, ch['side']) for ch in changes)
    require(events and len(fixed) <= 443 and fixed <= set(queue), 'unreviewed endpoint expansion')
    remaining = [dict(r, latest_review_reason=declines[k]) if k in declines else r for k, r in sorted(queue.items()) if k not in fixed]
    require(len(remaining) + len(fixed) == len(queue), 'queue partition differs')
    require(sha256(Path(census['database']['path'])) == census['database']['sha256'], 'current census full SHA differs')
    db = sqlite3.connect(Path(census['database']['path']).as_uri() + '?mode=ro', uri=True)
    try:
        for e in events:
            require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?', (e['claim_id'],)).fetchone() == (e['claim_sha256'], e['old_relation_id']), 'current census identity differs')
        shared = expected_shared(db, {e['claim_id']: e for e in events}, rows(c['current_shared_relations']['path']))
    finally: db.close()
    write_rows(OUTPUT / 'REMAINING_GENE_ENDPOINTS.jsonl', remaining)
    write_rows(OUTPUT / 'CURRENT_COMPLETE_NAME_COLLISIONS.jsonl', [node_projection(n) for n in nodes.values()])
    write_rows(OUTPUT / 'HISTORIC_DECLINES.jsonl', [dict(claim_id=cid, side=side, reason=reason) for (cid, side), reason in sorted(declines.items())])
    plan = dict(version=repair.VERSION, status='REVIEWED_NOT_APPLIED', at=j.utc_now(), graph=c['current_graph'], source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'], source_review=inspection_fp, source_full_sha_verified=True, source_census_full_sha_verified=True,
        historical_projection_rebound_by_fresh_current_claim_gene_node_incidence_and_full_source_hashes=True,
        code=code, events=events, edge_events=sorted(edge_events, key=lambda e: e['ordinal']), endpoint_reviews=reviews,
        new_literals=sorted(new.values(), key=lambda r: r['id']), existing_targets={**reused, **{g: w['node_sha256'] for g, w in witnesses.items()}},
        gene_witnesses=witnesses, expected_shared_claim_ids=shared, relation_grouping=POLICY,
        changed_claims=len(events), changed_endpoints=len(fixed), changed_edges=len(edge_events), added_literal_nodes=len(new), reused_existing_nodes=len(reused),
        old_watchlist_endpoints=len(queue), expanded_review_endpoints=len(queue), newly_registered_lexical_candidates=0, held_endpoints=len(remaining),
        original_watchlist_449_remaining_before=447, original_watchlist_449_repaired=len(fixed), original_watchlist_449_remaining_after=447-len(fixed),
        preflight_candidates=len(proposals), preflight_declines=dict(Counter(declines.values())),
        remaining_queue=j.fingerprint(OUTPUT / 'REMAINING_GENE_ENDPOINTS.jsonl'), protected_scientific_claims=sorted(protected),
        existing_name_collision_evidence=j.fingerprint(OUTPUT / 'CURRENT_COMPLETE_NAME_COLLISIONS.jsonl'), decline_evidence=j.fingerprint(OUTPUT / 'HISTORIC_DECLINES.jsonl'),
        record_preimages_saved=False, graph_backups=0,
        boundaries=['Finite historic full-name identity only; no extension to the entire lexical review queue.',
            'No new scientific claims, edges, statistics, aliases or metadata fields; existing node records unchanged.',
            'Generic complete mention literals do not assert an imaging or single-molecule type.',
            'All scientific fields and audit objects preserved; existing same names/case variants blocked unless separately reviewed safe reuse.',
            'Current R46 edge ordinals freshly scanned; old R47/R44 ordinals never reused.'])
    require(j.read_json(j.OUTPUT / 'CAMPAIGN.json') == c and [j.fingerprint(fp['path']) for fp in code] == code, 'source/code advanced')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    j.atomic_json(OUTPUT / 'PLAN.json', plan)
    progress('PLAN_COMPLETE_NOT_APPLIED', claims=len(events), endpoints=len(fixed), new_nodes=len(new), reused_nodes=len(reused),
        historic_remaining=447-len(fixed), expanded_remaining=len(remaining), declines=plan['preflight_declines'])


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        j.atomic_json(OUTPUT / 'PLAN_STATE.json', dict(status='FAILED', at=j.utc_now(), error=repr(error), graph_modified=False))
        raise
