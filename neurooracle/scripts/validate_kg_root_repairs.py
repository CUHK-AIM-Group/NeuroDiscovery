"""Independent original-stream validation of the frozen R69 plan, without a KG write.

No working selection/SQL prediction tables are read. Metadata expectations are
rebuilt from the frozen literal contract, not metadata_repair(). Group statistics
use compact in-memory sets rather than the producer's SQL grouping tables.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
import inspect_kg_root_repairs as inspect
from fetch_kg_fragmentation_examples import documents
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src import kg_root_repairs as repair
from neurooracle.src.correlation_grouping import IndexTerms
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.relation_evidence import relation_id
from neurooracle.src.shared_relation_catalog import check_file, current_shared_relations, find_shared_relations
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = inspect.OUTPUT
CLOZAPINE = {'CLM:case3_manual_12839431_001', 'CLM:case3_manual_22217436_001', 'CLM:case3_manual_25957394_001'}


def independently_normalized_metadata(record, contract):
    result = deepcopy(record); evidence = result['metadata'].get('evidence')
    if not isinstance(evidence, dict): return result
    current = evidence.get('study_type')
    if not isinstance(current, str): return result
    if current in contract['workflow_exact_values']:
        evidence.pop('study_type')
    elif current in contract['exact_design_aliases']:
        evidence['study_type'] = contract['exact_design_aliases'][current]
    elif current in contract['exact_method_labels']:
        method = evidence.get('methodology', '')
        if isinstance(method, str) or method is None:
            evidence.pop('study_type')
            if method is None or method == '': evidence['methodology'] = current
            elif current not in method.split('; '): evidence['methodology'] = method + '; ' + current
    return result


def join_distinct(existing, value):
    if value is None: return existing
    if existing is None: return value
    if isinstance(existing, set): existing.add(value); return existing
    return existing if existing == value else {existing, value}


def distinct_count(value):
    return len(value) if isinstance(value, set) else int(value is not None)


def endpoint_decision_index(records):
    result = defaultdict(list); seen = set()
    for record in records:
        key = (record['claim_id'], record['side'])
        require(record['side'] in ('subject', 'object') and key not in seen, 'multiple or invalid final endpoint decisions')
        seen.add(key); result[record['claim_id']].append(record)
    return result


def group_summary(groups):
    out = dict(fine_relations=len(groups), shared_groups=0, shared_claims=0,
        single_source_fine_relations=0, multi_source_fine_relations=0,
        multi_verified_source_groups=0, default_multi_verified_source_groups=0)
    for n, keys, verified, countable in groups.values():
        out['shared_groups'] += n > 1
        out['shared_claims'] += n if n > 1 else 0
        out['single_source_fine_relations'] += distinct_count(keys) == 1
        out['multi_source_fine_relations'] += distinct_count(keys) > 1
        out['multi_verified_source_groups'] += distinct_count(verified) > 1
        out['default_multi_verified_source_groups'] += distinct_count(countable) > 1
    return out


def main():
    require(not (OUTPUT / 'INDEPENDENT_VALIDATION.json').exists(), 'validation already frozen')
    plan_fp = j.fingerprint(OUTPUT / 'PLAN.json'); plan = j.read_json(plan_fp['path'])
    b = j.read_json(OUTPUT / 'BASELINE.json'); c = b['campaign']
    require(j.read_json(j.OUTPUT / 'CAMPAIGN.json') == c and plan['graph'] == c['current_graph'], 'source advanced')
    for fp in [*plan['artifacts'].values(), *plan['code'], plan['draft']]: check_file(fp, full_hash=True)
    validator_code = j.fingerprint(Path(__file__))
    rules = j.read_json(OUTPUT / 'METADATA_RULES.json')
    events = {e['claim_id']: e for e in rows(OUTPUT / 'CLAIM_EVENTS.jsonl')}
    structural = {e['claim_id']: e for e in rows(OUTPUT / 'STRUCTURAL_EVENTS.jsonl')}
    edge_edits = {e['ordinal']: e for e in rows(OUTPUT / 'EDGE_EVENTS.jsonl')}
    deleted = {e['claim_id']: e for e in rows(OUTPUT / 'CLAIM_DELETIONS.jsonl')}
    edge_deleted = {e['ordinal']: e for e in rows(OUTPUT / 'EDGE_DELETIONS.jsonl')}
    new_nodes = {n['id']: n for n in rows(OUTPUT / 'NEW_LITERAL_NODES.jsonl')}
    endpoints = rows(OUTPUT / 'ENDPOINT_DECISIONS.jsonl')
    deletion_ids = set(deleted)
    terms_base = VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])); terms = IndexTerms(terms_base)
    papers = VerifiedPaperIdentities(j.read_json(c['current_paper_identities']['path']))
    endpoint_by_claim = endpoint_decision_index(endpoints)
    source_decisions = j.read_json(OUTPUT / 'SOURCE_DECISIONS.json')
    source_docs = {}
    for value in source_decisions.values():
        check_file(value['source'], full_hash=True)
        source_docs.update(documents(Path(value['source']['path']).read_text(encoding='utf8')))
    source_claims = {}
    # The immutable CURRENT census is an input, not either temporary producer DB.
    census = sqlite3.connect(Path(b['census']['database']['path']).as_uri() + '?mode=ro', uri=True)
    current_shared = dict(census.execute('SELECT cid,relation_id FROM claims WHERE shared=1'))
    census.close(); old_group_to_new = defaultdict(set)
    for node in new_nodes.values():
        require(node == repair.unclassified_literal(node['preferred_name']), 'new literal is not the declared unclassified full text')
    duplicate_groups = [g for g in rows(OUTPUT / 'EXACT_EVIDENCE_DECISIONS.jsonl') if g['status'] != 'hold']
    duplicates = {cid: g['keeper'] for g in duplicate_groups for cid in g['members']}
    duplicate_nodes, duplicate_after = {}, {}
    duplicate_edge_expected, duplicate_edge_actual = defaultdict(set), defaultdict(Counter)
    counts, changed, polarity, before_study, after_study = (Counter() for _ in range(5))
    ids, entity_targets = set(new_nodes), set()
    seen_events, seen_struct, seen_edges, seen_deleted, seen_edge_deleted = set(), set(), set(), set(), set()
    node_union, edge_union = set(), set(); groups = {}; original_bad_endpoints = 0
    verified_targets = {}
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], b['census']['database']])
    inspect.progress('INDEPENDENT_FULL_SOURCE_PLAN_CHECK')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader, source_hash):
        for kind, key, original in walk_graph(reader):
            counts[kind] += 1; out = original
            if kind == 'node':
                require(key not in new_nodes, 'new ID collides with current graph')
                if key in terms_base.node_ids: verified_targets[key] = digest(original)
                if key.startswith('CLM:'):
                    counts['claims'] += 1; md = original['metadata']
                    if key in source_decisions: source_claims[key] = original
                    original_bad_endpoints += sum(md[s + '_id'].startswith('CLM:') for s in ('subject', 'object'))
                    before_study[compact((md.get('evidence') or {}).get('study_type'))] += 1
                    if key in duplicates: duplicate_nodes[key] = original
                    if key in deleted:
                        require(digest(original) == deleted[key]['record_sha256'], 'deletion hash differs')
                        seen_deleted.add(key); continue
                    expected = original
                    if key in structural:
                        proposal = structural[key]
                        for field in proposal['field_changes']:
                            p = field['path']
                            allowed = p in [['metadata', s + '_id'] for s in ('subject', 'object')] or p in [
                                ['metadata', 'metadata', s + '_id'] for s in ('subject', 'object')]
                            allowed |= key in CLOZAPINE and p in [['metadata', 'object_type'],
                                ['metadata', 'metadata', 'object_type'], ['metadata', 'evidence', 'study_type']]
                            require(allowed, 'unreviewed structural/scientific field change')
                        expected = repair.apply_event(original, proposal); seen_struct.add(key)
                        for e in endpoint_by_claim[key]:
                            side = e['side']
                            require(md[side + '_name'] == e['complete_name'] and md[side + '_id'] == e['old_id'], 'endpoint evidence differs')
                            require(expected['metadata'][side + '_id'] == e['target_id'], 'approved endpoint not applied')
                            if e['route'] == 'live_verified_whole_endpoint':
                                term = terms_base.term_for(md, side)
                                require(term and term.get('canonicalize', True) and term['target_id'] == e['target_id']
                                        and digest(term) == e['proof']['term_sha256'], 'whole endpoint authority does not approve')
                    expected = independently_normalized_metadata(expected, rules)
                    if key in events:
                        out = repair.apply_event(original, events[key]); seen_events.add(key); changed['claims'] += 1
                    require(digest(expected) == digest(out), 'missing/extra/incorrect composed metadata event')
                    if key in duplicates: duplicate_after[key] = out
                    md = out['metadata']; polarity[md['negated']] += 1
                    require(type(md['negated']) is bool and md['negated'] is original['metadata']['negated'], 'polarity lost')
                    for side in ('subject', 'object'):
                        require(not md[side + '_id'].startswith('CLM:'), 'remaining claim endpoint')
                        entity_targets.add(md[side + '_id'])
                    after_study[compact((md.get('evidence') or {}).get('study_type'))] += 1
                    identity = papers.resolve(md); pk = identity['paper_key']
                    verified = pk if identity['status'] == 'verified' else None
                    countable = verified if verified and papers.publication_review(pk)['status'] != 'retracted' else None
                    rid = relation_id(terms.relation_key(md))
                    if key in current_shared: old_group_to_new[current_shared[key]].add(rid)
                    if rid not in groups: groups[rid] = [1, pk, verified, countable]
                    else:
                        group = groups[rid]; group[0] += 1
                        for i, value in enumerate((pk, verified, countable), 1): group[i] = join_distinct(group[i], value)
                ids.add(key); node_union.update(out.get('metadata') or {})
            elif kind == 'edge':
                ordinal = int(key); owner = edge_owner(original)
                # Verify complete evidence-edge union survives each approved
                # duplicate group (including any uniquely retained science edge).
                if owner in duplicates:
                    keeper = duplicates[owner]; md0 = duplicate_nodes[owner]['metadata']; md1 = duplicate_after[keeper]['metadata']
                    target = deepcopy(original)
                    if target['relation_type'] == 'about':
                        sides = [s for s in ('subject', 'object') if target['target_id'] == md0[s + '_id']]
                        require(len(sides) == 1, 'ambiguous duplicate about edge')
                        target['target_id'] = md1[sides[0] + '_id']
                    else:
                        require((target['source_id'], target['target_id'], target['relation_type']) ==
                                (md0['subject_id'], md0['object_id'], md0['predicate']), 'duplicate science edge is inconsistent')
                        target['source_id'], target['target_id'] = md1['subject_id'], md1['object_id']
                    duplicate_edge_expected[keeper].add(digest(repair.normalized_owner(target, owner)))
                if ordinal in edge_deleted:
                    require(digest(original) == edge_deleted[ordinal]['record_sha256'], 'edge deletion hash differs')
                    seen_edge_deleted.add(ordinal); continue
                if ordinal in edge_edits:
                    out = repair.apply_event(original, edge_edits[ordinal]); seen_edges.add(ordinal); changed['edges'] += 1
                if owner in duplicates:
                    keeper = duplicates[owner]
                    require(edge_owner(out) == keeper, 'duplicate ownership not transferred')
                    duplicate_edge_actual[keeper][digest(repair.normalized_owner(out, keeper))] += 1
                require(out['source_id'] in ids and out['target_id'] in ids, 'dangling output edge')
                edge_union.update(out.get('metadata') or {})
            require(not list(exact_references(out, deletion_ids)), 'deleted record remains referenced')
            if counts[kind] % 250000 == 0: inspect.progress('VERIFY_' + kind, counts=dict(counts), changed=dict(changed))
        require(source_hash.hexdigest() == c['current_graph']['sha256'], 'independent source SHA differs')
    require(seen_events == set(events) and seen_struct == set(structural) and seen_edges == set(edge_edits), 'missing patch closure')
    require(seen_deleted == deletion_ids and seen_edge_deleted == set(edge_deleted), 'missing removal closure')
    require(not entity_targets - ids and original_bad_endpoints == 59, 'endpoint completeness changed')
    for term in terms_base.entries.values():
        for field, seal in (('target_id', 'target_sha256'), ('atom_id', 'atom_sha256'), ('parent_id', 'parent_sha256')):
            require(verified_targets.get(term[field]) == term[seal], 'registry witness changed')
    for g in duplicate_groups:
        normalized = []
        for cid in g['members']:
            row = deepcopy(duplicate_nodes[cid]); row.pop('id'); row['metadata'].pop('id', None); row['metadata'].pop('scope_reaudit', None)
            normalized.append(digest(row))
        require(len(set(normalized)) == 1, 'duplicate observation not identical')
        require(len({digest(repair.audit_decision(duplicate_nodes[cid])) for cid in g['members']}) == 1, 'audit decision conflict')
        keeper = g['keeper']
        require(keeper in ids and keeper not in deletion_ids, 'keeper missing')
        require(duplicate_edge_actual[keeper] == Counter({k: 1 for k in duplicate_edge_expected[keeper]}), 'duplicate edge union not preserved exactly once')
    require(set(source_claims) == set(source_decisions), 'source-reviewed claim closure differs')
    for cid, proof in source_decisions.items():
        row = source_claims[cid]; md = row['metadata']; paper = md['source_paper']; doc = source_docs[proof['pmid']]
        require(digest(row) == proof['claim_sha256'] and digest(doc['abstract']) == proof['source_abstract_sha256'], 'primary proof hash differs')
        require(str(paper['pmid']) == proof['pmid'] and paper['doi'].casefold() in {d.casefold() for d in doc['dois']}, 'own DOI/PMID binding differs')
        require(paper['title'].casefold() == doc['title'].casefold(), 'own title differs')
        if cid in CLOZAPINE:
            require('Review' in doc['publication_types'] and 'treatment-resistant schizophrenia' in doc['abstract'], 'TRS review evidence missing')
            proposed = repair.apply_event(row, events[cid])
            require(proposed['metadata']['object_id'] == 'CUI:C3544321' and proposed['metadata']['evidence']['study_type'] == 'review', 'source reviewed repair not present')
        else:
            require(cid in deleted and md['raw_text'] in doc['abstract'] and md['negated'] is False, 'non-result removal not source-supported')
            prefix = 'We hypothesize' if deleted[cid]['reason'] == 'source_hypothesis_not_observed_cause' else 'This study aims'
            require(md['raw_text'].startswith(prefix), 'removal is not a hypothesis/aim')
    for node in new_nodes.values(): node_union.update(node.get('metadata') or {})
    actual = dict(current_counts=c['counts'], proposed_counts=dict(nodes=len(ids), claims=counts['claims'] - len(deleted),
        edges=counts['edge'] - len(edge_deleted)), changed=dict(changed),
        study_type=dict(before_nonempty=sum(v for k, v in before_study.items() if k not in ('null', '""')),
            after_nonempty=sum(v for k, v in after_study.items() if k not in ('null', '""')),
            before_values=len(before_study), after_values=len(after_study), nonempty_is_not_source_verified_design_coverage=True),
        proposed_negated_claims=polarity[True], proposed_positive_claims=polarity[False],
        metadata_field_union=dict(nodes=len(node_union), edges=len(edge_union)), proposed_sharing=group_summary(groups),
        structural_claim_endpoints_remaining=0, existing_fine_relations_split=sum(len(v)>1 for v in old_group_to_new.values()),
        graph_modified=False, results_are_predictions=True)
    predicted = plan['summary']
    for key, value in actual.items(): require(value == predicted[key], 'independent prediction differs: ' + key)
    require(tuple(counts[k] for k in ('node', 'claims', 'edge')) == tuple(c['counts'][k] for k in ('nodes', 'claims', 'edges')), 'input counts differ')
    inspect.progress('FINAL_PROTECTED_FILE_HASH_BOUNDARY')
    for fp in [c['current_detail_store'], *c['formal_sources'].values(), b['census']['database']]: check_file(fp, full_hash=True)
    current_queries = dict(shared_groups=len(list(current_shared_relations(j.OUTPUT / 'CAMPAIGN.json'))),
        default_verified_groups=len(list(find_shared_relations(j.OUTPUT / 'CAMPAIGN.json', minimum_papers=2))),
        including_retracted=len(list(find_shared_relations(j.OUTPUT / 'CAMPAIGN.json', minimum_papers=2, include_retracted=True))))
    require(current_queries == dict(shared_groups=2588, default_verified_groups=519, including_retracted=521), 'current read query changed')
    require(j.fingerprint(OUTPUT / 'PLAN.json') == plan_fp and j.fingerprint(Path(__file__)) == validator_code, 'plan/validator changed')
    require(j.read_json(j.OUTPUT / 'CAMPAIGN.json') == c, 'campaign changed')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], b['census']['database']])
    result = dict(status='PASSED_INDEPENDENT_DRY_RUN_NOT_APPLIED', at=j.utc_now(), plan=plan_fp,
        validator=validator_code, source_full_sha_verified=True, full_protected_hashes_verified=True,
        claims_independently_checked=counts['claims'], evidence_duplicate_groups_verified=len(duplicate_groups),
        complete_duplicate_edge_union_preserved=True, independent_source_relation_counts=True,
        all_frozen_events_applied_exactly_once_in_memory=True, metadata_rules_independently_reconstructed=True,
        current_queries_unchanged=current_queries, predicted=actual, graph_writes=0, model_calls=0,
        scientific_scope_unresolved_holds_retained=True, actual_candidate_acceptance_not_yet_run=True)
    j.atomic_json(OUTPUT / 'INDEPENDENT_VALIDATION.json', result)
    inspect.progress('INDEPENDENT_VALIDATION_PASSED_KG_UNCHANGED', predicted=actual)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        j.atomic_json(OUTPUT / 'VALIDATION_FAILURE.json', dict(at=j.utc_now(), error=repr(error), graph_modified=False)); raise
