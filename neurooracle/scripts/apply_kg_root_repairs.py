"""R70: one candidate, actual-graph verification, evidence dossiers, atomic switch.

No model calls, scheduling, full_v2 writes, old KG or record-preimage retention.
Build and validate are separate resumable phases. Adoption requires all gates.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import CurrentCensus, small_check
from apply_kg_explicit_pmid_provenance import authorities, publication_row
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from fetch_kg_fragmentation_examples import documents
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows, guard_native
from reclaim_kg_backup_storage import native_info, sha256
from validate_kg_root_repairs import group_summary, join_distinct, CLOZAPINE
from neurooracle.src.correlation_grouping import IndexTerms, POLICY
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, bibliography, identifiers, title_key, census_identifier_projection
from neurooracle.src.kg_root_application import RootTransform, reverse_event, source_edge_ordinal
from neurooracle.src.metadata_field_audit import Coverage, node_class
from neurooracle.src.relation_evidence import relation_id, evidence_member
from neurooracle.src.relation_evidence_dossier import observation, summarize, REVIEW_TYPES
from neurooracle.src.shared_relation_catalog import check_file, find_shared_relations
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

R69 = j.OUTPUT / 'round69_root_cause_repairs'
OUTPUT = j.OUTPUT / 'round70_root_application'
SOURCE = j.OUTPUT / 'round23_source_deletion_candidate/knowledge_graph.candidate.json'
TEMP = SOURCE.with_name(SOURCE.name + '.root-repairs.tmp')
CATALOG = OUTPUT / 'CURRENT_SHARED_RELATIONS.jsonl'
DATABASE = OUTPUT / 'CURRENT_PAPER_CENSUS.sqlite'
QUEUE_FILES = {
    'current_issues': 'CURRENT_REMAINING_ISSUES.jsonl',
    'current_structure_holds': 'CURRENT_STRUCTURE_HOLDS.jsonl',
    'current_gene_holds': 'CURRENT_GENE_ENDPOINT_HOLDS.jsonl',
    'current_nominal_semantic_followups': 'CURRENT_SEMANTIC_FOLLOWUPS.jsonl',
}
NEW_CODE = [j.REPO / p for p in (
    'neurooracle/src/kg_root_application.py', 'neurooracle/src/relation_evidence_dossier.py',
    'neurooracle/scripts/apply_kg_root_repairs.py', 'neurooracle/scripts/query_kg_relation_evidence.py',
    'neurooracle/scripts/test_kg_root_application_regression.py',
    'neurooracle/tests/test_kg_root_application.py', 'neurooracle/tests/test_relation_evidence_dossier.py')]


def progress(phase, **values):
    state = dict(status='RUNNING', pid=os.getpid(), phase=phase, at=j.utc_now(), **values)
    j.atomic_json(OUTPUT / 'RUN_STATE.json', state)
    print(compact(state), flush=True)


def transform():
    return RootTransform(*(rows(R69 / n) for n in ('CLAIM_EVENTS.jsonl', 'EDGE_EVENTS.jsonl',
        'CLAIM_DELETIONS.jsonl', 'EDGE_DELETIONS.jsonl', 'NEW_LITERAL_NODES.jsonl')))


def fp_large(path):
    before = cheap(path)
    value = sha256(path)
    require(before == cheap(path), 'file changed while hashing')
    return dict(**before, sha256=value)


def codes(baseline):
    old = j.read_json(baseline['current_acceptance']['path'])
    plan = j.read_json(R69 / 'PLAN.json')
    paths = list(dict.fromkeys([Path(x['path']) for x in old['code'] + plan['code']] + NEW_CODE))
    return [j.fingerprint(p) for p in paths]


def test_receipt():
    path = OUTPUT / 'TEST_RESULTS.xml'
    cases = list(ET.parse(path).getroot().iter('testcase'))
    identities = {(c.get('classname'), c.get('name')) for c in cases}
    require(len(cases) == len(identities) >= 1734, 'insufficient or duplicated test selection')
    require(all(not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases), 'tests not passing')
    return dict(file=j.fingerprint(path), unique_passing_cases=len(cases), old_live_r59_cases_not_selected=18)


def load_frozen(baseline):
    accepted = j.read_json(R69 / 'BATCH_ACCEPTANCE.json')
    for key in ('plan', 'independent_validation', 'runtime_repair'):
        check_file(accepted[key], full_hash=True)
    plan = j.read_json(accepted['plan']['path'])
    independent = j.read_json(accepted['independent_validation']['path'])
    require(independent['status'] == 'PASSED_INDEPENDENT_DRY_RUN_NOT_APPLIED', 'R69 independent gate absent')
    require(independent['plan'] == accepted['plan'] and plan['graph'] == baseline['current_graph'], 'frozen source changed')
    require(plan['registry'] == baseline['current_entity_terms'] and plan['papers'] == baseline['current_paper_identities'], 'authority registry changed')
    for fp in [*plan['artifacts'].values(), *plan['code'], independent['validator'], plan['draft']]:
        check_file(fp, full_hash=True)
    require(plan['protected_formal_sources'] == baseline['formal_sources'], 'formal boundary changed')
    return plan


def seed_source_reviews(events):
    """Use cached primary publication records; do not infer roles from labels."""
    decisions = j.read_json(R69 / 'SOURCE_DECISIONS.json')
    docs, sources = {}, {}
    for row in decisions.values():
        path = row['source']['path']
        if path in sources:
            continue
        check_file(row['source'], full_hash=True)
        sources[path] = row['source']
        for pmid, doc in documents(Path(path).read_text(encoding='utf8')).items():
            if pmid in docs:
                require(digest(docs[pmid]['document']) == digest(doc), 'conflicting cached primary records')
            docs[pmid] = dict(document=doc, source=row['source'])
    return docs


def review_for(row, docs):
    md = row['metadata']; source = md.get('source_paper') or {}
    pmid = str(source.get('pmid') or '')
    entry = docs.get(pmid)
    if not entry:
        return None
    doc = entry['document']
    if source.get('title', '').casefold() != doc['title'].casefold():
        return None
    if not source.get('doi') or source['doi'].casefold() not in {d.casefold() for d in doc['dois']}:
        return None
    if not REVIEW_TYPES.intersection(doc['publication_types']):
        return None
    return dict(claim_sha256=digest(row), pmid=pmid, source_role='review_or_evidence_synthesis',
                primary_source=entry['source'], publication_types=doc['publication_types'],
                source_abstract_sha256=digest(doc['abstract']),
                own_empirical_result_not_inferred=True, cohort_independence_not_inferred=True)


def build():
    OUTPUT.mkdir(exist_ok=True)
    require(not any(p.exists() for p in (TEMP, CATALOG, DATABASE, OUTPUT / 'BUILD_STATE.json')), 'build exists; do not overwrite')
    baseline = j.read_json(j.OUTPUT / 'CAMPAIGN.json')
    require(baseline['status'] == 'COMPLETED' and baseline['active_process'] is None and not baseline['rollback_retention'], 'writer/retention boundary')
    require(Path(baseline['current_graph']['path']).resolve() == SOURCE.resolve(), 'unexpected graph target')
    require(shutil.disk_usage(SOURCE).free > baseline['current_graph']['bytes'] + 12 * 1024**3, 'insufficient candidate space')
    plan = load_frozen(baseline)
    tests = test_receipt()
    for field in (*QUEUE_FILES, 'current_scope_findings', 'current_acceptance', 'current_runtime_acceptance',
                  'current_coverage', 'current_shared_relations', 'current_paper_issues', 'current_paper_census'):
        small_check(baseline[field])
    code = codes(baseline)
    papers, _ = authorities(baseline)
    terms = IndexTerms(VerifiedEntityTerms(j.read_json(baseline['current_entity_terms']['path'])))
    old_census = j.read_json(baseline['current_paper_census']['path'])
    guards = [native_info(fp['path']) for fp in (baseline['current_graph'], baseline['current_detail_store'],
        old_census['database'], *baseline['formal_sources'].values())]
    night = j.fingerprint(baseline['night_window'])
    j.atomic_json(OUTPUT / 'BASELINE.json', baseline)
    campaign = dict(baseline, status='MANUAL_ACTIVE', phase='R70第三批：修复落地、跨论文证据归一验证、统一验收',
        updated_at=j.utc_now(), active_process=dict(kind='root_repair_application', pid=os.getpid(), state=str(OUTPUT / 'RUN_STATE.json')))
    j.atomic_json(j.OUTPUT / 'CAMPAIGN.json', campaign)
    patch = transform(); coverage = Coverage(); proof = SourceProof(terms, papers)
    census = CurrentCensus(DATABASE, terms, set())
    issues = []
    pub_path = Path(baseline['current_acceptance']['path']).parent / 'CURRENT_PUBLICATION_REVIEW.json'
    pub = j.read_json(pub_path); pub_rows = {r['claim_id']: r for r in pub['claims']}; current_pub = []
    require(pub['graph'] == baseline['current_graph'], 'stale publication review')
    def observed(records):
        for kind, key, row in records:
            if kind != 'metadata':
                coverage.add('node/' + node_class(key, row) if kind == 'node' else 'edge/all', row)
            if kind == 'node' and key.startswith('CLM:'):
                census.add(key, row)
                answer = papers.resolve(row['metadata'])
                if answer['status'] == 'conflict':
                    issues.append(dict(claim_id=key, claim_sha256=digest(row), reasons=answer['reasons'],
                        legacy_source_key=evidence_member(row['metadata'])['paper_key'], action='hold_no_claim_or_source_deletion'))
                if key in pub_rows:
                    original = reverse_event(row, patch.claims[key]) if key in patch.claims else row
                    current_pub.append(publication_row(pub_rows[key], original, row, papers))
            yield kind, key, row
    progress('BUILD_APPROVED_CANDIDATE')
    try:
        with TEMP.open('xb', buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader, h):
                result = stream_patch(proof.records(observed(patch.records(walk_graph(reader)))), handle, [], {}, CATALOG,
                    progress, relation_key_func=terms.relation_key, relation_version='kg.relation_evidence.v4',
                    papers=papers, extra_graph_metadata=dict(relation_grouping=POLICY))
                require(h.hexdigest() == baseline['current_graph']['sha256'], 'full source SHA differs')
            handle.flush(); os.fsync(handle.fileno())
        require(result['counts'] == plan['summary']['proposed_counts'] and not result['changed'], 'unexpected output counts')
        # Membership comes from the completed relation catalog, never from a
        # guessed subset of source records. Independent candidate scan checks it.
        census.db.executemany('INSERT INTO claims VALUES (?,?,?,?,?,?)', census.batch); census.batch.clear()
        for group in rows(CATALOG):
            contexts = {cid: v['signature'] for v in group['evidence_variants'] for cid in v['claim_ids']}
            for member in group['members']:
                cid = member['claim_id']
                legacy = census.db.execute('SELECT legacy_key FROM claims WHERE cid=?', (cid,)).fetchone()[0]
                census.db.execute('UPDATE claims SET shared=1 WHERE cid=?', (cid,))
                census.db.execute('INSERT INTO shared_members VALUES (?,?)', (cid, compact(dict(claim_id=cid,
                    paper_key=legacy, negated=member['negated'], context_signature=contexts[cid]))))
        census_result = census.finish()
    except BaseException:
        try: census.db.close()
        except Exception: pass
        raise
    audit = dict(claim_status=dict(proof.claim_status), verified_key_changes=proof.key_changes)
    proof.verify(result, audit)
    require(not census_result['outer_pmid_conflicts'], 'outer PMID conflicts')
    j.atomic_json(OUTPUT / 'IDENTITY_AUDIT.json', audit)
    write_rows(OUTPUT / 'CURRENT_METADATA_COVERAGE.jsonl', coverage.rows())
    write_rows(OUTPUT / 'CURRENT_PAPER_ISSUES.jsonl', issues)
    pub['claims'] = current_pub
    j.atomic_json(OUTPUT / 'PUBLICATION_BUILD.json', pub)
    j.atomic_json(OUTPUT / 'CENSUS_BUILD.json', dict(**census_result, database=fp_large(DATABASE)))
    guard_native(guards); require(codes(baseline) == code, 'code changed during build')
    state = dict(status='BUILT_NOT_ADOPTED', at=j.utc_now(), baseline=baseline, code=code, tests=tests,
        protected_native=guards, closed_night_window=night, temporary=cheap(TEMP), result=result,
        source_retained_canonical_digests={k: v.hexdigest() for k, v in patch.source_digests.items()},
        plan=j.fingerprint(R69 / 'PLAN.json'), prior_independent_validation=j.fingerprint(R69 / 'INDEPENDENT_VALIDATION.json'),
        catalog=j.fingerprint(CATALOG), census=j.fingerprint(OUTPUT / 'CENSUS_BUILD.json'),
        coverage=j.fingerprint(OUTPUT / 'CURRENT_METADATA_COVERAGE.jsonl'),
        paper_issues=j.fingerprint(OUTPUT / 'CURRENT_PAPER_ISSUES.jsonl'),
        publication=j.fingerprint(OUTPUT / 'PUBLICATION_BUILD.json'), identity_audit=j.fingerprint(OUTPUT / 'IDENTITY_AUDIT.json'),
        source_full_sha_verified=True, record_preimages_saved=False, graph_backups=0)
    j.atomic_json(OUTPUT / 'BUILD_STATE.json', state)
    progress('BUILT_NOT_ADOPTED', counts=result['counts'], sharing=result['catalog_stats'])


def validate():
    require(not (OUTPUT / 'VALIDATED.json').exists(), 'already validated')
    state = j.read_json(OUTPUT / 'BUILD_STATE.json'); baseline = state['baseline']
    require(codes(baseline) == state['code'] and cheap(TEMP) == state['temporary'], 'candidate/code changed')
    guard_native(state['protected_native'])
    for name in ('plan', 'prior_independent_validation', 'catalog', 'census', 'coverage', 'paper_issues', 'publication', 'identity_audit'):
        small_check(state[name])
    plan = load_frozen(baseline); patch = transform()
    terms = IndexTerms(VerifiedEntityTerms(j.read_json(baseline['current_entity_terms']['path'])))
    papers = VerifiedPaperIdentities(j.read_json(baseline['current_paper_identities']['path']))
    proof = SourceProof(terms, papers); coverage = Coverage()
    db = sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True)
    old_db = sqlite3.connect(Path(j.read_json(baseline['current_paper_census']['path'])['database']['path']).as_uri() + '?mode=ro', uri=True)
    old_shared = dict(old_db.execute('SELECT cid,relation_id FROM claims WHERE shared=1')); old_db.close()
    shared = {m['claim_id']: g['id'] for g in rows(CATALOG) for m in g['members']}
    source_docs = seed_source_reviews(patch.claims)
    inverse = {k: hashlib.sha256() for k in ('nodes', 'edges')}
    seen = {k: set() for k in ('claims', 'edges', 'new')}
    counts, polarity, study = Counter(), Counter(), Counter()
    groups, group_routes = {}, defaultdict(set)
    matched, paper_sigs = set(), set()
    dossiers, source_reviews = {}, {}
    current_pub = j.read_json(state['publication']['path']); pub_rows = {r['claim_id']: r for r in current_pub['claims']}
    seen_pub, issues = set(), []
    deleted_ids = set(patch.deleted_claims)
    def independent(records):
        for kind, key, row in records:
            original = row
            if kind == 'node':
                require(key not in deleted_ids, 'deleted claim still present')
                if key in patch.new_nodes:
                    require(row == patch.new_nodes[key], 'new literal differs'); seen['new'].add(key); original = None
                elif key in patch.claims:
                    original = reverse_event(row, patch.claims[key]); seen['claims'].add(key)
            elif kind == 'edge':
                ordinal = source_edge_ordinal(int(key), patch.deleted_edges)
                if ordinal in patch.edges:
                    original = reverse_event(row, patch.edges[ordinal]); seen['edges'].add(ordinal)
            if kind != 'metadata':
                counts[kind] += 1
                coverage.add('node/' + node_class(key, row) if kind == 'node' else 'edge/all', row)
                if original is not None:
                    inverse[kind + 's'].update(bytes.fromhex(digest(original)))
            require(not list(exact_references(row, deleted_ids)), 'deleted claim remains referenced')
            if kind == 'node' and key.startswith('CLM:'):
                md = row['metadata']; counts['claims'] += 1
                require(type(md['negated']) is bool, 'nonboolean negation')
                polarity[md['negated']] += 1
                require(all(not md[s + '_id'].startswith('CLM:') for s in ('subject', 'object')), 'claim still used as entity')
                if original is not None:
                    require(md['negated'] is original['metadata']['negated'], 'negation changed')
                    for field in ('raw_text', 'source_paper', 'conditions', 'population'):
                        require(md.get(field) == original['metadata'].get(field), 'scientific source/context changed')
                study[compact((md.get('evidence') or {}).get('study_type'))] += 1
                paper = bibliography(md['source_paper']); sig = digest(paper); member = evidence_member(md)
                rid = relation_id(terms.relation_key(md))
                actual = db.execute('SELECT node_sha,paper_sig,legacy_key,relation_id,shared FROM claims WHERE cid=?', (key,)).fetchone()
                require(actual == (digest(row), sig, member['paper_key'], rid, int(key in shared)), 'census claim differs')
                require(key not in matched, 'duplicate claim'); matched.add(key)
                if sig not in paper_sigs:
                    ids = identifiers(paper)
                    require(db.execute('SELECT payload,pmid,doi,pmcid,title,year FROM papers WHERE sig=?', (sig,)).fetchone() ==
                        (compact(paper), ids['pmid'], ids['doi'], ids['pmcid'], title_key(paper.get('title')), str(paper.get('year') or paper.get('publication_year') or '')), 'census bibliography differs')
                    paper_sigs.add(sig)
                answer = papers.resolve(md); pk = answer['paper_key']
                verified = pk if answer['status'] == 'verified' else None
                countable = verified if verified and papers.publication_review(pk)['status'] != 'retracted' else None
                if rid not in groups:
                    groups[rid] = [1, pk, verified, countable]
                else:
                    groups[rid][0] += 1
                    for index, value in enumerate((pk, verified, countable), 1):
                        groups[rid][index] = join_distinct(groups[rid][index], value)
                if key in old_shared:
                    group_routes[old_shared[key]].add(rid)
                if answer['status'] == 'conflict':
                    issues.append(dict(claim_id=key, claim_sha256=digest(row), reasons=answer['reasons'],
                        legacy_source_key=member['paper_key'], action='hold_no_claim_or_source_deletion'))
                if key in pub_rows:
                    require(publication_row(pub_rows[key], row, row, papers) == pub_rows[key], 'publication binding differs'); seen_pub.add(key)
                if key in shared:
                    require(db.execute('SELECT member_json FROM shared_members WHERE cid=?', (key,)).fetchone() == (compact(member),), 'shared member census differs')
                    review = review_for(row, source_docs)
                    if review:
                        source_reviews[key] = review
                    dossiers[key] = observation(row, papers, {key: review} if review else {})
            yield kind, key, row
    progress('INDEPENDENT_ACTUAL_CANDIDATE_SCAN')
    try:
        with hashed_reader(TEMP) as (reader, h):
            checks = verify_catalog_and_graph(proof.records(independent(walk_graph(reader))), state['result'], CATALOG,
                progress, relation_key_func=terms.relation_key, papers=papers)
            candidate_sha = h.hexdigest()
        require(len(matched) == db.execute('SELECT COUNT(*) FROM claims').fetchone()[0] == plan['summary']['proposed_counts']['claims'], 'claim census closure differs')
        require(len(paper_sigs) == db.execute('SELECT COUNT(*) FROM papers').fetchone()[0], 'stale bibliography')
        require(len(dossiers) == db.execute('SELECT COUNT(*) FROM shared_members').fetchone()[0] == plan['summary']['proposed_sharing']['shared_claims'], 'shared evidence closure differs')
        require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'census integrity failed')
        census_result = j.read_json(state['census']['path'])
        require(census_identifier_projection(db) == census_result['projection'], 'source collision projection differs')
    finally:
        db.close()
    require(seen['claims'] == set(patch.claims) and seen['edges'] == set(patch.edges) and seen['new'] == set(patch.new_nodes), 'incomplete applied event closure')
    require({k: v.hexdigest() for k, v in inverse.items()} == state['source_retained_canonical_digests'], 'unapproved retained node/edge change')
    require(group_summary(groups) == plan['summary']['proposed_sharing'], 'actual relation sharing differs from prediction')
    require(not any(len(v) > 1 for v in group_routes.values()), 'existing shared group split without approval')
    require(polarity[True] == plan['summary']['proposed_negated_claims'] == 6767, 'negative observations lost')
    require(sum(v for k, v in study.items() if k not in ('null', '""')) == plan['summary']['study_type']['after_nonempty'], 'study-type coverage differs')
    require(coverage.rows() == rows(state['coverage']['path']), 'independent metadata coverage differs')
    require(issues == rows(state['paper_issues']['path']) and seen_pub == set(pub_rows), 'source issue/publication closure differs')
    proof.verify(state['result'], j.read_json(state['identity_audit']['path']))
    output_dossiers = [summarize(g, [dossiers[m['claim_id']] for m in g['members']]) for g in rows(CATALOG)]
    write_rows(OUTPUT / 'CURRENT_EVIDENCE_DOSSIERS.jsonl', output_dossiers)
    require(rows(OUTPUT / 'CURRENT_EVIDENCE_DOSSIERS.jsonl') == output_dossiers, 'dossier serialization differs')
    j.atomic_json(OUTPUT / 'CURRENT_SOURCE_ROLE_REVIEWS.json', source_reviews)
    cloz_groups = [d for d in output_dossiers if CLOZAPINE <= {o['claim_id'] for o in d['observations']}]
    require(len(cloz_groups) == 1, 'three reviewed cross-paper observations did not converge')
    cloz = cloz_groups[0]
    require(all(dossiers[cid]['source_review']['source_role'] == 'review_or_evidence_synthesis' for cid in CLOZAPINE), 'three reviews misrepresented as independent trials')
    # All shared groups are machine-bound to the actual candidate. Source role
    # coverage below is explicitly bounded; missing cohort data stays unknown.
    science = dict(status='ACTUAL_SHARED_RELATION_EVIDENCE_VALIDATED', at=j.utc_now(),
        shared_groups=len(output_dossiers), current_observations=len(dossiers),
        source_review_bound_observations=len(source_reviews),
        all_review_groups=sum(d['independence_status'] == 'all_sources_are_reviews_not_primary_trials' for d in output_dossiers),
        independence_unestablished_groups=sum(d['independence_status'] == 'not_established' for d in output_dossiers),
        mixed_negation_groups=sum(d['mixed_negation_requires_context_review'] for d in output_dossiers),
        cross_source_identical_text_groups=sum(d['cross_source_identical_text_groups'] for d in output_dossiers),
        positive_control=dict(relation_id=cloz['relation_id'], claim_ids=sorted(CLOZAPINE),
            verified_articles=cloz['verified_article_count'], independent_primary_studies=cloz['independent_primary_study_count']),
        verified_source_count_is_not_independence=True, consensus_inferred=False,
        all_current_claims_polarity_checked=True, graph_wide_scientific_semantics_fully_reviewed=False)
    j.atomic_json(OUTPUT / 'SCIENTIFIC_QUERY_ACCEPTANCE.json', science)
    progress('FULL_PROTECTED_BOUNDARY_HASHES')
    for fp in [baseline['current_detail_store'], *baseline['formal_sources'].values()]:
        check_file(fp, full_hash=True)
    check_file(census_result['database'], full_hash=True)
    small_check(state['closed_night_window']); guard_native(state['protected_native'])
    require(codes(baseline) == state['code'], 'code changed during validation')
    # Remove inherited identity-only claims: R69 explicitly changes metadata
    # and removes 69 observations, so those old verifier labels do not apply.
    checks.pop('records_match_approved_identity_only_transform', None)
    checks.pop('all_nonidentity_claim_fields_preserved', None)
    checks.update(verified_identity_proofs_complete=True, paper_identity_witnesses_validated=True,
        publication_status_witnesses_validated=True, shared_observation_dossiers_complete=True,
        source_and_candidate_full_sha_verified=True, approved_field_changes_only=True,
        all_retained_source_record_canonical_digests_reproduced=True, all_current_census_rows_independently_verified=True,
        all_negative_observations_preserved=True, existing_shared_relations_split=0,
        source_conditions_and_statistical_values_not_rewritten=True, no_claim_entity_endpoints=True,
        formal_sources_full_sha_verified=True, metadata_coverage_full_scan_verified=True,
        distinct_articles_not_independent_studies=True)
    j.atomic_json(OUTPUT / 'VALIDATED.json', dict(status='VALIDATED_NOT_ADOPTED', at=j.utc_now(),
        graph=dict(**cheap(TEMP), sha256=candidate_sha), temporary_native=native_info(TEMP),
        database_native=native_info(DATABASE), build_state=j.fingerprint(OUTPUT / 'BUILD_STATE.json'), checks=checks,
        actual_sharing=group_summary(groups), actual_negated_claims=polarity[True],
        evidence_dossiers=j.fingerprint(OUTPUT / 'CURRENT_EVIDENCE_DOSSIERS.jsonl'),
        scientific_queries=j.fingerprint(OUTPUT / 'SCIENTIFIC_QUERY_ACCEPTANCE.json'),
        source_reviews=j.fingerprint(OUTPUT / 'CURRENT_SOURCE_ROLE_REVIEWS.json')))
    progress('VALIDATED_NOT_ADOPTED', counts=state['result']['counts'], scientific_queries=science)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build', 'validate'))
    phase = parser.parse_args().phase
    try:
        globals()[phase]()
    except BaseException as error:
        OUTPUT.mkdir(exist_ok=True)
        j.atomic_json(OUTPUT / 'FAILURE.json', dict(at=j.utc_now(), phase=phase, error=repr(error), current_graph_replaced=False))
        raise


if __name__ == '__main__':
    main()
