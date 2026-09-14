"""Current-only, read-only census of relation reuse and unsafe coarser groupings.

The SQLite working index lives in memory. Complete claim hashes, source
bibliographies and fine relation IDs must match the accepted current census.
Name-only and ID-only groupings are diagnostics, never merge instructions.
"""
from collections import Counter
from pathlib import Path
import json
import os
import sqlite3
import sys
import time
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from reclaim_kg_backup_storage import sha256
from neurooracle.src.correlation_grouping import IndexTerms, enabled
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, bibliography
from neurooracle.src.metadata_field_audit import node_class, nonempty
from neurooracle.src.relation_evidence import evidence_context, name_key, relation_id
from neurooracle.src.shared_relation_catalog import current_shared_relations, find_shared_relations
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = j.OUTPUT / 'round61_current_relation_reuse_review'
FILES = [Path(__file__), j.REPO / 'neurooracle/tests/test_kg_current_relation_reuse.py',
    *[j.REPO / ('neurooracle/src/' + name + '.py') for name in (
        'correlation_grouping', 'kg_identity_pilot', 'kg_paper_identity', 'metadata_field_audit',
        'relation_evidence', 'shared_relation_catalog', 'verified_entity_terms', 'kg_bulk_cleanup',
        'case_study_membership_contract')]]
GROUPINGS = ('rid', 'surface', 'folded', 'physical')


def diagnostic_triple(left, predicate, right):
    # This only repeats the already accepted correlation symmetry. No other
    # predicate is reversed, and no negation or study evidence is erased.
    if predicate == 'correlates_with' and left > right:
        left, right = right, left
    return (left, predicate, right)


def diagnostic_keys(md):
    sn, tn = name_key(md['subject_name']), name_key(md['object_name'])
    require(bool(sn and tn and md['subject_id'] and md['object_id'] and md['predicate']), 'incomplete relation')
    return dict(surface=digest(diagnostic_triple(sn, md['predicate'], tn)),
        folded=digest(diagnostic_triple(sn.casefold(), md['predicate'], tn.casefold())),
        physical=digest(diagnostic_triple(md['subject_id'], md['predicate'], md['object_id'])))


def claim_projection(row, terms, papers):
    md = row['metadata']
    require(md['id'] == row['id'], 'inner claim ID differs')
    paper = papers.resolve(md)
    countable = paper['status'] == 'verified' and papers.publication_review(paper['paper_key'])['status'] != 'retracted'
    keys = diagnostic_keys(md)
    return (row['id'], digest(row), digest(bibliography(md.get('source_paper'))),
        relation_id(terms.relation_key(md)), md['subject_id'], name_key(md['subject_name']), md['predicate'],
        md['object_id'], name_key(md['object_name']), keys['surface'], keys['folded'], keys['physical'],
        paper['paper_key'], paper['status'], int(countable), compact(md.get('negated')), digest(evidence_context(md)))


def initialize(db):
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('''CREATE TABLE claims(cid TEXT PRIMARY KEY,node_sha TEXT,paper_sig TEXT,rid TEXT,
        sid TEXT,sn TEXT,p TEXT,tid TEXT,tn TEXT,surface TEXT,folded TEXT,physical TEXT,
        pk TEXT,status TEXT,countable INT,neg TEXT,ctx TEXT)''')


def add_rows(db, values):
    db.executemany('INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)


def all_rows(db, sql, params=()):
    return [dict(r) for r in db.execute(sql, params)]


def group_statistics(db, field):
    require(field in GROUPINGS, 'unsupported grouping')
    table = 'groups_' + field
    db.execute(f'''CREATE TEMP TABLE {table} AS SELECT {field} k,COUNT(*) n,
        COUNT(DISTINCT pk) source_keys,
        COUNT(DISTINCT CASE WHEN status='verified' THEN pk END) verified_papers,
        COUNT(DISTINCT CASE WHEN countable=1 THEN pk END) counted_papers,
        COUNT(DISTINCT rid) fine_variants,COUNT(DISTINCT surface) exact_surface_variants,
        COUNT(DISTINCT physical) physical_variants,COUNT(DISTINCT neg) negation_variants,
        COUNT(DISTINCT ctx) evidence_variants
        FROM claims GROUP BY {field}''')
    return dict(db.execute(f'''SELECT COUNT(*) groups,COALESCE(SUM(n),0) claims,
        COALESCE(SUM(n=1),0) singleton_groups,COALESCE(SUM(n>1),0) shared_groups,
        COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0) shared_claims,
        COALESCE(SUM(source_keys>1),0) multi_source_key_groups,
        COALESCE(SUM(verified_papers>1),0) multi_verified_paper_groups,
        COALESCE(SUM(counted_papers>1),0) multi_verified_not_retracted_groups,
        COALESCE(SUM(fine_variants>1),0) multiple_fine_relation_groups,
        COALESCE(SUM(CASE WHEN fine_variants>1 THEN n ELSE 0 END),0) claims_in_multiple_fine_relation_groups,
        COALESCE(SUM(exact_surface_variants>1),0) multiple_exact_surface_groups,
        COALESCE(SUM(negation_variants>1),0) mixed_negation_groups,
        COALESCE(SUM(evidence_variants>1),0) multiple_evidence_context_groups,
        COALESCE(MAX(n),0) largest_group,COALESCE(MAX(counted_papers),0) most_counted_papers FROM {table}''').fetchone())


def bounded_examples(db, field, limit=5):
    require(field in GROUPINGS and type(limit) is int and 1 <= limit <= 10, 'invalid sample request')
    condition = 'n>1' if field == 'rid' else 'fine_variants>1'
    result = []
    for group in all_rows(db, f'''SELECT * FROM groups_{field} WHERE {condition}
            ORDER BY counted_papers DESC,source_keys DESC,n DESC,k LIMIT ?''', (limit,)):
        members = all_rows(db, f'''SELECT cid,rid,sid,sn,p,tid,tn,pk,status,countable,neg,ctx FROM claims
            WHERE {field}=? ORDER BY countable DESC,pk,cid LIMIT 4''', (group['k'],))
        result.append(dict(group=group, members=members, candidate_only=field != 'rid'))
    return result


def crosscheck_fine_groups(stats, campaign, shared, default, inclusive):
    expected = campaign['relation_evidence_counts']
    require(stats['groups'] == expected['all_fine_grained_relation_groups'], 'fine group census differs')
    require(stats['claims'] == expected['all_claims'] == campaign['counts']['claims'], 'claim census differs')
    require(stats['shared_groups'] == expected['shared_groups'] == len(shared), 'shared group census differs')
    require(stats['shared_claims'] == expected['indexed_claims'] == sum(g['claim_count'] for g in shared), 'shared member census differs')
    require(stats['multi_source_key_groups'] == expected['multi_paper_shared_groups'], 'raw source group census differs')
    require(stats['multi_verified_paper_groups'] == expected['verified_multi_paper_shared_groups'] == len(inclusive), 'verified group census differs')
    require(stats['multi_verified_not_retracted_groups'] == len(default), 'default current query differs')
    require(stats['groups'] == stats['singleton_groups'] + stats['shared_groups'], 'group decomposition differs')
    require(stats['claims'] == stats['singleton_groups'] + stats['shared_claims'], 'claim decomposition differs')


def distribution(counter):
    total = sum(counter.values())
    require(total > 0, 'empty distribution')
    def quantile(numerator, denominator):
        target = (total * numerator + denominator - 1) // denominator
        cumulative = 0
        for k, v in sorted(counter.items()):
            cumulative += v
            if cumulative >= target: return k
        raise AssertionError('unreachable quantile')
    return dict(records=total, mean=sum(k*v for k,v in counter.items())/total,
        minimum=min(counter), median=quantile(1,2), p90=quantile(9,10), maximum=max(counter),
        histogram={str(k):v for k,v in sorted(counter.items())})


def progress(phase, **values):
    state = dict(status='READ_ONLY_RUNNING', pid=os.getpid(), at=j.utc_now(), phase=phase, **values)
    j.atomic_json(OUTPUT / 'RUN_STATE.json', state)
    print(compact(state), flush=True)


def main():
    require(not (OUTPUT / 'REVIEW.json').exists(), 'current review already exists')
    control = j.OUTPUT / 'CAMPAIGN.json'; c = j.read_json(control)
    require(c['status'] == 'COMPLETED' and c['active_process'] is None, 'writer active')
    require(Path(c['current_acceptance']['path']).parent.name == 'round59_observational_semantics', 'R59 must be accepted first')
    require((Path(c['current_acceptance']['path']).parent / 'REPORT_VALIDATION.json').exists(), 'R59 report/retention incomplete')
    tests = ET.parse(OUTPUT / 'TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(tests.get('tests')) >= 15 and all(int(tests.get(k)) == 0 for k in ('failures','errors','skipped')), 'complete passing tests required')
    code = [j.fingerprint(p) for p in FILES]
    for k in ('current_acceptance','current_runtime_acceptance','current_entity_terms','current_paper_identities','current_paper_census','current_coverage'):
        require(j.fingerprint(c[k]['path']) == c[k], 'current proof changed: ' + k)
    receipt = j.read_json(c['current_acceptance']['path'])
    require(receipt['graph'] == c['current_graph'] and enabled(receipt), 'current symmetric relation policy differs')
    census = j.read_json(c['current_paper_census']['path'])
    require(census['graph'] == c['current_graph'] and census['independent_full_candidate_verified'], 'census not current')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    require(sha256(Path(census['database']['path'])) == census['database']['sha256'], 'current census full SHA differs')
    terms = IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    papers = VerifiedPaperIdentities(j.read_json(c['current_paper_identities']['path']))
    shared = list(current_shared_relations(control))
    default = list(find_shared_relations(control, minimum_papers=2))
    inclusive = list(find_shared_relations(control, minimum_papers=2, include_retracted=True))
    db = sqlite3.connect(':memory:', uri=True); initialize(db)
    db.execute('ATTACH DATABASE ? AS frozen', (Path(census['database']['path']).as_uri()+'?mode=ro',))
    counts, classes, source_shapes, edge_kinds, claim_outer, claim_inner = (Counter() for _ in range(6))
    node_union, edge_union = set(), set(); batch = []
    phase, last = 'FULL_CURRENT_GRAPH_REUSE_CENSUS', time.monotonic()
    def pulse():
        nonlocal last
        if time.monotonic()-last > 25:
            progress(phase); last = time.monotonic()
        return 0
    db.set_progress_handler(pulse, 200000)
    try:
        progress(phase)
        with hashed_reader(Path(c['current_graph']['path'])) as (reader, h):
            for kind, key, row in walk_graph(reader):
                counts[kind] += 1
                if kind == 'metadata':
                    require(enabled(row), 'graph symmetric policy differs')
                elif kind == 'node':
                    md = row.get('metadata') or {}; node_union.update(md)
                    category = node_class(key, row); classes[category] += 1
                    if category == 'claim':
                        source_shapes[type(md.get('source_paper')).__name__] += 1
                        claim_outer[len(md)] += 1; claim_inner[len(md.get('metadata') or {})] += 1
                        batch.append(claim_projection(row, terms, papers))
                        if len(batch) >= 5000: add_rows(db, batch); batch.clear()
                else:
                    md = row.get('metadata') or {}; edge_union.update(md)
                    edge_kinds['about' if row['relation_type']=='about' else 'claim_science' if md.get('claim_id') else 'other'] += 1
                if counts[kind] % 500000 == 0: progress(kind, counts=dict(counts))
            require(h.hexdigest() == c['current_graph']['sha256'], 'current graph full SHA differs')
        add_rows(db, batch); batch.clear(); db.commit()
        require((counts['node'],classes['claim'],counts['edge']) == tuple(c['counts'][k] for k in ('nodes','claims','edges')), 'current full counts differ')
        require((len(node_union),len(edge_union)) == (c['node_metadata_field_union'],c['edge_metadata_field_union']), 'metadata unions differ')
        phase = 'ALL_CLAIMS_VS_ACCEPTED_CENSUS'; progress(phase)
        mismatch = db.execute('''SELECT COUNT(*) FROM claims c LEFT JOIN frozen.claims f ON f.cid=c.cid
            WHERE f.cid IS NULL OR f.node_sha<>c.node_sha OR f.paper_sig<>c.paper_sig OR f.relation_id<>c.rid''').fetchone()[0]
        require(mismatch == 0 and db.execute('SELECT COUNT(*) FROM frozen.claims').fetchone()[0] == classes['claim'], 'claim/hash/paper/relation census mismatch')
        source_counts = all_rows(db, 'SELECT status,COUNT(*) claims,COUNT(DISTINCT pk) source_keys FROM claims GROUP BY status ORDER BY status')
        stats, samples = {}, {}
        for field in GROUPINGS:
            phase = 'GROUP_' + field; progress(phase)
            stats[field] = group_statistics(db, field); samples[field] = bounded_examples(db, field)
        crosscheck_fine_groups(stats['rid'], c, shared, default, inclusive)
        one_source = dict(db.execute('''SELECT COUNT(*) groups,COALESCE(SUM(n),0) claims FROM groups_rid WHERE source_keys=1''').fetchone())
        repeated_one_source = dict(db.execute('''SELECT COUNT(*) groups,COALESCE(SUM(n),0) claims FROM groups_rid WHERE source_keys=1 AND n>1''').fetchone())
        require(one_source['groups'] + stats['rid']['multi_source_key_groups'] == stats['rid']['groups'], 'source group decomposition differs')
        result = dict(status='CURRENT_READ_ONLY_REUSE_CENSUS_COMPLETE', at=j.utc_now(), graph=c['current_graph'],
            acceptance=c['current_acceptance'], runtime=c['current_runtime_acceptance'], census=c['current_paper_census'],
            census_database=census['database'], entity_terms=c['current_entity_terms'], paper_identities=c['current_paper_identities'],
            counts=c['counts'], node_classes=dict(classes), edge_kinds=dict(edge_kinds), source_paper_shapes=dict(source_shapes),
            source_status_counts=source_counts, groupings=stats, single_source_fine_groups=one_source,
            repeated_single_source_fine_groups=repeated_one_source, metadata_union=dict(nodes=len(node_union),edges=len(edge_union)),
            claim_outer_metadata=distribution(claim_outer), claim_inner_metadata=distribution(claim_inner),
            checked_claim_hashes=classes['claim'], actual_query_checks=dict(shared=len(shared),default=len(default),inclusive=len(inclusive)),
            full_current_graph_sha_verified=True, full_current_census_sha_verified=True,
            graph_mutations=0, graph_copies=0, record_preimages_saved=False, working_index_retained=False,
            scientific_equivalence_certified=False, cohorts_independence_certified=False,
            name_only_or_id_only_groupings_are_merge_authority=False, code=code,
            tests=j.fingerprint(OUTPUT / 'TEST_RESULTS.xml'),
            definitions=dict(rid='Accepted complete endpoint identities/names and predicate; correlation-only symmetry; evidence stays per claim.',
                surface='Case-preserving NFKC/whitespace full source endpoint names; ignores IDs. Diagnostic only.',
                folded='As surface with casefold; this can conflate species-specific symbols. Diagnostic only.',
                physical='Stored endpoint IDs and predicate; ignores complete surface scope. Diagnostic only.',
                source_keys='Resolved source keys; conflicted/unverified keys are not verified articles.',
                counted_papers='Verified article identity, excluding confirmed retracted sources; not a scientific quality or cohort guarantee.'))
    finally:
        db.close()
    require(j.read_json(control) == c and [j.fingerprint(p) for p in FILES] == code, 'source/code advanced during review')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    j.atomic_json(OUTPUT / 'BOUNDED_EXAMPLES.json', dict(graph=c['current_graph'], examples=samples, graph_modified=False))
    result['examples'] = j.fingerprint(OUTPUT / 'BOUNDED_EXAMPLES.json')
    j.atomic_json(OUTPUT / 'REVIEW.json', result)
    progress('COMPLETED', fine=stats['rid'], exact_surface_candidates=stats['surface'])


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        j.atomic_json(OUTPUT / 'RUN_STATE.json', dict(status='FAILED', at=j.utc_now(), pid=os.getpid(), error=repr(error), graph_mutations=0))
        raise
