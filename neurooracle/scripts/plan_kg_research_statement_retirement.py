"""Plan four exact research-intention retirements from a verified R45 source scan."""
from collections import Counter
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from apply_kg_explicit_pmid_provenance import authorities
from plan_kg_source_scope_resolution import expected_shared
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src import kg_research_statement_retirement as scope
from neurooracle.src.relation_evidence import strongest_paper_key
from neurooracle.src.correlation_grouping import POLICY

OUTPUT = j.OUTPUT/'round46_research_retirement'
REVIEW = j.OUTPUT/'round45_semantic_sources'


def expected_census(db, deleted, shared):
    require(deleted and len(deleted) == len(set(deleted)), 'unique nonempty deletion set required')
    slots = ','.join('?' for _ in deleted); args = tuple(sorted(deleted))
    kept = f'cid NOT IN ({slots})'
    paper_kept = f'sig IN (SELECT paper_sig FROM claims WHERE {kept})'
    return dict(
        claims=db.execute(f'SELECT COUNT(*) FROM claims WHERE {kept}', args).fetchone()[0],
        unique_bibliographies=db.execute(f'SELECT COUNT(DISTINCT paper_sig) FROM claims WHERE {kept}', args).fetchone()[0],
        distinct_legacy_source_keys=db.execute(f'SELECT COUNT(DISTINCT legacy_key) FROM claims WHERE {kept}', args).fetchone()[0],
        shared_claims=len(shared),
        missing_pmid_with_doi_claims=db.execute(f"SELECT COUNT(*) FROM claims c JOIN papers p ON p.sig=c.paper_sig WHERE c.{kept} AND p.pmid='' AND p.doi!=''", args).fetchone()[0],
        both_ids=db.execute(f"SELECT COUNT(DISTINCT pmid||'|'||doi) FROM papers WHERE {paper_kept} AND pmid!='' AND doi!=''", args).fetchone()[0])


def selected_reference_closure(exact, deleted, removals):
    selected = [r for r in exact if any(x['claim_id'] in deleted for x in r['references'])]
    allowed = {('node', cid) for cid in deleted} | {('edge', str(e['ordinal'])) for r in removals for e in r['owned_edges']}
    require({(r['kind'], str(r['key'])) for r in selected} == allowed, 'unexpected exact deletion reference')
    require(len(selected) == len(allowed), 'duplicate exact reference record')
    return selected


def main():
    require(not (OUTPUT/'PLAN.json').exists(), 'plan exists; inspect/resume')
    c = j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status'] == 'COMPLETED' and c['active_process'] is None and not c['rollback_retention'], 'accepted idle single-version boundary')
    require(Path(c['current_acceptance']['path']).parent.name == 'round44_source_resolution', 'expected accepted R44')
    receipt = j.read_json(c['current_acceptance']['path']); census = j.read_json(c['current_paper_census']['path'])
    for fp in receipt['code']: small_check(fp)
    inspection_fp = j.fingerprint(REVIEW/'SOURCE_INSPECTION.json'); inspection = j.read_json(inspection_fp['path'])
    require(inspection['status'] == 'INSPECTED_NOT_APPLIED' and inspection['graph'] == c['current_graph'] and
            inspection['source_acceptance'] == c['current_acceptance'] and inspection['source_census'] == c['current_paper_census'], 'current inspection boundary differs')
    require(inspection['full_source_sha_verified'] and inspection['full_census_sha_verified'] and not inspection['graph_modified'], 'complete read-only source proof required')
    for fp in [inspection['code'], inspection['public_manifest'], *inspection['artifacts'].values()]: small_check(fp)
    public = j.read_json(inspection['public_manifest']['path']); small_check(public['response'])
    proof = scope.public_source_proof(Path(public['response']['path']).read_text(encoding='utf8'))
    selected = set(scope.REVIEWS); projected = {r['claim_id']: r for r in rows(inspection['artifacts']['CURRENT_CLAIM_PROJECTIONS.jsonl']['path'])}
    owned = {r['claim_id']: r['edges'] for r in rows(inspection['artifacts']['CURRENT_OWNED_EDGE_PROJECTIONS.jsonl']['path'])}
    holds = {r['claim_id']: r for r in rows(c['current_issues']['path'])}
    require(selected <= set(holds) and len(holds) == 21, 'selected old semantic hold set differs')
    metadata = {}; removals = []
    for cid in sorted(selected):
        r = projected[cid]; md = dict(r['science'], id=cid, metadata=r['inner_science'], source_paper=r['source_paper'])
        review = scope.review_science(cid, r['claim_sha256'], md, proof)
        require(r['claim_sha256'] == holds[cid]['current_node_sha256'], 'current held claim differs')
        es = owned[cid]; require(len(es) == 3, 'selected closure size differs')
        expected = [(cid, md['subject_id'], 'about'), (cid, md['object_id'], 'about'),
                    (md['subject_id'], md['object_id'], md['predicate'])]
        require(sorted((e['source_id'],e['target_id'],e['relation_type']) for e in es) == sorted(expected), 'selected source closure endpoints differ')
        require(not any(inspection['detail_dependencies'][cid].values()), 'selected claim has offline detail dependency')
        require(sorted(holds[cid]['related_edge_ordinals']) == sorted(e['ordinal'] for e in es), 'held/source edge ordinals differ')
        removals.append(dict(claim_id=cid, claim_sha256=r['claim_sha256'], pmid=review['pmid'], doi=review['doi'], reason=review['reason'],
            source_proof=proof[cid], owned_edges=[dict(ordinal=e['ordinal'],edge_sha256=e['edge_sha256']) for e in sorted(es,key=lambda e:e['ordinal'])],
            no_null_or_replacement_claim_synthesized=True, source_document_not_deleted=True, record_preimages_saved=False))
        metadata[cid] = md
    exact = selected_reference_closure(rows(inspection['artifacts']['EXACT_SEMANTIC_REFERENCES.jsonl']['path']), selected, removals)
    removed_ordinals = sorted(e['ordinal'] for r in removals for e in r['owned_edges'])
    require(len(set(removed_ordinals)) == len(removed_ordinals) == 12, 'expected twelve distinct owned edges')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    db = sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro', uri=True)
    for cid in selected:
        require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone() == (scope.REVIEWS[cid]['sha'],), 'current census claim differs')
    shared = expected_shared(db, [], selected, rows(c['current_shared_relations']['path']))
    expected = expected_census(db, selected, shared)
    retired_papers = []
    for cid in sorted(selected):
        sig, legacy = db.execute('SELECT paper_sig,legacy_key FROM claims WHERE cid=?',(cid,)).fetchone()
        total = db.execute('SELECT COUNT(*) FROM claims WHERE paper_sig=?',(sig,)).fetchone()[0]
        require(total == inspection['paper_groups'][cid]['same_bibliography_claims'] == 1, 'last-claim bibliography scope differs')
        retired_papers.append(dict(claim_id=cid, pmid=scope.REVIEWS[cid]['pmid'], bibliography_sha256=sig, legacy_key=legacy,
            active_census_bibliography_retired=True, original_document_and_source_archive_preserved=True))
    db.close()
    papers, _ = authorities(c); audit = j.read_json(receipt['identity_audit']['path'])
    status = Counter(audit['claim_status']); reasons = Counter(audit['hold_reasons']); changed_keys = audit['verified_key_changes']
    source_answers = {}
    for cid, md in metadata.items():
        answer = papers.resolve(md); source_answers[cid] = answer
        status[answer['status']] -= 1; reasons.subtract(answer['reasons'])
        changed_keys -= int(answer['status'] == 'verified' and answer['paper_key'] != strongest_paper_key(md))
    require(all(v >= 0 for v in [*status.values(), *reasons.values(), changed_keys]), 'source status decrement differs')
    code = [j.fingerprint(p) for p in (Path(__file__), j.REPO/'neurooracle/src/kg_research_statement_retirement.py')]
    plan = dict(version=scope.VERSION, status='REVIEWED_NOT_APPLIED', at=j.utc_now(), graph=c['current_graph'],
        source_acceptance=c['current_acceptance'], detail_store=c['current_detail_store'], source_inspection=inspection_fp,
        source_public_manifest=inspection['public_manifest'], public_source=public['response'], public_source_proof=proof,
        review_notes=j.fingerprint(REVIEW/'PUBLIC_SOURCE_REVIEW.md'), source_full_sha_verified=True, source_census_full_sha_verified=True,
        code=code, deleted_claims=removals, removed_edge_ordinals=removed_ordinals, exact_deletion_references=exact,
        retired_active_bibliographies=retired_papers, source_paper_answers=source_answers,
        deletion_detail_dependencies={cid:inspection['detail_dependencies'][cid] for cid in sorted(selected)},
        expected_counts=dict(c['counts'], nodes=c['counts']['nodes']-4, claims=c['counts']['claims']-4, edges=c['counts']['edges']-12),
        expected_census_counts=expected, expected_shared_claim_ids=shared,
        expected_paper_claim_status={k:v for k,v in status.items() if v}, expected_paper_hold_reasons={k:v for k,v in reasons.items() if v},
        expected_verified_key_changes=changed_keys, relation_grouping=POLICY, deleted_claim_count=4, deleted_edges=12,
        expected_remaining_semantic_holds=17, new_scientific_claims=0, new_nodes=0, record_preimages_saved=False, graph_backups=0,
        boundaries=['Exact four research intentions only; no generic exclusion of reviews or weak/null studies.',
            'These are each their paper\'s sole current claim: four bibliographies leave the active census, original documents remain.',
            'All surviving nodes, scientific evidence, qualifiers and audit objects are unchanged; no new seals or negative findings.',
            'No models, training, re-extraction, formal full_v2 synchronization or rollback preimages.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json') == c and all(j.fingerprint(fp['path']) == fp for fp in code), 'source/code advanced')
    j.guards([c['current_graph'], c['current_detail_store'], c['formal_sources'], census['database']])
    OUTPUT.mkdir(exist_ok=True); j.atomic_json(OUTPUT/'PLAN.json', plan)
    print('R46_PLAN_COMPLETE_NOT_APPLIED', dict(deleted_claims=4, deleted_edges=12, retired_bibliographies=4,
        expected_counts=plan['expected_counts'], expected_census=expected, source_status=plan['expected_paper_claim_status']), flush=True)


if __name__ == '__main__': main()
