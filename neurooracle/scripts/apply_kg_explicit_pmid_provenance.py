"""R36: proven explicit PMIDs, exact edge references and current derived files.

One atomic candidate, no KG/record preimages. Independent full source/candidate
proofs mask only approved bibliography fields; models/formal KG are out of scope.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from apply_kg_bibliography_titles import CurrentCensus, FILES as PREVIOUS_FILES, small_check
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from audit_kg_publication_status import validate_current_identity_source, reviewed_payload
from audit_kg_explicit_pmid_provenance import Archive, review_pmid, proposed_node
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from inspect_kg_bibliography_sources import load_public_xml
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows, guard_native
from reclaim_kg_backup_storage import native_info, sha256
from resume_kg_bibliography_titles import require_process_ended
from neurooracle.src.kg_explicit_pmid_repair import reference_changes, apply_reference, masked_record, project_coverage
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, bibliography, identifiers, title_key, census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage, node_class
from neurooracle.src.relation_evidence import evidence_member, relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = journal.OUTPUT / "round36_explicit_pmid_provenance"
PREVIOUS = journal.OUTPUT / "round35_source_bibliography"
SOURCE = journal.OUTPUT / "round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP = SOURCE.with_name(SOURCE.name + ".explicit-pmid.tmp")
CATALOG = OUTPUT / "CURRENT_SHARED_RELATIONS.jsonl"
DATABASE = OUTPUT / "CURRENT_PAPER_CENSUS.sqlite"
FILES = list(dict.fromkeys([*PREVIOUS_FILES, Path(__file__), *[journal.REPO / p for p in (
    "neurooracle/scripts/audit_kg_explicit_pmid_provenance.py", "neurooracle/scripts/resume_kg_bibliography_titles.py",
    "neurooracle/src/kg_explicit_pmid_repair.py", "neurooracle/tests/test_kg_explicit_pmid_repair.py")]]))
_verified_census = None


def bindings(): return [journal.fingerprint(p) for p in FILES]


def progress(phase, **values):
    state = dict(status="RUNNING", pid=os.getpid(), phase=phase, at=journal.utc_now(), **values)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", state); print(compact(state), flush=True)


def authorities(baseline):
    global _verified_census
    small_check(baseline["current_paper_identities"]); small_check(baseline["current_paper_census"])
    payload = journal.read_json(baseline["current_paper_identities"]["path"])
    base = deepcopy(payload); base.pop("publication_reviews")
    manifests = [journal.read_json(journal.OUTPUT / p) for p in (
        "round31_paper_identity/AUTHORITY_FETCH.json", "round33_complete_title_evidence/XML_FETCH.json",
        "round34_publication_status/NOTICE_FETCH.json")]
    for manifest in manifests: small_check(manifest["request"])
    validate_current_identity_source(base, manifests[0], manifests[1])
    require(reviewed_payload(base, manifests[2]) == payload, "authority/publication evidence differs")
    census = journal.read_json(baseline["current_paper_census"]["path"])
    journal.guards(census["database"])
    if _verified_census != census["database"]:
        require(sha256(Path(census["database"]["path"])) == census["database"]["sha256"], "current source census SHA differs")
        db = sqlite3.connect(Path(census["database"]["path"]).as_uri() + "?mode=ro", uri=True)
        try: require(census_identifier_projection(db)["observed_collisions"] == payload["observed_collisions"], "global collision guards differ")
        finally: db.close()
        _verified_census = deepcopy(census["database"])
    return VerifiedPaperIdentities(payload), load_public_xml()


def load_plan(baseline):
    audit_fp = journal.fingerprint(OUTPUT / "PROVENANCE_AUDIT.json")
    audit = journal.read_json(audit_fp["path"])
    require(audit["graph"] == baseline["current_graph"] and audit["source_full_sha_verified"] and audit["census_full_sha_verified"], "plan source not current")
    for name, key in (("source_acceptance", "current_acceptance"), ("source_runtime", "current_runtime_acceptance"),
                      ("source_census", "current_paper_census"), ("source_registry", "current_paper_identities")):
        require(audit[name] == baseline[key], "plan baseline differs"); small_check(audit[name])
    for fp in [audit["repairs"], audit["edge_repairs"], audit["title_repair_predecessor"], audit["xml_fetch"], *audit["code"], *audit["original_archive_witnesses"]]:
        small_check(fp)
    nodes = {r["claim_id"]: r for r in rows(audit["repairs"]["path"])}
    edges = {r["ordinal"]: r for r in rows(audit["edge_repairs"]["path"])}
    titles = {r["claim_id"]: r for r in rows(audit["title_repair_predecessor"]["path"])}
    require(len(nodes) == audit["approved_pmid_nodes"] == 219 and not audit["pmid_holds"], "unreviewed node scope")
    require(len(edges) == audit["approved_edge_records"] == 673 and not audit["edge_reference_holds"], "unreviewed edge scope")
    require(len(titles) == 163 and not set(nodes) & set(titles), "title predecessor scope differs")
    return audit_fp, nodes, edges, titles


def test_evidence():
    tests, unique = Counter(), set()
    for suite in ET.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"):
        tests.update({k:int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
        for case in suite.findall("testcase"):
            key = (case.get("classname"), case.get("name")); require(key not in unique, "duplicate test"); unique.add(key)
    require(tests["tests"] == len(unique) >= 454 and not any(tests[k] for k in ("failures", "errors", "skipped")), "complete distinct passing tests required")
    return dict(tests)


def publication_row(prior, original, current, papers):
    require(digest(original) == prior["claim_sha256"], "publication reviewed source claim changed")
    out = deepcopy(prior); out["claim_sha256"] = digest(current)
    answer = papers.resolve(current["metadata"])
    if answer["status"] == "verified":
        require(answer["paper_key"] == "pmid:" + prior["pmid"], "publication identity changed to another source")
        require(papers.publication_review(answer["paper_key"])["status"] == prior["publication_status"], "publication status changed")
        out["identity_status"] = "verified_identity"
        out["disposition"] = "excluded_from_default_source_threshold_not_deleted" if out["publication_status"] == "retracted" else "corrected_version_retained_as_distinct_source"
    else:
        require(prior["identity_status"] == "source_reference_still_held", "previous verified publication became held")
    return out


def build():
    require(not any(p.exists() for p in (TEMP, CATALOG, DATABASE, OUTPUT / "BUILD_STATE.json")), "build exists; inspect/resume")
    baseline = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(baseline["status"] == "COMPLETED" and baseline["active_process"] is None and not baseline["rollback_retention"], "active writer/retention boundary")
    require(Path(baseline["current_graph"]["path"]).resolve() == SOURCE.resolve(), "unexpected current graph")
    require(shutil.disk_usage(SOURCE).free > baseline["current_graph"]["bytes"] + 12 * 1024**3, "insufficient atomic space")
    tests = test_evidence()
    for name in ("current_acceptance", "current_runtime_acceptance", "current_entity_terms", "current_shared_relations", "current_coverage", "current_issues", "current_structure_holds", "current_paper_issues"):
        small_check(baseline[name])
    old_receipt = journal.read_json(baseline["current_acceptance"]["path"])
    for fp in old_receipt["code"]: small_check(fp)
    plan_fp, node_events, edge_events, title_events = load_plan(baseline)
    papers, (articles, witnesses) = authorities(baseline)
    terms = VerifiedEntityTerms(journal.read_json(baseline["current_entity_terms"]["path"]))
    shared = {m["claim_id"] for g in rows(baseline["current_shared_relations"]["path"]) for m in g["members"]}
    old_census = journal.read_json(baseline["current_paper_census"]["path"])
    journal.guards([baseline["current_graph"], baseline["current_detail_store"], baseline["formal_sources"], old_census["database"]])
    targets = {r["claim_id"]:r for r in old_census["outer_pmid_conflicts"]}
    require(set(targets) == set(node_events), "outer PMID scope differs")
    pub_fp = journal.fingerprint(PREVIOUS / "CURRENT_PUBLICATION_REVIEW.json")
    pub = journal.read_json(pub_fp["path"]); prior_pub = {r["claim_id"]:r for r in pub["claims"]}
    require(pub["graph"] == baseline["current_graph"] and pub["registry"] == baseline["current_paper_identities"], "publication baseline differs")
    code = bindings()
    guards = [native_info(fp["path"]) for fp in (baseline["current_graph"], baseline["current_detail_store"], old_census["database"], *baseline["formal_sources"].values())]
    campaign = dict(baseline, status="MANUAL_ACTIVE", phase="R36显式PMID归位及边引用闭合；全图/普查/覆盖验收", updated_at=journal.utc_now(),
        active_process=dict(kind="explicit_pmid_provenance", pid=os.getpid(), state=str(OUTPUT / "RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json"); night["active_process"] = campaign["active_process"]
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    census, proof, archive = CurrentCensus(DATABASE, terms, shared), SourceProof(terms, papers), Archive()
    masked = {k:hashlib.sha256() for k in ("nodes", "edges")}
    before_status, after_status, reasons, selected_edges = Counter(), Counter(), Counter(), Counter()
    minus, plus = Coverage(), Coverage()
    originals, seen_edges, paper_issues, current_pub = {}, set(), [], []
    selected = set(node_events) | set(title_events)

    def transform(records):
        for kind, key, row in records:
            out = row; event = None
            if kind == "node" and key in selected:
                originals[key] = row
                if key in node_events:
                    reviewed, _ = review_pmid(row, targets[key], papers, articles, witnesses, archive)
                    require(reviewed == node_events[key], "original provenance plan did not reproduce")
                    event = node_events[key]; out = proposed_node(row, event)
                else: require(digest(row) == title_events[key]["current_node_sha256"], "R35 title claim changed")
            if kind == "edge":
                owner = (row.get("metadata") or {}).get("claim_id")
                if owner in selected:
                    selected_edges[owner] += 1
                    changed, bases, held = reference_changes(row, originals[owner], node_events.get(owner), title_events.get(owner))
                    require(not held, "edge provenance no longer complete")
                    if changed:
                        event = edge_events.get(int(key)); require(event is not None, "new unreviewed edge change")
                        require(event["set_fields"] == changed and event["reference_basis"] == bases and event["claim_id"] == owner, "edge review differs")
                        out = apply_reference(row, event); seen_edges.add(int(key))
                require(int(key) not in edge_events or event is not None, "planned edge missing")
            if kind == "node" and key.startswith("CLM:"):
                before_status[papers.resolve(row["metadata"])["status"]] += 1
                answer = papers.resolve(out["metadata"]); after_status[answer["status"]] += 1; reasons.update(answer["reasons"])
                if answer["status"] == "conflict": paper_issues.append(dict(claim_id=key, claim_sha256=digest(out), reasons=answer["reasons"], legacy_source_key=evidence_member(out["metadata"])["paper_key"], action="hold_no_claim_or_source_deletion"))
                census.add(key, out)
                if key in prior_pub: current_pub.append(publication_row(prior_pub[key], row, out, papers))
            if kind != "metadata":
                masked[kind + "s"].update(compact(masked_record(kind, row, event)).encode() + b"\n")
                if event:
                    scope = "node/claim" if kind == "node" else "edge/all"
                    minus.add(scope, row); plus.add(scope, out)
            yield kind, key, out

    progress("BUILD_EXPLICIT_PMID_AND_REFERENCES")
    try:
        with TEMP.open("xb", buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader, source_sha):
                result = stream_patch(proof.records(transform(walk_graph(reader))), handle, [], {}, CATALOG, progress,
                    relation_key_func=terms.relation_key, relation_version="kg.relation_evidence.v3", papers=papers)
                require(source_sha.hexdigest() == baseline["current_graph"]["sha256"], "full source SHA differs")
            handle.flush(); os.fsync(handle.fileno())
        require(set(originals) == selected and seen_edges == set(edge_events), "planned node/edge closure incomplete")
        require(set(selected_edges) == selected and dict(Counter(selected_edges.values())) == {3:347, 2:35}, "selected edge closure differs")
        require(dict(before_status) == dict(verified=10112, conflict=1671, unverified=893401), "baseline claim status differs")
        require(dict(after_status) == dict(verified=10331, conflict=1452, unverified=893401), "unapproved identity delta")
        require(result["counts"] == baseline["counts"] and not result["changed"], "unapproved stream change")
        for key in ("connected_components", "isolated_nodes", "self_loops"):
            require(result[key] == old_receipt[key], "topology changed: " + key)
        expected_unions = deepcopy(old_receipt["metadata_key_unions"]); expected_unions["nodes"].remove("pmid")
        require(result["metadata_key_unions"] == expected_unions, "metadata field union differs")
        for key in ("all_fine_grained_relation_groups", "all_claims", "shared_groups", "indexed_claims"):
            require(result["catalog_stats"][key] == baseline["relation_evidence_counts"][key], "relation membership changed")
        audit = dict(claim_status=dict(after_status), before_claim_status=dict(before_status), verified_key_changes=proof.key_changes,
            hold_reasons=dict(reasons), changed_pmid_claims=len(node_events), changed_edge_records=len(edge_events), relation_counts=result["catalog_stats"])
        proof.verify(result, audit)
        census_result = census.finish()
    except BaseException:
        try: census.db.close()
        except Exception: pass
        raise
    require(not census_result["outer_pmid_conflicts"], "outer PMID conflict remains")
    require(census_result["projection"]["observed_collisions"] == papers.export_payload()["observed_collisions"], "new global collision requires review")
    coverage = project_coverage(rows(baseline["current_coverage"]["path"]), minus, plus)
    require(len(coverage) == 434 and not any(r["scope"] == "node/claim" and r["field"] == "metadata.pmid" for r in coverage), "coverage projection differs")
    require({r["claim_id"] for r in current_pub} == set(prior_pub), "publication review closure incomplete")
    pub.update(claims=current_pub, binding_method="all claim hashes and publication identities reproduced during full current graph scans")
    write_rows(OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl", coverage)
    write_rows(OUTPUT / "CURRENT_PAPER_ISSUES.jsonl", paper_issues)
    journal.atomic_json(OUTPUT / "PAPER_IDENTITY_AUDIT.json", audit)
    journal.atomic_json(OUTPUT / "PUBLICATION_REVIEW_BUILD.json", pub)
    db_fp = {**cheap(DATABASE), "sha256":sha256(DATABASE)}
    journal.atomic_json(OUTPUT / "CENSUS_BUILD.json", dict(**census_result, database=db_fp))
    guard_native(guards); require(bindings() == code, "frozen code changed"); small_check(plan_fp); small_check(pub_fp)
    for fp in archive.files.values(): small_check(fp)
    state = dict(status="BUILT_NOT_ADOPTED", at=journal.utc_now(), baseline=baseline, protected_native=guards,
        temporary=cheap(TEMP), result=result, code=code, plan=plan_fp, catalog=journal.fingerprint(CATALOG),
        papers=baseline["current_paper_identities"], terms=baseline["current_entity_terms"], database=db_fp,
        masked_source_record_digests={k:v.hexdigest() for k,v in masked.items()}, test_counts=tests,
        source_full_sha_verified=True, graph_backups=0, record_preimages_saved=False)
    for key, file in dict(census_build="CENSUS_BUILD.json", audit="PAPER_IDENTITY_AUDIT.json", paper_issues="CURRENT_PAPER_ISSUES.jsonl",
        coverage="CURRENT_METADATA_COVERAGE.jsonl", publication="PUBLICATION_REVIEW_BUILD.json", tests="TEST_RESULTS.xml").items():
        state[key] = journal.fingerprint(OUTPUT / file)
    journal.atomic_json(OUTPUT / "BUILD_STATE.json", state)
    progress("BUILT_NOT_ADOPTED", identity=audit["claim_status"], relations=result["catalog_stats"])


def validate():
    require(not (OUTPUT / "VALIDATED.json").exists(), "validation exists; inspect/resume")
    state = journal.read_json(OUTPUT / "BUILD_STATE.json"); baseline = state["baseline"]
    require(bindings() == state["code"] and cheap(TEMP) == state["temporary"], "build/code changed")
    guard_native(state["protected_native"])
    for key in ("plan", "catalog", "papers", "terms", "census_build", "audit", "paper_issues", "coverage", "publication", "tests"): small_check(state[key])
    _, node_events, edge_events, title_events = load_plan(baseline)
    papers, (articles, witnesses) = authorities(baseline)
    terms = VerifiedEntityTerms(journal.read_json(state["terms"]["path"]))
    proof, archive, coverage = SourceProof(terms, papers), Archive(), Coverage()
    masked = {k:hashlib.sha256() for k in ("nodes", "edges")}
    seen_nodes, seen_edges, paper_sigs, shared_sigs, matched = set(), set(), set(), set(), 0
    issues, outer = [], []
    pub = journal.read_json(state["publication"]["path"]); pub_rows = {r["claim_id"]:r for r in pub["claims"]}; seen_pub = set()
    db = sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True)

    def independent(records):
        nonlocal matched
        for kind, key, row in records:
            event = None
            if kind == "node" and key in node_events:
                event = node_events[key]; require(digest(row) == event["current_node_sha256"], "PMID result hash differs")
                prior = deepcopy(row); prior["metadata"]["pmid"] = event["set_source_paper_pmid"]
                require(event["source_pmid_shape"] in ("missing", "original_queue_id"), "unsupported source PMID shape")
                prior["metadata"]["source_paper"]["pmid"] = "" if event["source_pmid_shape"] == "missing" else prior["metadata"]["paper_id"]
                expected = dict(claim_sha256=event["source_node_sha256"], alternate_pmid=event["set_source_paper_pmid"])
                reviewed, _ = review_pmid(prior, expected, papers, articles, witnesses, archive)
                require(reviewed == event and proposed_node(prior, event) == row, "independent original provenance not reproduced")
                seen_nodes.add(key)
            if kind == "edge" and int(key) in edge_events:
                event = edge_events[int(key)]
                require(digest(row) == event["current_edge_sha256"] and all(row[k] == v for k,v in event["set_fields"].items()), "reference result differs")
                require(row["metadata"]["claim_id"] == event["claim_id"], "reference owner changed")
                seen_edges.add(int(key))
            if kind != "metadata":
                coverage.add("node/" + node_class(key, row) if kind == "node" else "edge/all", row)
                masked[kind + "s"].update(compact(masked_record(kind, row, event)).encode() + b"\n")
            if kind == "node" and key.startswith("CLM:"):
                md = row["metadata"]; paper = bibliography(md["source_paper"]); sig = digest(paper)
                member = evidence_member(md); rid = relation_id(terms.relation_key(md))
                record = db.execute("SELECT node_sha,paper_sig,legacy_key,relation_id,shared FROM claims WHERE cid=?", (key,)).fetchone()
                require(record is not None and record[:4] == (digest(row), sig, member["paper_key"], rid), "census claim differs")
                if sig not in paper_sigs:
                    ids = identifiers(paper)
                    require(db.execute("SELECT payload,pmid,doi,pmcid,title,year FROM papers WHERE sig=?", (sig,)).fetchone() ==
                        (compact(paper), ids["pmid"], ids["doi"], ids["pmcid"], title_key(paper.get("title")), str(paper.get("year") or paper.get("publication_year") or "")), "census paper differs")
                    paper_sigs.add(sig)
                if record[4]:
                    require(db.execute("SELECT member_json FROM shared_members WHERE cid=?", (key,)).fetchone() == (compact(member),), "shared census differs")
                    shared_sigs.add(key)
                for holder in (md, md.get("metadata") or {}):
                    if holder.get("pmid") and str(holder["pmid"]) != str(paper.get("pmid") or ""): outer.append(key)
                answer = papers.resolve(md)
                if answer["status"] == "conflict": issues.append(dict(claim_id=key, claim_sha256=digest(row), reasons=answer["reasons"], legacy_source_key=member["paper_key"], action="hold_no_claim_or_source_deletion"))
                if key in pub_rows:
                    require(publication_row(pub_rows[key], row, row, papers) == pub_rows[key], "current publication review differs")
                    seen_pub.add(key)
                matched += 1
            yield kind, key, row

    progress("INDEPENDENT_FULL_CANDIDATE_CENSUS_COVERAGE_SCAN")
    try:
        with hashed_reader(TEMP) as (reader, graph_sha):
            checks = verify_catalog_and_graph(proof.records(independent(walk_graph(reader))), state["result"], CATALOG, progress,
                relation_key_func=terms.relation_key, papers=papers)
            candidate_sha = graph_sha.hexdigest()
        census = journal.read_json(state["census_build"]["path"])
        require(matched == db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 905184, "claim census closure differs")
        require(len(paper_sigs) == db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], "stale census paper")
        require(len(shared_sigs) == db.execute("SELECT COUNT(*) FROM shared_members").fetchone()[0] == 5861, "shared census closure differs")
        require(shared_sigs == {m["claim_id"] for g in rows(CATALOG) for m in g["members"]} and not outer and not census["outer_pmid_conflicts"], "shared flag/outer PMID differs")
        require(census_identifier_projection(db) == census["projection"], "census normalization differs")
        require(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "census integrity failed")
    finally: db.close()
    require(seen_nodes == set(node_events) and seen_edges == set(edge_events), "repair closure differs")
    require({k:v.hexdigest() for k,v in masked.items()} == state["masked_source_record_digests"], "non-approved record fields changed")
    require(issues == rows(state["paper_issues"]["path"]) and seen_pub == set(pub_rows), "current issue/publication scope differs")
    proof.verify(state["result"], journal.read_json(state["audit"]["path"]))
    require(coverage.rows() == rows(state["coverage"]["path"]), "independent coverage differs from exact source delta")
    progress("FULL_DETAIL_AND_CENSUS_SHA_BOUNDARY")
    require(sha256(Path(baseline["current_detail_store"]["path"])) == baseline["current_detail_store"]["sha256"], "detail SHA differs")
    require(sha256(DATABASE) == state["database"]["sha256"], "candidate census SHA differs")
    guard_native(state["protected_native"]); require(bindings() == state["code"], "code changed during validation")
    for fp in archive.files.values(): small_check(fp)
    for k in ("records_match_approved_identity_only_transform", "all_nonidentity_claim_fields_preserved"): checks.pop(k, None)
    checks.update(exact_approved_pmid_and_edge_reference_transform=True, all_other_node_and_edge_fields_preserved=True,
        original_quotes_negation_conditions_independent_sources_preserved=True, original_explicit_pmid_and_full_abstract_provenance_reproduced=True,
        current_census_all_claims_papers_and_shared_members_verified=True, metadata_coverage_full_scan_matches_exact_delta=True,
        verified_identity_proofs_complete=True, paper_identity_witnesses_validated=True, publication_status_witnesses_validated=True,
        no_claim_or_edge_metadata_fields_added=True, complete_title_and_identifier_conflicts_held=True,
        authority_verified_claims=proof.claim_status["verified"], verified_paper_key_changes=proof.key_changes,
        distinct_articles_not_independent_cohorts=True)
    journal.atomic_json(OUTPUT / "VALIDATED.json", dict(status="VALIDATED_NOT_ADOPTED", at=journal.utc_now(),
        graph={**cheap(TEMP), "sha256":candidate_sha}, temporary_native=native_info(TEMP), database_native=native_info(DATABASE),
        checks=checks, build_state=journal.fingerprint(OUTPUT / "BUILD_STATE.json"), metadata_coverage_rows=len(coverage.rows())))
    progress("VALIDATED_NOT_ADOPTED", claims=dict(proof.claim_status))


def apply():
    require(not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "already adopted")
    state = journal.read_json(OUTPUT / "BUILD_STATE.json"); accepted = journal.read_json(OUTPUT / "VALIDATED.json")
    small_check(accepted["build_state"]); require(bindings() == state["code"], "frozen code changed")
    guard_native([*state["protected_native"], accepted["temporary_native"], accepted["database_native"]])
    for key in ("plan", "catalog", "papers", "terms", "census_build", "audit", "paper_issues", "coverage", "publication", "tests"): small_check(state[key])
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json"); baseline = state["baseline"]
    require(campaign["current_graph"] == baseline["current_graph"] and campaign["active_process"]["kind"] == "explicit_pmid_provenance" and campaign["active_process"]["pid"] == os.getpid(), "source/writer advanced")
    require(TEMP.resolve().parent == SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()), "unsafe replacement")
    _, node_events, edge_events, _ = load_plan(baseline)
    held_files = {}
    for field, filename in (("current_issues", "CURRENT_REMAINING_ISSUES.jsonl"), ("current_structure_holds", "CURRENT_STRUCTURE_HOLDS.jsonl")):
        small_check(baseline[field]); held_files[filename] = rows(baseline[field]["path"])
        require(not set(node_events) & {r.get("claim_id") for r in held_files[filename]}, "scientific hold requires rebinding")
    os.replace(TEMP, SOURCE)
    graph = {**cheap(SOURCE), "sha256":accepted["graph"]["sha256"]}
    require(graph["bytes"] == accepted["graph"]["bytes"] and graph["mtime_ns"] == accepted["graph"]["mtime_ns"], "replacement differs")
    for filename, held in held_files.items(): write_rows(OUTPUT / filename, [dict(r, current_graph=graph, version_binding_status="FULL_CURRENT_GRAPH_VALIDATED") for r in held])
    census = journal.read_json(state["census_build"]["path"]); projection = census.pop("projection")
    journal.atomic_json(OUTPUT / "CENSUS.json", dict(**census, status="CURRENT_CENSUS_COMPLETE", at=journal.utc_now(), graph=graph,
        source_full_sha_verified=True, independent_full_candidate_verified=True, code=state["code"], record_preimages_saved=False))
    census_fp = journal.fingerprint(OUTPUT / "CENSUS.json")
    journal.atomic_json(OUTPUT / "CENSUS_NORMALIZATION.json", dict(**projection, census=census_fp))
    pub = journal.read_json(state["publication"]["path"]); pub.update(at=journal.utc_now(), graph=graph, current_census=census_fp)
    journal.atomic_json(OUTPUT / "CURRENT_PUBLICATION_REVIEW.json", pub)
    result = state["result"]; audit = journal.read_json(state["audit"]["path"])
    receipt = dict(status="CURRENT_EXPLICIT_PMID_PROVENANCE_APPLIED", at=journal.utc_now(), graph=graph, counts=result["counts"],
        changed=dict(claim_nodes=len(node_events), edge_records=len(edge_events), source_paper_pmid_fields=219, redundant_outer_pmid_fields_removed=219,
            edge_source_fields=219, edge_title_fields=454), connected_components=result["connected_components"], isolated_nodes=result["isolated_nodes"],
        self_loops=result["self_loops"], metadata_key_unions=result["metadata_key_unions"], metadata=result["metadata"], source_graph=baseline["current_graph"], source_kg_retained=False,
        shared_relations=state["catalog"], entity_terms=state["terms"], paper_identities=state["papers"], paper_issues=state["paper_issues"], metadata_coverage=state["coverage"],
        current_paper_census=census_fp, relation_counts=result["catalog_stats"], checks=accepted["checks"], tests=state["tests"], test_counts=state["test_counts"], code=state["code"],
        detail_store=baseline["current_detail_store"], identity_audit=state["audit"], provenance_plan=state["plan"], rollback_retention=False, record_preimages_saved=False, formal_apply_performed=False)
    journal.atomic_json(OUTPUT / "CURRENT_ACCEPTANCE.json", receipt)
    journal.atomic_json(OUTPUT / "CURRENT_RUNTIME_ACCEPTANCE.json", dict(status="CURRENT_RUNTIME_VALIDATED", at=receipt["at"], graph=graph,
        acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), code=state["code"], tests=state["tests"], test_counts=state["test_counts"]))
    campaign.update(status="COMPLETED", phase="R36显式PMID归位、边引用同步、metadata与当前普查已验收", active_process=None, updated_at=receipt["at"], current_graph=graph,
        current_acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), current_runtime_acceptance=journal.fingerprint(OUTPUT / "CURRENT_RUNTIME_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"), current_structure_holds=journal.fingerprint(OUTPUT / "CURRENT_STRUCTURE_HOLDS.jsonl"),
        current_shared_relations=state["catalog"], current_paper_identities=state["papers"], current_paper_issues=state["paper_issues"], current_coverage=state["coverage"],
        current_paper_census=census_fp, current_census_normalization=journal.fingerprint(OUTPUT / "CENSUS_NORMALIZATION.json"), relation_evidence_counts=result["catalog_stats"],
        last_deep_verification=accepted["at"], last_current_graph_content_verification=dict(graph=graph, independent_full_scan=True),
        next_steps=["219外层PMID已归位并去重，673边引用同步。剩1452书目待核，不猜合成ID后缀。",
            "当前普查/覆盖/共享索引在R36；无历史KG或记录前像。", "继续语义、完整复合表达、FANCE/VCP及结构问题；无模型训练或正式同步。"])
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json")
    night.update(active_process=None, latest_acceptance=campaign["current_acceptance"], latest_census=census_fp, latest_completed_batch="R36",
        latest_completed_runtime_batch="R36", latest_runtime_acceptance=campaign["current_runtime_acceptance"])
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", dict(status="COMPLETED", at=receipt["at"], graph=graph, changed=receipt["changed"], relations=result["catalog_stats"]))
    print(compact(dict(status=receipt["status"], changed=receipt["changed"], relations=result["catalog_stats"])), flush=True)


def resume():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    require(bindings() == state["code"] and not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "not a resumable build")
    c = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(c["current_graph"] == state["baseline"]["current_graph"] and c["active_process"]["kind"] == "explicit_pmid_provenance", "unexpected active batch")
    require_process_ended(c["active_process"]["pid"])
    c["active_process"]["pid"] = os.getpid(); journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", c)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json"); night["active_process"] = c["active_process"]
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    if not (OUTPUT / "VALIDATED.json").exists(): validate()
    apply()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("phase", choices=("run", "build", "resume"))
    phase = parser.parse_args().phase
    if phase == "run": build(); validate(); apply()
    elif phase == "build": build()
    else: resume()
