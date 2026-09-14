"""R35: repair isolated titles; rebuild the current census, never save preimages.

Two independent complete graph scans, exact non-title record proofs, owning
public XML and the unchanged scientific/structural checks gate atomic adoption.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from apply_kg_paper_identity import SourceProof, FILES as ENGINE_FILES
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from audit_kg_publication_status import validate_current_identity_source, reviewed_payload, FILES as AUTHORITY_FILES
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from inspect_kg_bibliography_sources import load_public_xml
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows, guard_native
from reclaim_kg_backup_storage import native_info, sha256
from neurooracle.src.kg_bibliography_repair import review_title, apply_title, verify_title_result, without_selected_title
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, bibliography, identifiers, title_key, census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage, node_class
from neurooracle.src.relation_evidence import evidence_member, relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = journal.OUTPUT / "round35_source_bibliography"
R31 = journal.OUTPUT / "round31_paper_identity"
R33 = journal.OUTPUT / "round33_complete_title_evidence"
R34 = journal.OUTPUT / "round34_publication_status"
SOURCE = journal.OUTPUT / "round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP = SOURCE.with_name(SOURCE.name + ".bibliography-titles.tmp")
CATALOG = OUTPUT / "CURRENT_SHARED_RELATIONS.jsonl"
DATABASE = OUTPUT / "CURRENT_PAPER_CENSUS.sqlite"
EVENTS = OUTPUT / "TITLE_REPAIRS.jsonl"
FILES = list(dict.fromkeys([*ENGINE_FILES, *AUTHORITY_FILES, Path(__file__),
    Path(__file__).with_name("inspect_kg_bibliography_sources.py"), *[journal.REPO / p for p in (
        "neurooracle/src/kg_bibliography_repair.py", "neurooracle/src/metadata_field_audit.py",
        "neurooracle/tests/test_kg_bibliography_repair.py", "neurooracle/tests/test_kg_publication_status.py")]]))
_verified = None


def bindings(): return [journal.fingerprint(p) for p in FILES]


def progress(phase, **values):
    state = dict(status="RUNNING", pid=os.getpid(), phase=phase, at=journal.utc_now(), **values)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", state)
    print(compact(state), flush=True)


def small_check(fp):
    require(journal.fingerprint(Path(fp["path"])) == fp, "frozen small artifact changed: " + fp["path"])


def authorities(baseline):
    global _verified
    small_check(baseline["current_paper_identities"])
    payload = journal.read_json(baseline["current_paper_identities"]["path"])
    base = deepcopy(payload); base.pop("publication_reviews")
    paths = [R31 / "AUTHORITY_FETCH.json", R33 / "XML_FETCH.json", R34 / "NOTICE_FETCH.json"]
    manifests = [journal.read_json(p) for p in paths]
    for manifest in manifests: small_check(manifest["request"])
    validate_current_identity_source(base, manifests[0], manifests[1])
    require(reviewed_payload(base, manifests[2]) == payload, "current publication/identity evidence differs")
    census = journal.read_json(R31 / "CENSUS.json"); journal.guards(census["database"])
    if _verified != census["database"]:
        require(sha256(Path(census["database"]["path"])) == census["database"]["sha256"], "source census SHA differs")
        db = sqlite3.connect(Path(census["database"]["path"]).as_uri() + "?mode=ro", uri=True)
        require(census_identifier_projection(db)["observed_collisions"] == payload["observed_collisions"], "global collision guards differ")
        db.close(); _verified = deepcopy(census["database"])
    return VerifiedPaperIdentities(payload), load_public_xml()


class CurrentCensus:
    """A fresh current derived database, not a copy of old bibliography rows."""
    def __init__(self, path, terms, shared):
        require(not path.exists(), "census already exists")
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE papers(sig TEXT PRIMARY KEY,payload TEXT,pmid TEXT,doi TEXT,pmcid TEXT,title TEXT,year TEXT)")
        self.db.execute("CREATE TABLE claims(cid TEXT PRIMARY KEY,node_sha TEXT,paper_sig TEXT,legacy_key TEXT,relation_id TEXT,shared INT)")
        self.db.execute("CREATE TABLE shared_members(cid TEXT PRIMARY KEY,member_json TEXT)")
        self.terms, self.shared = terms, shared
        self.seen, self.batch, self.outer, self.invalid = set(), [], [], Counter()

    def add(self, key, row):
        md = row["metadata"]; paper = bibliography(md.get("source_paper")); sig = digest(paper); ids = identifiers(paper)
        if sig not in self.seen:
            self.seen.add(sig)
            self.db.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?)", (sig, compact(paper), ids["pmid"], ids["doi"], ids["pmcid"],
                title_key(paper.get("title")), str(paper.get("year") or paper.get("publication_year") or "")))
        for field in ids: self.invalid[field] += bool(paper.get(field) and not ids[field])
        for holder in (md, md.get("metadata") or {}):
            if holder.get("pmid") and str(holder["pmid"]) != str(paper.get("pmid") or ""):
                self.outer.append(dict(claim_id=key, source_paper_pmid=paper.get("pmid"), alternate_pmid=holder["pmid"], claim_sha256=digest(row)))
        member = evidence_member(md)
        self.batch.append((key, digest(row), sig, member["paper_key"], relation_id(self.terms.relation_key(md)), int(key in self.shared)))
        if key in self.shared: self.db.execute("INSERT INTO shared_members VALUES (?,?)", (key, compact(member)))
        if len(self.batch) >= 5000:
            self.db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?)", self.batch); self.batch.clear(); self.db.commit()

    def finish(self):
        self.db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?)", self.batch); self.batch.clear()
        for col in ("pmid", "doi", "pmcid", "title"): self.db.execute(f"CREATE INDEX papers_{col} ON papers({col})")
        for col in ("paper_sig", "legacy_key", "relation_id"): self.db.execute(f"CREATE INDEX claims_{col} ON claims({col})")
        self.db.commit()
        counts = dict(claims=self.db.execute("SELECT COUNT(*) FROM claims").fetchone()[0], unique_bibliographies=len(self.seen),
            distinct_legacy_source_keys=self.db.execute("SELECT COUNT(DISTINCT legacy_key) FROM claims").fetchone()[0],
            shared_claims=self.db.execute("SELECT COUNT(*) FROM shared_members").fetchone()[0],
            missing_pmid_with_doi_claims=self.db.execute("SELECT COUNT(*) FROM claims c JOIN papers p ON p.sig=c.paper_sig WHERE p.pmid='' AND p.doi!=''").fetchone()[0],
            both_ids=self.db.execute("SELECT COUNT(DISTINCT pmid||'|'||doi) FROM papers WHERE pmid!='' AND doi!=''").fetchone()[0])
        projection = census_identifier_projection(self.db)
        require(self.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "census integrity failed")
        self.db.close()
        return dict(counts=counts, invalid_identifiers=dict(self.invalid), outer_pmid_conflicts=self.outer, projection=projection)


def build():
    require(not any(p.exists() for p in (TEMP, CATALOG, DATABASE, EVENTS, OUTPUT / "BUILD_STATE.json")), "build exists; inspect/resume")
    baseline = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(baseline["status"] == "COMPLETED" and baseline["active_process"] is None and not baseline["rollback_retention"], "active writer/retention boundary")
    require(Path(baseline["current_graph"]["path"]).resolve() == SOURCE.resolve(), "unexpected current graph")
    require(shutil.disk_usage(SOURCE).free > baseline["current_graph"]["bytes"] + 12 * 1024**3, "insufficient atomic space")
    test_counts = Counter()
    for suite in ET.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"):
        test_counts.update({k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
    require(test_counts["tests"] >= 416 and not any(test_counts[k] for k in ("failures", "errors", "skipped")), "complete passing tests required")
    for name in ("current_acceptance", "current_runtime_acceptance", "current_entity_terms", "current_shared_relations",
                 "current_coverage", "current_issues", "current_structure_holds", "current_paper_issues"):
        small_check(baseline[name])
    old_receipt = journal.read_json(baseline["current_acceptance"]["path"])
    for fp in old_receipt["code"]: small_check(fp)
    journal.guards([baseline["current_graph"], baseline["current_detail_store"], baseline["formal_sources"]])
    inspected = journal.read_json(OUTPUT / "SOURCE_INSPECTION.json")
    require(inspected["graph"] == baseline["current_graph"] and inspected["source_full_sha_verified"], "inspection not current")
    targets = {r["claim_id"]: r["claim_sha256"] for r in inspected["members"] if r["categories"] == ["title"]}
    require(len(targets) == 196, "unreviewed title scope")
    scope_fp = journal.fingerprint(OUTPUT / "REVIEWED_SCOPE.json")
    scope = journal.read_json(scope_fp["path"])
    approved = {r["claim_id"]: r for r in scope["eligible"]}
    require(scope["graph"] == baseline["current_graph"] and len(approved) == 163 and set(approved) <= set(targets), "reviewed scope not current")
    papers, (articles, witnesses) = authorities(baseline)
    terms = VerifiedEntityTerms(journal.read_json(baseline["current_entity_terms"]["path"]))
    shared = {m["claim_id"] for g in rows(baseline["current_shared_relations"]["path"]) for m in g["members"]}
    code = bindings(); inspection_fp = journal.fingerprint(OUTPUT / "SOURCE_INSPECTION.json")
    guarded = [native_info(fp["path"]) for fp in (baseline["current_graph"], baseline["current_detail_store"], *baseline["formal_sources"].values())]
    campaign = dict(baseline, status="MANUAL_ACTIVE", phase="R35仅书目标题纠正、全图与普查重建验收", updated_at=journal.utc_now(),
        active_process=dict(kind="bibliography_titles", pid=os.getpid(), state=str(OUTPUT / "RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json"); night["active_process"] = campaign["active_process"]
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    census = CurrentCensus(DATABASE, terms, shared)
    proof = SourceProof(terms, papers)
    masked = {k: hashlib.sha256() for k in ("nodes", "edges")}
    seen, events, holds = set(), [], []
    minus, plus = Coverage(), Coverage()
    before_status, after_status, reasons = Counter(), Counter(), Counter()
    paper_issues = []

    def transform(records):
        for kind, key, row in records:
            out = row
            if kind == "node" and key in targets:
                require(digest(row) == targets[key], "selected claim changed")
                seen.add(key)
                event, reason = review_title(row, papers, articles, witnesses)
                if event:
                    out = apply_title(row, event); events.append(event)
                    minus.add("node/claim", row); plus.add("node/claim", out)
                else:
                    holds.append(dict(claim_id=key, claim_sha256=digest(row), reason=reason))
            if kind == "node" and key.startswith("CLM:"):
                before_status[papers.resolve(row["metadata"])["status"]] += 1
                answer = papers.resolve(out["metadata"]); after_status[answer["status"]] += 1; reasons.update(answer["reasons"])
                if answer["status"] == "conflict":
                    paper_issues.append(dict(claim_id=key, claim_sha256=digest(out), reasons=answer["reasons"],
                        legacy_source_key=evidence_member(out["metadata"])["paper_key"], action="hold_no_claim_or_source_deletion"))
                census.add(key, out)
            if kind != "metadata":
                value = without_selected_title(row) if out is not row else row
                masked[kind + "s"].update(compact(value).encode() + b"\n")
            yield kind, key, out

    progress("BUILD_TITLES_AND_CURRENT_CENSUS")
    try:
        with TEMP.open("xb", buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader, source_sha):
                result = stream_patch(proof.records(transform(walk_graph(reader))), handle, [], {}, CATALOG, progress,
                    relation_key_func=terms.relation_key, relation_version="kg.relation_evidence.v3", papers=papers)
                require(source_sha.hexdigest() == baseline["current_graph"]["sha256"], "full current source SHA differs")
            handle.flush(); os.fsync(handle.fileno())
        require(seen == set(targets) and {e["claim_id"] for e in events} == set(approved), "title review batch differs; do not adopt")
        require(all(all(e[k] == approved[e["claim_id"]][k] for k in ("source_node_sha256", "current_node_sha256")) for e in events), "reviewed per-claim title result differs")
        require(dict(before_status) == dict(verified=9949, conflict=1834, unverified=893401), "baseline identity counts changed")
        require(dict(after_status) == dict(verified=10112, conflict=1671, unverified=893401), "unexpected identity delta")
        require(minus.rows() == plus.rows(), "metadata coverage changed; cannot reuse current counts")
        require(result["counts"] == baseline["counts"] and not result["changed"], "unapproved stream change")
        for key in ("connected_components", "isolated_nodes", "self_loops", "metadata_key_unions"):
            require(result[key] == old_receipt[key], "non-title structure differs: " + key)
        for key in ("all_fine_grained_relation_groups", "all_claims", "shared_groups", "indexed_claims"):
            require(result["catalog_stats"][key] == baseline["relation_evidence_counts"][key], "relation membership changed")
        audit = dict(claim_status=dict(after_status), verified_key_changes=proof.key_changes)
        proof.verify(result, audit)
        census_result = census.finish()
    except BaseException:
        try: census.db.close()
        except Exception: pass
        raise
    require(census_result["counts"]["claims"] == baseline["counts"]["claims"] and census_result["counts"]["shared_claims"] == len(shared), "incomplete current census")
    require(census_result["projection"]["observed_collisions"] == papers.export_payload()["observed_collisions"], "collision guards changed")
    require(len(census_result["outer_pmid_conflicts"]) == 219, "outer PMID conflict changed")
    write_rows(EVENTS, events); write_rows(OUTPUT / "TITLE_HOLDS.jsonl", holds)
    write_rows(OUTPUT / "CURRENT_PAPER_ISSUES.jsonl", paper_issues)
    audit.update(before_claim_status=dict(before_status), hold_reasons=dict(reasons), changed_claim_titles=len(events),
        reviewed_title_claims=len(targets), title_holds=len(holds), relation_counts=result["catalog_stats"])
    journal.atomic_json(OUTPUT / "PAPER_IDENTITY_AUDIT.json", audit)
    db_fp = {**cheap(DATABASE), "sha256": sha256(DATABASE)}
    journal.atomic_json(OUTPUT / "CENSUS_BUILD.json", dict(**census_result, database=db_fp))
    guard_native(guarded); require(bindings() == code, "code changed during build"); small_check(inspection_fp); small_check(scope_fp)
    state = dict(status="BUILT_NOT_ADOPTED", at=journal.utc_now(), baseline=baseline, protected_native=guarded,
        temporary=cheap(TEMP), result=result, code=code, inspection=inspection_fp, reviewed_scope=scope_fp, events=journal.fingerprint(EVENTS),
        masked_source_record_digests={k: v.hexdigest() for k, v in masked.items()},
        catalog=journal.fingerprint(CATALOG), papers=baseline["current_paper_identities"], terms=baseline["current_entity_terms"],
        census_build=journal.fingerprint(OUTPUT / "CENSUS_BUILD.json"), database=db_fp,
        audit=journal.fingerprint(OUTPUT / "PAPER_IDENTITY_AUDIT.json"), paper_issues=journal.fingerprint(OUTPUT / "CURRENT_PAPER_ISSUES.jsonl"),
        tests=journal.fingerprint(OUTPUT / "TEST_RESULTS.xml"), test_counts=dict(test_counts),
        metadata_coverage_unchanged_by_exact_delta=True, source_full_sha_verified=True, graph_backups=0, record_preimages_saved=False)
    journal.atomic_json(OUTPUT / "BUILD_STATE.json", state)
    progress("BUILT_NOT_ADOPTED", changed_titles=len(events), identity=audit["claim_status"], relations=result["catalog_stats"])


def validate():
    require(not (OUTPUT / "VALIDATED.json").exists(), "validation exists; inspect/resume")
    state = journal.read_json(OUTPUT / "BUILD_STATE.json"); baseline = state["baseline"]
    require(bindings() == state["code"] and cheap(TEMP) == state["temporary"], "build/code changed")
    guard_native(state["protected_native"])
    for key in ("inspection", "reviewed_scope", "events", "catalog", "papers", "terms", "census_build", "audit", "paper_issues", "tests"): small_check(state[key])
    papers, (articles, witnesses) = authorities(baseline)
    terms = VerifiedEntityTerms(journal.read_json(state["terms"]["path"]))
    proof = SourceProof(terms, papers)
    events = {r["claim_id"]: r for r in rows(EVENTS)}; require(len(events) == 163, "duplicate/incomplete repair plan")
    coverage, seen, matched, outer = Coverage(), set(), 0, []
    masked = {k: hashlib.sha256() for k in ("nodes", "edges")}
    db = sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True)
    paper_sigs, shared_sigs = set(), set()
    observed_issues = []

    def independent(records):
        nonlocal matched
        for kind, key, row in records:
            if kind != "metadata":
                coverage.add("node/" + node_class(key, row) if kind == "node" else "edge/all", row)
                value = row
                if kind == "node" and key in events:
                    verify_title_result(row, events[key], papers, articles, witnesses)
                    value = without_selected_title(row); seen.add(key)
                masked[kind + "s"].update(compact(value).encode() + b"\n")
            if kind == "node" and key.startswith("CLM:"):
                md = row["metadata"]; paper = bibliography(md.get("source_paper")); sig = digest(paper)
                member = evidence_member(md); rid = relation_id(terms.relation_key(md))
                record = db.execute("SELECT node_sha,paper_sig,legacy_key,relation_id,shared FROM claims WHERE cid=?", (key,)).fetchone()
                require(record is not None and record[:4] == (digest(row), sig, member["paper_key"], rid), "current census claim differs")
                if sig not in paper_sigs:
                    ids = identifiers(paper)
                    require(db.execute("SELECT payload,pmid,doi,pmcid,title,year FROM papers WHERE sig=?", (sig,)).fetchone() ==
                        (compact(paper), ids["pmid"], ids["doi"], ids["pmcid"], title_key(paper.get("title")), str(paper.get("year") or paper.get("publication_year") or "")),
                        "current census bibliography differs")
                    paper_sigs.add(sig)
                if record[4]:
                    require(db.execute("SELECT member_json FROM shared_members WHERE cid=?", (key,)).fetchone() == (compact(member),), "census shared member differs")
                    shared_sigs.add(key)
                for holder in (md, md.get("metadata") or {}):
                    if holder.get("pmid") and str(holder["pmid"]) != str(paper.get("pmid") or ""):
                        outer.append(dict(claim_id=key, source_paper_pmid=paper.get("pmid"), alternate_pmid=holder["pmid"], claim_sha256=digest(row)))
                answer = papers.resolve(md)
                if answer["status"] == "conflict": observed_issues.append(dict(claim_id=key, claim_sha256=digest(row), reasons=answer["reasons"],
                    legacy_source_key=member["paper_key"], action="hold_no_claim_or_source_deletion"))
                matched += 1
            yield kind, key, row

    progress("INDEPENDENT_CANDIDATE_SCIENCE_CENSUS_COVERAGE_SCAN")
    try:
        with hashed_reader(TEMP) as (reader, graph_sha):
            checks = verify_catalog_and_graph(proof.records(independent(walk_graph(reader))), state["result"], CATALOG, progress,
                relation_key_func=terms.relation_key, papers=papers)
            candidate_sha = graph_sha.hexdigest()
        census = journal.read_json(state["census_build"]["path"])
        require(matched == db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 905184, "census claim closure differs")
        require(len(paper_sigs) == db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], "orphan/stale census paper")
        require(len(shared_sigs) == db.execute("SELECT COUNT(*) FROM shared_members").fetchone()[0] == 5861, "census shared closure differs")
        actual_shared = {m["claim_id"] for g in rows(CATALOG) for m in g["members"]}
        require(shared_sigs == actual_shared and outer == census["outer_pmid_conflicts"], "census shared flag/outer PMID proof differs")
        require(census_identifier_projection(db) == census["projection"], "current census normalization differs")
        require(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "current census integrity failed")
    finally: db.close()
    require(seen == set(events) and {k: v.hexdigest() for k, v in masked.items()} == state["masked_source_record_digests"], "not an exact title-only source transform")
    require(observed_issues == rows(state["paper_issues"]["path"]), "current paper issue census differs")
    audit = journal.read_json(state["audit"]["path"]); proof.verify(state["result"], audit)
    require(coverage.rows() == rows(baseline["current_coverage"]["path"]), "full metadata coverage differs")
    progress("FULL_DETAIL_AND_CENSUS_SHA_BOUNDARY")
    require(sha256(Path(baseline["current_detail_store"]["path"])) == baseline["current_detail_store"]["sha256"], "detail SHA differs")
    require(sha256(DATABASE) == state["database"]["sha256"], "candidate census SHA differs")
    guard_native(state["protected_native"]); require(bindings() == state["code"], "code advanced during validation")
    checks.pop("records_match_approved_identity_only_transform", None)
    checks.pop("all_nonidentity_claim_fields_preserved", None)
    checks.update(exact_title_only_source_transform=True, all_other_node_and_edge_fields_preserved=True,
        original_quotes_negation_conditions_independent_sources_preserved=True, title_content_witnesses_independently_reproduced=True,
        current_census_all_claims_papers_and_shared_members_verified=True, metadata_coverage_full_scan_unchanged=True,
        verified_identity_proofs_complete=True, paper_identity_witnesses_validated=True, publication_status_witnesses_validated=True,
        complete_title_and_identifier_conflicts_held=True, no_claim_or_edge_metadata_fields_added=True,
        authority_verified_claims=proof.claim_status["verified"], verified_paper_key_changes=proof.key_changes,
        distinct_articles_not_independent_cohorts=True)
    journal.atomic_json(OUTPUT / "VALIDATED.json", dict(status="VALIDATED_NOT_ADOPTED", at=journal.utc_now(),
        graph={**cheap(TEMP), "sha256": candidate_sha}, temporary_native=native_info(TEMP), database_native=native_info(DATABASE),
        checks=checks, build_state=journal.fingerprint(OUTPUT / "BUILD_STATE.json"), metadata_coverage_rows=len(coverage.rows())))
    progress("VALIDATED_NOT_ADOPTED", repaired_titles=len(events), claims=dict(proof.claim_status))


def apply():
    require(not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "already adopted")
    state = journal.read_json(OUTPUT / "BUILD_STATE.json"); accepted = journal.read_json(OUTPUT / "VALIDATED.json")
    small_check(accepted["build_state"]); require(bindings() == state["code"], "frozen code changed")
    guard_native([*state["protected_native"], accepted["temporary_native"], accepted["database_native"]])
    for key in ("reviewed_scope", "events", "catalog", "papers", "terms", "census_build", "audit", "paper_issues", "tests"): small_check(state[key])
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json"); baseline = state["baseline"]
    require(campaign["current_graph"] == baseline["current_graph"] and campaign["active_process"]["kind"] == "bibliography_titles", "source advanced")
    require(TEMP.resolve().parent == SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()), "unsafe replacement")
    changed = {r["claim_id"] for r in rows(EVENTS)}
    held_files = {}
    for field, filename in (("current_issues", "CURRENT_REMAINING_ISSUES.jsonl"), ("current_structure_holds", "CURRENT_STRUCTURE_HOLDS.jsonl")):
        small_check(baseline[field])
        held_files[filename] = rows(baseline[field]["path"])
        require(not changed & {r.get("claim_id") for r in held_files[filename]}, "scientific hold overlaps title change")
    os.replace(TEMP, SOURCE)
    graph = {**cheap(SOURCE), "sha256": accepted["graph"]["sha256"]}
    require(graph["bytes"] == accepted["graph"]["bytes"] and graph["mtime_ns"] == accepted["graph"]["mtime_ns"], "replacement differs")
    result = state["result"]; audit = journal.read_json(state["audit"]["path"])
    # None of the existing scientific issue records overlaps this title batch.
    for filename, held in held_files.items():
        write_rows(OUTPUT / filename, [dict(row, current_graph=graph, version_binding_status="FULL_CURRENT_GRAPH_VALIDATED") for row in held])
    census = journal.read_json(state["census_build"]["path"]); projection = census.pop("projection")
    journal.atomic_json(OUTPUT / "CENSUS.json", dict(**census, status="CURRENT_CENSUS_COMPLETE", at=journal.utc_now(), graph=graph,
        source_full_sha_verified=True, independent_full_candidate_verified=True, code=state["code"], record_preimages_saved=False))
    census_fp = journal.fingerprint(OUTPUT / "CENSUS.json")
    journal.atomic_json(OUTPUT / "CENSUS_NORMALIZATION.json", dict(**projection, census=census_fp))
    receipt = dict(status="CURRENT_BIBLIOGRAPHY_TITLES_APPLIED", at=journal.utc_now(), graph=graph, counts=result["counts"],
        changed=dict(claim_nodes=len(changed), edge_records=0, title_fields=len(changed), verified_paper_keys=audit["verified_key_changes"]),
        connected_components=result["connected_components"], isolated_nodes=result["isolated_nodes"], self_loops=result["self_loops"],
        metadata_key_unions=result["metadata_key_unions"], metadata=result["metadata"], source_graph=baseline["current_graph"], source_kg_retained=False,
        shared_relations=state["catalog"], entity_terms=state["terms"], paper_identities=state["papers"], paper_issues=state["paper_issues"],
        metadata_coverage=baseline["current_coverage"], current_paper_census=census_fp, relation_counts=result["catalog_stats"], checks=accepted["checks"],
        tests=state["tests"], test_counts=state["test_counts"], code=state["code"], detail_store=baseline["current_detail_store"],
        identity_audit=state["audit"], title_repairs=state["events"], rollback_retention=False, record_preimages_saved=False, formal_apply_performed=False)
    journal.atomic_json(OUTPUT / "CURRENT_ACCEPTANCE.json", receipt)
    journal.atomic_json(OUTPUT / "CURRENT_RUNTIME_ACCEPTANCE.json", dict(status="CURRENT_RUNTIME_VALIDATED", at=receipt["at"], graph=graph,
        acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), code=state["code"], tests=state["tests"], test_counts=state["test_counts"]))
    campaign.update(status="COMPLETED", phase="R35来源标题修复与当前书目普查重建已应用", active_process=None, updated_at=receipt["at"], current_graph=graph,
        current_acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), current_runtime_acceptance=journal.fingerprint(OUTPUT / "CURRENT_RUNTIME_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"), current_structure_holds=journal.fingerprint(OUTPUT / "CURRENT_STRUCTURE_HOLDS.jsonl"),
        current_shared_relations=state["catalog"], current_paper_identities=state["papers"], current_paper_issues=state["paper_issues"],
        current_paper_census=census_fp, current_census_normalization=journal.fingerprint(OUTPUT / "CENSUS_NORMALIZATION.json"),
        relation_evidence_counts=result["catalog_stats"], last_deep_verification=accepted["at"], last_current_graph_content_verification=dict(graph=graph, independent_full_scan=True),
        next_steps=["标题冲突剩33条、外层PMID冲突219条；按原始导入与完整文章核对，不按合成前缀猜编号。",
            "书目普查已重建在R35；R31历史普查不再当前，不能继续作为新数据的基线。",
            "继续完整复合表达、FANCE/VCP及21语义/1结构待核；无模型训练或正式同步。"])
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json")
    night.update(active_process=None, latest_acceptance=campaign["current_acceptance"], latest_census=census_fp,
        latest_completed_batch="R35", latest_completed_runtime_batch="R35", latest_runtime_acceptance=campaign["current_runtime_acceptance"])
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", dict(status="COMPLETED", at=receipt["at"], graph=graph, changed=receipt["changed"], relations=result["catalog_stats"]))
    print(compact(dict(status=receipt["status"], changed=receipt["changed"], relations=result["catalog_stats"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "validate", "apply", "run"))
    phase = parser.parse_args().phase
    for step in ("build", "validate", "apply") if phase == "run" else (phase,): globals()[step]()
