"""Optional single cold-file extension, only after the first ten finish below 30 GiB."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reclaim_kg_backup_storage as base

OUTPUT = base.REPO / "neurooracle/data/storage_maintenance/20260908_ntfs_cold_kg_extension_v1"
TARGET = (base.UMLS / "claim_compatibility_repair_v1_20260907/knowledge_graph.candidate.json").resolve()
ACCEPTANCE = TARGET.parent / "REPAIR_ACCEPTANCE.json"
EXPECTED_SHA = "2c7604973d12a32ef8b43a32915b07f76bddc16f43398e4f2c56eb237fec1807"


def eligible(info, protected_ids, previous_ids):
    base.require(info["path"] == str(TARGET) and TARGET.is_relative_to(base.REPO.resolve()), "not the single authorized cold candidate")
    base.require(info["hardlink_count"] == 1 and info["file_id"] not in protected_ids | previous_ids, "shared or already processed file")
    base.require(not info["attributes"] & (base.REPARSE | base.ENCRYPTED | base.COMPRESSED), "unexpected file storage attributes")
    base.require(info["bytes"] == 7685834727 and info["mtime_ns"] == 1788723798709644900, "cold candidate fingerprint differs")


def stage(phase, **fields):
    value = {"status": phase, "updated_at": base.journal.utc_now(), **fields}
    base.journal.atomic_json(OUTPUT / "RUN_STATE.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    base.require(not (OUTPUT / "COMPLETE.json").exists() and not (OUTPUT / "BEFORE.json").exists(), "extension already started; inspect before resuming")
    tests = {k: sum(int(s.get(k, 0)) for s in ElementTree.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"))
             for k in ("tests", "failures", "errors", "skipped")}
    base.require(tests["tests"] >= 8 and not any(tests[k] for k in ("failures", "errors", "skipped")), "extension tests not passed")
    plan = base.load_plan()
    summary = base.journal.read_json(base.OUTPUT / "SUMMARY.json")
    run_state = base.journal.read_json(base.OUTPUT / "RUN_STATE.json")
    base.require(summary["status"] == run_state["status"] == "PARTIAL_VERIFIED" and summary["files_verified"] == 10, "initial batch must be fully finished first")
    receipts = []
    for fp in summary["receipts"]:
        base.require(base.journal.fingerprint(fp["path"]) == fp, "initial receipt changed")
        r = base.journal.read_json(fp["path"])
        base.require(base.native_info(r["after"]["path"]) == r["after"], "initial compressed file changed")
        base.require(r["verified_before_sha256"] == r["verified_after_sha256"], "initial content mismatch")
        receipts.append(r)
    reclaimed = sum(r["reclaimed_bytes"] for r in receipts)
    base.require(reclaimed == summary["reclaimed_bytes"] < base.GOAL, "extension not necessary or total differs")
    base.require([base.native_info(r["path"]) for r in plan["protected"]] == plan["protected"], "protected files changed")
    acceptance_fp = base.journal.fingerprint(ACCEPTANCE)
    expected = base.journal.read_json(ACCEPTANCE)["artifacts"]["knowledge_graph.candidate.json"]
    base.require(Path(expected["path"]).resolve() == TARGET and expected["sha256"] == EXPECTED_SHA, "frozen target binding differs")
    before = base.native_info(TARGET)
    eligible(before, {r["file_id"] for r in plan["protected"]}, {r["after"]["file_id"] for r in receipts})
    base.require(shutil.disk_usage(base.REPO).free >= 16 * base.GIB, "reserve not met")
    stage("HASHING_BEFORE", previous_reclaimed_GiB=reclaimed / base.GIB, path=str(TARGET))
    digest = base.sha256(TARGET)
    base.require(digest == EXPECTED_SHA, "full before SHA mismatch")
    base.stable_identity(before, base.native_info(TARGET))
    boundary = {"before": before, "verified_before_sha256": digest, "acceptance": acceptance_fp,
                "initial_summary": base.journal.fingerprint(base.OUTPUT / "SUMMARY.json"),
                "initial_plan": base.journal.fingerprint(base.OUTPUT / "PLAN.json"), "previous_reclaimed_bytes": reclaimed,
                "tests": tests, "test_results": base.journal.fingerprint(OUTPUT / "TEST_RESULTS.xml"),
                "implementation": [base.journal.fingerprint(Path(__file__)), base.journal.fingerprint(Path(base.__file__)),
                                   base.journal.fingerprint(base.REPO / "neurooracle/tests/test_kg_storage_extension.py")],
                "created_at": base.journal.utc_now()}
    base.journal.atomic_json(OUTPUT / "BEFORE.json", boundary)
    stage("COMPRESSING_SINGLE_EXTRA_COLD_CANDIDATE", path=str(TARGET))
    process = subprocess.run(base.compression_command(TARGET), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base.journal.atomic_text(OUTPUT / "COMPACT_LOG.txt", process.stdout.decode("utf-8", errors="replace"))
    stage("HASHING_AFTER", compact_exit_code=process.returncode)
    after_digest, after = base.sha256(TARGET), base.native_info(TARGET)
    base.stable_identity(before, after)
    base.require(after_digest == digest and process.returncode == 0 and after["ntfs_compressed"], "extension verification failed")
    base.require([base.native_info(r["path"]) for r in plan["protected"]] == plan["protected"], "protected files changed")
    for fp in [acceptance_fp, boundary["initial_summary"], boundary["initial_plan"], boundary["test_results"], *boundary["implementation"]]:
        base.require(base.journal.fingerprint(fp["path"]) == fp, "small boundary artifact changed")
    extra = before["allocated_bytes"] - after["allocated_bytes"]
    base.require(extra >= 0, "allocation increased")
    result = {**boundary, "status": "TARGET_REACHED" if reclaimed + extra >= base.GOAL else "PARTIAL_VERIFIED",
              "completed_at": base.journal.utc_now(), "after": after, "verified_after_sha256": after_digest,
              "extra_reclaimed_bytes": extra, "total_reclaimed_bytes": reclaimed + extra,
              "total_reclaimed_GiB": (reclaimed + extra) / base.GIB, "total_files_verified": 11,
              "protected_files_unchanged": True, "files_deleted": 0, "graph_record_changes": 0,
              "free_bytes_now": shutil.disk_usage(base.REPO).free,
              "rollback_command": [str(base.COMPACT), "/U", "/Q", str(TARGET)],
              "rollback_requires_free_bytes_at_least": extra + 16 * base.GIB}
    base.journal.atomic_json(OUTPUT / "COMPLETE.json", result)
    stage(result["status"], total_reclaimed_GiB=result["total_reclaimed_GiB"], total_files_verified=11)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        stage("FAILED_STOPPED_NO_DELETION", error=str(exc))
        raise
