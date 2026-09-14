"""Reclaim allocated NTFS space without deleting or rewriting KG records.

Only explicit, immutable cold files are eligible. Every real target is checked
against its frozen SHA before compression and fully rehashed afterwards.
No recursive compact flags, directory compression, hardlinks, graph copies,
source edits, model calls, D-drive writes, or automatic graph promotion.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal

REPO = journal.REPO
OUTPUT = REPO / "neurooracle/data/storage_maintenance/20260908_ntfs_cold_kg_v1"
ARCHIVE = REPO / "neurooracle/data/archive/kg_mutation_backups"
UMLS = journal.DATA
GIB = 1024**3
GOAL = 30 * GIB
COMPRESSED, REPARSE, ENCRYPTED = 0x800, 0x400, 0x4000
COMPACT = Path("C:/Windows/System32/compact.exe")
BACKUPS = (
    ("kg_v3_mixed_122k_full_injection_20260825_20260825_091939", "backup_manifest.json", ("knowledge_graph.json", "extracted_claims.jsonl")),
    ("brain_age_shadow_candidate_20260903_071738", "BACKUP_MANIFEST.json", ("knowledge_graph.json", "extracted_claims.jsonl")),
    ("infrastructure_taxonomy_v1_20260903_092208", "BACKUP_MANIFEST.json", ("knowledge_graph.json",)),
    ("spatial_mapping_schema_v1_20260903_183443", "BACKUP_MANIFEST.json", ("knowledge_graph.json",)),
    ("spatial_mapping_data_v1_20260903_213249", "BACKUP_MANIFEST.json", ("knowledge_graph.json",)),
    ("umls_2026AA_atomic_mentions_v1_20260906_102710", "BACKUP_MANIFEST.json", ("knowledge_graph.json",)),
)
FALLBACKS = (
    ("simplification_candidate_v1_20260906", "CANDIDATE_FREEZE.json"),
    ("simplification_candidate_v2_node_repairs_20260906", "REPAIR_FREEZE.json"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


class FileTime(ct.Structure):
    _fields_ = [("low", wt.DWORD), ("high", wt.DWORD)]


class HandleInfo(ct.Structure):
    _fields_ = [("attributes", wt.DWORD), ("creation", FileTime), ("access", FileTime), ("write", FileTime),
                ("volume", wt.DWORD), ("size_high", wt.DWORD), ("size_low", wt.DWORD),
                ("links", wt.DWORD), ("index_high", wt.DWORD), ("index_low", wt.DWORD)]


class StandardInfo(ct.Structure):
    _fields_ = [("allocated", ct.c_longlong), ("end", ct.c_longlong), ("links", wt.DWORD),
                ("delete_pending", ct.c_ubyte), ("directory", ct.c_ubyte)]


def native_info(path):
    require(os.name == "nt", "NTFS maintenance requires Windows")
    path = Path(path).resolve(strict=True)
    k = ct.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.c_void_p, wt.DWORD, wt.DWORD, ct.c_void_p]
    k.CreateFileW.restype = wt.HANDLE
    k.GetFileInformationByHandle.argtypes = [wt.HANDLE, ct.POINTER(HandleInfo)]
    k.GetFileInformationByHandle.restype = wt.BOOL
    k.GetFileInformationByHandleEx.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
    k.GetFileInformationByHandleEx.restype = wt.BOOL
    k.GetCompressedFileSizeW.argtypes = [wt.LPCWSTR, ct.POINTER(wt.DWORD)]
    k.GetCompressedFileSizeW.restype = wt.DWORD
    k.CloseHandle.argtypes = [wt.HANDLE]
    native_path = "\\\\?\\" + str(path)
    handle = k.CreateFileW(native_path, 0, 7, None, 3, 0, None)
    if handle == ct.c_void_p(-1).value:
        raise ct.WinError(ct.get_last_error())
    try:
        info, standard = HandleInfo(), StandardInfo()
        if not k.GetFileInformationByHandle(handle, ct.byref(info)):
            raise ct.WinError(ct.get_last_error())
        if not k.GetFileInformationByHandleEx(handle, 1, ct.byref(standard), ct.sizeof(standard)):
            raise ct.WinError(ct.get_last_error())
        allocated = standard.allocated
        if info.attributes & COMPRESSED:
            high = wt.DWORD()
            ct.set_last_error(0)
            low = k.GetCompressedFileSizeW(native_path, ct.byref(high))
            if low == 0xFFFFFFFF and ct.get_last_error():
                raise ct.WinError(ct.get_last_error())
            allocated = (high.value << 32) | low
        write_ticks = (info.write.high << 32) | info.write.low
        return {"path": str(path), "bytes": (info.size_high << 32) | info.size_low,
                "mtime_ns": (write_ticks - 116444736000000000) * 100,
                "allocated_bytes": allocated, "attributes": info.attributes,
                "file_id": f"{info.volume:08x}:{info.index_high:08x}{info.index_low:08x}",
                "hardlink_count": info.links, "ntfs_compressed": bool(info.attributes & COMPRESSED)}
    finally:
        k.CloseHandle(handle)


def stable_identity(before, after):
    for key in ("path", "bytes", "mtime_ns", "file_id", "hardlink_count"):
        require(before[key] == after[key], "file identity changed: " + key)


def allowed_paths():
    return {str((ARCHIVE / directory / name).resolve()) for directory, _, names in BACKUPS for name in names} | {
        str((UMLS / directory / "knowledge_graph.candidate.json").resolve()) for directory, _ in FALLBACKS}


def eligible(info, protected_ids):
    require(info["path"] in allowed_paths(), "target not in exact cold-file allowlist")
    require(Path(info["path"]).is_relative_to(REPO.resolve()), "target outside project")
    require(info["hardlink_count"] == 1 and info["file_id"] not in protected_ids, "file is shared with another path")
    require(not info["attributes"] & (REPARSE | ENCRYPTED), "reparse/encrypted target forbidden")
    require(0 < info["bytes"] < 30 * GIB, "unsupported target size")


def compression_command(path):
    # A single literal file argument: no directory, wildcard, /S, /EXE, /I or OS mode.
    return [str(COMPACT), "/C", "/F", "/Q", str(path)]


def fingerprints(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from fingerprints(child)
    elif isinstance(value, list):
        for child in value:
            yield from fingerprints(child)


def protected_info():
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["active_process"] is None and campaign["automation_id"] is None, "KG campaign active")
    accepted = journal.read_json(campaign["current_acceptance"]["path"])
    journal.guards(campaign)
    paths = [fp["path"] for fp in campaign["formal_sources"].values()]
    paths += [fp["path"] for name, fp in accepted["artifacts"].items() if name in (
        "knowledge_graph.candidate.json", "umls_details.sqlite", "NODE_REPAIRS.jsonl")]
    paths += [campaign["current_acceptance"]["path"]]
    require(len(paths) == 8, "protected file inventory differs")
    return [native_info(p) for p in paths]


def tests_ok():
    root = ElementTree.parse(OUTPUT / "TEST_RESULTS.xml").getroot()
    counts = {k: sum(int(s.get(k, 0)) for s in root.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] >= 15 and not any(counts[k] for k in ("failures", "errors", "skipped")), "tests incomplete/failed")
    return counts


def prepare():
    require(not (OUTPUT / "PLAN.json").exists(), "plan already exists")
    tests = tests_ok()
    protected = protected_info()
    protected_ids = {r["file_id"] for r in protected}
    targets = []
    for directory, manifest_name, names in BACKUPS:
        manifest = ARCHIVE / directory / manifest_name
        saved = journal.read_json(manifest)
        for name in names:
            expected = saved["files"][name] if "files" in saved else saved["source_fingerprints"]["knowledge_graph" if name == "knowledge_graph.json" else "extracted_claims"]
            info = native_info(ARCHIVE / directory / name)
            eligible(info, protected_ids)
            require(info["bytes"] == expected["bytes"], "backup size differs from original manifest")
            if "mtime_ns" in expected:
                require(info["mtime_ns"] == expected["mtime_ns"], "backup timestamp differs")
            targets.append({"before": info, "expected_sha256": expected["sha256"], "manifest": journal.fingerprint(manifest), "tier": "historical_backup"})
    for directory, manifest_name in FALLBACKS:
        manifest = UMLS / directory / manifest_name
        path = (UMLS / directory / "knowledge_graph.candidate.json").resolve()
        matches = [fp for fp in fingerprints(journal.read_json(manifest)) if Path(fp["path"]).resolve() == path]
        unique = {(fp["bytes"], fp["sha256"]) for fp in matches}
        require(len(unique) == 1, "old candidate frozen binding missing/ambiguous")
        size, digest = unique.pop()
        info = native_info(path)
        eligible(info, protected_ids)
        require(info["bytes"] == size, "old candidate size differs")
        targets.append({"before": info, "expected_sha256": digest, "manifest": journal.fingerprint(manifest), "tier": "old_candidate_only_if_needed"})
    require(len(targets) == 10 and len({r["before"]["file_id"] for r in targets}) == 10, "target identity scope differs")
    plan = {"schema": "kg.ntfs_cold_storage.v1", "created_at": journal.utc_now(), "target_reclaim_bytes": GOAL,
            "targets": targets, "protected": protected, "tests": tests, "test_results": journal.fingerprint(OUTPUT / "TEST_RESULTS.xml"),
            "implementation": [journal.fingerprint(Path(__file__)), journal.fingerprint(REPO / "neurooracle/tests/test_kg_backup_storage.py")],
            "free_bytes_before": shutil.disk_usage(REPO).free, "files_deleted": 0, "graph_record_changes": 0,
            "documentation": ["https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact",
                              "https://learn.microsoft.com/en-us/windows/win32/fileio/file-compression-and-decompression"]}
    journal.atomic_json(OUTPUT / "PLAN.json", plan)
    journal.atomic_json(OUTPUT / "PLAN_BINDING.json", journal.fingerprint(OUTPUT / "PLAN.json"))
    print(json.dumps({"status": "PLAN_READY", "targets": len(targets), "protected": len(protected), "target_reclaim_GiB": GOAL / GIB}), flush=True)


def load_plan():
    binding = journal.read_json(OUTPUT / "PLAN_BINDING.json")
    require(journal.fingerprint(OUTPUT / "PLAN.json") == binding, "plan binding changed")
    plan = journal.read_json(OUTPUT / "PLAN.json")
    for fp in [*plan["implementation"], plan["test_results"], *(r["manifest"] for r in plan["targets"])]:
        require(journal.fingerprint(fp["path"]) == fp, "implementation/test/source manifest changed")
    return plan


def status(phase, **data):
    value = {"status": phase, "updated_at": journal.utc_now(), **data}
    journal.atomic_json(OUTPUT / "RUN_STATE.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(max_files):
    plan = load_plan()
    receipts = []
    for path in sorted(OUTPUT.glob("FILE_*_COMPLETE.json")):
        receipt = journal.read_json(path)
        require(receipt["plan_sha256"] == journal.read_json(OUTPUT / "PLAN_BINDING.json")["sha256"], "receipt belongs to another plan")
        require(native_info(receipt["after"]["path"]) == receipt["after"], "completed file fingerprint changed")
        receipts.append(receipt)
    done = {r["after"]["path"] for r in receipts}
    reclaimed = sum(r["reclaimed_bytes"] for r in receipts)
    completed_now = 0
    for index, target in enumerate(plan["targets"], 1):
        path = Path(target["before"]["path"])
        if str(path) in done:
            continue
        if reclaimed >= plan["target_reclaim_bytes"] or completed_now >= max_files:
            break
        for original in plan["protected"]:
            require(native_info(original["path"]) == original, "protected file changed")
        current = native_info(path)
        stable_identity(target["before"], current)
        eligible(current, {r["file_id"] for r in plan["protected"]})
        require(shutil.disk_usage(REPO).free >= 16 * GIB, "free space below reserve")
        status("HASHING_BEFORE", target_index=index, path=str(path), reclaimed_GiB=reclaimed / GIB)
        before_sha = sha256(path)
        require(before_sha == target["expected_sha256"], "source full SHA differs: " + str(path))
        stable_identity(target["before"], native_info(path))
        before_receipt = {"plan_sha256": journal.read_json(OUTPUT / "PLAN_BINDING.json")["sha256"], "verified_before_sha256": before_sha,
                          "original_before": target["before"], "observed_before": current, "verified_at": journal.utc_now()}
        journal.atomic_json(OUTPUT / f"FILE_{index:02d}_BEFORE.json", before_receipt)
        status("COMPRESSING_ONE_LITERAL_FILE", target_index=index, path=str(path), reclaimed_GiB=reclaimed / GIB)
        process = subprocess.run(compression_command(path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        journal.atomic_text(OUTPUT / f"FILE_{index:02d}_COMPACT_LOG.txt", process.stdout.decode("utf-8", errors="replace"))
        status("HASHING_AFTER", target_index=index, path=str(path), compact_exit_code=process.returncode)
        after_sha = sha256(path)
        after = native_info(path)
        stable_identity(target["before"], after)
        require(after_sha == before_sha, "post-compression SHA mismatch")
        require(process.returncode == 0 and after["ntfs_compressed"], "compression failed/incomplete")
        require(after["allocated_bytes"] <= target["before"]["allocated_bytes"], "compression increased allocation")
        receipt = {**before_receipt, "completed_at": journal.utc_now(), "after": after, "verified_after_sha256": after_sha,
                   "compact_exit_code": process.returncode, "reclaimed_bytes": target["before"]["allocated_bytes"] - after["allocated_bytes"],
                   "content_size_mtime_file_id_preserved": True, "file_deleted": False,
                   "rollback": {"command": [str(COMPACT), "/U", "/Q", str(path)],
                                "requires_free_bytes_at_least": target["before"]["allocated_bytes"] - after["allocated_bytes"] + 16 * GIB}}
        journal.atomic_json(OUTPUT / f"FILE_{index:02d}_COMPLETE.json", receipt)
        receipts.append(receipt)
        reclaimed += receipt["reclaimed_bytes"]
        completed_now += 1
        status("FILE_VERIFIED", target_index=index, files_verified=len(receipts), reclaimed_GiB=reclaimed / GIB)
    protected_after = [native_info(r["path"]) for r in plan["protected"]]
    require(protected_after == plan["protected"], "protected files changed at boundary")
    value = {"status": "TARGET_REACHED" if reclaimed >= plan["target_reclaim_bytes"] else "PARTIAL_VERIFIED",
             "updated_at": journal.utc_now(), "files_verified": len(receipts), "reclaimed_bytes": reclaimed,
             "reclaimed_GiB": reclaimed / GIB, "free_bytes_now": shutil.disk_usage(REPO).free,
             "protected_files_unchanged": True, "files_deleted": 0, "graph_record_changes": 0,
             "models_called": 0, "training_jobs_started": 0, "d_drive_writes": 0,
             "plan": journal.fingerprint(OUTPUT / "PLAN.json"),
             "receipts": [journal.fingerprint(p) for p in sorted(OUTPUT.glob("FILE_*_COMPLETE.json"))]}
    journal.atomic_json(OUTPUT / "SUMMARY.json", value)
    status(value["status"], files_verified=len(receipts), reclaimed_GiB=reclaimed / GIB)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--max-files", type=int, default=10)
    args = parser.parse_args()
    try:
        require(args.max_files > 0, "positive max-files required")
        {"prepare": prepare, "run": lambda: run(args.max_files)}[args.phase]()
    except Exception as exc:
        status("FAILED_STOPPED_NO_DELETION", error=type(exc).__name__ + ": " + str(exc))
        raise
