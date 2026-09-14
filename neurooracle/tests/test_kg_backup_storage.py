from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reclaim_kg_backup_storage as storage


def item():
    return {"path": sorted(storage.allowed_paths())[0], "bytes": 1024, "mtime_ns": 12345,
            "allocated_bytes": 4096, "attributes": 32, "file_id": "one", "hardlink_count": 1, "ntfs_compressed": False}


@pytest.mark.parametrize("key", ["path", "bytes", "mtime_ns", "file_id", "hardlink_count"])
def test_stable_identity_fails_on_changes(key):
    before, after = item(), item()
    after[key] = "changed"
    with pytest.raises(ValueError): storage.stable_identity(before, after)


def test_only_storage_attributes_may_change():
    before, after = item(), item()
    after.update(allocated_bytes=1024, attributes=32 | storage.COMPRESSED, ntfs_compressed=True)
    storage.stable_identity(before, after)


@pytest.mark.parametrize("change", [{"path": str(storage.REPO)}, {"path": "C:/Windows/system.ini"},
                                    {"hardlink_count": 2}, {"attributes": storage.REPARSE},
                                    {"attributes": storage.ENCRYPTED}, {"bytes": 0}, {"bytes": 30 * storage.GIB}])
def test_unsafe_target_rejected(change):
    row = item(); row.update(change)
    with pytest.raises(ValueError): storage.eligible(row, set())


def test_shared_current_file_rejected():
    with pytest.raises(ValueError): storage.eligible(item(), {"one"})


def test_literal_command_no_broad_operation():
    path = Path("C:/example/one file.json")
    command = storage.compression_command(path)
    assert command == [str(storage.COMPACT), "/C", "/F", "/Q", str(path)]
    assert not any(x in command for x in ("/S", "/I", "/EXE", "/CompactOs"))


def test_exact_allowlist_no_current_or_formal_targets():
    values = storage.allowed_paths()
    assert len(values) == 10
    assert not any("full_v2\\" in p or "round12_conservative_candidate" in p or "remaining_endpoint_repair" in p for p in values)


def test_nested_frozen_binding_extraction():
    fp = {"path": "x", "bytes": 10, "sha256": "abc"}
    assert list(storage.fingerprints({"list": [fp], "other": {"bytes": 11}})) == [fp]


@pytest.mark.skipif(os.name != "nt", reason="Requires Windows NTFS compression via compact.exe")
def test_ntfs_transparent_compression_preserves_identity_and_hash(tmp_path):
    path = tmp_path / "synthetic_cold_graph.json"
    path.write_bytes(b'{"kind":"synthetic fixture","metadata":{"note":"repeated text"}}\n' * 32768)
    before = storage.native_info(path)
    before_hash = storage.sha256(path)
    result = subprocess.run(storage.compression_command(path), capture_output=True)
    assert result.returncode == 0, result.stdout
    after = storage.native_info(path)
    storage.stable_identity(before, after)
    assert after["ntfs_compressed"] and after["allocated_bytes"] < before["allocated_bytes"]
    assert storage.sha256(path) == before_hash
    undo = subprocess.run([str(storage.COMPACT), "/U", "/Q", str(path)], capture_output=True)
    assert undo.returncode == 0
    restored = storage.native_info(path)
    storage.stable_identity(before, restored)
    assert not restored["ntfs_compressed"] and storage.sha256(path) == before_hash
