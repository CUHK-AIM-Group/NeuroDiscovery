from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extend_kg_storage_reclaim as extension


def fixture():
    return {"path": str(extension.TARGET), "bytes": 7685834727, "mtime_ns": 1788723798709644900,
            "file_id": "cold-extra", "hardlink_count": 1, "attributes": 32}


def test_exact_extra_candidate_allowed():
    extension.eligible(fixture(), set(), set())


@pytest.mark.parametrize("change", [{"path": str(extension.base.REPO)}, {"bytes": 1}, {"mtime_ns": 1},
                                    {"hardlink_count": 2}, {"attributes": extension.base.REPARSE},
                                    {"attributes": extension.base.ENCRYPTED}, {"attributes": extension.base.COMPRESSED}])
def test_wrong_extra_target_rejected(change):
    row = fixture(); row.update(change)
    with pytest.raises(ValueError): extension.eligible(row, set(), set())


@pytest.mark.parametrize("protected,previous", [({"cold-extra"}, set()), (set(), {"cold-extra"})])
def test_reject_shared_or_already_counted_file(protected, previous):
    with pytest.raises(ValueError): extension.eligible(fixture(), protected, previous)
