from __future__ import annotations

import json
from pathlib import Path

from core.scripts.case1_codex_session_mixed_prefetch import _schema_pending_paths


def test_schema_selection_uses_full_created_order_before_offset(tmp_path: Path) -> None:
    pending_dir = tmp_path / "codex_session_shadow" / "pending"
    pending_dir.mkdir(parents=True)
    for index, created in enumerate((3.0, 1.0, 4.0, 2.0)):
        value = {
            "created_at": created,
            "request_sha256": f"sha-{index}",
            "request": {"json_schema": {"name": "hypothesis_evolution"}},
        }
        (pending_dir / f"sha-{index}.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
    selected = _schema_pending_paths(
        trial_dir=tmp_path,
        schema_name="hypothesis_evolution",
        offset=2,
        limit=2,
    )
    assert [path.stem for path in selected] == ["sha-0", "sha-2"]
