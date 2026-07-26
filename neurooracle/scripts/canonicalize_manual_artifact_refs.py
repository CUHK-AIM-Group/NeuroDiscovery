"""Canonicalize legacy MANUAL claim references in JSON experiment artifacts.

Only fields that carry claim foreign keys are rewritten. Human-readable text and
legacy provenance fields are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

if __package__:
    from neurooracle.scripts.canonicalize_manual_claim_ids import (
        atomic_swap_with_rollback,
        canonical_claim_id,
        is_legacy_manual_claim_id,
    )
else:
    from canonicalize_manual_claim_ids import (  # type: ignore[no-redef]
        atomic_swap_with_rollback,
        canonical_claim_id,
        is_legacy_manual_claim_id,
    )


MIGRATION_NAME = "canonicalize_manual_artifact_refs_20260722"
CLAIM_REFERENCE_KEYS = frozenset(
    {
        "claim_id",
        "claim_ids",
        "supporting_claim",
        "supporting_claims",
        "evidence_claim_id",
        "evidence_claim_ids",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def migrate_claim_references(value: Any) -> tuple[Any, dict[str, str], int]:
    mapping: dict[str, str] = {}
    changed = 0

    def walk(item: Any, parent_key: str = "") -> Any:
        nonlocal changed
        if isinstance(item, str):
            if parent_key in CLAIM_REFERENCE_KEYS and is_legacy_manual_claim_id(item):
                replacement = canonical_claim_id(item)
                mapping[item] = replacement
                changed += 1
                return replacement
            return item
        if isinstance(item, list):
            return [walk(child, parent_key) for child in item]
        if isinstance(item, dict):
            return {key: walk(child, key) for key, child in item.items()}
        return item

    return walk(value), mapping, changed


def find_legacy_claim_references(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def walk(item: Any, parent_key: str = "", path: str = "$") -> None:
        if isinstance(item, str):
            if parent_key in CLAIM_REFERENCE_KEYS and is_legacy_manual_claim_id(item):
                found.append((path, item))
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, parent_key, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                walk(child, key, f"{path}.{key}")

    walk(value)
    return found


def main() -> None:
    args = parse_args()
    artifacts = list(dict.fromkeys(path.resolve() for path in args.artifacts))
    staged: list[tuple[Path, Path, Path]] = []
    reports: list[dict[str, Any]] = []

    for artifact in artifacts:
        data = json.loads(artifact.read_text(encoding="utf-8"))
        migrated, mapping, changed = migrate_claim_references(data)
        remaining = find_legacy_claim_references(migrated)
        if remaining:
            raise ValueError(f"Legacy claim references remain in {artifact}: {remaining[:5]}")
        reports.append(
            {
                "artifact": str(artifact),
                "references_changed": changed,
                "unique_claim_ids_changed": len(mapping),
                "mapping": dict(sorted(mapping.items())),
                "legacy_claim_references_remaining": 0,
            }
        )
        if args.apply and changed:
            temp = artifact.with_suffix(artifact.suffix + f".{MIGRATION_NAME}.tmp")
            backup = artifact.with_suffix(artifact.suffix + f".{MIGRATION_NAME}.bak")
            if temp.exists():
                temp.unlink()
            if backup.exists():
                raise ValueError(f"Rollback file already exists: {backup}")
            temp.write_text(
                json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staged.append((temp, artifact, backup))

    swapped: list[tuple[Path, Path]] = []
    try:
        for temp, artifact, backup in staged:
            atomic_swap_with_rollback(temp, artifact, backup)
            swapped.append((artifact, backup))
    except Exception:
        for artifact, backup in reversed(swapped):
            if artifact.exists():
                artifact.unlink()
            os.replace(backup, artifact)
        raise

    report = {
        "migration": MIGRATION_NAME,
        "mode": "apply" if args.apply else "dry_run",
        "artifacts_scanned": len(artifacts),
        "artifacts_changed": len(staged) if args.apply else sum(
            1 for item in reports if item["references_changed"]
        ),
        "references_changed": sum(item["references_changed"] for item in reports),
        "unique_claim_ids_changed": len(
            {
                old_id
                for item in reports
                for old_id in item["mapping"]
            }
        ),
        "artifacts": reports,
        "rollback_files": [str(backup) for _temp, _artifact, backup in staged],
    }
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
