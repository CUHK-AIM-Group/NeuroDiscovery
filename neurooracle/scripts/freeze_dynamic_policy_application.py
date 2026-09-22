"""Freeze an unchanged dynamic policy for application to a newer KG release."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.src.experiment_source_bundle import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _file_record(path: Path, root: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        rendered = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        rendered = str(path)
    return {"path": rendered, "sha256": sha256_file(path)}


def _verify_record(record: Mapping[str, Any], root: Path) -> str:
    path = _resolve(str(record.get("path") or ""), root)
    expected = str(record.get("sha256") or "").upper()
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"canonical input hash mismatch: {path}")
    return expected


def _canonical_hashes_from_static(
    static: Mapping[str, Any],
    root: Path,
) -> dict[str, str]:
    canonical = static.get("canonical_release") or {}
    return {
        "knowledge_graph_sha256": _verify_record(canonical["knowledge_graph"], root),
        "claims_sha256": _verify_record(canonical["extracted_claims"], root),
        "state_sha256": _verify_record(canonical["current_state"], root),
    }


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def freeze_dynamic_policy_application(
    *,
    source_policy_path: Path,
    static_design_path: Path,
    output_path: Path,
    workspace_root: Path,
    additional_exposed_case_study_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind an already-selected policy to new KG bytes without retuning it."""

    workspace_root = workspace_root.resolve()
    source_policy_path = source_policy_path.resolve()
    static_design_path = static_design_path.resolve()
    output_path = output_path.resolve()

    source = json.loads(source_policy_path.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_for_confirmatory_application":
        raise ValueError("source dynamic policy is not frozen")
    selected_policy = dict(source.get("selected_policy") or {})
    if selected_policy.get("feedback_enabled") is not True:
        raise ValueError("source policy is not the selected closed-loop policy")

    static = json.loads(static_design_path.read_text(encoding="utf-8"))
    if static.get("status") != "frozen_before_formal_generation":
        raise ValueError("application static design is not frozen")
    application_release = _canonical_hashes_from_static(static, workspace_root)
    selection_release = {
        key: str((source.get("canonical_release") or {}).get(key) or "").upper()
        for key in (
            "knowledge_graph_sha256",
            "claims_sha256",
            "state_sha256",
        )
    }
    if not all(selection_release.values()):
        raise ValueError("source policy has incomplete selection-release hashes")

    original_ids = [
        str(value)
        for value in (
            (source.get("development_protocol") or {}).get("case_study_ids") or ()
        )
    ]
    if not original_ids:
        raise ValueError("source policy has no development Case Studies")
    additional_ids = [str(value) for value in additional_exposed_case_study_ids]
    if len(additional_ids) != len(set(additional_ids)):
        raise ValueError("additional exposed Case Studies must be unique")
    all_exposed_ids = list(dict.fromkeys([*original_ids, *additional_ids]))

    development = dict(source.get("development_protocol") or {})
    development["case_study_ids"] = all_exposed_ids
    development["original_policy_development_case_study_ids"] = original_ids
    development["additional_exposed_case_study_ids"] = additional_ids
    development["exclusion_basis"] = (
        "All Case Studies whose current-release outcomes were inspected before the "
        "formal cross-task generalization run are excluded."
    )

    application = {
        "schema_version": "neurodiscovery-frozen-dynamic-policy-application.v1",
        "registered_at": _utc_now(),
        "status": "frozen_for_confirmatory_application",
        "evidence_tier": "unchanged_cross_release_policy_application",
        "canonical_release": application_release,
        "development_protocol": development,
        "selected_policy": selected_policy,
        "policy_application": {
            "mode": "unchanged_cross_release_application",
            "source_policy": _file_record(source_policy_path, workspace_root),
            "source_policy_schema_version": source.get("schema_version"),
            "source_selected_policy_sha256": _canonical_json_sha256(selected_policy),
            "selection_canonical_release": selection_release,
            "application_canonical_release": application_release,
            "original_development_case_study_ids": original_ids,
            "additional_exposed_case_study_ids": additional_ids,
            "all_excluded_case_study_ids": all_exposed_ids,
            "frozen_runtime_unchanged": True,
            "new_release_outcomes_used_for_policy_selection": False,
            "post_update_retuning": False,
        },
    }

    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        application["registered_at"] = existing.get("registered_at")
        if existing != application:
            raise ValueError("existing dynamic policy application differs")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(application, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return application


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--static-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--additional-exposed-case-study", action="append", default=[])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = freeze_dynamic_policy_application(
        source_policy_path=args.source_policy,
        static_design_path=args.static_design,
        output_path=args.output,
        workspace_root=args.workspace_root,
        additional_exposed_case_study_ids=args.additional_exposed_case_study,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "excluded_case_study_ids": result["development_protocol"][
                    "case_study_ids"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
