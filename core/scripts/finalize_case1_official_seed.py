"""Commit one blinded Case Study 1 official-baseline seed before outcome reveal.

The official adapters already emit an exact-ID SearchPolicy.  This utility
validates that policy against the sealed public registry and commits every raw
generation artifact plus the selected fresh-session gateway audit trail.  It
does not load the exhaustive outcome table and never repairs or reruns a seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selected-gateway", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-kg-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-anchors", type=int, default=80)
    parser.add_argument("--control-note", default="")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def write_once_json(path: Path, payload: Any) -> None:
    encoded = canonical_json(payload).encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to overwrite non-identical artifact: {path}")
        return
    path.write_bytes(encoded)


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": (
            path.relative_to(root).as_posix() if root is not None else str(path)
        ),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def canonical_kg_sha(gateway: dict[str, Any]) -> str:
    release = gateway.get("canonical_release") or {}
    return str(
        (((release.get("files") or {}).get("knowledge_graph") or {}).get("sha256"))
        or ""
    ).lower()


def gateway_counts(path: Path) -> dict[str, int]:
    def count(name: str) -> int:
        directory = path / name
        return len(list(directory.glob("*.json"))) if directory.is_dir() else 0

    return {
        "requests": count("requests"),
        "responses": count("responses"),
        "receipts": count("receipts"),
        "rejected_attempts": count("rejected"),
    }


def validate_gateway(
    gateway_dir: Path,
    *,
    expected_task_sha256: str,
    expected_registry_sha256: str,
    expected_kg_sha256: str,
    expected_adapter_sha256: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    manifest_path = gateway_dir / "gateway_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_object(manifest_path)
    actual = {
        "task": str(manifest.get("task_sha256") or "").lower(),
        "registry": str(manifest.get("registry_sha256") or "").lower(),
        "kg": canonical_kg_sha(manifest),
        "adapter": str(manifest.get("sealed_adapter_sha256") or "").lower(),
    }
    expected = {
        "task": expected_task_sha256.lower(),
        "registry": expected_registry_sha256.lower(),
        "kg": expected_kg_sha256.lower(),
        "adapter": expected_adapter_sha256.lower(),
    }
    if actual != expected:
        raise RuntimeError(f"Selected gateway identity mismatch: {actual} != {expected}")
    if manifest.get("actual_model") != "gpt-5.6-luna":
        raise RuntimeError("Selected gateway actual model is not gpt-5.6-luna")
    if manifest.get("reasoning_effort") != "max":
        raise RuntimeError("Selected gateway reasoning effort is not max")
    if manifest.get("transport") != "fresh_codex_session_per_api_request":
        raise RuntimeError("Selected gateway did not use fresh sessions per request")
    if manifest.get("authorization_headers_persisted") is not False:
        raise RuntimeError("Selected gateway credential declaration is unsafe")
    counts = gateway_counts(gateway_dir)
    if counts["requests"] < 1:
        raise RuntimeError("Selected gateway has no requests")
    if not (
        counts["requests"] == counts["responses"] == counts["receipts"]
    ):
        raise RuntimeError(f"Selected gateway audit trail is incomplete: {counts}")
    return manifest, counts


def validate_policy(
    policy_path: Path,
    registry_path: Path,
    *,
    method: str,
    seed: int,
    expected_anchors: int,
) -> tuple[dict[str, Any], list[str]]:
    policy = load_object(policy_path)
    if policy.get("method") != method or int(policy.get("trial", -1)) != seed:
        raise RuntimeError("SearchPolicy method/trial does not match requested seed")
    anchors = policy.get("anchors")
    if not isinstance(anchors, list):
        raise RuntimeError("SearchPolicy anchors are not a list")
    candidate_ids = [
        str(anchor.get("candidate_id") or "")
        if isinstance(anchor, dict)
        else str(anchor)
        for anchor in anchors
    ]
    if not candidate_ids or len(candidate_ids) > expected_anchors:
        raise RuntimeError(
            f"SearchPolicy has {len(candidate_ids)} anchors; expected 1..{expected_anchors}"
        )
    if "" in candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("SearchPolicy contains blank or duplicate candidate IDs")
    registry_ids: set[str] = set()
    with registry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            registry_ids.add(str(row["candidate_id"]))
    invalid = sorted(set(candidate_ids) - registry_ids)
    if invalid:
        raise RuntimeError(
            f"SearchPolicy contains {len(invalid)} IDs outside the sealed registry"
        )
    return policy, candidate_ids


def bound_session_log_records(gateway_dir: Path) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for receipt_path in sorted((gateway_dir / "receipts").glob("*.json")):
        receipt = load_object(receipt_path)
        session_log = Path(str(receipt.get("session_log_path") or ""))
        if not session_log.is_file():
            raise FileNotFoundError(
                f"Bound Luna session log is missing for {receipt_path.name}: {session_log}"
            )
        record = file_record(session_log)
        record.update(
            {
                "thread_id": receipt.get("thread_id"),
                "turn_id": receipt.get("turn_id"),
                "response_content_sha256": receipt.get("response_content_sha256"),
            }
        )
        records[str(session_log)] = record
    return [records[key] for key in sorted(records)]


def verify_existing_commitment(path: Path, seed_dir: Path) -> dict[str, Any]:
    commitment = load_object(path)
    for record in commitment.get("generation_files") or []:
        target = seed_dir / str(record["path"])
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            raise RuntimeError(f"Committed generation artifact drifted: {target}")
    for record in commitment.get("bound_luna_session_logs") or []:
        target = Path(str(record["path"]))
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            raise RuntimeError(f"Committed Luna session log drifted: {target}")
    return commitment


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    seed_dir = (
        run_root
        / "baseline_generation"
        / "formal_official"
        / args.method
        / f"trial_{args.seed:02d}"
    )
    task_path = seed_dir / "task.json"
    policy_path = seed_dir / "search_policy.json"
    commitment_path = seed_dir / "generation_commitment.json"
    for required in (task_path, policy_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    if sha256_file(task_path) != args.expected_task_sha256.lower():
        raise RuntimeError("Frozen task hash mismatch")
    task = load_object(task_path)
    if task.get("method") != args.method or int(task.get("trial", -1)) != args.seed:
        raise RuntimeError("Frozen task method/trial mismatch")
    if task.get("model") != "deepseek-v4-pro" or task.get("reasoning_effort") != "high":
        raise RuntimeError("Frozen logical model contract is not deepseek-v4-pro/high")
    if int(task.get("n_anchors", -1)) != args.expected_anchors:
        raise RuntimeError("Frozen task anchor count mismatch")
    registry_path = Path(str(task["public_registry_path"]))
    if sha256_file(registry_path) != args.expected_registry_sha256.lower():
        raise RuntimeError("Sealed public registry hash mismatch")

    selected_gateway = seed_dir / args.selected_gateway
    manifest, selected_counts = validate_gateway(
        selected_gateway,
        expected_task_sha256=args.expected_task_sha256,
        expected_registry_sha256=args.expected_registry_sha256,
        expected_kg_sha256=args.expected_kg_sha256,
        expected_adapter_sha256=args.expected_adapter_sha256,
    )
    policy, candidate_ids = validate_policy(
        policy_path,
        registry_path,
        method=args.method,
        seed=args.seed,
        expected_anchors=args.expected_anchors,
    )

    if commitment_path.is_file():
        commitment = verify_existing_commitment(commitment_path, seed_dir)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "seed": args.seed,
                    "valid_anchors": commitment["valid_anchors"],
                    "missing_anchor_slots": commitment["missing_anchor_slots"],
                    "commitment_sha256": sha256_file(commitment_path),
                    "status": "existing_commitment_verified",
                },
                sort_keys=True,
            )
        )
        return

    excluded_names = {"generation_commitment.json"}
    generation_paths = [
        path
        for path in seed_dir.rglob("*")
        if path.is_file() and path.name not in excluded_names
    ]
    generation_files = [
        file_record(path, seed_dir)
        for path in sorted(generation_paths, key=lambda item: item.as_posix())
    ]
    gateway_audits = []
    for gateway_dir in sorted(seed_dir.glob("codex_session_chat_gateway*")):
        if not gateway_dir.is_dir():
            continue
        gateway_manifest = load_object(gateway_dir / "gateway_manifest.json")
        gateway_audits.append(
            {
                "directory": gateway_dir.name,
                "selected": gateway_dir == selected_gateway,
                "counts": gateway_counts(gateway_dir),
                "gateway_source_sha256": gateway_manifest.get("gateway_source_sha256"),
                "tool_contract": gateway_manifest.get("tool_contract"),
            }
        )
    commitment = {
        "schema_version": "case1-official-generation-commitment.v1",
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "seed": args.seed,
        "requested_anchors": args.expected_anchors,
        "valid_anchors": len(candidate_ids),
        "missing_anchor_slots": args.expected_anchors - len(candidate_ids),
        "all_valid_anchors_registered": True,
        "logical_model": task["model"],
        "logical_reasoning_effort": task["reasoning_effort"],
        "actual_backend_model": manifest["actual_model"],
        "actual_backend_reasoning_effort": manifest["reasoning_effort"],
        "transport": manifest["transport"],
        "selected_gateway": args.selected_gateway,
        "selected_gateway_counts": selected_counts,
        "gateway_audits": gateway_audits,
        "control_note": args.control_note,
        "sealed_identities": {
            "task_sha256": sha256_file(task_path),
            "public_registry_sha256": sha256_file(registry_path),
            "knowledge_graph_sha256": args.expected_kg_sha256.lower(),
            "official_adapter_sha256": args.expected_adapter_sha256.lower(),
            "search_policy_sha256": sha256_file(policy_path),
            "gateway_source_sha256": manifest.get("gateway_source_sha256"),
        },
        "policy_metadata": policy.get("metadata") or {},
        "outcomes_loaded_before_commitment": False,
        "repair_or_replacement_performed": False,
        "missing_slots_preserved_without_rerun": True,
        "authorization_headers_persisted": False,
        "generation_files": generation_files,
        "bound_luna_session_logs": bound_session_log_records(selected_gateway),
    }
    write_once_json(commitment_path, commitment)
    print(
        json.dumps(
            {
                "method": args.method,
                "seed": args.seed,
                "valid_anchors": len(candidate_ids),
                "missing_anchor_slots": args.expected_anchors - len(candidate_ids),
                "generation_files": len(generation_files),
                "bound_luna_sessions": len(commitment["bound_luna_session_logs"]),
                "commitment_sha256": sha256_file(commitment_path),
                "policy_sha256": sha256_file(policy_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
