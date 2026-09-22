"""Finalize one already-generated Case Study 1 native baseline seed.

This utility deliberately separates blinded generation from outcome reveal.  It
first commits the exact prompts and raw native outputs, validates/parses them,
and only then loads the exhaustive result table for exact candidate-id mapping.
It never retries, repairs, expands, or replaces a scientific proposal.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from core.scripts.case1_method_comparison import load_results
    from core.scripts.case1_native_baseline_experiment import (
        extract_json,
        map_validated,
        validate_seed_outputs,
    )
    from core.scripts.case1_search_policy import (
        PolicyAnchor,
        SearchPolicy,
        build_public_registry,
        policy_to_payload,
    )
except ModuleNotFoundError:
    from case1_method_comparison import load_results
    from case1_native_baseline_experiment import (
        extract_json,
        map_validated,
        validate_seed_outputs,
    )
    from case1_search_policy import (
        PolicyAnchor,
        SearchPolicy,
        build_public_registry,
        policy_to_payload,
    )


METHODS = ("brainpilot_native", "biomni_native")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--all-tests", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-kg-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-proposals", type=int, default=80)
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def write_once_text(path: Path, text: str) -> None:
    """Create an artifact, or prove an existing artifact is byte-identical."""

    encoded = text.encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to overwrite non-identical artifact: {path}")
        return
    path.write_bytes(encoded)


def write_once_json(path: Path, payload: Any) -> None:
    write_once_text(path, canonical_json(payload))


def relative_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def discover_batches(seed_dir: Path, expected_proposals: int) -> list[tuple[int, int, Path]]:
    batches: list[tuple[int, int, Path]] = []
    for batch_dir in sorted(seed_dir.glob("batch_*_*")):
        parts = batch_dir.name.split("_")
        if len(parts) != 3:
            continue
        batches.append((int(parts[1]), int(parts[2]), batch_dir))
    ranks = [rank for start, end, _ in batches for rank in range(start, end + 1)]
    expected = list(range(1, expected_proposals + 1))
    if ranks != expected:
        raise RuntimeError(
            f"Batch ranks do not cover 1..{expected_proposals} exactly: {ranks[:5]}..."
        )
    return batches


def verify_gateway(
    gateway: dict[str, Any],
    expected_registry_sha256: str,
    expected_kg_sha256: str,
    expected_adapter_sha256: str,
) -> None:
    registry_sha = str(gateway.get("registry_sha256") or "").lower()
    adapter_sha = str(gateway.get("sealed_adapter_sha256") or "").lower()
    canonical = gateway.get("canonical_release") or {}
    kg_sha = str(
        (((canonical.get("files") or {}).get("knowledge_graph") or {}).get("sha256"))
        or ""
    ).lower()
    expected = {
        "registry": expected_registry_sha256.lower(),
        "kg": expected_kg_sha256.lower(),
        "adapter": expected_adapter_sha256.lower(),
    }
    actual = {"registry": registry_sha, "kg": kg_sha, "adapter": adapter_sha}
    if actual != expected:
        raise RuntimeError(f"Gateway sealed identity mismatch: {actual} != {expected}")
    if gateway.get("transport") != "fresh_codex_session_per_api_request":
        raise RuntimeError("Gateway did not declare a fresh Codex session per API request")
    if gateway.get("authorization_headers_persisted") is not False:
        raise RuntimeError("Gateway credential-persistence declaration is unsafe or missing")


def main() -> None:
    args = parse_args()
    # Keep a mapped drive spelling when supplied.  Expanding R:\ to its long UNC
    # target can push deeply nested audit receipts beyond Win32's legacy path
    # limit even though the same files are valid through the mapped drive.
    run_root = args.run_root
    native_root = run_root / "baseline_generation" / "native"
    seed_dir = native_root / args.method / f"seed_{args.seed:02d}"
    registry_path = native_root / "cs1_public_registry.jsonl"
    task_path = seed_dir / "task.json"
    gateway_dir = seed_dir / "codex_session_chat_gateway"
    gateway_manifest_path = gateway_dir / "gateway_manifest.json"

    for required in (registry_path, task_path, gateway_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    registry_sha = sha256_file(registry_path)
    if registry_sha.lower() != args.expected_registry_sha256.lower():
        raise RuntimeError(f"Public registry hash mismatch: {registry_sha}")

    task = load_json(task_path)
    gateway = load_json(gateway_manifest_path)
    verify_gateway(
        gateway,
        args.expected_registry_sha256,
        args.expected_kg_sha256,
        args.expected_adapter_sha256,
    )
    if task.get("method") != args.method or int(task.get("trial", -1)) != args.seed:
        raise RuntimeError("Task method/trial does not match the requested finalization")
    if task.get("model") != "deepseek-v4-pro" or task.get("reasoning_effort") != "high":
        raise RuntimeError("Frozen logical model contract is not deepseek-v4-pro/high")
    if int(task.get("n_anchors", -1)) != args.expected_proposals:
        raise RuntimeError("Task proposal count does not match the frozen contract")

    batches = discover_batches(seed_dir, args.expected_proposals)
    generation_files: list[dict[str, Any]] = [
        relative_record(task_path, seed_dir),
        relative_record(gateway_manifest_path, seed_dir),
    ]
    batch_payloads: list[tuple[int, int, dict[str, Any] | None, str | None]] = []
    parsed_payloads: list[tuple[Path, dict[str, Any]]] = []
    for start, end, batch_dir in batches:
        prompt_path = batch_dir / "prompt.txt"
        final_path = batch_dir / "final.txt"
        for required in (prompt_path, final_path):
            if not required.is_file():
                raise FileNotFoundError(required)
            generation_files.append(relative_record(required, seed_dir))
        for optional_name in (
            "events.json",
            "client_meta.json",
            "agent_log.json",
            "control_intervention.json",
        ):
            optional_path = batch_dir / optional_name
            if optional_path.is_file():
                generation_files.append(relative_record(optional_path, seed_dir))
        payload = extract_json(final_path.read_text(encoding="utf-8"))
        batch_payloads.append((start, end, payload, None))
        parsed_payloads.append((batch_dir / "parsed.json", payload))

    receipt_paths = sorted((gateway_dir / "receipts").glob("*.json"))
    response_paths = sorted((gateway_dir / "responses").glob("*.json"))
    request_paths = sorted((gateway_dir / "requests").glob("*.json"))
    rejected_paths = sorted((gateway_dir / "rejected").glob("*.json"))
    if not receipt_paths or not response_paths or not request_paths:
        raise RuntimeError("Gateway audit trail is incomplete")
    for path in [*request_paths, *response_paths, *receipt_paths, *rejected_paths]:
        generation_files.append(relative_record(path, seed_dir))

    commitment_path = seed_dir / "generation_commitment.json"
    if commitment_path.is_file():
        commitment = load_json(commitment_path)
        if commitment.get("generation_files") != generation_files:
            raise RuntimeError("Existing generation commitment no longer matches raw files")
    else:
        commitment = {
            "schema_version": "case1-native-generation-commitment.v1",
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "method": args.method,
            "seed": args.seed,
            "requested_proposals": args.expected_proposals,
            "logical_model": task["model"],
            "logical_reasoning_effort": task["reasoning_effort"],
            "actual_backend_model": gateway.get("actual_model"),
            "actual_backend_reasoning_effort": gateway.get("reasoning_effort"),
            "transport": gateway.get("transport"),
            "sealed_identities": {
                "public_registry_sha256": registry_sha,
                "knowledge_graph_sha256": args.expected_kg_sha256.lower(),
                "official_adapter_sha256": args.expected_adapter_sha256.lower(),
                "task_sha256": sha256_file(task_path),
            },
            "gateway_audit_counts": {
                "requests": len(request_paths),
                "responses": len(response_paths),
                "receipts": len(receipt_paths),
                "rejected_attempts": len(rejected_paths),
            },
            "outcomes_loaded_before_commitment": False,
            "repair_or_replacement_allowed": False,
            "generation_files": generation_files,
        }
        write_once_json(commitment_path, commitment)

    # Parse artifacts are derivative of the already committed blind outputs.
    for parsed_path, payload in parsed_payloads:
        write_once_json(parsed_path, payload)

    # Outcome reveal begins only after generation_commitment.json is durable.
    scored = load_results(args.all_tests, args.gt_top_frac)
    public_registry = pd.read_json(registry_path, lines=True, dtype=False)
    rebuilt_registry = build_public_registry(scored)
    compare_columns = list(public_registry.columns)
    if list(rebuilt_registry.columns) != compare_columns:
        raise RuntimeError("Sealed registry columns do not match the result-derived registry")
    left = public_registry.fillna("").astype(str).reset_index(drop=True)
    right = rebuilt_registry.fillna("").astype(str).reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Sealed registry rows do not match the result-derived registry")

    validated = validate_seed_outputs(
        args.method, args.seed, batch_payloads, public_registry
    )
    if len(validated) != args.expected_proposals:
        raise RuntimeError("Validator did not preserve exactly one row per requested slot")
    mapped = map_validated(validated, scored)

    valid = validated[validated["native_schema_valid"]].copy()
    invalid = validated[~validated["native_schema_valid"]].copy()
    anchors = tuple(
        PolicyAnchor(
            candidate_id=str(row["generated_candidate_id"]),
            score=float(row["generated_confidence"]),
            rationale=str(row["generated_rationale"]),
        )
        for row in valid.sort_values("generated_rank", kind="mergesort").to_dict(
            orient="records"
        )
    )
    policy = policy_to_payload(
        SearchPolicy(
            method=args.method,
            trial=args.seed,
            anchors=anchors,
            metadata={
                "adapter": "native_official",
                "requested_proposals": args.expected_proposals,
                "valid_proposals": len(valid),
                "invalid_proposals": len(invalid),
                "invalid_outputs_preserved_without_repair": True,
                "logical_model": task["model"],
                "logical_reasoning_effort": task["reasoning_effort"],
                "actual_backend_model": gateway.get("actual_model"),
                "actual_backend_reasoning_effort": gateway.get("reasoning_effort"),
                "transport": gateway.get("transport"),
                "generation_commitment_sha256": sha256_file(commitment_path),
            },
        )
    )
    standardized = {
        "method": args.method,
        "seed": args.seed,
        "requested_proposals": args.expected_proposals,
        "valid_proposals": len(valid),
        "hypotheses": [
            {
                "rank": int(row["generated_rank"]),
                "candidate_id": str(row["generated_candidate_id"]),
                "rationale": str(row["generated_rationale"]),
                "confidence": float(row["generated_confidence"]),
            }
            for row in valid.sort_values(
                "generated_rank", kind="mergesort"
            ).to_dict(orient="records")
        ],
    }
    invalid_records = [
        {
            "rank": int(row["generated_rank"]),
            "status": str(row["native_validation_status"]),
            "candidate_id_sha256": hashlib.sha256(
                str(row["generated_candidate_id"]).encode("utf-8")
            ).hexdigest(),
        }
        for row in invalid.to_dict(orient="records")
    ]
    validation_summary = {
        "schema_version": "case1-native-seed-validation.v1",
        "method": args.method,
        "seed": args.seed,
        "requested_proposals": args.expected_proposals,
        "valid_unique_proposals": len(valid),
        "invalid_proposals": len(invalid),
        "invalid_records": invalid_records,
        "mapped_proposals": int(mapped["mapping_status"].eq("mapped").sum()),
        "generation_commitment_sha256": sha256_file(commitment_path),
        "all_tests_path": str(args.all_tests),
        "all_tests_sha256": sha256_file(args.all_tests),
        "reveal_after_generation_commitment": True,
        "repair_or_rerun_performed": False,
    }

    mapped_csv = mapped.to_csv(index=False, lineterminator="\n")
    write_once_text(seed_dir / "mapped_hypotheses.csv", mapped_csv)
    write_once_json(seed_dir / "standardized.json", standardized)
    write_once_json(seed_dir / "search_policy.json", policy)
    write_once_json(seed_dir / "validation_summary.json", validation_summary)

    print(
        json.dumps(
            {
                "method": args.method,
                "seed": args.seed,
                "requested": args.expected_proposals,
                "valid": len(valid),
                "invalid": len(invalid),
                "mapped": int(mapped["mapping_status"].eq("mapped").sum()),
                "commitment_sha256": sha256_file(commitment_path),
                "mapped_sha256": sha256_file(seed_dir / "mapped_hypotheses.csv"),
                "policy_sha256": sha256_file(seed_dir / "search_policy.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
