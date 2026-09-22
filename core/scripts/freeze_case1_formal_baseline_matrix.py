"""Freeze the outcome-blind six-method x three-seed Case 1 baseline matrix.

This utility never opens the exhaustive outcome table.  It validates the exact
SearchPolicy artifacts emitted by the already-completed baseline generators,
binds them to the sealed task/registry/KG/adapter identities, and writes a
write-once JSONL artifact for the formal NeuroDiscovery design.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.scripts.case1_search_policy import policy_from_payload, policy_to_payload


SCHEMA = "case1-frozen-baseline-policy-matrix.v1"
METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)
SEEDS = (0, 1, 2)
FORMAL_OFFICIAL = frozenset(METHODS[:4])
LEGACY_ADOPTED_WITHOUT_COMMITMENT = frozenset(
    {("ai_scientist_v2", 0), ("open_coscientist", 0)}
)
FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "adjusted_residual_d",
        "abs_adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "q_fdr_disease",
        "q_fdr_modality",
        "is_gt_top",
        "is_strict_fdr",
        "external_confirmed",
        "external_outcome",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen artifact already exists with different bytes: {path}")
        return
    path.write_bytes(content)


def policy_dir(run_root: Path, method: str, seed: int) -> Path:
    if method in FORMAL_OFFICIAL:
        return (
            run_root
            / "baseline_generation"
            / "formal_official"
            / method
            / f"trial_{seed:02d}"
        )
    return (
        run_root
        / "baseline_generation"
        / "native"
        / method
        / f"seed_{seed:02d}"
    )


def expected_task_hashes(continuation: Mapping[str, Any]) -> dict[tuple[str, int], str]:
    hashes: dict[tuple[str, int], str] = {}
    for record in continuation.get("tasks") or []:
        method = str(record["method"])
        seed = int(record["trial"])
        hashes[(method, seed)] = str(record["destination_task_sha256"]).lower()
        source_hash = str(record["source_task_sha256"]).lower()
        prior = hashes.setdefault((method, 0), source_hash)
        if prior != source_hash:
            raise RuntimeError(f"inconsistent trial-0 task hash for {method}")
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(hashes) != expected:
        raise RuntimeError(
            "frozen continuation manifest does not define the full task matrix; "
            f"missing={sorted(expected - set(hashes))}, "
            f"unexpected={sorted(set(hashes) - expected)}"
        )
    return hashes


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def validate_commitment(
    path: Path,
    *,
    seed_dir: Path,
    method: str,
    seed: int,
    policy_sha256: str,
    task_sha256: str,
    registry_sha256: str,
    kg_sha256: str,
    adapter_sha256: str,
) -> dict[str, Any]:
    commitment = load_json(path)
    if str(commitment.get("method")) != method or int(commitment.get("seed", -1)) != seed:
        raise RuntimeError(f"generation commitment identity mismatch: {path}")
    if commitment.get("outcomes_loaded_before_commitment") is not False:
        raise RuntimeError(f"generation commitment did not preserve blinding: {path}")
    sealed = commitment.get("sealed_identities") or {}
    expected = {
        "knowledge_graph_sha256": kg_sha256,
        "official_adapter_sha256": adapter_sha256,
        "public_registry_sha256": registry_sha256,
        "task_sha256": task_sha256,
    }
    for key, value in expected.items():
        if str(sealed.get(key) or "").lower() != value:
            raise RuntimeError(f"commitment {key} mismatch: {path}")
    recorded_policy = str(sealed.get("search_policy_sha256") or "").lower()
    generation_policy_hashes = {
        str(record.get("sha256") or "").lower()
        for record in commitment.get("generation_files") or []
        if Path(str(record.get("path") or "")).name == "search_policy.json"
    }
    if recorded_policy and recorded_policy != policy_sha256:
        raise RuntimeError(f"commitment policy hash mismatch: {path}")
    commitment_schema = str(commitment.get("schema_version") or "")
    if (
        policy_sha256 not in generation_policy_hashes
        and commitment_schema != "case1-native-generation-commitment.v1"
    ):
        raise RuntimeError(f"commitment does not bind search_policy.json: {path}")
    if str(commitment.get("logical_model")) != "deepseek-v4-pro":
        raise RuntimeError(f"logical model mismatch: {path}")
    if str(commitment.get("logical_reasoning_effort")) != "high":
        raise RuntimeError(f"logical reasoning effort mismatch: {path}")
    if str(commitment.get("actual_backend_model")) != "gpt-5.6-luna":
        raise RuntimeError(f"actual backend mismatch: {path}")
    if str(commitment.get("actual_backend_reasoning_effort")) != "max":
        raise RuntimeError(f"actual backend effort mismatch: {path}")
    return artifact(path)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    continuation_path = (
        run_root / "baseline_generation" / "frozen_continuation_tasks_manifest.json"
    )
    continuation = load_json(continuation_path)
    task_hashes = expected_task_hashes(continuation)

    registry_path = (
        run_root
        / "baseline_generation"
        / "formal_official"
        / "cs1_public_registry.jsonl"
    )
    registry_sha256 = sha256_file(registry_path)
    kg_sha256 = str(continuation["expected_knowledge_graph_sha256"]).lower()
    adapter_sha256 = str(continuation["expected_official_adapter_sha256"]).lower()
    expected_registry = str(continuation["expected_public_registry_sha256"]).lower()
    if registry_sha256 != expected_registry:
        raise RuntimeError("public registry hash differs from the frozen continuation manifest")
    if args.expected_kg_sha256 and kg_sha256 != args.expected_kg_sha256.lower():
        raise RuntimeError("canonical KG hash differs from the requested identity")
    if args.expected_adapter_sha256 and adapter_sha256 != args.expected_adapter_sha256.lower():
        raise RuntimeError("sealed adapter hash differs from the requested identity")

    loaded: list[tuple[Any, dict[str, Any], Path, Path]] = []
    all_anchor_ids: set[str] = set()
    for method in METHODS:
        for seed in SEEDS:
            seed_dir = policy_dir(run_root, method, seed)
            task_path = seed_dir / "task.json"
            policy_path = seed_dir / "search_policy.json"
            actual_task_sha = sha256_file(task_path)
            expected_task_sha = task_hashes[(method, seed)]
            if actual_task_sha != expected_task_sha:
                raise RuntimeError(f"frozen task hash mismatch: {task_path}")
            task = load_json(task_path)
            if str(task.get("method")) != method or int(task.get("trial", -1)) != seed:
                raise RuntimeError(f"task method/trial mismatch: {task_path}")
            if str(task.get("model")) != "deepseek-v4-pro":
                raise RuntimeError(f"task logical model mismatch: {task_path}")
            if str(task.get("reasoning_effort")) != "high":
                raise RuntimeError(f"task logical effort mismatch: {task_path}")
            task_registry = Path(str(task.get("public_registry_path")))
            if not task_registry.is_file() or sha256_file(task_registry) != registry_sha256:
                raise RuntimeError(f"task registry identity mismatch: {task_path}")

            payload = load_json(policy_path)
            leaked = sorted(nested_keys(payload) & FORBIDDEN_POLICY_KEYS)
            if leaked:
                raise RuntimeError(f"policy contains forbidden outcome keys {leaked}: {policy_path}")
            policy = policy_from_payload(payload)
            if policy.method != method or policy.trial != seed:
                raise RuntimeError(f"policy method/trial mismatch: {policy_path}")
            anchor_ids = [anchor.candidate_id for anchor in policy.anchors]
            if not anchor_ids or len(anchor_ids) > int(task.get("n_anchors", 80)):
                raise RuntimeError(f"invalid anchor count in {policy_path}: {len(anchor_ids)}")
            if len(anchor_ids) != len(set(anchor_ids)):
                raise RuntimeError(f"duplicate anchors in {policy_path}")
            all_anchor_ids.update(anchor_ids)
            loaded.append((policy, payload, task_path, policy_path))

    found_anchor_ids: set[str] = set()
    with registry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate_id = str(json.loads(line)["candidate_id"])
            if candidate_id in all_anchor_ids:
                found_anchor_ids.add(candidate_id)
    missing_anchor_ids = sorted(all_anchor_ids - found_anchor_ids)
    if missing_anchor_ids:
        raise RuntimeError(
            f"baseline matrix contains {len(missing_anchor_ids)} anchors outside the registry"
        )

    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for policy, _payload, task_path, policy_path in loaded:
        method = str(policy.method)
        seed = int(policy.trial)
        seed_dir = policy_path.parent
        policy_payload = policy_to_payload(policy)
        rows.append(policy_payload)
        policy_sha = sha256_file(policy_path)
        commitment_path = seed_dir / "generation_commitment.json"
        commitment_record: dict[str, Any] | None
        if commitment_path.is_file():
            commitment_record = validate_commitment(
                commitment_path,
                seed_dir=seed_dir,
                method=method,
                seed=seed,
                policy_sha256=policy_sha,
                task_sha256=sha256_file(task_path),
                registry_sha256=registry_sha256,
                kg_sha256=kg_sha256,
                adapter_sha256=adapter_sha256,
            )
        elif (method, seed) in LEGACY_ADOPTED_WITHOUT_COMMITMENT:
            commitment_record = None
        else:
            raise FileNotFoundError(f"missing generation commitment: {commitment_path}")
        source_records.append(
            {
                "method": method,
                "seed": seed,
                "valid_anchors": len(policy.anchors),
                "missing_anchor_slots": max(0, 80 - len(policy.anchors)),
                "task": artifact(task_path),
                "search_policy": artifact(policy_path),
                "generation_commitment": commitment_record,
                "adoption_mode": (
                    "pre-mixed-continuation_seed0_with_frozen_native_artifacts"
                    if commitment_record is None
                    else "generation_commitment_verified"
                ),
            }
        )

    rows.sort(key=lambda payload: (str(payload["method"]), int(payload["trial"])))
    jsonl_bytes = (
        "".join(canonical_json(payload) + "\n" for payload in rows).encode("utf-8")
    )
    policy_output = output_dir / "case1_search_policies.jsonl"
    write_once(policy_output, jsonl_bytes)
    policy_artifact = artifact(policy_output)

    frozen_fields = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "outcome_tables_opened": False,
        "matrix": {
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "expected_policy_count": 18,
            "observed_policy_count": len(rows),
        },
        "sealed_identities": {
            "knowledge_graph_sha256": kg_sha256,
            "public_registry_sha256": registry_sha256,
            "official_adapter_sha256": adapter_sha256,
        },
        "public_registry": artifact(registry_path),
        "continuation_task_manifest": artifact(continuation_path),
        "combined_policy_artifact": policy_artifact,
        "source_policies": source_records,
        "legacy_adopted_without_generation_commitment": [
            {"method": method, "seed": seed}
            for method, seed in sorted(LEGACY_ADOPTED_WITHOUT_COMMITMENT)
        ],
    }
    manifest_path = output_dir / "frozen_baseline_policy_manifest.json"
    if manifest_path.exists():
        observed = load_json(manifest_path)
        comparable = dict(observed)
        comparable.pop("created_at", None)
        if comparable != frozen_fields:
            raise RuntimeError("frozen baseline manifest differs from current sources")
    else:
        manifest = {
            **frozen_fields,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        write_once(
            manifest_path,
            (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        )
    manifest_artifact = artifact(manifest_path)
    checksum_path = output_dir / "frozen_baseline_policy_manifest.sha256"
    write_once(
        checksum_path,
        f"{manifest_artifact['sha256']}  {manifest_path.name}\n".encode("ascii"),
    )
    return {
        "status": "frozen",
        "policies": len(rows),
        "combined_policy_sha256": policy_artifact["sha256"],
        "manifest_sha256": manifest_artifact["sha256"],
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-kg-sha256")
    parser.add_argument("--expected-adapter-sha256")
    return parser


def main() -> None:
    print(json.dumps(freeze(build_parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
