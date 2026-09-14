"""Initialize, bind, and inspect a paired Luna-max claim campaign.

Primary and blinded Cross-QA shard topology is read from the frozen role
manifests/TARGET rather than assumed.  A registry may additionally contain a
dedicated review pool for adjudication and later QA stages.  This manager never
reads or writes the formal KG and never performs semantic extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_WRITER = REPO / "neurooracle/scripts/append_luna_max_claim_extraction_checkpoint.py"
FINALIZER = REPO / "neurooracle/scripts/finalize_luna_max_claim_extraction_fork.py"
CLAIM_EXTRACTOR = REPO / "neurooracle/src/claim_extractor.py"
ROLES = ("primary", "qa")
TARGET_NAME = "TARGET.json"
WORKFLOW_VERSION = "neurooracle.kg_update_workflow.v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256_prefix_lines(path: Path, count: int) -> str:
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        for line in handle:
            if seen >= count:
                break
            digest.update(line)
            seen += 1
    if seen != count:
        raise RuntimeError(f"{path} contains only {seen} rows; expected {count}")
    return digest.hexdigest()


def role_manifest(campaign_dir: Path, role: str) -> dict[str, Any]:
    return read_json(campaign_dir / role / "MANIFEST.json")


def campaign_shards(campaign_dir: Path) -> tuple[str, ...]:
    """Return the frozen source-shard inventory in manifest/TARGET order."""

    campaign_dir = campaign_dir.resolve()
    source_rows = role_manifest(campaign_dir, "primary").get("shards") or []
    source_shards = tuple(str(row.get("shard_id") or "") for row in source_rows)
    qa_rows = role_manifest(campaign_dir, "qa").get("shards") or []
    qa_shards = tuple(str(row.get("shard_id") or "") for row in qa_rows)
    if qa_shards != source_shards:
        raise RuntimeError("Primary/QA frozen shard inventories differ")
    target_path = campaign_dir / TARGET_NAME
    if target_path.is_file():
        rows = read_json(target_path).get("shards") or []
        shards = tuple(str(row.get("shard") or "") for row in rows)
        if shards != source_shards:
            raise RuntimeError("TARGET shard inventory differs from the frozen source manifest")
    else:
        shards = source_shards
    if not shards or any(not shard for shard in shards) or len(set(shards)) != len(shards):
        raise RuntimeError("campaign shard inventory is empty or contains duplicates")
    expected = tuple(f"W{index:02d}" for index in range(1, len(shards) + 1))
    if shards != expected:
        raise RuntimeError(
            f"campaign shards must be canonical contiguous IDs {expected}; got {shards}"
        )
    return shards


def verify_pair(
    campaign_dir: Path, *, allow_existing_output: bool = False
) -> dict[str, Any]:
    manifests = {role: role_manifest(campaign_dir, role) for role in ROLES}
    left = manifests["primary"]
    right = manifests["qa"]
    left_inventory = dict(left.get("inventory") or {})
    right_inventory = dict(right.get("inventory") or {})
    inventory_fields = ("papers", "abstracts", "abstract_coverage_percent", "shards")
    if any(left_inventory.get(field) != right_inventory.get(field) for field in inventory_fields):
        raise RuntimeError("Primary/QA frozen inventory differs")
    papers = int(left_inventory.get("papers") or 0)
    abstracts = int(left_inventory.get("abstracts") or 0)
    shard_count = int(left_inventory.get("shards") or 0)
    if papers <= 0 or abstracts != papers or shard_count <= 0:
        raise RuntimeError("paired inventory must contain complete abstracts for every paper")
    if float(left_inventory.get("abstract_coverage_percent") or 0.0) != 100.0:
        raise RuntimeError("paired inventory abstract coverage must be 100%")
    for role, manifest in manifests.items():
        inventory = manifest.get("inventory") or {}
        if any(inventory.get(field) != left_inventory.get(field) for field in inventory_fields):
            raise RuntimeError(f"{role} inventory does not match its paired role")
        if manifest.get("model") != "gpt-5.6-luna":
            raise RuntimeError(f"{role} model is not frozen to gpt-5.6-luna")
        if manifest.get("reasoning_effort") != "max":
            raise RuntimeError(f"{role} reasoning is not frozen to max")
        if manifest.get("formal_kg_mutation_permitted") is not False:
            raise RuntimeError(f"{role} does not fail closed for formal KG writes")
        outputs = [
            path
            for path in (campaign_dir / role / "forks").glob("W*/raw_items.jsonl")
            if path.stat().st_size > 0
        ]
        if outputs and not allow_existing_output:
            raise RuntimeError(f"{role} already contains worker output: {outputs[0]}")

    semantic_contract_fields = (
        "rubric_version",
        "rubric_sha256",
        "prompt_sha256",
        "audit_contract_version",
    )
    for field in semantic_contract_fields:
        if (left.get("frozen_contract") or {}).get(field) != (
            right.get("frozen_contract") or {}
        ).get(field):
            raise RuntimeError(f"Primary/QA frozen contract mismatch: {field}")

    paired_shards: list[dict[str, Any]] = []
    left_rows = list(left.get("shards") or [])
    right_rows = list(right.get("shards") or [])
    left_shards = {str(row["shard_id"]): row for row in left_rows}
    right_shards = {str(row["shard_id"]): row for row in right_rows}
    shard_order = tuple(str(row["shard_id"]) for row in left_rows)
    expected_shards = tuple(f"W{index:02d}" for index in range(1, shard_count + 1))
    if (
        len(left_rows) != shard_count
        or len(right_rows) != shard_count
        or len(left_shards) != shard_count
        or len(right_shards) != shard_count
        or shard_order != expected_shards
        or tuple(str(row["shard_id"]) for row in right_rows) != shard_order
        or set(left_shards) != set(right_shards)
    ):
        raise RuntimeError("Primary/QA shard inventories differ from the frozen inventory")
    if sum(int(row.get("paper_count") or 0) for row in left_rows) != papers:
        raise RuntimeError("Primary shard paper counts do not sum to inventory papers")
    if sum(int(row.get("paper_count") or 0) for row in right_rows) != papers:
        raise RuntimeError("QA shard paper counts do not sum to inventory papers")
    for shard in shard_order:
        primary = left_shards[shard]
        qa = right_shards[shard]
        comparable = ("queue_start", "queue_end", "paper_count", "task_sha256")
        for field in comparable:
            if primary.get(field) != qa.get(field):
                raise RuntimeError(f"{shard} Primary/QA mismatch: {field}")
        paired_shards.append(
            {
                "shard": shard,
                "queue_start": primary["queue_start"],
                "queue_end": primary["queue_end"],
                "paper_count": primary["paper_count"],
                "task_sha256": primary["task_sha256"],
            }
        )
    expected_start = 1
    for row in paired_shards:
        queue_start = int(row["queue_start"])
        queue_end = int(row["queue_end"])
        paper_count = int(row["paper_count"])
        if queue_start != expected_start or queue_end - queue_start + 1 != paper_count:
            raise RuntimeError("paired shard queue ranges are not contiguous and complete")
        expected_start = queue_end + 1
    if expected_start != papers + 1:
        raise RuntimeError("paired shard queue ranges do not cover the frozen inventory")
    return {
        "manifests": manifests,
        "paired_shards": paired_shards,
        "inventory": left_inventory,
        "shards": shard_order,
    }


def init_campaign(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    verified = verify_pair(campaign_dir)
    manifests = verified["manifests"]
    shard_count = len(verified["shards"])
    papers = int(verified["inventory"]["papers"])
    payload = {
        "schema_version": "neurooracle.paired_luna_campaign.v1",
        "workflow_version": WORKFLOW_VERSION,
        "created_at": utc_now(),
        "campaign_dir": str(campaign_dir),
        "model": "gpt-5.6-luna",
        "reasoning_effort": "max",
        "temperature": 0.0,
        "roles": {"primary": shard_count, "qa": shard_count},
        "papers": papers,
        "total_abstract_reads": papers * 2,
        "paired_shards": verified["paired_shards"],
        "role_manifests": {
            role: {
                "path": str((campaign_dir / role / "MANIFEST.json").resolve()),
                "sha256": sha256_file(campaign_dir / role / "MANIFEST.json"),
                "frozen_contract_sha256": (
                    manifests[role].get("frozen_contract") or {}
                ).get("sha256"),
            }
            for role in ROLES
        },
        "startup_policy": "independent_threads_two_phase_ready_then_begin",
        "primary_qa_blinded_until_both_shard_seals_exist": True,
        "formal_kg_mutation_permitted": False,
        "production_queue_mutation_permitted": False,
        "ready_for_thread_binding": True,
    }
    write_json_atomic(campaign_dir / "CAMPAIGN_MANIFEST.json", payload)
    return payload


def set_target(campaign_dir: Path, papers: int) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    verified = verify_pair(campaign_dir, allow_existing_output=True)
    shards = verified["shards"]
    inventory_papers = int(verified["inventory"]["papers"])
    if papers < len(shards) or papers > inventory_papers:
        raise RuntimeError(
            f"target papers must be between {len(shards)} and {inventory_papers}"
        )
    base, remainder = divmod(papers, len(shards))
    targets: list[dict[str, Any]] = []
    for index, paired in enumerate(verified["paired_shards"]):
        target_count = base + (1 if index < remainder else 0)
        if target_count > int(paired["paper_count"]):
            raise RuntimeError(f"target exceeds source shard {paired['shard']}")
        shard = str(paired["shard"])
        queue_start = int(paired["queue_start"])
        role_hashes: dict[str, str] = {}
        for role in ROLES:
            fork_dir = campaign_dir / role / "forks" / shard
            raw_path = fork_dir / "raw_items.jsonl"
            processed = 0
            if raw_path.is_file() and raw_path.stat().st_size:
                with raw_path.open("rb") as handle:
                    processed = sum(1 for line in handle if line.strip())
            if processed > target_count:
                raise RuntimeError(
                    f"{role}/{shard} already processed {processed}, "
                    f"beyond target {target_count}"
                )
            role_hashes[role] = sha256_prefix_lines(
                fork_dir / "task.jsonl", target_count
            )
        if role_hashes["primary"] != role_hashes["qa"]:
            raise RuntimeError(f"Primary/QA target prefix mismatch for {shard}")
        targets.append(
            {
                "shard": shard,
                "queue_start": queue_start,
                "queue_end": queue_start + target_count - 1,
                "paper_count": target_count,
                "task_prefix_sha256": role_hashes["primary"],
            }
        )
    payload = {
        "schema_version": "neurooracle.paired_luna_target.v1",
        "created_at": utc_now(),
        "selection_policy": "balanced_contiguous_prefix_per_frozen_source_shard",
        "target_unique_papers": papers,
        "target_total_abstract_reads": papers * 2,
        "shards": targets,
        "formal_kg_mutation_permitted": False,
    }
    parent_path = campaign_dir / TARGET_NAME
    if parent_path.is_file():
        previous = read_json(parent_path)
        if (
            previous.get("target_unique_papers") != papers
            or previous.get("shards") != targets
        ):
            raise RuntimeError("refusing to replace a different frozen target")
        return previous
    write_json_atomic(parent_path, payload)
    parent_sha = sha256_file(parent_path)
    for role in ROLES:
        write_json_atomic(
            campaign_dir / role / TARGET_NAME,
            {
                **payload,
                "role": role,
                "parent_target_path": str(parent_path),
                "parent_target_sha256": parent_sha,
            },
        )
    return payload


def register_threads(
    campaign_dir: Path,
    thread_ids: list[str],
    review_thread_ids: list[str] | None = None,
    *,
    wave_id: str = "wave-001",
    benchmark_override_by_user: bool = True,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    verified = verify_pair(campaign_dir)
    manifests = verified["manifests"]
    shards = verified["shards"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", wave_id):
        raise RuntimeError("wave_id must contain only letters, digits, dot, underscore, or dash")
    wave_root = campaign_dir / "waves" / wave_id
    wave_registry_path = wave_root / "THREAD_REGISTRY.json"
    if wave_registry_path.exists():
        raise RuntimeError(f"wave registry is already frozen: {wave_registry_path}")
    source_count = len(shards) * len(ROLES)
    source_thread_ids = list(thread_ids)
    dedicated_review_ids = list(review_thread_ids or [])
    if review_thread_ids is None and len(source_thread_ids) > source_count:
        dedicated_review_ids = source_thread_ids[source_count:]
        source_thread_ids = source_thread_ids[:source_count]
    if len(source_thread_ids) != source_count:
        raise RuntimeError(
            f"register requires exactly {source_count} Primary/QA thread IDs"
        )
    all_thread_ids = [*source_thread_ids, *dedicated_review_ids]
    if any(not str(thread_id).strip() for thread_id in all_thread_ids):
        raise RuntimeError("thread IDs must be non-empty")
    if len(set(all_thread_ids)) != len(all_thread_ids):
        raise RuntimeError("source and review thread IDs must all be unique")
    if (
        not CHECKPOINT_WRITER.is_file()
        or not FINALIZER.is_file()
        or not CLAIM_EXTRACTOR.is_file()
    ):
        raise FileNotFoundError("checkpoint writer/finalizer/claim extractor missing")

    created_at = utc_now()
    assignments: list[dict[str, Any]] = []
    cursor = 0
    for role in ROLES:
        workers: list[dict[str, Any]] = []
        root = campaign_dir / role
        for shard in shards:
            thread_id = source_thread_ids[cursor]
            cursor += 1
            fork_dir = (root / "forks" / shard).resolve()
            row = {
                "role": role,
                "shard": shard,
                "thread_id": thread_id,
                "allowed_write_root": str(fork_dir),
            }
            workers.append(row)
            assignments.append(row)
        registry = {
            "schema_version": "neurooracle.luna_max_fork_thread_registry.v2",
            "workflow_version": WORKFLOW_VERSION,
            "created_at": created_at,
            "wave_id": wave_id,
            "role": role,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "formal_kg_mutation_permitted": False,
            "canonical_workers": workers,
        }
        write_json_atomic(root / "FORK_THREAD_REGISTRY.json", registry)
        recovery = {
            "schema_version": "neurooracle.luna_max_recovery_attestation.v2",
            "created_at": created_at,
            "status": "verified_semantic_equivalence",
            "role": role,
            "frozen_contract_sha256": (
                manifests[role].get("frozen_contract") or {}
            ).get("sha256"),
            "basis": (
                f"Fresh immutable {int(verified['inventory']['papers'])}-paper queue; "
                "no prior raw_items; current "
                "checkpoint writer and finalizer bound by SHA-256."
            ),
            "current_files": {
                "checkpoint_writer": {
                    "path": str(CHECKPOINT_WRITER.resolve()),
                    "sha256": sha256_file(CHECKPOINT_WRITER),
                },
                "finalizer": {
                    "path": str(FINALIZER.resolve()),
                    "sha256": sha256_file(FINALIZER),
                },
                "claim_extractor": {
                    "path": str(CLAIM_EXTRACTOR.resolve()),
                    "sha256": sha256_file(CLAIM_EXTRACTOR),
                },
            },
            "formal_kg_mutated": False,
        }
        write_json_atomic(root / "RECOVERY_ATTESTATION.json", recovery)

    review_workers: list[dict[str, Any]] = []
    for index, thread_id in enumerate(dedicated_review_ids, 1):
        review_workers.append(
            {
                "role": "review",
                "worker_id": f"R{index:02d}",
                "thread_id": thread_id,
                "allowed_write_root": str(
                    (campaign_dir / "postreview" / "review_pool" / f"R{index:02d}").resolve()
                ),
            }
        )
    assignments.extend(review_workers)

    parent = {
        "schema_version": "neurooracle.paired_luna_thread_registry.v2",
        "workflow_version": WORKFLOW_VERSION,
        "created_at": created_at,
        "wave_id": wave_id,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "max",
        "assignments": assignments,
        "source_worker_count": source_count,
        "review_worker_count": len(review_workers),
        "review_pool_mode": "dedicated" if review_workers else "legacy_source_fallback",
        "task_layout": {
            "total": len(assignments),
            "primary": len(shards),
            "qa": len(shards),
            "review": len(review_workers),
        },
        "benchmark_override_by_user": bool(benchmark_override_by_user),
        "task_threads_reusable_across_waves": True,
        "wave_registry_binding_independent": True,
        "formal_kg_mutation_permitted": False,
        "ready_for_dispatch": True,
    }
    parent["registry_binding_sha256"] = canonical_hash(parent)
    write_json_atomic(campaign_dir / "THREAD_REGISTRY.json", parent)
    write_json_atomic(wave_registry_path, parent)
    wave_manifest = {
        "schema_version": "neurooracle.paired_luna_wave_manifest.v2",
        "workflow_version": WORKFLOW_VERSION,
        "created_at": created_at,
        "wave_id": wave_id,
        "campaign_dir": str(campaign_dir),
        "task_layout": parent["task_layout"],
        "benchmark_override_by_user": bool(benchmark_override_by_user),
        "registry_path": str(wave_registry_path.resolve()),
        "registry_file_sha256": sha256_file(wave_registry_path),
        "registry_binding_sha256": parent["registry_binding_sha256"],
        "same_task_threads_may_be_reused_in_later_waves": True,
        "formal_kg_mutation_permitted": False,
    }
    write_json_atomic(wave_root / "WAVE_MANIFEST.json", wave_manifest)
    return parent


def status(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    target_path = campaign_dir / TARGET_NAME
    target = read_json(target_path) if target_path.is_file() else {}
    target_by_shard = {
        str(row["shard"]): int(row["paper_count"])
        for row in target.get("shards") or []
    }
    total_papers = int(target.get("target_unique_papers") or 100000)
    shards = campaign_shards(campaign_dir)
    if target and (
        tuple(target_by_shard) != shards
        or sum(target_by_shard.values()) != total_papers
    ):
        raise RuntimeError("TARGET counts do not cover the complete frozen target inventory")
    if not target:
        total_papers = int(role_manifest(campaign_dir, "primary")["inventory"]["papers"])
    report: dict[str, Any] = {
        "schema_version": "neurooracle.paired_luna_campaign_status.v1",
        "campaign_dir": str(campaign_dir),
        "roles": {},
        "formal_kg_mutated": False,
    }
    for role in ROLES:
        processed = 0
        claims = 0
        complete_shards = 0
        shard_rows: list[dict[str, Any]] = []
        for shard in shards:
            fork_dir = campaign_dir / role / "forks" / shard
            progress_path = fork_dir / "progress.json"
            summary_path = fork_dir / "FINAL_SUMMARY.json"
            progress = read_json(progress_path) if progress_path.is_file() else {}
            summary = read_json(summary_path) if summary_path.is_file() else {}
            papers = int(progress.get("processed_papers") or summary.get("processed_papers") or 0)
            claim_count = int(
                progress.get("claims")
                or progress.get("claim_count")
                or summary.get("claims")
                or summary.get("claim_count")
                or 0
            )
            shard_target = target_by_shard.get(
                shard, int(progress.get("task_papers") or summary.get("task_papers") or 0)
            )
            complete = (
                papers == shard_target
                if target_by_shard
                else summary.get("complete") is True
            )
            processed += papers
            claims += claim_count
            complete_shards += int(complete)
            shard_rows.append(
                {
                    "shard": shard,
                    "processed_papers": papers,
                    "claim_count": claim_count,
                    "complete": complete,
                    "target_papers": shard_target,
                }
            )
        report["roles"][role] = {
            "processed_papers": processed,
            "total_papers": total_papers,
            "completion_percent": round(processed * 100 / total_papers, 6),
            "claim_count": claims,
            "complete_shards": complete_shards,
            "shards": shard_rows,
        }
    primary_by_shard = {
        row["shard"]: row["processed_papers"]
        for row in report["roles"]["primary"]["shards"]
    }
    qa_by_shard = {
        row["shard"]: row["processed_papers"]
        for row in report["roles"]["qa"]["shards"]
    }
    paired = sum(
        min(primary_by_shard[shard], qa_by_shard[shard])
        for shard in shards
    )
    report["paired_progress_papers"] = paired
    report["paired_completion_percent"] = round(paired * 100 / total_papers, 6)
    report["target_unique_papers"] = total_papers
    write_json_atomic(campaign_dir / "STATUS.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--campaign-dir", type=Path, required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--campaign-dir", type=Path, required=True)
    register_parser.add_argument("--thread-ids", nargs="+", required=True)
    register_parser.add_argument("--review-thread-ids", nargs="+", default=None)
    register_parser.add_argument("--wave-id", default="wave-001")
    register_parser.add_argument(
        "--benchmark-override-by-user",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    target_parser = subparsers.add_parser("target")
    target_parser.add_argument("--campaign-dir", type=Path, required=True)
    target_parser.add_argument("--papers", type=int, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        result = init_campaign(args.campaign_dir)
    elif args.command == "register":
        result = register_threads(
            args.campaign_dir,
            args.thread_ids,
            args.review_thread_ids,
            wave_id=args.wave_id,
            benchmark_override_by_user=args.benchmark_override_by_user,
        )
    elif args.command == "target":
        result = set_target(args.campaign_dir, args.papers)
    else:
        result = status(args.campaign_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
