"""Fail-closed adjudication, Secondary-QA, and sealing for paired Luna campaigns.

The extraction campaign remains immutable.  This manager writes only beneath
``<campaign>/postreview`` and never mutates the formal KG.  Production use is:

1. ``init`` after all Primary and blind-QA shards are complete;
2. workers submit adjudication checkpoints with ``append``;
3. ``prepare-secondary`` after every adjudication task is complete;
4. workers submit Secondary-QA checkpoints with ``append``;
5. ``finalize`` to create canonical outputs and block seals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neurooracle.scripts.compare_paired_luna_claim_campaign import (  # noqa: E402
    compare,
)
from neurooracle.scripts.finalize_luna_max_claim_extraction_fork import (  # noqa: E402
    _claim_from_item,
    _validate_raw_row,
)
from neurooracle.scripts.manage_paired_luna_claim_campaign import (  # noqa: E402
    campaign_shards,
    sha256_prefix_lines,
    status as extraction_status,
)
from neurooracle.src.case_study_membership_contract import (  # noqa: E402
    validate_final_scope_reaudit,
)


WORKFLOW_VERSION = "neurooracle.kg_update_workflow.v2"
POSTREVIEW_SCHEMA = "neurooracle.paired_luna_postreview.v1"
ADJUDICATION_SCHEMA = "neurooracle.paired_luna_adjudication.v1"
SECONDARY_SCHEMA = "neurooracle.paired_luna_secondary_qa.v1"
FINAL_SCHEMA = "neurooracle.paired_luna_final_seal.v1"
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
CHECKPOINT_SIZE = 10
LOW_SCOPE_CONFIDENCE_THRESHOLD = 0.75
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_bytes_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8"),
    )


def postreview_dir(campaign_dir: Path) -> Path:
    return campaign_dir.resolve() / "postreview"


def worker_id(assignment: dict[str, Any]) -> str:
    if assignment.get("worker_id"):
        return str(assignment["worker_id"])
    if assignment["role"] == "primary":
        return f"P_{assignment['shard']}"
    if assignment["role"] == "qa":
        return f"Q_{assignment['shard']}"
    raise RuntimeError("review-pool assignments require an explicit worker_id")


def source_workers(campaign_dir: Path) -> list[dict[str, Any]]:
    registry = read_json(campaign_dir / "THREAD_REGISTRY.json")
    shards = campaign_shards(campaign_dir)
    workers: list[dict[str, Any]] = []
    for assignment in registry.get("assignments") or []:
        if assignment.get("role") not in {"primary", "qa"}:
            continue
        row = dict(assignment)
        row["worker_id"] = worker_id(assignment)
        workers.append(row)
    workers.sort(key=lambda row: row["worker_id"])
    expected = len(shards) * 2
    if len(workers) != expected or len({row["thread_id"] for row in workers}) != expected:
        raise RuntimeError(
            f"post-review requires exactly {expected} distinct registered source workers"
        )
    inventory = Counter((str(row["role"]), str(row["shard"])) for row in workers)
    if any(inventory[(role, shard)] != 1 for role in ("primary", "qa") for shard in shards):
        raise RuntimeError("source registry must bind one Primary and one QA worker per shard")
    return workers


def review_workers(campaign_dir: Path) -> list[dict[str, Any]]:
    """Return a dedicated review pool, or source workers for legacy registries."""

    registry = read_json(campaign_dir / "THREAD_REGISTRY.json")
    dedicated: list[dict[str, Any]] = []
    for assignment in registry.get("assignments") or []:
        if assignment.get("role") != "review":
            continue
        row = dict(assignment)
        row["worker_id"] = worker_id(assignment)
        dedicated.append(row)
    if not dedicated:
        return source_workers(campaign_dir)
    dedicated.sort(key=lambda row: row["worker_id"])
    thread_ids = [str(row["thread_id"]) for row in dedicated]
    if len(thread_ids) != len(set(thread_ids)):
        raise RuntimeError("dedicated review-pool thread IDs are not unique")
    source_thread_ids = {str(row["thread_id"]) for row in source_workers(campaign_dir)}
    if source_thread_ids.intersection(thread_ids):
        raise RuntimeError("dedicated review pool overlaps Primary/QA source reviewers")
    return dedicated


def all_workers(campaign_dir: Path) -> list[dict[str, Any]]:
    """Backward-compatible name for the workers eligible for review stages."""

    return review_workers(campaign_dir)


def reviewer_threads_by_shard(campaign_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    shards = campaign_shards(campaign_dir)
    for worker in source_workers(campaign_dir):
        result[str(worker["shard"])].add(str(worker["thread_id"]))
    if any(len(result[shard]) != 2 for shard in shards):
        raise RuntimeError("each source shard must have one Primary and one blind-QA reviewer")
    return dict(result)


def target_by_shard(campaign_dir: Path) -> dict[str, dict[str, Any]]:
    target = read_json(campaign_dir / "TARGET.json")
    return {str(row["shard"]): row for row in target.get("shards") or []}


def block_id(campaign_dir: Path, shard: str, queue_index: int) -> str:
    target = target_by_shard(campaign_dir)[shard]
    offset = int(queue_index) - int(target["queue_start"])
    if offset < 0 or offset >= int(target["paper_count"]):
        raise RuntimeError(f"queue index {queue_index} is outside target shard {shard}")
    return f"{shard}.B{offset // 100 + 1:03d}"


def formal_kg_paths() -> list[Path]:
    root = REPO / "neurooracle" / "data" / "full_v2"
    return [root / "knowledge_graph.json", root / "extracted_claims.jsonl"]


def formal_kg_snapshot() -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for path in formal_kg_paths():
        if not path.is_file():
            raise RuntimeError(f"formal KG file is missing: {path}")
        stat = path.stat()
        snapshot.append(
            {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return snapshot


def assert_formal_kg_unchanged(expected: list[dict[str, Any]]) -> None:
    current = formal_kg_snapshot()
    if current != expected:
        raise RuntimeError("formal KG baseline changed during post-review")


def source_artifact_snapshot(campaign_dir: Path) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for role in ("primary", "qa"):
        for shard in campaign_shards(campaign_dir):
            directory = campaign_dir / role / "forks" / shard
            for name in (
                "task.jsonl",
                "raw_items.jsonl",
                "paper_results.jsonl",
                "claims.jsonl",
                "progress.json",
                "TASK_MANIFEST.json",
                "checkpoint_log.jsonl",
            ):
                path = directory / name
                if not path.is_file():
                    raise RuntimeError(f"completed source artifact is missing: {path}")
                snapshot.append(
                    {
                        "role": role,
                        "shard": shard,
                        "name": name,
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return snapshot


def assert_source_artifacts_unchanged(expected: list[dict[str, Any]]) -> None:
    current: list[dict[str, Any]] = []
    for row in expected:
        path = Path(str(row["path"]))
        if not path.is_file():
            raise RuntimeError(f"frozen extraction artifact is missing: {path}")
        current.append(
            {
                "role": str(row["role"]),
                "shard": str(row["shard"]),
                "name": str(row["name"]),
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if current != expected:
        raise RuntimeError("Primary/QA source artifacts changed after comparator freeze")


def assert_campaign_controls_unchanged(
    campaign_dir: Path, manifest: dict[str, Any]
) -> None:
    expected = {
        "target_sha256": sha256_file(campaign_dir / "TARGET.json"),
        "thread_registry_sha256": sha256_file(campaign_dir / "THREAD_REGISTRY.json"),
        "primary_frozen_contract_sha256": sha256_file(
            campaign_dir / "primary" / "FROZEN_CONTRACT.json"
        ),
        "qa_frozen_contract_sha256": sha256_file(
            campaign_dir / "qa" / "FROZEN_CONTRACT.json"
        ),
        "primary_role_manifest_sha256": sha256_file(
            campaign_dir / "primary" / "MANIFEST.json"
        ),
        "qa_role_manifest_sha256": sha256_file(
            campaign_dir / "qa" / "MANIFEST.json"
        ),
        "primary_recovery_attestation_sha256": sha256_file(
            campaign_dir / "primary" / "RECOVERY_ATTESTATION.json"
        ),
        "qa_recovery_attestation_sha256": sha256_file(
            campaign_dir / "qa" / "RECOVERY_ATTESTATION.json"
        ),
    }
    if any(str(manifest.get(key) or "") != value for key, value in expected.items()):
        raise RuntimeError("campaign target, registry, or frozen contract changed")


def build_target_source_seals(
    campaign_dir: Path, output_dir: Path
) -> list[dict[str, Any]]:
    """Seal exactly the selected target prefix of every role/shard."""

    targets = target_by_shard(campaign_dir)
    seals: list[dict[str, Any]] = []
    for role in ("primary", "qa"):
        for shard in campaign_shards(campaign_dir):
            directory = campaign_dir / role / "forks" / shard
            target = targets[shard]
            count = int(target["paper_count"])
            tasks = read_jsonl(directory / "task.jsonl")[:count]
            raw = read_jsonl(directory / "raw_items.jsonl")
            papers = read_jsonl(directory / "paper_results.jsonl")
            claims = read_jsonl(directory / "claims.jsonl")
            progress = read_json(directory / "progress.json")
            if len(tasks) != count or len(raw) != count or len(papers) != count:
                raise RuntimeError(
                    f"target-prefix coverage mismatch for {role}/{shard}: "
                    f"tasks={len(tasks)} raw={len(raw)} papers={len(papers)} target={count}"
                )
            if int(progress.get("processed_papers") or -1) != count:
                raise RuntimeError(f"progress does not seal target prefix for {role}/{shard}")
            expected_prefix = str(target["task_prefix_sha256"])
            actual_prefix = sha256_prefix_lines(directory / "task.jsonl", count)
            if actual_prefix != expected_prefix:
                raise RuntimeError(f"task prefix hash mismatch for {role}/{shard}")
            expected_queues = [int(task["queue_index"]) for task in tasks]
            if [int(row["queue_index"]) for row in raw] != expected_queues:
                raise RuntimeError(f"raw target prefix is out of order for {role}/{shard}")
            if [int(row["queue_index"]) for row in papers] != expected_queues:
                raise RuntimeError(f"paper target prefix is out of order for {role}/{shard}")
            for task, raw_row, paper in zip(tasks, raw, papers, strict=True):
                identity = (
                    int(task["queue_index"]),
                    str(task["paper_key"]),
                    str(task["source_context_sha256"]),
                )
                if identity != (
                    int(raw_row["queue_index"]),
                    str(raw_row["paper_key"]),
                    str(raw_row["source_context_sha256"]),
                ) or identity != (
                    int(paper["queue_index"]),
                    str(paper["paper_key"]),
                    str(paper["source_context_sha256"]),
                ):
                    raise RuntimeError(f"identity drift in {role}/{shard}/{identity[0]}")
            claim_ids = [str(row["id"]) for row in claims]
            referenced = [
                str(claim_id)
                for paper in papers
                for claim_id in paper.get("claim_ids") or []
            ]
            if len(claim_ids) != len(set(claim_ids)) or sorted(claim_ids) != sorted(referenced):
                raise RuntimeError(f"claim inventory mismatch for {role}/{shard}")
            seal = {
                "schema_version": f"{POSTREVIEW_SCHEMA}.target_source_seal.v1",
                "role": role,
                "shard": shard,
                "target_papers": count,
                "queue_start": int(target["queue_start"]),
                "queue_end": int(target["queue_end"]),
                "task_prefix_sha256": actual_prefix,
                "raw_items_sha256": sha256_file(directory / "raw_items.jsonl"),
                "paper_results_sha256": sha256_file(
                    directory / "paper_results.jsonl"
                ),
                "claims_sha256": sha256_file(directory / "claims.jsonl"),
                "claim_count": len(claims),
                "zero_claim_papers": sum(bool(row.get("zero_claim")) for row in papers),
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "formal_kg_mutated": False,
            }
            seal["seal_sha256"] = canonical_hash(seal)
            write_json(output_dir / f"{role}_{shard}.json", seal)
            seals.append(seal)
    return seals


def manager_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def source_claim_to_item(claim: dict[str, Any]) -> dict[str, Any]:
    metadata = claim.get("metadata") or {}
    evidence = claim.get("evidence") or {}
    reaudit = claim.get("scope_reaudit") or {}
    return {
        "subject": str(claim.get("subject_name") or ""),
        "predicate": str(claim.get("predicate") or ""),
        "object": str(claim.get("object_name") or ""),
        "negated": bool(claim.get("negated", False)),
        "raw_sentence": str(claim.get("raw_text") or ""),
        "subject_type": str(metadata.get("subject_type") or ""),
        "object_type": str(metadata.get("object_type") or ""),
        "subject_canonical_hint": str(metadata.get("subject_canonical_hint") or ""),
        "object_canonical_hint": str(metadata.get("object_canonical_hint") or ""),
        "subject_atlas": str(metadata.get("subject_atlas") or ""),
        "object_atlas": str(metadata.get("object_atlas") or ""),
        "study_type": str(evidence.get("study_type") or ""),
        "methodology": str(evidence.get("methodology") or ""),
        "p_value": evidence.get("p_value"),
        "effect_size": evidence.get("effect_size"),
        "effect_metric": str(evidence.get("effect_metric") or ""),
        "sample_size": evidence.get("sample_size"),
        "replicability": str(evidence.get("replicability") or "single_study"),
        "direction": str(evidence.get("direction") or ""),
        "conditions": list(metadata.get("conditions") or []),
        "population": metadata.get("population"),
        "case_study_ids": list(claim.get("claim_case_study_ids") or []),
        "case_study_gates": dict(reaudit.get("gates") or {}),
        "scope_evidence_spans": list(metadata.get("scope_evidence_spans") or []),
        "scope_confidence": float(
            metadata.get("scope_confidence", reaudit.get("confidence", 0.0)) or 0.0
        ),
        "scope_decision_basis": str(
            metadata.get("scope_decision_basis")
            or reaudit.get("decision_basis")
            or "No formal case-study membership; retained as general neuroscience evidence."
        ),
    }


def validate_items(items: Any, task: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise RuntimeError("items must be a list")
    validation_task = {
        "queue_index": int(task["source_queue_index"]),
        "paper_key": str(task["paper_key"]),
        "source_context_sha256": str(task["source_context_sha256"]),
        "abstract": str(task["abstract"]),
    }
    row = {
        "queue_index": int(task["source_queue_index"]),
        "paper_key": str(task["paper_key"]),
        "source_context_sha256": str(task["source_context_sha256"]),
        "items": items,
    }
    try:
        return _validate_raw_row(row, validation_task)
    except Exception as exc:  # pragma: no cover - message preservation matters
        raise RuntimeError(f"strict item validation failed: {exc}") from exc


def greedy_assign(
    rows: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    excluded_threads: Callable[[dict[str, Any]], set[str]],
) -> dict[str, list[dict[str, Any]]]:
    assigned: dict[str, list[dict[str, Any]]] = {
        str(worker["worker_id"]): [] for worker in workers
    }
    loads = Counter({str(worker["worker_id"]): 0 for worker in workers})
    for row in rows:
        excluded = excluded_threads(row)
        eligible = [
            worker for worker in workers if str(worker["thread_id"]) not in excluded
        ]
        if not eligible:
            raise RuntimeError(f"no eligible independent reviewer for {row['paper_key']}")
        chosen = min(
            eligible,
            key=lambda worker: (
                loads[str(worker["worker_id"])],
                canonical_hash([row["paper_key"], worker["worker_id"]]),
            ),
        )
        key = str(chosen["worker_id"])
        assigned[key].append(row)
        loads[key] += 1
    return assigned


def stage_root(campaign_dir: Path, stage: str) -> Path:
    return postreview_dir(campaign_dir) / stage


def write_stage(
    campaign_dir: Path,
    stage: str,
    assigned: dict[str, list[dict[str, Any]]],
    workers: list[dict[str, Any]],
) -> dict[str, Any]:
    root = stage_root(campaign_dir, stage)
    if stage == "adjudication":
        contract_text = """# Adjudication worker contract

Use only your `next_tasks.jsonl`, the complete abstract embedded in each task, and the frozen
rubric/contract. Review the next contiguous 10 `stage_index` values (or the final
remainder). Do not read or
modify another worker directory or the formal KG.

Write a checkpoint JSONL containing exactly:
`stage_index`, `outcome`, `decision_basis`, `items`.

- `outcome=primary` or `qa`: select that complete candidate inventory and leave
  `items=[]`; the canonical writer derives it.
- `outcome=synthesized`: provide the complete corrected raw-item inventory.
- `outcome=zero_claim`: use `items=[]` only when the abstract reports no valid result.
- `outcome=unresolved_source`: use `items=[]` only for a genuine source/identity defect.

Do not vote between reviewers. Decide from the full abstract. Every synthesized item
must use verbatim evidence, all nine gates, non-exclusive Case Study labels, and the
same frozen extraction standard. Submit only through the canonical manager `append`
command; never edit `decisions.jsonl` directly.
"""
    else:
        contract_text = """# Secondary-QA worker contract

Use only your `next_tasks.jsonl`, the complete abstract, and the proposed adjudicated or
exact-agreement inventory embedded in each task. Independently review the next
contiguous 10 `stage_index` values (or the final remainder). Do not read other worker
directories or the formal
KG.

Write a checkpoint JSONL containing exactly:
`stage_index`, `verdict`, `decision_basis`, `items`.

- `verdict=approve`: confirm the full proposed inventory and leave `items=[]`.
- `verdict=correct`: for an adjudication record, provide the entire corrected inventory;
  an empty inventory is allowed when zero-claim is correct.
- `verdict=material_error`: for an `exact_sample` only, provide the corrected inventory;
  this escalates the complete source 100-paper block.
- `verdict=unresolved_source`: leave `items=[]`; this prevents final sealing.

Verify recall as well as precision, evidence/polarity, entity types, claim granularity,
Case Study labels, all gates, confidence, and decision basis. Submit only through the
canonical manager `append` command; never edit `decisions.jsonl` directly.
"""
    write_bytes_atomic(root / "WORKER_CONTRACT.md", contract_text.encode("utf-8"))
    contract_sha256 = sha256_file(root / "WORKER_CONTRACT.md")
    worker_lookup = {str(row["worker_id"]): row for row in workers}
    registry_rows: list[dict[str, Any]] = []
    total = 0
    for key in sorted(assigned):
        worker = worker_lookup[key]
        directory = root / "workers" / key
        tasks: list[dict[str, Any]] = []
        for index, source in enumerate(assigned[key], 1):
            row = dict(source)
            row["stage_index"] = index
            row["stage_input_sha256"] = canonical_hash(source)
            tasks.append(row)
        write_jsonl(directory / "tasks.jsonl", tasks)
        write_jsonl(directory / "next_tasks.jsonl", tasks[:CHECKPOINT_SIZE])
        task_hash = sha256_file(directory / "tasks.jsonl")
        manifest = {
            "schema_version": f"{POSTREVIEW_SCHEMA}.{stage}.worker.v1",
            "stage": stage,
            "worker_id": key,
            "thread_id": str(worker["thread_id"]),
            "task_count": len(tasks),
            "tasks_sha256": task_hash,
            "worker_contract_sha256": contract_sha256,
            "checkpoint_size": CHECKPOINT_SIZE,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "formal_kg_mutation_permitted": False,
            "allowed_write_root": str(directory.resolve()),
        }
        write_json(directory / "TASK_MANIFEST.json", manifest)
        registry_rows.append(manifest)
        total += len(tasks)
    stage_manifest = {
        "schema_version": f"{POSTREVIEW_SCHEMA}.{stage}.manifest.v1",
        "created_at": utc_now(),
        "stage": stage,
        "workers": registry_rows,
        "task_count": total,
        "worker_contract_sha256": contract_sha256,
        "manager_sha256": manager_sha256(),
        "formal_kg_mutation_permitted": False,
    }
    write_json(root / "MANIFEST.json", stage_manifest)
    return stage_manifest


def init_postreview(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    current = extraction_status(campaign_dir)
    target = int(current["target_unique_papers"])
    shard_count = len(campaign_shards(campaign_dir))
    if (
        int(current["paired_progress_papers"]) != target
        or int(current["roles"]["primary"]["processed_papers"]) != target
        or int(current["roles"]["qa"]["processed_papers"]) != target
        or int(current["roles"]["primary"]["complete_shards"]) != shard_count
        or int(current["roles"]["qa"]["complete_shards"]) != shard_count
    ):
        raise RuntimeError("cannot freeze post-review before every paired extraction shard completes")
    comparison = compare(campaign_dir)
    if int(comparison["paired_papers"]) != target:
        raise RuntimeError("final comparator does not cover the complete target")

    root = postreview_dir(campaign_dir)
    if root.exists():
        raise RuntimeError(f"post-review directory already exists: {root}")
    frozen = root / "frozen"
    source_comparisons = campaign_dir / "comparison" / "comparisons_working.jsonl"
    source_disagreements = campaign_dir / "comparison" / "disagreements_working.jsonl"
    frozen.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_comparisons, frozen / "comparisons.jsonl")
    shutil.copyfile(source_disagreements, frozen / "disagreements.jsonl")
    write_json(frozen / "COMPARATOR_SUMMARY.json", comparison)
    source_seals = build_target_source_seals(
        campaign_dir, frozen / "target_source_seals"
    )

    workers = review_workers(campaign_dir)
    source_reviewers = reviewer_threads_by_shard(campaign_dir)
    disagreements = read_jsonl(frozen / "disagreements.jsonl")
    adjudication_rows: list[dict[str, Any]] = []
    for row in disagreements:
        source = {
            "schema_version": f"{ADJUDICATION_SCHEMA}.task.v1",
            "source_shard": str(row["shard"]),
            "source_queue_index": int(row["queue_index"]),
            "block_id": block_id(campaign_dir, str(row["shard"]), int(row["queue_index"])),
            "paper_key": str(row["paper_key"]),
            "source_context_sha256": str(row["source_context_sha256"]),
            "paper": row.get("paper") or {},
            "abstract": str(row.get("abstract") or ""),
            "comparison_decision": str(row["decision"]),
            "comparison_row_sha256": canonical_hash(row),
            "primary_claims": row.get("primary_claims") or [],
            "qa_claims": row.get("qa_claims") or [],
            "original_reviewer_thread_ids": sorted(
                source_reviewers[str(row["shard"])]
            ),
        }
        adjudication_rows.append(source)
    adjudication_rows.sort(
        key=lambda row: (row["source_shard"], row["source_queue_index"])
    )
    assigned = greedy_assign(
        adjudication_rows,
        workers,
        lambda row: set(row["original_reviewer_thread_ids"]),
    )
    stage_manifest = write_stage(campaign_dir, "adjudication", assigned, workers)

    formal_snapshot = formal_kg_snapshot()
    source_snapshot = source_artifact_snapshot(campaign_dir)
    manifest = {
        "schema_version": POSTREVIEW_SCHEMA,
        "workflow_version": WORKFLOW_VERSION,
        "created_at": utc_now(),
        "campaign_dir": str(campaign_dir),
        "target_papers": target,
        "target_sha256": sha256_file(campaign_dir / "TARGET.json"),
        "thread_registry_sha256": sha256_file(campaign_dir / "THREAD_REGISTRY.json"),
        "primary_frozen_contract_sha256": sha256_file(
            campaign_dir / "primary" / "FROZEN_CONTRACT.json"
        ),
        "qa_frozen_contract_sha256": sha256_file(
            campaign_dir / "qa" / "FROZEN_CONTRACT.json"
        ),
        "primary_role_manifest_sha256": sha256_file(
            campaign_dir / "primary" / "MANIFEST.json"
        ),
        "qa_role_manifest_sha256": sha256_file(
            campaign_dir / "qa" / "MANIFEST.json"
        ),
        "primary_recovery_attestation_sha256": sha256_file(
            campaign_dir / "primary" / "RECOVERY_ATTESTATION.json"
        ),
        "qa_recovery_attestation_sha256": sha256_file(
            campaign_dir / "qa" / "RECOVERY_ATTESTATION.json"
        ),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": 0.0,
        "comparison_schema": str(comparison["schema_version"]),
        "comparator_sha256": sha256_file(Path(compare.__code__.co_filename).resolve()),
        "comparison_summary_sha256": sha256_file(frozen / "COMPARATOR_SUMMARY.json"),
        "comparisons_sha256": sha256_file(frozen / "comparisons.jsonl"),
        "disagreements_sha256": sha256_file(frozen / "disagreements.jsonl"),
        "pending_adjudication_papers": len(disagreements),
        "target_source_seals": len(source_seals),
        "target_source_seal_inventory_sha256": canonical_hash(source_seals),
        "adjudication_manifest_sha256": sha256_file(
            stage_root(campaign_dir, "adjudication") / "MANIFEST.json"
        ),
        "adjudication_tasks": int(stage_manifest["task_count"]),
        "source_shards": shard_count,
        "review_pool_mode": (
            "dedicated"
            if all(str(row.get("role")) == "review" for row in workers)
            else "legacy_source_fallback"
        ),
        "review_pool_workers": len(workers),
        "review_pool_thread_ids_sha256": canonical_hash(
            sorted(str(row["thread_id"]) for row in workers)
        ),
        "formal_kg_baseline": formal_snapshot,
        "source_artifacts": source_snapshot,
        "source_artifact_inventory_sha256": canonical_hash(source_snapshot),
        "manager_sha256": manager_sha256(),
        "formal_kg_mutation_permitted": False,
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def read_checkpoint(path: Path, stage: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows or len(rows) > CHECKPOINT_SIZE:
        raise RuntimeError(f"checkpoint must contain 1..{CHECKPOINT_SIZE} rows")
    expected_keys = (
        {"stage_index", "outcome", "decision_basis", "items"}
        if stage == "adjudication"
        else {"stage_index", "verdict", "decision_basis", "items"}
    )
    for offset, row in enumerate(rows):
        if set(row) != expected_keys:
            raise RuntimeError(f"checkpoint row {offset + 1} has invalid fields")
        if not isinstance(row["stage_index"], int) or isinstance(row["stage_index"], bool):
            raise RuntimeError("stage_index must be an integer")
        if not isinstance(row["items"], list):
            raise RuntimeError("items must be a list")
        if len(str(row["decision_basis"]).strip()) < 20:
            raise RuntimeError("decision_basis must contain at least 20 characters")
    return rows


def stage_worker(campaign_dir: Path, stage: str, key: str) -> tuple[Path, dict[str, Any]]:
    directory = stage_root(campaign_dir, stage) / "workers" / key
    manifest = read_json(directory / "TASK_MANIFEST.json")
    stage_manifest = read_json(stage_root(campaign_dir, stage) / "MANIFEST.json")
    root_manifest = read_json(postreview_dir(campaign_dir) / "MANIFEST.json")
    if (
        str(manifest.get("stage")) != stage
        or str(manifest.get("worker_id")) != key
        or manifest.get("formal_kg_mutation_permitted") is not False
        or stage_manifest.get("formal_kg_mutation_permitted") is not False
        or root_manifest.get("formal_kg_mutation_permitted") is not False
    ):
        raise RuntimeError("post-review worker manifest is invalid")
    if Path(str(manifest.get("allowed_write_root") or "")).resolve() != directory.resolve():
        raise RuntimeError("worker allowed_write_root does not match its isolated directory")
    if str(root_manifest.get("manager_sha256")) != manager_sha256():
        raise RuntimeError("post-review manager changed after campaign freeze")
    if str(stage_manifest.get("manager_sha256")) != manager_sha256():
        raise RuntimeError("stage manager attestation changed")
    if sha256_file(directory / "tasks.jsonl") != str(manifest["tasks_sha256"]):
        raise RuntimeError("worker task file hash drift")
    contract_path = stage_root(campaign_dir, stage) / "WORKER_CONTRACT.md"
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != str(manifest["worker_contract_sha256"])
        or sha256_file(contract_path) != str(stage_manifest["worker_contract_sha256"])
    ):
        raise RuntimeError("worker contract hash drift")
    return directory, manifest


def adjudication_decision(
    checkpoint: dict[str, Any], task: dict[str, Any], thread_id: str
) -> dict[str, Any]:
    if thread_id in set(
        str(value) for value in task.get("original_reviewer_thread_ids") or []
    ):
        raise RuntimeError("adjudicator is not independent of Primary and blind QA")
    outcome = str(checkpoint["outcome"])
    allowed = {"primary", "qa", "synthesized", "zero_claim", "unresolved_source"}
    if outcome not in allowed:
        raise RuntimeError(f"invalid adjudication outcome: {outcome}")
    supplied_items = checkpoint["items"]
    if outcome == "primary":
        if supplied_items:
            raise RuntimeError("primary outcome derives items; supplied items must be empty")
        items = [source_claim_to_item(row) for row in task["primary_claims"]]
    elif outcome == "qa":
        if supplied_items:
            raise RuntimeError("qa outcome derives items; supplied items must be empty")
        items = [source_claim_to_item(row) for row in task["qa_claims"]]
    elif outcome in {"zero_claim", "unresolved_source"}:
        if supplied_items:
            raise RuntimeError(f"{outcome} outcome must have no claim items")
        items = []
    else:
        items = supplied_items
    canonical_items = validate_items(items, task)
    if outcome == "synthesized" and not canonical_items:
        raise RuntimeError("use zero_claim instead of empty synthesized output")
    return {
        "schema_version": ADJUDICATION_SCHEMA,
        "stage_index": int(checkpoint["stage_index"]),
        "source_shard": str(task["source_shard"]),
        "source_queue_index": int(task["source_queue_index"]),
        "block_id": str(task["block_id"]),
        "paper_key": str(task["paper_key"]),
        "source_context_sha256": str(task["source_context_sha256"]),
        "comparison_decision": str(task["comparison_decision"]),
        "comparison_row_sha256": str(task["comparison_row_sha256"]),
        "stage_input_sha256": str(task["stage_input_sha256"]),
        "outcome": outcome,
        "items": canonical_items,
        "decision_basis": str(checkpoint["decision_basis"]).strip(),
        "adjudicator_thread_id": thread_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "formal_kg_mutated": False,
    }


def secondary_decision(
    checkpoint: dict[str, Any], task: dict[str, Any], thread_id: str
) -> dict[str, Any]:
    excluded = set(
        str(value) for value in task.get("original_reviewer_thread_ids") or []
    )
    if task.get("adjudicator_thread_id"):
        excluded.add(str(task["adjudicator_thread_id"]))
    if thread_id in excluded:
        raise RuntimeError("Secondary-QA reviewer is not independent")
    verdict = str(checkpoint["verdict"])
    allowed = {"approve", "correct", "unresolved_source", "material_error"}
    if verdict not in allowed:
        raise RuntimeError(f"invalid Secondary-QA verdict: {verdict}")
    supplied_items = checkpoint["items"]
    if verdict == "approve":
        if task.get("proposed_outcome") == "unresolved_source":
            raise RuntimeError(
                "an unresolved-source adjudication cannot be approved as a zero claim"
            )
        if supplied_items:
            raise RuntimeError("approve derives proposed items; supplied items must be empty")
        items = task["proposed_items"]
    elif verdict == "unresolved_source":
        if supplied_items:
            raise RuntimeError("unresolved_source must have no claim items")
        items = []
    else:
        items = supplied_items
    canonical_items = validate_items(items, task)
    if verdict == "material_error" and task["review_kind"] != "exact_sample":
        raise RuntimeError("material_error is reserved for exact-agreement sampling")
    if verdict == "correct" and task["review_kind"] == "exact_sample":
        raise RuntimeError("an exact-sample correction must be marked material_error")
    return {
        "schema_version": SECONDARY_SCHEMA,
        "stage_index": int(checkpoint["stage_index"]),
        "review_kind": str(task["review_kind"]),
        "sample_stratum": str(task.get("sample_stratum") or ""),
        "low_scope_confidence": bool(task.get("low_scope_confidence", False)),
        "source_shard": str(task["source_shard"]),
        "source_queue_index": int(task["source_queue_index"]),
        "block_id": str(task["block_id"]),
        "paper_key": str(task["paper_key"]),
        "source_context_sha256": str(task["source_context_sha256"]),
        "stage_input_sha256": str(task["stage_input_sha256"]),
        "proposed_outcome": str(task.get("proposed_outcome") or ""),
        "adjudication_decision_sha256": str(
            task.get("adjudication_decision_sha256") or ""
        ),
        "verdict": verdict,
        "items": canonical_items,
        "decision_basis": str(checkpoint["decision_basis"]).strip(),
        "secondary_reviewer_thread_id": thread_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "formal_kg_mutated": False,
    }


def append_checkpoint(
    campaign_dir: Path,
    stage: str,
    key: str,
    input_path: Path,
    thread_id: str,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    if stage not in {"adjudication", "secondary"}:
        raise RuntimeError("stage must be adjudication or secondary")
    directory, manifest = stage_worker(campaign_dir, stage, key)
    if str(manifest["thread_id"]) != thread_id:
        raise RuntimeError("calling thread is not the registered stage owner")
    input_path = input_path.resolve()
    if input_path.parent != directory.resolve():
        raise RuntimeError("checkpoint must be inside the registered worker directory")
    tasks = read_jsonl(directory / "tasks.jsonl")
    existing_path = directory / "decisions.jsonl"
    existing = read_jsonl(existing_path)
    observed_exists = existing_path.exists()
    observed_bytes = existing_path.read_bytes() if observed_exists else b""
    observed_hash = sha256_file(existing_path) if observed_exists else ""
    checkpoint = read_checkpoint(input_path, stage)
    remaining = len(tasks) - len(existing)
    expected_size = min(CHECKPOINT_SIZE, remaining)
    if len(checkpoint) != expected_size:
        raise RuntimeError(f"next checkpoint must contain exactly {expected_size} rows")
    expected_start = len(existing) + 1
    for offset, row in enumerate(checkpoint):
        if int(row["stage_index"]) != expected_start + offset:
            raise RuntimeError(f"checkpoint must start at stage_index {expected_start}")

    lock = directory / ".checkpoint.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("another checkpoint writer is active") from exc
    try:
        os.close(descriptor)
        current_exists = existing_path.exists()
        current_hash = sha256_file(existing_path) if current_exists else ""
        if current_exists != observed_exists or current_hash != observed_hash:
            raise RuntimeError(
                "decision file changed before the checkpoint lock was acquired; retry"
            )
        original = observed_bytes
        new_rows: list[dict[str, Any]] = []
        for offset, raw in enumerate(checkpoint):
            task = tasks[len(existing) + offset]
            decision = (
                adjudication_decision(raw, task, thread_id)
                if stage == "adjudication"
                else secondary_decision(raw, task, thread_id)
            )
            decision["decision_sha256"] = canonical_hash(decision)
            new_rows.append(decision)
        if existing_path.exists() and sha256_file(existing_path) != observed_hash:
            raise RuntimeError("decision file changed during checkpoint validation")
        payload = original + "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in new_rows
        ).encode("utf-8")
        write_bytes_atomic(existing_path, payload)
        summary = {
            "schema_version": f"{POSTREVIEW_SCHEMA}.{stage}.worker_status.v1",
            "stage": stage,
            "worker_id": key,
            "processed": len(existing) + len(new_rows),
            "target": len(tasks),
            "complete": len(existing) + len(new_rows) == len(tasks),
            "decisions_sha256": sha256_file(existing_path),
            "formal_kg_mutated": False,
        }
        write_json(directory / "STATUS.json", summary)
        next_start = len(existing) + len(new_rows)
        write_jsonl(
            directory / "next_tasks.jsonl",
            tasks[next_start : next_start + CHECKPOINT_SIZE],
        )
        return summary
    finally:
        lock.unlink(missing_ok=True)


def source_rows(campaign_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    target = target_by_shard(campaign_dir)
    for shard in campaign_shards(campaign_dir):
        directory = campaign_dir / "primary" / "forks" / shard
        tasks = read_jsonl(directory / "task.jsonl")[: int(target[shard]["paper_count"])]
        papers = read_jsonl(directory / "paper_results.jsonl")
        claims = {str(row["id"]): row for row in read_jsonl(directory / "claims.jsonl")}
        if len(tasks) != len(papers):
            raise RuntimeError(f"Primary source coverage mismatch for {shard}")
        for task, paper in zip(tasks, papers, strict=True):
            key = (shard, int(task["queue_index"]))
            rows[key] = {
                "task": task,
                "paper_result": paper,
                "claims": [claims[str(claim_id)] for claim_id in paper.get("claim_ids") or []],
            }
    return rows


def exact_sample_stratum(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return "zero_claim"
    labels = sorted(
        {
            str(label)
            for claim in claims
            for label in claim.get("claim_case_study_ids") or []
        }
    )
    return f"case:{labels[0]}" if labels else "general_only"


def is_low_scope_confidence(claims: list[dict[str, Any]]) -> bool:
    from neurooracle.src.kg_bulk_cleanup import legacy_scope_metadata
    return any(
        float(legacy_scope_metadata(claim).get("scope_confidence") or 0.0)
        < LOW_SCOPE_CONFIDENCE_THRESHOLD
        for claim in claims
    )


def select_exact_samples(
    campaign_dir: Path,
    comparisons: list[dict[str, Any]],
    sources: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        if row["decision"] != "exact_agreement":
            continue
        shard = str(row["shard"])
        queue = int(row["queue_index"])
        source = sources[(shard, queue)]
        candidate = {
            "comparison": row,
            "source": source,
            "stratum": exact_sample_stratum(source["claims"]),
            "low_scope_confidence": is_low_scope_confidence(source["claims"]),
        }
        by_block[block_id(campaign_dir, shard, queue)].append(candidate)

    selected: list[dict[str, Any]] = []
    for current_block in sorted(by_block):
        candidates = sorted(
            by_block[current_block],
            key=lambda row: canonical_hash(
                [current_block, row["comparison"]["paper_key"]]
            ),
        )
        wanted = min(10, len(candidates))
        chosen: list[dict[str, Any]] = [
            candidate for candidate in candidates if candidate["low_scope_confidence"]
        ]
        if len(chosen) >= wanted:
            selected.extend(chosen)
            continue
        seen: set[str] = set()
        seen.update(candidate["stratum"] for candidate in chosen)
        for candidate in candidates:
            if candidate in chosen:
                continue
            if candidate["stratum"] in seen:
                continue
            chosen.append(candidate)
            seen.add(candidate["stratum"])
            if len(chosen) == wanted:
                break
        if len(chosen) < wanted:
            chosen_keys = {row["comparison"]["paper_key"] for row in chosen}
            chosen.extend(
                row
                for row in candidates
                if row["comparison"]["paper_key"] not in chosen_keys
            )
            chosen = chosen[:wanted]
        selected.extend(chosen)
    return selected


def complete_stage_decisions(campaign_dir: Path, stage: str) -> list[dict[str, Any]]:
    manifest = read_json(stage_root(campaign_dir, stage) / "MANIFEST.json")
    decisions: list[dict[str, Any]] = []
    for worker in manifest.get("workers") or []:
        directory = stage_root(campaign_dir, stage) / "workers" / str(worker["worker_id"])
        tasks = read_jsonl(directory / "tasks.jsonl")
        rows = read_jsonl(directory / "decisions.jsonl")
        if len(rows) != len(tasks):
            raise RuntimeError(
                f"{stage}/{worker['worker_id']} incomplete: {len(rows)}/{len(tasks)}"
            )
        decisions.extend(rows)
    if len(decisions) != int(manifest["task_count"]):
        raise RuntimeError(f"{stage} decision count does not match stage manifest")
    return decisions


def prepare_secondary(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    root = postreview_dir(campaign_dir)
    root_manifest = read_json(root / "MANIFEST.json")
    assert_formal_kg_unchanged(root_manifest["formal_kg_baseline"])
    assert_source_artifacts_unchanged(root_manifest["source_artifacts"])
    assert_campaign_controls_unchanged(campaign_dir, root_manifest)
    if (root / "secondary").exists():
        raise RuntimeError("Secondary-QA stage already exists")
    adjudications = complete_stage_decisions(campaign_dir, "adjudication")
    adjudication_by_key = {
        (str(row["source_shard"]), int(row["source_queue_index"])): row
        for row in adjudications
    }
    if len(adjudication_by_key) != len(adjudications):
        raise RuntimeError("duplicate adjudication paper identity")

    comparisons = read_jsonl(root / "frozen" / "comparisons.jsonl")
    sources = source_rows(campaign_dir)
    source_reviewers = reviewer_threads_by_shard(campaign_dir)
    secondary_rows: list[dict[str, Any]] = []
    for key, decision in sorted(adjudication_by_key.items()):
        source = sources[key]["task"]
        secondary_rows.append(
            {
                "schema_version": f"{SECONDARY_SCHEMA}.task.v1",
                "review_kind": "adjudication",
                "source_shard": key[0],
                "source_queue_index": key[1],
                "block_id": str(decision["block_id"]),
                "paper_key": str(decision["paper_key"]),
                "source_context_sha256": str(decision["source_context_sha256"]),
                "paper": source.get("paper") or {},
                "abstract": str(source.get("abstract") or ""),
                "proposed_outcome": str(decision["outcome"]),
                "proposed_items": decision["items"],
                "adjudication_decision_basis": str(decision["decision_basis"]),
                "adjudication_decision_sha256": str(decision["decision_sha256"]),
                "adjudicator_thread_id": str(decision["adjudicator_thread_id"]),
                "original_reviewer_thread_ids": sorted(source_reviewers[key[0]]),
            }
        )

    exact_samples = select_exact_samples(campaign_dir, comparisons, sources)
    for sample in exact_samples:
        comparison = sample["comparison"]
        source = sample["source"]
        shard = str(comparison["shard"])
        queue = int(comparison["queue_index"])
        task = source["task"]
        secondary_rows.append(
            {
                "schema_version": f"{SECONDARY_SCHEMA}.task.v1",
                "review_kind": "exact_sample",
                "sample_stratum": str(sample["stratum"]),
                "low_scope_confidence": bool(sample["low_scope_confidence"]),
                "source_shard": shard,
                "source_queue_index": queue,
                "block_id": block_id(campaign_dir, shard, queue),
                "paper_key": str(comparison["paper_key"]),
                "source_context_sha256": str(comparison["source_context_sha256"]),
                "paper": task.get("paper") or {},
                "abstract": str(task.get("abstract") or ""),
                "proposed_outcome": "exact_primary",
                "proposed_items": [source_claim_to_item(row) for row in source["claims"]],
                "adjudication_decision_basis": "",
                "adjudication_decision_sha256": "",
                "adjudicator_thread_id": "",
                "original_reviewer_thread_ids": sorted(source_reviewers[shard]),
            }
        )
    secondary_rows.sort(
        key=lambda row: (
            row["source_shard"],
            row["source_queue_index"],
            row["review_kind"],
        )
    )
    workers = review_workers(campaign_dir)
    assigned = greedy_assign(
        secondary_rows,
        workers,
        lambda row: set(row["original_reviewer_thread_ids"])
        | ({str(row["adjudicator_thread_id"])} if row["adjudicator_thread_id"] else set()),
    )
    stage_manifest = write_stage(campaign_dir, "secondary", assigned, workers)
    stage_manifest["adjudication_reviews"] = len(adjudications)
    stage_manifest["exact_agreement_samples"] = len(exact_samples)
    stage_manifest["sampling_policy"] = (
        "100% exact agreements below scope confidence 0.75, plus deterministic "
        "stratified sampling to min(10, available exact agreements) per source block"
    )
    stage_manifest["low_scope_confidence_threshold"] = LOW_SCOPE_CONFIDENCE_THRESHOLD
    write_json(stage_root(campaign_dir, "secondary") / "MANIFEST.json", stage_manifest)
    root_manifest["secondary_manifest_sha256"] = sha256_file(
        stage_root(campaign_dir, "secondary") / "MANIFEST.json"
    )
    root_manifest["secondary_tasks"] = int(stage_manifest["task_count"])
    root_manifest["secondary_exact_samples"] = len(exact_samples)
    write_json(root / "MANIFEST.json", root_manifest)
    return stage_manifest


def paper_labels(items: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(label)
            for item in items
            for label in item.get("case_study_ids") or []
        }
    )


def final_claims_from_items(
    items: list[dict[str, Any]],
    task: dict[str, Any],
    campaign_version: str,
    reviewed_at: str,
) -> list[dict[str, Any]]:
    labels = paper_labels(items)
    validation_task = {
        "queue_index": int(task["queue_index"]),
        "paper_key": str(task["paper_key"]),
        "source_context_sha256": str(task["source_context_sha256"]),
        "abstract": str(task["abstract"]),
        "paper": task.get("paper") or {},
    }
    canonical = _validate_raw_row(
        {
            "queue_index": int(task["queue_index"]),
            "paper_key": str(task["paper_key"]),
            "source_context_sha256": str(task["source_context_sha256"]),
            "items": items,
        },
        validation_task,
    )
    return [
        _claim_from_item(
            item,
            validation_task,
            item_index,
            campaign_version,
            labels,
            reviewed_at,
        )
        for item_index, item in enumerate(canonical)
    ]


def finalize_postreview(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    root = postreview_dir(campaign_dir)
    manifest = read_json(root / "MANIFEST.json")
    assert_formal_kg_unchanged(manifest["formal_kg_baseline"])
    assert_source_artifacts_unchanged(manifest["source_artifacts"])
    assert_campaign_controls_unchanged(campaign_dir, manifest)
    adjudications = complete_stage_decisions(campaign_dir, "adjudication")
    secondary = complete_stage_decisions(campaign_dir, "secondary")
    unresolved = [row for row in secondary if row["verdict"] == "unresolved_source"]
    material_errors = [row for row in secondary if row["verdict"] == "material_error"]
    if unresolved or material_errors:
        report = {
            "schema_version": f"{FINAL_SCHEMA}.blocked.v1",
            "unresolved_source": len(unresolved),
            "material_exact_sample_errors": len(material_errors),
            "escalated_blocks": sorted({row["block_id"] for row in material_errors}),
            "formal_kg_mutated": False,
        }
        write_json(root / "ESCALATION_REQUIRED.json", report)
        raise RuntimeError(canonical_json(report))

    secondary_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    sample_decisions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in secondary:
        key = (str(row["source_shard"]), int(row["source_queue_index"]))
        target = sample_decisions if row["review_kind"] == "exact_sample" else secondary_by_key
        if key in target:
            raise RuntimeError(f"duplicate Secondary-QA decision for {key}")
        target[key] = row

    comparisons = read_jsonl(root / "frozen" / "comparisons.jsonl")
    sources = source_rows(campaign_dir)
    contract = read_json(campaign_dir / "primary" / "FROZEN_CONTRACT.json")
    campaign_version = str(contract["campaign_version"])
    reviewed_at = str(manifest["created_at"])
    final_papers: list[dict[str, Any]] = []
    final_claims: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    block_papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    claim_ids: set[str] = set()

    for comparison in comparisons:
        shard = str(comparison["shard"])
        queue = int(comparison["queue_index"])
        key = (shard, queue)
        source = sources[key]
        task = source["task"]
        current_block = block_id(campaign_dir, shard, queue)
        if comparison["decision"] == "exact_agreement":
            claims = source["claims"]
            items = [source_claim_to_item(row) for row in claims]
            provenance = "exact_agreement"
        else:
            if key not in secondary_by_key:
                raise RuntimeError(f"missing Secondary-QA decision for disagreement {key}")
            decision = secondary_by_key[key]
            items = decision["items"]
            claims = final_claims_from_items(
                items, task, campaign_version, reviewed_at
            )
            provenance = "adjudicated_secondary_qa"
        ids = [str(row["id"]) for row in claims]
        duplicate = claim_ids.intersection(ids)
        if duplicate:
            raise RuntimeError(f"duplicate final claim IDs: {sorted(duplicate)[:3]}")
        claim_ids.update(ids)
        labels = sorted(
            {
                str(label)
                for claim in claims
                for label in claim.get("claim_case_study_ids") or []
            }
        )
        for claim in claims:
            validate_final_scope_reaudit(claim)
            claim_labels = set(str(value) for value in claim.get("claim_case_study_ids") or [])
            claim_paper_labels = set(
                str(value) for value in claim.get("paper_case_study_ids") or []
            )
            if not claim_labels.issubset(claim_paper_labels):
                raise RuntimeError(f"claim membership is outside paper membership for {key}")
            if sorted(claim_paper_labels) != labels:
                raise RuntimeError(f"claim paper union mismatch for {key}")
        paper_row = {
            "schema_version": "neurooracle.luna_max_fork_claim_extraction_result.v2",
            "queue_index": queue,
            "paper_key": str(task["paper_key"]),
            "pmid": str((task.get("paper") or {}).get("pmid") or ""),
            "source_context_sha256": str(task["source_context_sha256"]),
            "raw_item_count": 1,
            "claim_count": len(claims),
            "zero_claim": not claims,
            "claim_ids": ids,
            "paper_case_study_ids": labels,
            "validation_status": "final_complete",
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "postreview_provenance": provenance,
        }
        raw_rows.append(
            {
                "queue_index": queue,
                "paper_key": str(task["paper_key"]),
                "pmid": str((task.get("paper") or {}).get("pmid") or ""),
                "source_context_sha256": str(task["source_context_sha256"]),
                "items": items,
            }
        )
        final_papers.append(paper_row)
        final_claims.extend(claims)
        block_papers[current_block].append(paper_row)

    target = read_json(campaign_dir / "TARGET.json")
    if len(final_papers) != int(target["target_unique_papers"]):
        raise RuntimeError("final paper coverage does not equal the frozen target")
    final_dir = root / "final"
    write_jsonl(final_dir / "raw_items.jsonl", raw_rows)
    write_jsonl(final_dir / "paper_results.jsonl", final_papers)
    write_jsonl(final_dir / "claims.jsonl", final_claims)

    adj_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sec_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adjudications:
        adj_by_block[str(row["block_id"])].append(row)
    for row in secondary:
        target_map = sample_by_block if row["review_kind"] == "exact_sample" else sec_by_block
        target_map[str(row["block_id"])].append(row)

    block_seals: list[dict[str, Any]] = []
    for current_block in sorted(block_papers):
        papers = block_papers[current_block]
        disagreements = adj_by_block.get(current_block, [])
        secondary_rows = sec_by_block.get(current_block, [])
        samples = sample_by_block.get(current_block, [])
        if len(disagreements) != len(secondary_rows):
            raise RuntimeError(f"Secondary-QA disagreement coverage gap in {current_block}")
        exact_count = len(papers) - len(disagreements)
        required_sample = min(10, exact_count)
        if len(samples) < required_sample:
            raise RuntimeError(f"exact-agreement sample coverage gap in {current_block}")
        seal = {
            "schema_version": FINAL_SCHEMA,
            "workflow_version": WORKFLOW_VERSION,
            "block_id": current_block,
            "paper_count": len(papers),
            "disagreement_count": len(disagreements),
            "secondary_disagreement_reviews": len(secondary_rows),
            "exact_agreement_count": exact_count,
            "exact_sample_required": required_sample,
            "exact_sample_reviewed": len(samples),
            "open_disagreements": 0,
            "paper_inventory_sha256": canonical_hash(papers),
            "adjudication_inventory_sha256": canonical_hash(disagreements),
            "secondary_inventory_sha256": canonical_hash(secondary_rows),
            "exact_sample_inventory_sha256": canonical_hash(samples),
            "comparison_sha256": str(manifest["comparisons_sha256"]),
            "comparator_sha256": str(manifest["comparator_sha256"]),
            "source_artifact_inventory_sha256": str(
                manifest["source_artifact_inventory_sha256"]
            ),
            "target_source_seal_inventory_sha256": str(
                manifest["target_source_seal_inventory_sha256"]
            ),
            "target_sha256": str(manifest["target_sha256"]),
            "thread_registry_sha256": str(manifest["thread_registry_sha256"]),
            "primary_frozen_contract_sha256": str(
                manifest["primary_frozen_contract_sha256"]
            ),
            "qa_frozen_contract_sha256": str(
                manifest["qa_frozen_contract_sha256"]
            ),
            "primary_recovery_attestation_sha256": str(
                manifest["primary_recovery_attestation_sha256"]
            ),
            "qa_recovery_attestation_sha256": str(
                manifest["qa_recovery_attestation_sha256"]
            ),
            "rubric_version": str(contract["rubric_version"]),
            "rubric_sha256": str(contract["rubric_sha256"]),
            "case_study_registry_sha256": str(
                contract["case_study_registry_sha256"]
            ),
            "audit_contract_version": str(contract["audit_contract_version"]),
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": 0.0,
            "formal_kg_mutated": False,
        }
        seal["seal_sha256"] = canonical_hash(seal)
        write_json(final_dir / "block_seals" / f"{current_block}.json", seal)
        block_seals.append(seal)

    case_counts: Counter[str] = Counter()
    for claim in final_claims:
        case_counts.update(str(value) for value in claim.get("claim_case_study_ids") or [])
    assert_formal_kg_unchanged(manifest["formal_kg_baseline"])
    summary = {
        "schema_version": f"{FINAL_SCHEMA}.summary.v1",
        "completed_at": utc_now(),
        "workflow_version": WORKFLOW_VERSION,
        "target_papers": int(target["target_unique_papers"]),
        "final_complete_papers": len(final_papers),
        "final_claims": len(final_claims),
        "zero_claim_papers": sum(bool(row["zero_claim"]) for row in final_papers),
        "adjudicated_papers": len(adjudications),
        "secondary_adjudication_reviews": len(secondary_by_key),
        "secondary_exact_samples": len(sample_decisions),
        "block_seals": len(block_seals),
        "open_disagreements": 0,
        "case_study_claim_counts": dict(sorted(case_counts.items())),
        "artifacts": {
            "raw_items": {
                "path": str(final_dir / "raw_items.jsonl"),
                "sha256": sha256_file(final_dir / "raw_items.jsonl"),
            },
            "paper_results": {
                "path": str(final_dir / "paper_results.jsonl"),
                "sha256": sha256_file(final_dir / "paper_results.jsonl"),
            },
            "claims": {
                "path": str(final_dir / "claims.jsonl"),
                "sha256": sha256_file(final_dir / "claims.jsonl"),
            },
            "block_seal_inventory_sha256": canonical_hash(block_seals),
        },
        "formal_kg_baseline": manifest["formal_kg_baseline"],
        "source_artifact_inventory_sha256": str(
            manifest["source_artifact_inventory_sha256"]
        ),
        "target_source_seal_inventory_sha256": str(
            manifest["target_source_seal_inventory_sha256"]
        ),
        "formal_kg_mutated": False,
        "complete": True,
    }
    summary["final_summary_sha256"] = canonical_hash(summary)
    write_json(final_dir / "FINAL_SUMMARY.json", summary)
    return summary


def stage_progress(campaign_dir: Path, stage: str) -> dict[str, Any]:
    root = stage_root(campaign_dir, stage)
    if not root.is_dir():
        return {"stage": stage, "initialized": False, "processed": 0, "target": 0}
    manifest = read_json(root / "MANIFEST.json")
    processed = 0
    complete_workers = 0
    for worker in manifest.get("workers") or []:
        directory = root / "workers" / str(worker["worker_id"])
        count = len(read_jsonl(directory / "decisions.jsonl"))
        processed += count
        complete_workers += int(count == int(worker["task_count"]))
    return {
        "stage": stage,
        "initialized": True,
        "processed": processed,
        "target": int(manifest["task_count"]),
        "completion_percent": round(
            processed * 100 / int(manifest["task_count"]), 6
        ) if int(manifest["task_count"]) else 100.0,
        "complete_workers": complete_workers,
        "workers": len(manifest.get("workers") or []),
    }


def postreview_status(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    extraction = extraction_status(campaign_dir)
    root = postreview_dir(campaign_dir)
    final_path = root / "final" / "FINAL_SUMMARY.json"
    return {
        "schema_version": f"{POSTREVIEW_SCHEMA}.status.v1",
        "extraction": extraction,
        "postreview_initialized": (root / "MANIFEST.json").is_file(),
        "adjudication": stage_progress(campaign_dir, "adjudication"),
        "secondary": stage_progress(campaign_dir, "secondary"),
        "final": read_json(final_path) if final_path.is_file() else {"complete": False},
        "formal_kg_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "prepare-secondary", "finalize", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--campaign-dir", type=Path, required=True)
    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--campaign-dir", type=Path, required=True)
    append_parser.add_argument(
        "--stage", choices=("adjudication", "secondary"), required=True
    )
    append_parser.add_argument("--worker-id", required=True)
    append_parser.add_argument("--input", type=Path, required=True)
    append_parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()

    if args.command == "init":
        result = init_postreview(args.campaign_dir)
    elif args.command == "prepare-secondary":
        result = prepare_secondary(args.campaign_dir)
    elif args.command == "append":
        result = append_checkpoint(
            args.campaign_dir,
            args.stage,
            args.worker_id,
            args.input,
            args.thread_id,
        )
    elif args.command == "finalize":
        result = finalize_postreview(args.campaign_dir)
    else:
        result = postreview_status(args.campaign_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
