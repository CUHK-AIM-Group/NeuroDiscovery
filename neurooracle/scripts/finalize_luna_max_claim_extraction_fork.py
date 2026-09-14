"""Fail-closed finalizer for one Luna-max extraction fork.

This entry point is deliberately source based.  The campaign runs on the
bundled Python 3.12 runtime, so loading a 3.11/3.12 bytecode wrapper here is
unsafe (and can recursively load itself).  The finalizer binds worker output
to the immutable task queue, creates deterministic claim IDs and v4 scope
seals, and never touches the formal KG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The claim extractor imports these packages even though finalization is offline.
try:  # pragma: no cover - depends on the bundled runtime
    import httpx  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    httpx = types.ModuleType("httpx")
    httpx.Client = object  # type: ignore[attr-defined]
    sys.modules["httpx"] = httpx
try:  # pragma: no cover
    import openai  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    openai = types.ModuleType("openai")
    openai.OpenAI = object  # type: ignore[attr-defined]
    sys.modules["openai"] = openai

from neurooracle.src.case_study_membership_contract import (  # noqa: E402
    AUDIT_CONTRACT_VERSION,
    CASE_STUDY_REGISTRY_SHA256,
    build_final_scope_reaudit,
    source_text_sha256,
    validate_final_scope_reaudit,
)
from neurooracle.src.case_study_membership_policy import (  # noqa: E402
    GATE_NAMES,
    POLICY,
    validate_scope_decision,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS  # noqa: E402
from neurooracle.src.claim_extractor import ClaimExtractor, EXTRACTION_PROMPT  # noqa: E402
from neurooracle.src.schema import PaperRef  # noqa: E402


SCHEMA_VERSION = "neurooracle.luna_max_fork_claim_extraction_result.v2"
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
CORE_POLICY_FILES = (
    "case_studies.py",
    "case_study_scope.py",
    "case_study_membership_policy.py",
    "case_study_membership_contract.py",
    "CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_lf(path: Path) -> str:
    """Hash text after canonicalising platform line endings to LF."""

    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


# Backwards-compatible name used by early campaign tooling.
load_jsonl = read_jsonl


def paper_key(task: dict[str, Any]) -> str:
    return str(task.get("paper_key") or "").strip().lower()


def deterministic_claim_id(campaign: str, paper_key: str, source_hash: str, index: int) -> str:
    payload = f"{campaign}|{paper_key}|{source_hash}|{index}".encode("utf-8")
    return "CLM:" + hashlib.sha256(payload).hexdigest()[:32]


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _manifest(fork_dir: Path) -> dict[str, Any]:
    path = fork_dir / "TASK_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def _task_rows(fork_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(fork_dir / "task.jsonl")
    if not rows:
        raise ValueError("task.jsonl is empty")
    indices = [int(row.get("queue_index", -1)) for row in rows]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("task queue indices are not contiguous")
    return rows


def _validate_frozen_state(
    fork_dir: Path,
    manifest: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind a shard to the frozen semantic contract, tolerating only CRLF drift."""

    campaign_dir = fork_dir.parent.parent
    contract_path = campaign_dir / "FROZEN_CONTRACT.json"
    contract = read_json(contract_path)
    expected_contract_sha = str(manifest.get("frozen_contract_sha256") or "")
    if sha256_file(contract_path) != expected_contract_sha:
        raise ValueError("frozen contract hash mismatch")
    exact = {
        "campaign_version": manifest.get("campaign_version"),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": 0.0,
        "model_locked": True,
        "input_policy": "complete_abstract_no_truncation",
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "rubric_version": POLICY.version,
        "rubric_sha256": POLICY.rubric_sha256,
        "prompt_sha256": source_text_sha256(EXTRACTION_PROMPT),
        "case_study_registry_sha256": CASE_STUDY_REGISTRY_SHA256,
        "formal_kg_mutation_permitted": False,
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise ValueError(f"frozen contract {key} drift")
    if list(contract.get("case_study_ids") or []) != list(CASE_STUDY_IDS):
        raise ValueError("frozen Case Study registry drift")
    if list(contract.get("gate_names") or []) != list(GATE_NAMES):
        raise ValueError("frozen gate registry drift")
    if manifest.get("formal_kg_mutation_permitted") is not False:
        raise ValueError("task manifest permits formal KG mutation")
    for key in ("campaign_version", "model", "reasoning_effort", "temperature", "model_locked", "prompt_sha256"):
        if manifest.get(key) != contract.get(key):
            raise ValueError(f"task manifest {key} drift")

    task_path = fork_dir / "task.jsonl"
    if str(manifest.get("task_sha256") or "") != sha256_file(task_path):
        raise ValueError("task hash mismatch")
    if int(manifest.get("paper_count") or -1) != len(tasks):
        raise ValueError("task paper count mismatch")
    start = int(manifest.get("queue_start") or -1)
    end = int(manifest.get("queue_end") or -1)
    if start < 1 or end - start + 1 != len(tasks):
        raise ValueError("task queue range mismatch")

    seen_keys: set[str] = set()
    for offset, task in enumerate(tasks):
        queue_index = start + offset
        if int(task.get("queue_index", -1)) != queue_index:
            raise ValueError(f"task queue mismatch at {queue_index}")
        if task.get("campaign_version") != contract.get("campaign_version"):
            raise ValueError(f"task campaign mismatch at {queue_index}")
        if task.get("source_kind") != "complete_abstract":
            raise ValueError(f"task source kind mismatch at {queue_index}")
        abstract = str(task.get("abstract") or "")
        if len(abstract) < 200 or int(task.get("abstract_length") or -1) != len(abstract):
            raise ValueError(f"task abstract length mismatch at {queue_index}")
        if source_text_sha256(abstract) != str(task.get("source_context_sha256") or ""):
            raise ValueError(f"task source hash mismatch at {queue_index}")
        key = paper_key(task)
        if not key or ":" not in key or key in seen_keys:
            raise ValueError(f"task paper key missing or duplicated at {queue_index}")
        seen_keys.add(key)

    expected_policy_hashes = {
        Path(str(row.get("path") or "")).name: str(row.get("sha256") or "")
        for row in contract.get("policy_files") or []
        if isinstance(row, dict)
    }
    for name in CORE_POLICY_FILES:
        path = REPO_ROOT / "neurooracle" / "src" / name
        expected = expected_policy_hashes.get(name, "")
        if not path.is_file() or sha256_file_lf(path) != expected:
            raise ValueError(f"semantic policy drift: {name}")

    attestation_path = campaign_dir / "RECOVERY_ATTESTATION.json"
    if attestation_path.is_file():
        attestation = read_json(attestation_path)
        if attestation.get("status") != "verified_semantic_equivalence":
            raise ValueError("campaign recovery attestation is not verified")
        if attestation.get("frozen_contract_sha256") != expected_contract_sha:
            raise ValueError("campaign recovery attestation contract mismatch")
        current_files = attestation.get("current_files") or {}
        for name, path in {
            "claim_extractor": REPO_ROOT / "neurooracle/src/claim_extractor.py",
            "finalizer": Path(__file__).resolve(),
        }.items():
            expected = str((current_files.get(name) or {}).get("sha256") or "")
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"campaign recovery attestation drift: {name}")
    return contract


def _paper_ref(task: dict[str, Any]) -> PaperRef:
    paper = task.get("paper") or {}
    if not isinstance(paper, dict):
        paper = {}
    year = paper.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    return PaperRef(
        pmid=str(paper.get("pmid") or ""),
        doi=str(paper.get("doi") or ""),
        title=str(paper.get("title") or ""),
        authors=str(paper.get("authors") or ""),
        year=year,
        journal=str(paper.get("journal") or ""),
    )


def _validate_raw_row(row: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    expected_index = int(task["queue_index"])
    if int(row.get("queue_index", -1)) != expected_index:
        raise ValueError(f"queue_index mismatch at {expected_index}")
    expected_key = str(task.get("paper_key") or "")
    if str(row.get("paper_key") or "") != expected_key:
        raise ValueError(f"paper_key mismatch at {expected_index}")
    expected_hash = str(task.get("source_context_sha256") or "")
    if str(row.get("source_context_sha256") or "") != expected_hash:
        raise ValueError(f"source hash mismatch at {expected_index}")
    items = row.get("items")
    if not isinstance(items, list):
        raise ValueError(f"items must be a list at {expected_index}")
    if len(items) > 12:
        raise ValueError(f"too many claim items at {expected_index}: {len(items)}")
    abstract = str(task.get("abstract") or "")
    canonical_items: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"claim item is not an object at {expected_index}:{item_index}")
        if bool(item.get("negated", False)):
            raise ValueError(f"negated claim at {expected_index}:{item_index}")
        for field in ("subject", "predicate", "object", "raw_sentence"):
            if not str(item.get(field) or "").strip():
                raise ValueError(f"missing {field} at {expected_index}:{item_index}")
        raw_sentence = str(item["raw_sentence"])
        if raw_sentence not in abstract:
            raise ValueError(f"raw_sentence is not verbatim at {expected_index}:{item_index}")
        spans = item.get("scope_evidence_spans") or []
        if not isinstance(spans, list):
            raise ValueError(f"scope_evidence_spans must be a list at {expected_index}:{item_index}")
        if any(not isinstance(span, str) or not span.strip() for span in spans):
            raise ValueError(f"scope_evidence_spans contains a non-string/empty span at {expected_index}:{item_index}")
        if item.get("case_study_ids") and not spans:
            raise ValueError(f"labeled claim has no scope span at {expected_index}:{item_index}")
        for span in spans:
            if _normalized(span) and _normalized(span) not in _normalized(abstract):
                raise ValueError(f"scope span is not verbatim at {expected_index}:{item_index}")
        basis = str(item.get("scope_decision_basis") or "").strip()
        if len(basis) < 12:
            raise ValueError(f"scope decision basis is too short at {expected_index}:{item_index}")
        try:
            confidence = float(item.get("scope_confidence"))
        except (TypeError, ValueError):
            raise ValueError(f"scope confidence is not numeric at {expected_index}:{item_index}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"scope confidence outside 0..1 at {expected_index}:{item_index}")
        gates = item.get("case_study_gates")
        if not isinstance(gates, dict) or set(gates) != set(GATE_NAMES):
            raise ValueError(f"exactly the frozen nine gates are required at {expected_index}:{item_index}")
        if any(not isinstance(gates[name], bool) for name in GATE_NAMES):
            raise ValueError(f"all frozen gates must be booleans at {expected_index}:{item_index}")
        decision = validate_scope_decision(item.get("case_study_ids"), gates)
        canonical = dict(item)
        canonical["case_study_ids"] = list(decision.labels)
        canonical["case_study_gates"] = dict(decision.gates)
        canonical_items.append(canonical)
    return canonical_items


def _claim_from_item(
    item: dict[str, Any],
    task: dict[str, Any],
    item_index: int,
    campaign: str,
    paper_labels: list[str],
    reviewed_at: str,
) -> dict[str, Any]:
    abstract = str(task.get("abstract") or "")
    source_hash = str(task["source_context_sha256"])
    decision = validate_scope_decision(item.get("case_study_ids"), item.get("case_study_gates"))
    extractor = object.__new__(ClaimExtractor)
    extractor.reasoning_effort = REASONING_EFFORT
    extractor.lock_model = True
    claim = extractor._item_to_claim(
        item,
        _paper_ref(task),
        paper_case_study_ids=paper_labels,
        claim_case_study_ids=list(decision.labels),
        case_study_gates=dict(decision.gates),
        scope_context_sha256=source_hash,
        scope_source_text=abstract,
        scope_source_kind="abstract",
        scope_reviewer_id=MODEL,
        extraction_claim_index=item_index,
        finalize_scope_audit=False,
    )
    if claim is None:
        raise ValueError("claim item was rejected by the strict converter")
    claim.id = deterministic_claim_id(campaign, str(task["paper_key"]), source_hash, item_index)
    metadata = dict(claim.metadata)
    metadata.update(
        {
            "scope_rubric_version": POLICY.version,
            "scope_assignment_stage": "combined_extraction_and_scope_audit",
            "scope_review_status": "final_complete",
            "scope_confidence": float(item["scope_confidence"]),
            "scope_decision_basis": str(item["scope_decision_basis"]).strip(),
            "extraction_profile": {
                "schema_version": "neurooracle.claim_extraction_profile.v1",
                "prompt_sha256": "a39335c5d678881521cf0d1f177c7436e71112231a657651d6d6af587c55c277",
                "scope_rubric_version": POLICY.version,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "temperature": 0.0,
                "max_tokens": 16384,
                "model_locked": True,
                "input_policy": "complete_abstract",
            },
        }
    )
    claim.metadata = metadata
    payload = claim.to_dict()
    seal = build_final_scope_reaudit(
        payload,
        labels=list(decision.labels),
        gates=dict(decision.gates),
        confidence=float(item["scope_confidence"]),
        decision_basis=str(item["scope_decision_basis"]).strip(),
        scope_context_sha256=source_hash,
        reviewer_id=MODEL,
        reasoning_effort=REASONING_EFFORT,
        reviewed_at=reviewed_at,
        source_kind="complete_abstract",
    )
    # The campaign manifest's creation time is the deterministic review time.
    claim.scope_reaudit = seal
    payload = claim.to_dict()
    validate_final_scope_reaudit(payload)
    return payload


def finalize_shard(fork_dir: Path, allow_partial: bool = False) -> dict[str, Any]:
    fork_dir = fork_dir.resolve()
    manifest = _manifest(fork_dir)
    tasks = _task_rows(fork_dir)
    contract = _validate_frozen_state(fork_dir, manifest, tasks)
    raw_path = fork_dir / "raw_items.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw = read_jsonl(raw_path)
    if len(raw) > len(tasks):
        raise ValueError("raw output has more rows than task queue")
    if not allow_partial and len(raw) != len(tasks):
        raise ValueError(f"final output incomplete: {len(raw)}/{len(tasks)} rows")
    task_by_index = {int(row["queue_index"]): row for row in tasks}
    if [int(row.get("queue_index", -1)) for row in raw] != list(task_by_index)[:len(raw)]:
        raise ValueError("raw output is not a contiguous task prefix")
    campaign = str(manifest.get("campaign_version") or "")
    paper_results: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    label_claim_counts: Counter[str] = Counter()
    label_papers: dict[str, set[str]] = {value: set() for value in CASE_STUDY_IDS}
    for row in raw:
        task = task_by_index[int(row["queue_index"])]
        items = _validate_raw_row(row, task)
        paper_labels = [
            value
            for value in CASE_STUDY_IDS
            if any(value in item["case_study_ids"] for item in items)
        ]
        row_claims = [
            _claim_from_item(
                item,
                task,
                i,
                campaign,
                paper_labels,
                str(manifest.get("created_at") or contract.get("created_at") or ""),
            )
            for i, item in enumerate(items)
        ]
        for claim in row_claims:
            claims.append(claim)
            for label in claim.get("claim_case_study_ids") or []:
                label_claim_counts[label] += 1
        for label in paper_labels:
            label_papers[label].add(str(task["paper_key"]))
        paper_results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "queue_index": int(task["queue_index"]),
                "paper_key": str(task["paper_key"]),
                "pmid": str((task.get("paper") or {}).get("pmid") or ""),
                "source_context_sha256": str(task["source_context_sha256"]),
                "raw_item_count": len(items),
                "claim_count": len(row_claims),
                "zero_claim": not bool(row_claims),
                "claim_ids": [str(claim["id"]) for claim in row_claims],
                "paper_case_study_ids": paper_labels,
                "validation_status": "final_complete",
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
            }
        )
    processed = len(paper_results)
    zero_claim = sum(1 for row in paper_results if row["zero_claim"])
    complete = processed == len(tasks)
    write_jsonl(fork_dir / "paper_results.jsonl", paper_results)
    write_jsonl(fork_dir / "claims.jsonl", claims)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "shard_id": manifest.get("shard_id"),
        "updated_at": utc_now(),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "task_papers": len(tasks),
        "processed_papers": processed,
        "remaining_papers": len(tasks) - processed,
        "completion_percent": round(100.0 * processed / len(tasks), 4),
        "complete": complete,
        "zero_claim_papers": zero_claim,
        "claims": len(claims),
        "formal_kg_mutated": False,
    }
    write_json(fork_dir / "progress.json", progress)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "shard_id": manifest.get("shard_id"),
        "complete": complete,
        "task_sha256": sha256_file(fork_dir / "task.jsonl"),
        "raw_items_sha256": sha256_file(raw_path),
        "paper_results_sha256": sha256_file(fork_dir / "paper_results.jsonl"),
        "claims_sha256": sha256_file(fork_dir / "claims.jsonl"),
        "processed_papers": processed,
        "total_papers": len(tasks),
        "claim_count": len(claims),
        "zero_claim_papers": zero_claim,
        "formal_kg_mutated": False,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "rubric_version": POLICY.version,
        "rubric_sha256": POLICY.rubric_sha256,
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "frozen_contract_sha256": str(manifest.get("frozen_contract_sha256") or ""),
        "prompt_sha256": source_text_sha256(EXTRACTION_PROMPT),
        "claim_id_policy": "sha256(campaign|paper_key|source_hash|raw_item_index)[:32]",
        "case_study_claim_counts": {value: int(label_claim_counts[value]) for value in CASE_STUDY_IDS},
        "case_study_paper_counts": {value: len(label_papers[value]) for value in CASE_STUDY_IDS},
        "seal_policy": "strict current v4 contract; deterministic Case 2 component routing; fail-closed",
    }
    if complete:
        write_json(fork_dir / "FINAL_SUMMARY.json", summary)
    return summary


# Compatibility alias retained for early worker instructions.
finalize = finalize_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(finalize_shard(args.fork_dir, allow_partial=args.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
