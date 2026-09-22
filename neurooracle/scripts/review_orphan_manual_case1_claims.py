"""Audit quarantined manual Case-1 claims without mutating the formal KG.

The historical extraction pipelines appended every extracted candidate to the
JSONL store even when ``ingest_claims`` rejected it.  The full-graph re-audit
therefore quarantined rows that had no corresponding formal graph node.  This
script performs a fresh, content-addressed recovery audit for the manually
curated subset using the frozen peer-17 membership policy.

The workflow is deliberately staged:

1. collect the quarantined manual rows and their cached abstracts;
2. run a blind primary review and blind cross-QA review;
3. adjudicate only disagreements;
4. validate every decision and emit retained/excluded staging files;
5. never write ``full_v2`` canonical files.

API responses are append-only and resumable.  Final staged claims carry the
current ``case_study_reaudit_contract.v2`` seal, so a later injector can fail
closed before any KG mutation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable

from openai import OpenAI

from neurooracle.src.case_study_membership_contract import (
    build_final_scope_reaudit,
    validate_final_scope_reaudit,
)
from neurooracle.src.case_study_membership_policy import (
    GATE_NAMES,
    RUBRIC_SHA256,
    RUBRIC_VERSION,
    case_study_policy_prompt,
    validate_scope_decision,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


REPO = Path(__file__).resolve().parents[2]
DEFAULT_QUARANTINE = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "full_graph_v3"
    / "quarantine"
    / "extracted_orphans.jsonl"
)
DEFAULT_FORMAL_CLAIMS = (
    REPO / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "orphan_manual_case1_recovery_v1"
)
DEFAULT_ABSTRACT_CACHES = (
    REPO / "neurooracle" / "data" / "full_snapshot_v2" / "abstract_cache.jsonl",
    REPO
    / "neurooracle"
    / "data"
    / "archive"
    / "full_v2_history_DO_NOT_USE_20260801_1712"
    / "abstract_cache.jsonl",
)

SCHEMA_VERSION = "orphan_manual_case1_recovery.v1"
REVIEWER_VERSION = "peer17_orphan_recovery.primary_qa.v1"
WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def normalize_doi(value: object) -> str:
    doi = normalize_text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


def paper_key(claim: dict[str, Any]) -> str:
    paper = claim.get("source_paper") or {}
    pmid = normalize_text(paper.get("pmid"))
    if pmid:
        return f"pmid:{pmid}"
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = normalize_text(paper.get("title"))
    year = str(paper.get("year") or claim.get("year") or "")
    if title:
        return f"title_year:{title}|{year}"
    return f"claim_fallback:{claim.get('id') or ''}"


def evidence_signature(claim: dict[str, Any]) -> tuple[str, ...]:
    return (
        paper_key(claim),
        normalize_text(claim.get("subject_name")),
        normalize_text(claim.get("predicate")),
        normalize_text(claim.get("object_name")),
        normalize_text(claim.get("raw_text")),
    )


def load_manual_orphans(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            envelope = json.loads(line)
            row = envelope.get("row") if isinstance(envelope, dict) else None
            if not isinstance(row, dict):
                raise ValueError(f"invalid quarantine row at line {line_number}")
            if not bool((row.get("metadata") or {}).get("manual_curation")):
                continue
            claim_id = str(row.get("id") or "")
            if not claim_id or claim_id in seen:
                raise ValueError(f"invalid/duplicate manual claim ID: {claim_id!r}")
            seen.add(claim_id)
            copied = deepcopy(row)
            copied["_quarantine_source_line"] = envelope.get("source_line")
            rows.append(copied)
    return rows


def load_abstracts(
    caches: Iterable[Path], wanted_pmids: set[str]
) -> tuple[dict[str, str], dict[str, str]]:
    abstracts: dict[str, str] = {}
    sources: dict[str, str] = {}
    for cache in caches:
        if not cache.exists():
            continue
        with cache.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper = row.get("paper") or {}
                pmid = str(row.get("pmid") or paper.get("pmid") or "").strip()
                if pmid not in wanted_pmids or pmid in abstracts:
                    continue
                abstract = str(row.get("abstract") or paper.get("abstract") or "").strip()
                if abstract:
                    abstracts[pmid] = abstract
                    sources[pmid] = str(cache.resolve())
        if len(abstracts) == len(wanted_pmids):
            break
    return abstracts, sources


def load_formal_context(
    path: Path, wanted_papers: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]], set[tuple[str, ...]]]:
    claims: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paper_labels: dict[str, set[str]] = defaultdict(set)
    signatures: set[tuple[str, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = paper_key(row)
            if key not in wanted_papers:
                continue
            signatures.add(evidence_signature(row))
            paper_labels[key].update(row.get("claim_case_study_ids") or [])
            if len(claims[key]) < 24:
                claims[key].append(
                    {
                        "claim_id": row.get("id"),
                        "subject": row.get("subject_name"),
                        "predicate": row.get("predicate"),
                        "object": row.get("object_name"),
                        "evidence": row.get("raw_text"),
                        "case_study_ids": row.get("claim_case_study_ids") or [],
                    }
                )
    return claims, paper_labels, signatures


def claim_for_prompt(claim: dict[str, Any]) -> dict[str, Any]:
    paper = claim.get("source_paper") or {}
    evidence = claim.get("evidence") or {}
    metadata = claim.get("metadata") or {}
    return {
        "claim_id": claim.get("id"),
        "subject": claim.get("subject_name"),
        "predicate": claim.get("predicate"),
        "object": claim.get("object_name"),
        "raw_evidence": claim.get("raw_text"),
        "study_type": evidence.get("study_type"),
        "methodology": evidence.get("methodology"),
        "reported_confidence": claim.get("confidence"),
        "subject_type": metadata.get("subject_type"),
        "object_type": metadata.get("object_type"),
        "paper_title": paper.get("title"),
        "paper_year": paper.get("year"),
        "pmid": paper.get("pmid"),
        "doi": paper.get("doi"),
    }


def paper_payload(
    key: str,
    claims: list[dict[str, Any]],
    abstracts: dict[str, str],
    formal_context: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    paper = claims[0].get("source_paper") or {}
    pmid = str(paper.get("pmid") or "")
    return {
        "paper_key": key,
        "paper": {
            "pmid": paper.get("pmid"),
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "journal": paper.get("journal"),
        },
        "abstract": abstracts.get(pmid, ""),
        "claims_to_review": [claim_for_prompt(claim) for claim in claims],
        "existing_formal_claims_from_same_paper": formal_context.get(key, []),
    }


def review_prompt(payload: dict[str, Any], stage: str) -> str:
    gate_template = {name: False for name in GATE_NAMES}
    return f"""You are performing a {stage} scientific claim audit for NeuroOracle.

First decide whether each quarantined claim is a valid source-linked claim worth
retaining in the general neuroscience corpus. Retain only when the supplied
evidence sentence, title, or abstract semantically supports the complete
subject-predicate-object assertion. Exclude aims, methods alone, generic
background, speculation/future work, unsupported endpoints, and paraphrases
that overstate the source. A review/meta-analysis may be retained when its
abstract explicitly reports or synthesizes the asserted finding.

IMPORTANT OVERRIDE: an endpoint is not invalid merely because it contains MRI,
fMRI, DTI, PET, EEG, MEG, imaging, activation, connectivity, volume, thickness,
or a classifier. Concrete measured imaging/neural findings are legitimate.

For every retained claim, independently assign all applicable formal Case Study
IDs under the exact frozen policy below. Membership is claim-level and
non-exclusive. Same-paper context may clarify terminology, but it must not
donate an otherwise missing claim relation. Case 2 is derived only from a
direct genetic/pathway-to-neural or baseline-neural-to-later-outcome component.
Do not trust historical labels,
search labels, the filename, or the phrase "manual Case 1" as membership
evidence.

{case_study_policy_prompt(extraction=False)}

Return one JSON object only, with this exact structure:
{{
  "paper_key": {json.dumps(payload['paper_key'])},
  "decisions": [
    {{
      "claim_id": "exact supplied claim ID",
      "retain": true,
      "retain_reason": "specific evidence-grounded explanation",
      "case_study_ids": ["zero or more formal IDs"],
      "case_study_gates": {json.dumps(gate_template)},
      "scope_confidence": 0.0,
      "decision_basis": "why these labels, including why tempting alternatives fail"
    }}
  ]
}}

Rules for the JSON:
- Return exactly one decision for every supplied claim, in supplied order.
- case_study_gates must contain exactly all {len(GATE_NAMES)} boolean gate keys.
- If retain=false, case_study_ids must be [] and every gate must be false.
- scope_confidence is confidence in the retain+scope decision, not the claim's
  reported statistical confidence.
- Do not emit markdown or commentary outside the JSON.

PAPER AND CLAIM PAYLOAD:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def adjudication_prompt(
    payload: dict[str, Any],
    primary: dict[str, dict[str, Any]],
    qa: dict[str, dict[str, Any]],
    disputed_ids: list[str],
) -> str:
    reduced = deepcopy(payload)
    reduced["claims_to_review"] = [
        claim
        for claim in payload["claims_to_review"]
        if claim["claim_id"] in set(disputed_ids)
    ]
    comparisons = [
        {
            "claim_id": claim_id,
            "primary": primary[claim_id],
            "cross_qa": qa[claim_id],
        }
        for claim_id in disputed_ids
    ]
    return (
        review_prompt(reduced, "final blinded-disagreement adjudication")
        + "\nThe two independent reviewers disagreed. Resolve the disagreements "
        "from source evidence and the frozen policy; do not decide by voting.\n"
        + json.dumps(comparisons, ensure_ascii=False, indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("review response is not a JSON object")
    return value


def validate_review(
    value: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    if value.get("paper_key") != payload["paper_key"]:
        raise ValueError("review paper_key mismatch")
    expected = [claim["claim_id"] for claim in payload["claims_to_review"]]
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("review decisions must be a list")
    actual = [str(item.get("claim_id") or "") for item in decisions if isinstance(item, dict)]
    if actual != expected:
        raise ValueError(f"review claim inventory/order mismatch: {actual!r} != {expected!r}")

    normalized: list[dict[str, Any]] = []
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("every review decision must be an object")
        retain = item.get("retain")
        if not isinstance(retain, bool):
            raise ValueError("retain must be a JSON boolean")
        labels = item.get("case_study_ids")
        gates = item.get("case_study_gates")
        if not retain:
            if labels != []:
                raise ValueError("excluded claim must have no Case Study IDs")
            if not isinstance(gates, dict) or set(gates) != set(GATE_NAMES):
                raise ValueError("excluded claim has malformed gates")
            if any(gates.values()):
                raise ValueError("excluded claim gates must all be false")
        decision = validate_scope_decision(labels, gates)
        confidence = float(item.get("scope_confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("scope_confidence must be in 0..1")
        retain_reason = str(item.get("retain_reason") or "").strip()
        decision_basis = str(item.get("decision_basis") or "").strip()
        if len(retain_reason) < 12 or len(decision_basis) < 12:
            raise ValueError("review reasons are too short")
        normalized.append(
            {
                "claim_id": str(item["claim_id"]),
                "retain": retain,
                "retain_reason": retain_reason,
                "case_study_ids": list(decision.labels),
                "case_study_gates": dict(decision.gates),
                "scope_confidence": confidence,
                "decision_basis": decision_basis,
            }
        )
    return {"paper_key": payload["paper_key"], "decisions": normalized}


def api_review(
    *,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    payload: dict[str, Any],
    prompt: str,
    max_attempts: int,
) -> dict[str, Any]:
    backoff = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=prompt,
                max_output_tokens=16000,
                store=False,
            )
            parsed = extract_json_object(response.output_text)
            reviewed = validate_review(parsed, payload)
            reviewed.update(
                {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "response_id": getattr(response, "id", ""),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "reviewed_at": utc_now(),
                }
            )
            return reviewed
        except Exception as exc:  # network and strict response validation
            last_error = exc
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
    raise RuntimeError(
        f"review failed for {payload['paper_key']} after {max_attempts} attempts: {last_error}"
    )


def load_review_file(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("paper_key") or "")
            if not key or key in records:
                raise ValueError(f"invalid duplicate review at {path}:{line_number}")
            records[key] = record
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(compact_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def run_review_pass(
    *,
    name: str,
    path: Path,
    payloads: dict[str, dict[str, Any]],
    client_factory,
    model: str,
    reasoning_effort: str,
    max_workers: int,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    completed = load_review_file(path)
    pending = [key for key in payloads if key not in completed]
    if not pending:
        return completed

    local = threading.local()

    def worker(key: str) -> tuple[str, dict[str, Any]]:
        if not hasattr(local, "client"):
            local.client = client_factory()
        payload = payloads[key]
        result = api_review(
            client=local.client,
            model=model,
            reasoning_effort=reasoning_effort,
            payload=payload,
            prompt=review_prompt(payload, name),
            max_attempts=max_attempts,
        )
        result["review_stage"] = name
        return key, result

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                _, record = future.result()
                append_record(path, record)
                completed[key] = record
                done = len(completed)
                if done % 20 == 0 or done == len(payloads):
                    print(f"{name}: {done}/{len(payloads)} papers", flush=True)
            except Exception as exc:
                failures.append(f"{key}: {exc}")
    if failures:
        raise RuntimeError(f"{name} failures ({len(failures)}): {failures[:5]}")
    return completed


def index_decisions(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["claim_id"]): item for item in record["decisions"]}


def decision_signature(decision: dict[str, Any]) -> tuple[object, ...]:
    return (
        bool(decision["retain"]),
        tuple(decision["case_study_ids"]),
        tuple((name, decision["case_study_gates"][name]) for name in GATE_NAMES),
    )


def run_adjudications(
    *,
    path: Path,
    payloads: dict[str, dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    qa: dict[str, dict[str, Any]],
    client_factory,
    model: str,
    reasoning_effort: str,
    max_workers: int,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    completed = load_review_file(path)
    disputed: dict[str, list[str]] = {}
    for key in payloads:
        first = index_decisions(primary[key])
        second = index_decisions(qa[key])
        ids = [
            claim_id
            for claim_id in first
            if decision_signature(first[claim_id]) != decision_signature(second[claim_id])
        ]
        if ids:
            disputed[key] = ids
    pending = [key for key in disputed if key not in completed]
    if not pending:
        return completed

    local = threading.local()

    def worker(key: str) -> tuple[str, dict[str, Any]]:
        if not hasattr(local, "client"):
            local.client = client_factory()
        full_payload = payloads[key]
        wanted = disputed[key]
        reduced = deepcopy(full_payload)
        reduced["claims_to_review"] = [
            item for item in full_payload["claims_to_review"] if item["claim_id"] in set(wanted)
        ]
        first = index_decisions(primary[key])
        second = index_decisions(qa[key])
        result = api_review(
            client=local.client,
            model=model,
            reasoning_effort=reasoning_effort,
            payload=reduced,
            prompt=adjudication_prompt(full_payload, first, second, wanted),
            max_attempts=max_attempts,
        )
        result["review_stage"] = "adjudication"
        return key, result

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                _, record = future.result()
                append_record(path, record)
                completed[key] = record
                done = len(completed)
                if done % 20 == 0 or done == len(disputed):
                    print(f"adjudication: {done}/{len(disputed)} papers", flush=True)
            except Exception as exc:
                failures.append(f"{key}: {exc}")
    if failures:
        raise RuntimeError(f"adjudication failures ({len(failures)}): {failures[:5]}")
    return completed


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
            count += 1
    os.replace(temp, path)
    return count


def finalize(
    *,
    output_dir: Path,
    manual_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    qa: dict[str, dict[str, Any]],
    adjudications: dict[str, dict[str, Any]],
    existing_paper_labels: dict[str, set[str]],
    formal_signatures: set[tuple[str, ...]],
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    rows_by_id = {str(row["id"]): row for row in manual_rows}
    final_decisions: dict[str, dict[str, Any]] = {}
    consensus = 0
    adjudicated = 0
    for key in payloads:
        first = index_decisions(primary[key])
        second = index_decisions(qa[key])
        third = index_decisions(adjudications[key]) if key in adjudications else {}
        for claim_id in first:
            if decision_signature(first[claim_id]) == decision_signature(second[claim_id]):
                chosen = deepcopy(first[claim_id])
                chosen["final_source"] = "primary_qa_consensus"
                chosen["scope_confidence"] = min(
                    float(first[claim_id]["scope_confidence"]),
                    float(second[claim_id]["scope_confidence"]),
                )
                chosen["decision_basis"] = (
                    f"Blind primary/QA consensus. Primary: {first[claim_id]['decision_basis']} "
                    f"QA: {second[claim_id]['decision_basis']}"
                )
                chosen["retain_reason"] = (
                    f"Primary: {first[claim_id]['retain_reason']} "
                    f"QA: {second[claim_id]['retain_reason']}"
                )
                consensus += 1
            else:
                if claim_id not in third:
                    raise ValueError(f"missing adjudication for disputed claim {claim_id}")
                chosen = deepcopy(third[claim_id])
                chosen["final_source"] = "adjudicated"
                adjudicated += 1
            final_decisions[claim_id] = chosen
    if set(final_decisions) != set(rows_by_id):
        raise ValueError("final decision inventory differs from manual orphan inventory")

    retained_ids = {
        claim_id for claim_id, decision in final_decisions.items() if decision["retain"]
    }
    duplicate_ids = {
        claim_id
        for claim_id in retained_ids
        if evidence_signature(rows_by_id[claim_id]) in formal_signatures
    }
    retained_ids -= duplicate_ids

    new_paper_labels: dict[str, set[str]] = defaultdict(set)
    for claim_id in retained_ids:
        row = rows_by_id[claim_id]
        new_paper_labels[paper_key(row)].update(final_decisions[claim_id]["case_study_ids"])

    retained_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()
    for row in manual_rows:
        claim_id = str(row["id"])
        decision = deepcopy(final_decisions[claim_id])
        is_duplicate = claim_id in duplicate_ids
        final_retain = bool(decision["retain"]) and not is_duplicate
        decision_record = {
            "schema_version": SCHEMA_VERSION,
            "reviewer_version": REVIEWER_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": RUBRIC_SHA256,
            "claim_id": claim_id,
            "paper_key": paper_key(row),
            **decision,
            "retain": final_retain,
            "formal_exact_duplicate": is_duplicate,
        }
        decision_rows.append(decision_record)

        if not final_retain:
            excluded_rows.append(
                {
                    "decision": decision_record,
                    "claim": {k: v for k, v in row.items() if not k.startswith("_")},
                }
            )
            continue

        labels = list(decision["case_study_ids"])
        paper_labels = set(existing_paper_labels.get(paper_key(row), set()))
        paper_labels.update(new_paper_labels.get(paper_key(row), set()))
        ordered_paper_labels = [label for label in CASE_STUDY_IDS if label in paper_labels]
        staged = {k: deepcopy(v) for k, v in row.items() if not k.startswith("_")}
        staged["claim_case_study_ids"] = labels
        staged["paper_case_study_ids"] = ordered_paper_labels
        metadata = staged.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            staged["metadata"] = metadata
        metadata["claim_case_study_ids"] = labels
        metadata["paper_case_study_ids"] = ordered_paper_labels
        metadata["case_study_membership_schema_version"] = "case_study_membership.v2"

        context = payloads[paper_key(row)]
        context_hash = hashlib.sha256(
            compact_json(context).encode("utf-8")
        ).hexdigest()
        staged["scope_reaudit"] = build_final_scope_reaudit(
            staged,
            labels=labels,
            gates=decision["case_study_gates"],
            confidence=float(decision["scope_confidence"]),
            decision_basis=str(decision["decision_basis"]),
            scope_context_sha256=context_hash,
            reviewer_id=f"{model}.primary_qa_adjudicated",
            reasoning_effort=reasoning_effort,
            reviewed_at=utc_now(),
            source_kind="cached_abstract_plus_claim_and_same_paper_context",
        )
        validate_final_scope_reaudit(staged)
        retained_rows.append(staged)
        case_counts.update(labels)

    decision_rows.sort(key=lambda item: rows_by_id[item["claim_id"]]["_quarantine_source_line"])
    retained_rows.sort(key=lambda item: rows_by_id[item["id"]]["_quarantine_source_line"])
    excluded_rows.sort(
        key=lambda item: rows_by_id[item["claim"]["id"]]["_quarantine_source_line"]
    )
    write_jsonl(output_dir / "final_decisions.jsonl", decision_rows)
    write_jsonl(output_dir / "retained_claims.staged.jsonl", retained_rows)
    write_jsonl(output_dir / "excluded_claims.jsonl", excluded_rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "staged_not_injected",
        "formal_kg_mutated": False,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "manual_orphans_reviewed": len(manual_rows),
        "primary_qa_consensus": consensus,
        "adjudicated": adjudicated,
        "retained": len(retained_rows),
        "excluded": len(excluded_rows),
        "formal_exact_duplicates_excluded": len(duplicate_ids),
        "retained_case_study_counts": {
            label: case_counts[label] for label in CASE_STUDY_IDS
        },
        "outputs": {
            "final_decisions": str((output_dir / "final_decisions.jsonl").resolve()),
            "retained_claims": str(
                (output_dir / "retained_claims.staged.jsonl").resolve()
            ),
            "excluded_claims": str((output_dir / "excluded_claims.jsonl").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--formal-claims", type=Path, default=DEFAULT_FORMAL_CLAIMS)
    parser.add_argument("--abstract-cache", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_rows = load_manual_orphans(args.quarantine.resolve())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manual_rows:
        grouped[paper_key(row)].append(row)
    wanted_pmids = {
        str((row.get("source_paper") or {}).get("pmid") or "")
        for row in manual_rows
        if (row.get("source_paper") or {}).get("pmid")
    }
    caches = tuple(args.abstract_cache) if args.abstract_cache else DEFAULT_ABSTRACT_CACHES
    abstracts, abstract_sources = load_abstracts(caches, wanted_pmids)
    missing = sorted(wanted_pmids - set(abstracts))
    if missing:
        raise SystemExit(f"missing cached abstracts for {len(missing)} PMIDs: {missing[:10]}")

    formal_context, existing_paper_labels, formal_signatures = load_formal_context(
        args.formal_claims.resolve(), set(grouped)
    )
    payloads = {
        key: paper_payload(key, grouped[key], abstracts, formal_context)
        for key in sorted(grouped)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "review_in_progress",
        "formal_kg_mutated": False,
        "manual_claims": len(manual_rows),
        "papers": len(payloads),
        "abstracts": len(abstracts),
        "abstract_sources": dict(Counter(abstract_sources.values())),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_workers": args.max_workers,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "quarantine": str(args.quarantine.resolve()),
        "formal_claims": str(args.formal_claims.resolve()),
    }
    write_json(output_dir / "manifest.json", manifest)

    def client_factory() -> OpenAI:
        return OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=0,
        )

    primary = run_review_pass(
        name="blind_primary",
        path=output_dir / "primary_reviews.jsonl",
        payloads=payloads,
        client_factory=client_factory,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
    )
    qa = run_review_pass(
        name="blind_cross_qa",
        path=output_dir / "qa_reviews.jsonl",
        payloads=payloads,
        client_factory=client_factory,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
    )
    adjudications = run_adjudications(
        path=output_dir / "adjudications.jsonl",
        payloads=payloads,
        primary=primary,
        qa=qa,
        client_factory=client_factory,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
    )
    summary = finalize(
        output_dir=output_dir,
        manual_rows=manual_rows,
        payloads=payloads,
        primary=primary,
        qa=qa,
        adjudications=adjudications,
        existing_paper_labels=existing_paper_labels,
        formal_signatures=formal_signatures,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
