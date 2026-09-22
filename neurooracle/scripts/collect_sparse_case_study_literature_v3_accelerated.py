#!/usr/bin/env python3
"""Accelerated, deterministic literature collection for the KG-v3 top-up.

The legacy collector is intentionally conservative but serialises every
Europe PMC core-record request.  This companion keeps the same search queries,
normalisation rules, paper-identity gate, complete-abstract requirement, and
formal-KG fail-closed seal while parallelising only network I/O:

* query pages are fetched concurrently in deterministic round-robin waves;
* one coordinator performs every identity lookup and acceptance decision;
* 4-8 workers retrieve complete core records after the coordinator's preflight;
* records are committed in query/page/result order, never completion order;
* a shared per-source limiter/backoff circuit prevents 503 amplification.

It never writes a live legacy collection directory.  A stopped, page-aligned
partial collection may be copied into a new output directory with
``--carry-forward-dir``.  Unexhausted cursors are seeded from one or more prior
collections; high-recall queries that were never run start at ``*``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


from neurooracle.scripts import collect_sparse_case_study_literature_v3 as legacy
from neurooracle.src import academic_literature as academic
from neurooracle.src.paper_identity import (
    CandidatePaperDeduplicator,
    GlobalPaperIdentityIndex,
    paper_identity_aliases,
)


LOGGER = logging.getLogger("collect_sparse_case_study_literature_v3_accelerated")
SCHEMA = "sparse_case_study_literature_accelerated.v1"
DEFAULT_OUTPUT = (
    legacy.DATA_ROOT
    / "case_study_staging"
    / "kg_v3_100k_net_topup_supplement_42k_accelerated_20260814"
)
DEFAULT_SEED = (
    legacy.DATA_ROOT / "case_study_staging" / "kg_v3_sparse_case_studies_100k_20260811"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            yield value


def file_snapshot(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        stat = path.stat()
        result[str(path)] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hash_file(path),
        }
    return result


def assert_formal_stats_unchanged(
    paths: Iterable[Path], baseline: Mapping[str, Mapping[str, object]]
) -> None:
    """Cheap mid-run guard; full hashes are recomputed by the final seal."""

    for raw_path in paths:
        path = Path(raw_path).resolve()
        expected = baseline.get(str(path))
        if expected is None:
            raise RuntimeError(f"formal baseline lacks file: {path}")
        stat = path.stat()
        if (
            stat.st_size != int(expected["size_bytes"])
            or stat.st_mtime_ns != int(expected["mtime_ns"])
        ):
            raise RuntimeError(
                f"formal KG changed during accelerated collection: {path}"
            )


@dataclass(frozen=True)
class QueryPlan:
    rank: int
    target: legacy.Target
    query: legacy.SearchQuery
    key: str
    seed: dict[str, Any] | None
    seed_origin: str


@dataclass(frozen=True)
class PageFetch:
    plan: QueryPlan
    cursor: str
    payload: dict[str, Any] | None
    error: str = ""


@dataclass(frozen=True)
class CandidateTask:
    candidate: dict[str, Any]
    target: legacy.Target
    query: legacy.SearchQuery
    result_position: int
    extra_routes: tuple[tuple[legacy.Target, legacy.SearchQuery, dict[str, Any]], ...]


@dataclass(frozen=True)
class CoreOutcome:
    result: dict[str, Any] | None
    error: str = ""


class SourceCircuitBreaker:
    """Bound total source concurrency and pause all workers after final errors."""

    def __init__(
        self,
        *,
        max_inflight: int,
        minimum_start_interval: float = 0.04,
        failure_threshold: int = 3,
        initial_cooldown: float = 5.0,
        maximum_cooldown: float = 90.0,
        retry_base_delay: float = 1.0,
        maximum_retry_delay: float = 30.0,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self._max_inflight = max_inflight
        self._slots = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._minimum_start_interval = max(0.0, minimum_start_interval)
        self._failure_threshold = max(1, failure_threshold)
        self._initial_cooldown = max(0.1, initial_cooldown)
        self._maximum_cooldown = max(self._initial_cooldown, maximum_cooldown)
        self._retry_base_delay = max(0.0, retry_base_delay)
        self._maximum_retry_delay = max(
            self._retry_base_delay, maximum_retry_delay
        )
        self._last_start = 0.0
        self._consecutive_failures = 0
        self._open_until = 0.0

    def _before_call(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                delay = max(
                    0.0,
                    self._open_until - now,
                    self._minimum_start_interval - (now - self._last_start),
                )
                if delay <= 0:
                    self._last_start = now
                    return
            time.sleep(min(delay, 1.0))

    def call(
        self,
        function: Callable[..., Any],
        *args: object,
        attempts: int = 5,
        **kwargs: object,
    ) -> Any:
        """Retry one-attempt provider calls with source-wide failure visibility."""

        if attempts < 1:
            raise ValueError("attempts must be positive")
        last_error: Exception | None = None
        for attempt in range(attempts):
            with self._slots:
                self._before_call()
                try:
                    value = function(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    with self._lock:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self._failure_threshold:
                            exponent = self._consecutive_failures - self._failure_threshold
                            cooldown = min(
                                self._maximum_cooldown,
                                self._initial_cooldown * (2**exponent),
                            )
                            self._open_until = max(
                                self._open_until, time.monotonic() + cooldown
                            )
                else:
                    with self._lock:
                        self._consecutive_failures = 0
                        self._open_until = 0.0
                    return value
            if attempt + 1 < attempts:
                time.sleep(
                    min(
                        self._maximum_retry_delay,
                        self._retry_base_delay * (2.0**attempt),
                    )
                )
        raise RuntimeError(
            f"provider request failed after {attempts} circuit-visible attempts"
        ) from last_error

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "max_inflight": self._max_inflight,
                "consecutive_failures": self._consecutive_failures,
                "open_seconds_remaining": round(
                    max(0.0, self._open_until - time.monotonic()), 3
                ),
            }


def _query_key(target: legacy.Target, query: legacy.SearchQuery) -> str:
    return f"{target.case_study_id}/{query.query_id}"


def fetch_page_single_attempt(
    query: legacy.SearchQuery, cursor: str, page_size: int
) -> dict[str, Any]:
    """One provider attempt; SourceCircuitBreaker owns retries and backoff."""

    params = {
        "query": academic.full_query(query.expression),
        "resultType": "lite",
        "format": "json",
        "sort": "CITED desc",
        "pageSize": max(1, min(int(page_size), academic.MAX_PAGE_SIZE)),
        "cursorMark": str(cursor or "*"),
    }
    url = academic.EUROPE_PMC_SEARCH + "?" + urllib.parse.urlencode(params)
    return academic._request_json(url, attempts=1)


def fetch_core_single_attempt(source: str, source_id: str) -> dict[str, Any]:
    """Retrieve one complete core record without hiding failures from the circuit."""

    normalized_source = "".join(
        character for character in str(source or "").upper() if character.isalnum()
    )
    normalized_id = str(source_id or "").strip()
    if not normalized_source or not normalized_id or len(normalized_id) > 256:
        raise ValueError("a valid Europe PMC source and identifier are required")
    path = "/".join(
        (
            academic.EUROPE_PMC_ARTICLE.rstrip("/"),
            urllib.parse.quote(normalized_source, safe=""),
            urllib.parse.quote(normalized_id, safe=""),
        )
    )
    payload = academic._request_json(path + "?resultType=core&format=json", attempts=1)
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Europe PMC article response lacks a core result")
    return result


def load_seed_queries(
    seed_dirs: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, object]]]:
    """Choose the furthest checkpoint for each query from immutable seed states."""

    chosen: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    inventory: list[dict[str, object]] = []
    for raw_dir in seed_dirs:
        root = Path(raw_dir).resolve()
        state_path = root / "collection_state.json"
        if not state_path.exists():
            raise RuntimeError(f"seed collection lacks collection_state.json: {root}")
        state = read_json(state_path)
        inventory.append(
            {
                "path": str(root),
                "collection_state_sha256": hash_file(state_path),
                "ready_queue_sha256": hash_file(root / "abstracts_ready_for_extraction.jsonl")
                if (root / "abstracts_ready_for_extraction.jsonl").exists()
                else "",
            }
        )
        queries = state.get("queries") or {}
        if not isinstance(queries, Mapping):
            raise RuntimeError(f"seed queries are invalid: {state_path}")
        for key, raw_value in queries.items():
            if not isinstance(raw_value, Mapping):
                continue
            value = dict(raw_value)
            prior = chosen.get(str(key))
            score = (
                int(value.get("raw_results") or 0),
                int(value.get("pages") or 0),
                bool(value.get("exhausted")),
            )
            prior_score = (
                int((prior or {}).get("raw_results") or 0),
                int((prior or {}).get("pages") or 0),
                bool((prior or {}).get("exhausted")),
            )
            if prior is None or score > prior_score:
                chosen[str(key)] = value
                origins[str(key)] = str(root)
    return chosen, origins, inventory


def build_query_plans(
    seed_queries: Mapping[str, dict[str, Any]],
    seed_origins: Mapping[str, str],
    *,
    targets: Sequence[legacy.Target] | None = None,
) -> list[QueryPlan]:
    """Continue unfinished queries and activate never-run high-recall queries."""

    selected: list[QueryPlan] = []
    rank = 0
    for target in targets or legacy.TARGETS:
        for query in target.queries:
            key = _query_key(target, query)
            seed = seed_queries.get(key)
            if seed is not None and bool(seed.get("exhausted")):
                continue
            if seed is None and "high_recall" not in query.query_id:
                continue
            selected.append(
                QueryPlan(
                    rank=rank,
                    target=target,
                    query=query,
                    key=key,
                    seed=dict(seed) if seed is not None else None,
                    seed_origin=str(seed_origins.get(key) or "never_run_high_recall"),
                )
            )
            rank += 1
    return selected


def _copy_carry_forward(carry_dir: Path, output_dir: Path) -> dict[str, object]:
    source = carry_dir.resolve()
    ready = source / "abstracts_ready_for_extraction.jsonl"
    links = source / "dedup_audit_search_provenance_links.jsonl"
    state = source / "collection_state.json"
    audit = source / "dedup_audit_excluded_papers.jsonl"
    if not ready.exists() or not state.exists():
        raise RuntimeError(f"carry-forward collection is incomplete: {source}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("carry-forward is allowed only for a new empty output directory")
    tracked = [path for path in (ready, links, audit, state) if path.exists()]
    before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked
    }
    state_mtime = state.stat().st_mtime_ns
    newer_than_state = [
        str(path)
        for path in (ready, links, audit)
        if path.exists() and path.stat().st_mtime_ns > state_mtime
    ]
    if newer_than_state:
        raise RuntimeError(
            "carry-forward is not at a page checkpoint; stop the legacy collector "
            f"after collection_state.json is written: {newer_than_state}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ready, output_dir / ready.name)
    if links.exists():
        shutil.copy2(links, output_dir / links.name)
    else:
        (output_dir / links.name).touch()
    if audit.exists():
        shutil.copy2(audit, output_dir / audit.name)
    after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked
    }
    if after != before:
        raise RuntimeError(
            "carry-forward source changed while it was copied; discard the new output "
            "directory and retry after the legacy collector is stopped"
        )
    copied_ready = output_dir / ready.name
    if hash_file(copied_ready) != hash_file(ready):
        raise RuntimeError("carry-forward ready queue copy failed hash verification")
    return {
        "path": str(source),
        "ready_queue_sha256": hash_file(ready),
        "collection_state_sha256": hash_file(state),
        "dedup_audit_sha256": hash_file(audit) if audit.exists() else "",
        "page_checkpoint_verified": True,
        "collection_complete": (source / "COLLECTION_COMPLETE.json").exists(),
        "copied_at": utc_now(),
    }


def initialize_state(
    *,
    plans: Sequence[QueryPlan],
    target_total: int,
    seed_inventory: Sequence[dict[str, object]],
    carry_inventory: dict[str, object] | None,
) -> dict[str, Any]:
    queries: dict[str, dict[str, Any]] = {}
    for plan in plans:
        seed = plan.seed or {}
        queries[plan.key] = {
            "case_study_id": plan.target.case_study_id,
            "query_id": plan.query.query_id,
            "cursor": str(seed.get("cursor") or "*"),
            "pages": int(seed.get("pages") or 0),
            "raw_results": int(seed.get("raw_results") or 0),
            "reported_hit_count": seed.get("reported_hit_count"),
            "exhausted": False,
            "suspended": False,
            "errors": int(seed.get("errors") or 0),
            "run_pages": 0,
            "run_raw_results": 0,
            "seed_origin": plan.seed_origin,
            "seed_cursor": str(seed.get("cursor") or "*"),
            "seed_raw_results": int(seed.get("raw_results") or 0),
        }
    return {
        "schema_version": f"{SCHEMA}.state",
        "created_at": utc_now(),
        "target_unique_papers": target_total,
        "queries": queries,
        "pages_completed_this_run": 0,
        "raw_results_scanned_this_run": 0,
        "invalid_results": {},
        "seed_collections": list(seed_inventory),
        "carry_forward": carry_inventory,
        "formal_kg_mutated": False,
    }


def load_or_initialize_state(
    state_path: Path,
    *,
    plans: Sequence[QueryPlan],
    target_total: int,
    seed_inventory: Sequence[dict[str, object]],
    carry_inventory: dict[str, object] | None,
) -> dict[str, Any]:
    if not state_path.exists():
        return initialize_state(
            plans=plans,
            target_total=target_total,
            seed_inventory=seed_inventory,
            carry_inventory=carry_inventory,
        )
    state = read_json(state_path)
    if state.get("schema_version") != f"{SCHEMA}.state":
        raise RuntimeError(f"incompatible accelerator state: {state_path}")
    if int(state.get("target_unique_papers") or 0) != target_total:
        raise RuntimeError("target_total differs from the resumable accelerator state")
    expected_keys = {plan.key for plan in plans}
    if set((state.get("queries") or {}).keys()) != expected_keys:
        raise RuntimeError("query plan differs from the resumable accelerator state")
    return state


def _record_link(
    *,
    links_path: Path,
    links: set[tuple[str, str, str]],
    paper_cases: dict[str, set[str]],
    paper_queries: dict[str, set[str]],
    paper_id: str,
    target: legacy.Target,
    query: legacy.SearchQuery,
) -> None:
    key = (paper_id, target.case_study_id, query.query_id)
    paper_cases[paper_id].add(target.case_study_id)
    paper_queries[paper_id].add(query.query_id)
    if key in links:
        return
    links.add(key)
    append_jsonl(
        links_path,
        {
            "schema_version": "search_provenance_link.v1",
            "paper_id": paper_id,
            "case_study_id": target.case_study_id,
            "query_id": query.query_id,
            "linked_at": utc_now(),
            "membership_asserted": False,
        },
    )


def _local_match(record: Mapping[str, Any], alias_to_paper: Mapping[str, str]) -> str:
    return next(
        (
            alias_to_paper[alias]
            for alias in paper_identity_aliases(record)
            if alias in alias_to_paper
        ),
        "",
    )


def _fetch_page(
    plan: QueryPlan,
    cursor: str,
    page_size: int,
    breaker: SourceCircuitBreaker,
    fetch_page_fn: Callable[[legacy.SearchQuery, str, int], dict[str, Any]],
) -> PageFetch:
    try:
        payload = breaker.call(fetch_page_fn, plan.query, cursor, page_size)
        if not isinstance(payload, dict):
            raise RuntimeError("search response is not an object")
        return PageFetch(plan=plan, cursor=cursor, payload=payload)
    except Exception as exc:
        return PageFetch(plan=plan, cursor=cursor, payload=None, error=repr(exc))


def fetch_page_wave(
    *,
    plans: Sequence[QueryPlan],
    state: Mapping[str, Any],
    page_size: int,
    search_workers: int,
    breaker: SourceCircuitBreaker,
    fetch_page_fn: Callable[[legacy.SearchQuery, str, int], dict[str, Any]],
) -> list[PageFetch]:
    active = [
        plan
        for plan in plans
        if not bool(state["queries"][plan.key].get("exhausted"))
        and not bool(state["queries"][plan.key].get("suspended"))
    ]
    if not active:
        return []
    fetched: list[PageFetch] = []
    with ThreadPoolExecutor(max_workers=min(search_workers, len(active))) as executor:
        futures: list[Future[PageFetch]] = []
        for plan in active:
            cursor = str(state["queries"][plan.key].get("cursor") or "*")
            futures.append(
                executor.submit(
                    _fetch_page,
                    plan,
                    cursor,
                    page_size,
                    breaker,
                    fetch_page_fn,
                )
            )
        for future in as_completed(futures):
            fetched.append(future.result())
    return sorted(fetched, key=lambda item: item.plan.rank)


def _fetch_core(
    task: CandidateTask,
    breaker: SourceCircuitBreaker,
    fetch_core_fn: Callable[[str, str], dict[str, Any]],
) -> CoreOutcome:
    try:
        result = breaker.call(
            fetch_core_fn,
            str(task.candidate["europepmc_source"]),
            str(task.candidate["europepmc_id"]),
        )
        if not isinstance(result, dict):
            raise RuntimeError("core response is not an object")
        return CoreOutcome(result=result)
    except Exception as exc:
        return CoreOutcome(result=None, error=repr(exc))


def process_page(
    *,
    page: PageFetch,
    output_dir: Path,
    state: dict[str, Any],
    alias_to_paper: dict[str, str],
    paper_cases: dict[str, set[str]],
    paper_queries: dict[str, set[str]],
    links: set[tuple[str, str, str]],
    primary_counts: Counter[str],
    total_ref: list[int],
    target_total: int,
    deduplicator: CandidatePaperDeduplicator,
    breaker: SourceCircuitBreaker,
    core_workers: int,
    core_chunk_size: int,
    fetch_core_fn: Callable[[str, str], dict[str, Any]],
    max_query_errors: int,
) -> None:
    qstate = state["queries"][page.plan.key]
    state_path = output_dir / "collection_state.json"
    papers_path = output_dir / "abstracts_ready_for_extraction.jsonl"
    links_path = output_dir / "dedup_audit_search_provenance_links.jsonl"
    invalid = Counter(state.get("invalid_results") or {})

    if page.error:
        qstate["errors"] = int(qstate.get("errors") or 0) + 1
        if int(qstate["errors"]) >= max_query_errors:
            qstate["suspended"] = True
        append_jsonl(
            output_dir / "search_errors.jsonl",
            {
                "at": utc_now(),
                "case_study_id": page.plan.target.case_study_id,
                "query_id": page.plan.query.query_id,
                "cursor": page.cursor,
                "error": page.error,
                "suspended": bool(qstate.get("suspended")),
            },
        )
        state["invalid_results"] = dict(sorted(invalid.items()))
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        LOGGER.error("query failed: %s: %s", page.plan.key, page.error)
        return

    payload = page.payload or {}
    results = ((payload.get("resultList") or {}).get("result")) or []
    if not isinstance(results, list):
        results = []
    next_cursor = str(payload.get("nextCursorMark") or "")
    qstate["reported_hit_count"] = int(payload.get("hitCount") or 0)
    qstate["pages"] = int(qstate.get("pages") or 0) + 1
    qstate["run_pages"] = int(qstate.get("run_pages") or 0) + 1
    qstate["raw_results"] = int(qstate.get("raw_results") or 0) + len(results)
    qstate["run_raw_results"] = int(qstate.get("run_raw_results") or 0) + len(results)
    state["pages_completed_this_run"] = int(
        state.get("pages_completed_this_run") or 0
    ) + 1
    state["raw_results_scanned_this_run"] = int(
        state.get("raw_results_scanned_this_run") or 0
    ) + len(results)

    tasks: list[CandidateTask] = []
    pending_alias_to_task: dict[str, int] = {}
    pending_extras: dict[int, list[tuple[legacy.Target, legacy.SearchQuery, dict[str, Any]]]] = defaultdict(list)
    for result_position, raw_result in enumerate(results):
        if not isinstance(raw_result, dict):
            invalid["non_object_result"] += 1
            continue
        candidate, status = legacy.normalize_candidate_result(
            raw_result,
            target=page.plan.target,
            query=page.plan.query,
        )
        if candidate is None:
            invalid[status] += 1
            continue
        within = _local_match(candidate, alias_to_paper)
        if within:
            _record_link(
                links_path=links_path,
                links=links,
                paper_cases=paper_cases,
                paper_queries=paper_queries,
                paper_id=within,
                target=page.plan.target,
                query=page.plan.query,
            )
            deduplicator.is_seen(
                candidate,
                source="europepmc",
                preset=page.plan.target.case_study_id,
                stage="within_current_collection_merge",
            )
            continue
        if deduplicator.is_seen(
            candidate,
            source="europepmc",
            preset=page.plan.target.case_study_id,
            stage="post_search_pre_abstract",
        ):
            continue
        pending_index = next(
            (
                pending_alias_to_task[alias]
                for alias in paper_identity_aliases(candidate)
                if alias in pending_alias_to_task
            ),
            None,
        )
        if pending_index is not None:
            pending_extras[pending_index].append(
                (page.plan.target, page.plan.query, candidate)
            )
            continue
        task_index = len(tasks)
        for alias in paper_identity_aliases(candidate):
            pending_alias_to_task[alias] = task_index
        tasks.append(
            CandidateTask(
                candidate=candidate,
                target=page.plan.target,
                query=page.plan.query,
                result_position=result_position,
                extra_routes=(),
            )
        )

    if pending_extras:
        tasks = [
            CandidateTask(
                candidate=task.candidate,
                target=task.target,
                query=task.query,
                result_position=task.result_position,
                extra_routes=tuple(pending_extras.get(index, [])),
            )
            for index, task in enumerate(tasks)
        ]

    with ThreadPoolExecutor(max_workers=core_workers) as executor:
        for chunk_start in range(0, len(tasks), core_chunk_size):
            if total_ref[0] >= target_total:
                break
            chunk = tasks[chunk_start : chunk_start + core_chunk_size]
            outcomes = executor.map(
                lambda task: _fetch_core(task, breaker, fetch_core_fn),
                chunk,
            )
            for task, outcome in zip(chunk, outcomes):
                if total_ref[0] >= target_total:
                    break
                if outcome.result is None:
                    invalid["abstract_retrieval_failed"] += 1
                    append_jsonl(
                        output_dir / "abstract_errors.jsonl",
                        {
                            "at": utc_now(),
                            "case_study_id": task.target.case_study_id,
                            "query_id": task.query.query_id,
                            "paper_id": task.candidate.get("paper_id"),
                            "europepmc_source": task.candidate.get("europepmc_source"),
                            "europepmc_id": task.candidate.get("europepmc_id"),
                            "error": outcome.error,
                        },
                    )
                    continue
                record, status = legacy.normalize_result(
                    outcome.result,
                    target=task.target,
                    query=task.query,
                    expected_candidate=task.candidate,
                )
                if record is None:
                    invalid[status] += 1
                    continue
                if not deduplicator.accept(
                    record,
                    source="europepmc",
                    preset=task.target.case_study_id,
                    collection_path=str(papers_path.resolve()),
                ):
                    continue
                append_jsonl(papers_path, record)
                paper_id = str(record["paper_id"])
                for alias in paper_identity_aliases(record):
                    alias_to_paper[alias] = paper_id
                primary_counts[task.target.case_study_id] += 1
                total_ref[0] += 1
                _record_link(
                    links_path=links_path,
                    links=links,
                    paper_cases=paper_cases,
                    paper_queries=paper_queries,
                    paper_id=paper_id,
                    target=task.target,
                    query=task.query,
                )
                for extra_target, extra_query, extra_candidate in task.extra_routes:
                    deduplicator.is_seen(
                        extra_candidate,
                        source="europepmc",
                        preset=extra_target.case_study_id,
                        stage="within_current_collection_merge",
                    )
                    _record_link(
                        links_path=links_path,
                        links=links,
                        paper_cases=paper_cases,
                        paper_queries=paper_queries,
                        paper_id=paper_id,
                        target=extra_target,
                        query=extra_query,
                    )

    qstate["cursor"] = next_cursor or page.cursor
    if not results or not next_cursor or next_cursor == page.cursor:
        qstate["exhausted"] = True
    deduplicator.flush()
    state["invalid_results"] = dict(sorted(invalid.items()))
    state["accepted_unique_papers"] = total_ref[0]
    state["primary_counts"] = dict(sorted(primary_counts.items()))
    state["updated_at"] = utc_now()
    state["source_circuit"] = breaker.status()
    write_json(state_path, state)
    print(
        f"[{page.plan.key}] total={total_ref[0]:,}/{target_total:,} "
        f"run_pages={state['pages_completed_this_run']:,} "
        f"run_raw={state['raw_results_scanned_this_run']:,} "
        f"dedup_excluded={deduplicator.excluded:,}",
        flush=True,
    )


def _load_existing_output(
    *,
    papers_path: Path,
    links_path: Path,
    identity_index: GlobalPaperIdentityIndex,
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
    dict[str, set[str]],
    set[tuple[str, str, str]],
    Counter[str],
    int,
]:
    return legacy.load_existing_collection(papers_path, links_path, identity_index)


def seal_collection(
    *,
    output_dir: Path,
    target_total: int,
    batch_size: int,
    formal_files: Sequence[Path],
    formal_baseline: dict[str, dict[str, object]],
    state: dict[str, Any],
    identity_sync: dict[str, Any],
    deduplication: dict[str, Any],
    started: float,
    core_workers: int,
    search_workers: int,
    source_inflight: int,
    identity_resync_pages: int,
) -> dict[str, Any]:
    papers_path = output_dir / "abstracts_ready_for_extraction.jsonl"
    links_path = output_dir / "dedup_audit_search_provenance_links.jsonl"
    paper_cases: dict[str, set[str]] = defaultdict(set)
    paper_queries: dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(links_path):
        paper_id = str(row.get("paper_id") or "")
        case_id = str(row.get("case_study_id") or "")
        query_id = str(row.get("query_id") or "")
        if paper_id and case_id:
            paper_cases[paper_id].add(case_id)
        if paper_id and query_id:
            paper_queries[paper_id].add(query_id)

    batch_dir = output_dir / "extraction_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for stale in batch_dir.glob("batch_*.jsonl"):
        stale.unlink()
    temp_queue = papers_path.with_suffix(".jsonl.finalizing")
    aliases_seen: dict[str, str] = {}
    abstract_lengths: list[int] = []
    count = 0
    batch_rows: list[dict[str, Any]] = []
    batch_count = 0
    with temp_queue.open("w", encoding="utf-8", newline="\n") as queue_handle:
        for row in iter_jsonl(papers_path):
            paper_id = str(row.get("paper_id") or "")
            cases = set(str(value) for value in row.get("search_target_case_study_ids") or [])
            queries = set(str(value) for value in row.get("search_query_ids") or [])
            cases.update(paper_cases.get(paper_id, set()))
            queries.update(paper_queries.get(paper_id, set()))
            abstract = legacy.clean_text(row.get("abstract"))
            if len(abstract) < legacy.MIN_ABSTRACT_CHARACTERS:
                raise RuntimeError(f"final queue contains short abstract: {paper_id}")
            year = legacy.parse_year(row.get("year"))
            if year is None:
                raise RuntimeError(f"final queue contains out-of-window year: {paper_id}")
            row["abstract"] = abstract
            row["abstract_characters"] = len(abstract)
            row["abstract_words"] = len(abstract.split())
            row["search_target_case_study_ids"] = sorted(cases)
            row["search_query_ids"] = sorted(queries)
            row["search_provenance_only"] = True
            row["assignment_policy"] = legacy.ASSIGNMENT_POLICY
            row["kg_injection"] = False
            for alias in paper_identity_aliases(row):
                prior = aliases_seen.setdefault(alias, paper_id)
                if prior != paper_id:
                    raise RuntimeError(f"final queue identity collision: {alias}")
            count += 1
            abstract_lengths.append(len(abstract))
            queue_handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            batch_row = dict(row)
            batch_row["batch_id"] = ((count - 1) // batch_size) + 1
            batch_row["batch_position"] = ((count - 1) % batch_size) + 1
            batch_rows.append(batch_row)
            if len(batch_rows) == batch_size:
                batch_count += 1
                batch_path = batch_dir / f"batch_{batch_count:04d}.jsonl"
                with batch_path.open("w", encoding="utf-8", newline="\n") as handle:
                    for value in batch_rows:
                        handle.write(
                            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                batch_rows.clear()
    if batch_rows:
        batch_count += 1
        batch_path = batch_dir / f"batch_{batch_count:04d}.jsonl"
        with batch_path.open("w", encoding="utf-8", newline="\n") as handle:
            for value in batch_rows:
                handle.write(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    os.replace(temp_queue, papers_path)

    after = file_snapshot(formal_files)
    if after != formal_baseline:
        raise RuntimeError("formal KG changed during accelerated collection; refusing seal")
    status = "complete" if count >= target_total else "source_exhausted_shortfall"
    manifest = {
        "schema_version": SCHEMA,
        "created_at": state["created_at"],
        "completed_at": utc_now(),
        "status": status,
        "mode": "parallel_search_single_identity_coordinator_bounded_core_download",
        "years": {"start": legacy.YEAR_START, "end": legacy.YEAR_END},
        "target_unique_papers": target_total,
        "collected_unique_papers": count,
        "batch_size": batch_size,
        "total_batches": batch_count,
        "source": "Europe PMC official REST API",
        "parallelism": {
            "search_workers": search_workers,
            "core_abstract_workers": core_workers,
            "source_max_inflight": source_inflight,
            "identity_resync_pages": identity_resync_pages,
            "deterministic_commit_order": "query_rank,page_wave,result_position",
            "single_identity_coordinator": True,
        },
        "abstracts": {
            "full_provider_abstracts_not_truncated": True,
            "minimum_characters_gate": legacy.MIN_ABSTRACT_CHARACTERS,
            "minimum_observed_characters": min(abstract_lengths) if abstract_lengths else None,
            "maximum_observed_characters": max(abstract_lengths) if abstract_lengths else None,
            "mean_observed_characters": round(sum(abstract_lengths) / count, 3)
            if count
            else None,
        },
        "deduplication": {
            **deduplication,
            "identity_order": [
                "PMID",
                "DOI",
                "PMCID",
                "arXiv",
                "OpenAlex",
                "normalized title + year",
            ],
            "formal_kg_and_all_active_staging_checked_before_abstract_retrieval": True,
            "within_campaign_unique_aliases": len(aliases_seen),
        },
        "identity_sync": identity_sync,
        "search_state": state,
        "assignment_policy": legacy.ASSIGNMENT_POLICY,
        "claim_extraction_performed": False,
        "kg_injection": False,
        "formal_kg_mutated": False,
        "formal_kg_files": after,
        "outputs": {
            "ready_queue": str(papers_path.resolve()),
            "extraction_batches": str(batch_dir.resolve()),
            "dedup_audit": str(
                (output_dir / "dedup_audit_excluded_papers.jsonl").resolve()
            ),
            "search_provenance_links": str(links_path.resolve()),
        },
        "output_sha256": {
            "ready_queue": hash_file(papers_path),
            "collection_state": hash_file(output_dir / "collection_state.json"),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "manifest.json", manifest)
    complete = {
        "schema_version": f"{SCHEMA}.complete",
        "completed_at": manifest["completed_at"],
        "status": status,
        "papers": count,
        "batches": batch_count,
        "ready_queue_sha256": manifest["output_sha256"]["ready_queue"],
        "manifest_sha256": hash_file(output_dir / "manifest.json"),
        "formal_kg_inventory_sha256": canonical_hash(after),
        "kg_injection": False,
        "formal_kg_mutated": False,
    }
    write_json(output_dir / "COLLECTION_COMPLETE.json", complete)
    (output_dir / "README.md").write_text(
        "# Accelerated KG-v3 literature top-up\n\n"
        "Collection-only output. Search pages and complete abstract downloads were "
        "parallelised, while a single coordinator performed deterministic paper "
        "identity decisions and ordered commits. Search provenance is not Case Study "
        "membership. No formal KG file was modified.\n",
        encoding="utf-8",
    )
    return manifest


def run_collection(
    *,
    output_dir: Path,
    target_total: int,
    seed_dirs: Sequence[Path],
    carry_forward_dir: Path | None = None,
    page_size: int = legacy.DEFAULT_PAGE_SIZE,
    batch_size: int = 100,
    search_workers: int = 4,
    core_workers: int = 6,
    source_inflight: int = 6,
    core_chunk_size: int = 128,
    max_query_errors: int = 5,
    identity_resync_pages: int = 12,
    formal_claim_store: Path = legacy.FORMAL_CLAIMS,
    formal_files: Sequence[Path] = legacy.FORMAL_FILES,
    staging_roots: Sequence[Path] = legacy.STAGING_ROOTS,
    identity_index_path: Path = legacy.IDENTITY_INDEX,
    targets: Sequence[legacy.Target] | None = None,
    fetch_page_fn: Callable[
        [legacy.SearchQuery, str, int], dict[str, Any]
    ] = fetch_page_single_attempt,
    fetch_core_fn: Callable[[str, str], dict[str, Any]] = fetch_core_single_attempt,
) -> dict[str, Any]:
    if not 4 <= core_workers <= 8:
        raise ValueError("core_workers must remain in the bounded range 4..8")
    if search_workers < 1 or source_inflight < 1:
        raise ValueError("search_workers and source_inflight must be positive")
    output_dir = output_dir.resolve()
    if (output_dir / "COLLECTION_COMPLETE.json").exists():
        raise RuntimeError("output is already sealed; refusing overwrite")
    started = time.time()
    seed_queries, seed_origins, seed_inventory = load_seed_queries(seed_dirs)
    plans = build_query_plans(seed_queries, seed_origins, targets=targets)
    if not plans:
        raise RuntimeError("no unexhausted or never-run high-recall queries are available")

    carry_inventory: dict[str, object] | None = None
    if carry_forward_dir is not None and (
        not output_dir.exists() or not any(output_dir.iterdir())
    ):
        carry_inventory = _copy_carry_forward(carry_forward_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_path = output_dir / "abstracts_ready_for_extraction.jsonl"
    links_path = output_dir / "dedup_audit_search_provenance_links.jsonl"
    papers_path.touch(exist_ok=True)
    links_path.touch(exist_ok=True)
    baseline_path = output_dir / "formal_kg_baseline.json"
    current_formal = file_snapshot(formal_files)
    if baseline_path.exists():
        baseline_payload = read_json(baseline_path)
        formal_baseline = baseline_payload.get("files")
        if formal_baseline != current_formal:
            raise RuntimeError("formal KG changed since accelerated collection start")
    else:
        formal_baseline = current_formal
        write_json(
            baseline_path,
            {"captured_at": utc_now(), "files": formal_baseline},
        )

    state_path = output_dir / "collection_state.json"
    state = load_or_initialize_state(
        state_path,
        plans=plans,
        target_total=target_total,
        seed_inventory=seed_inventory,
        carry_inventory=carry_inventory,
    )
    breaker = SourceCircuitBreaker(max_inflight=source_inflight)
    with GlobalPaperIdentityIndex(identity_index_path) as identity_index:
        identity_sync = identity_index.sync(
            formal_claim_store=formal_claim_store,
            staging_roots=staging_roots,
        )
        (
            alias_to_paper,
            paper_cases,
            paper_queries,
            links,
            primary_counts,
            existing_total,
        ) = _load_existing_output(
            papers_path=papers_path,
            links_path=links_path,
            identity_index=identity_index,
        )
        total_ref = [existing_total]
        deduplicator = CandidatePaperDeduplicator(
            identity_index,
            output_dir / "dedup_audit_excluded_papers.jsonl",
        )
        state["accepted_unique_papers"] = existing_total
        state["primary_counts"] = dict(sorted(primary_counts.items()))
        write_json(state_path, state)

        last_identity_sync_page = int(state.get("pages_completed_this_run") or 0)
        while total_ref[0] < target_total:
            wave = fetch_page_wave(
                plans=plans,
                state=state,
                page_size=page_size,
                search_workers=search_workers,
                breaker=breaker,
                fetch_page_fn=fetch_page_fn,
            )
            if not wave:
                break
            before = total_ref[0]
            for page in wave:
                if total_ref[0] >= target_total:
                    break
                process_page(
                    page=page,
                    output_dir=output_dir,
                    state=state,
                    alias_to_paper=alias_to_paper,
                    paper_cases=paper_cases,
                    paper_queries=paper_queries,
                    links=links,
                    primary_counts=primary_counts,
                    total_ref=total_ref,
                    target_total=target_total,
                    deduplicator=deduplicator,
                    breaker=breaker,
                    core_workers=core_workers,
                    core_chunk_size=core_chunk_size,
                    fetch_core_fn=fetch_core_fn,
                    max_query_errors=max_query_errors,
                )
            completed_pages = int(state.get("pages_completed_this_run") or 0)
            if (
                identity_resync_pages > 0
                and completed_pages - last_identity_sync_page >= identity_resync_pages
            ):
                assert_formal_stats_unchanged(formal_files, formal_baseline)
                identity_sync = identity_index.sync(
                    formal_claim_store=formal_claim_store,
                    staging_roots=staging_roots,
                )
                last_identity_sync_page = completed_pages
            if total_ref[0] == before and all(
                bool(state["queries"][plan.key].get("exhausted"))
                or bool(state["queries"][plan.key].get("suspended"))
                for plan in plans
            ):
                break
        deduplicator.flush()
        state["updated_at"] = utc_now()
        state["accepted_unique_papers"] = total_ref[0]
        state["primary_counts"] = dict(sorted(primary_counts.items()))
        state["deduplication"] = deduplicator.summary()
        state["identity_sync"] = identity_sync
        write_json(state_path, state)
        deduplication = deduplicator.summary()

    return seal_collection(
        output_dir=output_dir,
        target_total=target_total,
        batch_size=batch_size,
        formal_files=formal_files,
        formal_baseline=formal_baseline,
        state=state,
        identity_sync=identity_sync,
        deduplication=deduplication,
        started=started,
        core_workers=core_workers,
        search_workers=search_workers,
        source_inflight=source_inflight,
        identity_resync_pages=identity_resync_pages,
    )


def plan_payload(seed_dirs: Sequence[Path]) -> dict[str, Any]:
    seed_queries, seed_origins, seed_inventory = load_seed_queries(seed_dirs)
    plans = build_query_plans(seed_queries, seed_origins)
    return {
        "schema_version": f"{SCHEMA}.plan",
        "years": [legacy.YEAR_START, legacy.YEAR_END],
        "seed_collections": seed_inventory,
        "queries": [
            {
                "rank": plan.rank,
                "case_study_id": plan.target.case_study_id,
                "query_id": plan.query.query_id,
                "cursor": str((plan.seed or {}).get("cursor") or "*"),
                "raw_results_already_scanned": int(
                    (plan.seed or {}).get("raw_results") or 0
                ),
                "reported_hit_count": (plan.seed or {}).get("reported_hit_count"),
                "seed_origin": plan.seed_origin,
                "mode": "resume_cursor" if plan.seed is not None else "start_high_recall",
            }
            for plan in plans
        ],
        "claim_extraction": False,
        "kg_injection": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-total", type=int, default=42_000)
    parser.add_argument(
        "--seed-collection-dir",
        type=Path,
        action="append",
        dest="seed_dirs",
        help="Prior collection whose furthest per-query cursor should be reused",
    )
    parser.add_argument(
        "--carry-forward-dir",
        type=Path,
        help="Stopped partial collection copied into a new output directory",
    )
    parser.add_argument("--page-size", type=int, default=legacy.DEFAULT_PAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--core-workers", type=int, choices=range(4, 9), default=6)
    parser.add_argument("--source-inflight", type=int, default=6)
    parser.add_argument("--core-chunk-size", type=int, default=128)
    parser.add_argument("--max-query-errors", type=int, default=5)
    parser.add_argument("--identity-resync-pages", type=int, default=12)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    seed_dirs = args.seed_dirs or [DEFAULT_SEED]
    if args.plan:
        print(json.dumps(plan_payload(seed_dirs), ensure_ascii=False, indent=2))
        return 0
    manifest = run_collection(
        output_dir=args.output_dir,
        target_total=args.target_total,
        seed_dirs=seed_dirs,
        carry_forward_dir=args.carry_forward_dir,
        page_size=args.page_size,
        batch_size=args.batch_size,
        search_workers=args.search_workers,
        core_workers=args.core_workers,
        source_inflight=args.source_inflight,
        core_chunk_size=args.core_chunk_size,
        max_query_errors=args.max_query_errors,
        identity_resync_pages=args.identity_resync_pages,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
