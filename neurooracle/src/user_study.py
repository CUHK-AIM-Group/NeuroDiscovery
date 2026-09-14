"""Persistent expert-ranking studies for NeuroDiscovery case studies.

The service deliberately keeps study data outside the repository and runtime
bundle.  It stores an append-only interaction log alongside immutable candidate
snapshots so a ranking can be reproduced after the source files change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL_VERSION = "case1-pairwise-v8-timeboxed-6x10m"
DEFAULT_MAX_PAIRS = 100
MAX_CURATED_PAIRS = 2000
DEFAULT_SESSION_ACTIVE_SECONDS = 600
CONDITIONS = {"manual", "assisted", "generator"}
RESULT_STATUSES = {"confirmed", "not_confirmed", "inconclusive", "execution_failed"}
SCORE_FIELDS = {
    "confidence_score",
    "novelty_score",
    "evidence_score",
    "testability_score",
    "composite_score",
    "critic_score",
    "kge_score",
    "plausibility_score",
    "generator_rank",
    "rank",
}


def default_study_root() -> Path:
    configured = os.environ.get("NEURODISCOVERY_STUDY_ROOT", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".neurodiscovery" / "user-studies"


def _study_protocol(payload: Any) -> dict[str, Any]:
    raw = payload.get("session_protocol") if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    try:
        required_sessions = max(1, min(24, int(raw.get("required_sessions") or 1)))
    except (TypeError, ValueError):
        required_sessions = 1
    try:
        active_seconds = max(
            60,
            min(4 * 60 * 60, int(raw.get("active_seconds_per_session") or DEFAULT_SESSION_ACTIVE_SECONDS)),
        )
    except (TypeError, ValueError):
        active_seconds = DEFAULT_SESSION_ACTIVE_SECONDS
    try:
        pair_pool_per_session = max(0, int(raw.get("pair_pool_per_session") or 0))
    except (TypeError, ValueError):
        pair_pool_per_session = 0
    return {
        "completion_basis": "active_time",
        "required_sessions": required_sessions,
        "active_seconds_per_session": active_seconds,
        "pair_pool_per_session": pair_pool_per_session,
        "assignment_policy": str(
            raw.get("assignment_policy") or "participant_seeded"
        ),
        "shared_random_seed": int(raw.get("shared_random_seed") or 0),
        "same_questions_for_all_participants": bool(
            raw.get("same_questions_for_all_participants")
        ),
    }


def _now() -> float:
    return time.time()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path | None) -> str:
    if not path or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = next(
            (
                payload[key]
                for key in (
                    "scheduled_hypotheses",
                    "hypotheses",
                    "candidates",
                    "results",
                    "items",
                )
                if isinstance(payload.get(key), list)
            ),
            [],
        )
    else:
        raw = []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _candidate_id(item: dict[str, Any], index: int) -> str:
    value = str(item.get("id") or item.get("hypothesis_id") or "").strip()
    if value:
        return value
    basis = _json(item).encode("utf-8")
    return f"hyp-{index + 1:04d}-{_sha256_bytes(basis)[:10]}"


def _normalize_candidate(item: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(item)
    out["id"] = _candidate_id(out, index)
    source = str(out.get("source_name") or out.get("source") or "").strip()
    target = str(out.get("target_name") or out.get("target") or "").strip()
    metadata = dict(out.get("metadata") or {}) if isinstance(out.get("metadata"), dict) else {}
    title = str(
        out.get("title") or out.get("name") or metadata.get("display_title") or ""
    ).strip()
    if not title:
        title = f"{source} → {target}" if source and target else source or target or out["id"]
    out["title"] = title
    out["summary"] = str(
        out.get("summary") or out.get("explanation") or out.get("hypothesis") or out.get("text") or ""
    ).strip()
    path = out.get("path")
    out["path"] = path if isinstance(path, list) else []
    claims = out.get("supporting_claims")
    out["supporting_claims"] = claims if isinstance(claims, list) else []
    out["metadata"] = metadata
    out["comparison_task_id"] = candidate_comparison_task(out)
    metadata.setdefault("comparison_task_id", out["comparison_task_id"])
    return out


def candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the blinded structural positions used to stage pair difficulty."""
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    candidate_tuple = (
        metadata.get("candidate_tuple") if isinstance(metadata.get("candidate_tuple"), dict) else {}
    )
    diseases = candidate_tuple.get("disease_ids") or metadata.get("disease_ids") or []
    if not diseases:
        diseases = candidate_tuple.get("diseases") or metadata.get("diseases") or []
    disease_ids = tuple(sorted({str(item).strip() for item in diseases if str(item).strip()}))
    region = str(
        candidate_tuple.get("region_id")
        or metadata.get("cluster_region_id")
        or candidate.get("target_id")
        or candidate_tuple.get("region_name")
        or metadata.get("cluster_region_name")
        or candidate.get("target_name")
        or ""
    ).strip()
    feature = str(
        candidate_tuple.get("feature_id")
        or metadata.get("cluster_modality")
        or candidate_tuple.get("feature_name")
        or ""
    ).strip()
    direction = str(
        candidate_tuple.get("direction")
        or metadata.get("cluster_sign")
        or metadata.get("direction_assumption")
        or ""
    ).strip()
    return {
        "diseases": disease_ids,
        "region": region,
        "feature": feature,
        "direction": direction,
    }


def candidate_semantic_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Return a normalized identity for the scientific hypothesis itself.

    Candidate IDs identify generator records, not necessarily unique scientific
    propositions.  Different generation methods can emit the same Case 1 tuple,
    so pair construction must compare this key rather than IDs alone.
    """
    signature = candidate_signature(candidate)
    diseases = tuple(str(item).strip().casefold() for item in signature["diseases"])
    region = str(signature["region"]).strip().casefold()
    feature = str(signature["feature"]).strip().casefold()
    direction = str(signature["direction"]).strip().casefold()
    if not (diseases or region or feature or direction):
        return ()
    return diseases, region, feature, direction


def candidate_comparison_task(candidate: dict[str, Any]) -> str:
    """Return the scientific question within which pairwise ranking is valid.

    For the transdiagnostic atlas, the disease set defines the research task;
    brain region, imaging feature, and direction are competing hypothesis
    positions within that task. Other hypothesis families fall back to their
    explicit task or chain scope, so unrelated questions are never compared.
    """
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    explicit = str(
        candidate.get("comparison_task_id") or metadata.get("comparison_task_id") or ""
    ).strip()
    if explicit:
        return explicit

    signature = candidate_signature(candidate)
    task_name = str(
        metadata.get("task_name")
        or metadata.get("chain_name")
        or candidate.get("task_name")
        or candidate.get("hypothesis_type")
        or "unspecified"
    ).strip()
    if signature["diseases"]:
        scope = {
            "task": task_name,
            "diseases": list(signature["diseases"]),
        }
    elif metadata.get("chain_name") or candidate.get("hypothesis_type") == "chain":
        scope = {
            "task": task_name,
            "source": str(candidate.get("source_id") or candidate.get("source_name") or ""),
            "target": str(candidate.get("target_id") or candidate.get("target_name") or ""),
        }
    else:
        scope = {"task": task_name}
    return f"task-{_sha256_bytes(_json(scope).encode('utf-8'))[:12]}"


def candidate_structural_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Count changed hypothesis-chain positions; one disease substitution is one change."""
    a = candidate_signature(left)
    b = candidate_signature(right)
    a_diseases, b_diseases = set(a["diseases"]), set(b["diseases"])
    disease_changes = max(len(a_diseases), len(b_diseases)) - len(a_diseases & b_diseases)
    return int(
        disease_changes
        + (a["region"] != b["region"])
        + (a["feature"] != b["feature"])
        + (bool(a["direction"] or b["direction"]) and a["direction"] != b["direction"])
    )


def candidate_reference_keys(candidate: dict[str, Any]) -> set[str]:
    """Return stable identities for the literature displayed with a candidate."""
    references = candidate.get("literature") if isinstance(candidate.get("literature"), list) else []
    keys: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict):
            continue
        value = reference.get("pmid") or reference.get("doi") or reference.get("title") or ""
        normalized = " ".join(str(value).strip().casefold().split())
        if normalized:
            keys.add(normalized)
    return keys


def build_progressive_pair_schedule(
    candidates: list[dict[str, Any]],
    *,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seed: int = 20260615,
    max_shared_references: int = 2,
    max_reference_jaccard: float = 0.34,
) -> list[dict[str, Any]]:
    """Build easy, medium, and hard comparisons with discriminative evidence.

    Difficulty is the number of hypothesis positions that differ inside the
    same scientific task: one position is easy, two are medium, and three or
    more are hard. The default 100-question study uses a 30/40/30 split.
    """
    if len(candidates) < 2 or max_pairs <= 0:
        return []
    rng = random.Random(int(seed))
    exposure = {str(item["id"]): 0 for item in candidates}
    edges: list[dict[str, Any]] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left, right = candidates[left_index], candidates[right_index]
            if candidate_comparison_task(left) != candidate_comparison_task(right):
                continue
            left_key = candidate_semantic_key(left)
            if left_key and left_key == candidate_semantic_key(right):
                continue
            distance = candidate_structural_distance(left, right)
            if distance <= 0:
                continue
            left_references = candidate_reference_keys(left)
            right_references = candidate_reference_keys(right)
            shared_references = len(left_references & right_references)
            reference_union = len(left_references | right_references)
            reference_jaccard = shared_references / reference_union if reference_union else 0.0
            # Apply this guard only when both hypotheses have literature. Empty
            # evidence sets in generic/legacy studies must not erase all pairs.
            if left_references and right_references and (
                shared_references > max(0, int(max_shared_references))
                or reference_jaccard > max(0.0, float(max_reference_jaccard))
            ):
                continue
            difficulty = "easy" if distance == 1 else "medium" if distance == 2 else "hard"
            tie_breaker = rng.random()
            edges.append(
                {
                    "left_id": str(left["id"]),
                    "right_id": str(right["id"]),
                    "distance": distance,
                    "difficulty": difficulty,
                    "shared_references": shared_references,
                    "reference_jaccard": round(reference_jaccard, 4),
                    "_tie": tie_breaker,
                }
            )

    easy_quota = round(max_pairs * 0.30)
    hard_quota = round(max_pairs * 0.30)
    quotas = {
        "easy": easy_quota,
        "medium": max_pairs - easy_quota - hard_quota,
        "hard": hard_quota,
    }
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()

    def add_best(pool: list[dict[str, Any]], limit: int) -> None:
        for _ in range(limit):
            available = [
                edge
                for edge in pool
                if tuple(sorted((edge["left_id"], edge["right_id"]))) not in selected_keys
            ]
            if not available:
                return
            edge = min(
                available,
                key=lambda item: (
                    exposure[item["left_id"]] + exposure[item["right_id"]],
                    max(exposure[item["left_id"]], exposure[item["right_id"]]),
                    item["distance"] if item["difficulty"] != "hard" else -item["distance"],
                    item["shared_references"],
                    item["reference_jaccard"],
                    item["_tie"],
                ),
            )
            selected.append(edge)
            selected_keys.add(tuple(sorted((edge["left_id"], edge["right_id"]))))
            exposure[edge["left_id"]] += 1
            exposure[edge["right_id"]] += 1

    for stage in ("easy", "medium", "hard"):
        add_best([edge for edge in edges if edge["difficulty"] == stage], quotas[stage])

    # Guarantee at least one observation per candidate when the comparison budget permits it.
    uncovered = [candidate_id for candidate_id, count in exposure.items() if count == 0]
    for candidate_id in uncovered:
        if len(selected) >= max_pairs:
            break
        pool = [
            edge
            for edge in edges
            if candidate_id in (edge["left_id"], edge["right_id"])
            and tuple(sorted((edge["left_id"], edge["right_id"]))) not in selected_keys
        ]
        if not pool:
            continue
        edge = min(
            pool,
            key=lambda item: (
                item["distance"],
                item["shared_references"],
                item["reference_jaccard"],
                item["_tie"],
            ),
        )
        selected.append(edge)
        selected_keys.add(tuple(sorted((edge["left_id"], edge["right_id"]))))
        exposure[edge["left_id"]] += 1
        exposure[edge["right_id"]] += 1

    if len(selected) < max_pairs:
        add_best(edges, max_pairs - len(selected))

    stage_order = {"easy": 0, "medium": 1, "hard": 2}
    selected.sort(key=lambda item: (stage_order[item["difficulty"]], item["distance"], item["_tie"]))
    for index, pair in enumerate(selected, start=1):
        if rng.random() < 0.5:
            pair["left_id"], pair["right_id"] = pair["right_id"], pair["left_id"]
        pair["pair_id"] = f"pair-{index:03d}"
        pair.pop("_tie", None)
    return selected


def _curated_pair_schedule(
    payload: Any,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and return an approved schedule embedded in a bank."""
    if not isinstance(payload, dict):
        return []
    curation = payload.get("curation")
    if not isinstance(curation, dict) or curation.get("status") not in {
        "manually_reviewed",
        "external_validation_curated",
    }:
        return []
    raw_schedule = payload.get("pair_schedule")
    if not isinstance(raw_schedule, list):
        raise ValueError("A manually reviewed candidate bank must contain pair_schedule")

    by_id = {str(item["id"]): item for item in candidates}
    allow_same_disease_external_pairs = (
        curation.get("status") == "external_validation_curated"
        and payload.get("case_study") == "case1_tcp_external_validation"
    )

    def candidate_disease(candidate: dict[str, Any]) -> str:
        metadata = candidate.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        candidate_tuple = metadata.get("candidate_tuple")
        candidate_tuple = candidate_tuple if isinstance(candidate_tuple, dict) else {}
        return str(
            candidate_tuple.get("disease")
            or candidate_tuple.get("disease_id")
            or metadata.get("disease")
            or ""
        ).strip()

    seen_pairs: set[tuple[str, str]] = set()
    schedule: list[dict[str, Any]] = []
    for index, raw_pair in enumerate(raw_schedule, start=1):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"Curated pair {index} is not an object")
        left_id = str(raw_pair.get("left_id") or "").strip()
        right_id = str(raw_pair.get("right_id") or "").strip()
        if not left_id or not right_id or left_id == right_id:
            raise ValueError(f"Curated pair {index} has invalid hypothesis IDs")
        if left_id not in by_id or right_id not in by_id:
            raise ValueError(f"Curated pair {index} references a missing hypothesis")
        pair_key = tuple(sorted((left_id, right_id)))
        if pair_key in seen_pairs:
            raise ValueError(f"Curated pair {index} duplicates an earlier comparison")
        left_task = candidate_comparison_task(by_id[left_id])
        right_task = candidate_comparison_task(by_id[right_id])
        if left_task != right_task:
            pair_disease = str(raw_pair.get("disease") or "").strip()
            declared_task = str(raw_pair.get("comparison_task") or "").strip()
            same_disease_external_task = (
                allow_same_disease_external_pairs
                and pair_disease
                and candidate_disease(by_id[left_id]) == pair_disease
                and candidate_disease(by_id[right_id]) == pair_disease
                and declared_task == f"{pair_disease}|case-vs-control"
            )
            if not same_disease_external_task:
                raise ValueError(f"Curated pair {index} crosses scientific tasks")
        difficulty = str(raw_pair.get("difficulty") or "").strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"Curated pair {index} has invalid difficulty")
        manual_review = raw_pair.get("manual_review")
        if not isinstance(manual_review, dict) or manual_review.get("quality") != "keep":
            raise ValueError(f"Curated pair {index} has not passed manual quality review")

        seen_pairs.add(pair_key)
        schedule.append(
            {
                **raw_pair,
                "left_id": left_id,
                "right_id": right_id,
                "difficulty": difficulty,
                "pair_id": str(raw_pair.get("pair_id") or f"pair-{index:03d}"),
            }
        )
    if len(schedule) > MAX_CURATED_PAIRS:
        raise ValueError(f"Curated schedule exceeds the {MAX_CURATED_PAIRS}-pair safety limit")
    return schedule


def _public_candidate(candidate: dict[str, Any], condition: str) -> dict[str, Any]:
    out = dict(candidate)
    if condition == "manual":
        for field in SCORE_FIELDS:
            out.pop(field, None)
        metadata = out.get("metadata")
        if isinstance(metadata, dict):
            out["metadata"] = {
                key: value
                for key, value in metadata.items()
                if key not in SCORE_FIELDS and not key.endswith("_score") and "rank" not in key.lower()
            }
    return out


def _generator_score(candidate: dict[str, Any]) -> float:
    for key in ("composite_score", "critic_score", "plausibility_score", "evidence_score", "confidence_score"):
        try:
            return float(candidate.get(key))
        except (TypeError, ValueError):
            continue
    try:
        return -float(candidate.get("generator_rank") or candidate.get("rank"))
    except (TypeError, ValueError):
        return 0.0


class UserStudyService:
    """SQLite-backed study protocol and analysis service."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_study_root()
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "study.sqlite3"
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # The desktop reset flow can remove the study directory while an
        # already-running local backend is still winding down. Recreate both
        # the directory and schema on the next request so the service heals
        # instead of surfacing SQLite's opaque "unable to open database file".
        self.root.mkdir(parents=True, exist_ok=True)
        database_missing = not self.db_path.exists()
        conn = sqlite3.connect(str(self.db_path), timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            if database_missing:
                self._ensure_schema(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    case_study TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    candidate_source TEXT NOT NULL,
                    candidate_manifest_hash TEXT NOT NULL,
                    graph_source TEXT NOT NULL DEFAULT '',
                    graph_snapshot_hash TEXT NOT NULL DEFAULT '',
                    random_seed INTEGER NOT NULL,
                    protocol_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    submitted_at REAL,
                    active_seconds REAL,
                    wall_seconds REAL,
                    session_number INTEGER NOT NULL DEFAULT 1,
                    required_sessions INTEGER NOT NULL DEFAULT 1,
                    target_active_seconds REAL NOT NULL DEFAULT 600,
                    completion_reason TEXT NOT NULL DEFAULT '',
                    final_ranking_json TEXT,
                    pair_schedule_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS sessions_study_idx ON sessions(study_id, condition, status);
                CREATE TABLE IF NOT EXISTS session_candidates (
                    session_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    display_order INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, candidate_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    hypothesis_id TEXT,
                    client_elapsed_ms REAL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS events_session_idx ON events(session_id, event_id);
                CREATE TABLE IF NOT EXISTS execution_results (
                    study_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(study_id, hypothesis_id)
                );
            """
        )
        session_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "pair_schedule_json" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN pair_schedule_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "session_number" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN session_number INTEGER NOT NULL DEFAULT 1"
            )
        if "required_sessions" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN required_sessions INTEGER NOT NULL DEFAULT 1"
            )
        if "target_active_seconds" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN target_active_seconds REAL NOT NULL DEFAULT 600"
            )
        if "completion_reason" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN completion_reason TEXT NOT NULL DEFAULT ''"
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            self._ensure_schema(conn)

    def load_candidates(
        self, path: Path | str
    ) -> tuple[list[dict[str, Any]], str, Path, list[dict[str, Any]]]:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".json":
            raise ValueError("Candidate source must be a JSON file")
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Candidate source not found: {source}")
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
        candidates = [_normalize_candidate(item, idx) for idx, item in enumerate(_candidate_list(payload))]
        if not candidates:
            raise ValueError("Candidate source contains no hypotheses")
        ids = [item["id"] for item in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate source contains duplicate hypothesis IDs")
        pair_schedule = _curated_pair_schedule(payload, candidates)
        return candidates, _sha256_bytes(raw), source, pair_schedule

    def load_study_protocol(self, path: Path | str) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".json" or not source.is_file():
            raise FileNotFoundError(f"Candidate source not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        return _study_protocol(payload)

    def create_session(
        self,
        *,
        study_id: str,
        participant_id: str,
        condition: str,
        candidate_path: Path | str,
        case_study: str = "case1_transdiagnostic",
        random_seed: int = 0,
        graph_path: Path | str | None = None,
    ) -> dict[str, Any]:
        study_id = study_id.strip()
        participant_id = participant_id.strip()
        condition = condition.strip().lower()
        if not study_id or not participant_id:
            raise ValueError("study_id and participant_id are required")
        if condition not in CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}")
        candidates, manifest_hash, source, curated_pairs = self.load_candidates(candidate_path)
        protocol = self.load_study_protocol(source)
        required_sessions = int(protocol["required_sessions"])
        target_active_seconds = float(protocol["active_seconds_per_session"])
        if protocol["same_questions_for_all_participants"]:
            random_seed = int(protocol["shared_random_seed"])
        graph_source = Path(graph_path).expanduser().resolve() if graph_path else None
        order = list(range(len(candidates)))
        if condition in {"manual", "assisted"}:
            random.Random(int(random_seed)).shuffle(order)
        else:
            order.sort(key=lambda idx: (-_generator_score(candidates[idx]), candidates[idx]["id"]))
        if condition == "generator":
            session_number = 1
        else:
            with self._connect() as conn:
                prior_rows = conn.execute(
                    """SELECT session_number FROM sessions
                       WHERE study_id = ? AND participant_id = ? AND candidate_manifest_hash = ?
                         AND condition IN ('manual', 'assisted')
                         AND status IN ('active', 'completed')""",
                    (study_id, participant_id, manifest_hash),
                ).fetchall()
            occupied = {
                max(1, int(row["session_number"] or 1))
                for row in prior_rows
            }
            session_number = next(
                (number for number in range(1, required_sessions + 1) if number not in occupied),
                required_sessions + 1,
            )
            if session_number > required_sessions:
                raise ValueError(
                    f"All {required_sessions} required sessions are already active or completed "
                    f"for participant {participant_id}"
                )
        if condition == "generator":
            session_pairs: list[dict[str, Any]] = []
        elif required_sessions > 1 and curated_pairs:
            session_pairs = [
                pair
                for pair in curated_pairs
                if int(pair.get("session_number") or 0) == session_number
            ]
            if not session_pairs:
                session_pairs = [
                    pair
                    for index, pair in enumerate(curated_pairs)
                    if index % required_sessions == session_number - 1
                ]
        else:
            session_pairs = curated_pairs
        session_id = str(uuid.uuid4())
        started_at = _now()
        initial_status = "completed" if condition == "generator" else "active"
        ranking = [candidates[idx]["id"] for idx in order] if condition == "generator" else None
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions (
                    session_id, study_id, case_study, participant_id, condition,
                    candidate_source, candidate_manifest_hash, graph_source, graph_snapshot_hash,
                    random_seed, protocol_version, status, started_at, submitted_at,
                    active_seconds, wall_seconds, session_number, required_sessions,
                    target_active_seconds, completion_reason, final_ranking_json, pair_schedule_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    study_id,
                    case_study,
                    participant_id,
                    condition,
                    str(source),
                    manifest_hash,
                    str(graph_source or ""),
                    _sha256_file(graph_source),
                    int(random_seed),
                    PROTOCOL_VERSION,
                    initial_status,
                    started_at,
                    started_at if ranking else None,
                    0.0 if ranking else None,
                    0.0 if ranking else None,
                    session_number,
                    required_sessions,
                    target_active_seconds,
                    "generator_baseline" if ranking else "",
                    _json(ranking) if ranking else None,
                    _json(session_pairs),
                ),
            )
            conn.executemany(
                "INSERT INTO session_candidates(session_id, candidate_id, display_order, payload_json) VALUES (?, ?, ?, ?)",
                [(session_id, candidates[idx]["id"], pos, _json(candidates[idx])) for pos, idx in enumerate(order)],
            )
            conn.execute(
                "INSERT INTO events(session_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    "session_started",
                    _json(
                        {
                            "condition": condition,
                            "generator_scores_visible": condition == "assisted",
                            "session_number": session_number,
                            "required_sessions": required_sessions,
                            "target_active_seconds": target_active_seconds,
                        }
                    ),
                    started_at,
                ),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str, *, include_events: bool = False) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            candidate_rows = conn.execute(
                "SELECT payload_json FROM session_candidates WHERE session_id = ? ORDER BY display_order",
                (session_id,),
            ).fetchall()
            condition = str(row["condition"])
            out = dict(row)
            out["candidates"] = [
                _public_candidate(json.loads(item["payload_json"]), condition) for item in candidate_rows
            ]
            curated_pairs = json.loads(row["pair_schedule_json"] or "[]")
            out["pairs"] = (
                curated_pairs
                or build_progressive_pair_schedule(
                    out["candidates"],
                    max_pairs=DEFAULT_MAX_PAIRS,
                    seed=int(row["random_seed"]),
                )
            ) if condition != "generator" else []
            out["final_ranking"] = json.loads(row["final_ranking_json"]) if row["final_ranking_json"] else None
            out.pop("final_ranking_json", None)
            out.pop("pair_schedule_json", None)
            if include_events:
                out["events"] = [
                    {
                        **dict(event),
                        "payload": json.loads(event["payload_json"]),
                    }
                    for event in conn.execute(
                        "SELECT * FROM events WHERE session_id = ? ORDER BY event_id", (session_id,)
                    ).fetchall()
                ]
                for event in out["events"]:
                    event.pop("payload_json", None)
            return out

    def delete_active_session(self, session_id: str) -> dict[str, Any]:
        """Delete one unfinished session and its dependent records.

        Completed sessions are immutable study records and must never be
        removed through the resume/restart workflow.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT session_id, study_id, participant_id, condition, status,
                          session_number, required_sessions
                   FROM sessions WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if str(row["status"]) != "active":
                raise ValueError("Only an unfinished active session can be deleted")
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            deleted = dict(row)
        deleted["deleted"] = True
        return deleted

    def append_events(self, session_id: str, events: Iterable[dict[str, Any]]) -> int:
        with self._connect() as conn:
            session = conn.execute("SELECT status FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if session is None:
                raise KeyError(session_id)
            rows = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type") or event.get("event_type") or "").strip()
                if not event_type:
                    continue
                rows.append(
                    (
                        session_id,
                        event_type[:80],
                        str(event.get("hypothesis_id") or "") or None,
                        float(event["elapsed_ms"]) if event.get("elapsed_ms") is not None else None,
                        _json(event.get("payload") if isinstance(event.get("payload"), dict) else {}),
                        _now(),
                    )
                )
            conn.executemany(
                """INSERT INTO events(
                    session_id, event_type, hypothesis_id, client_elapsed_ms, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def submit_session(
        self,
        session_id: str,
        *,
        ranking: list[str],
        active_seconds: float,
        wall_seconds: float,
        buckets: dict[str, str] | None = None,
        completion_reason: str = "submitted",
    ) -> dict[str, Any]:
        completion_reason = str(completion_reason or "submitted").strip()[:80]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT candidate_id FROM session_candidates WHERE session_id = ?", (session_id,)
            ).fetchall()
            if not rows:
                raise KeyError(session_id)
            expected = {str(row["candidate_id"]) for row in rows}
            cleaned = [str(item) for item in ranking]
            if len(cleaned) != len(expected) or set(cleaned) != expected:
                raise ValueError("Final ranking must contain every candidate exactly once")
            submitted_at = _now()
            conn.execute(
                """UPDATE sessions SET status = 'completed', submitted_at = ?, active_seconds = ?,
                   wall_seconds = ?, completion_reason = ?, final_ranking_json = ?
                   WHERE session_id = ?""",
                (
                    submitted_at,
                    max(0.0, float(active_seconds)),
                    max(0.0, float(wall_seconds)),
                    completion_reason,
                    _json(cleaned),
                    session_id,
                ),
            )
            conn.execute(
                """INSERT INTO events(
                    session_id, event_type, payload_json, created_at
                ) VALUES (?, 'ranking_submitted', ?, ?)""",
                (
                    session_id,
                    _json(
                        {
                            "ranking": cleaned,
                            "buckets": buckets or {},
                            "completion_reason": completion_reason,
                        }
                    ),
                    submitted_at,
                ),
            )
        return self.get_session(session_id)

    def list_sessions(self, study_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT session_id, study_id, case_study, participant_id, condition, status,
                          candidate_manifest_hash, graph_snapshot_hash, random_seed, protocol_version,
                          started_at, submitted_at, active_seconds, wall_seconds, session_number,
                          required_sessions, target_active_seconds, completion_reason
                   FROM sessions WHERE study_id = ? ORDER BY started_at""",
                (study_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def import_execution_results(self, study_id: str, path: Path | str) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".json" or not source.exists():
            raise FileNotFoundError(f"Execution result JSON not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        items = _candidate_list(payload)
        if not items and isinstance(payload, dict) and isinstance(payload.get("execution_results"), list):
            items = [dict(item) for item in payload["execution_results"] if isinstance(item, dict)]
        imported = 0
        with self._connect() as conn:
            for item in items:
                hypothesis_id = str(item.get("hypothesis_id") or item.get("id") or "").strip()
                status = str(item.get("status") or item.get("outcome") or "").strip().lower()
                aliases = {"success": "confirmed", "negative": "not_confirmed", "failed": "execution_failed", "error": "execution_failed"}
                status = aliases.get(status, status)
                if not hypothesis_id or status not in RESULT_STATUSES:
                    continue
                duration = item.get("duration_seconds", item.get("runtime_seconds", item.get("experiment_seconds", 0)))
                try:
                    duration = max(0.0, float(duration))
                except (TypeError, ValueError):
                    continue
                conn.execute(
                    """INSERT INTO execution_results(
                        study_id, hypothesis_id, status, duration_seconds, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(study_id, hypothesis_id) DO UPDATE SET
                        status=excluded.status, duration_seconds=excluded.duration_seconds,
                        payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
                    (study_id, hypothesis_id, status, duration, _json(item), _now()),
                )
                imported += 1
        return {"study_id": study_id, "source": str(source), "imported": imported}

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def results(self, study_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            sessions = conn.execute(
                "SELECT * FROM sessions WHERE study_id = ? AND status = 'completed' ORDER BY started_at",
                (study_id,),
            ).fetchall()
            result_rows = conn.execute(
                "SELECT * FROM execution_results WHERE study_id = ?", (study_id,)
            ).fetchall()
        runtime = {str(row["hypothesis_id"]): dict(row) for row in result_rows}
        duration_bases: set[str] = set()
        for item in runtime.values():
            try:
                payload = json.loads(item.get("payload_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            basis = str(payload.get("duration_basis") or "").strip()
            if basis:
                duration_bases.add(basis)
        confirmed_total = sum(1 for item in runtime.values() if item["status"] == "confirmed")
        quantiles = [index / 10 for index in range(1, 11)]
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in sorted(CONDITIONS)}
        active_by_condition: dict[str, list[float]] = {condition: [] for condition in sorted(CONDITIONS)}
        for row in sessions:
            condition = str(row["condition"])
            ranking = json.loads(row["final_ranking_json"] or "[]")
            active = float(row["active_seconds"] or 0.0)
            active_by_condition.setdefault(condition, []).append(active)
            discovered = 0
            experiment_time = 0.0
            points: list[dict[str, Any]] = []
            targets = {q: max(1, math.ceil(confirmed_total * q)) for q in quantiles} if confirmed_total else {}
            reached: dict[float, float] = {}
            for hypothesis_id in ranking:
                result = runtime.get(str(hypothesis_id))
                if not result:
                    continue
                experiment_time += float(result["duration_seconds"])
                if result["status"] == "confirmed":
                    discovered += 1
                for q, target in targets.items():
                    if q not in reached and discovered >= target:
                        reached[q] = experiment_time
            for q in quantiles:
                seconds = reached.get(q)
                points.append(
                    {
                        "fraction": q,
                        "confirmed_target": targets.get(q, 0),
                        "experiment_seconds": seconds,
                        "end_to_end_seconds": (active + seconds) if seconds is not None else None,
                    }
                )
            by_condition.setdefault(condition, []).append(
                {"session_id": row["session_id"], "participant_id": row["participant_id"], "points": points}
            )

        summaries: dict[str, Any] = {}
        curves: dict[str, list[dict[str, Any]]] = {}
        for condition in sorted(by_condition):
            active_values = active_by_condition.get(condition, [])
            summaries[condition] = {
                "n": len(active_values),
                "mean_active_seconds": self._mean(active_values),
                "median_active_seconds": statistics.median(active_values) if active_values else None,
            }
            curve = []
            for idx, q in enumerate(quantiles):
                experiment = [
                    float(session["points"][idx]["experiment_seconds"])
                    for session in by_condition[condition]
                    if session["points"][idx]["experiment_seconds"] is not None
                ]
                end_to_end = [
                    float(session["points"][idx]["end_to_end_seconds"])
                    for session in by_condition[condition]
                    if session["points"][idx]["end_to_end_seconds"] is not None
                ]
                curve.append(
                    {
                        "fraction": q,
                        "experiment_seconds": self._mean(experiment),
                        "end_to_end_seconds": self._mean(end_to_end),
                        "n": len(experiment),
                    }
                )
            curves[condition] = curve

        manual = summaries.get("manual", {}).get("mean_active_seconds")
        assisted = summaries.get("assisted", {}).get("mean_active_seconds")
        saving = ((manual - assisted) / manual * 100.0) if manual and assisted is not None else None
        return {
            "study_id": study_id,
            "protocol_version": PROTOCOL_VERSION,
            "sessions": self.list_sessions(study_id),
            "execution_results": len(runtime),
            "confirmed_hypotheses": confirmed_total,
            "duration_basis": (
                next(iter(duration_bases))
                if len(duration_bases) == 1
                else "seconds"
            ),
            "ranking_time": summaries,
            "ranking_time_saving_percent": saving,
            "curves": curves,
        }
