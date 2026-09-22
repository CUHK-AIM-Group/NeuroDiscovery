"""Build a compact Case Study 1 index from a large NeuroOracle JSON graph."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import IO, Any, Iterable

from neurooracle.src.case_study_scope import (
    canonical_case_study_id,
    claim_case_study_ids_from_dict,
)

CACHE_SCHEMA_VERSION = 2
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_CASE_STUDY_ID = "case1_transdiagnostic"


def _normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


class _JsonStream:
    """Incrementally decode one JSON value at a time without loading the file."""

    def __init__(self, handle: IO[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _read_more(self) -> bool:
        if self.eof:
            return False
        if self.pos:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _skip_whitespace(self) -> None:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or not self._read_more():
                return

    def peek(self) -> str:
        self._skip_whitespace()
        if self.pos >= len(self.buffer):
            raise ValueError("Unexpected end of JSON input")
        return self.buffer[self.pos]

    def consume(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r} in JSON stream, found {actual!r}")
        self.pos += 1

    def decode(self) -> Any:
        self._skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if not self._read_more():
                    raise
            else:
                self.pos = end
                return value

    def iter_object_items(self) -> Iterable[tuple[str, Any]]:
        self.consume("{")
        if self.peek() == "}":
            self.consume("}")
            return
        while True:
            key = self.decode()
            if not isinstance(key, str):
                raise ValueError("JSON object key is not a string")
            self.consume(":")
            yield key, self.decode()
            delimiter = self.peek()
            if delimiter == "}":
                self.consume("}")
                return
            self.consume(",")

    def iter_array_items(self) -> Iterable[Any]:
        self.consume("[")
        if self.peek() == "]":
            self.consume("]")
            return
        while True:
            yield self.decode()
            delimiter = self.peek()
            if delimiter == "]":
                self.consume("]")
                return
            self.consume(",")


def _query_signature(query_terms: Iterable[str] | None) -> tuple[set[str] | None, str]:
    if query_terms is None:
        return None, "all"
    normalized = {_normalize_text(term) for term in query_terms}
    normalized.discard("")
    digest = hashlib.sha256("\n".join(sorted(normalized)).encode("utf-8")).hexdigest()
    return normalized, digest


def case1_kg_cache_path(
    path: Path,
    query_terms: Iterable[str] | None,
    case_study_id: str = DEFAULT_CASE_STUDY_ID,
) -> Path:
    _, digest = _query_signature(query_terms)
    scope = canonical_case_study_id(case_study_id)
    if not scope:
        raise ValueError(f"unknown case-study ID: {case_study_id!r}")
    scope_digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:8]
    return path.with_name(
        f"{path.name}.case1-index-v{CACHE_SCHEMA_VERSION}-{scope_digest}-{digest[:16]}.pkl"
    )


def _edge_relation(edge: dict[str, Any]) -> str:
    return _normalize_text(
        edge.get("relation_type")
        or edge.get("relation")
        or edge.get("predicate")
        or edge.get("edge_type")
        or ""
    )


def _is_semantic_support_edge(edge: dict[str, Any], relation: str) -> tuple[bool, str]:
    if relation == "about":
        return False, "about"
    evidence_state = _normalize_text(
        edge.get("evidence_state") or edge.get("status") or ""
    )
    metadata = edge.get("metadata")
    is_negated = isinstance(metadata, dict) and bool(metadata.get("negated"))
    if "contradict" in relation or "contradict" in evidence_state or is_negated:
        return False, "contradicted"
    return True, "semantic"


def _concept_names(concept: dict[str, Any]) -> list[object]:
    names: list[object] = [concept.get("preferred_name", "")]
    aliases = concept.get("aliases") or []
    if isinstance(aliases, list):
        names.extend(aliases[:30])
    return names


def build_case1_kg_index_payload(
    path: Path,
    query_terms: Iterable[str] | None = None,
    *,
    case_study_id: str = DEFAULT_CASE_STUDY_ID,
) -> dict[str, Any]:
    """Build global and task-specific support for the CS1 candidate space."""

    wanted_names, query_digest = _query_signature(query_terms)
    scope = canonical_case_study_id(case_study_id)
    if not scope:
        raise ValueError(f"unknown case-study ID: {case_study_id!r}")
    name_to_ids: dict[str, set[str]] = defaultdict(set)
    relevant_ids: set[str] = set()
    degrees: Counter[str] = Counter()
    adjacency: dict[str, set[str]] = defaultdict(set)
    directed_support: dict[tuple[str, str], float] = {}
    scoped_degrees: Counter[str] = Counter()
    scoped_adjacency: dict[str, set[str]] = defaultdict(set)
    scoped_directed_support: dict[tuple[str, str], float] = {}
    stats = {
        "concepts_seen": 0,
        "matched_concepts": 0,
        "edges_seen": 0,
        "semantic_edges_seen": 0,
        "about_edges_skipped": 0,
        "contradicted_edges_skipped": 0,
        "malformed_edges_skipped": 0,
        "case_study_id": scope,
        "case_study_semantic_edges_seen": 0,
        "query_digest": query_digest,
    }

    with _open_text(path) as handle:
        stream = _JsonStream(handle)
        stream.consume("{")
        while True:
            if stream.peek() == "}":
                stream.consume("}")
                break
            top_level_key = stream.decode()
            if not isinstance(top_level_key, str):
                raise ValueError("Top-level JSON object key is not a string")
            stream.consume(":")

            if top_level_key == "concepts":
                for concept_id, raw_concept in stream.iter_object_items():
                    stats["concepts_seen"] += 1
                    if not isinstance(raw_concept, dict):
                        continue
                    matched_names: set[str] = set()
                    for name in _concept_names(raw_concept):
                        normalized = _normalize_text(name)
                        if normalized and (
                            wanted_names is None or normalized in wanted_names
                        ):
                            matched_names.add(normalized)
                    if not matched_names:
                        continue
                    cid = str(concept_id)
                    relevant_ids.add(cid)
                    stats["matched_concepts"] += 1
                    for normalized in matched_names:
                        name_to_ids[normalized].add(cid)
            elif top_level_key == "edges":
                for raw_edge in stream.iter_array_items():
                    stats["edges_seen"] += 1
                    if not isinstance(raw_edge, dict):
                        stats["malformed_edges_skipped"] += 1
                        continue
                    relation = _edge_relation(raw_edge)
                    include, edge_kind = _is_semantic_support_edge(
                        raw_edge,
                        relation,
                    )
                    if not include:
                        stats[f"{edge_kind}_edges_skipped"] += 1
                        continue
                    source = raw_edge.get("source_id") or raw_edge.get("source")
                    target = raw_edge.get("target_id") or raw_edge.get("target")
                    if not source or not target:
                        stats["malformed_edges_skipped"] += 1
                        continue
                    source_id = str(source)
                    target_id = str(target)
                    source_relevant = wanted_names is None or source_id in relevant_ids
                    target_relevant = wanted_names is None or target_id in relevant_ids
                    stats["semantic_edges_seen"] += 1
                    edge_case_study_ids = set(
                        claim_case_study_ids_from_dict(raw_edge)
                    )
                    is_scoped = scope in edge_case_study_ids
                    if is_scoped:
                        stats["case_study_semantic_edges_seen"] += 1

                    if source_relevant:
                        degrees[source_id] += 1
                        adjacency[source_id].add(target_id)
                        if is_scoped:
                            scoped_degrees[source_id] += 1
                            scoped_adjacency[source_id].add(target_id)
                    if target_relevant:
                        degrees[target_id] += 1
                        adjacency[target_id].add(source_id)
                        if is_scoped:
                            scoped_degrees[target_id] += 1
                            scoped_adjacency[target_id].add(source_id)
                    if not (source_relevant and target_relevant):
                        continue

                    raw_confidence = (
                        raw_edge.get("confidence")
                        if raw_edge.get("confidence") is not None
                        else raw_edge.get("weight", 1.0)
                    )
                    try:
                        confidence = float(raw_confidence)
                    except (TypeError, ValueError):
                        confidence = 1.0
                    confidence = min(1.0, max(0.05, confidence))
                    key = (source_id, target_id)
                    directed_support[key] = max(
                        confidence,
                        directed_support.get(key, 0.0),
                    )
                    if is_scoped:
                        scoped_directed_support[key] = max(
                            confidence,
                            scoped_directed_support.get(key, 0.0),
                        )
            else:
                stream.decode()

            delimiter = stream.peek()
            if delimiter == "}":
                stream.consume("}")
                break
            stream.consume(",")

    name_to_degree: dict[str, int] = {}
    name_to_scoped_degree: dict[str, int] = {}
    for name, concept_ids in name_to_ids.items():
        name_to_degree[name] = max(
            (int(degrees.get(concept_id, 0)) for concept_id in concept_ids),
            default=0,
        )
        name_to_scoped_degree[name] = max(
            (int(scoped_degrees.get(concept_id, 0)) for concept_id in concept_ids),
            default=0,
        )

    return {
        "degrees": dict(degrees),
        "name_to_ids": {
            name: tuple(sorted(concept_ids))
            for name, concept_ids in name_to_ids.items()
        },
        "name_to_degree": name_to_degree,
        "adjacency": dict(adjacency),
        "directed_support": directed_support,
        "case_study_id": scope,
        "scoped_degrees": dict(scoped_degrees),
        "name_to_scoped_degree": name_to_scoped_degree,
        "scoped_adjacency": dict(scoped_adjacency),
        "scoped_directed_support": scoped_directed_support,
        "stats": stats,
    }


def load_case1_kg_index_payload(
    path: Path,
    query_terms: Iterable[str] | None = None,
    *,
    cache: bool = True,
    case_study_id: str = DEFAULT_CASE_STUDY_ID,
) -> dict[str, Any]:
    """Load a source-signature-aware local cache or build it by streaming JSON."""

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"knowledge graph does not exist: {path}")
    scope = canonical_case_study_id(case_study_id)
    if not scope:
        raise ValueError(f"unknown case-study ID: {case_study_id!r}")
    normalized_terms, query_digest = _query_signature(query_terms)
    cache_path = case1_kg_cache_path(path, normalized_terms, scope)
    stat = path.stat()
    signature = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "query_digest": query_digest,
        "case_study_id": scope,
    }

    if cache and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if all(cached.get(key) == value for key, value in signature.items()):
                payload = cached.get("payload")
                if isinstance(payload, dict):
                    return payload
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            pass

    payload = build_case1_kg_index_payload(
        path,
        normalized_terms,
        case_study_id=scope,
    )
    if not cache:
        return payload

    cache_record = {**signature, "payload": payload}
    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(cache_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return payload
