"""Execute one official autoresearch adapter and emit a strict SearchPolicy.

The native framework orchestration is case-study agnostic. Case-specific task
files define the public registry, executable ID grammar, policy schema, domain
roles, and ontology fields. Exact registered candidate IDs are accepted by
default. A task may explicitly enable a frozen public-coordinate compiler for
native final artifacts that cannot carry executable IDs in their schema.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Annotated, Any
from xml.etree import ElementTree

PRIMARY_METHODS = {
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
}
SUPPLEMENTARY_METHODS = {"data_to_paper", "openscholar_rag"}
METHODS = PRIMARY_METHODS | SUPPLEMENTARY_METHODS

_PUBMED_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}
_PUBMED_LOCK = threading.Lock()
_PUBMED_LAST_REQUEST = 0.0
_RETRIEVAL_AUDIT: list[dict[str, Any]] = []


def _xml_text(element: Any) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _pubmed_query_variants(query: str) -> tuple[str, ...]:
    """Progressively relax natural-language searches that PubMed treats as AND queries."""

    normalized = " ".join(re.sub(r"[|_/]+", " ", str(query)).split())[:500]
    words = normalized.split()
    stopwords = {
        "a",
        "alteration",
        "alterations",
        "an",
        "and",
        "candidate",
        "case-control",
        "direction",
        "experiment",
        "for",
        "hypothesis",
        "in",
        "is",
        "left",
        "mean",
        "not",
        "outcome-blind",
        "partial",
        "registered",
        "reproducibility",
        "reproducible",
        "right",
        "roi",
        "specified",
        "test",
        "tests",
        "the",
        "top",
    }
    filtered_words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", normalized)
        if word.casefold() not in stopwords
    ]
    filtered_casefold = {word.casefold() for word in filtered_words}
    if "depression" in filtered_casefold:
        filtered_words = [word for word in filtered_words if word.casefold() != "mdd"]
    variants: list[str] = []
    for source_words, sizes in (
        (words, (len(words),)),
        (filtered_words, (12, 8, 6, 4)),
        (words, (12, 8, 6, 4)),
    ):
        for size in sizes:
            if not source_words:
                break
            candidate = " ".join(source_words[: min(size, len(source_words))])
            if candidate and candidate not in variants:
                variants.append(candidate)
    return tuple(variants)


def _pubmed_fallback_papers(query: str, limit: int) -> list[dict[str, Any]]:
    """Retrieve public PubMed abstracts when a framework's primary search is unavailable."""

    import requests

    query_variants = _pubmed_query_variants(query)
    normalized_query = query_variants[0] if query_variants else ""
    cache_key = (normalized_query.casefold(), int(limit))
    if cache_key in _PUBMED_CACHE:
        return list(_PUBMED_CACHE[cache_key])

    global _PUBMED_LAST_REQUEST
    started = time.time()
    try:
        with _PUBMED_LOCK:
            delay = 0.36 - (time.time() - _PUBMED_LAST_REQUEST)
            if delay > 0:
                time.sleep(delay)
            common = {
                "db": "pubmed",
                "tool": "NeuroClawBenchmark",
            }
            email = os.environ.get("NEUROCLAW_PUBMED_EMAIL", "").strip()
            if email:
                common["email"] = email
            pmids: list[str] = []
            search_attempts: list[dict[str, Any]] = []
            used_query = normalized_query
            for candidate_query in query_variants:
                delay = 0.36 - (time.time() - _PUBMED_LAST_REQUEST)
                if delay > 0:
                    time.sleep(delay)
                search = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={
                        **common,
                        "term": candidate_query,
                        "retmax": max(1, int(limit)),
                        "sort": "relevance",
                        "retmode": "json",
                    },
                    timeout=30,
                )
                _PUBMED_LAST_REQUEST = time.time()
                search.raise_for_status()
                pmids = [
                    str(value)
                    for value in search.json()["esearchresult"]["idlist"]
                ]
                search_attempts.append(
                    {
                        "query_sha256": hashlib.sha256(
                            candidate_query.encode()
                        ).hexdigest(),
                        "terms": len(candidate_query.split()),
                        "pmids": len(pmids),
                    }
                )
                used_query = candidate_query
                if pmids:
                    break
            if not pmids:
                papers: list[dict[str, Any]] = []
            else:
                delay = 0.36 - (time.time() - _PUBMED_LAST_REQUEST)
                if delay > 0:
                    time.sleep(delay)
                fetch = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={
                        **common,
                        "id": ",".join(pmids),
                        "retmode": "xml",
                    },
                    timeout=45,
                )
                _PUBMED_LAST_REQUEST = time.time()
                fetch.raise_for_status()
                root = ElementTree.fromstring(fetch.content)
                papers = []
                for article in root.findall(".//PubmedArticle"):
                    citation = article.find("MedlineCitation")
                    pmid = _xml_text(citation.find("PMID") if citation is not None else None)
                    journal_article = (
                        citation.find("Article") if citation is not None else None
                    )
                    if journal_article is None:
                        continue
                    title = _xml_text(journal_article.find("ArticleTitle"))
                    abstract = " ".join(
                        _xml_text(value)
                        for value in journal_article.findall("Abstract/AbstractText")
                        if _xml_text(value)
                    )
                    journal = _xml_text(journal_article.find("Journal/Title"))
                    year = _xml_text(
                        journal_article.find("Journal/JournalIssue/PubDate/Year")
                    )
                    if not year:
                        year = _xml_text(
                            journal_article.find(
                                "Journal/JournalIssue/PubDate/MedlineDate"
                            )
                        )[:4]
                    authors = []
                    for author in journal_article.findall("AuthorList/Author"):
                        name = " ".join(
                            part
                            for part in (
                                _xml_text(author.find("ForeName")),
                                _xml_text(author.find("LastName")),
                            )
                            if part
                        )
                        if name:
                            authors.append({"name": name})
                    papers.append(
                        {
                            "paperId": f"PMID:{pmid}",
                            "title": title,
                            "authors": authors,
                            "venue": journal,
                            "year": int(year) if year.isdigit() else year,
                            "abstract": abstract,
                            "citationCount": 0,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        }
                    )
        _PUBMED_CACHE[cache_key] = papers
        _RETRIEVAL_AUDIT.append(
            {
                "source": "pubmed_fallback",
                "status": "complete",
                "query_sha256": hashlib.sha256(normalized_query.encode()).hexdigest(),
                "used_query_sha256": hashlib.sha256(used_query.encode()).hexdigest(),
                "query_relaxed": used_query != normalized_query,
                "search_attempts": search_attempts,
                "papers": len(papers),
                "duration_seconds": time.time() - started,
            }
        )
        return list(papers)
    except Exception as exc:
        _RETRIEVAL_AUDIT.append(
            {
                "source": "pubmed_fallback",
                "status": "failed",
                "query_sha256": hashlib.sha256(normalized_query.encode()).hexdigest(),
                "error_type": type(exc).__name__,
                "duration_seconds": time.time() - started,
            }
        )
        return []


def _request_timeout_seconds() -> float:
    try:
        return max(
            30.0,
            float(os.environ.get("CASE_STUDY_LLM_REQUEST_TIMEOUT", "600")),
        )
    except ValueError:
        return 600.0


def _openai_strict_json_schema(value: Any) -> Any:
    """Return an OpenAI-strict copy of a framework-provided JSON schema.

    Open Co-Scientist marks its response schemas as strict at runtime, but a
    small number of nested objects retain optional properties or allow free
    form additional properties.  The Responses API rejects those schemas
    before generation.  Strict schemas require every declared property in the
    object's ``required`` list and ``additionalProperties`` set to false.
    """

    if isinstance(value, list):
        return [_openai_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _openai_strict_json_schema(item) for key, item in value.items()
    }
    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["required"] = list(properties)
        normalized["additionalProperties"] = False
    return normalized


def _open_coscientist_max_tokens() -> int:
    try:
        return max(
            1024,
            int(os.environ.get("OPEN_COSCIENTIST_MAX_OUTPUT_TOKENS", "8192")),
        )
    except ValueError:
        return 8192


def _open_coscientist_debate_turns() -> int:
    try:
        return min(
            5,
            max(3, int(os.environ.get("OPEN_COSCIENTIST_DEBATE_TURNS", "3"))),
        )
    except ValueError:
        return 3


def _open_coscientist_pool_size(task: dict[str, Any], target: int) -> int:
    factor = float(task.get("open_coscientist_overgeneration_factor", 1.0))
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("Open Co-Scientist overgeneration factor must be finite and >= 1")
    return max(2, int(math.ceil(target * factor)))


OPEN_COSCIENTIST_CACHE_SCHEMA = "neuroclaw.open-coscientist-trial-cache.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _streaming_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _python_source_tree_sha256(root: Path) -> str:
    """Hash imported Python sources without depending on Git cleanliness."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _ancestor_json_identity(start: Path, filename: str) -> dict[str, Any] | None:
    for directory in (start, *start.parents):
        candidate = directory / filename
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        identity: dict[str, Any] = {
            "sha256": _streaming_file_sha256(candidate),
        }
        if filename == "canonical_kg_release.json":
            identity.update(
                {
                    "release_id": payload.get("release_id"),
                    "status": payload.get("status"),
                    "knowledge_graph_sha256": (
                        (payload.get("files") or {}).get("knowledge_graph") or {}
                    ).get("sha256"),
                    "extracted_claims_sha256": (
                        (payload.get("files") or {}).get("extracted_claims") or {}
                    ).get("sha256"),
                    "current_state_sha256": (
                        (payload.get("files") or {}).get("current_state") or {}
                    ).get("sha256"),
                }
            )
        return identity
    return None


def _open_coscientist_cache_stats(cache_dir: Path) -> dict[str, Any]:
    files = list(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    return {
        "cache_files": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _open_coscientist_cache_identity(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    model_name: str,
    native_pool_size: int,
) -> dict[str, Any]:
    registry_path = Path(str(task["public_registry_path"]))
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    upstream_source_root = args.repo / "src" / "open_coscientist"
    if not upstream_source_root.is_dir():
        raise FileNotFoundError(upstream_source_root)
    return {
        "schema_version": OPEN_COSCIENTIST_CACHE_SCHEMA,
        "scope": {
            "case_study_id": _case_study_id(task),
            "method": "open_coscientist",
            "trial": int(task["trial"]),
            "round_index": int(task.get("round_index", 0)),
        },
        "sealed_inputs": {
            "task_sha256": _canonical_sha256(task),
            "public_registry_sha256": _streaming_file_sha256(registry_path),
            "public_registry_size_bytes": registry_path.stat().st_size,
            "canonical_kg_release": _ancestor_json_identity(
                args.out, "canonical_kg_release.json"
            ),
            "formal_protocol_manifest": _ancestor_json_identity(
                args.out, "official_adapter_manifest.json"
            ),
        },
        "model_configuration": {
            "declared_model": str(args.model),
            "litellm_model": model_name,
            "reasoning_effort": str(args.reasoning_effort),
            "thinking_enabled": True,
            "response_storage": False,
            "approved_route_scope": "deepseek-v4-pro approved channels",
        },
        "native_workflow_configuration": {
            "native_retrieval_enabled": bool(args.enable_native_retrieval),
            "tool_calling_generation_enabled": bool(
                task.get("open_coscientist_tool_generation_enabled", False)
            ),
            "max_iterations": max(
                0,
                int(task.get("open_coscientist_max_iterations", 1)),
            ),
            "initial_hypotheses_count": native_pool_size,
            "evolution_max_count": native_pool_size,
            "debate_turns": _open_coscientist_debate_turns(),
            "max_output_tokens_floor": _open_coscientist_max_tokens(),
            "max_concurrent_llm_calls": str(
                os.environ.get("OPEN_COSCIENTIST_MAX_CONCURRENT_LLM_CALLS", "4")
            ),
            "rate_limit_retries": str(
                os.environ.get("OPEN_COSCIENTIST_RATE_LIMIT_RETRIES", "8")
            ),
            "transient_retries": str(
                os.environ.get("OPEN_COSCIENTIST_TRANSIENT_RETRIES", "")
            ),
            "request_timeout_seconds": _request_timeout_seconds(),
            "json_schema_strict": os.environ.get(
                "OPENAI_JSON_SCHEMA_STRICT", ""
            ).lower()
            in {"1", "true", "yes"},
            "json_schema_embedded_in_prompt": os.environ.get(
                "OPENAI_EMBED_JSON_SCHEMA_IN_PROMPT", ""
            ).lower()
            in {"1", "true", "yes"},
        },
        "implementation": {
            "adapter_source_sha256": _streaming_file_sha256(Path(__file__)),
            "upstream_python_source_tree_sha256": _python_source_tree_sha256(
                upstream_source_root
            ),
        },
        "cache_policy": {
            "scope": "single method + trial + round + sealed identity",
            "successful_nonempty_responses_only": True,
            "exact_request_key_fields": [
                "prompt",
                "model",
                "temperature",
                "max_tokens",
                "tools",
                "json_schema",
                "force_json",
            ],
            "node_cache_enabled": False,
            "cross_trial_reuse": False,
            "cross_seed_reuse": False,
            "cross_kg_reuse": False,
        },
    }


def _prepare_open_coscientist_trial_cache(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    model_name: str,
    native_pool_size: int,
) -> tuple[Path, Path, str]:
    """Create or validate an exact-request cache isolated to one formal trial."""

    cache_dir = args.out / "open_coscientist_trial_cache"
    manifest_path = args.out / "open_coscientist_cache_manifest.json"
    identity = _open_coscientist_cache_identity(
        task,
        args,
        model_name=model_name,
        native_pool_size=native_pool_size,
    )
    identity_sha256 = _canonical_sha256(identity)
    existing: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Open Co-Scientist cache manifest is unreadable: {manifest_path}"
            ) from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"Open Co-Scientist cache manifest is not an object: {manifest_path}"
            )
        existing = loaded
        if (
            existing.get("identity_sha256") != identity_sha256
            or existing.get("identity") != identity
        ):
            raise RuntimeError(
                "Open Co-Scientist trial cache identity mismatch; archive this trial "
                "directory and start a fresh trial instead of reusing responses"
            )
    elif cache_dir.is_dir() and any(cache_dir.iterdir()):
        raise RuntimeError(
            "Open Co-Scientist cache files exist without a sealed identity manifest"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    manifest = dict(existing or {})
    manifest.update(
        {
            "schema_version": OPEN_COSCIENTIST_CACHE_SCHEMA,
            "identity_sha256": identity_sha256,
            "identity": identity,
            "cache_dir": str(cache_dir),
            "created_at": manifest.get("created_at", now),
            "last_opened_at": now,
            "open_count": int(manifest.get("open_count", 0)) + 1,
            "status": "active",
            "stats_at_open": _open_coscientist_cache_stats(cache_dir),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return cache_dir, manifest_path, identity_sha256


def _update_open_coscientist_cache_manifest(
    manifest_path: Path,
    cache_dir: Path,
    *,
    status: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "last_closed_at": time.time(),
            "status": status,
            "stats_at_close": _open_coscientist_cache_stats(cache_dir),
        }
    )
    _write_json_atomic(manifest_path, manifest)


def _install_open_coscientist_cache_audit(
    cache: Any,
    *,
    out_dir: Path,
    identity_sha256: str,
) -> None:
    """Audit cache hits/stores by request hash without recording prompts or outputs."""

    if getattr(cache, "_neuroclaw_trial_cache_audited", False):
        return
    audit_path = out_dir / "open_coscientist_cache_events.jsonl"
    audit_lock = threading.Lock()
    original_get = cache.get
    original_set = cache.set

    def request_key(
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        tools: Any = None,
        json_schema: Any = None,
        force_json: Any = None,
    ) -> str:
        return cache._generate_cache_key(
            prompt,
            model_name,
            temperature,
            max_tokens,
            tools,
            json_schema,
            force_json,
        )

    def append_event(event: dict[str, Any]) -> None:
        payload = {
            "timestamp": time.time(),
            "identity_sha256": identity_sha256,
            **event,
        }
        with audit_lock, audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def audited_get(
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        tools: Any = None,
        json_schema: Any = None,
        force_json: Any = None,
    ) -> Any:
        key = request_key(
            prompt,
            model_name,
            temperature,
            max_tokens,
            tools,
            json_schema,
            force_json,
        )
        result = original_get(
            prompt,
            model_name,
            temperature,
            max_tokens,
            tools,
            json_schema,
            force_json,
        )
        append_event(
            {
                "event": "lookup",
                "request_sha256": key,
                "hit": result is not None,
            }
        )
        return result

    def audited_set(
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        response: Any,
        tools: Any = None,
        json_schema: Any = None,
        force_json: Any = None,
    ) -> None:
        key = request_key(
            prompt,
            model_name,
            temperature,
            max_tokens,
            tools,
            json_schema,
            force_json,
        )
        original_set(
            prompt,
            model_name,
            temperature,
            max_tokens,
            response,
            tools,
            json_schema,
            force_json,
        )
        append_event(
            {
                "event": "store",
                "request_sha256": key,
                "stored": (Path(cache.cache_dir) / f"{key}.json").is_file(),
            }
        )

    cache.get = audited_get
    cache.set = audited_set
    cache._neuroclaw_trial_cache_audited = True


def _open_coscientist_research_goal(
    task: dict[str, Any],
    *,
    target: int,
    native_delivery: str,
) -> str:
    """Translate the shared batch contract into Open Co-Scientist semantics."""

    shared_goal = str(task["research_goal"])
    portfolio_marker = "\n\nReturn the requested framework-native artifact"
    shared_goal = shared_goal.split(portfolio_marker, 1)[0].rstrip()
    return f"""{shared_goal}

Open Co-Scientist batching contract: the request for {target} experiments
applies to the final collection across framework-native Hypothesis objects.
Each individual Hypothesis object and each generation response must propose
exactly one experiment, not a ranked portfolio. The complete workflow must
generate and rank at least {target} distinct Hypothesis objects before the
adapter reads the final ranking. Duplicate or missing candidates remain failed
experiment slots and are not recovered from planning, review, or debate prose.

Native-delivery requirement: {native_delivery}"""


def _install_openai_request_guards() -> None:
    """Disable provider storage and bound requests inside upstream frameworks."""

    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except (ImportError, AttributeError):
        return

    timeout = _request_timeout_seconds()
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "high")

    def guarded_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.setdefault("store", False)
        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("reasoning_effort", reasoning_effort)
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("thinking", {"type": "enabled"})
        kwargs["extra_body"] = extra_body
        return kwargs

    for resource_class in (Completions, AsyncCompletions):
        original = resource_class.create
        if getattr(original, "_case_study_request_guard", False):
            continue
        is_async = inspect.iscoroutinefunction(original)
        if is_async:

            @functools.wraps(original)
            async def guarded_async(self: Any, *args: Any, __original=original, **kwargs: Any) -> Any:
                kwargs = guarded_kwargs(kwargs)
                return await __original(self, *args, **kwargs)

            guarded_async._case_study_request_guard = True  # type: ignore[attr-defined]
            resource_class.create = guarded_async
        else:

            @functools.wraps(original)
            def guarded_sync(self: Any, *args: Any, __original=original, **kwargs: Any) -> Any:
                kwargs = guarded_kwargs(kwargs)
                return __original(self, *args, **kwargs)

            guarded_sync._case_study_request_guard = True  # type: ignore[attr-defined]
            resource_class.create = guarded_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--native-artifact", type=Path, default=None)
    parser.add_argument("--enable-native-retrieval", action="store_true")
    return parser.parse_args()


CANDIDATE_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_/-])"
    r"("
    r"[A-Za-z0-9_./-]+(?:\|[A-Za-z0-9_./-]+)+"
    r"|[A-Za-z0-9_.-]+:[A-Fa-f0-9]{12,64}"
    r")"
    r"(?![A-Za-z0-9_/-])"
)

PUBLIC_COORDINATE_COMPILER_SCHEMA = "public_coordinate_aliases_v1"


def _task_value(task: dict[str, Any], key: str, default: str) -> str:
    value = str(task.get(key) or "").strip()
    return value or default


def _case_study_id(task: dict[str, Any]) -> str:
    return _task_value(task, "case_study_id", "cs1")


def _candidate_id_template(task: dict[str, Any]) -> str:
    return _task_value(
        task,
        "candidate_id_template",
        "modality|source|disease|feature|roi_index",
    )


def _candidate_mentions(value: str) -> list[str]:
    text = str(value).replace(r"\|", "|")
    return [match.group(1) for match in CANDIDATE_ID_PATTERN.finditer(text)]


def _registry_rows(registry_path: Path) -> list[dict[str, Any]]:
    suffix = registry_path.suffix.casefold()
    if suffix == ".csv":
        with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".json":
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            rows = payload.get("candidates") or payload.get("records") or []
            return [dict(row) for row in rows if isinstance(row, dict)]
        return []

    rows: list[dict[str, Any]] = []
    with registry_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _registry_lookup(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    registry_path = Path(str(task["public_registry_path"]))
    lookup: dict[str, dict[str, Any]] = {}
    for row in _registry_rows(registry_path):
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            lookup[candidate_id] = row
    if not lookup:
        raise ValueError(f"public registry contains no candidates: {registry_path}")
    return lookup


def _normalized_coordinate_phrase(value: Any) -> str:
    """Normalize public coordinate labels without semantic expansion."""

    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _public_coordinate_compiler(
    task: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any] | None:
    """Validate and materialize one explicitly enabled public alias compiler."""

    raw = task.get("native_coordinate_compiler")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("native_coordinate_compiler must be one object")
    if raw.get("enabled") is not True:
        return None
    if str(raw.get("schema_version") or "") != PUBLIC_COORDINATE_COMPILER_SCHEMA:
        raise ValueError("unsupported native coordinate compiler schema")

    applicable_methods = tuple(str(value) for value in raw.get("applicable_methods") or ())
    if not applicable_methods:
        raise ValueError("native coordinate compiler has no applicable methods")
    if method not in applicable_methods:
        return None

    candidate_fields = tuple(
        str(value)
        for value in task.get("candidate_id_fields") or ()
        if str(value)
    )
    match_fields = tuple(str(value) for value in raw.get("match_fields") or ())
    if not match_fields or len(set(match_fields)) != len(match_fields):
        raise ValueError("native coordinate compiler match_fields are empty or repeated")
    if any(field not in candidate_fields for field in match_fields):
        raise ValueError("native coordinate compiler uses a non-candidate field")

    rows = list(lookup.values())
    global_coordinate_values = raw.get("global_coordinate_values") or {}
    if not isinstance(global_coordinate_values, dict):
        raise ValueError("global_coordinate_values must be one object")
    canonical_values: dict[str, set[str]] = {}
    aliases: dict[str, dict[str, set[str]]] = {}
    for field in match_fields:
        current_values = {str(row.get(field) or "").strip() for row in rows}
        if "" in current_values:
            raise ValueError(f"public registry has an empty coordinate field: {field}")
        declared_values = global_coordinate_values.get(field)
        if declared_values is None:
            values = current_values
        else:
            if not isinstance(declared_values, list):
                raise ValueError(f"global coordinate values for {field} must be one list")
            values = {str(value).strip() for value in declared_values}
            if "" in values or not values:
                raise ValueError(f"global coordinate values for {field} are empty")
            if not current_values <= values:
                raise ValueError(f"public registry exceeds global coordinate values: {field}")
        canonical_values[field] = values
        aliases[field] = {
            value: {_normalized_coordinate_phrase(value)} for value in sorted(values)
        }

    registry_alias_fields = raw.get("registry_alias_fields") or {}
    if not isinstance(registry_alias_fields, dict):
        raise ValueError("registry_alias_fields must be one object")
    for field, alias_fields in registry_alias_fields.items():
        if field not in aliases or not isinstance(alias_fields, list):
            raise ValueError(f"invalid registry alias field declaration: {field}")
        for row in rows:
            canonical = str(row.get(field) or "").strip()
            for alias_field in alias_fields:
                alias = _normalized_coordinate_phrase(row.get(str(alias_field)) or "")
                if alias:
                    aliases[field][canonical].add(alias)

    explicit_aliases = raw.get("aliases") or {}
    if not isinstance(explicit_aliases, dict):
        raise ValueError("native coordinate aliases must be one object")
    for field, by_value in explicit_aliases.items():
        if field not in aliases or not isinstance(by_value, dict):
            raise ValueError(f"invalid explicit alias field: {field}")
        for canonical, values in by_value.items():
            canonical = str(canonical)
            if canonical not in canonical_values[field]:
                raise ValueError(f"alias names an unknown {field}: {canonical}")
            if not isinstance(values, list):
                raise ValueError(f"aliases for {field}={canonical} must be one list")
            for value in values:
                alias = _normalized_coordinate_phrase(value)
                if not alias:
                    raise ValueError(f"empty alias for {field}={canonical}")
                aliases[field][canonical].add(alias)

    for field, by_value in aliases.items():
        owners: dict[str, set[str]] = {}
        for canonical, values in by_value.items():
            for alias in values:
                if alias:
                    owners.setdefault(alias, set()).add(canonical)
        collisions = {
            alias: sorted(values)
            for alias, values in owners.items()
            if len(values) > 1
        }
        if collisions:
            raise ValueError(
                f"ambiguous public coordinate aliases for {field}: "
                + json.dumps(collisions, sort_keys=True)
            )

    canonical_raw = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": PUBLIC_COORDINATE_COMPILER_SCHEMA,
        "config_sha256": hashlib.sha256(canonical_raw.encode("utf-8")).hexdigest(),
        "match_fields": match_fields,
        "aliases": {
            field: {
                canonical: tuple(sorted(values))
                for canonical, values in by_value.items()
            }
            for field, by_value in aliases.items()
        },
    }


def _candidate_from_public_coordinates(
    value: str,
    *,
    compiler: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Resolve one atomic native statement using public aliases only."""

    normalized = _normalized_coordinate_phrase(value)
    record: dict[str, Any] = {
        "text_sha256": hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
        "status": "unmapped",
        "matched_coordinates": {},
        "matched_aliases": {},
    }
    if not normalized:
        record["status"] = "empty_text"
        return "", record
    padded = f" {normalized} "
    coordinates: dict[str, str] = {}
    for field in compiler["match_fields"]:
        hits: dict[str, tuple[tuple[int, int], str]] = {}
        for canonical, aliases in compiler["aliases"][field].items():
            found = [alias for alias in aliases if f" {alias} " in padded]
            if not found:
                continue
            best_alias = max(found, key=lambda alias: (len(alias.split()), len(alias)))
            hits[canonical] = ((len(best_alias.split()), len(best_alias)), best_alias)
        if not hits:
            record["status"] = f"missing_coordinate:{field}"
            return "", record
        best_score = max(score for score, _alias in hits.values())
        winners = sorted(
            canonical
            for canonical, (score, _alias) in hits.items()
            if score == best_score
        )
        if len(winners) != 1:
            record["status"] = f"ambiguous_coordinate:{field}"
            record["ambiguous_values"] = winners
            return "", record
        canonical = winners[0]
        coordinates[field] = canonical
        record["matched_coordinates"][field] = canonical
        record["matched_aliases"][field] = hits[canonical][1]

    candidates = [
        candidate_id
        for candidate_id, row in lookup.items()
        if all(
            str(row.get(field) or "").strip() == canonical
            for field, canonical in coordinates.items()
        )
    ]
    if len(candidates) != 1:
        record["status"] = (
            "no_registered_coordinate_conjunction"
            if not candidates
            else "ambiguous_registered_coordinate_conjunction"
        )
        record["candidate_count"] = len(candidates)
        return "", record
    record["status"] = "mapped"
    record["candidate_id"] = candidates[0]
    return candidates[0], record


def _open_coscientist_debate_cohorts(
    task: dict[str, Any], target: int
) -> dict[int, tuple[str, ...]]:
    """Partition an outcome-blind menu across parallel native debates."""

    if target < 1 or _factor_rule_mode(task):
        return {}
    if not bool(task.get("open_coscientist_debate_cohorts_enabled", True)):
        return {}
    registry_path = Path(str(task.get("public_registry_path") or ""))
    if not registry_path.is_file():
        return {}
    candidate_ids = [
        str(row.get("candidate_id") or "").strip()
        for row in _registry_rows(registry_path)
    ]
    candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id]
    if len(candidate_ids) < target:
        return {}
    return {
        debate_id: tuple(candidate_ids[debate_id::target])
        for debate_id in range(target)
    }


def _open_coscientist_evolution_guidance(value: Any) -> dict[str, Any]:
    """Add an immutable experiment-ID constraint without changing native content."""

    guidance = dict(value) if isinstance(value, dict) else {}
    workflow_plan = dict(guidance.get("workflow_plan") or {})
    evolution_phase = dict(workflow_plan.get("evolution_phase") or {})
    guard = (
        "Treat the exact candidate_id in the original hypothesis as an immutable "
        "experiment primary key. Refine only the rationale and design. Never change, "
        "remove, punctuate, or copy another hypothesis's candidate_id."
    )
    priorities = evolution_phase.get("refinement_priorities")
    if isinstance(priorities, list):
        priorities = [str(item) for item in priorities]
    elif priorities:
        priorities = [str(priorities)]
    else:
        priorities = []
    if guard not in priorities:
        priorities.append(guard)
    evolution_phase["refinement_priorities"] = priorities
    strategy = str(evolution_phase.get("iteration_strategy") or "").strip()
    evolution_phase["iteration_strategy"] = f"{strategy} {guard}".strip()
    workflow_plan["evolution_phase"] = evolution_phase
    guidance["workflow_plan"] = workflow_plan
    return guidance


def _explicit_candidate_id(
    value: dict[str, Any],
    known_ids: set[str],
    task: dict[str, Any],
) -> str:
    candidate_id = str(value.get("candidate_id") or "").strip()
    if candidate_id in known_ids:
        return candidate_id

    fields = tuple(
        str(field).strip()
        for field in (
            task.get("candidate_id_fields")
            or ("modality", "source", "disease", "feature", "roi_index")
        )
        if str(field).strip()
    )
    components = tuple(str(value.get(field) or "").strip() for field in fields)
    if all(components):
        candidate_id = "|".join(components)
        if candidate_id in known_ids:
            return candidate_id

    # Preserve the historical CS1 anatomy_id reconstruction contract.
    disease = str(value.get("disease") or "").strip()
    feature = str(value.get("feature") or "").strip()
    anatomy_id = str(value.get("anatomy_id") or "").strip()
    if disease and feature and anatomy_id.count("|") == 2:
        modality, source, roi_index = anatomy_id.split("|")
        candidate_id = "|".join((modality, source, disease, feature, roi_index))
        if candidate_id in known_ids:
            return candidate_id
    return ""


def _native_confidence(value: dict[str, Any]) -> float:
    for field in ("confidence", "score", "priority_score"):
        try:
            confidence = float(value.get(field))
        except (TypeError, ValueError):
            continue
        if confidence == confidence:
            return min(1.0, max(0.0, confidence))
    return 1.0


def _native_rationale(value: dict[str, Any], fallback: str = "") -> str:
    for field in (
        "rationale",
        "reasoning",
        "hypothesis",
        "Short Hypothesis",
        "description",
        "summary",
    ):
        text = str(value.get(field) or "").strip()
        if text:
            return text[:1000]
    return fallback.strip()[:1000]


def _factor_rule_mode(task: dict[str, Any]) -> bool:
    return str(task.get("policy_mode") or "") == "factor_rules"


def _json_values_from_text(value: str) -> list[Any]:
    """Extract complete JSON objects from a native text artifact."""

    text = str(value).strip()
    if not text or "{" not in text:
        return []
    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    decoded: list[Any] = []
    seen: set[str] = set()
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            for index, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                signature = json.dumps(payload, sort_keys=True, default=str)
                if signature not in seen:
                    seen.add(signature)
                    decoded.append(payload)
        else:
            signature = json.dumps(payload, sort_keys=True, default=str)
            if signature not in seen:
                seen.add(signature)
                decoded.append(payload)
    return decoded


def _factor_rule_count(value: Any) -> int:
    """Count native rule objects without interpreting or repairing their values."""

    if isinstance(value, dict):
        count = int(isinstance(value.get("when"), dict) and bool(value["when"]))
        return count + sum(_factor_rule_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_factor_rule_count(item) for item in value)
    return 0


def _sciagents_final_artifact(
    history: Any,
    *,
    factor_rule_mode: bool,
) -> str:
    """Select the last substantive native artifact before AG2 termination chatter."""

    entries = history if isinstance(history, (list, tuple)) else ()
    if factor_rule_mode:
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            decoded = _json_values_from_text(content)
            if sum(_factor_rule_count(value) for value in decoded) > 0:
                return content

    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content") or "").strip()
        speaker = str(entry.get("name") or entry.get("role") or "").casefold()
        if content and content != "TERMINATE" and (
            "critic" in speaker or content.endswith("TERMINATE")
        ):
            return content
    return ""


def compile_native_policy(
    *,
    method: str,
    task: dict[str, Any],
    native_result: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile only exact native candidate mentions; never reinterpret ideas."""

    lookup = _registry_lookup(task)
    known_ids = set(lookup)
    coordinate_compiler = _public_coordinate_compiler(task, lookup, method=method)
    mapping_mode = (
        "deterministic_exact_or_public_coordinate_aliases_v1"
        if coordinate_compiler is not None
        else "deterministic_exact_native_only"
    )
    rule_fields = tuple(
        str(field)
        for field in (task.get("policy_rule_fields") or ())
        if str(field)
    )
    allowed_rule_values = {
        field: {str(row.get(field) or "") for row in lookup.values()}
        for field in rule_fields
    }
    anchors: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_rules: set[tuple[tuple[str, str], ...]] = set()
    invalid_mentions: list[str] = []
    invalid_rules: list[dict[str, Any]] = []
    exact_mentions = 0
    coordinate_projection_records: list[dict[str, Any]] = []
    candidate_artifact: Any = native_result
    artifact_scope = "complete_native_result"
    if isinstance(native_result, dict) and method == "sciagents":
        candidate_artifact = native_result.get("final_artifact", "")
        artifact_scope = "sciagents_final_artifact"
        if _factor_rule_mode(task):
            decoded = _json_values_from_text(str(candidate_artifact))
            if sum(_factor_rule_count(value) for value in decoded) == 0:
                recovered = _sciagents_final_artifact(
                    native_result.get("chat_history"),
                    factor_rule_mode=True,
                )
                if recovered:
                    candidate_artifact = recovered
                    artifact_scope = "sciagents_last_rule_artifact_from_chat_history"
    elif isinstance(native_result, dict) and method == "virtual_lab":
        candidate_artifact = str(native_result.get("summary", "")).replace(r"\|", "|")
        artifact_scope = "virtual_lab_final_summary"
    elif (
        isinstance(native_result, dict)
        and method == "open_coscientist"
        and not _factor_rule_mode(task)
    ):
        ranked_hypotheses = [
            hypothesis.get("text", "")
            for hypothesis in native_result.get("hypotheses", [])
            if isinstance(hypothesis, dict)
        ]
        candidate_artifact = ranked_hypotheses
        artifact_scope = "open_coscientist_final_ranked_hypotheses_only"

    def add_candidate(
        candidate_id: str,
        *,
        score: float = 1.0,
        rationale: str = "",
    ) -> None:
        nonlocal exact_mentions
        if candidate_id == _candidate_id_template(task):
            return
        exact_mentions += 1
        if candidate_id not in known_ids:
            if candidate_id not in invalid_mentions:
                invalid_mentions.append(candidate_id)
            return
        if candidate_id in seen or len(anchors) >= int(task["n_anchors"]):
            return
        seen.add(candidate_id)
        anchors.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "rationale": rationale,
                "evidence_ids": [],
            }
        )

    def add_rule(value: dict[str, Any]) -> None:
        raw_when = value.get("when")
        if not isinstance(raw_when, dict) or not raw_when:
            return
        when = {str(key): str(raw).strip() for key, raw in raw_when.items()}
        try:
            weight = float(value.get("weight"))
        except (TypeError, ValueError):
            invalid_rules.append({"reason": "invalid_weight", "when": when})
            return
        errors = []
        for field, raw in when.items():
            if field not in rule_fields:
                errors.append(f"unknown_field:{field}")
            elif raw not in allowed_rule_values[field]:
                errors.append(f"unknown_value:{field}={raw}")
        if not math.isfinite(weight) or not -1.0 <= weight <= 1.0:
            errors.append("weight_out_of_range")
        signature = tuple(sorted(when.items()))
        if signature in seen_rules:
            errors.append("duplicate")
        if errors:
            invalid_rules.append({"reason": "|".join(errors), "when": when})
            return
        if len(rules) >= int(task["n_anchors"]):
            return
        seen_rules.add(signature)
        rules.append(
            {
                "weight": weight,
                "when": when,
                "rationale": _native_rationale(value),
            }
        )

    def visit(value: Any, *, project_coordinates: bool = True) -> None:
        if isinstance(value, dict):
            if _factor_rule_mode(task):
                add_rule(value)
            explicit_id = _explicit_candidate_id(value, known_ids, task)
            if explicit_id:
                add_candidate(
                    explicit_id,
                    score=_native_confidence(value),
                    rationale=_native_rationale(value),
                )
            candidate_fields = tuple(
                str(field)
                for field in task.get("candidate_id_fields") or ()
                if str(field)
            )
            has_explicit_candidate_shape = bool(
                str(value.get("candidate_id") or "").strip()
            ) or (
                bool(candidate_fields)
                and all(str(value.get(field) or "").strip() for field in candidate_fields)
            )
            for nested in value.values():
                visit(
                    nested,
                    project_coordinates=project_coordinates and not has_explicit_candidate_shape,
                )
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, project_coordinates=project_coordinates)
            return
        if not isinstance(value, str):
            return
        if _factor_rule_mode(task):
            for decoded in _json_values_from_text(value):
                visit(decoded, project_coordinates=project_coordinates)
        mentions = _candidate_mentions(value)
        for candidate_id in mentions:
            add_candidate(
                candidate_id,
                rationale=value,
            )
        if coordinate_compiler is not None and project_coordinates and not mentions:
            candidate_id, projection = _candidate_from_public_coordinates(
                value,
                compiler=coordinate_compiler,
                lookup=lookup,
            )
            if len(coordinate_projection_records) < 100:
                coordinate_projection_records.append(projection)
            if candidate_id:
                add_candidate(candidate_id, rationale=value)

    visit(candidate_artifact)
    audit = {
        "method": method,
        "trial": int(task["trial"]),
        "requested_anchors": int(task["n_anchors"]),
        "valid_unique_anchors": len(anchors),
        "valid_unique_rules": len(rules),
        "exact_mentions_seen": exact_mentions,
        "invalid_exact_mentions": invalid_mentions[:100],
        "invalid_rules": invalid_rules[:100],
        "mapping_mode": mapping_mode,
        "artifact_scope": artifact_scope,
        "coordinate_compiler_schema": (
            coordinate_compiler["schema_version"] if coordinate_compiler else None
        ),
        "coordinate_compiler_config_sha256": (
            coordinate_compiler["config_sha256"] if coordinate_compiler else None
        ),
        "coordinate_projection_attempts": len(coordinate_projection_records),
        "coordinate_projection_mapped": sum(
            row.get("status") == "mapped" for row in coordinate_projection_records
        ),
        "coordinate_projection_records": coordinate_projection_records,
    }
    policy_payload = {
        "schema_version": _task_value(
            task, "policy_schema_version", "case1-search-policy-v1"
        ),
        "method": method,
        "trial": int(task["trial"]),
        "anchors": anchors,
        "rules": rules,
        "quotas": {},
        "metadata": {
            "mapping_mode": mapping_mode,
            "native_artifact_only": True,
            "artifact_scope": artifact_scope,
            "coordinate_compiler_schema": (
                coordinate_compiler["schema_version"] if coordinate_compiler else None
            ),
            "coordinate_compiler_config_sha256": (
                coordinate_compiler["config_sha256"] if coordinate_compiler else None
            ),
        },
    }
    return policy_payload, audit


def native_workflow_audit(method: str, native_result: Any) -> dict[str, Any]:
    """Summarize evidence that the method's upstream workflow actually ran."""

    audit: dict[str, Any] = {
        "method": method,
        "shared_heuristic_expansion": False,
    }
    if method == "ai_scientist_v2":
        audit.update(
            {
                "native_component": "perform_ideation_temp_free.generate_temp_free_idea",
                "native_idea_artifacts": (
                    len(native_result) if isinstance(native_result, list) else 0
                ),
            }
        )
    elif method == "open_coscientist":
        payload = native_result if isinstance(native_result, dict) else {}
        hypotheses = [
            value
            for value in payload.get("hypotheses", [])
            if isinstance(value, dict)
        ]
        audit.update(
            {
                "native_component": "HypothesisGenerator.generate_hypotheses",
                "native_hypotheses": len(hypotheses),
                "hypotheses_with_literature_grounding": sum(
                    bool(str(value.get("literature_grounding") or "").strip())
                    for value in hypotheses
                ),
                "hypotheses_with_citation_map": sum(
                    bool(value.get("citation_map")) for value in hypotheses
                ),
                "citation_records": sum(
                    len(value.get("citation_map") or {}) for value in hypotheses
                ),
                "native_review_present": bool(payload.get("meta_review")),
                "native_tournament_present": bool(payload.get("tournament_matchups")),
                "native_evolution_present": bool(payload.get("evolution_details")),
            }
        )
    elif method == "sciagents":
        payload = native_result if isinstance(native_result, dict) else {}
        history = payload.get("chat_history", [])
        tool_calls: dict[str, int] = {}
        for message in history if isinstance(history, list) else ():
            if not isinstance(message, dict):
                continue
            calls = list(message.get("tool_calls") or [])
            if message.get("function_call"):
                calls.append({"function": message["function_call"]})
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = (
                    str(function.get("name") or "")
                    if isinstance(function, dict)
                    else ""
                )
                if name:
                    tool_calls[name] = tool_calls.get(name, 0) + 1
        audit.update(
            {
                "native_component": "complete_upstream_12_role_groupchat",
                "participating_roles": list(payload.get("participating_roles", [])),
                "all_upstream_roles_participated": bool(
                    payload.get("all_upstream_roles_participated")
                ),
                "tool_calls_by_name": tool_calls,
            }
        )
    elif method == "virtual_lab":
        payload = native_result if isinstance(native_result, dict) else {}
        audit.update(
            {
                "native_component": "virtual_lab.run_meeting",
                "meeting_summary_present": bool(
                    str(payload.get("summary") or "").strip()
                ),
                "pubmed_article_cache_entries": int(
                    payload.get("article_cache_entries") or 0
                ),
                "retrieval_failures_isolated": int(
                    payload.get("retrieval_failures_isolated") or 0
                ),
            }
        )
    else:
        audit["native_component"] = "external_native_artifact"
    return audit


def run_ai_scientist(task: dict[str, Any], args: argparse.Namespace) -> Any:
    sys.path.insert(0, str(args.repo.resolve()))
    import ai_scientist.llm as ai_scientist_llm
    from ai_scientist.llm import create_client
    import ai_scientist.perform_ideation_temp_free as ideation

    # High-effort reasoning can consume AI-Scientist-v2's historical 4096-token
    # ceiling before a final answer is emitted.  Match the registered experiment
    # transport budget while leaving the upstream workflow otherwise unchanged.
    ai_scientist_llm.MAX_NUM_TOKENS = max(
        int(getattr(ai_scientist_llm, "MAX_NUM_TOKENS", 4096)),
        8192,
    )

    if not args.enable_native_retrieval:
        ideation.tools_dict = {}
        ideation.tool_names_str = '"FinalizeIdea"'
        ideation.system_prompt += (
            "\n\nFor this benchmark run literature search is disabled. Do not select "
            "SearchSemanticScholar; proceed directly to FinalizeIdea."
        )
    else:
        literature_tool = getattr(ideation, "tools_dict", {}).get(
            "SearchSemanticScholar"
        )
        if literature_tool is not None:
            original_search = literature_tool.search_for_papers

            def resilient_search(query: str) -> list[dict[str, Any]] | None:
                try:
                    papers = original_search(query)
                except Exception as exc:
                    _RETRIEVAL_AUDIT.append(
                        {
                            "source": "semantic_scholar",
                            "status": "failed",
                            "error_type": type(exc).__name__,
                        }
                    )
                    papers = None
                if papers:
                    _RETRIEVAL_AUDIT.append(
                        {
                            "source": "semantic_scholar",
                            "status": "complete",
                            "papers": len(papers),
                        }
                    )
                    return papers
                _RETRIEVAL_AUDIT.append(
                    {
                        "source": "semantic_scholar",
                        "status": "empty_or_unavailable",
                        "papers": 0,
                    }
                )
                return _pubmed_fallback_papers(query, literature_tool.max_results)

            literature_tool.search_for_papers = resilient_search

    ideas_path = args.out / "ai_scientist_native_ideas.json"
    if args.model == "deepseek-v4-pro":
        # AI-Scientist-v2 dispatches OpenAI-compatible calls only when the model
        # name contains ``gpt``.  The local gateway always replaces the incoming
        # model with deepseek-v4-pro, so use an explicit dispatcher alias here
        # without changing the declared or provider-side model.
        client, _ = create_client("gpt-4o")
        client_model = "gpt-deepseek-v4-pro"
        (args.out / "ai_scientist_model_bridge.json").write_text(
            json.dumps(
                {
                    "schema_version": "neuroclaw.ai-scientist-model-bridge.v1",
                    "declared_model": args.model,
                    "upstream_dispatcher_alias": client_model,
                    "provider_model": "deepseek-v4-pro",
                    "transport": "local_deepseek_v4_pro_gateway",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        client, client_model = create_client(args.model)
    target = max(1, int(task["n_anchors"]))
    portfolio_size = min(8, target)
    target_ideas = (target + portfolio_size - 1) // portfolio_size
    if _factor_rule_mode(task):
        delivery = (
            f"Native-delivery requirement: each final idea is an ordered portfolio "
            f"of {portfolio_size} distinct search rules. Preserve AI Scientist-v2's "
            "native `ACTION: FinalizeIdea` and `ARGUMENTS: {\"idea\": ...}` envelope. "
            "Place the exact rules JSON object requested in the research goal inside "
            "the idea's `Experiments` field. Preserve every factor value verbatim; "
            "prose-only rules receive zero credit."
        )
    else:
        delivery = (
            f"Native-delivery requirement: treat each final idea as an ordered portfolio "
            f"of {portfolio_size} distinct executable hypotheses. Include one line per "
            "hypothesis in the exact form "
            f"`candidate_id: <{_candidate_id_template(task)}>`, followed by "
            "its concise scientific rationale. Rank the lines from strongest to weakest "
            "and do not repeat IDs used in earlier portfolios. Only verbatim IDs will "
            "be accepted; prose-only ideas receive zero credit. The upstream host parser "
            "rejects a bare final JSON object. Even if the embedded research goal says "
            "to end with JSON, wrap that JSON inside the official envelope exactly as:\n"
            "ACTION: FinalizeIdea\n"
            "ARGUMENTS: {\"idea\":{\"Name\":\"registered_portfolio\","
            "\"hypotheses\":[{\"rank\":1,\"candidate_id\":\"...\","
            "\"rationale\":\"...\",\"confidence\":0.75}]}}\n"
            "Print nothing after ARGUMENTS."
        )
    retrieval_requirement = ""
    if args.enable_native_retrieval:
        retrieval_requirement = (
            "\n\nBefore FinalizeIdea, call the official SearchSemanticScholar tool at "
            "least once with a focused neuroscience query. Use the returned records "
            "only for plausibility and novelty reasoning; never infer benchmark outcomes."
        )
    workshop_description = (
        f"{task['research_goal']}\n\n{delivery}{retrieval_requirement}"
    )
    ideas: list[dict[str, Any]] = []
    if ideas_path.exists():
        existing = json.loads(ideas_path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            ideas = existing

    def compiled_anchor_count() -> int:
        try:
            policy, _audit = compile_native_policy(
                method="ai_scientist_v2",
                task=task,
                native_result=ideas,
            )
        except ValueError:
            return 0
        return len(policy["anchors"])

    no_progress_attempts = 0
    max_no_progress_attempts = 4
    while compiled_anchor_count() < target and len(ideas) < target_ideas:
        existing_count = len(ideas)
        ideas = ideation.generate_temp_free_idea(
            idea_fname=str(ideas_path),
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            max_num_generations=1,
            num_reflections=2 if target < 8 else 3,
            reload_ideas=True,
        )
        if len(ideas) <= existing_count:
            no_progress_attempts += 1
            if no_progress_attempts >= max_no_progress_attempts:
                raise RuntimeError(
                    "AI Scientist-v2 made no checkpoint progress after "
                    f"{max_no_progress_attempts} native retries while generating "
                    f"{target_ideas} hypothesis portfolios"
                )
            continue
        no_progress_attempts = 0
    return ideas[:target_ideas]


async def _run_open_coscientist_async(
    task: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    sys.path.insert(0, str((args.repo / "src").resolve()))
    os.environ.setdefault("OPEN_COSCIENTIST_MAX_CONCURRENT_LLM_CALLS", "4")
    os.environ.setdefault("OPEN_COSCIENTIST_RATE_LIMIT_RETRIES", "8")
    from open_coscientist import HypothesisGenerator

    try:
        debate_module = importlib.import_module(
            "open_coscientist.nodes.generation.debate"
        )
    except ImportError:
        debate_module = None
    if debate_module is not None:
        original_single_debate = debate_module._run_single_debate
        debate_turns = _open_coscientist_debate_turns()
        debate_cohorts = _open_coscientist_debate_cohorts(
            task, max(1, int(task["n_anchors"]))
        )

        @functools.wraps(original_single_debate)
        async def bounded_single_debate(
            state: Any,
            debate_id: int | None = None,
            num_turns: int = debate_turns,
            articles_with_reasoning: str | None = None,
            reference_index: Any = None,
        ) -> Any:
            local_state = state
            cohort = debate_cohorts.get(int(debate_id)) if debate_id is not None else None
            if cohort:
                local_state = dict(state)
                cohort_lines = "\n".join(f"- {candidate_id}" for candidate_id in cohort)
                local_state["research_goal"] = (
                    f"{state['research_goal']}\n\n"
                    "PARALLEL DEBATE DIVERSITY ALLOCATION\n"
                    f"This is debate slot {int(debate_id) + 1} of {len(debate_cohorts)}. "
                    "To prevent independent debates from converging on the same proposal, "
                    "you must choose exactly one candidate_id from the outcome-blind "
                    "assigned cohort below. Do not select an ID outside this cohort. "
                    "The later native review and tournament stages will compare all "
                    "cohort winners globally.\n"
                    f"{cohort_lines}"
                )
            return await original_single_debate(
                local_state,
                debate_id=debate_id,
                num_turns=min(num_turns, debate_turns),
                articles_with_reasoning=articles_with_reasoning,
                reference_index=reference_index,
            )

        debate_module._run_single_debate = bounded_single_debate

    try:
        evolve_module = importlib.import_module("open_coscientist.nodes.evolve")
    except ImportError:
        evolve_module = None
    if evolve_module is not None:
        original_evolve_single = evolve_module.evolve_single_hypothesis

        @functools.wraps(original_evolve_single)
        async def identity_guarded_evolve_single(
            hypothesis: Any,
            other_hypotheses_texts: list[str],
            meta_review: dict[str, Any],
            model_name: str,
            removed_duplicates: list[str],
            supervisor_guidance: dict[str, Any] | None = None,
            articles_with_reasoning: str | None = None,
            run_id: str | None = None,
            hypothesis_index: int | None = None,
            tool_registry: Any | None = None,
        ) -> Any:
            return await original_evolve_single(
                hypothesis=hypothesis,
                other_hypotheses_texts=other_hypotheses_texts,
                meta_review=meta_review,
                model_name=model_name,
                removed_duplicates=removed_duplicates,
                supervisor_guidance=_open_coscientist_evolution_guidance(
                    supervisor_guidance
                ),
                articles_with_reasoning=articles_with_reasoning,
                run_id=run_id,
                hypothesis_index=hypothesis_index,
                tool_registry=tool_registry,
            )

        evolve_module.evolve_single_hypothesis = identity_guarded_evolve_single

    try:
        open_coscientist_llm = importlib.import_module("open_coscientist.llm")
    except ImportError:
        open_coscientist_llm = None
    if open_coscientist_llm is not None:
        original_acompletion = open_coscientist_llm.litellm.acompletion
        call_counter = 0
        call_audit_path = args.out / "open_coscientist_llm_calls.jsonl"

        async def guarded_acompletion(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal call_counter
            call_counter += 1
            call_index = call_counter
            call_kwargs.setdefault("store", False)
            call_kwargs.setdefault("timeout", _request_timeout_seconds())
            call_kwargs.setdefault("reasoning_effort", args.reasoning_effort)
            extra_body = dict(call_kwargs.get("extra_body") or {})
            extra_body.setdefault("thinking", {"type": "enabled"})
            call_kwargs["extra_body"] = extra_body
            response_format = call_kwargs.get("response_format")
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and isinstance(response_format.get("json_schema"), dict)
            ):
                json_schema = dict(response_format["json_schema"])
                json_schema["schema"] = _openai_strict_json_schema(
                    json_schema.get("schema", {})
                )
                json_schema["strict"] = True
                call_kwargs["response_format"] = {
                    **response_format,
                    "json_schema": json_schema,
                }
            call_kwargs["max_tokens"] = max(
                int(call_kwargs.get("max_tokens") or 0),
                _open_coscientist_max_tokens(),
            )
            started = time.time()
            audit = {
                "call": call_index,
                "started_at": started,
                "event": "started",
                "model": str(call_kwargs.get("model") or ""),
                "max_tokens": call_kwargs.get("max_tokens"),
                "response_format": str(
                    (call_kwargs.get("response_format") or {}).get("type") or ""
                ),
                "store": bool(call_kwargs.get("store")),
                "timeout_seconds": call_kwargs.get("timeout"),
                "thinking_enabled": bool(
                    (call_kwargs.get("extra_body") or {}).get("thinking")
                ),
            }
            with call_audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            try:
                response = await original_acompletion(*call_args, **call_kwargs)
            except Exception as exc:
                audit.update(
                    {
                        "status": "failed",
                        "event": "finished",
                        "duration_seconds": time.time() - started,
                        "error_type": type(exc).__name__,
                    }
                )
                with call_audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                raise
            audit.update(
                {
                    "status": "complete",
                    "event": "finished",
                    "duration_seconds": time.time() - started,
                }
            )
            with call_audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            return response

        open_coscientist_llm.litellm.acompletion = guarded_acompletion

    model_name = args.model if "/" in args.model else f"openai/{args.model}"
    target = max(1, int(task["n_anchors"]))
    if _factor_rule_mode(task):
        native_delivery = (
            "Treat each framework-native Hypothesis object as exactly one proposed "
            "factor rule. Preserve every registered value verbatim, keep the rule "
            "machine-readable through review, ranking, and evolution, and make all "
            "generated hypotheses distinct. Do not place the complete benchmark "
            "portfolio inside each individual hypothesis."
        )
    else:
        native_delivery = (
            "Treat each framework-native Hypothesis object as exactly one executable "
            "experiment. Its hypothesis text must contain exactly one line in the "
            f"form `candidate_id: <{_candidate_id_template(task)}>`, followed by a "
            "concise rationale. Keep that exact ID unchanged through review, ranking, "
            "deduplication, and evolution, and make all generated hypotheses distinct. "
            "Do not place the complete benchmark JSON portfolio inside each individual "
            "hypothesis. Only verbatim registered IDs are accepted."
        )
    research_goal = _open_coscientist_research_goal(
        task,
        target=target,
        native_delivery=native_delivery,
    )
    native_pool_size = _open_coscientist_pool_size(task, target)
    cache_dir, cache_manifest_path, cache_identity_sha256 = (
        _prepare_open_coscientist_trial_cache(
            task,
            args,
            model_name=model_name,
            native_pool_size=native_pool_size,
        )
    )
    open_coscientist_cache = importlib.import_module("open_coscientist.cache")
    os.environ["COSCIENTIST_CACHE_DIR"] = str(cache_dir)

    # Open Co-Scientist controls LLM and whole-node caches with one environment
    # flag.  Initialize the node cache while disabled, then enable only the
    # exact-request LLM cache.  Replaying a whole node would bypass native
    # retrieval/planning and is therefore forbidden for formal trials.
    os.environ["COSCIENTIST_CACHE_ENABLED"] = "false"
    node_cache = open_coscientist_cache.get_node_cache()
    if bool(getattr(node_cache, "enabled", True)):
        raise RuntimeError("Open Co-Scientist node cache could not be disabled")
    os.environ["COSCIENTIST_CACHE_ENABLED"] = "true"
    llm_cache = open_coscientist_cache.get_cache()
    if not bool(getattr(llm_cache, "enabled", False)):
        raise RuntimeError("Open Co-Scientist exact-request LLM cache is disabled")
    if Path(llm_cache.cache_dir) != cache_dir:
        raise RuntimeError(
            "Open Co-Scientist LLM cache was initialized outside the sealed trial directory"
        )
    _install_open_coscientist_cache_audit(
        llm_cache,
        out_dir=args.out,
        identity_sha256=cache_identity_sha256,
    )
    generator = HypothesisGenerator(
        model_name=model_name,
        max_iterations=max(
            0,
            int(task.get("open_coscientist_max_iterations", 1)),
        ),
        initial_hypotheses_count=native_pool_size,
        evolution_max_count=native_pool_size,
        enable_cache=True,
        cache_dir=str(cache_dir),
    )
    try:
        native_result = await generator.generate_hypotheses(
            research_goal=research_goal,
            opts={
                "enable_literature_review_node": bool(args.enable_native_retrieval),
                "enable_tool_calling_generation": bool(
                    task.get("open_coscientist_tool_generation_enabled", False)
                ),
                "constraints": [
                    _task_value(
                        task,
                        "coordinate_constraint",
                        "Use only registered disease, feature, and anatomy coordinates.",
                    ),
                    "Do not inspect experimental outcomes.",
                    (
                        f"The final ranked set must retain at least {target} distinct "
                        "framework-native hypotheses, each containing one exact executable "
                        "candidate_id verbatim; prose-only proposals are invalid."
                    ),
                ],
            },
            stream=False,
            run_id=(
                f"{_case_study_id(task)}-{task['trial']}"
                f"-round-{int(task.get('round_index', 0))}"
            ),
        )
    except BaseException:
        try:
            _update_open_coscientist_cache_manifest(
                cache_manifest_path,
                cache_dir,
                status="interrupted_or_failed",
            )
        except Exception:
            pass
        raise
    _update_open_coscientist_cache_manifest(
        cache_manifest_path,
        cache_dir,
        status="completed",
    )
    return native_result


def run_open_coscientist(task: dict[str, Any], args: argparse.Namespace) -> Any:
    return asyncio.run(_run_open_coscientist_async(task, args))


def run_virtual_lab(task: dict[str, Any], args: argparse.Namespace) -> Any:
    sys.path.insert(0, str((args.repo / "src").resolve()))
    from virtual_lab import Agent

    meeting_module = importlib.import_module("virtual_lab.run_meeting")
    utils_module = importlib.import_module("virtual_lab.utils")
    retrieval_errors: list[dict[str, str]] = []
    article_cache: dict[tuple[str, bool], tuple[Any, Any]] = {}
    original_article_fetch = utils_module.get_pubmed_central_article

    def resilient_article_fetch(
        pmcid: str,
        abstract_only: bool = False,
    ) -> tuple[Any, Any]:
        key = (str(pmcid), bool(abstract_only))
        if key in article_cache:
            return article_cache[key]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                value = original_article_fetch(
                    pmcid=pmcid,
                    abstract_only=abstract_only,
                )
                article_cache[key] = value
                return value
            except Exception as exc:  # Upstream retrieval must not abort a meeting.
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        retrieval_errors.append(
            {
                "stage": "article_fetch",
                "pmcid": str(pmcid),
                "error": f"{type(last_error).__name__}: {last_error}",
            }
        )
        article_cache[key] = (None, None)
        return article_cache[key]

    utils_module.get_pubmed_central_article = resilient_article_fetch
    original_run_tools = meeting_module.run_tools

    def resilient_run_tools(tool_calls: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
        try:
            return original_run_tools(tool_calls)
        except Exception as exc:
            retrieval_errors.append(
                {
                    "stage": "pubmed_tool",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            output = (
                "PubMed retrieval was temporarily unavailable. Continue the blinded "
                "scientific discussion using the registered candidate metadata; do not "
                "infer missing evidence."
            )
            return (
                [output for _ in tool_calls],
                [
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                    for tool_call in tool_calls
                ],
            )

    meeting_module.run_tools = resilient_run_tools

    shared_goal = _task_value(
        task,
        "virtual_lab_goal",
        "rank blinded, executable hypotheses while respecting the exact public registry",
    )
    lead = Agent(
        title="principal investigator",
        expertise=_task_value(
            task, "lead_expertise", "neuroimaging and research prioritization"
        ),
        goal=shared_goal,
        role="synthesize and deliver the final ranked proposals",
        model=args.model,
    )
    members = (
        Agent(
            title="neuroimaging scientist",
            expertise=_task_value(
                task,
                "imaging_expertise",
                "functional and structural neuroimaging measurements",
            ),
            goal=shared_goal,
            role="evaluate anatomical and imaging-feature plausibility",
            model=args.model,
        ),
        Agent(
            title="psychiatric neuroscientist",
            expertise=_task_value(
                task, "biological_expertise", "clinical and molecular neuroscience"
            ),
            goal=shared_goal,
            role="evaluate disease-general and disease-specific mechanisms",
            model=args.model,
        ),
        Agent(
            title="statistician",
            expertise="high-dimensional blinded experiment prioritization",
            goal=shared_goal,
            role="criticize redundancy and improve search diversity",
            model=args.model,
        ),
    )
    summary = meeting_module.run_meeting(
        meeting_type="team",
        agenda=task["research_goal"],
        save_dir=args.out,
        save_name="virtual_lab_native_discussion",
        team_lead=lead,
        team_members=members,
        agenda_questions=(
            (
                "Which exact registered factor rules should be prioritized? Return "
                "the final rules in the requested JSON shape."
                if _factor_rule_mode(task)
                else "Which exact registered candidate_id values should be prioritized? "
                "Print every selected candidate_id verbatim."
            ),
            "How should proposals be ranked while preserving diversity?",
        ),
        agenda_rules=(
            f"Never inspect {_case_study_id(task)} outcomes or hidden labels.",
            (
                "Use exact public factor values and emit machine-readable factor rules."
                if _factor_rule_mode(task)
                else "Use exact public registry coordinates and include one verbatim "
                "candidate_id in every final proposal."
            ),
        ),
        summaries=tuple(
            str(value)
            for value in task.get("previous_round_summaries", ())
            if str(value).strip()
        ),
        contexts=tuple(
            str(value)
            for value in task.get("round_contexts", ())
            if str(value).strip()
        ),
        num_rounds=1 if int(task["n_anchors"]) < 8 else 2,
        pubmed_search=bool(args.enable_native_retrieval),
        return_summary=True,
    )
    return {
        "summary": summary,
        "native_retrieval_enabled": bool(args.enable_native_retrieval),
        "retrieval_failures_isolated": len(retrieval_errors),
        "retrieval_errors": retrieval_errors,
        "article_cache_entries": len(article_cache),
    }


def sciagents_path_context(
    task: dict[str, Any],
    *,
    max_paths: int = 48,
) -> str:
    """Sample blinded public-registry paths without consulting KG evidence."""

    registry_path = Path(str(task["public_registry_path"]))
    selected: list[tuple[str, dict[str, Any]]] = []
    trial = int(task["trial"])
    for row in _registry_rows(registry_path):
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        priority = hashlib.sha256(
            f"{trial}|{candidate_id}".encode("utf-8")
        ).hexdigest()
        selected.append((priority, row))
        if len(selected) > max_paths * 8:
            selected = sorted(selected, key=lambda item: item[0])[:max_paths]
    selected = sorted(selected, key=lambda item: item[0])[:max_paths]
    raw_fields = task.get("ontology_path_fields") or (
        "disease",
        "feature",
        "anatomy_full",
        "network",
    )
    fields = [str(field).strip() for field in raw_fields if str(field).strip()]
    labels = dict(task.get("ontology_path_labels") or {})
    paths = []
    for _, row in selected:
        components = [
            f"{labels.get(field, field)}:{row.get(field, '')}" for field in fields
        ]
        components.append(f"candidate_id:{row.get('candidate_id', '')}")
        paths.append(" -> ".join(components))
    return "\n".join(f"- {path}" for path in paths)


def sciagents_llm_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build an AG2 config that delegates retries to the finite formal router."""

    max_retries = max(0, int(os.environ.get("SCIAGENTS_LLM_MAX_RETRIES", "0")))
    return {
        "config_list": [
            {
                "model": args.model,
                "api_key": (
                    os.environ.get("CASE_STUDY_LOCAL_API_KEY")
                    or os.environ["CS1_LOCAL_API_KEY"]
                ),
                "base_url": args.base_url,
                "max_retries": max_retries,
                "store": False,
                "timeout": _request_timeout_seconds(),
            }
        ],
        "temperature": 0.2,
        "reasoning_effort": args.reasoning_effort,
    }


def _sciagents_official_role_specs(repo: Path) -> tuple[dict[str, dict[str, str]], Path]:
    """Load role prompts from upstream source without importing its fixed graph."""

    source_path = (repo / "ScienceDiscovery" / "agents.py").resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    specs: dict[str, dict[str, str]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        function_name = (
            call.func.id
            if isinstance(call.func, ast.Name)
            else call.func.attr
            if isinstance(call.func, ast.Attribute)
            else ""
        )
        if function_name != "AssistantAgent":
            continue
        values: dict[str, str] = {}
        for keyword in call.keywords:
            if keyword.arg not in {"name", "system_message", "description"}:
                continue
            try:
                literal = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                continue
            if isinstance(literal, str):
                values[keyword.arg] = literal
        name = values.get("name", "").strip()
        if name:
            specs[name] = values
    required = {
        "planner",
        "assistant",
        "ontologist",
        "scientist",
        "hypothesis_agent",
        "outcome_agent",
        "mechanism_agent",
        "design_principles_agent",
        "unexpected_properties_agent",
        "comparison_agent",
        "novelty_agent",
        "critic_agent",
    }
    missing = sorted(required - set(specs))
    if missing:
        raise RuntimeError(f"upstream SciAgents roles are missing: {missing}")
    return specs, source_path


def run_sciagents(task: dict[str, Any], args: argparse.Namespace) -> Any:
    """Run the complete upstream role topology on the registered CS search graph."""

    sys.path.insert(0, str(args.repo.resolve()))
    import autogen

    config = sciagents_llm_config(args)
    official_specs, official_source = _sciagents_official_role_specs(args.repo)
    role_order = (
        "planner",
        "assistant",
        "ontologist",
        "scientist",
        "hypothesis_agent",
        "outcome_agent",
        "mechanism_agent",
        "design_principles_agent",
        "unexpected_properties_agent",
        "comparison_agent",
        "novelty_agent",
        "critic_agent",
    )
    start_rank = int(task.get("round_index", 0)) * int(task["n_anchors"]) + 1
    end_rank = start_rank + int(task["n_anchors"]) - 1
    shared_domain_adapter = (
        "\n\nDOMAIN ADAPTER: Work only on the outcome-blind neuroimaging candidate "
        "registry supplied by the user. Replace materials-specific examples in the "
        "upstream prompt with disease, brain anatomy, atlas, and imaging-feature "
        "reasoning. Never infer an experimental result. Preserve candidate_id values "
        "verbatim and do not create coordinates outside the current menu."
    )
    roles: dict[str, Any] = {}
    for role_name in role_order:
        spec = official_specs[role_name]
        system_message = spec["system_message"] + shared_domain_adapter
        if role_name == "planner":
            system_message += (
                " Plan work for the official ontologist, scientist, seven specialist "
                "review roles, tool-calling assistant, and final critic."
            )
        elif role_name == "assistant":
            system_message += (
                " Before handing the workflow to the specialist reviewers, call both "
                "registered domain tools at least once: generate_path for blinded "
                "registry structure and rate_novelty_feasibility for literature evidence."
            )
        elif role_name == "critic_agent":
            system_message += (
                f" You are the final decision maker. Deduplicate and rank exactly "
                f"{task['n_anchors']} executable menu candidates at consecutive ranks "
                f"{start_rank}-{end_rank}. End with the exact JSON object requested in "
                "the research goal and then print TERMINATE on a new line."
            )
        roles[role_name] = autogen.AssistantAgent(
            name=role_name,
            llm_config=config,
            system_message=system_message,
            description=spec.get("description", ""),
            is_termination_msg=(
                (lambda message: str(message.get("content") or "").rstrip().endswith("TERMINATE"))
                if role_name == "critic_agent"
                else None
            ),
        )
    user = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=False,
        is_termination_msg=lambda message: str(message.get("content") or "")
        .rstrip()
        .endswith("TERMINATE"),
    )

    registry = _registry_rows(Path(str(task["public_registry_path"])))
    path_context = sciagents_path_context(task)

    def generate_path(
        keyword_1: Annotated[str, "optional first public-registry keyword"] = "",
        keyword_2: Annotated[str, "optional second public-registry keyword"] = "",
    ) -> str:
        keywords = [
            str(value).strip().casefold()
            for value in (keyword_1, keyword_2)
            if str(value).strip()
        ]
        matches: list[str] = []
        for row in registry:
            searchable = " ".join(str(value) for value in row.values()).casefold()
            if keywords and not all(keyword in searchable for keyword in keywords):
                continue
            matches.append(
                " -> ".join(
                    (
                        f"disease:{row.get('disease', '')}",
                        f"feature:{row.get('feature', '')}",
                        f"anatomy:{row.get('anatomy_full', '')}",
                        f"candidate_id:{row.get('candidate_id', '')}",
                    )
                )
            )
            if len(matches) >= 12:
                break
        return "\n".join(matches) if matches else "No registered path matched."

    def rate_novelty_feasibility(
        hypothesis: Annotated[str, "hypothesis text to compare with literature"],
    ) -> str:
        if not args.enable_native_retrieval:
            return "Native literature retrieval is disabled for this run."
        papers: list[dict[str, Any]] = []
        try:
            import requests

            headers = {}
            semantic_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
            if semantic_key:
                headers["x-api-key"] = semantic_key
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": str(hypothesis)[:350],
                    "limit": 5,
                    "fields": "title,abstract,url,year",
                },
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            papers = []
            for paper in (response.json().get("data") or [])[:5]:
                papers.append(
                    {
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "url": paper.get("url"),
                        "abstract": str(paper.get("abstract") or "")[:1000],
                    }
                )
            if papers:
                _RETRIEVAL_AUDIT.append(
                    {
                        "source": "semantic_scholar",
                        "status": "complete",
                        "papers": len(papers),
                    }
                )
        except Exception as exc:
            _RETRIEVAL_AUDIT.append(
                {
                    "source": "semantic_scholar",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
        if not papers:
            papers = _pubmed_fallback_papers(str(hypothesis), 5)
        return json.dumps(
            {
                "literature_evidence": papers,
                "instruction": (
                    "Use these records to assess novelty and feasibility; they do "
                    "not contain benchmark outcomes."
                ),
            },
            ensure_ascii=False,
        )

    for caller_name in ("planner", "assistant"):
        roles[caller_name].register_for_llm(
            name="generate_path",
            description=(
                "Retrieve blinded disease-feature-anatomy paths from the current "
                "registered candidate menu."
            ),
        )(generate_path)
        roles[caller_name].register_for_llm(
            name="rate_novelty_feasibility",
            description=(
                "Retrieve current literature evidence for assessing the novelty and "
                "feasibility of a proposed hypothesis."
            ),
        )(rate_novelty_feasibility)
    user.register_for_execution(name="generate_path")(generate_path)
    user.register_for_execution(name="rate_novelty_feasibility")(
        rate_novelty_feasibility
    )

    def complete_role_topology_selector(last_speaker: Any, groupchat: Any) -> Any:
        del last_speaker
        last_message = groupchat.messages[-1] if groupchat.messages else {}
        if last_message.get("function_call") or last_message.get("tool_calls"):
            return user
        spoken = {
            str(message.get("name") or "")
            for message in groupchat.messages
            if isinstance(message, dict)
        }
        for role_name in role_order:
            if role_name not in spoken:
                return roles[role_name]
        return "auto"

    max_rounds = max(16, min(50, int(task.get("sciagents_max_rounds", 50))))
    group = autogen.GroupChat(
        agents=[user, *(roles[name] for name in role_order)],
        messages=[],
        max_round=max_rounds,
        admin_name="user",
        send_introductions=True,
        allow_repeat_speaker=True,
        speaker_selection_method=complete_role_topology_selector,
    )
    manager = autogen.GroupChatManager(groupchat=group, llm_config=config)
    message = (
        f"{task['research_goal']}\n\n"
        "Sampled outcome-blind candidate-ontology paths for native graph reasoning:\n"
        f"{path_context}\n\n"
        "Use these as structural examples, not as evidence of experimental support. "
        "The tool-calling assistant must invoke both generate_path and "
        "rate_novelty_feasibility before the final critic ranks candidates."
    )
    result = user.initiate_chat(manager, message=message, max_turns=max_rounds)
    history = getattr(result, "chat_history", group.messages)
    final_artifact = _sciagents_final_artifact(
        history,
        factor_rule_mode=_factor_rule_mode(task),
    )
    if not final_artifact:
        final_artifact = str(getattr(result, "summary", "") or "").strip()
    participating_roles = sorted(
        {
            str(message.get("name") or "")
            for message in history
            if isinstance(message, dict)
            and str(message.get("name") or "") in set(role_order)
        }
    )
    return {
        "summary": getattr(result, "summary", ""),
        "final_artifact": final_artifact,
        "chat_history": history,
        "path_context_type": "public_registry_candidate_ontology",
        "sampled_path_count": len(path_context.splitlines()),
        "adapter_fidelity": "complete_upstream_role_topology_with_domain_tools",
        "upstream_role_source": str(official_source),
        "upstream_role_source_sha256": hashlib.sha256(
            official_source.read_bytes()
        ).hexdigest(),
        "upstream_roles": list(role_order),
        "participating_roles": participating_roles,
        "all_upstream_roles_participated": set(participating_roles) == set(role_order),
        "registered_tools": ["generate_path", "rate_novelty_feasibility"],
        "native_literature_retrieval_enabled": bool(args.enable_native_retrieval),
        "max_rounds": max_rounds,
    }


def load_supplementary_artifact(args: argparse.Namespace) -> Any:
    if args.native_artifact is None:
        raise RuntimeError(
            f"{args.method} is supplementary and requires --native-artifact from "
            "its official upstream workflow"
        )
    return json.loads(args.native_artifact.read_text(encoding="utf-8"))


def run_native(task: dict[str, Any], args: argparse.Namespace) -> Any:
    if args.native_artifact is not None:
        return json.loads(args.native_artifact.read_text(encoding="utf-8"))
    if args.method == "ai_scientist_v2":
        return run_ai_scientist(task, args)
    if args.method == "open_coscientist":
        return run_open_coscientist(task, args)
    if args.method == "sciagents":
        return run_sciagents(task, args)
    if args.method == "virtual_lab":
        return run_virtual_lab(task, args)
    return load_supplementary_artifact(args)


def main() -> None:
    args = parse_args()
    secret = os.environ.get("CASE_STUDY_LOCAL_API_KEY") or os.environ.get(
        "CS1_LOCAL_API_KEY"
    )
    if args.native_artifact is None and not secret:
        raise RuntimeError("CASE_STUDY_LOCAL_API_KEY is required")
    args.out.mkdir(parents=True, exist_ok=True)
    task = json.loads(args.task.read_text(encoding="utf-8"))

    if secret:
        os.environ["OPENAI_API_KEY"] = secret
        os.environ["OPENAI_BASE_URL"] = args.base_url
        os.environ["OPENAI_API_BASE"] = args.base_url
        os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
        os.environ["OPENAI_JSON_SCHEMA_STRICT"] = "true"
        os.environ["OPENAI_EMBED_JSON_SCHEMA_IN_PROMPT"] = "true"
        _install_openai_request_guards()
    started = time.time()
    native_result = run_native(task, args)
    workflow_audit = native_workflow_audit(args.method, native_result)
    native_path = args.out / "native_result.json"
    native_path.write_text(
        json.dumps(native_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    policy_payload, compile_audit = compile_native_policy(
        method=args.method,
        task=task,
        native_result=native_result,
    )
    policy_payload["schema_version"] = _task_value(
        task, "policy_schema_version", "case1-search-policy-v1"
    )
    policy_payload["method"] = args.method
    policy_payload["trial"] = int(task["trial"])
    metadata = dict(policy_payload.get("metadata") or {})
    mapping_mode = (
        "deterministic_exact_factor_rules"
        if _factor_rule_mode(task)
        else str(compile_audit["mapping_mode"])
    )
    metadata.update(
        {
            "official_repo": str(args.repo),
            "native_artifact": str(native_path),
            "mapping_mode": mapping_mode,
            "coordinate_compiler_schema": compile_audit.get("coordinate_compiler_schema"),
            "coordinate_compiler_config_sha256": compile_audit.get(
                "coordinate_compiler_config_sha256"
            ),
            "rule_matching": (
                "hierarchical_partial_plus_joint"
                if _factor_rule_mode(task)
                else "exact_conjunction"
            ),
            "native_retrieval_enabled": bool(args.enable_native_retrieval),
            "open_coscientist_debate_turns": (
                _open_coscientist_debate_turns()
                if args.method == "open_coscientist"
                else None
            ),
            "open_coscientist_max_iterations": (
                max(0, int(task.get("open_coscientist_max_iterations", 1)))
                if args.method == "open_coscientist"
                else None
            ),
            "open_coscientist_overgeneration_factor": (
                float(task.get("open_coscientist_overgeneration_factor", 1.0))
                if args.method == "open_coscientist"
                else None
            ),
            "open_coscientist_tool_generation_enabled": (
                bool(task.get("open_coscientist_tool_generation_enabled", False))
                if args.method == "open_coscientist"
                else None
            ),
            "open_coscientist_debate_cohorts_enabled": (
                bool(task.get("open_coscientist_debate_cohorts_enabled", True))
                if args.method == "open_coscientist"
                else None
            ),
            "open_coscientist_debate_cohort_mode": (
                "outcome_blind_disjoint_public_menu_partition"
                if args.method == "open_coscientist"
                and bool(task.get("open_coscientist_debate_cohorts_enabled", True))
                else None
            ),
            "retrieval_audit_records": len(_RETRIEVAL_AUDIT),
            "retrieval_sources_used": sorted(
                {
                    str(row.get("source") or "")
                    for row in _RETRIEVAL_AUDIT
                    if row.get("status") == "complete"
                }
                - {""}
            ),
            "native_workflow_audit": workflow_audit,
        }
    )
    policy_payload["metadata"] = metadata
    (args.out / "native_compile_audit.json").write_text(
        json.dumps(compile_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "native_workflow_audit.json").write_text(
        json.dumps(workflow_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "search_policy.json").write_text(
        json.dumps(policy_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "adapter_client_meta.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "trial": task["trial"],
                "model": args.model,
                "base_url": args.base_url,
                "reasoning_effort": args.reasoning_effort,
                "extra_body": {"thinking": {"type": "enabled"}},
                "duration_seconds": time.time() - started,
                "native_retrieval_enabled": bool(args.enable_native_retrieval),
                "mapping_mode": mapping_mode,
                "valid_unique_anchors": len(policy_payload["anchors"]),
                "valid_unique_rules": len(policy_payload["rules"]),
                "open_coscientist_debate_cohorts_enabled": (
                    bool(task.get("open_coscientist_debate_cohorts_enabled", True))
                    if args.method == "open_coscientist"
                    else None
                ),
                "retrieval_audit_records": len(_RETRIEVAL_AUDIT),
                "retrieval_sources_used": sorted(
                    {
                        str(row.get("source") or "")
                        for row in _RETRIEVAL_AUDIT
                        if row.get("status") == "complete"
                    }
                    - {""}
                ),
                "native_workflow_audit": workflow_audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if _RETRIEVAL_AUDIT:
        (args.out / "native_retrieval_audit.json").write_text(
            json.dumps(_RETRIEVAL_AUDIT, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(args.out / "search_policy.json")


if __name__ == "__main__":
    main()


# Last Updated At: 2026-08-01 10:20 HKT
