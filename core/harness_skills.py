"""The skill capability: definition, provider and model-facing consumers.

DeepSeek Harness makes skills a capability family of three roles
(``docs/subsystems/skills.md``): a Service Definition (``ctx.skills``), one or
more Service Providers (``dsh-skill-filesystem`` and packaged providers), and a
Consumer that is usually a model-facing tool (``packages/skill/tool-skill``,
whose tool is named ``skill``). Its criterion is explicit: "one role alone is
not a seam".

This module fills the same three roles for NeuroRuntime:

* **Definition** — :class:`SkillCapability` is the interface every caller uses:
  ``catalog()``, ``search()``, ``load()`` and ``load_section()``.
* **Provider** — :class:`FilesystemSkillProvider` scans the ``skills/``
  directories through the existing :class:`core.skill_loader.loader.SkillLoader`
  and is the only place that touches the filesystem. Further providers can be
  added later without changing the Definition or the Consumers.
* **Consumer** — ``search_skills``, ``read_skill`` and ``read_skill_section`` are registered in
  :mod:`core.tool_registry`, so the model can discover a skill and then load its
  full instructions on demand.

Before this module the repository had only the Provider: ``SkillLoader`` scanned
102 skill directories, but no model-facing tool could load one, so skills
reached the model solely as prompt-injected summaries built by
``_build_skill_exec_summary`` in ``core/agent/main.py``.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.skill_loader.loader import SkillLoader


DEFAULT_READ_MAX_CHARS = 20000
DEFAULT_SECTION_MAX_CHARS = 4800
MAX_SECTION_SOURCE_CHARS = 200000
MAX_SECTION_CACHE_ENTRIES = 32
MAX_SUMMARY_CHARS = 240


@dataclass(frozen=True)
class SkillRecord:
    """One discoverable skill: its identity and where its body lives."""

    name: str
    description: str
    summary: str
    path: Path
    skill_md: Path
    layer: str = ""
    skill_type: str = ""


class FilesystemSkillProvider:
    """Provider role: discover skills from the local ``skills/`` directories."""

    def __init__(self, skills_dir: Path, *, include_cwd: bool = True) -> None:
        self._skills_dir = Path(skills_dir)
        self._include_cwd = include_cwd

    def list(self) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        # Campaigns pin one frozen provider; interactive discovery keeps its
        # historical cwd convention unless the caller explicitly opts out.
        for raw in SkillLoader(self._skills_dir, include_cwd=self._include_cwd).load_all():
            name = str(raw.get("name") or "").strip()
            skill_md = raw.get("skill_md")
            if not name or skill_md is None:
                continue
            records.append(
                SkillRecord(
                    name=name,
                    description=re.sub(r"\s+", " ", str(raw.get("description") or "")).strip(),
                    summary=re.sub(r"\s+", " ", str(raw.get("summary_en") or "")).strip(),
                    path=Path(raw.get("path")),
                    skill_md=Path(skill_md),
                    layer=str(raw.get("layer") or ""),
                    skill_type=str(raw.get("skill_type") or ""),
                )
            )
        records.sort(key=lambda item: item.name.lower())
        return records


class SkillCapability:
    """Definition role: the interface the consumers and any caller share."""

    def __init__(self, providers: Sequence[FilesystemSkillProvider]) -> None:
        if not providers:
            raise ValueError("at least one skill provider is required")
        self._providers = list(providers)
        self._cache: list[SkillRecord] | None = None
        self._section_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()

    def catalog(self) -> list[SkillRecord]:
        """All known skills, merged across providers, nearest provider winning."""
        if self._cache is None:
            merged: dict[str, SkillRecord] = {}
            for provider in self._providers:
                for record in provider.list():
                    merged.setdefault(record.name.lower(), record)
            self._cache = sorted(merged.values(), key=lambda item: item.name.lower())
        return list(self._cache)

    def search(self, query: str, *, limit: int = 8) -> list[SkillRecord]:
        """Rank skills by keyword overlap against name, description and summary."""
        keywords = [token for token in re.split(r"[\s,;/]+", (query or "").lower()) if token]
        if not keywords:
            return []
        scored: list[tuple[int, int, SkillRecord]] = []
        for record in self.catalog():
            name = record.name.lower()
            haystack = f"{name} {record.description} {record.summary} {record.skill_type} {record.layer}".lower()
            score = 0
            for keyword in keywords:
                if keyword in name:
                    score += 3
                elif keyword in haystack:
                    score += 1
            if score:
                scored.append((-score, len(record.name), record))
        scored.sort(key=lambda item: (item[0], item[1], item[2].name.lower()))
        return [record for _, _, record in scored[: max(1, limit)]]

    def load(self, name: str, *, max_chars: int = DEFAULT_READ_MAX_CHARS, offset: int = 0) -> dict[str, Any]:
        """Load one skill body by exact name, bounded and workspace-safe."""
        wanted = (name or "").strip().lower()
        if not wanted:
            return {"success": False, "error_type": "invalid_skill_name", "error": "Skill name is required."}
        record = next((item for item in self.catalog() if item.name.lower() == wanted), None)
        if record is None:
            known = ", ".join(item.name for item in self.catalog()[:20])
            return {
                "success": False,
                "error_type": "unknown_skill",
                "error": f"Unknown skill '{name}'. Known skills include: {known}",
            }
        try:
            source_bytes = record.skill_md.read_bytes()
            body = source_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except OSError as exc:
            return {"success": False, "error_type": "skill_read_failed", "error": str(exc)}
        start = max(0, int(offset))
        limit = max(1, int(max_chars))
        chunk = body[start : start + limit]
        truncated = start + limit < len(body)
        return {
            "success": True,
            "name": record.name,
            "path": str(record.skill_md),
            "skill_dir": str(record.path),
            "description": record.description,
            "content_kind": "method_guidance",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "workflow_executed": False,
            "usage_guidance": (
                "Use the requested stage and actual inputs/output paths. Examples are not observed data. "
                "Loading a skill does not execute commands, delegate work or run verification. "
                "Use only exposed tools; proposed executable code needs real calls, failure handling "
                "and distinct running/completed/verified states, rather than print-only placeholders."
            ),
            "content": chunk,
            "offset": start,
            "total_chars": len(body),
            "truncated": truncated,
            "next_offset": start + limit if truncated else None,
        }

    def load_section(self, name: str, query: str, *, max_chars: int = DEFAULT_SECTION_MAX_CHARS) -> dict[str, Any]:
        """Select bounded method paragraphs, caching only source-bound results."""
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            return {"success": False, "error_type": "invalid_section_query", "error": "A nonempty query of at most 4096 characters is required."}
        if type(max_chars) is not int or not 1 <= max_chars <= DEFAULT_READ_MAX_CHARS:
            return {"success": False, "error_type": "invalid_section_limit", "error": "max_chars must be between 1 and 20000."}
        raw = self.load(name, max_chars=MAX_SECTION_SOURCE_CHARS)
        if not raw.get("success"):
            return raw
        # Reread the source hash before using a session cache entry, so an edited
        # skill cannot return paragraphs from its former contents.
        key = (raw["name"], query.strip(), max_chars, raw["source_sha256"])
        if key in self._section_cache:
            self._section_cache.move_to_end(key)
            return {**deepcopy(self._section_cache[key]), "cache_hit": True}
        words = [word.casefold() for word in query.split() if len(word) > 2]
        paragraphs = raw["content"].split("\n\n")
        ranked = sorted(enumerate(paragraphs),
                        key=lambda item: (-sum(word in item[1].casefold() for word in words), item[0]))
        selected, length = [], 0
        for index, paragraph in ranked:
            remaining = max_chars - length - (2 if selected else 0)
            if remaining <= 0:
                break
            piece = paragraph[:remaining]
            if piece:
                selected.append((index, piece))
                length += len(piece) + (2 if len(selected) > 1 else 0)
        content = "\n\n".join(piece for _, piece in sorted(selected))
        value = {
            "success": True, "name": raw["name"], "query": query.strip(),
            "path": raw["path"], "source_sha256": raw["source_sha256"],
            "source_content_sha256": raw["content_sha256"],
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content_kind": "method_guidance", "content": content,
            "source_total_chars": raw["total_chars"],
            "source_scanned_chars": len(raw["content"]),
            "source_scan_truncated": raw["truncated"], "source_next_offset": raw["next_offset"],
            "cached": True, "cache_hit": False, "workflow_executed": False,
            "usage_guidance": raw["usage_guidance"],
        }
        self._section_cache[key] = deepcopy(value)
        while len(self._section_cache) > MAX_SECTION_CACHE_ENTRIES:
            self._section_cache.popitem(last=False)
        return value

    def summary_line(self, record: SkillRecord) -> str:
        text = record.summary or record.description
        if len(text) > MAX_SUMMARY_CHARS:
            text = text[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."
        return f"{record.name}: {text}"


def default_capability(skills_dir: Path, *, include_cwd: bool = True) -> SkillCapability:
    return SkillCapability([FilesystemSkillProvider(skills_dir, include_cwd=include_cwd)])
