"""Persistent fact-based memory storage.

Each memory is one markdown file with YAML frontmatter:

    ---
    name: kebab-case-slug
    description: one-line summary
    type: user|feedback|project|reference
    created_at: ISO 8601
    updated_at: ISO 8601
    ---
    body text (with optional **Why:** / **How to apply:** for feedback/project)

Files live in ``<root>/<name>.md``. ``<root>/MEMORY.md`` is an auto-generated
index of one line per file, designed to be injected into the system prompt.

Storage root defaults to ``~/.neuroclaw/projects/<sha256(workspace)[:16]>/memory/``
so memory is per-workspace, user-local, and never written into the project repo.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

logger = logging.getLogger(__name__)

MemoryType = Literal["user", "feedback", "project", "reference"]
VALID_TYPES: tuple[str, ...] = ("user", "feedback", "project", "reference")

INDEX_FILENAME = "MEMORY.md"
MAX_INDEX_LINES = 200
MAX_DESCRIPTION_LEN = 150
MAX_NAME_LEN = 80

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def default_memory_root(workspace: Path) -> Path:
    """Return the per-workspace memory root under the user home directory.

    Layout: ``~/.neuroclaw/projects/<sha256(workspace)[:16]>/memory/``
    """
    ws = Path(workspace).resolve()
    digest = hashlib.sha256(str(ws).encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".neuroclaw" / "projects" / digest / "memory"


def slugify(text: str) -> str:
    """Convert free text to a safe kebab-case slug."""
    text = text.strip().lower()
    text = _SLUG_RE.sub("-", text).strip("-")
    if not text:
        text = "memory"
    return text[:MAX_NAME_LEN]


@dataclass
class MemoryEntry:
    """A single memory record."""

    name: str
    description: str
    type: MemoryType
    body: str
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if self.type not in VALID_TYPES:
            raise ValueError(
                f"Invalid memory type {self.type!r}; must be one of {VALID_TYPES}"
            )
        self.name = slugify(self.name)
        self.description = self.description.strip()[:MAX_DESCRIPTION_LEN]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_markdown(self) -> str:
        return (
            "---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"type: {self.type}\n"
            f"created_at: {self.created_at}\n"
            f"updated_at: {self.updated_at}\n"
            "---\n"
            f"{self.body.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "MemoryEntry":
        front, body = _split_frontmatter(text)
        return cls(
            name=front.get("name", "unnamed"),
            description=front.get("description", ""),
            type=front.get("type", "project"),  # type: ignore[arg-type]
            body=body,
            created_at=front.get("created_at", ""),
            updated_at=front.get("updated_at", ""),
        )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the ``---`` YAML-ish frontmatter block from a markdown file.

    Only flat ``key: value`` pairs are supported (sufficient for this format).
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    front_raw, body = parts[1], parts[2]
    meta: dict[str, str] = {}
    for line in front_raw.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.strip()


@dataclass
class MemoryStore:
    """Reads, writes, and indexes memory entries on disk.

    Parameters
    ----------
    root : Path
        Directory holding memory files. Created if it does not exist.
    """

    root: Path
    _entries: list[MemoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._reload()

    # ── disk I/O ──────────────────────────────────────────────────────────

    def _reload(self) -> None:
        self._entries.clear()
        for path in sorted(self.root.glob("*.md")):
            if path.name == INDEX_FILENAME:
                continue
            try:
                entry = MemoryEntry.from_markdown(path.read_text(encoding="utf-8"))
                self._entries.append(entry)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Skipping unreadable memory file %s: %s", path, exc)

    def _path_for(self, name: str) -> Path:
        return self.root / f"{slugify(name)}.md"

    # ── public API ────────────────────────────────────────────────────────

    def list_entries(self) -> list[MemoryEntry]:
        """Return all entries (in-memory copy)."""
        return list(self._entries)

    def get(self, name: str) -> Optional[MemoryEntry]:
        slug = slugify(name)
        for entry in self._entries:
            if entry.name == slug:
                return entry
        return None

    def upsert(self, entry: MemoryEntry) -> MemoryEntry:
        """Insert or update by name. Updates ``updated_at`` on overwrite."""
        existing = self.get(entry.name)
        if existing is not None:
            entry.created_at = existing.created_at
            entry.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._entries = [e for e in self._entries if e.name != entry.name]
        self._entries.append(entry)
        self._path_for(entry.name).write_text(entry.to_markdown(), encoding="utf-8")
        self._write_index()
        return entry

    def remove(self, name: str) -> bool:
        slug = slugify(name)
        path = self._path_for(slug)
        if not path.exists():
            return False
        path.unlink()
        self._entries = [e for e in self._entries if e.name != slug]
        self._write_index()
        return True

    # ── index ─────────────────────────────────────────────────────────────

    def _write_index(self) -> None:
        """Rewrite ``MEMORY.md`` from the current entry list."""
        # Group by type, ordered for stable output
        order = {"feedback": 0, "user": 1, "project": 2, "reference": 3}
        sorted_entries = sorted(
            self._entries, key=lambda e: (order.get(e.type, 99), e.name)
        )

        lines: list[str] = []
        for entry in sorted_entries[:MAX_INDEX_LINES]:
            desc = entry.description or "(no description)"
            lines.append(f"- [{entry.name}]({entry.name}.md) — {desc}")

        content = "\n".join(lines) + ("\n" if lines else "")
        (self.root / INDEX_FILENAME).write_text(content, encoding="utf-8")

    def render_index(self) -> str:
        """Return the index content for system-prompt injection.

        Empty string when no memories exist.
        """
        if not self._entries:
            return ""
        index_path = self.root / INDEX_FILENAME
        if not index_path.exists():
            self._write_index()
        body = index_path.read_text(encoding="utf-8").strip()
        if not body:
            return ""
        header = (
            "[Persistent Memory Index]\n"
            "Cross-session facts about the user and project. "
            "Entries with names you want to use must be confirmed against "
            "current files/code before acting on them.\n"
        )
        return f"\n\n{header}\n{body}"

    def render_full_context(self, names: Iterable[str]) -> str:
        """Return concatenated full bodies for the named entries.

        Used when the agent decides to actually load specific memory bodies
        rather than just the index.
        """
        chunks: list[str] = []
        for name in names:
            entry = self.get(name)
            if entry is not None:
                chunks.append(f"## {entry.name} ({entry.type})\n{entry.body}")
        return "\n\n".join(chunks)
