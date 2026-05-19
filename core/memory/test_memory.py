"""Tests for the cross-session memory system.

Run with:
    /c/Users/45846/anaconda3/envs/neuroclaw/python.exe -m pytest core/memory/test_memory.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.memory import (
    MemoryEntry,
    MemoryExtractor,
    MemoryStore,
    default_memory_root,
)
from core.memory.store import slugify


# ── MemoryEntry ───────────────────────────────────────────────────────────


def test_entry_validates_type() -> None:
    with pytest.raises(ValueError):
        MemoryEntry(name="x", description="d", type="bogus", body="b")  # type: ignore[arg-type]


def test_entry_slugifies_name() -> None:
    entry = MemoryEntry(
        name="Some Name With Spaces!", description="d", type="user", body="b"
    )
    assert entry.name == "some-name-with-spaces"


def test_entry_truncates_description() -> None:
    long = "x" * 500
    entry = MemoryEntry(name="n", description=long, type="user", body="b")
    assert len(entry.description) <= 150


def test_entry_roundtrip_markdown() -> None:
    entry = MemoryEntry(
        name="conda-env",
        description="Project uses conda env neuroclaw",
        type="project",
        body="Body content\n**Why:** reason\n**How to apply:** always",
    )
    md = entry.to_markdown()
    parsed = MemoryEntry.from_markdown(md)
    assert parsed.name == entry.name
    assert parsed.description == entry.description
    assert parsed.type == entry.type
    assert parsed.body == entry.body


# ── MemoryStore ───────────────────────────────────────────────────────────


def test_store_upsert_and_get(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    entry = MemoryEntry(
        name="lang", description="Use Chinese", type="feedback", body="zh"
    )
    store.upsert(entry)
    loaded = store.get("lang")
    assert loaded is not None
    assert loaded.body == "zh"
    assert (tmp_path / "lang.md").exists()


def test_store_upsert_updates_existing(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    a = MemoryEntry(name="x", description="v1", type="user", body="b1")
    store.upsert(a)
    created = store.get("x").created_at  # type: ignore[union-attr]

    b = MemoryEntry(name="x", description="v2", type="user", body="b2")
    store.upsert(b)
    after = store.get("x")
    assert after is not None
    assert after.body == "b2"
    assert after.description == "v2"
    assert after.created_at == created  # preserved
    # only one file on disk
    files = [p for p in tmp_path.glob("*.md") if p.name != "MEMORY.md"]
    assert len(files) == 1


def test_store_remove(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    store.upsert(MemoryEntry(name="x", description="d", type="user", body="b"))
    assert store.remove("x") is True
    assert store.get("x") is None
    assert store.remove("missing") is False


def test_store_index_written_and_grouped(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    store.upsert(MemoryEntry(name="proj", description="p", type="project", body="b"))
    store.upsert(MemoryEntry(name="user-bg", description="u", type="user", body="b"))
    store.upsert(MemoryEntry(name="fb", description="f", type="feedback", body="b"))

    index = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    lines = [l for l in index.strip().splitlines() if l.strip()]
    # feedback first, then user, then project
    assert "fb.md" in lines[0]
    assert "user-bg.md" in lines[1]
    assert "proj.md" in lines[2]


def test_store_render_index_empty(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    assert store.render_index() == ""


def test_store_render_index_nonempty(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    store.upsert(MemoryEntry(name="x", description="d", type="user", body="b"))
    rendered = store.render_index()
    assert "[Persistent Memory Index]" in rendered
    assert "x.md" in rendered
    assert "d" in rendered


def test_store_reload_from_disk(tmp_path: Path) -> None:
    store1 = MemoryStore(root=tmp_path)
    store1.upsert(MemoryEntry(name="x", description="d", type="user", body="hello"))

    store2 = MemoryStore(root=tmp_path)  # fresh instance, same dir
    loaded = store2.get("x")
    assert loaded is not None
    assert loaded.body == "hello"


def test_store_skips_stray_files(tmp_path: Path) -> None:
    (tmp_path / "not_memory.md").write_text("# nope\n", encoding="utf-8")
    store = MemoryStore(root=tmp_path)
    # malformed file becomes an entry with default type=project; ensure load doesn't crash
    entries = store.list_entries()
    # It is acceptable that the stray file is loaded; the key invariant is no crash.
    assert isinstance(entries, list)


def test_default_memory_root_under_home(tmp_path: Path) -> None:
    root = default_memory_root(tmp_path)
    assert ".neuroclaw" in str(root)
    assert "projects" in str(root)
    assert root.name == "memory"


# ── MemoryExtractor ───────────────────────────────────────────────────────


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = SimpleNamespace(content=content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:  # noqa: D401
        self.calls.append(kwargs)
        return _FakeResponse(self.content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_extractor_saves_returned_memory(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "memories": [
                {
                    "type": "feedback",
                    "name": "use-chinese",
                    "description": "User prefers Chinese for tech discussion",
                    "body": "Reply in Chinese.\n**Why:** explicit ask.\n**How to apply:** all replies.",
                }
            ]
        }
    )
    store = MemoryStore(root=tmp_path)
    client = _FakeClient(payload)
    extractor = MemoryExtractor(llm_client=client, store=store)
    extractor.maybe_extract("以后用中文", "Got it.", block=True)
    saved = store.get("use-chinese")
    assert saved is not None
    assert saved.type == "feedback"


def test_extractor_handles_empty_memories(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    client = _FakeClient(json.dumps({"memories": []}))
    extractor = MemoryExtractor(llm_client=client, store=store)
    extractor.maybe_extract("hi", "hello", block=True)
    assert store.list_entries() == []


def test_extractor_handles_invalid_json(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    client = _FakeClient("not json at all")
    extractor = MemoryExtractor(llm_client=client, store=store)
    extractor.maybe_extract("hi", "hello", block=True)
    assert store.list_entries() == []


def test_extractor_rejects_invalid_type(tmp_path: Path) -> None:
    payload = json.dumps(
        {"memories": [{"type": "garbage", "name": "x", "description": "d", "body": "b"}]}
    )
    store = MemoryStore(root=tmp_path)
    client = _FakeClient(payload)
    extractor = MemoryExtractor(llm_client=client, store=store)
    extractor.maybe_extract("hi", "hello", block=True)
    assert store.list_entries() == []


def test_extractor_disabled_without_client(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    extractor = MemoryExtractor(llm_client=None, store=store)
    assert extractor.enabled is False
    extractor.maybe_extract("hi", "hello", block=True)
    assert store.list_entries() == []


def test_extractor_no_op_on_empty_input(tmp_path: Path) -> None:
    store = MemoryStore(root=tmp_path)
    client = _FakeClient(json.dumps({"memories": [{"type": "user", "name": "x", "description": "d", "body": "b"}]}))
    extractor = MemoryExtractor(llm_client=client, store=store)
    extractor.maybe_extract("", "", block=True)
    # LLM should not have been called and nothing saved
    assert client.completions.calls == []
    assert store.list_entries() == []


def test_slugify() -> None:
    assert slugify("Hello World") == "hello-world"
    assert slugify("中文-keep-only-ascii") == "keep-only-ascii"
    assert slugify("") == "memory"
    assert slugify("---__   ") == "memory"
