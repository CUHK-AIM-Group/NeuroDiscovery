"""NeuroClaw cross-session memory system.

Stores user preferences, project context, feedback, and external references
as fact-based markdown files under ``~/.neuroclaw/projects/<hash>/memory/``.

Public API:
    MemoryStore: persistence + index management
    MemoryEntry: single memory record (user/feedback/project/reference)
    MemoryExtractor: LLM-driven signal detection from conversation turns
    default_memory_root: per-workspace memory directory under user home
"""

from .store import MemoryEntry, MemoryStore, MemoryType, default_memory_root
from .extractor import MemoryExtractor

__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "MemoryType",
    "MemoryExtractor",
    "default_memory_root",
]
