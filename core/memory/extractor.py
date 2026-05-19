"""LLM-driven memory signal detection from conversation turns.

Runs after every chat turn. Sends a small prompt to a lightweight model
(``gpt-4o-mini`` by default) describing the latest user/assistant exchange
and asks whether it contains anything worth saving as a long-lived memory.

The extractor is conservative: it returns ``[]`` for the vast majority of
turns. It writes only when the turn carries an explicit signal (user
correction, user identity disclosure, project decision/deadline, external
resource pointer, or an explicit ``"remember this"`` from the user).

Output schema (the model is asked to return JSON):

    {
      "memories": [
        {
          "type": "user|feedback|project|reference",
          "name": "kebab-case-slug",
          "description": "one-line summary",
          "body": "full memory body, with **Why:** / **How to apply:** for feedback/project"
        }
      ]
    }

If the model returns ``{"memories": []}`` or anything unparseable, nothing
is saved. Errors never propagate to the caller.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

from .store import MemoryEntry, MemoryStore, VALID_TYPES

logger = logging.getLogger(__name__)


_EXTRACTOR_SYSTEM_PROMPT = """You decide whether the latest conversation turn contains a fact worth saving as long-lived memory for an AI coding assistant. Return strict JSON: {"memories": [...]}. Default to {"memories": []} unless a clear signal is present.

Save when the turn shows ANY of:
- user identity, role, expertise, or stable preference (type=user)
- user correction of the assistant, or explicit confirmation of a non-obvious approach (type=feedback)
- project decision, motivation, deadline, named milestone, environment/version pin, or constraint that future sessions need (type=project)
- pointer to an external resource (Linear board, dashboard URL, internal repo, channel) (type=reference)
- the user literally says "remember this" / "记住这个" / similar.

Do NOT save:
- code patterns, file paths, function names that can be re-read from the repo
- bug fixes already applied (the diff is the record)
- transient task state ("working on X right now")
- restating something already obvious from the user's request
- anything sounding like a critique of the user

For type=feedback or type=project, the body MUST contain two lines:
**Why:** <reason the user gave>
**How to apply:** <when this guidance kicks in>

Body language: prefer the user's language (Chinese stays Chinese).
Names: short kebab-case slug, e.g. feedback-chinese, project-merge-freeze-2026-03.
Descriptions: under 150 chars, one line, specific.

Return JSON only, no prose.
"""


_USER_PROMPT_TEMPLATE = """Existing memory index (do not duplicate these):
{index}

Latest user message:
{user_msg}

Latest assistant reply:
{assistant_msg}

Return JSON: {{"memories": [...]}}
"""


class MemoryExtractor:
    """Wraps an LLM client to extract memory entries from a single turn.

    Parameters
    ----------
    llm_client : Any
        OpenAI-compatible client (must expose ``chat.completions.create``).
    store : MemoryStore
        Destination store. ``upsert`` is called for each accepted memory.
    model : str
        Model name. Defaults to ``gpt-4o-mini`` for low cost.
    enabled : bool
        Master switch. When False, ``maybe_extract`` is a no-op.
    """

    def __init__(
        self,
        llm_client: Any,
        store: MemoryStore,
        *,
        model: str = "gpt-4o-mini",
        enabled: bool = True,
    ) -> None:
        self._llm = llm_client
        self._store = store
        self._model = model
        self.enabled = enabled and llm_client is not None
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────

    def maybe_extract(
        self,
        user_msg: str,
        assistant_msg: str,
        *,
        block: bool = False,
    ) -> Optional[threading.Thread]:
        """Run extraction in a background thread (default) or inline.

        Returns the worker thread when ``block=False``, or ``None`` when
        extraction is disabled or run inline.
        """
        if not self.enabled:
            return None
        if not user_msg.strip() and not assistant_msg.strip():
            return None

        if block:
            self._run(user_msg, assistant_msg)
            return None

        thread = threading.Thread(
            target=self._run,
            args=(user_msg, assistant_msg),
            daemon=True,
            name="memory-extractor",
        )
        thread.start()
        return thread

    # ── internals ─────────────────────────────────────────────────────────

    def _run(self, user_msg: str, assistant_msg: str) -> None:
        try:
            entries = self._call_llm(user_msg, assistant_msg)
        except Exception as exc:
            logger.debug("Memory extraction failed: %s", exc)
            return

        if not entries:
            return

        with self._lock:
            for raw in entries:
                entry = self._coerce_entry(raw)
                if entry is None:
                    continue
                try:
                    self._store.upsert(entry)
                    logger.info("Saved memory: %s (%s)", entry.name, entry.type)
                except Exception as exc:
                    logger.debug("Skipped invalid memory %r: %s", raw, exc)

    def _call_llm(self, user_msg: str, assistant_msg: str) -> list[dict[str, Any]]:
        index = self._render_index_for_prompt()
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            index=index or "(empty)",
            user_msg=_truncate(user_msg, 4000),
            assistant_msg=_truncate(assistant_msg, 4000),
        )

        response = self._llm.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        memories = data.get("memories", [])
        if not isinstance(memories, list):
            return []
        return memories

    def _render_index_for_prompt(self) -> str:
        entries = self._store.list_entries()
        if not entries:
            return ""
        return "\n".join(
            f"- {e.name} ({e.type}): {e.description}" for e in entries
        )

    @staticmethod
    def _coerce_entry(raw: dict[str, Any]) -> Optional[MemoryEntry]:
        if not isinstance(raw, dict):
            return None
        type_ = raw.get("type")
        name = raw.get("name") or ""
        description = raw.get("description") or ""
        body = raw.get("body") or ""

        if type_ not in VALID_TYPES:
            return None
        if not name or not body.strip():
            return None
        try:
            return MemoryEntry(
                name=name,
                description=description,
                type=type_,  # type: ignore[arg-type]
                body=body,
            )
        except Exception:
            return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"
