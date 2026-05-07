"""
NeuroClaw Session Manager

Manages conversation history, context-window compression, and lightweight
checkpointing so long-running neuroscience sessions can be resumed.

Design decisions
----------------
- Compression uses a sliding window: the oldest messages (beyond a configurable
  keep_recent count) are summarised into a single "context summary" assistant
  message.  The system prompt is always preserved.
- Supports two compression modes:
  * stub (default): Simple placeholder text, zero cost
  * llm_summary: LLM-generated semantic summary, requires LLM client
- Checkpoints are written as JSON to workspace/.neuroclaw_checkpoints/.
- No external dependencies beyond the Python standard library.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CHECKPOINT_DIR = REPO_ROOT / ".neuroclaw_checkpoints"

# Maximum number of recent turns to keep verbatim before compression triggers
DEFAULT_KEEP_RECENT = 20
# Maximum number of checkpoint files to retain
MAX_CHECKPOINTS = 5


class SessionManager:
    """
    Manages context compression and checkpointing for a NeuroClaw session.

    Parameters
    ----------
    env : dict
        Loaded neuroclaw_environment.json content.
    keep_recent : int
        Number of recent turns to keep verbatim during compression.
    llm_client : Any, optional
        LLM client for generating semantic summaries. If None, uses stub mode.
    compression_mode : str
        "stub" (default) or "llm_summary"
    """

    def __init__(
        self,
        env: dict,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        llm_client: Optional[Any] = None,
        compression_mode: str = "stub"
    ) -> None:
        self.env = env
        self.keep_recent = keep_recent
        self.llm_client = llm_client
        self.compression_mode = compression_mode
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Context compression ────────────────────────────────────────────────────

    def maybe_compress(self, history: list[dict]) -> None:
        """
        Compress history in-place if it exceeds the keep_recent threshold.

        The system prompt (index 0) is always preserved.
        Messages beyond keep_recent are replaced by a summary (stub or LLM-generated).
        """
        # Count non-system messages
        user_assistant = [m for m in history if m["role"] != "system"]
        if len(user_assistant) <= self.keep_recent:
            return

        system_msgs = [m for m in history if m["role"] == "system"]
        recent = user_assistant[-self.keep_recent :]
        old_messages = user_assistant[: -self.keep_recent]
        compressed_count = len(old_messages)

        # Generate summary based on compression mode
        if self.compression_mode == "llm_summary" and self.llm_client is not None:
            summary_content = self._generate_llm_summary(old_messages, compressed_count)
        else:
            # Fallback to stub mode
            summary_content = (
                f"[Context summary: {compressed_count} earlier message(s) compressed "
                f"to save context space. Key topics covered in prior turns are available "
                f"in the session checkpoint.]"
            )

        summary = {
            "role": "assistant",
            "content": summary_content,
        }

        history.clear()
        history.extend(system_msgs)
        history.append(summary)
        history.extend(recent)

    def _generate_llm_summary(self, old_messages: list[dict], count: int) -> str:
        """
        Generate a semantic summary of old messages using LLM.

        Parameters
        ----------
        old_messages : list[dict]
            The messages to be summarized.
        count : int
            Number of messages being compressed.

        Returns
        -------
        str
            LLM-generated summary text.
        """
        try:
            # Build prompt for summarization
            messages_text = "\n\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in old_messages
            ])

            prompt = (
                f"Summarize the following {count} conversation turn(s) into 2-3 concise sentences. "
                f"Focus on key topics, decisions, and outcomes. Be specific and factual.\n\n"
                f"{messages_text}\n\n"
                f"Summary:"
            )

            # Call LLM (use lightweight model for cost efficiency)
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Lightweight model for summarization
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            summary_text = response.choices[0].message.content.strip()

            # Format as context summary
            return f"[Context summary: {summary_text}]"

        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}. Falling back to stub.")
            # Fallback to stub on error
            return (
                f"[Context summary: {count} earlier message(s) compressed "
                f"to save context space. Key topics covered in prior turns are available "
                f"in the session checkpoint.]"
            )

    # ── Checkpointing ──────────────────────────────────────────────────────────

    def save_checkpoint(self, history: list[dict], metadata: dict | None = None) -> Path:
        """
        Save current history to a timestamped JSON checkpoint file.

        Returns the path of the written checkpoint.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{ts}.json"

        payload: dict[str, Any] = {
            "timestamp": ts,
            "history": history,
            "metadata": metadata or {},
            "env_snapshot": {
                "setup_type": self.env.get("setup_type"),
                "python_path": self.env.get("python_path"),
                "conda_env": self.env.get("conda_env"),
                "cuda": self.env.get("cuda"),
                "llm_backend": {
                    k: v
                    for k, v in self.env.get("llm_backend", {}).items()
                    if k != "api_key_env"  # never persist key names to disk
                },
            },
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2))
        self._prune_old_checkpoints()
        return checkpoint_path

    def load_latest_checkpoint(self) -> dict | None:
        """Return the most recent checkpoint payload, or None if none exist."""
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        with checkpoints[-1].open() as f:
            return json.load(f)

    def _prune_old_checkpoints(self) -> None:
        """Keep at most MAX_CHECKPOINTS files, deleting the oldest."""
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"))
        for old in checkpoints[:-MAX_CHECKPOINTS]:
            old.unlink(missing_ok=True)
