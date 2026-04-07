"""
NeuroClaw Session Manager

Manages conversation history, context-window compression, and lightweight
checkpointing so long-running neuroscience sessions can be resumed.

Design decisions
----------------
- Compression uses a sliding window: the oldest messages (beyond a configurable
  keep_recent count) are summarised into a single "context summary" assistant
  message.  The system prompt is always preserved.
- Checkpoints are written as JSON to workspace/.neuroclaw_checkpoints/.
- No external dependencies beyond the Python standard library.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    """

    def __init__(self, env: dict, keep_recent: int = DEFAULT_KEEP_RECENT) -> None:
        self.env = env
        self.keep_recent = keep_recent
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Context compression ────────────────────────────────────────────────────

    def maybe_compress(self, history: list[dict]) -> None:
        """
        Compress history in-place if it exceeds the keep_recent threshold.

        The system prompt (index 0) is always preserved.
        Messages beyond keep_recent are replaced by a summary stub.
        """
        # Count non-system messages
        user_assistant = [m for m in history if m["role"] != "system"]
        if len(user_assistant) <= self.keep_recent:
            return

        system_msgs = [m for m in history if m["role"] == "system"]
        recent = user_assistant[-self.keep_recent :]
        compressed_count = len(user_assistant) - self.keep_recent

        summary = {
            "role": "assistant",
            "content": (
                f"[Context summary: {compressed_count} earlier message(s) compressed "
                f"to save context space. Key topics covered in prior turns are available "
                f"in the session checkpoint.]"
            ),
        }

        history.clear()
        history.extend(system_msgs)
        history.append(summary)
        history.extend(recent)

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
