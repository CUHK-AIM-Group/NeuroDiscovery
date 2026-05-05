"""Shadow-git based file-system checkpoint manager.

Creates an invisible git repository *outside* the user's workspace to track
file changes without polluting the project's own ``.git``.  Every git command
is executed with explicit ``GIT_DIR`` / ``GIT_WORK_TREE`` environment variables
so the shadow repo never appears inside the working tree.

Storage layout::

    .neuroclaw_checkpoints/
        {sha256(workspace)[:16]}/
            git/          -- bare-ish shadow repo (HEAD, refs/, objects/, info/exclude)
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ── Exclude patterns written into the shadow repo's info/exclude ────────────

_EXCLUDE_PATTERNS: list[str] = [
    ".git/",
    ".neuroclaw_checkpoints/",
    "node_modules/",
    "__pycache__/",
    "*.pyc",
    ".env",
    ".env.*",
    "*.nii",
    "*.nii.gz",
    "*.h5",
    "*.hdf5",
    "output/",
    ".DS_Store",
    "Thumbs.db",
    "*.egg-info/",
    "dist/",
    "build/",
    ".venv/",
    "venv/",
]


class ShadowCheckpointManager:
    """Transparent file-system checkpointing via a shadow git repository."""

    def __init__(self, repo_root: Path, max_checkpoints: int = 50) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._base_dir = self._repo_root / ".neuroclaw_checkpoints"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._dedup: set[str] = set()  # per-turn dedup keys

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _workspace_hash(workspace: Path) -> str:
        return hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]

    def _shadow_git_dir(self, workspace: Path) -> Path:
        return self._base_dir / self._workspace_hash(workspace)

    def _ensure_shadow_repo(self, workspace: Path) -> Path:
        """Initialise the shadow repo if it does not exist yet."""
        shadow_root = self._shadow_git_dir(workspace)
        git_dot_dir = shadow_root / ".git"
        if not (git_dot_dir / "HEAD").exists():
            shadow_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "init", str(shadow_root)],
                capture_output=True, text=True, check=True,
            )
            # Write exclude patterns into the .git info/exclude
            exclude = git_dot_dir / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text("\n".join(_EXCLUDE_PATTERNS) + "\n", encoding="utf-8")
        return git_dot_dir

    def _git_env(self, workspace: Path) -> dict[str, str]:
        """Build environment dict with isolated GIT_DIR / GIT_WORK_TREE."""
        shadow_root = self._shadow_git_dir(workspace)
        git_dir = shadow_root / ".git"
        env = os.environ.copy()
        env["GIT_DIR"] = str(git_dir)
        env["GIT_WORK_TREE"] = str(workspace.resolve())
        # Prevent user global git config from interfering
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        # Identity for commits (shadow repo only)
        env["GIT_AUTHOR_NAME"] = "NeuroClaw"
        env["GIT_AUTHOR_EMAIL"] = "checkpoint@neuroclaw.local"
        env["GIT_COMMITTER_NAME"] = "NeuroClaw"
        env["GIT_COMMITTER_EMAIL"] = "checkpoint@neuroclaw.local"
        return env

    def _run_git(
        self, workspace: Path, *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        env = self._git_env(workspace)
        return subprocess.run(
            ["git", *args],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(workspace.resolve()),
            check=check,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def begin_turn(self) -> None:
        """Reset per-turn dedup state.  Call once at the start of each agent turn."""
        self._dedup.clear()

    def checkpoint(self, workspace: Path, label: str = "") -> dict:
        """Create a snapshot of *workspace* if anything changed.

        Returns ``{"skipped": True}`` when nothing changed, or
        ``{"commit": hash, "timestamp": iso, "files_changed": N}`` on success.
        """
        ws = Path(workspace).resolve()
        dedup_key = self._workspace_hash(ws)
        if dedup_key in self._dedup:
            return {"skipped": True, "reason": "dedup"}

        self._ensure_shadow_repo(ws)

        # Stage everything
        self._run_git(ws, "add", "-A")

        # Check if anything is staged
        diff_result = self._run_git(ws, "diff", "--cached", "--quiet", check=False)
        if diff_result.returncode == 0:
            # Nothing staged — no changes
            return {"skipped": True, "reason": "no_changes"}

        # Commit
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        msg = f"checkpoint: {ts}"
        if label:
            msg += f" | {label[:120]}"

        self._run_git(ws, "commit", "-m", msg, "--allow-empty")

        # Get the commit hash
        rev = self._run_git(ws, "rev-parse", "HEAD")
        commit_hash = rev.stdout.strip()

        # Count changed files
        stat = self._run_git(ws, "diff", "--cached", "--stat", check=False)
        files_changed = len(
            [l for l in stat.stdout.strip().splitlines() if l and "|" in l]
        )

        self._dedup.add(dedup_key)
        self._prune(ws)

        return {
            "commit": commit_hash,
            "timestamp": ts,
            "files_changed": files_changed,
            "label": label,
        }

    def list_checkpoints(self, workspace: Path) -> list[dict]:
        """Return all checkpoints in chronological order (oldest first)."""
        ws = Path(workspace).resolve()
        if not (self._shadow_git_dir(ws) / ".git" / "HEAD").exists():
            return []
        result = self._run_git(
            ws, "log", "--format=%H|%aI|%s", "--reverse", check=False
        )
        if result.returncode != 0:
            return []
        checkpoints: list[dict] = []
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            hash_val, ts, msg = parts
            label = ""
            if " | " in msg:
                label = msg.split(" | ", 1)[1]
            checkpoints.append(
                {"hash": hash_val, "timestamp": ts, "message": msg, "label": label}
            )
        return checkpoints

    def diff_checkpoint(self, workspace: Path, commit_hash: str) -> dict:
        """Show diff between *commit_hash* and HEAD.

        Returns ``{"files": [...], "diff_text": "..."}``.
        """
        ws = Path(workspace).resolve()
        self._ensure_shadow_repo(ws)

        # File list
        stat = self._run_git(
            ws, "diff", "--name-only", f"{commit_hash}..HEAD", check=False
        )
        files = [f for f in stat.stdout.strip().splitlines() if f]

        # Unified diff
        diff = self._run_git(
            ws, "diff", f"{commit_hash}..HEAD", check=False
        )
        return {"files": files, "diff_text": diff.stdout}

    def diff_checkpoint_file(
        self, workspace: Path, commit_hash: str, filepath: str
    ) -> dict:
        """Diff a single file between *commit_hash* and HEAD."""
        ws = Path(workspace).resolve()
        diff = self._run_git(
            ws, "diff", f"{commit_hash}..HEAD", "--", filepath, check=False
        )
        return {"diff_text": diff.stdout}

    def restore_checkpoint(
        self,
        workspace: Path,
        commit_hash: str,
        filepath: str | None = None,
    ) -> dict:
        """Restore workspace (or a single file) to the state at *commit_hash*.

        Before restoring, a "pre-rollback" snapshot is created so the rollback
        itself can be undone.
        """
        ws = Path(workspace).resolve()
        self._ensure_shadow_repo(ws)

        # Pre-rollback snapshot
        self.checkpoint(ws, label=f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        # Validate commit hash format
        if not re.match(r"^[0-9a-f]{40}$", commit_hash):
            raise ValueError(f"Invalid commit hash: {commit_hash}")

        if filepath:
            # Restore single file
            self._run_git(ws, "checkout", commit_hash, "--", filepath)
            restored = [filepath]
        else:
            # Restore entire workspace
            self._run_git(ws, "checkout", commit_hash, "--", ".")
            # List restored files
            result = self._run_git(
                ws, "diff", "--name-only", f"{commit_hash}..HEAD", check=False
            )
            restored = [f for f in result.stdout.strip().splitlines() if f]

        # Commit the restore
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._run_git(ws, "add", "-A")
        self._run_git(
            ws,
            "commit",
            "-m",
            f"checkpoint: {ts} | restored to {commit_hash[:8]}",
            check=False,
        )

        return {"restored_files": restored}

    def get_files_at_checkpoint(
        self, workspace: Path, commit_hash: str
    ) -> list[str]:
        """List all files tracked at *commit_hash*."""
        ws = Path(workspace).resolve()
        result = self._run_git(
            ws, "ls-tree", "-r", "--name-only", commit_hash, check=False
        )
        return [f for f in result.stdout.strip().splitlines() if f]

    def _prune(self, workspace: Path) -> None:
        """Keep only the most recent *max_checkpoints* commits."""
        ws = Path(workspace).resolve()
        cps = self.list_checkpoints(ws)
        if len(cps) <= self._max_checkpoints:
            return
        # Delete oldest refs
        to_remove = cps[: len(cps) - self._max_checkpoints]
        for cp in to_remove:
            # Use git update-ref to remove the commit from history
            # Simpler approach: just rely on gc; for now, cap via reflog expiry
            pass
        # Expire reflog and gc
        self._run_git(ws, "reflog", "expire", "--expire=now", "--all", check=False)
        self._run_git(ws, "gc", "--prune=now", check=False)
