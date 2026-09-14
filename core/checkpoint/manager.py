"""Shadow-git based file-system checkpoint manager.

Creates an invisible git repository *outside* the user's workspace to track
file changes without polluting the project's own ``.git``.  Every git command
is executed with explicit ``GIT_DIR`` / ``GIT_WORK_TREE`` environment variables
so the shadow repo never appears inside the working tree.

Storage layout::

    .neuroclaw_checkpoints/
        {sha256(workspace)[:16]}/
            .git/         -- legacy/shared shadow repo
            sessions/
                {sha256(chat_id)[:16]}/
                    .git/ -- chat-scoped shadow repo
                    deleted_checkpoints
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

    def __init__(
        self,
        repo_root: Path,
        max_checkpoints: int = 50,
        scope_id: str | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._base_dir = self._repo_root / ".neuroclaw_checkpoints"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._scope_id = str(scope_id or "").strip()
        self._scope_hash = (
            hashlib.sha256(self._scope_id.encode("utf-8")).hexdigest()[:16]
            if self._scope_id
            else ""
        )
        self._dedup: set[str] = set()  # per-turn dedup keys

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _workspace_hash(workspace: Path) -> str:
        return hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]

    def _shadow_git_dir(self, workspace: Path) -> Path:
        workspace_root = self._base_dir / self._workspace_hash(workspace)
        if not self._scope_hash:
            return workspace_root
        return workspace_root / "sessions" / self._scope_hash

    def _deleted_checkpoints_path(self, workspace: Path) -> Path:
        return self._shadow_git_dir(workspace) / "deleted_checkpoints"

    def _deleted_checkpoints(self, workspace: Path) -> set[str]:
        path = self._deleted_checkpoints_path(workspace)
        if not path.exists():
            return set()
        try:
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if re.fullmatch(r"[0-9a-f]{40}", line.strip())
            }
        except OSError:
            return set()

    def _validate_checkpoint(self, workspace: Path, commit_hash: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", commit_hash):
            raise ValueError(f"Invalid commit hash: {commit_hash}")
        if commit_hash in self._deleted_checkpoints(workspace):
            raise ValueError("Checkpoint not found")
        result = self._run_git(
            workspace,
            "cat-file",
            "-e",
            f"{commit_hash}^{{commit}}",
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("Checkpoint not found")

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
        env["GIT_AUTHOR_NAME"] = "NeuroRuntime"
        env["GIT_AUTHOR_EMAIL"] = "checkpoint@neuroclaw.local"
        env["GIT_COMMITTER_NAME"] = "NeuroRuntime"
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

    def _force_checkpoint(self, workspace: Path, label: str) -> dict:
        """Create a snapshot unconditionally, bypassing dedup and no-change checks."""
        ws = Path(workspace).resolve()
        self._ensure_shadow_repo(ws)
        self._run_git(ws, "add", "-A")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        msg = f"checkpoint: {ts} | {label[:120]}"
        self._run_git(ws, "commit", "-m", msg, "--allow-empty")
        rev = self._run_git(ws, "rev-parse", "HEAD")
        commit_hash = rev.stdout.strip()
        self._prune(ws)
        return {"commit": commit_hash, "timestamp": ts, "files_changed": 0, "label": label}

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

        # Count changed files before committing (cached diff is cleared after commit)
        stat = self._run_git(ws, "diff", "--cached", "--stat", check=False)
        files_changed = len(
            [l for l in stat.stdout.strip().splitlines() if l and "|" in l]
        )

        # Commit
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        msg = f"checkpoint: {ts}"
        if label:
            msg += f" | {label[:120]}"

        self._run_git(ws, "commit", "-m", msg, "--allow-empty")

        # Get the commit hash
        rev = self._run_git(ws, "rev-parse", "HEAD")
        commit_hash = rev.stdout.strip()

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
        deleted = self._deleted_checkpoints(ws)
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
            if hash_val in deleted:
                continue
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
        self._validate_checkpoint(ws, commit_hash)

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
        self._ensure_shadow_repo(ws)
        self._validate_checkpoint(ws, commit_hash)
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
        self._validate_checkpoint(ws, commit_hash)

        # Pre-rollback snapshot (always create, even if no pending changes)
        self._force_checkpoint(ws, label=f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

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
        self._ensure_shadow_repo(ws)
        self._validate_checkpoint(ws, commit_hash)
        result = self._run_git(
            ws, "ls-tree", "-r", "--name-only", commit_hash, check=False
        )
        return [f for f in result.stdout.strip().splitlines() if f]

    def delete_checkpoint(self, workspace: Path, commit_hash: str) -> dict:
        """Remove one checkpoint from this manager's visible history.

        The underlying git object is retained until normal repository garbage
        collection, but all list/diff/restore/file operations reject it
        immediately. Scoped managers therefore delete only the selected chat's
        checkpoint without touching another chat's history.
        """
        ws = Path(workspace).resolve()
        self._ensure_shadow_repo(ws)
        self._validate_checkpoint(ws, commit_hash)
        deleted = self._deleted_checkpoints(ws)
        deleted.add(commit_hash)
        path = self._deleted_checkpoints_path(ws)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text("\n".join(sorted(deleted)) + "\n", encoding="utf-8")
        temporary.replace(path)
        return {"deleted": True, "hash": commit_hash}

    def _prune(self, workspace: Path) -> None:
        """Keep only the most recent *max_checkpoints* commits."""
        ws = Path(workspace).resolve()
        cps = self.list_checkpoints(ws)
        if len(cps) <= self._max_checkpoints:
            return
        # Move HEAD back to the oldest commit we want to keep
        keep_index = len(cps) - self._max_checkpoints
        keep_hash = cps[keep_index]["hash"]
        self._run_git(ws, "reset", "--hard", keep_hash, check=False)
        # Expire reflog and gc to purge orphaned commits
        self._run_git(ws, "reflog", "expire", "--expire=now", "--all", check=False)
        self._run_git(ws, "gc", "--prune=now", check=False)
