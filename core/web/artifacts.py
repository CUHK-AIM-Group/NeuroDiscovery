"""Read-only artifact access, confined to a saved conversation's workspace.

No directory crawl, generated index, model call or modification of artifacts.
"""
from pathlib import Path
import re


TEXT_LIMIT = 2 * 1024 * 1024
BINARY_LIMIT = 32 * 1024 * 1024
IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    ".bmp": "image/bmp", ".avif": "image/avif",
}


def resolve_artifact(state: dict, chat_id: str, reference: str, default_workspace: Path,
                     *, download: bool = False) -> tuple[Path, str]:
    """Resolve only an explicit file in this chat's workspace, including symlinks."""
    session = next((s for s in state.get("sessions", []) if s.get("id") == chat_id), None)
    if not session:
        raise FileNotFoundError("Conversation not found")
    workspace = session.get("workspacePath")
    if session.get("projectId"):
        project = next((p for p in state.get("projects", []) if p.get("id") == session["projectId"]), None)
        if not project:
            raise FileNotFoundError("Workspace not found")
        workspace = project.get("workspacePath")
    root = Path(workspace or default_workspace).resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError("Workspace not found")
    value = str(reference).strip()
    if (not value or len(value) > 4096 or "\x00" in value
            or value.startswith(("\\\\", "//"))
            or re.search(r"[<>|?*]", value)
            or ":" in re.sub(r"^[A-Za-z]:[/\\]", "", value)):
        raise PermissionError("Invalid artifact path")
    # Do not turn the preview API into a credential/configuration file browser.
    parts = re.split(r"[/\\]", value)
    if any(p.startswith(".") and p not in {".", ".."} for p in parts):
        raise PermissionError("Hidden files are not artifacts")
    target = Path(value)
    target = (target if target.is_absolute() else root / target).resolve(strict=True)
    if not target.is_relative_to(root):
        raise PermissionError("Artifact is outside this conversation's workspace")
    if any(p.startswith(".") for p in target.relative_to(root).parts):
        raise PermissionError("Hidden files are not artifacts")
    if not target.is_file():
        raise FileNotFoundError("Artifact is not a file")
    media_type = IMAGE_TYPES.get(target.suffix.lower(), "application/pdf" if target.suffix.lower() == ".pdf" else "text/plain; charset=utf-8")
    limit = BINARY_LIMIT if media_type.startswith("image/") or media_type == "application/pdf" else TEXT_LIMIT
    if not download and target.stat().st_size > limit:
        raise OverflowError("File is too large to preview; download the original file")
    return target, media_type
