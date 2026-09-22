"""Freeze the exact source and configuration used by a long experiment.

Bundled files are copied into an immutable directory. Large data inputs are
recorded as hash-only references so the bundle stays small while still tying a
run to exact bytes. Environment variables are deliberately not captured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "neurodiscovery-experiment-source-bundle.v1"
MAX_BUNDLED_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_RELATIVE_PATH_CHARS = 96
MAX_WINDOWS_DESTINATION_PATH_CHARS = 220
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".codegraph",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _expand_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: dict[Path, Path] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if any(part in IGNORED_DIRECTORY_NAMES for part in candidate.parts):
                continue
            if candidate.suffix.lower() in {".pyc", ".pyo"}:
                continue
            resolved = candidate.resolve()
            expanded[resolved] = resolved
    return sorted(expanded.values(), key=lambda value: str(value).casefold())


def _archive_name(
    path: Path,
    repo_root: Path,
    *,
    destination_prefix_chars: int = 0,
) -> Path:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        relative = Path("external") / path.name
    archive = Path("files") / relative
    destination_chars = destination_prefix_chars + 1 + len(archive.as_posix())
    if (
        len(archive.as_posix()) <= MAX_BUNDLE_RELATIVE_PATH_CHARS
        and destination_chars <= MAX_WINDOWS_DESTINATION_PATH_CHARS
    ):
        return relative
    # Windows still commonly enforces MAX_PATH. Preserve the original path in
    # the manifest and shorten only the copied archive location.
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
    return Path("long") / f"{digest}{path.suffix.lower()}"


def _file_records(
    *,
    source_paths: Sequence[Path],
    config_paths: Sequence[Path],
    repo_root: Path,
    destination_prefix_chars: int = 0,
) -> list[dict[str, Any]]:
    roles_by_path: dict[Path, set[str]] = {}
    for role, paths in (("source", source_paths), ("config", config_paths)):
        for path in _expand_paths(paths):
            roles_by_path.setdefault(path, set()).add(role)

    records: list[dict[str, Any]] = []
    names: dict[str, Path] = {}
    for path in sorted(roles_by_path, key=lambda value: str(value).casefold()):
        size = path.stat().st_size
        if size > MAX_BUNDLED_FILE_BYTES:
            raise ValueError(
                f"bundled file exceeds {MAX_BUNDLED_FILE_BYTES} bytes; "
                f"record it as --reference instead: {path}"
            )
        archive_path = _archive_name(
            path,
            repo_root,
            destination_prefix_chars=destination_prefix_chars,
        ).as_posix()
        collision = names.get(archive_path.casefold())
        if collision is not None and collision != path:
            raise ValueError(
                f"two files map to the same bundle path {archive_path}: "
                f"{collision} and {path}"
            )
        names[archive_path.casefold()] = path
        records.append(
            {
                "source_path": str(path),
                "bundle_path": f"files/{archive_path}",
                "roles": sorted(roles_by_path[path]),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    return records


def _reference_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in _expand_paths(paths):
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "tracked_worktree_dirty": bool(
            command("status", "--porcelain", "--untracked-files=no")
        ),
    }


def _verify_existing_bundle(
    output_dir: Path,
    *,
    files: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    command: str | None,
) -> dict[str, Any]:
    manifest_path = output_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("bundle directory exists without bundle_manifest.json")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("schema_version") != SCHEMA:
        raise ValueError("existing source bundle uses an incompatible schema")
    if existing.get("files") != list(files):
        raise ValueError("source or configuration bytes changed after bundle freeze")
    if existing.get("references") != list(references):
        raise ValueError("referenced input bytes changed after bundle freeze")
    if existing.get("command") != command:
        raise ValueError("experiment command differs from the frozen bundle")
    for record in files:
        archived = output_dir / str(record["bundle_path"])
        if not archived.is_file() or _sha256(archived) != record["sha256"]:
            raise ValueError(f"bundled file was modified or removed: {archived}")
    return existing


def freeze_source_bundle(
    *,
    source_paths: Sequence[Path],
    config_paths: Sequence[Path],
    reference_paths: Sequence[Path],
    output_dir: Path,
    command: str | None = None,
    repo_root: Path = REPO,
    registered_at: str | None = None,
) -> dict[str, Any]:
    """Create or verify one immutable source/configuration bundle."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    temporary_prefix_chars = len(str(output_dir.parent / ".fb-12345678"))
    files = _file_records(
        source_paths=source_paths,
        config_paths=config_paths,
        repo_root=repo_root,
        destination_prefix_chars=temporary_prefix_chars,
    )
    if not files:
        raise ValueError("at least one source or configuration file is required")
    references = _reference_records(reference_paths)

    if output_dir.exists():
        return _verify_existing_bundle(
            output_dir,
            files=files,
            references=references,
            command=command,
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fb-", dir=output_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        for record in files:
            source = Path(str(record["source_path"]))
            destination = temporary / str(record["bundle_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if _sha256(destination) != record["sha256"]:
                raise RuntimeError(f"source copy verification failed: {source}")
        manifest = {
            "schema_version": SCHEMA,
            "registered_at": registered_at or datetime.now(timezone.utc).isoformat(),
            "repo_root": str(repo_root),
            "command": command,
            "environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "environment_variables_captured": False,
            },
            "git": _git_metadata(repo_root),
            "files": files,
            "references": references,
        }
        _atomic_json(temporary / "bundle_manifest.json", manifest)
        temporary.replace(output_dir)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--config", action="append", type=Path, default=[])
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--command")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = freeze_source_bundle(
        source_paths=args.source,
        config_paths=args.config,
        reference_paths=args.reference,
        output_dir=args.output_dir,
        command=args.command,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "bundled_files": len(manifest["files"]),
                "referenced_files": len(manifest["references"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
