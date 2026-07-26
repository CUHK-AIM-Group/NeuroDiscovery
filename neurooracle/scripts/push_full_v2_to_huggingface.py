"""Publish the current NeuroOracle full_v2 graph bundle to Hugging Face Spaces.

The Space downloader in ``core/web/server.py`` expects:

    neurooracle/data/knowledge_graph.json.gz

This script refreshes the gzip from ``full_v2/knowledge_graph.json`` and uploads
the runtime graph bundle, not local backups or staging artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "zxcvb20001/NeuroOracle"
DEFAULT_REPO_TYPE = "space"
DEFAULT_KEYS = ROOT / ".env.keys"
GRAPH_SRC = ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
GRAPH_GZ = ROOT / "neurooracle" / "data" / "knowledge_graph.json.gz"


def _load_env_keys(path: Path) -> None:
    if not path.exists():
        return
    pattern = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$""")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        key, raw = m.groups()
        value = raw.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gzip_graph(src: Path, dst: Path, compresslevel: int = 6) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as f_in, gzip.open(tmp, "wb", compresslevel=compresslevel) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    tmp.replace(dst)


def _upload(api: HfApi, repo_id: str, repo_type: str, local: Path, remote: str, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Update {remote}",
            )
        except Exception as exc:  # network errors from httpx/huggingface_hub vary by version
            last_error = exc
            if attempt >= retries:
                break
            wait = min(60, 5 * attempt)
            print(f"Upload failed for {remote} on attempt {attempt}/{retries}: {type(exc).__name__}: {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)
    raise last_error or RuntimeError(f"Upload failed for {remote}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    parser.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
    parser.add_argument("--skip-compress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    _load_env_keys(args.keys)
    token = (
        os.environ.get("HUGGINGFACE_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token and not args.dry_run:
        raise SystemExit("Missing Hugging Face token: set HUGGINGFACE_KEY or HF_TOKEN")

    if not GRAPH_SRC.exists():
        raise SystemExit(f"Missing graph source: {GRAPH_SRC}")
    if not args.skip_compress:
        print(f"Compressing {GRAPH_SRC} -> {GRAPH_GZ}")
        _gzip_graph(GRAPH_SRC, GRAPH_GZ)

    uploads = [(GRAPH_GZ, "neurooracle/data/knowledge_graph.json.gz")]

    print(f"Target: {args.repo_type}:{args.repo_id}")
    for local, remote in uploads:
        print(f"{remote}\t{local.stat().st_size}\tsha256={_sha256(local)[:16]}")

    if args.dry_run:
        return

    api = HfApi(token=token)
    info = api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    print(f"Remote repo ok: {info.id}")

    for local, remote in uploads:
        url = _upload(api, args.repo_id, args.repo_type, local, remote, retries=max(1, args.retries))
        print(f"Uploaded {remote}: {url}")


if __name__ == "__main__":
    main()
