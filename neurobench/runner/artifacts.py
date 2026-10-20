"""Durable campaign evidence and immutable input files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"


def freeze_text(path: Path, text: str) -> None:
    freeze_bytes(path, text.encode("utf-8"))


def freeze_bytes(path: Path, data: bytes) -> None:
    """Publish exact bytes, including original source-file line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError("Frozen artifact changed: %s" % path)
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class EventLog:
    """One exclusive transcript per attempt, flushed before every dispatch."""

    def __init__(self, path: Path, *, secrets=()):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8", newline="")
        self.secrets = [value for value in secrets if value]

    def write(self, kind: str, **payload) -> None:
        import datetime as dt
        row = {"event": kind, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), **payload}
        text = json.dumps(row, ensure_ascii=False, default=str)
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        self.stream.write(text + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()
