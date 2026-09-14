"""Explicit, verifiable deliverable contracts for research skill outputs.

This does not certify scientific or clinical validity. Only user-selected input
artifacts are fingerprinted; never pass credentials or credential files here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    Path(path).write_text(json.dumps(value, default=convert, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def subject_ids(values, n):
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f"subject_ids must be a list of length {n}")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Subject IDs must be nonempty strings")
    if len(set(values)) != n:
        raise ValueError("Subject IDs must be unique (one row per subject)")
    return values


def _within(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Artifact escapes output directory: {name}")
    return path


def _check_artifact(path, spec):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty artifact: {path.name}")
    kind = spec.get("kind", "file")
    if kind == "json":
        read_json(path)
    elif kind == "csv":
        import pandas as pd
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(data) == 0 or not set(spec.get("columns", [])).issubset(data.columns):
            raise ValueError(f"Empty table or missing columns: {path.name}")
        if "rows" in spec and len(data) != spec["rows"]:
            raise ValueError(f"Wrong row count: {path.name}")
        if "subject_ids" in spec and data["subject_id"].tolist() != spec["subject_ids"]:
            raise ValueError(f"Subject identity/order mismatch: {path.name}")
    elif kind in {"npy", "nifti"}:
        if kind == "npy":
            data = np.load(path, allow_pickle=False)
        else:
            import nibabel as nib
            img = nib.load(path)
            data = img.get_fdata()
            if "affine" in spec and not np.allclose(img.affine, spec["affine"], atol=1e-5):
                raise ValueError(f"Affine mismatch: {path.name}")
        if list(data.shape) != spec["shape"]:
            raise ValueError(f"Wrong array shape: {path.name}")
        if np.isinf(data).any() or not np.isfinite(data).any():
            raise ValueError(f"Invalid numeric artifact: {path.name}")
        if spec.get("finite", True) and not np.isfinite(data).all():
            raise ValueError(f"Nonfinite array: {path.name}")


def validate_manifest(path, required=()):
    path = Path(path)
    manifest = read_json(path)
    if manifest.get("schema") != "neurodiscovery.research-outputs.v1":
        raise ValueError("Unsupported output manifest")
    artifacts = manifest.get("artifacts", [])
    roles = {item["role"] for item in artifacts}
    missing = (set(required) | set(manifest.get("required", []))) - roles
    if missing:
        raise ValueError(f"Missing required outputs: {', '.join(sorted(missing))}")
    if not artifacts or len({x["path"] for x in artifacts}) != len(artifacts):
        raise ValueError("Artifact list must be nonempty with unique paths")
    for item in artifacts:
        artifact = _within(path.parent, item["path"])
        _check_artifact(artifact, item)
        actual = fingerprint(artifact)
        if any(actual[key] != item[key] for key in ("sha256", "bytes")):
            raise ValueError(f"Artifact changed: {item['path']}")
    return manifest


class OutputBundle:
    """Never overwrite an existing run; the completion manifest is written last."""

    def __init__(self, directory, workflow, config, inputs=()):
        self.root = Path(directory)
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise ValueError("Output directory must be new or empty; preserve previous runs")
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "schema": "neurodiscovery.research-outputs.v1",
            "workflow": workflow, "research_only": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "config": config,
            "inputs": [fingerprint(p) for p in inputs], "artifacts": [],
        }

    def add(self, name, role, kind="file", **spec):
        path = _within(self.root, name)
        item = {"path": name, "role": role, "kind": kind, **spec}
        _check_artifact(path, item)
        item.update({k: v for k, v in fingerprint(path).items() if k != "name"})
        self.manifest["artifacts"].append(item)

    def finish(self, required):
        self.manifest["required"] = list(required)
        pending = self.root / "manifest.pending.json"
        write_json(pending, self.manifest)
        validate_manifest(pending)
        path = self.root / "run_manifest.json"
        pending.replace(path)
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require", nargs="+", default=[])
    args = parser.parse_args(argv)
    validate_manifest(args.manifest, args.require)
    print("Output contract verified (not clinical validation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
