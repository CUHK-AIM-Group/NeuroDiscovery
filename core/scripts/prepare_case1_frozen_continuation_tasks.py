"""Derive Case Study 1 trials 1 and 2 from each frozen trial-0 task.

The formal prompt generator evolved while trial 0 was running.  Regenerating
later tasks from the current source would therefore mix prompt versions.  This
utility preserves the exact frozen method prompt and public registry, changing
only the independent trial identifier.  Every output is write-once and the
derivation is recorded in a hash manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
NATIVE_METHODS = ("brainpilot_native", "biomni_native")
ALL_METHODS = (*OFFICIAL_METHODS, *NATIVE_METHODS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trials", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-kg-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def task_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def write_once(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite non-identical task: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def task_path(run_root: Path, method: str, trial: int) -> Path:
    if method in OFFICIAL_METHODS:
        return (
            run_root
            / "baseline_generation"
            / "formal_official"
            / method
            / f"trial_{trial:02d}"
            / "task.json"
        )
    return (
        run_root
        / "baseline_generation"
        / "native"
        / method
        / f"seed_{trial:02d}"
        / "task.json"
    )


def validate_source(
    payload: dict[str, Any],
    *,
    method: str,
    expected_registry_sha256: str,
) -> Path:
    if (
        payload.get("method") != method
        or int(payload.get("trial", -1)) != 0
        or int(payload.get("n_anchors", -1)) != 80
        or payload.get("model") != "deepseek-v4-pro"
        or payload.get("reasoning_effort") != "high"
    ):
        raise RuntimeError(f"Frozen source task contract mismatch for {method}")
    blinding = payload.get("blinding") or {}
    if not blinding or any(value is not False for value in blinding.values()):
        raise RuntimeError(f"Frozen source task blinding mismatch for {method}")
    registry_path = Path(str(payload.get("public_registry_path") or ""))
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    if sha256_file(registry_path) != expected_registry_sha256:
        raise RuntimeError(f"Public registry hash mismatch for {method}")
    goal = payload.get("research_goal")
    if not isinstance(goal, str) or goal.count("Independent trial ID: 0") != 1:
        raise RuntimeError(f"Frozen source task has an ambiguous trial marker: {method}")
    return registry_path


def main() -> None:
    args = parse_args()
    trials = tuple(args.trials)
    if trials != (1, 2):
        raise ValueError(f"Formal continuation requires exactly trials 1 2: {trials}")
    run_root = args.run_root
    expected_registry_sha256 = args.expected_registry_sha256.lower()
    records: list[dict[str, Any]] = []

    for method in ALL_METHODS:
        source_path = task_path(run_root, method, 0)
        source = load_object(source_path)
        registry_path = validate_source(
            source,
            method=method,
            expected_registry_sha256=expected_registry_sha256,
        )
        source_sha256 = sha256_file(source_path)
        for trial in trials:
            derived = json.loads(json.dumps(source, ensure_ascii=False))
            derived["trial"] = trial
            derived["research_goal"] = derived["research_goal"].replace(
                "Independent trial ID: 0",
                f"Independent trial ID: {trial}",
            )
            destination = task_path(run_root, method, trial)
            write_once(destination, task_bytes(derived))
            records.append(
                {
                    "method": method,
                    "trial": trial,
                    "source_task": str(source_path),
                    "source_task_sha256": source_sha256,
                    "destination_task": str(destination),
                    "destination_task_sha256": sha256_file(destination),
                    "public_registry": str(registry_path),
                    "public_registry_sha256": expected_registry_sha256,
                    "changed_fields": ["trial", "research_goal.trial_marker"],
                }
            )

    manifest = {
        "schema_version": "case1-frozen-continuation-tasks.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trials": list(trials),
        "methods": list(ALL_METHODS),
        "logical_model": "deepseek-v4-pro",
        "logical_reasoning_effort": "high",
        "expected_knowledge_graph_sha256": args.expected_kg_sha256.lower(),
        "expected_public_registry_sha256": expected_registry_sha256,
        "expected_official_adapter_sha256": args.expected_adapter_sha256.lower(),
        "derivation": (
            "Exact frozen trial-0 task with only the structured trial field and "
            "the unique Independent trial ID marker changed."
        ),
        "tasks": records,
    }
    manifest_path = (
        run_root
        / "baseline_generation"
        / "frozen_continuation_tasks_manifest.json"
    )
    encoded_manifest = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    if manifest_path.is_file():
        existing = load_object(manifest_path)
        existing.pop("created_at", None)
        comparable = dict(manifest)
        comparable.pop("created_at", None)
        if existing != comparable:
            raise RuntimeError(
                f"Existing continuation manifest differs from derived tasks: {manifest_path}"
            )
    else:
        write_once(manifest_path, encoded_manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "task_count": len(records),
                "methods": len(ALL_METHODS),
                "trials": list(trials),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
