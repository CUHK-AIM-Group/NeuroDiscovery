"""Resolve the published graph, never the newest file or an old snapshot.

The explorer's existing campaign pointer takes precedence over the formal base
release. Publication receipts are small and hash-checked; graph metadata is
checked against its accepted fingerprint without rereading gigabytes here.
Explicit paths remain available for reproducible, version-pinned experiments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .shared_relation_catalog import check_file
from .kg_storage import storage_path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in graph configuration: {path}")
    return data


def _relative_to_root(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Empty graph source path")
    path = Path(value)
    return storage_path(path if path.is_absolute() else root / path)


def current_graph_path(repo_root: Path | None = None) -> Path:
    """Return the latest accepted graph, failing closed on stale publication.

    A configured but missing/in-progress campaign is an error, not permission to
    silently fall back to an older graph. No experiment or publication is run.
    """
    root = Path(repo_root or REPO_ROOT).resolve()
    configured = os.environ.get("NEUROCLAW_KG_CAMPAIGN")
    config_path = root / "neurooracle/configs/graph_explorer.json"
    if configured is None and config_path.exists():
        config = _read_json(config_path)
        if config.get("version") != 1:
            raise ValueError("Invalid graph explorer configuration version")
        configured = config.get("campaign")
        if not isinstance(configured, str):
            raise ValueError("Graph explorer configuration needs a campaign path")
    if configured is not None:
        campaign_path = _relative_to_root(configured, root)
        campaign = _read_json(campaign_path)
        if campaign.get("status") != "COMPLETED" or campaign.get("active_process") is not None:
            raise ValueError("Current graph publication is not ready; use an explicit frozen --graph or retry after acceptance")
        receipt_path = check_file(campaign["current_acceptance"], full_hash=True)
        receipt = _read_json(receipt_path)
        graph = campaign["current_graph"]
        checks = receipt.get("checks") or {}
        if receipt.get("graph") != graph or not checks.get("independent_full_structure_scan"):
            raise ValueError("Current graph does not match its validated acceptance")
        path = check_file(graph).resolve()
        if _read_json(campaign_path) != campaign:
            raise ValueError("Current graph publication changed during resolution; retry")
        return path

    state_path = root / "neurooracle/data/full_v2/CURRENT_STATE.json"
    state = _read_json(state_path)
    if state.get("status") != "canonical_current":
        raise ValueError("Formal graph state is not canonical_current")
    graph = state["canonical_files"]["knowledge_graph"]
    path = _relative_to_root(graph["path"], root)
    if not path.is_file() or path.stat().st_size != graph["bytes"]:
        raise ValueError("Formal graph is missing or differs from CURRENT_STATE.json")
    if _read_json(state_path) != state:
        raise ValueError("Formal graph state changed during resolution; retry")
    return path


def resolve_graph_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit experiment input, or the current published default."""
    if path is None:
        return current_graph_path()
    resolved = storage_path(Path(path).expanduser())
    if not resolved.is_file():
        compressed = resolved.with_suffix(resolved.suffix + ".gz")
        if compressed.is_file():
            return compressed
        raise FileNotFoundError(f"Knowledge graph is missing: {resolved}")
    return resolved


def separate_graph_output(input_path: Path, output_path: str | Path | None) -> Path:
    """Keep ordinary editing commands from writing into their published input."""
    if output_path is None:
        raise ValueError("Specify a separate --output path; the published input graph is read-only")
    output = Path(output_path).expanduser().resolve()
    if output == Path(input_path).resolve():
        raise ValueError("--output must differ from the input graph; publication is a separate operation")
    if output.exists() and output.samefile(input_path):
        raise ValueError("--output refers to the same file as the input graph")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the current published KG path without loading the graph")
    parser.add_argument("--graph", help="Explicit version-pinned input instead of the published default")
    args = parser.parse_args()
    print(resolve_graph_path(args.graph))


if __name__ == "__main__":
    main()
