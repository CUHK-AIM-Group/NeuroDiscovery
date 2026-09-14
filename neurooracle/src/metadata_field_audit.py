"""Coverage and conservative static-read evidence for graph metadata fields."""

from __future__ import annotations

import ast
import math
from collections import Counter, defaultdict
from pathlib import Path


PLACEHOLDERS = frozenset({"unknown", "n/a", "na", "none", "null", "unspecified", "not reported", "not available"})


def nonempty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(nonempty(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(nonempty(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True  # numeric zero and False are recorded, meaningful values


def node_class(node_id: str, record: dict) -> str:
    if node_id.startswith("CLM:"):
        return "claim"
    if node_id.startswith("CLM_CONCEPT:"):
        return "source_mention"
    if node_id.startswith("CLM_ATOM:"):
        return "umls_atom"
    if str((record.get("metadata") or {}).get("audit_ref") or "").startswith("cuis/"):
        return "new_umls_cui"
    return "existing_entity_or_infrastructure"


class Coverage:
    """Denominators are record counts, never only records with metadata."""
    def __init__(self):
        self.denominators = Counter()
        self.fields = defaultdict(Counter)
        self.types = defaultdict(Counter)

    def add(self, scope: str, record: dict) -> None:
        self.denominators[scope] += 1
        md = record.get("metadata") or {}
        if not isinstance(md, dict):
            raise ValueError("metadata is not an object")
        for key, value in md.items():
            self._field(scope, "metadata." + key, value)
            # One nested level exposes evidence and source-paper sparsity; no
            # list expansion or assumption that a nonempty object is complete.
            if isinstance(value, dict):
                for child, nested in value.items():
                    self._field(scope, f"metadata.{key}.{child}", nested)
        for key, value in record.items():
            if key != "metadata":
                self._field(scope, "top." + key, value)

    def _field(self, scope: str, field: str, value) -> None:
        key = (scope, field)
        item = self.fields[key]
        item["present"] += 1
        item["nonempty"] += int(nonempty(value))
        item["placeholder_like"] += int(isinstance(value, str) and value.strip().casefold() in PLACEHOLDERS)
        self.types[key][type(value).__name__] += 1

    def rows(self) -> list[dict]:
        result = []
        for (scope, field), counts in sorted(self.fields.items()):
            total = self.denominators[scope]
            result.append({"scope": scope, "field": field, "denominator": total,
                           "present": counts["present"], "nonempty": counts["nonempty"],
                           "empty_or_null": counts["present"] - counts["nonempty"],
                           "absent": total - counts["present"],
                           "placeholder_like": counts["placeholder_like"],
                           "present_pct": round(100 * counts["present"] / total, 6),
                           "nonempty_pct": round(100 * counts["nonempty"] / total, 6),
                           "types": dict(self.types[scope, field])})
        return result


def static_key_reads(path: Path, keys: set[str]) -> tuple[list[dict], list[dict]]:
    """Key reads are candidates, not proof of graph-field lineage or liveness.

    Includes literal-key .get and read subscripts; assignments are excluded.
    Dynamic lookups are disclosed separately, not mislabelled unused.
    """
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    reads, dynamic = [], []
    for node in ast.walk(tree):
        key, holder, mode = None, None, None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            key, holder, mode = node.args[0], node.func.value, "get"
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key, holder, mode = node.slice, node.value, "subscript_read"
        if key is None:
            continue
        name = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
        if name is not None and name not in keys:
            continue
        function, parent = "<module>", node
        while parent in parents:
            parent = parents[parent]
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = parent.name
                break
        holder_text = ast.unparse(holder)
        row = {"path": str(path.resolve()), "line": node.lineno, "function": function,
               "key": name, "holder": holder_text[:180], "operation": mode,
               "source_line": lines[node.lineno - 1].strip()[:300],
               "confidence": "static_key_read_candidate_not_proven_metadata_lineage"}
        if name is None:
            if any(token in holder_text.lower() for token in ("meta", "claim", "evidence", "paper", "node", "edge")):
                dynamic.append(row)
        else:
            reads.append(row)
    return reads, dynamic
