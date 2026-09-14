"""Lossless, standalone KG reading views, never a replacement graph schema.

No schema decoder, numerical coercion, entity merging, model or network call.
Every original JSON leaf (including empty containers) keeps its exact path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from html import escape
import json

SCHEMA = "kg.reading_view.v1"
GROUPS = (
    ("core", "关系与原文", True),
    ("context", "科学限定与类型", True),
    ("source", "来源与可选统计", False),
    ("other", "审核与其余字段（完整保留）", False),
)
CORE = frozenset("id preferred_name definition subject_id subject_name predicate object_id object_name negated confidence raw_text source_id target_id relation_type claim_id".split())
CONTEXT = frozenset("subject_type object_type conditions population atom_type atom_types semantic_types domain_tags scope_evidence_spans evidence_span original_predicate predicate_normalized_from predicate_normalized_to conditioning polarity direction study_type methodology replicability".split())
SOURCE = frozenset("source_paper evidence raw_stats source source_vocab evidence_ref external_ids audit_ref source_mention_id mapping_status mapping_count".split())
LAYERS = ("relations", "claim_anchors", "mappings")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _check_json(value):
    if isinstance(value, dict):
        _require(all(isinstance(k, str) for k in value), "JSON keys must be strings")
        for child in value.values():
            _check_json(child)
    elif isinstance(value, list):
        for child in value:
            _check_json(child)
    else:
        _require(value is None or type(value) in (str, int, float, bool), "non-JSON value")
        encoded(value)  # rejects NaN and infinity


def pointer(path):
    return "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in path)


def _fields(value, path=()):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _fields(child, (*path, key))
    else:  # arrays stay atomic; empty object/null/false/zero remain explicit
        yield list(path), deepcopy(value)


def field_group(path):
    keys = list(path)
    # Ignore only actual metadata-container prefixes, not arbitrary extension
    # ancestors. Unknown extensions are retained, never declared unused.
    while keys and keys[0] == "metadata":
        keys.pop(0)
    if not keys:
        return "other"
    if keys[0] in CONTEXT or keys[:1] == ["evidence"] and len(keys) > 1 and keys[1] in CONTEXT:
        return "context"
    if keys[0] in CORE:
        return "core"
    if keys[0] in SOURCE:
        return "source"
    return "other"


def value_state(value):
    if value is None:
        return "explicit_null"
    if value in ("", [], {}) or isinstance(value, str) and not value.strip():
        return "explicit_empty"
    return "recorded"  # includes False and zero; not a validity claim


def make_view(record, *, kind="node", notices=()):
    _require(kind in ("node", "edge") and isinstance(record, dict), "expected node/edge object")
    _check_json(record)
    _require(all(isinstance(n, str) for n in notices), "notices must be strings")
    groups = {key: {"key": key, "label": label, "expanded": expanded, "fields": []}
              for key, label, expanded in GROUPS}
    for order, (path, value) in enumerate(_fields(record)):
        groups[field_group(path)]["fields"].append({"order": order, "path": path,
            "pointer": pointer(path), "value": value, "state": value_state(value)})
    return {"schema": SCHEMA, "kind": kind, "record_sha256": digest(record),
            "notices": list(notices), "groups": list(groups.values()),
            "graph_mutation": False, "scientific_validity_certified": False}


def restore_record(view):
    _require(view["schema"] == SCHEMA and view["kind"] in ("node", "edge"), "unsupported reading view")
    _require(view["graph_mutation"] is False and view["scientific_validity_certified"] is False, "view is not a graph/science acceptance")
    _require(all(isinstance(n, str) for n in view["notices"]), "invalid notices")
    _require(all(type(g["expanded"]) is bool for g in view["groups"]), "expanded must be boolean")
    _require([(g["key"], g["label"], g["expanded"]) for g in view["groups"]] == list(GROUPS), "reading groups changed")
    fields = []
    for group in view["groups"]:
        for item in group["fields"]:
            path = item["path"]
            _require(isinstance(path, list) and all(isinstance(k, str) for k in path), "invalid JSON path")
            _require(type(item["order"]) is int and item["order"] >= 0, "invalid field order")
            _require(item["pointer"] == pointer(path) and group["key"] == field_group(path), "path/group changed")
            _check_json(item["value"])
            _require(item["state"] == value_state(item["value"]), "value state changed")
            fields.append(item)
    _require(sorted(f["order"] for f in fields) == list(range(len(fields))) and fields, "missing/duplicate field order")
    result, terminals = {}, set()
    for item in sorted(fields, key=lambda x: x["order"]):
        path = tuple(item["path"])
        _require(path not in terminals, "duplicate JSON path")
        _require(not any(path[:i] in terminals for i in range(len(path))), "overlapping JSON path")
        _require(not any(p[:len(path)] == path for p in terminals), "overlapping JSON container")
        terminals.add(path)
        if not path:
            _require(len(fields) == 1 and item["value"] == {}, "invalid root object")
            result = {}
            continue
        parent = result
        for part in path[:-1]:
            parent = parent.setdefault(part, {})
            _require(isinstance(parent, dict), "path traverses a value")
        _require(path[-1] not in parent, "duplicate terminal")
        parent[path[-1]] = deepcopy(item["value"])
    _require(digest(result) == view["record_sha256"], "record content/type hash changed")
    return result


def edge_layer(edge):
    relation = edge.get("relation_type")
    return "claim_anchors" if relation == "about" else "mappings" if relation == "maps_to" else "relations"


def iter_visible_edges(edges, *, layers=("relations",)):
    """Display filter only: no aggregation, reordering or endpoint/node edits."""
    chosen = tuple(layers)
    _require(len(set(chosen)) == len(chosen) and set(chosen) <= set(LAYERS), "unknown/duplicate edge layer")
    for edge in edges:
        if edge_layer(edge) in chosen:
            yield deepcopy(edge)


def render_record(view):
    record = restore_record(view)
    identity = record.get("id") if view["kind"] == "node" else " → ".join(str(record.get(k, "<absent>")) for k in ("source_id", "relation_type", "target_id"))
    warnings = "".join(f'<p class="warning">{escape(n)}</p>' for n in view["notices"])
    parts = []
    for group in view["groups"]:
        rows = "".join(f'<tr><td><code>{escape(f["pointer"] or "/")}</code></td><td><pre>{escape(json.dumps(f["value"], ensure_ascii=False, indent=2, allow_nan=False))}</pre><small>{escape(f["state"])}</small></td></tr>' for f in group["fields"])
        attr = " open" if group["expanded"] else ""
        parts.append(f'<details{attr}><summary>{escape(group["label"])} · {len(group["fields"])} 项</summary><table>{rows}</table></details>')
    raw = escape(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return f'<article><h3>{escape(str(identity))}</h3>{warnings}{"".join(parts)}<details><summary>完整原记录 JSON（未删字段）</summary><pre>{raw}</pre></details></article>'
