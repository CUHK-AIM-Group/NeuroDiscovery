"""Render prompt shape text from the schema, and detect drift.

Consolidation step 3 of docs/KG_PIPELINE_CONSOLIDATION_PLAN_20260927.md.

The defect this addresses: field names, required keys and enums were written out
longhand inside several prompt strings, so a prompt and the validator that judged
its output could disagree and nothing would notice. The schema file is now the
only place those facts live, the shape block is rendered from it, and the checks
here report drift in both directions without repairing anything.

The schema does NOT own reasoning guidance. Prose about what not to infer, and
the contract's harder rules, stay in the prompt text because that is where a
model reads them.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "candidate.schema.json"
SHAPE_MARKER = "structured_result shape:"
WIRE_VALUE_TOKEN = re.compile(r"source_result_v\d+")
QUOTED_TOKEN = re.compile(r'"([a-z][a-z0-9_]{2,})"')
BARE_VALUE = re.compile(r"(?<=:)(binding_or_null|binding|slot)(?=[,\s}])")

# Envelopes that shape a whole artifact, so a prompt is expected to state them.
CHECKED_REQUIRED = ("observation", "structured_result", "root")


def load_schema(path=None):
    return json.loads((Path(path) if path else SCHEMA_PATH).read_text(encoding="utf-8"))


def walk_properties(node, found):
    """Collect every property name declared anywhere in the schema.

    Only ``properties`` keys are collected, never the keys of ``definitions``.
    Subtracting definition names would delete genuine properties that share a
    name with a definition (``study``, ``result``, ``conditions`` ...).
    """
    if isinstance(node, dict):
        for key, value in (node.get("properties") or {}).items():
            found.add(key)
            walk_properties(value, found)
        for value in node.values():
            walk_properties(value, found)
    elif isinstance(node, list):
        for item in node:
            walk_properties(item, found)
    return found


def schema_fields(schema=None):
    """Every declared property name."""
    return walk_properties(schema or load_schema(), set())


def enums(schema=None):
    """Map dotted path -> allowed values for every enum in the schema."""
    schema = schema or load_schema()
    found = {}

    def visit(node, path):
        if isinstance(node, dict):
            if "enum" in node:
                found[path or "root"] = list(node["enum"])
            for key, value in (node.get("properties") or {}).items():
                visit(value, "%s.%s" % (path, key) if path else key)
            for key, value in (node.get("definitions") or {}).items():
                visit(value, key)
        elif isinstance(node, list):
            for item in node:
                visit(item, path)

    visit(schema, "")
    return found


def required_fields(schema=None):
    """Map envelope/definition name -> required keys."""
    schema = schema or load_schema()
    found = {"root": list(schema.get("required") or [])}
    for name, definition in (schema.get("definitions") or {}).items():
        if isinstance(definition, dict) and definition.get("required"):
            found[name] = list(definition["required"])
    return found


def wire_schema_const(schema=None):
    """The one frozen wire value, read from the schema that declares it."""
    schema = schema or load_schema()
    return schema["definitions"]["structured_result"]["properties"]["schema"]["const"]


def render_shape(schema=None):
    """One compact shape line per object definition, from the schema alone.

    Deterministic, so it can be hashed and compared across runs. Optional keys
    are marked ``?`` rather than dropped: a shape that hides an optional field
    would understate what the extractor may return. The definition order comes
    from the schema, so a new definition cannot be forgotten here.
    """
    schema = schema or load_schema()
    lines = ["JSON shape (keys marked ? are optional; null or [] for missing; no extra commentary):",
             "{" + ", ".join('"%s"' % key for key in schema["required"]) + "}"]
    for name, definition in (schema.get("definitions") or {}).items():
        if not isinstance(definition, dict) or not definition.get("properties"):
            continue
        required = set(definition.get("required") or [])
        parts = []
        for key, value in definition["properties"].items():
            allowed = value.get("enum") if isinstance(value, dict) else None
            rendered = ("|".join("null" if item is None else str(item) for item in allowed)
                        if allowed else "null")
            mark = "?" if required and key not in required else ""
            parts.append('"%s"%s:%s' % (key, mark, rendered))
        lines.append("%s: {%s}" % (name, ", ".join(parts)))
    return "\n".join(lines)


def shape_sha(schema=None):
    return hashlib.sha256(render_shape(schema).encode("utf-8")).hexdigest()


def prompt_drift(prompt_text, schema=None):
    """Where a prompt disagrees with the schema it is supposed to describe.

    Reports a required key the prompt never states as an exact JSON key, and a
    wire value it names that the schema has superseded. Read-only observation.
    """
    schema = schema or load_schema()
    declared = schema_fields(schema)
    problems = []

    stated = any('"%s"' % name in prompt_text for name in declared)
    for name in CHECKED_REQUIRED:
        keys = required_fields(schema).get(name) or []
        missing = [key for key in keys if '"%s"' % key not in prompt_text]
        if missing and stated:
            problems.append({"kind": "prompt_omits_required_keys",
                             "definition": name, "keys": missing})

    declared_wire = wire_schema_const(schema)
    for value in sorted(set(WIRE_VALUE_TOKEN.findall(prompt_text))):
        if value != declared_wire:
            problems.append({"kind": "prompt_mentions_superseded_wire_value",
                             "value": value, "declared": declared_wire})
    return problems


def fields_without_json_key(prompt_text, schema=None):
    """Declared fields whose exact JSON key form (``"name"``) never appears.

    Deliberately strict: a field named only in prose counts as missing, because
    what a model must emit is the quoted key. The live prompt currently misses
    exactly one, ``structured_result``, which its amendment names in prose.
    """
    return sorted(name for name in schema_fields(schema) if '"%s"' % name not in prompt_text)


def undefined_field_mentions(prompt_text, schema=None):
    """Quoted snake_case tokens no field, enum value or wire value explains.

    The reverse direction: catches a prompt that invents a key the schema never
    declares. Prose in quotes is reported too, so this is a review aid, not a
    pass/fail gate.
    """
    schema = schema or load_schema()
    known = schema_fields(schema) | {str(value) for values in enums(schema).values() for value in values}
    known.add(wire_schema_const(schema))
    return sorted(set(QUOTED_TOKEN.findall(prompt_text)) - known)


def _first_json_object(text, marker):
    """Parse the first balanced JSON object after ``marker``, or None."""
    if marker not in text:
        return None
    body = text[text.index(marker):]
    start = body.index("{")
    depth = 0
    for offset in range(start, len(body)):
        if body[offset] == "{":
            depth += 1
        elif body[offset] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(BARE_VALUE.sub(r'"__\1__"', body[start:offset + 1]))
    return None


def shape_delta(prompt_text, schema=None, definition="structured_result", marker=SHAPE_MARKER):
    """How a prompt's hand-written shape block differs from the schema.

    None means the prompt carries no such block; an empty list means it agrees
    exactly at every level. This is the evidence the plan requires before an
    embedded shape block may be replaced by rendered schema text. Deltas are
    reported, never repaired.
    """
    schema = schema or load_schema()
    stated = _first_json_object(prompt_text, marker)
    if stated is None:
        return None
    deltas = []
    _shape_walk(stated, schema["definitions"][definition], definition, schema, deltas)
    return deltas


def shape_block(prompt_text, marker=SHAPE_MARKER):
    """The parsed hand-written shape block of a prompt, or None if it has none.

    Public so read-only reporters do not reach into the private parser; returns
    the nested object, not text, because that is what the comparison needs.
    """
    return _first_json_object(prompt_text, marker)


BASE_SHAPE_MARKER = "JSON shape"


def _resolve(declared, schema):
    """Follow ``$ref`` / ``oneOf`` / ``items`` to the object a shape key states."""
    if not isinstance(declared, dict):
        return None
    if "$ref" in declared:
        name = declared["$ref"].rsplit("/", 1)[-1]
        return schema["definitions"].get(name)
    if "items" in declared:
        return _resolve(declared["items"], schema)
    for option in declared.get("oneOf") or []:
        resolved = _resolve(option, schema)
        if resolved and resolved.get("properties"):
            return resolved
    return declared


def shape_delta_full(prompt_text, schema=None, marker=BASE_SHAPE_MARKER):
    """Compare a prompt's *whole* stated shape block with the schema, recursively.

    ``shape_delta`` only covers the amendment's ``structured_result`` block. The
    three older prompts use a ``JSON shape`` marker, so their blocks were never
    compared at all -- the drift report showed them as "no block". This walks the
    envelope and every nested definition the prompt actually states, so the
    difference list required before deletion covers all four embedded structures.
    Deltas are reported, never repaired. ``None`` means the prompt has no block.
    """
    schema = schema or load_schema()
    stated = _first_json_object(prompt_text, marker)
    if stated is None:
        return None
    deltas = []
    envelope = schema["required"]
    if set(stated) != set(envelope):
        deltas.append({"path": "root", "kind": "keys",
                       "absent_from_prompt": sorted(set(envelope) - set(stated)),
                       "not_declared_by_schema": sorted(set(stated) - set(envelope))})
    properties = schema.get("properties") or {}
    for key, value in stated.items():
        definition = _resolve(properties.get(key), schema)
        if definition and definition.get("properties") and isinstance(value, dict):
            _shape_walk(value, definition, key, schema, deltas)
        elif definition and definition.get("properties") and isinstance(value, list) and value \
                and isinstance(value[0], dict):
            _shape_walk(value[0], definition, key + "[0]", schema, deltas)
    return deltas


def _shape_walk(stated, definition, path, schema, deltas):
    properties = definition.get("properties") or {}
    if set(stated) != set(properties):
        deltas.append({"path": path, "kind": "keys",
                       "absent_from_prompt": sorted(set(properties) - set(stated)),
                       "not_declared_by_schema": sorted(set(stated) - set(properties))})
    required = definition.get("required") or []
    if required and set(stated) != set(required):
        deltas.append({"path": path, "kind": "required",
                       "difference": sorted(set(stated) ^ set(required))})
    for key, value in stated.items():
        declared = properties.get(key)
        if not isinstance(declared, dict):
            continue
        if isinstance(value, dict):
            resolved = _resolve(declared, schema)
            if resolved.get("properties"):
                _shape_walk(value, resolved, "%s.%s" % (path, key), schema, deltas)
        elif "const" in declared:
            if value != declared["const"]:
                deltas.append({"path": "%s.%s" % (path, key), "kind": "const",
                               "difference": [value, declared["const"]]})
        elif isinstance(value, str) and "|" in value and declared.get("enum"):
            expected = {"null" if item is None else str(item) for item in declared["enum"]}
            if set(value.split("|")) != expected:
                deltas.append({"path": "%s.%s" % (path, key), "kind": "enum",
                               "difference": sorted(set(value.split("|")) ^ expected)})
