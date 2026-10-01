"""Source-bound validation contracts; no graph, model, or provider access."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
from collections import defaultdict


class ValidationError(ValueError):
    pass


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).casefold().split())


def literal_mentions(text: str, aliases: dict[str, list[str]]) -> dict:
    """Return recall candidates, never a semantic identity decision."""
    normalized = normalized_text(text)
    owners = defaultdict(set)
    for entity_id, labels in aliases.items():
        for label in labels:
            alias = normalized_text(label)
            if alias:
                owners[alias].add(entity_id)
    matches = []
    for alias, entity_ids in sorted(owners.items()):
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized):
            matches.append({
                "alias": alias, "entity_ids": sorted(entity_ids),
                "normalized_span": [match.start(), match.end()],
                "identity_status": "ambiguous" if len(entity_ids) > 1 else "pending_review",
            })
    return {"matches": matches, "automatic_entity_resolution": False}


def _required_text(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"missing {field}")
    return value


def _source_ids(row: dict, field: str) -> set[str]:
    values = row.get(field)
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValidationError(f"missing {field}")
    return set(values)


def check_item(row: dict, sources: dict, protected_works: set[str]) -> None:
    """Validate bindings and explicit decisions, not the truth of those decisions."""
    for field in ("item_id", "source_id", "relation", "target_id", "fact_family",
                  "relation_family", "measurement", "role", "rationale"):
        _required_text(row, field)
    if not isinstance(row.get("scope"), dict) or not row["scope"]:
        raise ValidationError("missing explicit scope")
    if row.get("mapping_status") != "reviewed_exact":
        raise ValidationError("entity mapping not reviewed")
    works = _source_ids(row, "work_keys")
    families = _source_ids(row, "source_families")
    if works & protected_works:
        raise ValidationError("protected heldout work")
    label = row.get("label")
    if label not in {"supported", "contradicted", "unresolved"}:
        raise ValidationError("unknown label")
    if label == "contradicted" and row.get("negative_basis") != "explicit_source_contradiction":
        raise ValidationError("absence, reversal or null is not a negative label")
    if label != "unresolved" and row.get("scope_match") is not True:
        raise ValidationError("measurement/scope not verified")
    anchors = row.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValidationError("missing source anchors")
    anchored_works = set()
    anchored_families = set()
    for anchor in anchors:
        source_hash = anchor.get("source_sha256")
        source = sources.get(source_hash)
        if not isinstance(source, dict):
            raise ValidationError("source snapshot missing")
        if source.get("fold") != "corpus":
            raise ValidationError("non-corpus source")
        source_work = _required_text(source, "work_key")
        source_family = _required_text(source, "source_family")
        if source_work in protected_works or source_work not in works or source_family not in families:
            raise ValidationError("source identity mismatch or protected work")
        text = _required_text(source, "text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != source_hash:
            raise ValidationError("source snapshot hash mismatch")
        start, end = anchor.get("start"), anchor.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValidationError("invalid raw-text offsets")
        quote = _required_text(anchor, "quote")
        if text[start:end] != quote:
            raise ValidationError("literal anchor mismatch")
        if label == "contradicted" and anchor.get("stance") != "contradicts_exact_scoped_claim":
            raise ValidationError("negative needs an exact scoped contradiction")
        anchored_works.add(source_work)
        anchored_families.add(source_family)
    if anchored_works != works or anchored_families != families:
        raise ValidationError("unanchored work or source family")


def proxy_keys(row: dict) -> set[tuple]:
    """Conservative: block every relation between held-out endpoint pairs."""
    source, target = sorted((_required_text(row, "source_id"), _required_text(row, "target_id")))
    return {("endpoints", source, target), ("fact", _required_text(row, "fact_family"))}


def family_split(items: list[dict], seed: str) -> dict[str, str]:
    """Connected components prevent cross-work, inverse and duplicate leakage."""
    parent = list(range(len(items)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    ids = set()
    for index, row in enumerate(items):
        item_id = _required_text(row, "item_id")
        if item_id in ids:
            raise ValidationError("duplicate item id")
        ids.add(item_id)
        keys = proxy_keys(row)
        keys |= {("work", value) for value in _source_ids(row, "work_keys")}
        keys |= {("family", value) for value in _source_ids(row, "source_families")}
        for key in keys:
            if key in seen:
                parent[root(index)] = root(seen[key])
            seen[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(items):
        groups[root(index)].append(row["item_id"])
    assignments = {}
    for members in groups.values():
        draw = int(fingerprint([seed, sorted(members)])[:16], 16) / 2**64
        fold = "train" if draw < 0.6 else "validation" if draw < 0.8 else "test"
        assignments.update({item_id: fold for item_id in members})
    return assignments


def filter_training_graph(edges: list[dict], heldout: list[dict], protected_works: set[str]) -> dict:
    blocked_keys = set().union(*(proxy_keys(row) for row in heldout)) if heldout else set()
    blocked_works = set(protected_works)
    blocked_families = set()
    for row in heldout:
        blocked_works.update(_source_ids(row, "work_keys"))
        blocked_families.update(_source_ids(row, "source_families"))
    kept, excluded = [], []
    for edge in edges:
        try:
            works = _source_ids(edge, "work_keys")
            families = _source_ids(edge, "source_families")
            keys = proxy_keys(edge)
            _required_text(edge, "relation")
            feature_works = edge.get("feature_work_keys")
            if not isinstance(feature_works, list) or any(
                not isinstance(work, str) or not work for work in feature_works
            ) or edge.get("feature_provenance_complete") is not True:
                raise ValidationError("unknown feature provenance")
            if edge.get("provenance_verified") is not True:
                raise ValidationError("unknown edge provenance")
            if keys & blocked_keys or (works | set(feature_works)) & blocked_works or families & blocked_families:
                raise ValidationError("heldout family, fact, endpoint proxy or feature")
        except ValidationError as exc:
            excluded.append({"edge": edge, "reason": str(exc)})
        else:
            kept.append(edge)
    return {"edges": kept, "excluded": excluded}


def direction_summary(pairs: list[tuple[float, float]], expected: int) -> dict:
    if type(expected) is not int or expected < len(pairs):
        raise ValidationError("scored count exceeds expected count")
    wins = ties = losses = 0
    for forward, reverse in pairs:
        if not math.isfinite(forward) or not math.isfinite(reverse):
            raise ValidationError("nonfinite model score")
        if math.isclose(forward, reverse, rel_tol=1e-4, abs_tol=1e-6):
            ties += 1
        elif forward > reverse:
            wins += 1
        else:
            losses += 1
    return {"scored": len(pairs), "unmeasured": expected - len(pairs),
            "wins": wins, "ties": ties, "losses": losses,
            "strict_win_rate": wins / len(pairs) if pairs else None,
            "tie_adjusted_preference": (wins + ties / 2) / len(pairs) if pairs else None,
            "scientific_accuracy": None}


def validate_path(hops: list[dict]) -> None:
    if len(hops) < 2:
        raise ValidationError("at least two hops required")
    vertices = [_required_text(hops[0], "source_id")]
    for hop in hops:
        if _required_text(hop, "source_id") != vertices[-1]:
            raise ValidationError("disconnected ordered path")
        _required_text(hop, "relation")
        vertices.append(_required_text(hop, "target_id"))
    if len(set(vertices)) != len(vertices):
        raise ValidationError("cycle or inverse round trip is not a new endpoint path")


def preflight(items: list[dict], sources: dict, assignments: dict, edges: list[dict],
              protected_works: set[str], exposed_works: set[str], *,
              require_contradictions: bool = True) -> dict:
    if not items or set(assignments) != {row.get("item_id") for row in items}:
        raise ValidationError("exact nonempty item/split binding required")
    if len(items) != len(assignments):
        raise ValidationError("duplicate item id")
    if set(assignments.values()) != {"train", "validation", "test"}:
        raise ValidationError("all three folds required; never move test items into train")
    seen = {}
    for row in items:
        check_item(row, sources, protected_works)
        fold = assignments[row["item_id"]]
        works = _source_ids(row, "work_keys")
        if fold != "train" and (works & exposed_works or row.get("previously_exposed") is not False):
            raise ValidationError("previously exposed or unknown-exposure validation/test item")
        keys = proxy_keys(row)
        keys |= {("work", value) for value in works}
        keys |= {("family", value) for value in row["source_families"]}
        for key in keys:
            if key in seen and seen[key] != fold:
                raise ValidationError("cross-fold source family or proxy leakage")
            seen[key] = fold
    for fold in ("validation", "test"):
        labels = {row["label"] for row in items if assignments[row["item_id"]] == fold}
        required = {"supported", "contradicted", "unresolved"} if require_contradictions else {"supported"}
        if not required <= labels:
            raise ValidationError(f"{fold} lacks required labels: {sorted(required)}")
    heldout = [row for row in items if assignments[row["item_id"]] != "train"]
    filtered = filter_training_graph(edges, heldout, protected_works)
    if filtered["excluded"] or not filtered["edges"]:
        raise ValidationError("message graph contains unbound/leaking edges or is empty")
    return {"status": "STRUCTURAL_CHECKS_ONLY", "binding_sha256": fingerprint({
        "items": items, "sources": sources, "assignments": assignments, "edges": edges,
        "protected_works": sorted(protected_works), "exposed_works": sorted(exposed_works),
        "require_contradictions": require_contradictions,
    }), "scientific_acceptance": False, "training_authorized": False}


class ValidationStopper:
    """Select a best checkpoint using validation only; never training or test loss."""

    def __init__(self, patience: int = 5):
        if type(patience) is not int or patience < 1:
            raise ValueError("positive patience required")
        self.patience = patience
        self.best = -math.inf
        self.stale = 0
        self.checkpoint = None

    def observe(self, metric: float, checkpoint: bytes, *, split: str) -> bool:
        if split != "validation" or not math.isfinite(metric) or not isinstance(checkpoint, bytes):
            raise ValidationError("finite validation metric and immutable checkpoint required")
        if metric > self.best:
            self.best, self.stale, self.checkpoint = metric, 0, checkpoint
        else:
            self.stale += 1
        return self.stale >= self.patience
