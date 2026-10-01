"""Cross-paper proposition identity for the rebuilt paper knowledge layer.

The retired claim store recorded a proposition as a free-text triple that kept
the paper's own wording, so two papers stating the same finding almost never
produced the same claim row (measured: 99.74% of 905,274 claims resolve to a
single work). This module supplies the layer that was missing: a stable
proposition identity that deliberately excludes paper, bibliography, raw text
and statistics, plus an append-only, reversible decision log.

Nothing here writes the graph, reads the fixed layer, calls a model or
authorizes publication. It performs no science by itself: a proposition key is
an identity, and every reuse/merge decision still requires an adjudicated
judgement with a source-bound reason.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

VERSION = "kg.shared_proposition_registry.v1"

# Identity fields. A proposition key is built only from these. Paper id, title,
# authors, year, journal, sample size, effect size, p value, confidence and the
# supporting sentence are excluded by construction so that a second paper can
# reach the same proposition.
IDENTITY_FIELDS = (
    "subject",
    "relation",
    "object",
    "polarity",
    "qualifiers",
)

# Fields that must never enter a key even if a caller supplies them.
FORBIDDEN_KEY_FIELDS = frozenset({
    "paper", "paper_id", "pmid", "doi", "pmcid", "arxiv_id", "openalex_id",
    "title", "authors", "year", "journal", "publication_year", "batch",
    "sample_size", "effect_size", "effect_metric", "p_value", "ci", "direction",
    "confidence", "raw_text", "raw_sentence", "sentence", "quote", "claim_id",
    "observation_id", "evidence_link_id", "extraction_timestamp",
})

POLARITIES = frozenset({"asserted", "negated"})

# Controlled relation vocabulary. Unknown relations are preserved as declared
# but are not silently folded into another family.
RELATION_FAMILIES = {
    "is_associated_with": "association",
    "associated_with": "association",
    "correlates_with": "association",
    "correlated_with": "association",
    "is_correlated_with": "association",
    "is_biomarker_of": "prediction",
    "biomarker_of": "prediction",
    "predicts": "prediction",
    "is_predictive_of": "prediction",
    "distinguishes": "discrimination",
    "differentiates": "discrimination",
    "discriminates": "discrimination",
    "increases": "increase",
    "reduces": "decrease",
    "modulates": "modulation",
    "regulates": "modulation",
    "mediates": "mediation",
    "causes": "causation",
    "treats": "treatment",
    "is_treatment_for": "treatment",
    "is_risk_factor_for": "risk",
    "has_adverse_effect": "adverse",
    "inhibits": "inhibition",
    "activates": "activation",
}

# Qualifier slots that materially change what a proposition asserts. A
# difference in any listed slot is a scope difference, not a wording difference.
QUALIFIER_SLOTS = (
    "species",
    "population",
    "disease_stage",
    "anatomy",
    "modality",
    "measurement",
    "task",
    "intervention",
    "comparator",
    "dose",
    "timepoint",
    "adjustment",
)

DECISIONS = frozenset({"reuse", "create", "related", "unresolved"})
RELATIONS = frozenset({
    "equivalent", "narrower_than", "broader_than", "related_to", "distinct",
    "unresolved",
})
EVIDENCE_ROLES = frozenset({"supports", "opposes", "partial", "contextual", "unresolved"})


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_term(value):
    """Case/whitespace/punctuation normalization only.

    This is deliberately shallow. Measured effect of surface normalization alone
    was negligible (0.11% -> 0.12% multi-work), so it must never be presented as
    semantic merging.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    for char in "_-/":
        text = text.replace(char, " ")
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())


def normalize_qualifiers(qualifiers=None):
    """Return a canonical, ordered qualifier mapping.

    Absent slots are omitted, not filled with a default, so "unknown" cannot be
    mistaken for "same as the other paper".
    """
    source = dict(qualifiers or {})
    unknown = set(source) - set(QUALIFIER_SLOTS)
    require(not unknown, "unknown qualifier slots: %s" % sorted(unknown))
    result = {}
    for slot in QUALIFIER_SLOTS:
        if slot not in source:
            continue
        raw = source[slot]
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            values = sorted({normalize_term(item) for item in raw} - {""})
            if values:
                result[slot] = values
            continue
        text = normalize_term(raw)
        if text:
            result[slot] = text
    return result


def normalize_relation(value):
    """Fold separators to underscores so relation synonyms actually match.

    Relation names are identifiers, not free text: "correlates_with" and
    "correlates with" must reach the same family.
    """
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace("/", "_")
    text = "".join(ch if (ch.isalnum() or ch == "_") else " " for ch in text)
    return "_".join(text.split())


def relation_family(relation):
    text = normalize_relation(relation)
    require(text, "proposition needs a relation")
    return RELATION_FAMILIES.get(text, text)


def work_identity(value):
    """Work identifiers are opaque and must not be punctuation-normalized."""
    if value is None:
        return ""
    return str(value).strip()


def proposition_identity(subject, relation, object_, *, polarity="asserted", qualifiers=None):
    """Build the identity payload for one proposition.

    Endpoints must already be canonical references (for example an existing
    concept id). Passing a raw sentence is rejected so the old failure cannot
    recur: paper wording must never be the identity of a shared proposition.
    """
    subject_term = normalize_term(subject)
    object_term = normalize_term(object_)
    require(subject_term, "proposition needs a subject")
    require(object_term, "proposition needs an object")
    require(polarity in POLARITIES, "polarity must be one of %s" % sorted(POLARITIES))
    identity = {
        "subject": subject_term,
        "relation": relation_family(relation),
        "object": object_term,
        "polarity": polarity,
        "qualifiers": normalize_qualifiers(qualifiers),
    }
    return identity


def proposition_key(subject, relation, object_, *, polarity="asserted", qualifiers=None):
    return digest(proposition_identity(subject, relation, object_, polarity=polarity, qualifiers=qualifiers))


def recall_keys(subject, relation, object_, *, polarity="asserted"):
    """Blocking keys used to fetch candidate propositions for adjudication.

    Recall is intentionally generous: it proposes candidates, it never decides
    equivalence. Candidate retrieval is measured separately from final merging.
    """
    identity = proposition_identity(subject, relation, object_, polarity=polarity)
    return {
        "subject_relation": digest([identity["subject"], identity["relation"]]),
        "relation_object": digest([identity["relation"], identity["object"]]),
        "subject_object": digest([identity["subject"], identity["object"]]),
        "relation_only": digest([identity["relation"]]),
    }


def assert_no_forbidden_fields(payload):
    """Guard used by callers that assemble identity payloads dynamically."""
    present = sorted(FORBIDDEN_KEY_FIELDS & set(payload))
    require(not present, "identity payload must not contain paper-bound fields: %s" % present)
    return True


class SharedPropositionRegistry:
    """In-memory registry with an append-only, reversible decision log.

    Persistence and graph publication are separate concerns; this type is the
    identity authority the pipeline consults before creating a new proposition.
    """

    def __init__(self):
        self._propositions = {}
        self._by_key = {}
        self._members = defaultdict(set)          # proposition_key -> {work key}
        self._evidence = defaultdict(lambda: defaultdict(set))
        self._log = []
        self._merged_into = {}                    # retired key -> surviving key

    # ---- creation and reuse -------------------------------------------------
    def register(self, subject, relation, object_, *, polarity="asserted", qualifiers=None,
                 statement=None, decision="create", related_to=None, reason=""):
        """Create or reuse a proposition, recording how the decision was reached."""
        require(decision in DECISIONS, "decision must be one of %s" % sorted(DECISIONS))
        require(reason, "every reuse/create decision needs a source-bound reason")
        identity = proposition_identity(subject, relation, object_, polarity=polarity, qualifiers=qualifiers)
        key = digest(identity)
        existing = self._by_key.get(key)
        if existing is not None:
            self._record("reuse", key, reason, {"requested_decision": decision})
            return key, False
        require(decision != "reuse", "cannot reuse a proposition that does not exist yet")
        self._propositions[key] = {
            "key": key,
            "identity": identity,
            "statement": statement,
            "related_to": list(related_to or []),
        }
        self._by_key[key] = identity
        self._record("create", key, reason, {"decision": decision})
        return key, True

    def candidates(self, subject, relation, object_, *, polarity="asserted"):
        """Return propositions sharing any blocking key, for adjudication."""
        wanted = set(recall_keys(subject, relation, object_, polarity=polarity).values())
        found = set()
        for key, identity in self._by_key.items():
            keys = set(
                recall_keys(
                    identity["subject"], identity["relation"], identity["object"],
                    polarity=identity["polarity"],
                ).values()
            )
            if keys & wanted:
                found.add(key)
        return sorted(found)

    # ---- evidence -----------------------------------------------------------
    def add_evidence(self, key, work, role, *, reason=""):
        """Attach one work to one proposition, counted once per work."""
        require(role in EVIDENCE_ROLES, "role must be one of %s" % sorted(EVIDENCE_ROLES))
        require(work, "evidence needs a work identity")
        # Evidence may be attached through a retired key after a merge; resolve
        # to the surviving proposition instead of failing or orphaning it.
        key = self.resolve(key)
        require(key in self._propositions, "unknown proposition key")
        work = work_identity(work)
        self._members[key].add(work)
        self._evidence[key][role].add(work)
        self._record("evidence", key, reason or "evidence attached", {"work": work, "role": role})

    def resolve(self, key):
        """Follow merge links to the proposition that currently owns the key."""
        seen = set()
        while key in self._merged_into:
            require(key not in seen, "cyclic merge link at %s" % key)
            seen.add(key)
            key = self._merged_into[key]
        return key

    def works(self, key):
        return sorted(self._members[self.resolve(key)])

    def multipaper_keys(self):
        return sorted(k for k, works in self._members.items() if len(works) >= 2)

    # ---- reversible restructuring -------------------------------------------
    def merge(self, source_key, target_key, *, reason):
        """Collapse two propositions already adjudicated as equivalent."""
        require(reason, "merge needs a reason")
        require(source_key in self._propositions and target_key in self._propositions, "unknown proposition key")
        require(source_key != target_key, "cannot merge a proposition into itself")
        for work in self._members.pop(source_key, set()):
            self._members[target_key].add(work)
        for role, works in self._evidence.pop(source_key, {}).items():
            self._evidence[target_key][role].update(works)
        self._merged_into[source_key] = target_key
        self._record("merge", target_key, reason, {"source": source_key})

    def split(self, key, *, reason):
        """Reverse a merge; the retired key becomes an independent proposition again."""
        require(reason, "split needs a reason")
        require(key in self._merged_into, "proposition is not merged")
        target = self._merged_into.pop(key)
        self._propositions[key] = {
            "key": key,
            "identity": self._by_key[key],
            "statement": None,
            "related_to": [],
        }
        self._members.setdefault(key, set())
        self._evidence.setdefault(key, defaultdict(set))
        self._record("split", key, reason, {"restored_from": target})

    def relate(self, key, other_key, relation, *, reason):
        require(relation in RELATIONS, "relation must be one of %s" % sorted(RELATIONS))
        require(reason, "relation needs a reason")
        require(key in self._propositions and other_key in self._propositions, "unknown proposition key")
        entry = {"other": other_key, "relation": relation, "reason": reason}
        self._propositions[key]["related_to"].append(entry)
        self._record("relate", key, reason, entry)

    # ---- audit --------------------------------------------------------------
    def _record(self, action, key, reason, extra=None):
        self._log.append({"seq": len(self._log), "action": action, "key": key, "reason": reason, "extra": extra or {}})

    @property
    def log(self):
        return list(self._log)

    def counts(self):
        works_per_key = [len(v) for v in self._members.values() if v]
        multi = len(self.multipaper_keys())
        active = [k for k in self._propositions if k not in self._merged_into]
        return {
            "propositions": len(active),
            "merged_away": len(self._merged_into),
            "with_evidence": len(works_per_key),
            "multipaper": multi,
            "single_work": len([n for n in works_per_key if n == 1]),
            "works_histogram": dict(sorted(Counter(works_per_key).items())),
        }
