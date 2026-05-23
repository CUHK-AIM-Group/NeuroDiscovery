"""Path specificity filter — flags hypotheses whose nodes are too generic.

Diagnoses three contamination modes observed in the Phase 4.3 dry-run that
make ``surprise_gap = 1.0`` look meaningful when it is actually vacuous:

  1. Hub umbrella terms (e.g. "Nervous System Diseases", "Cerebrum") whose
     ComplEx embedding is a generic average vector that scores high against
     anything; PubMed never literally co-mentions them with specific concepts.
  2. CLM_CONCEPT noise nodes the claim_extractor pulled out of free text
     without ontology grounding ("250 healthy controls",
     "much heterogeneity among studies", "transport").
  3. VS:* visual-stimulus nodes leaking outside visual-decoding paths
     ("VS:coco:person" appearing in an AD-imaging hypothesis).

Each rule contributes one issue per offending node. The score is
``1 - n_problems / n_unique_nodes`` so a clean two-hop path scores 1.0 and a
path with two contaminated nodes out of three scores 0.33.

The function takes ``concepts`` (the KG ``concepts`` dict) and ``degrees``
(node-id → degree) so callers in tests can pass tiny stubs without loading
the full KG. Both are currently optional context for future rules; the four
active rules above run with just the IDs and on-link names.
"""

from __future__ import annotations

import re
from typing import Iterable


# ── explicit hub blacklist ─────────────────────────────────────────────
# MSH IDs whose name is an umbrella term that aggregates dozens of specific
# diseases / brain regions. The KG keeps them because some claims literally
# reference them, but they're useless as path endpoints — anything connects
# to anything via "Nervous System Diseases".
HUB_BLACKLIST_IDS = frozenset({
    "MSH:D009422",  # Nervous System Diseases
    "MSH:D001523",  # Mental Disorders
    "MSH:D001927",  # Brain Diseases
    "MSH:D003704",  # Dementia (umbrella; specific dementias are kept)
    "MSH:D019636",  # Neurodegenerative Diseases
    "MSH:D009069",  # Movement Disorders
    "MSH:D019964",  # Mood Disorders
    "MSH:D019954",  # Neurobehavioral Manifestations
    "MSH:D003072",  # Cognition Disorders (umbrella)
    "MSH:D054022",  # Cerebrum
    "MSH:D001921",  # Brain
    "MSH:D009457",  # Neuroglia
    "MSH:D013134",  # Stem Cells (umbrella)
})

# Names that are themselves umbrella regardless of the specific MSH ID.
# Cheap fallback for nodes added later that we don't have IDs for.
GENERIC_NODE_NAMES = frozenset({
    "Nervous System Diseases",
    "Mental Disorders",
    "Brain Diseases",
    "Mood Disorders",
    "Cognition Disorders",
    "Movement Disorders",
    "Neurodegenerative Diseases",
    "Neurobehavioral Manifestations",
    "Cerebrum",
    "Brain",
    "Neuroglia",
})

# Local-name patterns inside the CLM_CONCEPT namespace that mark methodology
# snippets, study-cohort descriptions, or quantifier phrases the LLM picked up
# instead of biological entities. Anchored / boundary-aware regexes.
_VAGUE_CLM_PATTERNS = [
    re.compile(r"^\d"),                                  # "250_healthy_controls", "5_year_follow_up"
    re.compile(r"(?:^|_)controls?(?:_|$)", re.I),        # "...controls..."
    re.compile(r"(?:^|_)patients?(?:_|$)", re.I),
    re.compile(r"(?:^|_)studies?(?:_|$)", re.I),
    re.compile(r"(?:^|_)factors?(?:_|$)", re.I),
    re.compile(r"(?:^|_)label(?:s)?(?:_|$)", re.I),
    re.compile(r"^interventions?$", re.I),
    re.compile(r"^heterogeneity", re.I),
    re.compile(r"^much_", re.I),
    re.compile(r"^various_", re.I),
    re.compile(r"^individual_", re.I),
    re.compile(r"^class_", re.I),
    re.compile(r"^all_", re.I),
    re.compile(r"^transport$", re.I),                    # too abstract w/o substrate
    re.compile(r"^passive_diffusion$", re.I),
    re.compile(r"^death_in_", re.I),
]


def _node_issue(
    nid: str,
    name: str,
    domain_tags: list[str] | None,
    degree: int,
    task_kind: str,
) -> str | None:
    """Return a short issue label for this node, or None if it's clean."""

    # Rule 1: explicit hub blacklist by ID.
    if nid in HUB_BLACKLIST_IDS:
        return f"hub_blacklist:{name}"

    # Rule 2: generic name (catches MSH IDs we didn't list, future imports).
    if name in GENERIC_NODE_NAMES:
        return f"generic_name:{name}"

    # Rule 3: CLM_CONCEPT methodology / cohort / quantifier phrases.
    if nid.startswith("CLM_CONCEPT:"):
        local = nid.split(":", 1)[1]
        for pat in _VAGUE_CLM_PATTERNS:
            if pat.search(local):
                return f"vague_clm:{local}"

    # Rule 4: VS:* visual stimulus nodes leaking outside visual decoding.
    if nid.startswith("VS:"):
        if "visual" not in task_kind.lower():
            return f"misplaced_visual_stimulus:{name}"

    return None


def path_specificity(
    hypothesis,
    concepts: dict | None = None,
    degrees: dict[str, int] | None = None,
) -> tuple[float, list[str]]:
    """Return (specificity_score in [0,1], list of issue strings).

    A score of 1.0 means every node in the path passed all rules. A score
    of 0.0 means every unique node was flagged.

    ``concepts`` and ``degrees`` are accepted but not currently consumed by
    the four active rules — the heuristics fire from the link's ``from_id``,
    ``to_id``, and on-link names plus the hypothesis ``metadata.task_kind``
    for the visual-stimulus rule. Both args remain in the signature so a
    future degree- or ontology-based rule can plug in without touching
    callers.
    """
    concepts = concepts or {}
    degrees = degrees or {}
    task_kind = ""
    meta = getattr(hypothesis, "metadata", None) or {}
    if isinstance(meta, dict):
        task_kind = str(meta.get("task_kind") or meta.get("task_signature") or "")

    seen: set[str] = set()
    issues: list[str] = []
    for link in getattr(hypothesis, "path", []):
        for nid, name in (
            (link.from_id, link.from_name),
            (link.to_id, link.to_name),
        ):
            if not nid or nid in seen:
                continue
            seen.add(nid)
            node = concepts.get(nid) or {}
            tags = node.get("domain_tags") if isinstance(node, dict) else None
            issue = _node_issue(
                nid=nid,
                name=name or node.get("name", "") if isinstance(node, dict) else (name or ""),
                domain_tags=tags,
                degree=int(degrees.get(nid, 0)),
                task_kind=task_kind,
            )
            if issue:
                issues.append(issue)

    if not seen:
        return 1.0, []
    score = 1.0 - len(issues) / len(seen)
    return max(0.0, score), issues


def build_degree_map(edges: Iterable[dict]) -> dict[str, int]:
    """Helper for callers that have raw KG edges and want a degree map."""
    deg: dict[str, int] = {}
    for e in edges:
        s, t = e.get("source_id"), e.get("target_id")
        if not (s and t) or s == t:
            continue
        deg[s] = deg.get(s, 0) + 1
        deg[t] = deg.get(t, 0) + 1
    return deg
