"""Input Recipe generator: data inventory -> LLM brainstorm -> KG links.

Three stages, kept separate so each is testable in isolation:

1. ``build_inventory(concepts)`` walks the KG and returns a short list per
   "input bucket" (modalities + representative gene/biomarker/score names).
   The point is to give the LLM ENOUGH to know the data shape, not so much
   that it just paraphrases the KG.

2. ``brainstorm_recipes(inventory, n, client, model)`` issues one LLM call
   asking for N computable quantities. The prompt deliberately withholds
   any downstream task framing per spec. ``brainstorm_recipes_batched``
   wraps this in a multi-batch loop with rotating focus and de-duplication
   for large N (>= 100).

3. ``link_to_concepts(recipes, concepts)`` token-matches recipe names back
   to KG concept ids so the recipe nodes can later be edge-anchored.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Buckets the inventory exposes to the LLM. Each entry is
# (domain, display_label, cap, type_hint, prefer_long_names). The type_hint
# is shown to the LLM so it understands what kind of value each bucket holds,
# which prevents nonsense like averaging a brain-region label with a
# diffusion coefficient. ``prefer_long_names`` flips the within-bucket sort:
# when True, multi-word / longer names win the cap (right for disease and
# drug, where short tokens are mostly noisy acronyms like DSM/DTI/FDG/PIB);
# when False, short canonical tokens win (right for genes/biomarkers where
# APOE/BDNF/IL6 are the real names).
_BUCKETS: tuple[tuple[str, str, int, str, bool], ...] = (
    ("modality",               "imaging modalities", 12,
     "imaging acquisition types - each item identifies WHICH raw signal is available, not a scalar value",
     False),
    ("gene",                   "genes / variants", 15,
     "gene or variant labels - quantifiable as allele dosage, mRNA expression, or carrier indicator (NOT directly a number)",
     False),
    ("biomarker",              "biomarkers / lab measures", 15,
     "biomarker labels - quantifiable as concentration / level (each has its own unit; do not combine across biochemical classes without justification)",
     False),
    ("neuroanatomy",           "brain regions", 20,
     "anatomical region labels - quantifiable per modality as volume (mm^3), cortical thickness (mm), activation, or connectivity (NOT a scalar by itself)",
     False),
    ("cognitive_function",     "cognitive functions", 12,
     "cognitive constructs - each is a latent ability, NOT a directly measured number; only meaningful when paired with a behavioural score that operationalises it",
     False),
    ("paradigm",               "task paradigms", 10,
     "experimental paradigms - each yields task-specific behavioural measures (RT, accuracy, error rate); paradigms cannot be added or subtracted from each other",
     False),
    ("dataset_variable",       "clinical / behavioural scores", 12,
     "clinical or behavioural scores - actual scalar values from datasets (CDR, MMSE, BMI, etc.)",
     False),
    ("neurotransmitter",       "neurotransmitter systems", 8,
     "neurotransmitter labels - quantifiable as concentration, receptor density, or release rate",
     False),
    ("visual_stimulus",        "stimulus categories", 8,
     "visual stimulus category labels - quantifiable as mean evoked response (BOLD, EEG amplitude) WHEN paired with an imaging modality",
     False),
    ("individual_data_anchor", "individual-level covariates", 8,
     "subject-level covariates (age, sex, handedness) - usable as stratifiers or interaction terms, not standalone outcomes",
     False),
    ("disease",                "diseases / clinical labels", 15,
     "diagnostic labels - quantifiable as criterion-based indicator (1/0 by DSM/NINCDS/ICD), severity score, time-since-onset, or contrast between two diagnoses",
     True),
    ("drug",                   "drugs / interventions", 12,
     "pharmacological or stimulation interventions - quantifiable as dose (mg, mg/kg/day), cumulative exposure (mg-years), time since first administration, on/off status, or dose change",
     True),
    ("treatment_outcome",      "treatment outcomes", 10,
     "outcome categories used in trials - quantifiable as event indicator, time-to-event, or score change versus baseline",
     True),
)


# Hand-curated long-form expansions for cryptic short tokens that appear in
# the KG with no aliases. Without these the LLM treats 'C3' / 'Pu' / 'SNc'
# / 'Aβ' as algebraic variables and emits formulas like 'C3 + C7' or
# 'Pu / ADC'. Keys are the EXACT preferred_name strings; values are the
# long form shown to the LLM (the original short token is retained inside
# parentheses so inputs_used can still cite it).
_LONGFORM: dict[str, str] = {
    # neuroanatomy - basal ganglia / brainstem / visual cortex shorthand
    "Pu": "Putamen",
    "Ca": "Caudate",
    "GPe": "Globus Pallidus externa",
    "GPi": "Globus Pallidus interna",
    "SNc": "Substantia Nigra pars compacta",
    "SNr": "Substantia Nigra pars reticulata",
    "VeP": "Ventral Pallidum",
    "V3v": "ventral V3 visual area",
    "LC": "Locus Coeruleus",
    "LC_L": "Locus Coeruleus (left)",
    "LC_R": "Locus Coeruleus (right)",
    "HN": "Hypoglossal Nucleus",
    "MN": "Motor Nucleus",
    "RN": "Red Nucleus",
    "BB4": "Brodmann area 4",
    "EXA": "external amygdala",
    "HTH": "Hypothalamus",
    "PBP": "parabrachial pigmented nucleus",
    "STH": "Subthalamic nucleus",
    "VTA_L": "Ventral Tegmental Area (left)",
    "VTA_R": "Ventral Tegmental Area (right)",
    # genes / proteins / Greek-letter biomarkers
    "C3": "complement C3",
    "C7": "complement C7",
    "FH": "complement factor H",
    "GK": "glycerol kinase (GK)",
    "CP": "ceruloplasmin (CP)",
    "AR": "androgen receptor (AR)",
    "Aβ": "amyloid-beta peptide",
    "Aß42": "amyloid-beta 1-42 peptide",
    "PGC1α": "PGC-1 alpha (mitochondrial biogenesis regulator)",
    "PPARα": "PPAR-alpha nuclear receptor",
    "Gβγ": "G-protein beta-gamma subunit complex",
    # biomarker Greek symbols seen as bare letters
    "ΔR2*": "delta R2-star (BOLD susceptibility change)",
    # cognitive_function
    "Id": "psychoanalytic Id construct",
}


# Generic Greek-letter -> long-form expansions, applied when a name is
# JUST a single non-ASCII letter (these almost always mean a biomarker
# the KG failed to contextualise). Keep terse - the LLM only needs a hint.
_GREEK_FALLBACK: dict[str, str] = {
    "α": "alpha (rhythm or subunit)",
    "β": "beta (rhythm, peptide, or subunit)",
    "γ": "gamma (rhythm or subunit)",
    "δ": "delta (rhythm or peptide)",
    "θ": "theta (rhythm)",
    "τ": "tau (protein or time constant)",
    "Δ": "delta (change)",
    "Ψ": "psi",
    "Ω": "omega",
    "Φ": "phi",
}


# Names too short / generic / non-ASCII to be useful inventory tokens. These
# get DROPPED from the inventory entirely - no amount of long-form context
# rescues a single-letter ASCII token like "F" or a stray "X".
def _is_unusable_name(name: str) -> bool:
    if not name:
        return True
    stripped = name.strip()
    # Single ASCII letter / digit tokens are useless as data labels.
    if len(stripped) <= 1 and stripped.isascii():
        return True
    # Bare numeric / percentage tokens like "45%", "90%", "7T", "1.5T".
    # Field strengths like 7T/3T/1.5T are real but belong in modality, not
    # biomarker; we drop them here to avoid LLM treating them as scalars.
    if re.fullmatch(r"\d+(\.\d+)?\s*[%TK]?", stripped):
        return True
    # Sentence-like fragments (KG sometimes contains misclassified claim
    # snippets in disease/drug). Reject anything that looks more like a
    # narrative phrase than a label: too long OR has multiple commas /
    # connective words that no real disease/drug name would carry.
    if len(stripped) > 80:
        return True
    return False


def _looks_like_sentence_fragment(name: str) -> bool:
    """Heuristic: reject narrative phrases that slipped into label fields."""
    if " and " in name or " of " in name or " with " in name and len(name) > 40:
        return True
    if name.count(",") >= 2:
        return True
    if name.count(" ") >= 8:
        return True
    return False


# Source vocabularies whose entries are canonical (curated upstream by MeSH /
# ATC / ClinicalOutcomes / etc.) rather than free-text claim fragments.
_CANONICAL_SOURCE_VOCABS: frozenset[str] = frozenset({
    "MeSH",
    "DisGeNET",
    "CognitiveAtlas",
    "ATC",
    "RxNorm",
    "SNOMED-CT",
    "ClinicalOutcomes",
    "MedDRA-SOC",
    "HGNC",
    "UniProt",
})


def _is_canonical_concept(c: dict) -> bool:
    """Return True if concept comes from a curated source or has a UMLS CUI.

    Used as a positive filter for the prefer-long buckets (disease, drug,
    treatment_outcome) where the KG mixes free-text claim fragments with
    real entries and we want the LLM to see only the canonical ones.
    """
    if c.get("source_vocab", "") in _CANONICAL_SOURCE_VOCABS:
        return True
    md = c.get("metadata") or {}
    if md.get("umls_cui"):
        return True
    return False


def _display_name(name: str) -> str:
    """Return the LLM-facing string for an inventory token.

    Adds a hand-curated long form (or a Greek-letter fallback) so the LLM
    treats cryptic short tokens as named biological entities rather than
    algebraic variables. The original token is kept inside parentheses so
    a recipe's inputs_used can still cite it verbatim.
    """
    if name in _LONGFORM:
        return f"{_LONGFORM[name]} ({name})"
    if name in _GREEK_FALLBACK:
        return f"{_GREEK_FALLBACK[name]} ({name})"
    return name


def _has_longform(name: str) -> bool:
    return name in _LONGFORM or name in _GREEK_FALLBACK


@dataclass
class Recipe:
    """A single computable quantity proposed by the LLM."""
    id: str
    name: str
    formula: str
    inputs_used: list[str] = field(default_factory=list)
    linked_concept_ids: list[str] = field(default_factory=list)
    atoms: list[str] = field(default_factory=list)
    llm_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── inventory ──────────────────────────────────────────────────────────


def _domain_of(concept: dict) -> str:
    tags = concept.get("domain_tags") or []
    return tags[0] if tags else concept.get("domain", "")


def build_inventory(concepts: dict[str, dict],
                    edges: Optional[Iterable[dict]] = None) -> dict[str, list[str]]:
    """Group concept names by bucket, capped per bucket.

    Filters performed:
      * single-character ASCII tokens are dropped entirely;
      * cryptic short tokens listed in ``_LONGFORM`` (or single
        Greek-letter names in ``_GREEK_FALLBACK``) are expanded to
        ``"<long form> (<short>)"`` so the LLM treats them as named
        biological entities rather than algebraic variables.

    Within each bucket, items with a hand-curated long form sort FIRST
    (so they survive the per-bucket cap), then remaining items are
    ordered by length, shortest first.

    For prefer-long buckets (disease/drug/treatment_outcome), if ``edges``
    is supplied, concepts are additionally ranked by edge degree so the
    most-connected entities (Alzheimer, Schizophrenia, donepezil, etc.)
    win the cap over rare hereditary subtypes that happen to sort earlier
    alphabetically.
    """
    degree: dict[str, int] = {}
    if edges is not None:
        for e in edges:
            sid = e.get("source_id")
            tid = e.get("target_id")
            if sid:
                degree[sid] = degree.get(sid, 0) + 1
            if tid:
                degree[tid] = degree.get(tid, 0) + 1

    by_dom: dict[str, list[tuple[int, int, str, str]]] = {dom: [] for dom, _, _, _, _ in _BUCKETS}
    seen_per_dom: dict[str, set[str]] = {dom: set() for dom, _, _, _, _ in _BUCKETS}
    prefer_long: dict[str, bool] = {dom: pl for dom, _, _, _, pl in _BUCKETS}
    for cid, c in concepts.items():
        dom = _domain_of(c)
        if dom not in by_dom:
            continue
        name = (c.get("preferred_name") or c.get("name") or "").strip()
        if not name or _is_unusable_name(name):
            continue
        # Prefer-long buckets (disease/drug/treatment_outcome) host claim
        # free-text alongside canonical entries. Drop non-canonical and
        # narrative-shaped names so the LLM sees only curated labels.
        if prefer_long.get(dom):
            if not _is_canonical_concept(c):
                continue
            if _looks_like_sentence_fragment(name):
                continue
        if name in seen_per_dom[dom]:
            continue
        seen_per_dom[dom].add(name)
        priority = 0 if _has_longform(name) else 1
        deg = degree.get(cid, 0)
        by_dom[dom].append((priority, deg, name, _display_name(name)))

    inventory: dict[str, list[str]] = {}
    for dom, _label, cap, _type, _pl in _BUCKETS:
        items = by_dom.get(dom, [])
        if prefer_long.get(dom):
            # Multi-word labels first; short uppercase acronyms (DSM, FDG,
            # PIB) are mostly noise so push them last. Within the canonical
            # pool, prefer Title Case names over SCREAMING CAPS rare-subtype
            # entries from MeSH, then prefer high-degree (well-connected)
            # concepts so Alzheimer / Schizophrenia / SSRIs beat rare
            # hereditary subtypes that happen to sort earlier alphabetically.
            def _key(t: tuple[int, int, str, str]) -> tuple:
                _, deg, raw, _disp = t
                is_acronym = raw.isupper() and len(raw) <= 4 and raw.isascii()
                is_screaming = raw.isupper() and len(raw) > 4
                n_words = len(raw.split())
                length_band = 0 if 2 <= n_words <= 6 else (1 if n_words == 1 else 2)
                return (
                    1 if is_acronym else 0,
                    1 if is_screaming else 0,
                    length_band,
                    -deg,
                    -n_words,
                    raw.lower(),
                )
            items.sort(key=_key)
        else:
            items.sort(key=lambda t: (t[0], len(t[2]), t[2].lower()))
        if items:
            inventory[dom] = [display for _, _, _, display in items[:cap]]
    return inventory


def _render_inventory(inventory: dict[str, list[str]]) -> str:
    """Render the inventory as a labelled, type-annotated list for the LLM."""
    lines: list[str] = []
    meta = {dom: (label, type_hint) for dom, label, _, type_hint, _ in _BUCKETS}
    for dom, items in inventory.items():
        label, type_hint = meta.get(dom, (dom, ""))
        head = f"- {label}"
        if type_hint:
            head += f" [{type_hint}]"
        head += f": {', '.join(items)}"
        lines.append(head)
    return "\n".join(lines)


# ── LLM brainstorm ─────────────────────────────────────────────────────


_SYSTEM_PROMPT = (
    "You propose computable quantities from raw biomedical data. "
    "You receive ONLY a data inventory (with type hints per bucket); "
    "you do not know the downstream task and must not speculate about it. "
    "Treat each inventory item as the kind of object its bucket type hint "
    "describes - a brain region label is NOT a number, a paradigm is NOT a "
    "scalar, a gene label is NOT directly numeric. Only propose quantities "
    "that respect those types. Output strictly valid JSON."
)


# Rules block reused by both single-shot and batched prompts. Centralised so
# behavioural fixes don't have to be duplicated in two strings.
_PROMPT_RULES = (
    "RULES (a recipe that violates any of these is invalid):\n"
    "  1. A formula combining two items must be unit-compatible. Do NOT "
    "average, sum, or take ratios across incompatible types (e.g. a brain "
    "region label with a diffusion coefficient, a paradigm with a "
    "biomarker, a gene label with a stimulus category).\n"
    "  2. Brain regions, genes, neurotransmitters, and stimulus categories "
    "are LABELS, not scalars. To turn a label into a number you MUST pair "
    "it with a measurement modality or a quantification verb (volume of, "
    "thickness of, allele dosage of, concentration of, mean BOLD response "
    "to, etc.).\n"
    "  3. A single-token formula like just \"Galanin\" or just "
    "\"vehicle\" is NOT a recipe - it is a label. Reject such items.\n"
    "  4. If you cannot propose a sensible quantification for an inventory "
    "item under these rules, skip it; do not pad with mechanical "
    "A_op_B combinations.\n"
)


def _build_prompt(inventory: dict[str, list[str]], n: int) -> str:
    return (
        f"Available raw data:\n{_render_inventory(inventory)}\n\n"
        f"{_PROMPT_RULES}\n"
        f"List {n} distinct quantities that can be computed from this data. "
        "Each item is a single scalar or vector value, defined by a short "
        "name and a formula. Mix simple (one-region one-measure) and "
        "compositional (ratios, differences, network summaries) forms, "
        "but ONLY when the combination is biologically meaningful. "
        "Do not justify, do not group, do not consider any downstream use.\n\n"
        "Return JSON array. Each element:\n"
        '  {"name": "<short label>", '
        '"formula": "<one-line definition referencing inventory items>", '
        '"inputs_used": ["<inventory item>", ...]}'
    )


def brainstorm_recipes(
    inventory: dict[str, list[str]],
    n: int,
    llm_call: Callable[[str, str], str],
    model_name: str = "",
) -> list[Recipe]:
    """Single LLM call -> list of Recipe.

    ``llm_call(prompt, system_prompt) -> raw_text`` is injected so tests
    can pass a stub. Production callers wire it to the OpenAI client used
    by ``critic_agent.py``.
    """
    prompt = _build_prompt(inventory, n)
    raw = llm_call(prompt, _SYSTEM_PROMPT)
    data = _extract_json(raw)
    if not isinstance(data, list):
        logger.warning("LLM did not return a JSON array; got %r", type(data))
        return []

    recipes: list[Recipe] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        formula = str(item.get("formula") or "").strip()
        if not name or not formula:
            continue
        inputs_used = [str(x).strip() for x in (item.get("inputs_used") or []) if str(x).strip()]
        recipes.append(Recipe(
            id=f"recipe_{i+1:04d}",
            name=name,
            formula=formula,
            inputs_used=inputs_used,
            llm_model=model_name,
        ))
    return recipes


def _extract_json(text: str) -> Any:
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    for s, e in (("[", "]"), ("{", "}")):
        i, j = text.find(s), text.rfind(e)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ── junk filter ────────────────────────────────────────────────────────


# Single uppercase letter optionally followed by digits, e.g. "R1", "C7", "E1".
# Real biomarkers (CD4, IL6, APOE) survive because they carry >= 2 letters or
# a known prefix, so the rule "ALL inputs_used match this pattern" rarely hits
# legitimate items.
_PLACEHOLDER_TOKEN = re.compile(r"^[A-Z]\d{0,2}$")


def _is_placeholder_recipe(name: str, formula: str, inputs_used: list[str],
                           inventory_lc: set[str]) -> bool:
    """Return True if a recipe looks like a fabricated placeholder.

    Triggers:
      * inputs_used is empty or every token is a single-letter variable, OR
      * none of inputs_used overlap with the inventory (case-insensitive).
    """
    if not inputs_used:
        return True
    if all(_PLACEHOLDER_TOKEN.match(s.strip()) for s in inputs_used):
        return True
    lowered = {s.strip().lower() for s in inputs_used}
    if not (lowered & inventory_lc):
        return True
    # Bare-number ratios like "90% - 45%" or "0.5 + 0.5".
    if re.fullmatch(r"[\d\s\.\-\+\*\/%]+", formula.strip()):
        return True
    return False


def _flatten_inventory_lc(inventory: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    for items in inventory.values():
        for it in items:
            out.add(it.strip().lower())
    return out


# ── batched brainstorm ─────────────────────────────────────────────────


def _normalise_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", " ", s.strip().lower())


def brainstorm_recipes_batched(
    inventory: dict[str, list[str]],
    n_total: int,
    llm_call: Callable[[str, str], str],
    model_name: str = "",
    batch_size: int = 50,
    seed: int = 0,
) -> list[Recipe]:
    """Generate ``n_total`` recipes via repeated batched LLM calls.

    Each batch:
      * picks 3-4 focus buckets (rotated deterministically by seed) so the
        LLM doesn't keep repeating the same hippocampus/APOE recipes;
      * receives up to 30 already-emitted names as a "do not repeat" hint;
      * is filtered for placeholder/junk items before being merged.

    Names are de-duplicated globally by ``_normalise_name``. Returns up to
    ``n_total`` Recipe objects, all with sequentially assigned ids
    (``recipe_0001`` ...). May return fewer if the LLM's unique output
    saturates.
    """
    inv_lc = _flatten_inventory_lc(inventory)
    bucket_names = list(inventory.keys())
    rng = random.Random(seed)

    seen_names: set[str] = set()
    accepted: list[Recipe] = []
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        if len(accepted) >= n_total:
            break
        focus = _pick_focus_buckets(bucket_names, batch_idx, rng)
        sub_inventory = {b: inventory[b] for b in focus if b in inventory}
        if not sub_inventory:
            sub_inventory = inventory
        sample_existing = _sample_existing_names(accepted, rng, k=30)
        prompt = _build_prompt_batched(sub_inventory, batch_size,
                                        sample_existing, batch_idx)
        try:
            raw = llm_call(prompt, _SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch %d LLM call failed: %s", batch_idx, exc)
            continue
        data = _extract_json(raw)
        if not isinstance(data, list):
            logger.warning("batch %d: not a JSON array (%r)", batch_idx,
                            type(data))
            continue
        added_this_batch = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            formula = str(item.get("formula") or "").strip()
            if not name or not formula:
                continue
            inputs_used = [str(x).strip() for x in (item.get("inputs_used") or [])
                            if str(x).strip()]
            if _is_placeholder_recipe(name, formula, inputs_used, inv_lc):
                continue
            key = _normalise_name(name)
            if key in seen_names:
                continue
            seen_names.add(key)
            accepted.append(Recipe(
                id="",  # assigned after dedup
                name=name,
                formula=formula,
                inputs_used=inputs_used,
                llm_model=model_name,
            ))
            added_this_batch += 1
            if len(accepted) >= n_total:
                break
        logger.info("batch %d/%d focus=%s -> +%d (total %d/%d)",
                    batch_idx + 1, n_batches, focus, added_this_batch,
                    len(accepted), n_total)

    accepted = accepted[:n_total]
    for i, r in enumerate(accepted):
        r.id = f"recipe_{i+1:04d}"
    return accepted


def _pick_focus_buckets(buckets: list[str], batch_idx: int,
                        rng: random.Random) -> list[str]:
    """Rotate through bucket combinations so each batch has a different focus."""
    if len(buckets) <= 4:
        return buckets
    # Shift by batch_idx to walk the bucket list, plus 1-2 random extras for
    # cross-bucket compositional recipes.
    primary = buckets[batch_idx % len(buckets)]
    secondary = buckets[(batch_idx + 1) % len(buckets)]
    pool = [b for b in buckets if b not in (primary, secondary)]
    extras = rng.sample(pool, k=min(2, len(pool)))
    return [primary, secondary, *extras]


def _sample_existing_names(accepted: list[Recipe], rng: random.Random,
                            k: int) -> list[str]:
    if not accepted:
        return []
    if len(accepted) <= k:
        return [r.name for r in accepted]
    return [r.name for r in rng.sample(accepted, k=k)]


def _build_prompt_batched(inventory: dict[str, list[str]], n: int,
                          existing: list[str], batch_idx: int) -> str:
    base = (
        f"Available raw data (focus buckets for THIS batch):\n"
        f"{_render_inventory(inventory)}\n\n"
        f"{_PROMPT_RULES}\n"
        f"Propose {n} distinct quantities computable from this data. "
        "Each item is a single scalar or vector value, defined by a short "
        "name and a formula. Mix simple (one-region one-measure) and "
        "compositional (ratios, differences, network summaries, "
        "cross-modality combinations) forms - but combinations MUST be "
        "biologically meaningful per the rules above. Every formula MUST "
        "reference at least one item from the inventory by its actual name "
        "AND pair label-only items with a quantification verb (volume of, "
        "concentration of, dosage of, mean response to, ...). "
        "Do NOT invent variable names like R1, C3, E1, X, Y. "
        "Do NOT pad with mechanical 'A op B' combinations between bare "
        "labels. Do not justify, do not group, do not consider any "
        "downstream use.\n\n"
    )
    if existing:
        base += (
            "Already proposed in earlier batches (do NOT repeat or "
            "trivially rephrase these):\n"
        )
        for nm in existing:
            base += f"  - {nm}\n"
        base += "\n"
    base += (
        "Return JSON array. Each element:\n"
        '  {"name": "<short label>", '
        '"formula": "<one-line definition referencing inventory items>", '
        '"inputs_used": ["<inventory item>", ...]}'
    )
    return base


# ── link to KG concepts ────────────────────────────────────────────────


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def link_to_concepts(
    recipes: list[Recipe],
    concepts: dict[str, dict],
    linkable_domains: Iterable[str] = ("neuroanatomy", "biomarker", "gene",
                                        "cognitive_function", "modality",
                                        "neurotransmitter", "paradigm",
                                        "visual_stimulus",
                                        "individual_data_anchor",
                                        "dataset_variable",
                                        "disease", "drug",
                                        "treatment_outcome"),
) -> list[Recipe]:
    """Populate ``recipe.linked_concept_ids`` by name/substring match.

    Two indices are built over the linkable concepts:

    * **long keys** (``len >= 4``) are matched as bare substrings against the
      lowercased haystack of ``name + formula + inputs_used``;
    * **short keys** (``2-3`` chars, e.g. ``Pu``, ``C3``, ``Aβ``, ``AR``) are
      matched only on whole-word boundaries to avoid spurious hits like
      ``Pu`` matching ``"output"``.

    Each cryptic preferred_name listed in ``_LONGFORM`` also contributes its
    long form (e.g. ``"complement c3"``) and bracketed display form
    (e.g. ``"complement c3 (c3)"``) as additional long keys, so a recipe
    that cites the LLM-facing display string still resolves to the KG node.
    """
    linkable = set(linkable_domains)
    long_keys: dict[str, list[str]] = {}
    short_keys: dict[str, list[str]] = {}

    def _add_key(table: dict[str, list[str]], key: str, cid: str) -> None:
        ids = table.setdefault(key, [])
        if cid not in ids:
            ids.append(cid)

    for cid, c in concepts.items():
        if _domain_of(c) not in linkable:
            continue
        candidates: set[str] = set()
        pn = c.get("preferred_name") or ""
        if pn:
            candidates.add(pn)
        for alias in c.get("aliases") or []:
            if alias:
                candidates.add(alias)
        # Inject _LONGFORM display strings so display-form citations link back.
        if pn in _LONGFORM:
            candidates.add(_LONGFORM[pn])
            candidates.add(f"{_LONGFORM[pn]} ({pn})")
        if pn in _GREEK_FALLBACK:
            candidates.add(_GREEK_FALLBACK[pn])
        for nm in candidates:
            key = _normalise(nm)
            if len(key) >= 4:
                _add_key(long_keys, key, cid)
            elif len(key) >= 2:
                _add_key(short_keys, key, cid)

    long_keys_sorted = sorted(long_keys.keys(), key=len, reverse=True)
    short_keys_sorted = sorted(short_keys.keys(), key=len, reverse=True)

    for r in recipes:
        haystack = _normalise(f"{r.name} {r.formula} {' '.join(r.inputs_used)}")
        hits: list[str] = []
        seen: set[str] = set()
        for key in long_keys_sorted:
            if key in haystack:
                for cid in long_keys[key]:
                    if cid not in seen:
                        seen.add(cid)
                        hits.append(cid)
                if len(hits) >= 20:
                    break
        # Whole-word match for short keys (Pu, C3, AR, ...). Use \b on both
        # sides so "C3" doesn't fire on "object" (it wouldn't anyway, but
        # "AR" on "MAGNETIC RESONANCE" would).
        for key in short_keys_sorted:
            if len(hits) >= 20:
                break
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])"
            if re.search(pattern, haystack):
                for cid in short_keys[key]:
                    if cid not in seen:
                        seen.add(cid)
                        hits.append(cid)
        r.linked_concept_ids = hits
    return recipes


# ── atom tagging ────────────────────────────────────────────────────────


# Modality tokens that are NOT imaging signals - they capture molecular or
# behavioural data and must not be tagged IMAGING_MARKER on their own.
_NON_IMAGING_MODALITY_TOKENS: frozenset[str] = frozenset({
    "genetics",
    "clinical",
    "physical",
    "environment",
    "eye_tracking",
})


def tag_atoms(recipes: list[Recipe], concepts: dict[str, dict]) -> list[Recipe]:
    """Populate ``recipe.atoms`` from linked concepts and modality inputs.

    A recipe is tagged with every atom realised by:
      * the primary domain of any linked KG concept (via
        :func:`atoms_for_domain` from :mod:`neurooracle.src.atoms`);
      * any *imaging* modality token cited in ``inputs_used`` (sMRI, fMRI,
        dMRI, PET, EEG, MEG, ...) - these add IMAGING_MARKER even when
        the recipe links no imaging-domain concepts.

    Non-imaging modality strings (``genetics``, ``clinical``, ``physical``,
    ``environment``, ``eye_tracking``) do NOT contribute IMAGING_MARKER on
    their own; they pass through whatever atom their accompanying linked
    concept implies.
    """
    # Local import to avoid a circular import in case Recipe is later used
    # inside neurooracle.src.atoms.
    from ..atoms import atoms_for_domain

    inv_modalities: set[str] = set()
    # We don't have inventory here, so fall back to the KG: any concept whose
    # primary domain is "modality" and whose preferred_name is NOT a
    # non-imaging signal counts as an imaging modality token.
    for c in concepts.values():
        if _domain_of(c) == "modality":
            pn = (c.get("preferred_name") or "").strip().lower()
            if pn and pn not in _NON_IMAGING_MODALITY_TOKENS:
                inv_modalities.add(pn)

    for r in recipes:
        atoms_seen: set[str] = set()
        for cid in r.linked_concept_ids:
            c = concepts.get(cid) or {}
            dom = _domain_of(c)
            for a in atoms_for_domain(dom):
                atoms_seen.add(a.name)
        used_lc = {s.strip().lower() for s in r.inputs_used}
        if used_lc & inv_modalities:
            atoms_seen.add("IMAGING_MARKER")
        # Preserve a stable order (sorted) so JSON diffs are clean.
        r.atoms = sorted(atoms_seen)
    return recipes
