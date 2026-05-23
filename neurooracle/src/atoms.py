"""Compositional task algebra for NeuroClaw / NeuroOracle.

A small alphabet of meta-concepts (atoms) — disease, drug, imaging marker,
gene target, cognitive task, outcome, individual data — combines under
direction to form research tasks. Hypothesis generation, evaluation, and
benchmarking are then organised over this task alphabet rather than
ad-hoc node-level pairs.

The alphabet is intentionally small (~7) so the task space stays
tractable. Each atom maps to one or more KG domain tags via
ATOM_TO_DOMAINS; a node in the graph can serve as a given atom whenever
its domain tag appears in that atom's domain set.

A canonical task is `(inputs ⊂ Atoms, output ∈ Atoms, modifier?)`. The
registry CANONICAL_TASKS lists ~15 tasks the platform currently supports;
this is the *lower bound* of coverage — adding new atoms automatically
expands the combinatorial space, with humans selecting which combinations
are scientifically meaningful (see `candidate_tasks_with_atom`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations


# ── Atom alphabet ──────────────────────────────────────────────────────────────

class Atom(str, Enum):
    """Meta-concepts that compose into research tasks."""
    DISEASE         = "disease"
    DRUG            = "drug"
    IMAGING_MARKER  = "imaging_marker"
    GENE_TARGET     = "gene_target"
    COGNITIVE_TASK  = "cognitive_task"
    OUTCOME         = "outcome"
    INDIVIDUAL_DATA = "individual_data"


# Short labels for compact task signatures: e.g. "{IM,Idv}->D[subtype]".
_SHORT_NAMES: dict[Atom, str] = {
    Atom.DISEASE:         "D",
    Atom.DRUG:            "Rx",
    Atom.IMAGING_MARKER:  "IM",
    Atom.GENE_TARGET:     "G",
    Atom.COGNITIVE_TASK:  "Tk",
    Atom.OUTCOME:         "O",
    Atom.INDIVIDUAL_DATA: "Idv",
}

# Display order — used by signature() to emit deterministic strings.
_DISPLAY_ORDER: tuple[Atom, ...] = (
    Atom.DISEASE, Atom.DRUG, Atom.IMAGING_MARKER, Atom.GENE_TARGET,
    Atom.COGNITIVE_TASK, Atom.OUTCOME, Atom.INDIVIDUAL_DATA,
)


# ── Atom ↔ KG-domain mapping ──────────────────────────────────────────────────
#
# Each atom is realised in the KG by nodes carrying one of the listed domain
# tags. Multi-mapping is intentional: atoms abstract over related domains
# ("imaging marker" covers imaging_feature + connectivity + biomarker +
# neuroanatomy-as-target). The same domain tag may also appear under multiple
# atoms — for instance `dataset_variable` serves both OUTCOME (ΔADAS-Cog,
# responder labels) and INDIVIDUAL_DATA (age, lifestyle); disambiguation
# happens at use site, either by task semantics or node metadata.
#
# Infrastructure tags (atlas, modality, dataset, ml_model) and meta tags
# (claim, recipe) are intentionally *not* members of any atom — they describe
# the experimental apparatus, not the scientific content of a hypothesis.
ATOM_TO_DOMAINS: dict[Atom, frozenset[str]] = {
    Atom.DISEASE:         frozenset({"disease"}),
    Atom.DRUG:            frozenset({"drug"}),
    Atom.IMAGING_MARKER:  frozenset({
        "imaging_feature",   # cortical thickness, FA, SUVR, ALFF, ReHo …
        "connectivity",      # FC matrix entries, structural connectivity
        "biomarker",         # CSF Aβ42, p-tau, blood biomarkers
        "neuroanatomy",      # used as measurement target (volume, activation)
    }),
    Atom.GENE_TARGET:     frozenset({"gene", "neurotransmitter"}),
    Atom.COGNITIVE_TASK:  frozenset({
        "paradigm",            # n-back, stop-signal …
        "cognitive_function",  # working memory, attention …
        "visual_stimulus",     # face / scene / object / motion …
        "emotion",             # affective state labels
        "vigilance",           # alertness levels
    }),
    Atom.OUTCOME:         frozenset({"treatment_outcome", "dataset_variable"}),
    Atom.INDIVIDUAL_DATA: frozenset({
        "dataset_variable",         # UKB/ADNI/HCP host categories (smoking, demographics, …)
        "individual_data_anchor",   # concept-level anchors (Aging, APOE, Big-5 traits, …)
                                    # seeded by ingestion.individual_data_anchors;
                                    # bridges concept-side IM/disease/gene nodes to
                                    # the dataset_variable hubs so brain_age /
                                    # connectome_behavior / task_brain_behavior /
                                    # disease_biomarker_prognosis can route IM ↔ Idv.
    }),
}


def _build_domain_to_atoms() -> dict[str, frozenset[Atom]]:
    rev: dict[str, set[Atom]] = {}
    for atom, doms in ATOM_TO_DOMAINS.items():
        for d in doms:
            rev.setdefault(d, set()).add(atom)
    return {d: frozenset(atoms) for d, atoms in rev.items()}


# Inverse: domain tag → atoms that domain can play.
DOMAIN_TO_ATOMS: dict[str, frozenset[Atom]] = _build_domain_to_atoms()


# ── Task definition ──────────────────────────────────────────────────────────

class TaskModifier(str, Enum):
    """Optional qualifiers attached to a task."""
    NONE         = ""
    LONGITUDINAL = "longitudinal"   # output observed at later timepoint
    CONTRASTIVE  = "contrastive"    # disease A vs disease B
    SUBTYPE      = "subtype"        # output is a sub-class of the output atom
    CONDITIONAL  = "conditional"    # path conditional on a particular input atom


@dataclass(frozen=True)
class Task:
    """A research task expressed as a directed combination of atoms.

    A task is identified by its input atom set, output atom, and optional
    modifier. Tasks form an alphabet over which hypothesis generation,
    evaluation, and benchmarking are organised.
    """
    name: str
    inputs: frozenset[Atom]
    output: Atom
    modifier: TaskModifier = TaskModifier.NONE
    description: str = ""
    example: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task.name must be non-empty")
        if not self.inputs:
            raise ValueError(f"Task '{self.name}': inputs must be non-empty")

    @property
    def signature(self) -> str:
        """Compact textual signature, e.g. '{IM,Idv}->D[subtype]'."""
        ins = ",".join(_SHORT_NAMES[a] for a in _DISPLAY_ORDER if a in self.inputs)
        out = _SHORT_NAMES[self.output]
        mod = f"[{self.modifier.value}]" if self.modifier != TaskModifier.NONE else ""
        return f"{{{ins}}}->{out}{mod}"

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "name": self.name,
            "inputs": [a.value for a in _DISPLAY_ORDER if a in self.inputs],
            "output": self.output.value,
            "modifier": self.modifier.value,
            "description": self.description,
            "example": self.example,
            "signature": self.signature,
        }


# ── Task chain (mediation: ordered ≥3-hop atom sequence) ─────────────────────
#
# A flat ``Task`` says "given inputs together, predict output" — input atoms
# are parallel. A ``TaskChain`` says "X acts on Z *through* Y" — atoms occur
# in a fixed order along the path. Mediation is a different scientific claim:
# discovering "BDNF → hippocampus → depression" is informationally richer
# than "BDNF → depression" because the intermediate is the mechanism.

@dataclass(frozen=True)
class TaskChain:
    """A research task expressed as an ordered chain of ≥3 atoms.

    Path generation must visit atom domains in the listed order — start in
    ``chain[0]``, transit ``chain[1..-2]`` as mediators, terminate in
    ``chain[-1]``. Distinct from :class:`Task`, where inputs are parallel.
    """
    name: str
    chain: tuple[Atom, ...]
    modifier: TaskModifier = TaskModifier.NONE
    description: str = ""
    example: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TaskChain.name must be non-empty")
        if len(self.chain) < 3:
            raise ValueError(
                f"TaskChain '{self.name}': chain must have ≥3 atoms "
                f"(got {len(self.chain)})"
            )

    @property
    def signature(self) -> str:
        """Compact textual signature, e.g. 'G->IM->D'."""
        body = "->".join(_SHORT_NAMES[a] for a in self.chain)
        mod = f"[{self.modifier.value}]" if self.modifier != TaskModifier.NONE else ""
        return f"{body}{mod}"

    @property
    def source(self) -> Atom:
        return self.chain[0]

    @property
    def target(self) -> Atom:
        return self.chain[-1]

    @property
    def mediators(self) -> tuple[Atom, ...]:
        return self.chain[1:-1]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "chain": [a.value for a in self.chain],
            "modifier": self.modifier.value,
            "description": self.description,
            "example": self.example,
            "signature": self.signature,
        }


# ── Canonical task registry ──────────────────────────────────────────────────
# Lower bound of platform coverage; not exhaustive. Adding a new task here
# makes it visible to the hypothesis engine, NeuroBench, and the explorer.

CANONICAL_TASKS: tuple[Task, ...] = (
    # ── A. Disease understanding ─────────────────────────────────
    Task(
        name="biomarker_discovery",
        inputs=frozenset({Atom.IMAGING_MARKER}),
        output=Atom.DISEASE,
        description="Identify imaging markers that distinguish or predict a disease.",
        example="Hippocampal volume → Alzheimer's Disease",
    ),
    Task(
        name="disease_subtyping",
        inputs=frozenset({Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA}),
        output=Atom.DISEASE,
        modifier=TaskModifier.SUBTYPE,
        description="Multivariate features → disease subtype label.",
        example="DMN connectivity + age → AD-typical vs AD-hippocampal-sparing",
    ),
    Task(
        name="progression_prediction",
        inputs=frozenset({Atom.IMAGING_MARKER}),
        output=Atom.DISEASE,
        modifier=TaskModifier.LONGITUDINAL,
        description="Baseline imaging → future disease conversion.",
        example="MCI baseline imaging → AD conversion within 24 months",
    ),
    Task(
        name="imaging_genetics",
        inputs=frozenset({Atom.GENE_TARGET}),
        output=Atom.IMAGING_MARKER,
        description="Genotype / molecular target → brain phenotype.",
        example="APOE-ε4 → hippocampal atrophy",
    ),
    Task(
        name="differential_diagnosis",
        inputs=frozenset({Atom.IMAGING_MARKER}),
        output=Atom.DISEASE,
        modifier=TaskModifier.CONTRASTIVE,
        description="Distinguish disease A from disease B using imaging features.",
        example="rs-fMRI features distinguish bipolar from major depression",
    ),

    # ── B. Treatment optimisation ───────────────────────────────
    Task(
        name="drug_response_prediction",
        inputs=frozenset({Atom.DRUG, Atom.IMAGING_MARKER, Atom.DISEASE}),
        output=Atom.OUTCOME,
        description="Patient profile + drug → expected response magnitude.",
        example="Baseline DaTSCAN + levodopa → ΔMDS-UPDRS at 12 months",
    ),
    Task(
        name="personalised_treatment",
        inputs=frozenset({Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA, Atom.DISEASE}),
        output=Atom.DRUG,
        description="Patient profile → recommended drug.",
        example="MCI + APOE-ε4 + low hippocampal volume → cholinesterase inhibitor",
    ),
    Task(
        name="drug_repurposing",
        inputs=frozenset({Atom.DRUG}),
        output=Atom.DISEASE,
        description="Existing drug → novel indication via mechanism path.",
        example="Memantine → traumatic brain injury (via NMDA modulation)",
    ),
    Task(
        name="adverse_event_prediction",
        inputs=frozenset({Atom.DRUG}),
        output=Atom.OUTCOME,
        description="Drug → adverse-event likelihood (off-target effects).",
        example="Levodopa long-term → impulse control disorder",
    ),
    Task(
        name="neuromodulation_target",
        inputs=frozenset({Atom.DISEASE, Atom.COGNITIVE_TASK}),
        output=Atom.IMAGING_MARKER,
        description="Disease + symptom → optimal stimulation site.",
        example="Treatment-resistant depression + anhedonia → DMN node target",
    ),

    # ── C. Brain function mapping ───────────────────────────────
    Task(
        name="functional_localization",
        inputs=frozenset({Atom.COGNITIVE_TASK}),
        output=Atom.IMAGING_MARKER,
        description="Stimulus / cognitive task → brain region or activation.",
        example="Face stimulus → FFA activation",
    ),
    Task(
        name="cognitive_decoding",
        inputs=frozenset({Atom.IMAGING_MARKER}),
        output=Atom.COGNITIVE_TASK,
        description="Brain activity → stimulus / mental-state label.",
        example="Visual cortex BOLD pattern → seen-image category",
    ),
    Task(
        name="connectome_behavior",
        inputs=frozenset({Atom.IMAGING_MARKER}),
        output=Atom.INDIVIDUAL_DATA,
        description="Connectivity → behavioural / cognitive trait score.",
        example="DMN FC → fluid intelligence score",
    ),

    # ── D. Health monitoring ────────────────────────────────────
    Task(
        name="brain_age",
        inputs=frozenset({Atom.IMAGING_MARKER}),
        output=Atom.INDIVIDUAL_DATA,
        description="Imaging features → biological age estimate.",
        example="Cortical thickness + GM volume → predicted age",
    ),
    Task(
        name="prognosis",
        inputs=frozenset({Atom.IMAGING_MARKER, Atom.DISEASE}),
        output=Atom.OUTCOME,
        modifier=TaskModifier.LONGITUDINAL,
        description="Acute / baseline state + disease → outcome at follow-up.",
        example="Stroke acute lesion → 6-month NIHSS",
    ),
)


# ── Canonical chain registry (3-hop mediation tasks) ─────────────────────────
# Mediated forms of the flat tasks above: same atoms appear, but in a fixed
# order along the path so the intermediate atom is treated as a mechanism,
# not a parallel input. Initial set covers the four research families A–D.

CANONICAL_CHAINS: tuple[TaskChain, ...] = (
    # A. Disease understanding — gene → brain phenotype → disease
    TaskChain(
        name="genetic_imaging_disease",
        chain=(Atom.GENE_TARGET, Atom.IMAGING_MARKER, Atom.DISEASE),
        description=(
            "Genetic risk variant acts on disease through an imaging "
            "endophenotype — full imaging-genetics mediation chain."
        ),
        example="APOE-ε4 → hippocampal atrophy → Alzheimer's Disease",
    ),
    # B. Treatment optimisation — drug → brain change → outcome
    TaskChain(
        name="drug_imaging_outcome",
        chain=(Atom.DRUG, Atom.IMAGING_MARKER, Atom.OUTCOME),
        description=(
            "Drug effect on outcome mediated by an imaging change — "
            "mechanistic pharmacodynamics via brain imaging."
        ),
        example="SSRI → amygdala reactivity reduction → HAM-D improvement",
    ),
    # C. Brain function mapping — task → brain → behaviour
    TaskChain(
        name="task_brain_behavior",
        chain=(Atom.COGNITIVE_TASK, Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA),
        description=(
            "Cognitive task elicits brain activation that explains a "
            "behavioural / individual-difference trait."
        ),
        example="n-back → DLPFC activation → working-memory capacity",
    ),
    # D. Health monitoring — disease → biomarker → prognosis
    TaskChain(
        name="disease_biomarker_prognosis",
        chain=(Atom.DISEASE, Atom.IMAGING_MARKER, Atom.OUTCOME),
        modifier=TaskModifier.LONGITUDINAL,
        description=(
            "Disease state is mediated by an imaging biomarker that "
            "predicts longitudinal outcome — biomarker-as-mediator prognosis."
        ),
        example="MCI → hippocampal volume → 24-month MMSE decline",
    ),
)

def task_by_name(name: str) -> Task:
    """Look up a canonical task by its registry name. Raises KeyError."""
    for t in CANONICAL_TASKS:
        if t.name == name:
            return t
    raise KeyError(f"unknown task: {name}")


def chain_by_name(name: str) -> TaskChain:
    """Look up a canonical task chain by its registry name. Raises KeyError."""
    for c in CANONICAL_CHAINS:
        if c.name == name:
            return c
    raise KeyError(f"unknown chain: {name}")

def tasks_by_atom(atom: Atom, role: str = "any") -> tuple[Task, ...]:
    """All canonical tasks involving `atom`.

    role: 'input' (atom appears in inputs), 'output' (atom is the output),
    or 'any' (either, default).
    """
    if role == "input":
        return tuple(t for t in CANONICAL_TASKS if atom in t.inputs)
    if role == "output":
        return tuple(t for t in CANONICAL_TASKS if t.output == atom)
    if role == "any":
        return tuple(t for t in CANONICAL_TASKS
                     if atom in t.inputs or t.output == atom)
    raise ValueError(f"role must be input | output | any (got {role!r})")


def domains_for_atom(atom: Atom) -> frozenset[str]:
    """KG domain tags whose nodes can serve as `atom`."""
    return ATOM_TO_DOMAINS[atom]


def atoms_for_domain(domain: str) -> frozenset[Atom]:
    """Atoms a given KG domain tag can play.

    Returns an empty frozenset for domains that are intentionally outside
    the alphabet (atlas / modality / dataset / ml_model / claim / recipe),
    or for unknown tags. Callers can use the empty result to detect
    "infrastructure" nodes.
    """
    return DOMAIN_TO_ATOMS.get(domain, frozenset())


def candidate_tasks_with_atom(
    new_atom: Atom,
    max_inputs: int = 3,
) -> list[frozenset[Atom]]:
    """All input-atom subsets that include `new_atom`, up to `max_inputs`.

    Used when adding a new atom to enumerate the space of possible new
    tasks. Selecting which combinations are scientifically meaningful
    is human review — this just lists the combinatorial options.
    """
    if max_inputs < 1:
        raise ValueError("max_inputs must be ≥ 1")
    others = [a for a in Atom if a != new_atom]
    result: list[frozenset[Atom]] = [frozenset({new_atom})]
    for k in range(1, max_inputs):
        for combo in combinations(others, k):
            result.append(frozenset(combo) | {new_atom})
    return result


__all__ = [
    "Atom",
    "TaskModifier",
    "Task",
    "TaskChain",
    "ATOM_TO_DOMAINS",
    "DOMAIN_TO_ATOMS",
    "CANONICAL_TASKS",
    "CANONICAL_CHAINS",
    "task_by_name",
    "chain_by_name",
    "tasks_by_atom",
    "domains_for_atom",
    "atoms_for_domain",
    "candidate_tasks_with_atom",
]
