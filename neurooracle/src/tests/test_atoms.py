"""Tests for the compositional task algebra (atoms.py)."""

from __future__ import annotations

import pytest

from neurooracle.src.atoms import (
    Atom,
    TaskModifier,
    Task,
    TaskChain,
    ATOM_TO_DOMAINS,
    DOMAIN_TO_ATOMS,
    CANONICAL_TASKS,
    CANONICAL_CHAINS,
    task_by_name,
    chain_by_name,
    tasks_by_atom,
    domains_for_atom,
    atoms_for_domain,
    candidate_tasks_with_atom,
)
from neurooracle.src.schema import DomainTag


# ── Atom alphabet ──────────────────────────────────────────────────────────

def test_atom_alphabet_is_small_and_distinct():
    """The alphabet should be intentionally small (~7) and value-distinct."""
    members = list(Atom)
    assert 5 <= len(members) <= 12, f"alphabet should stay small, got {len(members)}"
    values = [a.value for a in members]
    assert len(set(values)) == len(values), "atom values must be distinct"


def test_atom_str_inheritance():
    """Atoms are str-Enums so they JSON-serialise cleanly."""
    assert Atom.DISEASE == "disease"
    assert Atom.IMAGING_MARKER.value == "imaging_marker"


# ── Atom ↔ Domain mapping ──────────────────────────────────────────────────

def test_every_atom_has_at_least_one_domain():
    for atom in Atom:
        domains = ATOM_TO_DOMAINS.get(atom)
        assert domains and len(domains) >= 1, f"{atom} has no domains mapped"


def test_atom_domains_reference_real_domain_tags():
    """Every domain string under ATOM_TO_DOMAINS must be a valid DomainTag."""
    valid = {d.value for d in DomainTag}
    for atom, domains in ATOM_TO_DOMAINS.items():
        for d in domains:
            assert d in valid, f"{atom} maps to unknown domain {d!r}"


def test_treatment_outcome_tag_exists():
    """Atom.OUTCOME requires treatment_outcome — schema must declare it."""
    assert "treatment_outcome" in {d.value for d in DomainTag}
    assert "treatment_outcome" in ATOM_TO_DOMAINS[Atom.OUTCOME]


def test_inverse_mapping_is_consistent():
    """DOMAIN_TO_ATOMS is the exact inverse of ATOM_TO_DOMAINS."""
    expected_pairs = {(a, d) for a, ds in ATOM_TO_DOMAINS.items() for d in ds}
    actual_pairs = {(a, d) for d, atoms in DOMAIN_TO_ATOMS.items() for a in atoms}
    assert expected_pairs == actual_pairs


def test_dataset_variable_serves_two_atoms():
    """dataset_variable is intentionally polysemous (OUTCOME + INDIVIDUAL_DATA)."""
    atoms = atoms_for_domain("dataset_variable")
    assert Atom.OUTCOME in atoms and Atom.INDIVIDUAL_DATA in atoms


def test_infrastructure_domains_are_outside_alphabet():
    """Atlas/modality/dataset/ml_model/claim/recipe describe apparatus, not science."""
    for d in ("atlas", "modality", "dataset", "ml_model", "claim", "recipe"):
        assert atoms_for_domain(d) == frozenset(), f"{d} should not map to any atom"


# ── Task class ─────────────────────────────────────────────────────────────

def test_task_requires_name_and_inputs():
    with pytest.raises(ValueError):
        Task(name="", inputs=frozenset({Atom.DISEASE}), output=Atom.DRUG)
    with pytest.raises(ValueError):
        Task(name="x", inputs=frozenset(), output=Atom.DRUG)


def test_task_is_frozen_and_hashable():
    t = Task(name="t1", inputs=frozenset({Atom.DRUG}), output=Atom.DISEASE)
    {t}  # noqa: B018  — verifies hashability
    with pytest.raises(Exception):
        t.name = "renamed"  # type: ignore[misc]


def test_task_signature_format():
    t = Task(
        name="example",
        inputs=frozenset({Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA}),
        output=Atom.DISEASE,
        modifier=TaskModifier.SUBTYPE,
    )
    assert t.signature == "{IM,Idv}->D[subtype]"


def test_task_signature_no_modifier():
    t = Task(name="ex", inputs=frozenset({Atom.DRUG}), output=Atom.DISEASE)
    assert t.signature == "{Rx}->D"


def test_task_to_dict_roundtrip_keys():
    t = task_by_name("biomarker_discovery")
    d = t.to_dict()
    for k in ("name", "inputs", "output", "modifier",
              "description", "example", "signature"):
        assert k in d
    assert d["inputs"] == ["imaging_marker"]
    assert d["output"] == "disease"


# ── Canonical task registry ────────────────────────────────────────────────

def test_canonical_task_names_unique():
    names = [t.name for t in CANONICAL_TASKS]
    assert len(set(names)) == len(names), "duplicate task names in registry"


def test_canonical_task_signatures_allow_meaningful_collisions():
    """Two tasks may share a signature when the atom shape is identical but
    the scientific intent differs (e.g. brain_age vs connectome_behavior both
    map IM → Idv, but predict age vs behavioural trait — the polysemy of
    the INDIVIDUAL_DATA atom is precisely what the alphabet allows).

    What we *do* require is that any colliding pair carry distinct names
    and descriptions — i.e. no truly duplicated entries.
    """
    by_sig: dict[str, list[Task]] = {}
    for t in CANONICAL_TASKS:
        by_sig.setdefault(t.signature, []).append(t)
    for sig, tasks in by_sig.items():
        names = {t.name for t in tasks}
        descs = {t.description for t in tasks}
        assert len(names) == len(tasks), \
            f"signature {sig}: duplicate names {[t.name for t in tasks]}"
        assert len(descs) == len(tasks), \
            f"signature {sig}: duplicate descriptions"


def test_canonical_tasks_have_resolvable_atom_pools():
    """Every canonical task's inputs and output map to a non-empty domain pool,
    so the hypothesis engine can sample real KG nodes for each role."""
    for t in CANONICAL_TASKS:
        for a in t.inputs:
            assert domains_for_atom(a), f"{t.name}: input atom {a} has empty domain pool"
        assert domains_for_atom(t.output), \
            f"{t.name}: output atom {t.output} has empty domain pool"


def test_canonical_minimum_coverage():
    """Sanity: all four task families A/B/C/D should be represented."""
    names = {t.name for t in CANONICAL_TASKS}
    must_have = {
        "biomarker_discovery",     # A. Disease understanding
        "drug_response_prediction",# B. Treatment optimisation
        "functional_localization", # C. Brain function mapping
        "brain_age",               # D. Health monitoring
    }
    assert must_have <= names, f"missing canonical tasks: {must_have - names}"


# ── Helpers ────────────────────────────────────────────────────────────────

def test_task_by_name_unknown_raises():
    with pytest.raises(KeyError):
        task_by_name("does_not_exist")


def test_tasks_by_atom_roles():
    in_imaging = tasks_by_atom(Atom.IMAGING_MARKER, role="input")
    out_imaging = tasks_by_atom(Atom.IMAGING_MARKER, role="output")
    any_imaging = tasks_by_atom(Atom.IMAGING_MARKER, role="any")
    assert all(Atom.IMAGING_MARKER in t.inputs for t in in_imaging)
    assert all(t.output == Atom.IMAGING_MARKER for t in out_imaging)
    assert set(any_imaging) >= set(in_imaging) | set(out_imaging)
    with pytest.raises(ValueError):
        tasks_by_atom(Atom.DISEASE, role="bogus")


def test_candidate_tasks_with_atom_includes_singleton():
    cands = candidate_tasks_with_atom(Atom.DISEASE, max_inputs=1)
    assert frozenset({Atom.DISEASE}) in cands
    assert all(Atom.DISEASE in c for c in cands)


def test_candidate_tasks_grow_with_max_inputs():
    c1 = candidate_tasks_with_atom(Atom.DISEASE, max_inputs=1)
    c2 = candidate_tasks_with_atom(Atom.DISEASE, max_inputs=2)
    c3 = candidate_tasks_with_atom(Atom.DISEASE, max_inputs=3)
    assert len(c1) <= len(c2) <= len(c3)
    # max_inputs=2 means 1-atom (just new) + 2-atom (new + each other)
    assert len(c2) == 1 + (len(Atom) - 1)


def test_candidate_tasks_rejects_zero():
    with pytest.raises(ValueError):
        candidate_tasks_with_atom(Atom.DISEASE, max_inputs=0)


# ── TaskChain (3-hop mediation) ───────────────────────────────────────────

def test_task_chain_requires_min_three_atoms():
    with pytest.raises(ValueError):
        TaskChain(name="too_short", chain=(Atom.DRUG, Atom.DISEASE))
    with pytest.raises(ValueError):
        TaskChain(name="empty", chain=())


def test_task_chain_requires_name():
    with pytest.raises(ValueError):
        TaskChain(
            name="",
            chain=(Atom.GENE_TARGET, Atom.IMAGING_MARKER, Atom.DISEASE),
        )


def test_task_chain_signature_format():
    c = TaskChain(
        name="x",
        chain=(Atom.GENE_TARGET, Atom.IMAGING_MARKER, Atom.DISEASE),
    )
    assert c.signature == "G->IM->D"


def test_task_chain_signature_with_modifier():
    c = TaskChain(
        name="x",
        chain=(Atom.DISEASE, Atom.IMAGING_MARKER, Atom.OUTCOME),
        modifier=TaskModifier.LONGITUDINAL,
    )
    assert c.signature == "D->IM->O[longitudinal]"


def test_task_chain_endpoints_and_mediators():
    c = TaskChain(
        name="x",
        chain=(Atom.COGNITIVE_TASK, Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA),
    )
    assert c.source == Atom.COGNITIVE_TASK
    assert c.target == Atom.INDIVIDUAL_DATA
    assert c.mediators == (Atom.IMAGING_MARKER,)


def test_task_chain_is_frozen_and_hashable():
    c = TaskChain(
        name="x",
        chain=(Atom.DRUG, Atom.IMAGING_MARKER, Atom.OUTCOME),
    )
    {c}  # noqa: B018 — verifies hashability
    with pytest.raises(Exception):
        c.name = "renamed"  # type: ignore[misc]


def test_task_chain_to_dict_roundtrip_keys():
    c = chain_by_name("genetic_imaging_disease")
    d = c.to_dict()
    for k in ("name", "chain", "modifier", "description", "example", "signature"):
        assert k in d
    assert d["chain"] == ["gene_target", "imaging_marker", "disease"]


def test_canonical_chains_present_for_all_registered_families():
    names = {c.name for c in CANONICAL_CHAINS}
    assert names == {
        "genetic_imaging_disease",
        "drug_imaging_outcome",
        "task_brain_behavior",
        "disease_biomarker_prognosis",
        "pathway_polygenic_mediation",
    }


def test_pathway_polygenic_mediation_keeps_disease_implicit_in_outcome():
    """CS3 is intentionally G -> IM -> O, with disease encoded by outcome."""
    c = chain_by_name("pathway_polygenic_mediation")
    assert c.signature == "G->IM->O[longitudinal]"
    assert c.chain == (Atom.GENE_TARGET, Atom.IMAGING_MARKER, Atom.OUTCOME)


def test_canonical_chain_names_unique():
    names = [c.name for c in CANONICAL_CHAINS]
    assert len(set(names)) == len(names)


def test_canonical_chains_have_resolvable_atom_pools():
    for c in CANONICAL_CHAINS:
        for a in c.chain:
            assert domains_for_atom(a), \
                f"{c.name}: atom {a} has empty domain pool"


def test_chain_by_name_unknown_raises():
    with pytest.raises(KeyError):
        chain_by_name("does_not_exist")
