from __future__ import annotations

from types import SimpleNamespace

from neurooracle.scripts.build_umls_atomic_mention_impact import (
    resolve_unit,
    scan_mrconso,
    scan_mrsty,
)
from neurooracle.src import umls_canonicalize
from neurooracle.src.merge_by_umls import _group_by_cui
from neurooracle.src.umls_integration import _collect_graph_names
from neurooracle.src.umls_mention_mapping import (
    atomize_mention_candidates,
    biomedical_context,
    is_lexically_eligible,
    normalize_term,
    select_atomic_units,
    semantic_compatibility,
)


def test_normalize_term_is_conservative_and_deterministic() -> None:
    assert normalize_term("  Hippocampal_volume—loss  ") == "hippocampal volume-loss"
    assert normalize_term("APOE ε4") == "apoe ε4"


def test_atomizer_emits_traceable_top_level_components() -> None:
    result = atomize_mention_candidates(
        "CLM_CONCEPT:test",
        "hippocampal atrophy and cognitive decline",
    )
    assert result["has_syntactic_composite"] is True
    assert [item["text"] for item in result["split"]] == [
        "hippocampal atrophy",
        "cognitive decline",
    ]
    assert [(item["start"], item["end"]) for item in result["split"]] == [
        (0, 19),
        (24, 41),
    ]


def test_atomizer_does_not_split_inside_parentheses_or_shared_head_fragment() -> None:
    parenthetical = atomize_mention_candidates(
        "CLM_CONCEPT:parenthetical",
        "attention (visual and auditory)",
    )
    assert parenthetical["split"] == []

    shared_head = atomize_mention_candidates(
        "CLM_CONCEPT:shared-head",
        "left and right hippocampal volume",
    )
    assert shared_head["split"] == []


def test_full_exact_match_has_precedence_over_syntax_split() -> None:
    record = atomize_mention_candidates(
        "CLM_CONCEPT:test",
        "signs and symptoms",
    )
    selected, reason = select_atomic_units(record, full_has_compatible_match=True)
    assert [item["text"] for item in selected] == ["signs and symptoms"]
    assert reason == "full_exact_preferred_over_split"


def test_biomedical_and_lexical_denominator_rules_are_explicit() -> None:
    assert biomedical_context(["disease"], [])[0] is True
    assert biomedical_context(["spatial_reference"], [])[0] is False
    assert biomedical_context([], ["gene_target"])[1] == ("gene",)
    assert is_lexically_eligible("hippocampal atrophy")[0] is True
    assert is_lexically_eligible("results") == (False, "generic_non_entity")


def test_semantic_compatibility_uses_mrsty_names() -> None:
    status, overlap, basis = semantic_compatibility(
        ["disease"],
        [],
        [{"tui": "T047", "sty": "Disease or Syndrome"}],
    )
    assert status == "compatible"
    assert overlap == ("Disease or Syndrome",)
    assert basis == "domain:disease"

    status, _, _ = semantic_compatibility(
        ["disease"],
        [],
        [{"tui": "T121", "sty": "Pharmacologic Substance"}],
    )
    assert status == "incompatible"


def test_destructive_canonicalizer_explicitly_skips_clm_concept() -> None:
    assert "CLM_CONCEPT" in umls_canonicalize.SKIP_PREFIXES
    assert "CLM_CONCEPT" not in umls_canonicalize.ELIGIBLE_PREFIXES


def test_legacy_alignment_and_merge_do_not_fold_clm_mentions() -> None:
    mention = SimpleNamespace(
        preferred_name="Alzheimer disease",
        aliases=[],
        domain_tags=["disease"],
        metadata={"umls_cui": "C0000001"},
    )
    canonical = SimpleNamespace(
        preferred_name="Alzheimer Disease",
        aliases=[],
        domain_tags=["disease"],
        metadata={"umls_cui": "C0000001"},
    )
    kg = SimpleNamespace(
        _index={
            "CLM_CONCEPT:mention": mention,
            "MSH:D000001": canonical,
        }
    )
    names = _collect_graph_names(kg)
    assert names == {"alzheimer disease": ["MSH:D000001"]}
    assert _group_by_cui(kg) == {}


def test_streamed_rrf_lookup_resolves_split_atoms_with_mrsty(tmp_path) -> None:
    mrconso = tmp_path / "MRCONSO.RRF"
    mrsty = tmp_path / "MRSTY.RRF"

    def conso_row(cui: str, code: str, term: str) -> str:
        columns = [
            cui,
            "ENG",
            "P",
            "L1",
            "PF",
            "S1",
            "Y",
            "A1",
            "",
            "",
            "",
            "MSH",
            "MH",
            code,
            term,
            "0",
            "N",
            "",
        ]
        return "|".join(columns) + "\n"

    mrconso.write_text(
        conso_row("C0000001", "D000001", "Alzheimer disease")
        + conso_row("C0000002", "D000002", "Parkinson disease"),
        encoding="utf-8",
    )
    mrsty.write_text(
        "C0000001|T047|B2.2.1.2.1|Disease or Syndrome|||\n"
        "C0000002|T047|B2.2.1.2.1|Disease or Syndrome|||\n",
        encoding="utf-8",
    )

    record = {
        "source_mention_id": "CLM_CONCEPT:composite",
        "domain_tags": ["disease"],
        "atom_types": [],
        "existing_direct_cuis": [],
        "external_code_keys": [],
        **atomize_mention_candidates(
            "CLM_CONCEPT:composite",
            "Alzheimer disease and Parkinson disease",
        ),
    }
    target_terms = {
        variant["normalized"]
        for unit in [record["full"], *record["split"]]
        for variant in unit["variants"]
    }
    term_matches, code_matches, overflow_terms, conso_stats = scan_mrconso(
        mrconso,
        target_terms,
        set(),
        tmp_path / "RUN_STATE.json",
    )
    assert conso_stats["rows_scanned"] == 2
    matched_cuis = {cui for matches in term_matches.values() for cui in matches}
    semantics, sty_stats = scan_mrsty(
        mrsty,
        matched_cuis,
        tmp_path / "RUN_STATE.json",
    )
    assert sty_stats["cuis_with_semantics"] == 2

    full = resolve_unit(
        record,
        record["full"],
        term_matches,
        code_matches,
        semantics,
        overflow_terms,
        include_source_identifiers=True,
    )
    assert full["mappings"] == []
    selected, reason = select_atomic_units(record, full_has_compatible_match=False)
    assert reason == "composite_split_after_full_exact_miss"
    resolved = [
        resolve_unit(
            record,
            unit,
            term_matches,
            code_matches,
            semantics,
            overflow_terms,
            include_source_identifiers=False,
        )
        for unit in selected
    ]
    assert [item["mappings"][0]["cui"] for item in resolved] == [
        "C0000001",
        "C0000002",
    ]
    assert all(
        item["mappings"][0]["method"] == "atomic_surface_normalized_exact"
        for item in resolved
    )
