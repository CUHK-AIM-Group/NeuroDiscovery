"""Tests for IM (imaging marker) brainstorming."""

from __future__ import annotations

import json

from neurooracle.src.recipe import (
    ImagingMarker,
    build_im_palette,
    brainstorm_ims,
    validate_ims,
    link_ims_to_kg,
    tag_atoms,
)


def _concept(cid: str, name: str, domain: str,
             aliases=None, source_vocab: str = "",
             metadata: dict | None = None) -> tuple[str, dict]:
    return cid, {
        "id": cid,
        "preferred_name": name,
        "domain_tags": [domain],
        "aliases": aliases or [],
        "source_vocab": source_vocab,
        "metadata": metadata or {},
    }


def _stub_kg() -> dict:
    """Tiny KG with all the slots IMPalette pulls from."""
    concepts = dict([
        # Modalities
        _concept("MODALITY:sMRI", "sMRI", "modality"),
        _concept("MODALITY:fMRI", "fMRI", "modality"),
        _concept("MODALITY:dMRI", "dMRI", "modality"),
        _concept("MODALITY:PET",  "PET",  "modality"),
        _concept("MODALITY:EEG",  "EEG",  "modality"),
        _concept("MODALITY:MEG",  "MEG",  "modality"),
        _concept("MODALITY:genetics", "genetics", "modality"),  # filtered out
        # IF operations
        _concept("IF:cortical_thickness", "cortical thickness",
                  "imaging_feature", metadata={"modality": "sMRI"}),
        _concept("IF:regional_volume", "regional volume",
                  "imaging_feature", metadata={"modality": "sMRI"}),
        _concept("IF:fractional_anisotropy", "fractional anisotropy",
                  "imaging_feature", metadata={"modality": "dMRI"}),
        _concept("IF:functional_connectivity", "functional connectivity",
                  "imaging_feature", metadata={"modality": "fMRI"}),
        _concept("IF:bold_amplitude", "task BOLD amplitude",
                  "imaging_feature", metadata={"modality": "fMRI"}),
        _concept("IF:amyloid_suvr", "amyloid SUVR",
                  "imaging_feature", metadata={"modality": "PET"}),
        # Regions
        _concept("NN:1", "Hippocampus", "neuroanatomy"),
        _concept("NN:2", "Entorhinal Cortex", "neuroanatomy"),
        _concept("NN:3", "Anterior Cingulate Cortex", "neuroanatomy",
                 aliases=["ACC"]),
        _concept("NN:4", "Cerebral Cortex", "neuroanatomy"),  # filtered as container
        _concept("VROI:FFA", "Fusiform Face Area (FFA, visual ROI)",
                 "neuroanatomy"),  # VROI prefix routes to visual_rois
        # Tasks / concepts
        _concept("COGAT_TASK:t1", "n-back task", "paradigm",
                 source_vocab="CognitiveAtlas"),
        _concept("COGAT_TASK:t2", "stop-signal task", "paradigm",
                 source_vocab="CognitiveAtlas"),
        _concept("COGAT_CONCEPT:c1", "working memory", "cognitive_function",
                 source_vocab="CognitiveAtlas"),
        # MSH-tagged 'cognitive_function' should NOT enter palette concepts
        _concept("MSH:bad", "Pain", "cognitive_function", source_vocab="MeSH"),
    ])
    edges = [
        # give some edge degree so degree-sort returns deterministic order
        {"source_id": "NN:1", "target_id": "NN:2", "relation_type": "part_of"},
        {"source_id": "NN:1", "target_id": "NN:3", "relation_type": "part_of"},
        {"source_id": "NN:2", "target_id": "NN:3", "relation_type": "part_of"},
        {"source_id": "COGAT_TASK:t1", "target_id": "NN:1",
         "relation_type": "activates"},
    ]
    return {"concepts": concepts, "edges": edges}


def test_palette_keeps_only_imaging_modalities():
    kg = _stub_kg()
    palette = build_im_palette(kg["concepts"], edges=kg["edges"])
    mod_names = {m["name"] for m in palette.modalities}
    assert mod_names == {"sMRI", "fMRI", "dMRI", "PET", "EEG", "MEG"}
    assert "genetics" not in mod_names


def test_palette_filters_container_regions():
    kg = _stub_kg()
    palette = build_im_palette(kg["concepts"], edges=kg["edges"])
    core_names = {r["name"] for r in palette.core_regions}
    assert "Cerebral Cortex" not in core_names
    assert "Hippocampus" in core_names
    # VROI routes to a separate slot
    vroi_names = {r["name"] for r in palette.visual_rois}
    assert any("FFA" in n for n in vroi_names)


def test_palette_concepts_only_from_cognitive_atlas():
    kg = _stub_kg()
    palette = build_im_palette(kg["concepts"], edges=kg["edges"])
    concept_names = {c["name"] for c in palette.concepts}
    assert "working memory" in concept_names
    assert "Pain" not in concept_names  # MSH-tagged, must be excluded


def test_brainstorm_ims_parses_schema():
    palette = build_im_palette(*_stub_kg().values()) if False else build_im_palette(
        _stub_kg()["concepts"], edges=_stub_kg()["edges"])
    payload = [
        {"name": "hipp_volume",
         "family": "univariate",
         "modality": "sMRI",
         "operation": "regional_volume",
         "regions": ["Hippocampus"],
         "conditioning": None,
         "formula": "regional_volume(Hippocampus) on sMRI",
         "rationale": "Hippocampal atrophy is the canonical AD imaging biomarker."},
        {"name": "ffa_face_response",
         "family": "task_evoked",
         "modality": "fMRI",
         "operation": "bold_amplitude",
         "regions": ["Fusiform Face Area (FFA, visual ROI)"],
         "conditioning": {"task": "n-back task", "concept": ""},
         "formula": "GLM beta in FFA during n-back",
         "rationale": "Face-selective response in FFA."},
    ]

    def stub(prompt: str, sys: str) -> str:
        # Spec: prompt must mention modality & operation slots
        assert "Modalities:" in prompt
        assert "Operations" in prompt
        return json.dumps(payload)

    raw = brainstorm_ims(palette, n=2, llm_call=stub, model_name="stub-1")
    assert len(raw) == 2
    assert raw[0].operation == "regional_volume"
    assert raw[1].family == "task_evoked"


def test_validate_rejects_modality_op_mismatch():
    palette = build_im_palette(_stub_kg()["concepts"], edges=_stub_kg()["edges"])
    bad = ImagingMarker(
        id="im_0001", name="bad", family="univariate",
        modality="sMRI", operation="fractional_anisotropy",
        region_names=["Hippocampus"],
    )
    report = validate_ims([bad], palette)
    assert report.n_accepted == 0
    assert "operation_modality_incompatible" in report.reject_reasons()


def test_validate_rejects_unresolvable_region():
    palette = build_im_palette(_stub_kg()["concepts"], edges=_stub_kg()["edges"])
    bad = ImagingMarker(
        id="im_0001", name="bad", family="univariate",
        modality="sMRI", operation="cortical_thickness",
        region_names=["Made-Up Region"],
    )
    report = validate_ims([bad], palette)
    assert report.n_accepted == 0
    assert "region_resolution_failed" in report.reject_reasons()


def test_validate_rejects_task_evoked_without_conditioning():
    palette = build_im_palette(_stub_kg()["concepts"], edges=_stub_kg()["edges"])
    bad = ImagingMarker(
        id="im_0001", name="bad", family="task_evoked",
        modality="fMRI", operation="bold_amplitude",
        region_names=["Hippocampus"],
        conditioning=None,
    )
    report = validate_ims([bad], palette)
    assert report.n_accepted == 0
    assert "conditioning_missing_or_unknown" in report.reject_reasons()


def test_validate_rejects_ratio_with_one_region():
    palette = build_im_palette(_stub_kg()["concepts"], edges=_stub_kg()["edges"])
    bad = ImagingMarker(
        id="im_0001", name="bad", family="ratio",
        modality="sMRI", operation="regional_volume",
        region_names=["Hippocampus"],
    )
    report = validate_ims([bad], palette)
    assert report.n_accepted == 0
    # Either count or resolution can trip — we just need rejection.
    assert any(r.startswith("region_count<") for r in report.reject_reasons())


def test_validate_accepts_well_formed_marker():
    palette = build_im_palette(_stub_kg()["concepts"], edges=_stub_kg()["edges"])
    good = ImagingMarker(
        id="im_0001", name="hipp_vol", family="univariate",
        modality="sMRI", operation="regional_volume",
        region_names=["Hippocampus"],
    )
    report = validate_ims([good], palette)
    assert report.n_accepted == 1
    assert report.accepted[0].regions == ["NN:1"]


def test_link_and_tag_atoms():
    palette = build_im_palette(_stub_kg()["concepts"], edges=_stub_kg()["edges"])
    a = ImagingMarker(
        id="im_0001", name="hipp_vol", family="univariate",
        modality="sMRI", operation="regional_volume",
        region_names=["Hippocampus"],
    )
    b = ImagingMarker(
        id="im_0002", name="ffa_faces", family="task_evoked",
        modality="fMRI", operation="bold_amplitude",
        region_names=["Fusiform Face Area (FFA, visual ROI)"],
        conditioning={"task": "n-back task"},
    )
    report = validate_ims([a, b], palette)
    assert report.n_accepted == 2
    link_ims_to_kg(report.accepted, palette)
    tag_atoms(report.accepted)
    a_, b_ = report.accepted
    assert a_.modality_id == "MODALITY:sMRI"
    assert a_.operation_id == "IF:regional_volume"
    assert a_.atoms == ["IMAGING_MARKER"]
    assert "COGNITIVE_TASK" in b_.atoms
