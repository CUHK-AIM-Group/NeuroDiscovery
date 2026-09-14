from neurooracle.src.atoms import Atom
from neurooracle.src.claim_semantics import (
    SEMANTIC_ENDPOINT_IDENTITY_VERSION,
    audit_claim_endpoints,
    declared_type_atoms,
    is_specific_functional_imaging_readout,
    looks_like_cognitive_task_or_stimulus,
    looks_like_concrete_imaging_measurement,
    looks_like_functional_imaging_readout,
    looks_like_imaging_measurement,
    looks_like_non_imaging_assay_entity,
    looks_like_non_task_construct,
    name_compatibility_score,
    semantic_claim_pair,
)
from neurooracle.src.schema import ConceptNode


def _concepts() -> dict[str, ConceptNode]:
    return {
        "REGION:PONS": ConceptNode(
            id="REGION:PONS",
            preferred_name="Pons",
            domain_tags=["neuroanatomy"],
        ),
        "TASK:FEAR": ConceptNode(
            id="TASK:FEAR",
            preferred_name="Fear",
            domain_tags=["emotion"],
        ),
        "DISEASE:PSYCHOSIS": ConceptNode(
            id="DISEASE:PSYCHOSIS",
            preferred_name="Psychosis",
            domain_tags=["disease"],
        ),
        "CUI:BROAD_CORTEX": ConceptNode(
            id="CUI:BROAD_CORTEX",
            preferred_name="Cerebral Cortex",
            aliases=["Cortical"],
            domain_tags=["neuroanatomy"],
        ),
        "CUI:CEREBELLUM": ConceptNode(
            id="CUI:CEREBELLUM",
            preferred_name="Cerebellum",
            aliases=["Cerebellar"],
            domain_tags=["neuroanatomy"],
        ),
        "CUI:CORPUS_CALLOSUM": ConceptNode(
            id="CUI:CORPUS_CALLOSUM",
            preferred_name="Corpus Callosum",
            domain_tags=["neuroanatomy"],
        ),
    }


def test_name_compatibility_is_specific() -> None:
    assert SEMANTIC_ENDPOINT_IDENTITY_VERSION == "semantic-endpoint-identity.v8"
    assert name_compatibility_score("Pons", "pons activation") < 0.70
    assert name_compatibility_score(
        "pons activation", "task-evoked pons activation"
    ) >= 0.70
    assert name_compatibility_score("Pons", "diminished EEG response amplitude") == 0.0
    assert name_compatibility_score("amygdala volume", "amygdala activation") < 0.70
    assert name_compatibility_score(
        "bipolar I disorder", "adolescent bipolar I disorder"
    ) >= 0.70
    assert name_compatibility_score(
        "entorhinal cortical thickness", "Cortical"
    ) < 0.70
    assert name_compatibility_score("ZNF804A variants", "ZNF804A") >= 0.70
    assert name_compatibility_score(
        "medial temporal lobe atrophy", "Temporal lobe"
    ) < 0.70
    assert name_compatibility_score(
        "posterior-cingulate and cerebellar Crus I default-mode network homogeneity elevation",
        "Cerebellar",
    ) == 0.0
    assert name_compatibility_score(
        "corpus callosum circularity", "Corpus Callosum"
    ) == 0.0
    assert name_compatibility_score("Corpus Callosum", "corpus callosum") == 1.0


def test_declared_type_atoms_cover_core_roles() -> None:
    assert declared_type_atoms("IMAGING_MARKER") == {Atom.IMAGING_MARKER}
    assert declared_type_atoms("task_fmri_biomarker") == {Atom.IMAGING_MARKER}
    assert declared_type_atoms("GENETIC_MARKER") == {Atom.GENE_TARGET}
    assert declared_type_atoms("CLINICAL_OUTCOME") == {Atom.OUTCOME}
    assert declared_type_atoms("COGNITIVE_TASK") == {Atom.COGNITIVE_TASK}
    assert declared_type_atoms(
        "",
        "regional task-fMRI reward-anticipation activation patterns",
    ) == {Atom.IMAGING_MARKER}
    assert declared_type_atoms("", "working memory task") == {Atom.COGNITIVE_TASK}
    assert declared_type_atoms(
        "COGNITIVE_TASK", "working-memory performance"
    ) == {Atom.OUTCOME, Atom.INDIVIDUAL_DATA}
    assert declared_type_atoms(
        "COGNITIVE_TASK", "attention deficits"
    ) == {Atom.OUTCOME, Atom.INDIVIDUAL_DATA}


def test_imaging_measurement_detection_separates_task_from_readout() -> None:
    assert looks_like_imaging_measurement("task-fMRI amygdala activation")
    assert looks_like_imaging_measurement("task-induced network entropy")
    assert looks_like_imaging_measurement(
        "hypoactivation of personal and social identity task regions"
    )
    assert looks_like_imaging_measurement(
        "reduced alpha-band fronto-temporal synchronization during working memory"
    )
    assert looks_like_imaging_measurement(
        "altered causal interactions among salience and default-mode networks"
    )
    assert looks_like_imaging_measurement(
        "altered ERP components during memory encoding"
    )
    assert looks_like_imaging_measurement(
        "amygdala responsiveness to anxiety-related stimuli"
    )
    assert not looks_like_imaging_measurement("working memory task")
    assert not looks_like_imaging_measurement("pleasant IAPS stimuli")


def test_non_imaging_assay_detection_excludes_peripheral_physio_readouts() -> None:
    assert looks_like_non_imaging_assay_entity("fetal autonomic brain age score")
    assert looks_like_non_imaging_assay_entity("heart-rate variability")
    assert looks_like_non_imaging_assay_entity("ECG amplitude")
    assert looks_like_non_imaging_assay_entity("magnetocardiography MCG index")
    assert looks_like_non_imaging_assay_entity("CSF aminotransferase activity")
    assert looks_like_non_imaging_assay_entity("serum neurofilament light level")
    assert not looks_like_non_imaging_assay_entity("EEG alpha-band power")
    assert not looks_like_non_imaging_assay_entity("MEG functional connectivity")
    assert not looks_like_non_imaging_assay_entity("cerebral blood flow")
    assert not looks_like_non_imaging_assay_entity("BOLD activity in the amygdala")


def test_task_detection_requires_task_stimulus_or_cognitive_operation() -> None:
    assert looks_like_cognitive_task_or_stimulus("pleasant IAPS stimuli")
    assert looks_like_cognitive_task_or_stimulus(
        "visuo-spatial imagery and episodic memory retrieval"
    )
    assert looks_like_cognitive_task_or_stimulus("working memory task")
    assert looks_like_cognitive_task_or_stimulus("perseverative cognition")
    assert looks_like_cognitive_task_or_stimulus("inhibitory control")
    assert looks_like_cognitive_task_or_stimulus("self-processing")
    assert looks_like_cognitive_task_or_stimulus("stop-signal task")
    assert not looks_like_cognitive_task_or_stimulus("reduced global cognitive performance")
    assert not looks_like_cognitive_task_or_stimulus(
        "altered causal interactions among salience and default-mode networks"
    )


def test_functional_readout_detection_excludes_structural_morphometry() -> None:
    assert looks_like_functional_imaging_readout("left temporal pole connectivity")
    assert looks_like_functional_imaging_readout("precuneus network module size")
    assert looks_like_functional_imaging_readout("right-amygdala hyperactivation")
    assert not looks_like_functional_imaging_readout("medial prefrontal cortex thickness")
    assert not looks_like_functional_imaging_readout("hippocampal volume")
    assert is_specific_functional_imaging_readout("left temporal pole connectivity")
    assert is_specific_functional_imaging_readout("precuneus network module size")
    assert is_specific_functional_imaging_readout(
        "Default Mode Network functional connectivity"
    )
    assert not is_specific_functional_imaging_readout("Default Mode Network")
    assert not is_specific_functional_imaging_readout("salience network")
    assert not is_specific_functional_imaging_readout("functional connectivity")
    assert not is_specific_functional_imaging_readout("task BOLD amplitude")
    assert not is_specific_functional_imaging_readout(
        "amplitude of low-frequency fluctuation"
    )


def test_imaging_detection_normalizes_underscored_feature_names() -> None:
    assert looks_like_imaging_measurement("task-evoked bold_amplitude in ACC")
    assert looks_like_imaging_measurement("theta power_band over frontal sources")
    assert looks_like_imaging_measurement("evoked_potential in anterior cingulate")


def test_non_task_construct_detection_is_task_specific() -> None:
    assert looks_like_non_task_construct("Emotion Regulation Questionnaire")
    assert looks_like_non_task_construct("Positive and Negative Affect Scale")
    assert looks_like_non_task_construct("psychosis")
    assert looks_like_non_task_construct("autism spectrum quotient")
    assert looks_like_non_task_construct("working-memory performance")
    assert looks_like_non_task_construct("attention deficits")
    assert looks_like_non_task_construct("executive-function score")
    assert not looks_like_non_task_construct("Emotion Regulation Task")
    assert not looks_like_non_task_construct("theory of mind task in psychosis")
    assert not looks_like_non_task_construct("anxiety")


def test_claim_endpoint_audit_rejects_name_collision() -> None:
    audit = audit_claim_endpoints(
        {
            "subject_id": "REGION:PONS",
            "subject_name": "diminished EEG response amplitude",
            "subject_type": "ELECTROPHYSIOLOGY_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "negative symptoms across psychosis probands",
            "object_type": "OUTCOME",
        },
        _concepts(),
    )
    assert not audit.valid
    assert audit.reason == "subject_name_mismatch"


def test_claim_endpoint_audit_rejects_atom_collision() -> None:
    audit = audit_claim_endpoints(
        {
            "subject_id": "TASK:FEAR",
            "subject_name": "fear",
            "subject_type": "IMAGING_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        },
        _concepts(),
    )
    assert not audit.valid
    assert audit.reason == "subject_atom_mismatch"


def test_claim_endpoint_audit_splits_readout_from_bare_anatomy() -> None:
    audit = audit_claim_endpoints(
        {
            "subject_id": "REGION:PONS",
            "subject_name": "pons activation",
            "subject_type": "IMAGING_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        },
        _concepts(),
    )
    assert not audit.valid
    assert audit.reason == "subject_name_mismatch"


def test_semantic_projection_keeps_valid_id_and_splits_collision() -> None:
    concepts = _concepts()
    aligned = semantic_claim_pair(
        {
            "subject_id": "REGION:PONS",
            "subject_name": "pons activation",
            "subject_type": "IMAGING_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        },
        concepts,
    )
    collided = semantic_claim_pair(
        {
            "subject_id": "REGION:PONS",
            "subject_name": "diminished EEG response amplitude",
            "subject_type": "ELECTROPHYSIOLOGY_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "negative symptoms across psychosis probands",
            "object_type": "OUTCOME",
        },
        concepts,
    )

    assert aligned is not None and collided is not None
    assert aligned[0].entity_id.startswith("CLAIM_ENTITY:")
    assert not aligned[0].uses_canonical_id
    assert collided[0].entity_id.startswith("CLAIM_ENTITY:")
    assert collided[0].entity_id != "REGION:PONS"
    assert not collided[0].uses_canonical_id


def test_semantic_projection_is_stable_across_wrong_canonical_ids() -> None:
    concepts = _concepts()
    first = semantic_claim_pair(
        {
            "subject_id": "REGION:PONS",
            "subject_name": "diminished EEG response amplitude",
            "subject_type": "ELECTROPHYSIOLOGY_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        },
        concepts,
    )
    second = semantic_claim_pair(
        {
            "subject_id": "TASK:FEAR",
            "subject_name": "diminished EEG response amplitude",
            "subject_type": "ELECTROPHYSIOLOGY_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        },
        concepts,
    )

    assert first is not None and second is not None
    assert first[0].entity_id == second[0].entity_id


def test_semantic_projection_splits_specific_readout_from_broad_cortical_alias() -> None:
    projected = semantic_claim_pair(
        {
            "subject_id": "CUI:BROAD_CORTEX",
            "subject_name": "entorhinal cortical thickness",
            "subject_type": "IMAGING_MARKER",
            "object_id": "DISEASE:PSYCHOSIS",
            "object_name": "psychosis",
            "object_type": "DISEASE",
        },
        _concepts(),
    )

    assert projected is not None
    assert projected[0].entity_id.startswith("CLAIM_ENTITY:")
    assert projected[0].entity_id != "CUI:BROAD_CORTEX"
    assert not projected[0].uses_canonical_id


def test_concrete_imaging_measurement_separates_modality_from_readout() -> None:
    for modality in ("MRI", "FDG-PET", "resting-state fMRI", "SPECT imaging"):
        assert not looks_like_concrete_imaging_measurement(modality)

    for readout in (
        "entorhinal cortical thickness",
        "amyloid PET SUVR",
        "FDG hypometabolism",
        "amygdala functional connectivity",
        "amygdala reactivity",
        "default-mode network homogeneity",
        "corpus callosum circularity",
        "precentral cortical gyrification",
        "brain-predicted age gap",
        "white-matter FA in the cingulum",
        "whole-brain white-matter FA and MD",
        "infant face-processing ERP features",
        "positive cerebral amyloid PET scan",
        "amyloid PET positivity",
        "regional vulnerability index from structural MRI",
        "right amygdala structural connectome node strength",
        "amygdala response to threatening faces",
    ):
        assert looks_like_concrete_imaging_measurement(readout)

    for non_readout in (
        "default mode network",
        "salience network",
        "human cortex",
        "extensive cortical regions",
    ):
        assert not looks_like_concrete_imaging_measurement(non_readout)


def test_semantic_projection_splits_specific_readout_from_broad_anatomy() -> None:
    concepts = _concepts()
    cases = (
        (
            "CUI:CEREBELLUM",
            "posterior-cingulate and cerebellar Crus I default-mode network homogeneity elevation",
        ),
        ("CUI:CORPUS_CALLOSUM", "corpus callosum circularity"),
    )
    for canonical_id, readout_name in cases:
        projected = semantic_claim_pair(
            {
                "subject_id": canonical_id,
                "subject_name": readout_name,
                "subject_type": "IMAGING_MARKER",
                "object_id": "DISEASE:PSYCHOSIS",
                "object_name": "psychosis",
                "object_type": "DISEASE",
            },
            concepts,
        )

        assert projected is not None
        assert projected[0].entity_id.startswith("CLAIM_ENTITY:")
        assert projected[0].entity_id != canonical_id
        assert not projected[0].uses_canonical_id


# Updated: 2026-08-12 13:49:00 HKT - cover modality-versus-readout semantic QA.
# Updated: 2026-08-13 05:40:10 HKT - cover broad cortical alias collisions in semantic endpoint projection.
# Updated: 2026-08-13 05:43:26 HKT - cover the lightweight semantic identity cache contract.
# Updated: 2026-08-13 06:09:34 HKT - cover measurement-versus-anatomy endpoint identity separation.
# Updated: 2026-08-13 16:20:00 HKT - split functional and morphometric readouts from broad anatomy.
