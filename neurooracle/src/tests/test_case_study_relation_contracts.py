from __future__ import annotations

from neurooracle.src.atoms import Atom
from neurooracle.src.case_study_relation_contracts import (
    case_study_endpoint_names_allowed,
    case_study_endpoint_atom_routes,
    case_study_pair_allowed,
    endpoint_matches_atom,
    requires_complete_path,
)


def test_endpoint_atom_routes_include_contract_admitted_outcomes() -> None:
    assert case_study_endpoint_atom_routes(
        "progression_prediction", (Atom.IMAGING_MARKER,)
    ) == (
        (Atom.IMAGING_MARKER, Atom.DISEASE),
        (Atom.IMAGING_MARKER, Atom.OUTCOME),
    )
    assert case_study_endpoint_atom_routes(
        "connectome_behavior", (Atom.IMAGING_MARKER,)
    ) == (
        (Atom.IMAGING_MARKER, Atom.INDIVIDUAL_DATA),
        (Atom.IMAGING_MARKER, Atom.OUTCOME),
    )
from neurooracle.src.claim_semantics import SemanticEndpoint


def _endpoint(
    entity_id: str,
    name: str,
    *atoms: str,
) -> SemanticEndpoint:
    return SemanticEndpoint(
        entity_id=entity_id,
        name=name,
        canonical_id=entity_id,
        atoms=tuple(atoms),
        uses_canonical_id=True,
        name_score=1.0,
        role_compatible=True,
    )


def test_adverse_event_requires_a_drug_not_a_stimulation_procedure() -> None:
    outcome = _endpoint("OUTCOME:HEADACHE", "headache", "outcome")
    tms = _endpoint("INTERVENTION:TMS", "TMS", "drug")
    fluoxetine = _endpoint("DRUG:FLUOXETINE", "fluoxetine", "drug")
    claim = {"predicate": "causes"}

    assert not case_study_pair_allowed(
        claim, {}, (tms, outcome), "adverse_event_prediction"
    )
    assert case_study_pair_allowed(
        claim, {}, (fluoxetine, outcome), "adverse_event_prediction"
    )
    ziprasidone = _endpoint("CLM_CONCEPT:ZIPRASIDONE", "ziprasidone", "drug")
    assert case_study_pair_allowed(
        claim, {}, (ziprasidone, outcome), "adverse_event_prediction"
    )

    mistyped_risk = _endpoint(
        "RISK:BONE_DENSITY",
        "higher risk of osteoporotic bone mineral density in women",
        "drug",
        "outcome",
    )
    assert not case_study_pair_allowed(
        claim, {}, (mistyped_risk, outcome), "adverse_event_prediction"
    )
    hyponatremia = _endpoint(
        "CLM_CONCEPT:HYPONATREMIA", "severe hyponatremia", "drug"
    )
    assert not case_study_pair_allowed(
        claim, {}, (hyponatremia, outcome), "adverse_event_prediction"
    )


def test_adverse_event_requires_an_adverse_outcome() -> None:
    fluoxetine = _endpoint("DRUG:FLUOXETINE", "fluoxetine", "drug")
    metabolism = _endpoint(
        "OUTCOME:PET", "altered PET cerebral glucose metabolism", "outcome"
    )
    assert not case_study_pair_allowed(
        {"predicate": "is_associated_with"},
        {},
        (fluoxetine, metabolism),
        "adverse_event_prediction",
    )


def test_imaging_genetics_requires_a_real_imaging_readout() -> None:
    gene = _endpoint("GENE:APOE", "APOE", "gene_target")
    immunoreactivity = _endpoint(
        "ASSAY:TRPV4", "TRPV4 immunoreactivity", "imaging_marker"
    )
    hippocampal_volume = _endpoint(
        "IM:HIPPOCAMPAL_VOLUME", "hippocampal volume", "imaging_marker"
    )
    concepts = {
        "IM:HIPPOCAMPAL_VOLUME": {"domain_tags": ["imaging_feature"]},
    }
    claim = {"predicate": "predicts"}

    assert not case_study_pair_allowed(
        claim, concepts, (gene, immunoreactivity), "imaging_genetics"
    )
    assert case_study_pair_allowed(
        claim, concepts, (gene, hippocampal_volume), "imaging_genetics"
    )


def test_imaging_genetics_requires_a_specific_genetic_endpoint() -> None:
    imaging = _endpoint(
        "IM:WMH", "regional white matter hyperintensity volume", "imaging_marker"
    )
    mistyped_clinical = _endpoint(
        "CLM_CONCEPT:CLINICAL_PHENOMENA",
        "clinical and pharmacological phenomena in mood disorders",
        "gene_target",
    )
    variant = _endpoint(
        "CLM_CONCEPT:BDNF_VAL66MET", "BDNF Val66Met polymorphism", "gene_target"
    )
    mistyped_neurodegeneration = _endpoint(
        "CLM_CONCEPT:MCI_PD",
        "neurodegeneration in MCI and early Parkinson disease",
        "gene_target",
    )
    concepts = {"IM:WMH": {"domain_tags": ["imaging_feature"]}}
    claim = {"predicate": "modulates"}

    assert not case_study_pair_allowed(
        claim, concepts, (mistyped_clinical, imaging), "imaging_genetics"
    )
    assert case_study_pair_allowed(
        claim, concepts, (variant, imaging), "imaging_genetics"
    )
    assert not case_study_pair_allowed(
        claim, concepts, (mistyped_neurodegeneration, imaging), "imaging_genetics"
    )


def test_functional_localization_is_directed_task_to_neural_readout() -> None:
    task = _endpoint("TASK:NBACK", "n-back task", "cognitive_task")
    activation = _endpoint(
        "IM:DLPFC", "DLPFC activation", "imaging_marker"
    )
    concepts = {"IM:DLPFC": {"domain_tags": ["neuroanatomy"]}}
    claim = {"predicate": "activates"}

    assert case_study_pair_allowed(
        claim, concepts, (task, activation), "functional_localization"
    )
    assert not case_study_pair_allowed(
        claim, concepts, (activation, task), "functional_localization"
    )


def test_generation_support_edges_do_not_become_future_endpoint_relations() -> None:
    first = _endpoint("IM:DLPFC", "DLPFC activation", "imaging_marker")
    second = _endpoint("IM:AMYGDALA", "amygdala activation", "imaging_marker")
    concepts = {
        "IM:DLPFC": {"domain_tags": ["neuroanatomy"]},
        "IM:AMYGDALA": {"domain_tags": ["neuroanatomy"]},
    }
    claim = {"predicate": "is_associated_with"}

    assert not case_study_pair_allowed(
        claim, concepts, (first, second), "functional_localization"
    )
    assert case_study_pair_allowed(
        claim,
        concepts,
        (first, second),
        "functional_localization",
        include_support=True,
    )


def test_prognosis_rejects_an_imaging_network_as_the_outcome() -> None:
    disease = _endpoint("D:MDD", "major depressive disorder", "disease")
    network = _endpoint(
        "OUTCOME:NETWORK",
        "working-memory and thalamocortical networks",
        "outcome",
    )
    assert not case_study_pair_allowed(
        {"predicate": "predicts"}, {}, (disease, network), "prognosis"
    )


def test_prognosis_accepts_specific_longitudinal_clinical_changes() -> None:
    thickness = _endpoint(
        "IM:PFC_THICKNESS",
        "right prefrontal cortex cortical thickness",
        "imaging_marker",
    )
    response = _endpoint(
        "OUTCOME:CBT_RESPONSE",
        "cognitive-behavioral therapy response",
        "outcome",
    )
    symptoms = _endpoint(
        "OUTCOME:SYMPTOM_CHANGE",
        "future PTSD and depression symptom severity",
        "outcome",
    )
    validation = _endpoint(
        "OUTCOME:MODEL_VALIDATION",
        "treatment-response biomarker validation",
        "outcome",
    )

    assert case_study_pair_allowed(
        {"predicate": "predicts"}, {}, (thickness, response), "prognosis"
    )
    assert case_study_pair_allowed(
        {"predicate": "predicts"}, {}, (thickness, symptoms), "prognosis"
    )
    assert not case_study_pair_allowed(
        {"predicate": "predicts"}, {}, (thickness, validation), "prognosis"
    )


def test_connectome_behavior_accepts_phenotype_but_not_bare_network() -> None:
    bare_network = _endpoint(
        "IM:DMN", "default mode network", "imaging_marker"
    )
    connectivity = _endpoint(
        "IM:DMN_FC", "default mode network functional connectivity", "imaging_marker"
    )
    performance = _endpoint(
        "OUTCOME:MEMORY", "memory performance", "individual_data", "outcome"
    )

    assert not case_study_pair_allowed(
        {"predicate": "correlates_with"},
        {},
        (bare_network, performance),
        "connectome_behavior",
    )
    assert case_study_pair_allowed(
        {"predicate": "correlates_with"},
        {},
        (connectivity, performance),
        "connectome_behavior",
    )


def test_disease_atom_rejects_prognosis_phrases_as_baseline_diagnoses() -> None:
    concepts = {
        "CUI:C0036341": {"domain_tags": ["disease"]},
        "CUI:C0033975": {"domain_tags": ["disease"]},
    }
    schizophrenia = _endpoint("CUI:C0036341", "schizophrenia", "disease")
    transition = _endpoint(
        "CUI:C0033975", "transition to psychosis", "disease", "outcome"
    )
    progression = _endpoint(
        "CLM_CONCEPT:PROGRESSION",
        "clinical progression in subjective cognitive decline",
        "disease",
        "outcome",
    )
    rapid_progression = _endpoint(
        "CLM_CONCEPT:RAPID_PROGRESSION",
        "rapid progression from mild cognitive impairment to Alzheimer dementia",
        "disease",
        "outcome",
    )
    risk = _endpoint(
        "CLM_CONCEPT:RISK",
        "long-term cerebrovascular disease and dementia risk",
        "disease",
        "outcome",
    )
    parkinson_dementia = _endpoint(
        "CLM_CONCEPT:PDD", "Parkinson disease dementia", "disease"
    )
    treatment_resistant = _endpoint(
        "CLM_CONCEPT:TRS", "treatment-resistant schizophrenia", "disease"
    )
    efficacy_response = _endpoint(
        "CLM_CONCEPT:EFFICACY_RESPONSE",
        "a positive rosiglitazone efficacy response in Alzheimer disease patients",
        "disease",
        "outcome",
    )
    dementia_development = _endpoint(
        "CLM_CONCEPT:DEMENTIA_DEVELOPMENT",
        "dementia development",
        "disease",
        "outcome",
    )
    burden = _endpoint(
        "CLM_CONCEPT:BURDEN",
        "high humanistic burden including cognitive impairment",
        "disease",
        "outcome",
    )
    comparison = _endpoint(
        "CLM_CONCEPT:COMPARISON",
        "schizophrenia than schizoaffective disorder",
        "disease",
    )
    generic_disorder = _endpoint(
        "CLM_CONCEPT:GENERIC",
        "a lifelong neurodevelopmental disorder",
        "disease",
    )

    assert endpoint_matches_atom(schizophrenia, Atom.DISEASE, concepts)
    assert endpoint_matches_atom(parkinson_dementia, Atom.DISEASE, concepts)
    assert endpoint_matches_atom(treatment_resistant, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(transition, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(progression, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(rapid_progression, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(risk, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(efficacy_response, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(dementia_development, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(burden, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(comparison, Atom.DISEASE, concepts)
    assert not endpoint_matches_atom(generic_disorder, Atom.DISEASE, concepts)


def test_imaging_atom_rejects_an_algorithm_that_mentions_mri() -> None:
    method = _endpoint(
        "METHOD:SVM",
        "PSO-based least square support vector machine with MRI biomarkers",
        "imaging_marker",
    )
    measurement = _endpoint(
        "IM:HIPPOCAMPUS",
        "hippocampal MRI volume",
        "imaging_marker",
    )
    concepts = {
        "METHOD:SVM": {"domain_tags": ["imaging_feature"]},
        "IM:HIPPOCAMPUS": {"domain_tags": ["imaging_feature"]},
    }

    assert not endpoint_matches_atom(method, Atom.IMAGING_MARKER, concepts)
    assert endpoint_matches_atom(measurement, Atom.IMAGING_MARKER, concepts)


def test_imaging_atom_rejects_plural_algorithms_and_mvpa_pipelines() -> None:
    plural_algorithm = _endpoint(
        "METHOD:FMRI_ALGORITHMS",
        "functional MRI classification algorithms",
        "imaging_marker",
    )
    mvpa_pipeline = _endpoint(
        "METHOD:MVPA",
        "multi-biomarker resting-state fMRI MVPA with extreme learning machine",
        "imaging_marker",
    )
    connectivity = _endpoint(
        "IM:CONNECTIVITY",
        "resting-state fMRI functional connectivity features",
        "imaging_marker",
    )

    assert not endpoint_matches_atom(plural_algorithm, Atom.IMAGING_MARKER, {})
    assert not endpoint_matches_atom(mvpa_pipeline, Atom.IMAGING_MARKER, {})
    assert endpoint_matches_atom(connectivity, Atom.IMAGING_MARKER, {})


def test_imaging_atom_rejects_a_molecular_assay_but_keeps_pet_binding() -> None:
    kinase = _endpoint(
        "ASSAY:PI3K",
        "spinal cord PI 3-kinase activity and protein level",
        "imaging_marker",
    )
    pet_binding = _endpoint(
        "IM:DAT",
        "dopamine transporter binding potential",
        "imaging_marker",
    )

    assert not endpoint_matches_atom(kinase, Atom.IMAGING_MARKER, {})
    assert endpoint_matches_atom(pet_binding, Atom.IMAGING_MARKER, {})


def test_cognitive_decoding_rejects_a_non_task_construct() -> None:
    activation = _endpoint(
        "IM:VISUAL", "visual cortex activation", "imaging_marker"
    )
    task = _endpoint("TASK:NBACK", "n-back task", "cognitive_task")
    ego = _endpoint("CONCEPT:EGO", "ego", "cognitive_task")
    concepts = {"IM:VISUAL": {"domain_tags": ["neuroanatomy"]}}
    claim = {"predicate": "predicts"}

    assert case_study_pair_allowed(
        claim, concepts, (activation, task), "cognitive_decoding"
    )
    assert not case_study_pair_allowed(
        claim, concepts, (activation, ego), "cognitive_decoding"
    )


def test_brain_age_requires_an_age_endpoint() -> None:
    thickness = _endpoint(
        "IM:THICKNESS", "cortical thickness", "imaging_marker"
    )
    age = _endpoint("OUTCOME:AGE", "predicted brain age", "individual_data")
    sex = _endpoint("OUTCOME:SEX", "biological sex", "individual_data")
    concepts = {"IM:THICKNESS": {"domain_tags": ["imaging_feature"]}}
    claim = {"predicate": "predicts"}

    assert case_study_pair_allowed(
        claim, concepts, (thickness, age), "brain_age"
    )
    assert not case_study_pair_allowed(
        claim, concepts, (thickness, sex), "brain_age"
    )


def test_brain_age_rejects_method_and_non_imaging_inputs() -> None:
    model = _endpoint(
        "CLM_CONCEPT:CNN",
        "ten-layer 3D convolutional neural network",
        "imaging_marker",
    )
    parity = _endpoint(
        "CLM_CONCEPT:PARITY",
        "higher parity",
        "imaging_marker",
    )
    thickness = _endpoint(
        "IM:THICKNESS",
        "cortical thickness",
        "imaging_marker",
    )
    age = _endpoint("OUTCOME:AGE", "predicted brain age", "individual_data")
    claim = {"predicate": "predicts"}

    assert not case_study_pair_allowed(claim, {}, (model, age), "brain_age")
    assert not case_study_pair_allowed(claim, {}, (parity, age), "brain_age")
    assert case_study_pair_allowed(claim, {}, (thickness, age), "brain_age")


def test_brain_age_rejects_brain_age_as_the_imaging_predictor_of_an_outcome() -> None:
    predicted_brain_age = _endpoint(
        "CLM_CONCEPT:BRAIN_AGE",
        "predicted brain age",
        "imaging_marker",
    )
    stroke_risk = _endpoint(
        "CLM_CONCEPT:STROKE_RISK",
        "stroke risk",
        "individual_data",
    )

    assert not case_study_pair_allowed(
        {"predicate": "predicts"},
        {},
        (predicted_brain_age, stroke_risk),
        "brain_age",
    )


def test_feature_cases_reject_bare_modalities_and_fully_convolutional_networks() -> None:
    age = _endpoint("OUTCOME:AGE", "predicted brain age", "individual_data")
    for name in ("FDG-PET", "MRI", "Simple Fully Convolutional Network"):
        endpoint = _endpoint(f"CLM_CONCEPT:{name}", name, "imaging_marker")
        assert not case_study_pair_allowed(
            {"predicate": "predicts"}, {}, (endpoint, age), "brain_age"
        )

    suvr = _endpoint("IM:SUVR", "amyloid PET SUVR", "imaging_marker")
    assert case_study_pair_allowed(
        {"predicate": "predicts"}, {}, (suvr, age), "brain_age"
    )


def test_case1_requires_a_concrete_imaging_readout_not_a_bare_region() -> None:
    disease = _endpoint("D:MDD", "major depressive disorder", "disease")
    bare_region = _endpoint(
        "CUI:NUCLEUS_ACCUMBENS", "Nucleus Accumbens", "imaging_marker"
    )
    connectivity = _endpoint(
        "IM:NACC_CONNECTIVITY",
        "nucleus accumbens functional connectivity",
        "imaging_marker",
    )

    for scope in ("case1_transdiagnostic", "transdiagnostic_clustering"):
        assert not case_study_endpoint_names_allowed(
            scope, bare_region, disease
        )
        assert case_study_endpoint_names_allowed(
            scope, connectivity, disease
        )
    assert not case_study_pair_allowed(
        {"predicate": "correlates_with"},
        {},
        (bare_region, disease),
        "case1_transdiagnostic",
    )
    assert case_study_pair_allowed(
        {"predicate": "correlates_with"},
        {},
        (connectivity, disease),
        "case1_transdiagnostic",
    )


def test_mechanism_tasks_require_complete_path_support() -> None:
    for case_study_id in (
        "case2_pathway_mediation",
        "disease_subtyping",
        "drug_response_prediction",
        "neuromodulation_target",
        "personalised_treatment",
        "prognosis",
    ):
        assert requires_complete_path(case_study_id)

    assert not requires_complete_path("biomarker_discovery")
    assert not requires_complete_path("functional_localization")


# Updated: 2026-08-12 16:02:00 HKT - cover contract-derived routes and imaging endpoint guards.
# Updated: 2026-08-12 21:37 HKT - reject outcome prediction mislabeled as brain-age estimation.
# Updated: 2026-08-13 16:40 HKT - require executable imaging readouts for Case Study 1.
