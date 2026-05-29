"""Phase 1.5: Experiment infrastructure ingestion.

Adds static NeuroClaw experiment components into the knowledge graph:
- Brain atlases (Schaefer*, AAL*, Desikan, Destrieux, Glasser, etc.)
- Imaging/data modalities (fMRI, dMRI, sMRI, PET, MEG, EEG, genetics, clinical)
- ML model architectures (BrainGNN, NeuroStorm, BrainLM, SwiFT, 3D-CNN, XGBoost, SVM)
- Datasets (UKB, ADNI, HCP_YA)
- Internal relations: Model supports Modality, Dataset provides Modality

These nodes:
- Are NOT in UMLS — they're engineering/methodological concepts, not biomedical entities
- Are referenced by downstream phases (hypothesis generation, biomarker scan)
- Serve as the static backbone for graph retrieval ("all experiments using Schaefer400")

Run as part of Phase 1 ingestion; align_graph_to_umls() skips these domain tags.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

# ── Atlas registry ─────────────────────────────────────────────────────

SUPPORTED_ATLASES: dict[str, dict] = {
    # Cortical functional parcellations (Schaefer 2018)
    "Schaefer100":  {"n_regions": 100,  "family": "Schaefer", "kind": "cortical_functional",
                     "ref": "Schaefer et al. 2018 Cereb Cortex",
                     "aliases": ["Schaefer 100 parcellation", "Schaefer-100",
                                 "Schaefer 100-region atlas"]},
    "Schaefer200":  {"n_regions": 200,  "family": "Schaefer", "kind": "cortical_functional",
                     "ref": "Schaefer et al. 2018 Cereb Cortex",
                     "aliases": ["Schaefer 200 parcellation", "Schaefer-200"]},
    "Schaefer400":  {"n_regions": 400,  "family": "Schaefer", "kind": "cortical_functional",
                     "ref": "Schaefer et al. 2018 Cereb Cortex",
                     "aliases": ["Schaefer 400 parcellation", "Schaefer-400",
                                 "Schaefer 400-region atlas"]},
    "Schaefer1000": {"n_regions": 1000, "family": "Schaefer", "kind": "cortical_functional",
                     "ref": "Schaefer et al. 2018 Cereb Cortex",
                     "aliases": ["Schaefer 1000 parcellation", "Schaefer-1000"]},
    # Automated Anatomical Labeling
    "AAL90":   {"n_regions": 90,  "family": "AAL", "kind": "anatomical",
                "ref": "Tzourio-Mazoyer et al. 2002 NeuroImage",
                "aliases": ["AAL atlas", "AAL", "AAL 90-region atlas",
                            "Automated Anatomical Labeling atlas"]},
    "AAL116":  {"n_regions": 116, "family": "AAL", "kind": "anatomical",
                "ref": "Tzourio-Mazoyer et al. 2002 NeuroImage",
                "aliases": ["AAL 116 atlas", "AAL-116",
                            "Automated Anatomical Labeling atlas"]},
    # Desikan-Killiany / Destrieux (FreeSurfer)
    "Desikan":   {"n_regions": 68,  "family": "FreeSurfer", "kind": "anatomical",
                  "ref": "Desikan et al. 2006 NeuroImage",
                  "aliases": ["Desikan-Killiany atlas", "Desikan atlas",
                              "Desikan-Killiany cortical atlas",
                              "Desikan-Killiany parcellation"]},
    "Destrieux": {"n_regions": 148, "family": "FreeSurfer", "kind": "anatomical",
                  "ref": "Destrieux et al. 2010 NeuroImage",
                  "aliases": ["Destrieux atlas", "Destrieux parcellation",
                              "a2009s atlas"]},
    # Subcortical
    "HarvardOxford_sub": {"n_regions": 21, "family": "HarvardOxford", "kind": "subcortical",
                          "ref": "Harvard-Oxford subcortical atlas",
                          "aliases": ["Harvard-Oxford subcortical atlas",
                                      "Harvard-Oxford atlas"]},
    # HCP multi-modal parcellation
    "Glasser": {"n_regions": 360, "family": "HCP", "kind": "multimodal",
                "ref": "Glasser et al. 2016 Nature",
                "aliases": ["Glasser atlas", "Glasser multi-modal parcellation",
                            "HCP-MMP1", "HCP multi-modal parcellation",
                            "Glasser 360-region atlas"]},
    # Voxel-level / whole-brain CNN input
    "voxel": {"n_regions": 0, "family": "voxel", "kind": "whole_brain",
              "ref": "whole-brain 3D voxel grid",
              "aliases": ["voxel-level", "whole-brain voxel grid"]},
    # ── EEG electrode layouts (treated as "atlases" for uniform node model) ─
    "EEG_10_20":     {"n_regions": 19, "family": "10-20", "kind": "eeg_layout",
                      "ref": "Jasper 1958 international 10-20 system",
                      "aliases": ["10-20 system", "international 10-20 system"]},
    "EEG_10_10":     {"n_regions": 64, "family": "10-10", "kind": "eeg_layout",
                      "ref": "Chatrian et al. 1988 modified 10-10 system",
                      "aliases": ["10-10 system", "modified 10-10 system"]},
    "EEG_SEED_62":   {"n_regions": 62, "family": "SEED", "kind": "eeg_layout",
                      "ref": "62-channel ESI NeuroScan layout used by SEED/SEED-IV/SEED-V/SEED-VII/SEED-DV",
                      "aliases": ["SEED 62-channel layout", "ESI NeuroScan 62 layout"]},
    "EEG_SEED_VIG_17": {"n_regions": 17, "family": "SEED-VIG", "kind": "eeg_layout",
                        "ref": "17-channel layout used by SEED-VIG (plus 4-ch EOG)",
                        "aliases": ["SEED-VIG 17-channel layout"]},
    "EEG_BCI_32":    {"n_regions": 32, "family": "BCI2000", "kind": "eeg_layout",
                      "ref": "32-channel standard BCI/biosemi layout",
                      "aliases": ["32-channel BCI layout", "Biosemi 32 layout"]},
}

# ── Modality registry ──────────────────────────────────────────────────

CANONICAL_MODALITIES: dict[str, dict] = {
    "fMRI":        {"kind": "functional_imaging",
                    "description": "functional MRI (BOLD-based brain activity)"},
    "dMRI":        {"kind": "diffusion_imaging",
                    "description": "diffusion MRI (white matter microstructure)"},
    "sMRI":        {"kind": "structural_imaging",
                    "description": "structural MRI (T1/T2-weighted anatomical)"},
    "PET":         {"kind": "nuclear_imaging",
                    "description": "positron emission tomography (tracer uptake)"},
    "MEG":         {"kind": "electrophysiology",
                    "description": "magnetoencephalography (neural magnetic fields)"},
    "EEG":         {"kind": "electrophysiology",
                    "description": "electroencephalography (scalp electrical activity)"},
    "EOG":         {"kind": "electrophysiology",
                    "description": "electrooculography (eye-movement potentials)"},
    "eye_tracking":{"kind": "behavior",
                    "description": "gaze position / saccade / fixation / pupillometry"},
    "genetics":    {"kind": "omics",
                    "description": "germline genetic variants (SNPs/WGS)"},
    "clinical":    {"kind": "questionnaire",
                    "description": "clinical questionnaires / diagnosis codes"},
    "environment": {"kind": "questionnaire",
                    "description": "environmental exposures / lifestyle"},
    "physical":    {"kind": "measurement",
                    "description": "anthropometric / physical measurements"},
}

# Modality normalization: common variants → canonical
MODALITY_ALIASES: dict[str, str] = {
    # fMRI family
    "rfmri":                 "fMRI",
    "rs-fmri":               "fMRI",
    "resting-state fmri":    "fMRI",
    "resting state fmri":    "fMRI",
    "tfmri":                 "fMRI",
    "task-fmri":             "fMRI",
    "task fmri":             "fMRI",
    "functional mri":        "fMRI",
    "bold":                  "fMRI",
    # dMRI family
    "dti":                   "dMRI",
    "dwi":                   "dMRI",
    "diffusion mri":         "dMRI",
    "diffusion-weighted":    "dMRI",
    # sMRI family
    "t1":                    "sMRI",
    "t1w":                   "sMRI",
    "t1-weighted":           "sMRI",
    "t2":                    "sMRI",
    "t2w":                   "sMRI",
    "t2-weighted":           "sMRI",
    "flair":                 "sMRI",
    "swi":                   "sMRI",
    "mri":                   "sMRI",
    "structural mri":        "sMRI",
    "anatomical mri":        "sMRI",
    # PET family
    "amyloid pet":           "PET",
    "tau pet":               "PET",
    "fdg pet":               "PET",
    "fdg-pet":               "PET",
    "pib pet":               "PET",
    # identity / canonical self-mappings
    "fmri":                  "fMRI",
    "dmri":                  "dMRI",
    "smri":                  "sMRI",
    "pet":                   "PET",
    "meg":                   "MEG",
    "eeg":                   "EEG",
    # EOG family
    "eog":                   "EOG",
    "electrooculography":    "EOG",
    "electrooculogram":      "EOG",
    # eye tracking family
    "eye tracking":          "eye_tracking",
    "eye-tracking":          "eye_tracking",
    "eye_tracking":          "eye_tracking",
    "eye movement":          "eye_tracking",
    "eye-movement":          "eye_tracking",
    "gaze":                  "eye_tracking",
    "pupillometry":          "eye_tracking",
    "genetics":              "genetics",
    "clinical":              "clinical",
    "environment":           "environment",
    "physical":              "physical",
}


def normalize_modality(mod: str) -> str:
    """Return the canonical modality label for any alias.

    Case-insensitive. Unknown strings are returned unchanged (stripped).
    """
    if not mod:
        return mod
    return MODALITY_ALIASES.get(mod.strip().lower(), mod.strip())


# ── ML model registry ─────────────────────────────────────────────────

ML_MODELS: dict[str, dict] = {
    "BrainGNN": {
        "family": "graph_neural_network",
        "input_level": ["ROI", "connectivity"],
        "modalities":  ["fMRI", "sMRI", "dMRI"],
        "description": "graph neural network operating on parcellated brain graphs",
        "ref": "Li et al. 2021 Med Image Anal",
        "kg_node": True,
    },
    "NeuroStorm": {
        "family": "foundation_model",
        "input_level": ["voxel", "connectivity"],
        "modalities":  ["fMRI"],
        "description": "4D fMRI foundation model (large-scale pretraining)",
        "ref": "NeuroStorm Nat BME 2026",
        # Engineering brand name with no measured paper occurrences -- kept
        # as metadata only so dataset-level specificity scoring can still
        # pick it up without polluting KG traversal with a dead-end node.
        "kg_node": False,
    },
    "BrainLM": {
        "family": "foundation_model",
        "input_level": ["ROI", "connectivity"],
        "modalities":  ["fMRI"],
        "description": "time-series transformer over ROI-level BOLD",
        "ref": "Caro et al. 2024 ICLR",
        "kg_node": False,
    },
    "SwiFT": {
        "family": "vision_transformer",
        "input_level": ["voxel"],
        "modalities":  ["fMRI"],
        "description": "swin transformer for 4D fMRI",
        "ref": "Kim et al. 2023 NeurIPS",
        "kg_node": True,
    },
    "3D-CNN": {
        "family": "convolutional_network",
        "input_level": ["voxel"],
        "modalities":  ["sMRI", "PET"],
        "description": "3D convolutional network for volumetric images",
        "ref": "generic 3D CNN",
        "kg_node": True,
    },
    "XGBoost": {
        "family": "gradient_boosting",
        "input_level": ["ROI", "connectivity", "variable", "channel"],
        "modalities":  ["sMRI", "fMRI", "dMRI", "PET", "EEG", "EOG",
                        "eye_tracking", "genetics", "clinical"],
        "description": "gradient-boosted decision trees on tabular features",
        "ref": "Chen & Guestrin 2016 KDD",
        "kg_node": True,
    },
    "SVM": {
        "family": "kernel_method",
        "input_level": ["ROI", "variable", "channel"],
        "modalities":  ["sMRI", "dMRI", "PET", "EEG", "EOG", "genetics"],
        "description": "support vector machine classifier/regressor",
        "ref": "Cortes & Vapnik 1995",
        "kg_node": True,
    },
    # ── EEG-specific deep learning models ──────────────────────────────
    "EEGNet": {
        "family": "convolutional_network",
        "input_level": ["channel"],
        "modalities":  ["EEG"],
        "description": "compact 2D conv net (depthwise + separable) for EEG classification",
        "ref": "Lawhern et al. 2018 J Neural Eng",
        "kg_node": True,
    },
    "ShallowConvNet": {
        "family": "convolutional_network",
        "input_level": ["channel"],
        "modalities":  ["EEG"],
        "description": "shallow temporal+spatial conv net, inspired by FBCSP",
        "ref": "Schirrmeister et al. 2017 Hum Brain Mapp",
        "kg_node": True,
    },
    "DeepConvNet": {
        "family": "convolutional_network",
        "input_level": ["channel"],
        "modalities":  ["EEG"],
        "description": "deep 5-block conv net for raw EEG classification",
        "ref": "Schirrmeister et al. 2017 Hum Brain Mapp",
        "kg_node": True,
    },
    "EEGConformer": {
        "family": "transformer",
        "input_level": ["channel"],
        "modalities":  ["EEG"],
        "description": "conv + self-attention hybrid for EEG emotion/MI/visual decoding",
        "ref": "Song et al. 2023 IEEE TNSRE",
        "kg_node": False,
    },
    "LaBraM": {
        "family": "foundation_model",
        "input_level": ["channel"],
        "modalities":  ["EEG"],
        "description": "large brain foundation model for EEG (cross-task transfer)",
        "ref": "Jiang et al. 2024 ICLR",
        "kg_node": False,
    },
}

# ── Dataset registry ──────────────────────────────────────────────────
# Mirrors hypothesis_engine.DATASET_FEATURES but with provenance for Phase 1.

DATASETS: dict[str, dict] = {
    "UKB": {
        "full_name": "UK Biobank",
        "n_subjects_imaging": "≈45,000",
        "modalities": ["sMRI", "dMRI", "fMRI", "genetics", "clinical", "environment", "physical"],
        "description": "population cohort with imaging, genetics, ICD-10 outcomes",
        "url": "https://www.ukbiobank.ac.uk/",
    },
    "ADNI": {
        "full_name": "Alzheimer's Disease Neuroimaging Initiative",
        "n_subjects": "≈2,000",
        "modalities": ["sMRI", "PET", "fMRI", "dMRI", "genetics", "clinical"],
        "description": "longitudinal AD continuum cohort (CN/SMC/EMCI/LMCI/AD)",
        "url": "https://adni.loni.usc.edu/",
    },
    "HCP_YA": {
        "full_name": "Human Connectome Project — Young Adult",
        "n_subjects": "≈1,200",
        "modalities": ["sMRI", "fMRI", "dMRI", "MEG"],
        "description": "healthy young adults with high-resolution multi-modal imaging",
        "url": "https://www.humanconnectome.org/study/hcp-young-adult",
    },
    # ── visual decoding (fMRI) ──────────────────────────────────────────
    "NSD": {
        "full_name": "Natural Scenes Dataset",
        "n_subjects": 8,
        "modalities": ["sMRI", "fMRI"],
        "description": "8 subjects × ≈10K natural images, high-density visual task fMRI",
        "url": "https://naturalscenesdataset.org/",
    },
    "BOLD5000": {
        "full_name": "BOLD5000",
        "n_subjects": 4,
        "modalities": ["sMRI", "fMRI"],
        "description": "4 subjects × 5,000 ImageNet/COCO/Scene visual task fMRI",
        "url": "https://bold5000-dataset.github.io/",
    },
    # ── visual decoding (EEG) ───────────────────────────────────────────
    "SEED_DV": {
        "full_name": "SEED-DV (EEG2Video)",
        "n_subjects": 20,
        "modalities": ["EEG"],
        "description": "EEG → dynamic video decoding / reconstruction (NeurIPS 2024)",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/",
    },
    # ── emotion decoding (EEG) ──────────────────────────────────────────
    "SEED": {
        "full_name": "SEED",
        "n_subjects": 15,
        "modalities": ["EEG", "eye_tracking"],
        "description": "3-class emotion (positive/neutral/negative), 15 Chinese film clips",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/seed.html",
    },
    "SEED_IV": {
        "full_name": "SEED-IV",
        "n_subjects": 15,
        "modalities": ["EEG", "eye_tracking"],
        "description": "4-class emotion (happy/sad/fear/neutral), 72 film clips × 3 sessions",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html",
    },
    "SEED_V": {
        "full_name": "SEED-V",
        "n_subjects": 16,
        "modalities": ["EEG", "eye_tracking"],
        "description": "5-class emotion (happy/sad/disgust/neutral/fear) from film clips",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/seed-v.html",
    },
    "SEED_VII": {
        "full_name": "SEED-VII",
        "n_subjects": 20,
        "modalities": ["EEG", "eye_tracking"],
        "description": "7-class emotion + continuous affect labels (IEEE TAFFC 2024)",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/",
    },
    "SEED_GER": {
        "full_name": "SEED-GER",
        "n_subjects": 8,
        "modalities": ["EEG", "eye_tracking"],
        "description": "cross-cultural German cohort, 3-class emotion",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/",
    },
    "SEED_FRA": {
        "full_name": "SEED-FRA",
        "n_subjects": 8,
        "modalities": ["EEG", "eye_tracking"],
        "description": "cross-cultural French cohort, 3-class emotion",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/",
    },
    # ── vigilance decoding (EEG) ────────────────────────────────────────
    "SEED_VIG": {
        "full_name": "SEED-VIG",
        "n_subjects": 23,
        "modalities": ["EEG", "EOG", "eye_tracking"],
        "description": "sustained-attention driving simulator, continuous PERCLOS regression",
        "url": "https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html",
    },
}


# ── Ingestion ─────────────────────────────────────────────────────────

def _ensure(kg: KnowledgeGraph, node: ConceptNode) -> bool:
    """Add node if absent. Returns True if newly created.

    If a same-id node exists it is left untouched (KnowledgeGraph.add_concept
    already does alias/tag union when called with the same id). If a
    *different-id* node has an identical preferred_name (cross-source name
    collision, e.g. seed `Fusiform Face Area` vs an existing claim node of
    the same name), the seed's aliases / domain_tags are merged into that
    pre-existing node instead of creating a duplicate or silently dropping
    the seed.
    """
    if kg.has_concept(node.id):
        return False
    collisions = kg.find_by_name_exact(
        node.preferred_name,
        exclude_source_vocab=node.source_vocab,
    )
    if collisions:
        target = collisions[0]
        seed_meta = dict(node.metadata) if node.metadata else {}
        seed_meta.setdefault("aliased_from_seed", node.id)
        seed_meta.setdefault("aliased_from_source", node.source_vocab)
        kg.merge_seed_into_existing(
            target.id,
            seed_aliases=[node.preferred_name, *node.aliases],
            seed_metadata=seed_meta,
            seed_domain_tags=node.domain_tags,
        )
        logger.debug(
            "name collision: seed %s merged into existing %s",
            node.id, target.id,
        )
        return False
    kg.add_concept(node)
    return True


def _build_atlas_node(name: str, info: dict) -> ConceptNode:
    aliases = list(info.get("aliases", []))
    if name not in aliases:
        aliases.insert(0, name)
    return ConceptNode(
        id=f"ATLAS:{name}",
        preferred_name=name,
        domain_tags=[DomainTag.ATLAS.value],
        source_vocab="experiment_infra",
        aliases=aliases,
        definition=f"Brain parcellation ({info['kind']}): {info['n_regions']} regions. "
                   f"Reference: {info['ref']}",
        metadata={k: info[k] for k in ("n_regions", "family", "kind", "ref")},
    )


def _build_modality_node(name: str, info: dict) -> ConceptNode:
    aliases = sorted({k for k, v in MODALITY_ALIASES.items() if v == name and k != name.lower()})
    return ConceptNode(
        id=f"MODALITY:{name}",
        preferred_name=name,
        domain_tags=[DomainTag.MODALITY.value],
        source_vocab="experiment_infra",
        aliases=aliases,
        definition=info["description"],
        metadata={"kind": info["kind"]},
    )


def _build_model_node(name: str, info: dict) -> ConceptNode:
    return ConceptNode(
        id=f"MODEL:{name}",
        preferred_name=name,
        domain_tags=[DomainTag.ML_MODEL.value],
        source_vocab="experiment_infra",
        definition=f"{info['description']}. Reference: {info['ref']}",
        metadata={
            "family": info["family"],
            "input_level": info["input_level"],
            "modalities":  info["modalities"],
        },
    )


def _build_dataset_node(name: str, info: dict) -> ConceptNode:
    return ConceptNode(
        id=f"DATASET:{name}",
        preferred_name=name,
        domain_tags=[DomainTag.DATASET.value],
        source_vocab="experiment_infra",
        aliases=[info.get("full_name", "")],
        definition=info["description"],
        metadata={k: v for k, v in info.items() if k not in ("description",)},
    )


def ingest_experiment_infrastructure(kg: KnowledgeGraph) -> dict:
    """Ingest atlases, modalities, models, datasets and their cross-relations.

    Idempotent: running twice does not duplicate nodes or edges.

    Returns a stats dict.
    """
    stats = {
        "atlases_added":    0,
        "modalities_added": 0,
        "models_added":     0,
        "datasets_added":   0,
        "edges_added":      0,
    }

    # 1. Atlases
    for name, info in SUPPORTED_ATLASES.items():
        if _ensure(kg, _build_atlas_node(name, info)):
            stats["atlases_added"] += 1

    # 2. Modalities
    for name, info in CANONICAL_MODALITIES.items():
        if _ensure(kg, _build_modality_node(name, info)):
            stats["modalities_added"] += 1

    # 3. ML models (skip those marked kg_node=False -- kept as engineering
    #    metadata only, see ML_MODELS docstring)
    for name, info in ML_MODELS.items():
        if not info.get("kg_node", True):
            continue
        if _ensure(kg, _build_model_node(name, info)):
            stats["models_added"] += 1

    # 4. Datasets
    for name, info in DATASETS.items():
        if _ensure(kg, _build_dataset_node(name, info)):
            stats["datasets_added"] += 1

    # 5. Model --supports_modality--> Modality (only for KG-node-backed models)
    for model, info in ML_MODELS.items():
        if not info.get("kg_node", True):
            continue
        for mod in info["modalities"]:
            mod_id = f"MODALITY:{mod}"
            if not kg.has_concept(mod_id):
                continue
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=f"MODEL:{model}",
                target_id=mod_id,
                relation_type="supports_modality",
                source="experiment_infra",
                confidence=1.0,
            ))
            if kg.G.number_of_edges() > before:
                stats["edges_added"] += 1

    # 6. Dataset --provides_modality--> Modality
    for ds, info in DATASETS.items():
        for mod in info.get("modalities", []):
            mod_id = f"MODALITY:{mod}"
            if not kg.has_concept(mod_id):
                continue
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=f"DATASET:{ds}",
                target_id=mod_id,
                relation_type="provides_modality",
                source="experiment_infra",
                confidence=1.0,
            ))
            if kg.G.number_of_edges() > before:
                stats["edges_added"] += 1

    logger.info(
        "experiment_infra ingest: %d atlases, %d modalities, %d models, %d datasets, %d edges",
        stats["atlases_added"], stats["modalities_added"], stats["models_added"],
        stats["datasets_added"], stats["edges_added"],
    )
    return stats


# ── UMLS-alignment skip predicate ─────────────────────────────────────

#: Domain tags that should be excluded from UMLS MRCONSO alignment.
#: These are engineering/methodological concepts with no UMLS CUI.
UMLS_SKIP_DOMAINS = {
    DomainTag.ATLAS.value,
    DomainTag.MODALITY.value,
    DomainTag.ML_MODEL.value,
    DomainTag.DATASET.value,
    DomainTag.RECIPE.value,
}


def should_skip_umls_alignment(node: ConceptNode) -> bool:
    """Return True if this node should NOT be aligned to UMLS CUIs."""
    if node.source_vocab == "experiment_infra":
        return True
    return any(tag in UMLS_SKIP_DOMAINS for tag in node.domain_tags)
