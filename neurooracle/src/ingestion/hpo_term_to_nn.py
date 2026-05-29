"""HPO term -> NeuroNames region mapping (curation table).

Layer-1 of the GENE -> IM coverage plan. This table is hand-curated from
HPO's anatomical phenotype subtree (HP:0012443 Abnormality of brain
morphology + HP:0002977 Aplasia/Hypoplasia involving CNS), restricted
to high-coverage terms (>=5 genes each, in practice all >=20 genes after
audit pass 2).

Curation principles
-------------------
1. Only HP terms whose phenotype is anatomically localizable are mapped.
   Pure functional terms (seizures, ataxia, autism, ID) are excluded.
2. Microcephaly subterms are mapped to Cerebral_Cortex, not Cerebrum,
   because microcephaly is fundamentally a neurogenesis / cortical
   neural-progenitor defect rather than a whole-telencephalon issue.
3. Cerebrovascular events (stroke, hemorrhage, etc.) map to Cerebrum
   as the broadest reasonable region (the actual lesion site varies).
4. Pituitary / Optic_Nerve buckets are intentionally dropped: NeuroNames
   does not carry direct nodes for them, and these regions are covered
   by the AHBA / GWAS layers instead.
5. The 'Brain' bucket (HP:0002283 Global brain atrophy / HP:0012444
   Brain atrophy) is folded into Cerebrum.

The audit logic that produced this table lives in c:/tmp/hpo_audit*.py
(not committed; see commit message / PR for per-region gene-count tables).
"""
from __future__ import annotations

# Maps HP term ID -> region label. Region labels are resolved to actual
# NN: node IDs at ingest time via HPOAnatomyImporter._resolve_region_targets.
HP_TO_REGION_LABEL: dict[str, str] = {
    # ---- Cerebral_Cortex (microcephaly + cortical malformations) ----
    # Microcephaly: cortical neurogenesis defect, not whole-telencephalon
    "HP:0000252": "Cerebral_Cortex",     # Microcephaly
    "HP:0005484": "Cerebral_Cortex",     # Secondary microcephaly
    "HP:0011451": "Cerebral_Cortex",     # Primary microcephaly
    "HP:0000253": "Cerebral_Cortex",     # Progressive microcephaly
    "HP:0001355": "Cerebral_Cortex",     # Megalencephaly
    "HP:0002120": "Cerebral_Cortex",     # Cerebral cortical atrophy
    "HP:0002126": "Cerebral_Cortex",     # Polymicrogyria
    "HP:0001302": "Cerebral_Cortex",     # Pachygyria
    "HP:0009879": "Cerebral_Cortex",     # Simplified gyral pattern
    "HP:0001339": "Cerebral_Cortex",     # Lissencephaly
    "HP:0002536": "Cerebral_Cortex",     # Abnormal cortical gyration
    "HP:0002539": "Cerebral_Cortex",     # Cortical dysplasia
    "HP:0012650": "Cerebral_Cortex",     # Perisylvian polymicrogyria

    # ---- Cerebrum (whole-telencephalon-scale + cerebrovascular) ----
    "HP:0001360": "Cerebrum",            # Holoprosencephaly
    "HP:0002323": "Cerebrum",            # Anencephaly
    "HP:0006870": "Cerebrum",            # Lobar holoprosencephaly
    "HP:0002059": "Cerebrum",            # Cerebral atrophy
    "HP:0002506": "Cerebrum",            # Diffuse cerebral atrophy
    "HP:0002283": "Cerebrum",            # Global brain atrophy (folded in)
    "HP:0012444": "Cerebrum",            # Brain atrophy (folded in)
    # cerebrovascular -> Cerebrum (location varies, broadest defensible)
    "HP:0001297": "Cerebrum",            # Stroke
    "HP:0002326": "Cerebrum",            # TIA
    "HP:0002140": "Cerebrum",            # Ischemic stroke
    "HP:0002401": "Cerebrum",            # Stroke-like episode
    "HP:0001342": "Cerebrum",            # Cerebral hemorrhage
    "HP:0002170": "Cerebrum",            # Intracranial hemorrhage
    "HP:0002138": "Cerebrum",            # Subarachnoid hemorrhage
    "HP:0100659": "Cerebrum",            # Abnormal cerebral vascular morphology
    "HP:0004944": "Cerebrum",            # Dilatation of cerebral artery
    # septum / forebrain
    "HP:0001331": "Cerebrum",            # Absent septum pellucidum
    "HP:0002389": "Cerebrum",            # Cavum septum pellucidum
    "HP:0002139": "Cerebrum",            # Arrhinencephaly
    "HP:0002132": "Cerebrum",            # Porencephalic cyst
    "HP:0002181": "Cerebrum",            # Cerebral edema
    # pathology markers (no localized HP region available)
    "HP:0100315": "Cerebrum",            # Lewy bodies
    "HP:0002185": "Cerebrum",            # Neurofibrillary tangles
    "HP:0002514": "Cerebrum",            # Cerebral calcification

    # ---- Frontal_Lobe ----
    "HP:0007333": "Frontal_Lobe",        # Hypoplasia of the frontal lobes

    # ---- Corpus_Callosum ----
    "HP:0002079": "Corpus_Callosum",     # Hypoplasia of CC
    "HP:0001274": "Corpus_Callosum",     # Agenesis of CC
    "HP:0033725": "Corpus_Callosum",     # Thin corpus callosum
    "HP:0001273": "Corpus_Callosum",     # Abnormal CC morphology
    "HP:0007370": "Corpus_Callosum",     # Aplasia/Hypoplasia of CC
    "HP:0001338": "Corpus_Callosum",     # Partial agenesis of CC
    "HP:0006989": "Corpus_Callosum",     # Dysplastic CC
    "HP:0007371": "Corpus_Callosum",     # CC atrophy

    # ---- White_Matter (cerebral white matter) ----
    "HP:0002500": "White_Matter",        # Abnormal cerebral WM morphology
    "HP:0002352": "White_Matter",        # Leukoencephalopathy
    "HP:0034295": "White_Matter",        # Reduced cerebral WM volume
    "HP:0002518": "White_Matter",        # Abnormal periventricular WM
    "HP:0002188": "White_Matter",        # Delayed CNS myelination
    "HP:0030890": "White_Matter",        # Hyperintensity of cerebral WM on MRI
    "HP:0006970": "White_Matter",        # Periventricular leukomalacia
    "HP:0007204": "White_Matter",        # Diffuse white matter abnormalities
    "HP:0012762": "White_Matter",        # Cerebral WM atrophy
    "HP:0030891": "White_Matter",        # Periventricular WM hyperintensities

    # ---- Ventricles (ventricular system) ----
    "HP:0002119": "Ventricles",          # Ventriculomegaly
    "HP:0000238": "Ventricles",          # Hydrocephalus
    "HP:0006956": "Ventricles",          # Lateral ventricle dilatation
    "HP:0002198": "Ventricles",          # Dilated 4th ventricle
    "HP:0030048": "Ventricles",          # Colpocephaly
    "HP:0002410": "Ventricles",          # Aqueductal stenosis
    "HP:0001334": "Ventricles",          # Communicating hydrocephalus

    # ---- Cerebellum ----
    "HP:0001272": "Cerebellum",          # Cerebellar atrophy
    "HP:0001321": "Cerebellum",          # Cerebellar hypoplasia
    "HP:0001320": "Cerebellum",          # Cerebellar vermis hypoplasia
    "HP:0001317": "Cerebellum",          # Abnormal cerebellum morphology
    "HP:0006855": "Cerebellum",          # Cerebellar vermis atrophy
    "HP:0001305": "Cerebellum",          # Dandy-Walker malformation
    "HP:0007360": "Cerebellum",          # Aplasia/Hypoplasia of cerebellum
    "HP:0007033": "Cerebellum",          # Cerebellar dysplasia
    "HP:0006817": "Cerebellum",          # Aplasia/Hypoplasia of cerebellar vermis
    "HP:0100275": "Cerebellum",          # Diffuse cerebellar atrophy
    "HP:0002308": "Cerebellum",          # Chiari malformation
    "HP:0007099": "Cerebellum",          # Chiari type I
    "HP:0002350": "Cerebellum",          # Cerebellar cyst

    # ---- Brainstem ----
    "HP:0002365": "Brainstem",           # Hypoplasia of brainstem
    "HP:0012110": "Brainstem",           # Hypoplasia of pons
    "HP:0002363": "Brainstem",           # Abnormal brainstem morphology
    "HP:0007366": "Brainstem",           # Atrophy/Degeneration affecting brainstem
    "HP:0002419": "Brainstem",           # Molar tooth sign on MRI
    "HP:0012748": "Brainstem",           # Focal T2 hyperintense brainstem lesion

    # ---- Basal_Ganglia ----
    "HP:0002135": "Basal_Ganglia",       # Basal ganglia calcification
    "HP:0007183": "Basal_Ganglia",       # Focal T2 hyperintense BG lesion

    # ---- Hypothalamus ----
    "HP:0012285": "Hypothalamus",        # Abnormal hypothalamus physiology
}


# Region label -> candidate names for resolving to NN: nodes via the KG.
# Order: literal label (with underscore->space substitution) tried first,
# then domain-specific synonyms. The first NN node whose preferred_name or
# alias matches any candidate (case-insensitive, with optional underscores)
# is used as the target.
REGION_LABEL_TO_CANDIDATES: dict[str, list[str]] = {
    "Cerebrum":        ["Cerebrum", "Telencephalon"],
    "Cerebral_Cortex": ["Cerebral cortex", "Neocortex"],
    "Frontal_Lobe":    ["Frontal lobe"],
    "Corpus_Callosum": ["Corpus callosum"],
    "White_Matter":    ["Cerebral white matter", "White matter of cerebrum",
                        "White matter"],
    "Ventricles":      ["Ventricular system", "Cerebral ventricles",
                        "Lateral ventricle"],
    "Cerebellum":      ["Cerebellum"],
    "Brainstem":       ["Brainstem", "Brain stem"],
    "Basal_Ganglia":   ["Basal ganglia", "Basal nuclei"],
    "Hypothalamus":    ["Hypothalamus"],
}
