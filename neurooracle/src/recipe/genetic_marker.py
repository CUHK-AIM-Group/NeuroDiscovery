"""GM (genetic marker) brainstorming: structured composition over KG primitives.

Phase 1 catalogue of genetic-derived markers, each one a scalar (or vector)
quantity computable from one subject's genetic / transcriptomic / epigenetic
data. Composition uses a fixed palette:

  * data types          (genotype_array / wgs / wes / rnaseq /
                         methylation_array / mtdna_seq)
  * operations          (PRS, rare-variant burden, expression aggregate,
                         methylation clock, imputed expression / TWAS,
                         mtDNA copy / heteroplasmy, ...)
  * curated gene sets   (AD_risk / PD_risk / Synaptic / Dopaminergic / ...)
  * single-gene pool    (top neuropsych genes by KG degree)
  * tissue palette      (GTEx brain v9 + whole blood)
  * disease GWAS sources (top neuropsych diseases by gene-edge count)
  * methylation clocks  (Horvath / Hannum / PhenoAge / GrimAge / DunedinPACE)

Mirrors imaging_marker.py: palette -> brainstorm -> validate -> link -> tag.
Atoms by construction: every passing GM is GENE_TARGET (+ DISEASE when the
GM is built off a specific disease GWAS, e.g. polygenic_risk).

Output is genetic_markers.json. Markers are NOT injected into the KG;
disease/anatomy -> GM edges come from Phase 2 paper extraction.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# == GM families and operation/data-type compatibility ======================

GM_FAMILIES: tuple[str, ...] = (
    "polygenic_risk",
    "single_locus",
    "mutation_burden",
    "expression_single",
    "expression_aggregate",
    "imputed_expression",
    "methylation_clock",
    "methylation_aggregate",
    "mtdna_metric",
    "cross_tissue_expression",
)
# Operation slug -> {valid data_type}. Data types are abstractions over the
# raw assay; the LLM picks both. validate_gms enforces compatibility.
OP_TO_DATA_TYPES: dict[str, frozenset[str]] = {
    # PRS-style: needs whole-genome variants + GWAS sumstats
    "polygenic_risk":           frozenset({"genotype_array", "wgs", "wes"}),
    "partition_prs":            frozenset({"genotype_array", "wgs", "wes"}),
    "carrier_status":           frozenset({"genotype_array", "wgs", "wes"}),
    "genotype_count":           frozenset({"genotype_array", "wgs", "wes"}),
    # Rare-variant aggregation: needs sequencing
    "ploF_burden":              frozenset({"wgs", "wes"}),
    "missense_burden":          frozenset({"wgs", "wes"}),
    "geneset_burden":           frozenset({"wgs", "wes"}),
    "loeuf_weighted_burden":    frozenset({"wgs", "wes"}),
    # Direct transcriptomics (rare in human-brain studies but valid for
    # blood-derived expression panels)
    "expression_level":         frozenset({"rnaseq"}),
    "expression_geneset_mean":  frozenset({"rnaseq"}),
    "expression_pc1":           frozenset({"rnaseq"}),
    # Imputed expression / TWAS (needs genotypes + GTEx weights)
    "imputed_expression":       frozenset({"genotype_array", "wgs", "wes"}),
    "twas_zscore":              frozenset({"genotype_array", "wgs", "wes"}),
    # Methylation
    "epigenetic_clock":         frozenset({"methylation_array"}),
    "methylation_geneset_mean": frozenset({"methylation_array"}),
    "methylation_single_cpg":   frozenset({"methylation_array"}),
    # Mitochondrial
    "mtdna_copy_number":        frozenset({"genotype_array", "wgs", "mtdna_seq"}),
    "mtdna_heteroplasmy":       frozenset({"wgs", "mtdna_seq"}),
}

ALL_OPERATIONS: frozenset[str] = frozenset(OP_TO_DATA_TYPES.keys())

DATA_TYPES: frozenset[str] = frozenset({
    "genotype_array", "wgs", "wes", "rnaseq", "methylation_array", "mtdna_seq",
})

# Family -> the operation set it can use. Enforced in validate_gms.
FAMILY_TO_OPS: dict[str, frozenset[str]] = {
    "polygenic_risk":          frozenset({"polygenic_risk", "partition_prs"}),
    "single_locus":            frozenset({"carrier_status", "genotype_count"}),
    "mutation_burden":         frozenset({"ploF_burden", "missense_burden",
                                          "geneset_burden",
                                          "loeuf_weighted_burden"}),
    "expression_single":       frozenset({"expression_level"}),
    "expression_aggregate":    frozenset({"expression_geneset_mean",
                                          "expression_pc1"}),
    "imputed_expression":      frozenset({"imputed_expression",
                                          "twas_zscore"}),
    "methylation_clock":       frozenset({"epigenetic_clock"}),
    "methylation_aggregate":   frozenset({"methylation_geneset_mean",
                                          "methylation_single_cpg"}),
    "mtdna_metric":            frozenset({"mtdna_copy_number",
                                          "mtdna_heteroplasmy"}),
    "cross_tissue_expression": frozenset({"expression_geneset_mean",
                                          "expression_pc1",
                                          "imputed_expression"}),
}

# Families that REQUIRE a disease/trait label (PRS/TWAS need GWAS source).
DISEASE_REQUIRED_FAMILIES: frozenset[str] = frozenset({
    "polygenic_risk", "imputed_expression",
})

# Families that REQUIRE a tissue (expression / TWAS / methylation that
# routes through a brain region or blood).
TISSUE_REQUIRED_FAMILIES: frozenset[str] = frozenset({
    "expression_single", "expression_aggregate", "imputed_expression",
    "cross_tissue_expression", "methylation_aggregate",
})

# Families that REQUIRE a clock name (Horvath/Hannum/PhenoAge/...).
CLOCK_REQUIRED_FAMILIES: frozenset[str] = frozenset({"methylation_clock"})

# Families that REQUIRE >=1 specific gene (carrier status, single expression).
GENE_REQUIRED_FAMILIES: frozenset[str] = frozenset({
    "single_locus", "expression_single",
})

# Families that REQUIRE a gene set (mutation burden, geneset expression).
GENESET_REQUIRED_FAMILIES: frozenset[str] = frozenset({
    "mutation_burden", "expression_aggregate", "methylation_aggregate",
    "cross_tissue_expression",
})


# == Curated gene sets and tissue palette ==================================


# Manually curated gene sets the LLM can reference by name. Members are HGNC
# symbols that overlap with the KG's GENE:* nodes; validation only checks
# that the gene set NAME is a palette token, not that every member resolves.
CURATED_GENE_SETS: dict[str, dict] = {
    "AD_risk_GWAS": {
        "description": "Common-variant AD risk loci from Bellenguez 2022 et al.",
        "members": ["APOE", "BIN1", "CR1", "CLU", "PICALM", "MS4A6A",
                     "ABCA7", "TREM2", "SORL1", "EPHA1", "CD33"],
    },
    "PD_risk_GWAS": {
        "description": "Common-variant PD risk loci from Nalls 2019 et al.",
        "members": ["SNCA", "LRRK2", "GBA", "MAPT", "PARK7", "PINK1",
                     "VPS35", "GCH1", "TMEM175", "BST1"],
    },
    "Mendelian_AD": {
        "description": "Autosomal-dominant early-onset AD genes.",
        "members": ["APP", "PSEN1", "PSEN2"],
    },
    "Mendelian_PD": {
        "description": "Autosomal-dominant or recessive PD genes.",
        "members": ["SNCA", "LRRK2", "PRKN", "PARK7", "PINK1", "VPS35",
                     "ATP13A2"],
    },
    "FTD_ALS": {
        "description": "FTD/ALS spectrum genes.",
        "members": ["MAPT", "GRN", "C9orf72", "TARDBP", "FUS", "SOD1",
                     "VCP", "CHCHD10"],
    },
    "Synaptic": {
        "description": "Synaptic and neurotransmission gene set "
                       "(SynGO core).",
        "members": ["DLG4", "SYN1", "SHANK3", "SYNGAP1", "GRIN2A",
                     "GRIN2B", "SLC17A7", "SLC32A1", "GAD1", "GAD2"],
    },
    "Dopaminergic": {
        "description": "Dopamine synthesis, transport, and receptor genes.",
        "members": ["TH", "DDC", "DBH", "DAT1", "SLC6A3", "DRD1", "DRD2",
                     "DRD3", "DRD4", "DRD5", "COMT", "MAOA", "MAOB"],
    },
    "Serotonergic": {
        "description": "Serotonin synthesis, transport, and receptor genes.",
        "members": ["TPH1", "TPH2", "SLC6A4", "HTR1A", "HTR1B", "HTR2A",
                     "HTR2C", "MAOA"],
    },
    "Cholinergic": {
        "description": "Acetylcholine synthesis and receptor genes.",
        "members": ["CHAT", "ACHE", "BCHE", "CHRNA4", "CHRNB2", "CHRM1"],
    },
    "Schizophrenia_GWAS": {
        "description": "Schizophrenia GWAS hits (PGC3 subset).",
        "members": ["DRD2", "GRIN2A", "CACNA1C", "ZNF804A", "TCF4",
                     "MIR137", "C4A"],
    },
    "Bipolar_GWAS": {
        "description": "Bipolar disorder GWAS hits.",
        "members": ["CACNA1C", "ANK3", "ODZ4", "TRANK1", "ITIH3"],
    },
    "MDD_GWAS": {
        "description": "Major depressive disorder GWAS hits.",
        "members": ["SIRT1", "LHPP", "OLFM4", "NEGR1"],
    },
    "Autism_GWAS": {
        "description": "Autism GWAS / rare-variant hits.",
        "members": ["CHD8", "SCN2A", "SHANK3", "PTEN", "MECP2", "TBR1",
                     "ARID1B", "GRIN2B"],
    },
    "Constrained_Genes_LOEUF": {
        "description": "Top-decile LOEUF-constrained protein-coding genes "
                       "(haploinsufficient).",
        "members": [],  # palette-only token; burden over gnomAD constraint set
    },
    "Microglia_immune": {
        "description": "Microglia / neuroinflammation gene set.",
        "members": ["TREM2", "TYROBP", "CR1", "CD33", "AIF1", "CX3CR1",
                     "P2RY12", "ITGAM"],
    },
    "Myelin": {
        "description": "Myelin and oligodendrocyte gene set.",
        "members": ["MBP", "PLP1", "MOG", "MAG", "OLIG1", "OLIG2", "SOX10"],
    },
    "GABA_glutamate": {
        "description": "GABA / glutamate receptor and transporter genes.",
        "members": ["GAD1", "GAD2", "SLC32A1", "GABBR1", "GABBR2",
                     "GRIA1", "GRIA2", "GRIN2A", "GRIN2B", "SLC17A7"],
    },
}


# Methylation clocks the LLM can pick by name.
METHYLATION_CLOCKS: tuple[dict, ...] = (
    {"name": "Horvath",      "ref": "Horvath 2013",  "n_cpg": 353},
    {"name": "Hannum",       "ref": "Hannum 2013",   "n_cpg": 71},
    {"name": "PhenoAge",     "ref": "Levine 2018",   "n_cpg": 513},
    {"name": "GrimAge",      "ref": "Lu 2019",       "n_cpg": 1030},
    {"name": "DunedinPACE",  "ref": "Belsky 2022",   "n_cpg": 173},
    {"name": "Horvath_Skin", "ref": "Horvath 2018",  "n_cpg": 391},
    {"name": "DNAmTL",       "ref": "Lu 2019",       "n_cpg": 140},
)


# Tissues / brain regions where expression / methylation can be measured.
# The brain regions mirror GTEx Brain v9 sample labels (the bulk-tissue
# transcriptomes most often used for TWAS in neuro studies).
GTEX_BRAIN_TISSUES: tuple[dict, ...] = (
    {"name": "Brain - Cortex",                   "kind": "brain"},
    {"name": "Brain - Frontal Cortex (BA9)",     "kind": "brain"},
    {"name": "Brain - Anterior cingulate cortex (BA24)", "kind": "brain"},
    {"name": "Brain - Hippocampus",              "kind": "brain"},
    {"name": "Brain - Amygdala",                 "kind": "brain"},
    {"name": "Brain - Hypothalamus",             "kind": "brain"},
    {"name": "Brain - Caudate (basal ganglia)",  "kind": "brain"},
    {"name": "Brain - Putamen (basal ganglia)",  "kind": "brain"},
    {"name": "Brain - Nucleus accumbens (basal ganglia)", "kind": "brain"},
    {"name": "Brain - Substantia nigra",         "kind": "brain"},
    {"name": "Brain - Cerebellum",               "kind": "brain"},
    {"name": "Brain - Cerebellar Hemisphere",    "kind": "brain"},
    {"name": "Brain - Spinal cord (cervical c-1)", "kind": "brain"},
    {"name": "Whole Blood",                      "kind": "blood"},
)


# Disease GWAS sources the LLM can attach to PRS / TWAS markers.
NEURO_GWAS_SOURCES: tuple[dict, ...] = (
    {"name": "Alzheimer Disease",        "consortium": "IGAP / Bellenguez 2022"},
    {"name": "Parkinson Disease",        "consortium": "Nalls 2019"},
    {"name": "Schizophrenia",            "consortium": "PGC3 2022"},
    {"name": "Bipolar Disorder",         "consortium": "PGC-BIP 2021"},
    {"name": "Major Depressive Disorder", "consortium": "PGC-MDD 2019 / Howard 2019"},
    {"name": "Autism Spectrum Disorder", "consortium": "iPSYCH-PGC 2019"},
    {"name": "Attention Deficit Hyperactivity Disorder",
                                          "consortium": "PGC-ADHD 2019"},
    {"name": "Anorexia Nervosa",         "consortium": "PGC-ED 2019"},
    {"name": "Anxiety",                  "consortium": "UKB-anxiety / Purves 2020"},
    {"name": "Migraine",                 "consortium": "Hautakangas 2022"},
    {"name": "Multiple Sclerosis",       "consortium": "IMSGC 2019"},
    {"name": "Amyotrophic Lateral Sclerosis", "consortium": "van Rheenen 2021"},
    {"name": "Stroke",                   "consortium": "MEGASTROKE 2018"},
    {"name": "Educational Attainment",   "consortium": "SSGAC 2018"},
    {"name": "Cognitive Performance",    "consortium": "Lee 2018"},
    {"name": "Insomnia",                 "consortium": "Jansen 2019"},
)


# == Palette construction ==================================================


@dataclass
class GMPalette:
    """KG-grounded primitives shown to the LLM for GM composition."""
    data_types:  list[dict]
    operations:  list[dict]
    families:    list[str]
    gene_sets:   list[dict]
    top_genes:   list[dict]   # {id, name, neuropsych_diseases:int}
    tissues:     list[dict]
    clocks:      list[dict]
    diseases:    list[dict]   # {id, name, gwas_consortium}

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def gene_set_names(self) -> set[str]:
        return {g["name"] for g in self.gene_sets}

    @property
    def gene_symbols(self) -> set[str]:
        return {g["name"] for g in self.top_genes}

    @property
    def tissue_names(self) -> set[str]:
        return {t["name"] for t in self.tissues}

    @property
    def clock_names(self) -> set[str]:
        return {c["name"] for c in self.clocks}

    @property
    def disease_names(self) -> set[str]:
        return {d["name"].lower() for d in self.diseases}


_NEURO_KEYWORDS: tuple[str, ...] = (
    "alzheim", "parkinson", "schizo", "autism", "depress", "bipolar",
    "epilep", "huntington", "adhd", "cognit", "migraine", "dyslex",
    "frontotemp", "dementia", "sclerosis", "stroke", "tourette",
    "obsessive", "panic", "psychos", "ptsd", "anxiet", "chorea",
    "ataxia", "encephalop", "tremor", "amyotroph", "narcoleps",
)


def _gene_disease_degree(
    concepts: dict[str, dict],
    edges: Iterable[dict],
) -> dict[str, int]:
    """For each GENE:* node, count its disease-associated edges, restricted
    to neuropsych-relevant disease nodes (so APP scores higher than housekeeping)."""
    deg: dict[str, int] = collections.Counter()
    for e in edges:
        if e.get("relation_type") != "gene_associated_with_disease":
            continue
        s, t = e.get("source_id", ""), e.get("target_id", "")
        sd = (concepts.get(s, {}).get("domain_tags") or [""])[0]
        td = (concepts.get(t, {}).get("domain_tags") or [""])[0]
        if sd == "gene" and td == "disease":
            gene_id, dis_id = s, t
        elif sd == "disease" and td == "gene":
            gene_id, dis_id = t, s
        else:
            continue
        nm = (concepts.get(dis_id, {}).get("preferred_name") or "").lower()
        if any(k in nm for k in _NEURO_KEYWORDS):
            deg[gene_id] += 1
    return deg


def _disease_gene_degree(
    concepts: dict[str, dict],
    edges: Iterable[dict],
) -> dict[str, int]:
    deg: dict[str, int] = collections.Counter()
    for e in edges:
        if e.get("relation_type") != "gene_associated_with_disease":
            continue
        s, t = e.get("source_id", ""), e.get("target_id", "")
        sd = (concepts.get(s, {}).get("domain_tags") or [""])[0]
        td = (concepts.get(t, {}).get("domain_tags") or [""])[0]
        if sd == "disease":
            deg[s] += 1
        elif td == "disease":
            deg[t] += 1
    return deg


def build_gm_palette(
    concepts: dict[str, dict],
    edges: Optional[Iterable[dict]] = None,
    n_top_genes: int = 80,
    n_diseases: int = 16,
) -> GMPalette:
    """Extract GM-relevant primitives from KG + curated lists."""
    edges = list(edges or [])
    gene_deg = _gene_disease_degree(concepts, edges)
    dis_deg = _disease_gene_degree(concepts, edges)

    data_types = [
        {"name": "genotype_array",
         "description": "imputed common-variant SNP array (e.g. UKB, ADNI)"},
        {"name": "wgs",
         "description": "whole-genome sequencing"},
        {"name": "wes",
         "description": "whole-exome sequencing"},
        {"name": "rnaseq",
         "description": "bulk RNA-seq (blood or post-mortem brain)"},
        {"name": "methylation_array",
         "description": "Illumina 450K / EPIC methylation array (blood)"},
        {"name": "mtdna_seq",
         "description": "mitochondrial DNA sequencing"},
    ]

    operations: list[dict] = []
    op_descriptions = {
        "polygenic_risk":         "weighted sum of GWAS-significant variants for a trait",
        "partition_prs":          "PRS restricted to a pathway / gene set",
        "carrier_status":         "binary indicator of >=1 risk variant in a gene",
        "genotype_count":         "0/1/2 risk-allele dosage at a single SNP",
        "ploF_burden":            "count of high-confidence loss-of-function variants",
        "missense_burden":        "count of CADD-deleterious missense variants",
        "geneset_burden":         "rare-variant burden summed over a curated gene set",
        "loeuf_weighted_burden":  "rare-variant burden weighted by gene LOEUF score",
        "expression_level":       "TPM expression of a single gene in a tissue",
        "expression_geneset_mean": "mean log-TPM across a gene set in a tissue",
        "expression_pc1":         "first principal component of a gene set's expression",
        "imputed_expression":     "PrediXcan-imputed expression in a tissue",
        "twas_zscore":            "TWAS Z-score of imputed expression vs. trait",
        "epigenetic_clock":       "DNA-methylation age estimate from a CpG panel",
        "methylation_geneset_mean": "mean methylation beta over a gene set's promoters",
        "methylation_single_cpg": "methylation beta at a single CpG",
        "mtdna_copy_number":      "estimated mitochondrial DNA copies per cell",
        "mtdna_heteroplasmy":     "fraction of mtDNA reads carrying a variant",
    }
    for slug in OP_TO_DATA_TYPES:
        operations.append({
            "slug":        slug,
            "description": op_descriptions.get(slug, ""),
            "data_types":  sorted(OP_TO_DATA_TYPES[slug]),
        })

    gene_sets = [{"name": k,
                   "description": v["description"],
                   "n_members":   len(v["members"])}
                  for k, v in CURATED_GENE_SETS.items()]

    top_genes: list[dict] = []
    for gid, n in sorted(gene_deg.items(), key=lambda kv: -kv[1])[:n_top_genes]:
        sym = concepts.get(gid, {}).get("preferred_name") or gid.split(":", 1)[-1]
        top_genes.append({"id": gid, "name": sym, "neuropsych_disease_edges": n})

    tissues = [{"name": t["name"], "kind": t["kind"]} for t in GTEX_BRAIN_TISSUES]
    clocks = [{"name": c["name"], "ref": c["ref"], "n_cpg": c["n_cpg"]}
              for c in METHYLATION_CLOCKS]

    diseases: list[dict] = []
    for d in NEURO_GWAS_SOURCES[:n_diseases]:
        diseases.append({
            "name": d["name"],
            "gwas_consortium": d["consortium"],
        })

    return GMPalette(
        data_types=data_types,
        operations=operations,
        families=list(GM_FAMILIES),
        gene_sets=gene_sets,
        top_genes=top_genes,
        tissues=tissues,
        clocks=clocks,
        diseases=diseases,
    )


def _render_palette(palette: GMPalette) -> str:
    lines: list[str] = []
    lines.append("Data types:")
    for d in palette.data_types:
        lines.append(f"  - {d['name']:18s} {d['description']}")
    lines.append("")
    lines.append("Operations (each constrained to specific data types):")
    for op in palette.operations:
        dts = "/".join(op["data_types"])
        lines.append(f"  - {op['slug']:26s} [{dts}] {op['description']}")
    lines.append("")
    lines.append(f"Families: {', '.join(palette.families)}")
    lines.append("")
    lines.append("Curated gene sets:")
    for gs in palette.gene_sets:
        lines.append(f"  - {gs['name']:24s} (n~{gs['n_members']}) {gs['description']}")
    lines.append("")
    lines.append("Top neuropsych genes (HGNC symbols, palette-only):")
    names = [g["name"] for g in palette.top_genes]
    lines.append("  " + ", ".join(names))
    lines.append("")
    lines.append("Tissues (GTEx Brain v9 + blood):")
    for t in palette.tissues:
        lines.append(f"  - {t['name']}")
    lines.append("")
    lines.append("Methylation clocks:")
    for c in palette.clocks:
        lines.append(f"  - {c['name']:14s} ({c['ref']}, {c['n_cpg']} CpGs)")
    lines.append("")
    lines.append("Disease GWAS sources (for PRS / TWAS):")
    for d in palette.diseases:
        lines.append(f"  - {d['name']}  [{d['gwas_consortium']}]")
    return "\n".join(lines)


# == GeneticMarker dataclass ===============================================


@dataclass
class GeneticMarker:
    id: str
    name: str
    family: str
    operation: str
    data_type: str
    gene_symbols: list[str] = field(default_factory=list)
    gene_set: Optional[str] = None
    tissue: Optional[str] = None
    clock: Optional[str] = None
    disease: Optional[str] = None
    formula: str = ""
    rationale: str = ""
    gene_ids: list[str] = field(default_factory=list)   # KG GENE:* ids after linking
    atoms: list[str] = field(default_factory=list)
    llm_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# == LLM brainstorm =========================================================


_SYSTEM_PROMPT = (
    "You design genetic-derived markers (GMs) for a neuroimaging knowledge "
    "graph. A GM is a scalar (or vector) quantity that an analysis pipeline "
    "can compute from one subject's genetic / transcriptomic / epigenetic "
    "data, e.g. 'AD polygenic risk score on UKB genotype array', 'rare-variant "
    "LoF burden in synaptic gene set from WES', or 'Horvath epigenetic age "
    "from blood methylation'. You receive a fixed palette of data types, "
    "operations, gene sets, gene symbols, tissues, clocks and disease GWAS "
    "sources; only those tokens may appear in your output. Do not invent "
    "new operations, gene sets, clocks, or tissues. Output strictly valid "
    "JSON, no prose."
)


_RULES = (
    "RULES (any GM violating these is invalid):\n"
    "  1. `family` must be one of the listed families.\n"
    "  2. `operation` must be a palette slug AND its family must permit it "
    "(see Operations / Families). `data_type` must be a palette name AND "
    "compatible with the chosen operation.\n"
    "  3. polygenic_risk and imputed_expression MUST set `disease` to a "
    "palette disease GWAS source.\n"
    "  4. expression_single, expression_aggregate, imputed_expression, "
    "cross_tissue_expression, methylation_aggregate MUST set `tissue` to a "
    "palette tissue.\n"
    "  5. methylation_clock MUST set `clock` to a palette clock name.\n"
    "  6. single_locus, expression_single MUST list >=1 specific gene from "
    "the palette gene-symbols pool in `gene_symbols`.\n"
    "  7. mutation_burden, expression_aggregate, methylation_aggregate, "
    "cross_tissue_expression MUST set `gene_set` to a palette curated set "
    "name (do not list individual genes).\n"
    "  8. The GM must be physically computable from a single subject's data. "
    "No cross-subject means, no multi-omics joint scores beyond what one "
    "operation produces, no behavioural-only quantities.\n"
    "  9. `name` must be a short snake_case slug; do not duplicate existing "
    "names.\n"
)


def _build_prompt(palette: GMPalette,
                  n: int,
                  family_focus: Optional[list[str]] = None,
                  existing_names: Optional[list[str]] = None) -> str:
    pal = _render_palette(palette)
    fam = (
        f"\nFamily focus for this batch (mix freely among these): "
        f"{', '.join(family_focus)}\n" if family_focus else ""
    )
    avoid = ""
    if existing_names:
        avoid = "\nAlready proposed (do NOT repeat or trivially rephrase):\n"
        for nm in existing_names:
            avoid += f"  - {nm}\n"
    return (
        f"Palette:\n{pal}\n"
        f"{fam}"
        f"\n{_RULES}\n"
        f"{avoid}"
        f"\nPropose {n} distinct genetic markers. Return a JSON array; each "
        "element MUST follow this schema exactly:\n"
        '  {"name": "<short slug>",\n'
        '   "family": "<one of the listed families>",\n'
        '   "operation": "<palette operation slug>",\n'
        '   "data_type": "<palette data type>",\n'
        '   "gene_symbols": ["<HGNC symbol>", ...] | [],\n'
        '   "gene_set": "<palette gene-set name>" | null,\n'
        '   "tissue": "<palette tissue name>" | null,\n'
        '   "clock": "<palette clock name>" | null,\n'
        '   "disease": "<palette disease name>" | null,\n'
        '   "formula": "<one-line definition referencing palette tokens>",\n'
        '   "rationale": "<one short sentence: why this GM is meaningful>"}\n'
    )


def brainstorm_gms(
    palette: GMPalette,
    n: int,
    llm_call: Callable[[str, str], str],
    model_name: str = "",
    family_focus: Optional[list[str]] = None,
    existing_names: Optional[list[str]] = None,
) -> list[GeneticMarker]:
    """Single LLM call -> list of raw GeneticMarker (pre-validation)."""
    prompt = _build_prompt(palette, n, family_focus, existing_names)
    raw = llm_call(prompt, _SYSTEM_PROMPT)
    data = _extract_json(raw)
    if not isinstance(data, list):
        logger.warning("LLM did not return a JSON array; got %r", type(data))
        return []
    out: list[GeneticMarker] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        family = str(item.get("family") or "").strip().lower()
        op = str(item.get("operation") or "").strip()
        dt = str(item.get("data_type") or "").strip()
        if not (name and family and op and dt):
            continue
        gs_raw = item.get("gene_symbols") or []
        if not isinstance(gs_raw, list):
            gs_raw = []
        out.append(GeneticMarker(
            id=f"gm_{i+1:04d}",
            name=name,
            family=family,
            operation=op,
            data_type=dt,
            gene_symbols=[str(s).strip() for s in gs_raw if str(s).strip()],
            gene_set=(item.get("gene_set") or None),
            tissue=(item.get("tissue") or None),
            clock=(item.get("clock") or None),
            disease=(item.get("disease") or None),
            formula=str(item.get("formula") or "").strip(),
            rationale=str(item.get("rationale") or "").strip(),
            llm_model=model_name,
        ))
    return out


_FAMILY_ROTATION: tuple[tuple[str, ...], ...] = (
    ("polygenic_risk",),
    ("mutation_burden",),
    ("expression_single", "expression_aggregate"),
    ("imputed_expression",),
    ("methylation_clock", "methylation_aggregate"),
    ("single_locus",),
    ("mtdna_metric", "cross_tissue_expression"),
)


def _normalise_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", " ", s.strip().lower())


def brainstorm_gms_batched(
    palette: GMPalette,
    n_total: int,
    llm_call: Callable[[str, str], str],
    model_name: str = "",
    batch_size: int = 30,
    seed: int = 0,
) -> list[GeneticMarker]:
    """Generate `n_total` raw GMs via repeated LLM calls with rotating family focus."""
    rng = random.Random(seed)
    accepted: list[GeneticMarker] = []
    seen_keys: set[str] = set()
    n_batches = (n_total + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        if len(accepted) >= n_total:
            break
        focus = _FAMILY_ROTATION[batch_idx % len(_FAMILY_ROTATION)]
        sample_existing = (
            [r.name for r in rng.sample(accepted, k=min(20, len(accepted)))]
            if accepted else None
        )
        try:
            batch = brainstorm_gms(
                palette, n=batch_size, llm_call=llm_call,
                model_name=model_name,
                family_focus=list(focus),
                existing_names=sample_existing,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch %d LLM call failed: %s", batch_idx, exc)
            continue
        added = 0
        for gm in batch:
            key = _normalise_name(gm.name)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            accepted.append(gm)
            added += 1
            if len(accepted) >= n_total:
                break
        logger.info("batch %d/%d focus=%s -> +%d (total %d/%d)",
                    batch_idx + 1, n_batches, focus, added,
                    len(accepted), n_total)
    accepted = accepted[:n_total]
    for i, gm in enumerate(accepted):
        gm.id = f"gm_{i+1:04d}"
    return accepted


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


# == Validation =============================================================


@dataclass
class ValidationReport:
    accepted: list[GeneticMarker]
    rejected: list[tuple[GeneticMarker, str]]

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def reject_reasons(self) -> dict[str, int]:
        c: dict[str, int] = collections.Counter()
        for _, reason in self.rejected:
            c[reason] += 1
        return dict(c)


def validate_gms(gms: Iterable[GeneticMarker], palette: GMPalette) -> ValidationReport:
    accepted: list[GeneticMarker] = []
    rejected: list[tuple[GeneticMarker, str]] = []
    gene_set_names = palette.gene_set_names
    gene_symbols = palette.gene_symbols
    tissue_names = palette.tissue_names
    clock_names = palette.clock_names
    disease_names = palette.disease_names

    for gm in gms:
        if gm.family not in GM_FAMILIES:
            rejected.append((gm, "family_invalid"))
            continue
        if gm.operation not in ALL_OPERATIONS:
            rejected.append((gm, "operation_unknown"))
            continue
        if gm.operation not in FAMILY_TO_OPS[gm.family]:
            rejected.append((gm, "operation_family_incompatible"))
            continue
        if gm.data_type not in DATA_TYPES:
            rejected.append((gm, "data_type_unknown"))
            continue
        if gm.data_type not in OP_TO_DATA_TYPES[gm.operation]:
            rejected.append((gm, "operation_data_type_incompatible"))
            continue
        if gm.family in DISEASE_REQUIRED_FAMILIES:
            if not gm.disease or gm.disease.lower() not in disease_names:
                rejected.append((gm, "disease_missing_or_unknown"))
                continue
        if gm.family in TISSUE_REQUIRED_FAMILIES:
            if not gm.tissue or gm.tissue not in tissue_names:
                rejected.append((gm, "tissue_missing_or_unknown"))
                continue
        if gm.family in CLOCK_REQUIRED_FAMILIES:
            if not gm.clock or gm.clock not in clock_names:
                rejected.append((gm, "clock_missing_or_unknown"))
                continue
        if gm.family in GENE_REQUIRED_FAMILIES:
            resolved = [s for s in gm.gene_symbols if s in gene_symbols]
            if not resolved:
                rejected.append((gm, "gene_missing_or_unknown"))
                continue
            gm.gene_symbols = resolved
        if gm.family in GENESET_REQUIRED_FAMILIES:
            if not gm.gene_set or gm.gene_set not in gene_set_names:
                rejected.append((gm, "gene_set_missing_or_unknown"))
                continue
        accepted.append(gm)
    return ValidationReport(accepted=accepted, rejected=rejected)


# == KG linking + atom tagging =============================================


def link_gms_to_kg(gms: Iterable[GeneticMarker], palette: GMPalette,
                    concepts: dict[str, dict]) -> None:
    """Resolve gene_symbols (and gene_set members for GENESET families) to
    KG GENE:* ids. The palette only carries top-degree genes, so we fall
    back to a global symbol -> id lookup over `concepts` for completeness."""
    sym_to_id: dict[str, str] = {}
    for cid, c in concepts.items():
        if cid.startswith("GENE:"):
            sym = c.get("preferred_name") or cid.split(":", 1)[-1]
            sym_to_id.setdefault(sym, cid)
    for gm in gms:
        ids: list[str] = []
        for s in gm.gene_symbols:
            rid = sym_to_id.get(s)
            if rid:
                ids.append(rid)
        if not ids and gm.gene_set:
            members = (CURATED_GENE_SETS.get(gm.gene_set) or {}).get("members") or []
            for s in members:
                rid = sym_to_id.get(s)
                if rid:
                    ids.append(rid)
        gm.gene_ids = sorted(set(ids))


def tag_atoms(gms: Iterable[GeneticMarker]) -> None:
    """Atoms by construction. GENE_TARGET is universal; PRS / TWAS that
    weight by a disease GWAS additionally carry DISEASE since they encode
    a disease-specific weighting, not a generic gene readout."""
    for gm in gms:
        atoms = ["GENE_TARGET"]
        if gm.family in DISEASE_REQUIRED_FAMILIES and gm.disease:
            atoms.append("DISEASE")
        gm.atoms = atoms


__all__ = [
    "GeneticMarker",
    "GMPalette",
    "ValidationReport",
    "GM_FAMILIES",
    "OP_TO_DATA_TYPES",
    "ALL_OPERATIONS",
    "DATA_TYPES",
    "FAMILY_TO_OPS",
    "CURATED_GENE_SETS",
    "METHYLATION_CLOCKS",
    "GTEX_BRAIN_TISSUES",
    "NEURO_GWAS_SOURCES",
    "build_gm_palette",
    "brainstorm_gms",
    "brainstorm_gms_batched",
    "validate_gms",
    "link_gms_to_kg",
    "tag_atoms",
]
