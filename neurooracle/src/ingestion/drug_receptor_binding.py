"""Phase 1 ingester: curated drug -> receptor binding dictionary.

Static, hand-curated table of primary pharmacological targets for the
neuropsychiatric drugs already added by `atc_drugs`. We deliberately keep
this dictionary small and high-confidence - the goal is to anchor each drug
to its 1-3 canonical mechanism-of-action targets, not to model every
off-target interaction one might find on PDSP / DrugBank. A targeted, curated
table beats a noisy parsed dump for hypothesis generation: paths through
HTR2A for psilocybin should reflect the field's consensus mechanism, not the
500 nM Ki of some screening hit.

Every target is the HGNC gene symbol of the receptor / transporter protein,
matching the GENE:* node IDs already in the KG and the genes used by
`hansen_receptor_density`. This means a hypothesis path

    DRUG psilocybin --binds_to--> GENE HTR2A --receptor_density_in--> NN region

is now expressible end-to-end without ad hoc bridge nodes.

Sources for curation: Goodman & Gilman, Stahl's Essential Psychopharmacology,
PDSP Ki database (top-affinity hits only), and recent reviews
(e.g. Vollenweider & Preller 2020 for psychedelics, Stahl 2019 for atypicals).

Edge predicate: `binds_to` (already in RELATION_TYPES), confidence 1.0 for
primary targets, 0.85 for secondary (clinically relevant but not the headline
mechanism).
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import Edge

logger = logging.getLogger(__name__)


# Each entry: ATC drug node id -> list of (gene_symbol, action, confidence).
# action is informational only; confidence stays in the edge.
# Genes listed are HGNC symbols matching GENE:<symbol> in the KG.
DRUG_TARGETS: dict[str, list[tuple[str, str, float]]] = {
    # SSRIs (N06AB) - all primarily SERT (SLC6A4)
    "ATC:N06AB03": [("SLC6A4", "primary inhibitor", 1.0)],                       # fluoxetine
    "ATC:N06AB04": [("SLC6A4", "primary inhibitor", 1.0)],                       # citalopram
    "ATC:N06AB05": [("SLC6A4", "primary inhibitor", 1.0)],                       # paroxetine
    "ATC:N06AB06": [("SLC6A4", "primary inhibitor", 1.0)],                       # sertraline
    "ATC:N06AB08": [("SLC6A4", "primary inhibitor", 1.0)],                       # fluvoxamine
    "ATC:N06AB10": [("SLC6A4", "primary inhibitor", 1.0)],                       # escitalopram
    # SNRIs / atypical (N06AX)
    "ATC:N06AX16": [("SLC6A4", "primary inhibitor", 1.0),
                    ("SLC6A2", "primary inhibitor", 1.0)],                        # venlafaxine
    "ATC:N06AX21": [("SLC6A4", "primary inhibitor", 1.0),
                    ("SLC6A2", "primary inhibitor", 1.0)],                        # duloxetine
    "ATC:N06AX11": [("HTR2A",  "antagonist",        1.0),
                    ("ADRA2A", "antagonist",        0.85)],                       # mirtazapine
    "ATC:N06AX12": [("SLC6A3", "primary inhibitor", 1.0),
                    ("SLC6A2", "primary inhibitor", 1.0)],                        # bupropion (NDRI)
    "ATC:N06AX26": [("HTR1A",  "agonist",           1.0),
                    ("HTR2A",  "antagonist",        0.85),
                    ("SLC6A4", "primary inhibitor", 1.0)],                        # vortioxetine (multimodal)
    # TCAs
    "ATC:N06AA09": [("SLC6A4", "inhibitor",         1.0),
                    ("SLC6A2", "inhibitor",         1.0)],                        # amitriptyline
    "ATC:N06AA10": [("SLC6A2", "inhibitor",         1.0),
                    ("SLC6A4", "inhibitor",         0.85)],                       # nortriptyline
    "ATC:N06AA12": [("SLC6A2", "inhibitor",         1.0),
                    ("HRH1",   "antagonist",        0.85)],                       # doxepin
    "ATC:N06AA04": [("SLC6A4", "primary inhibitor", 1.0),
                    ("SLC6A2", "inhibitor",         0.85)],                       # clomipramine
    # Atypical antipsychotics - primary D2 + 5-HT2A
    "ATC:N05AH02": [("DRD2",   "antagonist",        1.0),
                    ("HTR2A",  "antagonist",        1.0),
                    ("HTR1A",  "partial agonist",   0.85)],                       # clozapine
    "ATC:N05AH03": [("DRD2",   "antagonist",        1.0),
                    ("HTR2A",  "antagonist",        1.0)],                        # olanzapine
    "ATC:N05AH04": [("DRD2",   "antagonist",        1.0),
                    ("HTR2A",  "antagonist",        1.0),
                    ("HRH1",   "antagonist",        0.85)],                       # quetiapine
    "ATC:N05AX08": [("DRD2",   "antagonist",        1.0),
                    ("HTR2A",  "antagonist",        1.0)],                        # risperidone
    "ATC:N05AX12": [("DRD2",   "partial agonist",   1.0),
                    ("HTR2A",  "antagonist",        1.0),
                    ("HTR1A",  "partial agonist",   0.85)],                       # aripiprazole
    "ATC:N05AE04": [("DRD2",   "antagonist",        1.0),
                    ("HTR2A",  "antagonist",        1.0)],                        # ziprasidone
    "ATC:N05AX13": [("DRD2",   "antagonist",        1.0),
                    ("HTR2A",  "antagonist",        1.0)],                        # paliperidone
    # Typical antipsychotics - D2 dominant
    "ATC:N05AA01": [("DRD2",   "antagonist",        1.0),
                    ("HRH1",   "antagonist",        0.85)],                       # chlorpromazine
    "ATC:N05AD01": [("DRD2",   "antagonist",        1.0)],                        # haloperidol
    # Mood stabilisers - lithium has no direct receptor; skip rather than fake
    # Benzodiazepines - GABA-A allosteric (alpha1 subunit dominant)
    "ATC:N05BA01": [("GABRA1", "positive modulator", 1.0)],                       # diazepam
    "ATC:N05BA04": [("GABRA1", "positive modulator", 1.0)],                       # oxazepam
    "ATC:N05BA06": [("GABRA1", "positive modulator", 1.0)],                       # lorazepam
    "ATC:N05BA12": [("GABRA1", "positive modulator", 1.0)],                       # alprazolam
    "ATC:N05CD08": [("GABRA1", "positive modulator", 1.0)],                       # midazolam
    "ATC:N05CD07": [("GABRA1", "positive modulator", 1.0)],                       # temazepam
    # Z-drugs - GABA-A alpha1 selective
    "ATC:N05CF01": [("GABRA1", "positive modulator", 1.0)],                       # zopiclone
    "ATC:N05CF02": [("GABRA1", "positive modulator", 1.0)],                       # zolpidem
    # Cholinesterase inhibitors - target AChE (ACHE) not a receptor; we still
    # link via binds_to since downstream paths can ask "drug -> ACHE -> region"
    "ATC:N06DA02": [("ACHE",   "inhibitor",         1.0)],                        # donepezil
    "ATC:N06DA03": [("ACHE",   "inhibitor",         1.0),
                    ("BCHE",   "inhibitor",         1.0)],                        # rivastigmine
    "ATC:N06DA04": [("ACHE",   "inhibitor",         1.0),
                    ("CHRNA4", "positive modulator", 0.85)],                      # galantamine
    # NMDA antagonists
    "ATC:N06DX01": [("GRIN2B", "antagonist",        1.0)],                        # memantine (NMDA)
    # ADHD stimulants
    "ATC:N06BA04": [("SLC6A3", "primary inhibitor", 1.0),
                    ("SLC6A2", "primary inhibitor", 1.0)],                        # methylphenidate
    "ATC:N06BA02": [("SLC6A3", "releaser",          1.0),
                    ("SLC6A2", "releaser",          1.0)],                        # dexamphetamine
    "ATC:N06BA12": [("SLC6A3", "releaser",          1.0),
                    ("SLC6A2", "releaser",          1.0)],                        # lisdexamfetamine
    "ATC:N06BA09": [("SLC6A2", "primary inhibitor", 1.0)],                        # atomoxetine
    "ATC:N06BA07": [("SLC6A3", "weak inhibitor",    0.85)],                       # modafinil
    # Anti-Parkinson - levodopa is precursor (not a receptor binder);
    # agonists target DRD2 / DRD3
    "ATC:N04BC04": [("DRD2",   "agonist",           1.0),
                    ("DRD3",   "agonist",           1.0)],                        # ropinirole
    "ATC:N04BC05": [("DRD2",   "agonist",           1.0),
                    ("DRD3",   "agonist",           1.0)],                        # pramipexole
    "ATC:N04BD01": [("MAOB",   "inhibitor",         1.0)],                        # selegiline
    "ATC:N04BD02": [("MAOB",   "inhibitor",         1.0)],                        # rasagiline
    # Anti-epileptics - VGSC / SV2A / GABA / Ca channel ligands
    "ATC:N03AE01": [("GABRA1", "positive modulator", 1.0)],                       # clonazepam
    "ATC:N03AF01": [("SCN1A",  "blocker",           1.0)],                        # carbamazepine
    "ATC:N03AG01": [("GABRA1", "indirect enhancer", 0.85)],                       # valproate
    "ATC:N03AX09": [("SCN1A",  "blocker",           1.0)],                        # lamotrigine
    "ATC:N03AX12": [("CACNA2D1", "ligand",          1.0)],                        # gabapentin
    "ATC:N03AX14": [("SV2A",   "modulator",         1.0)],                        # levetiracetam
    "ATC:N03AX16": [("CACNA2D1", "ligand",          1.0)],                        # pregabalin
    # Addiction
    "ATC:N07BA01": [("CHRNA4", "agonist",           1.0),
                    ("CHRNB2", "agonist",           1.0)],                        # nicotine
    "ATC:N07BA03": [("CHRNA4", "partial agonist",   1.0),
                    ("CHRNB2", "partial agonist",   1.0)],                        # varenicline
    "ATC:N07BB04": [("OPRM1",  "antagonist",        1.0)],                        # naltrexone
    "ATC:N07BC02": [("OPRM1",  "agonist",           1.0)],                        # methadone
    "ATC:N07BC01": [("OPRM1",  "partial agonist",   1.0)],                        # buprenorphine
    # Other / emerging
    "ATC:N07XX_KETAMINE":   [("GRIN2B", "antagonist", 1.0)],                      # ketamine
    "ATC:N07XX_PSILOCYBIN": [("HTR2A",  "agonist",    1.0),
                             ("HTR1A",  "agonist",    0.85)],                     # psilocybin
    "ATC:N03AX_CANNABIDIOL":[("CNR1",   "weak modulator", 0.85)],                 # CBD
}


def ingest_drug_receptor_binding(kg: KnowledgeGraph) -> dict:
    """Add DRUG -> GENE binds_to edges for the curated table. Idempotent.

    Skips drugs absent from the KG and gene targets absent from the KG. Logs
    the breakdown so missing targets surface immediately.
    """
    edges_added = 0
    drugs_processed = 0
    drugs_skipped = []
    targets_skipped = []
    new_genes_added = 0

    for atc_id, targets in DRUG_TARGETS.items():
        if not kg.has_concept(atc_id):
            drugs_skipped.append(atc_id)
            continue
        drugs_processed += 1
        for gene, action, conf in targets:
            gene_id = f"GENE:{gene}"
            if not kg.has_concept(gene_id):
                # add a stub gene node so the edge is not dropped silently;
                # downstream UMLS merge or future HGNC ingest can enrich it.
                from ..schema import ConceptNode, DomainTag
                kg.add_concept(ConceptNode(
                    id=gene_id,
                    preferred_name=gene,
                    domain_tags=[DomainTag.GENE.value],
                    source_vocab="curated_pharmacology",
                    definition=f"HGNC gene symbol {gene} (drug target stub).",
                    aliases=[gene.upper()],
                    external_ids={"HGNC": gene},
                ))
                new_genes_added += 1
                targets_skipped.append((atc_id, gene))
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=atc_id,
                target_id=gene_id,
                relation_type="binds_to",
                source="curated_pharmacology",
                confidence=float(conf),
                evidence_ref=f"{action} (Stahl/Goodman & Gilman/PDSP curation 2026-05-27)",
                metadata={"action": action},
            ))
            if kg.G.number_of_edges() > before:
                edges_added += 1

    summary = {
        "drugs_processed":  drugs_processed,
        "drugs_skipped":    len(drugs_skipped),
        "edges_added":      edges_added,
        "gene_stubs_added": new_genes_added,
    }
    logger.info(f"drug-receptor binding ingestion complete: {summary}")
    if drugs_skipped:
        logger.info(f"  skipped (drug not in KG): {drugs_skipped}")
    return summary


__all__ = ["ingest_drug_receptor_binding", "DRUG_TARGETS"]
