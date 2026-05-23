"""Phase 1 ingester: ATC neuropsychiatric drugs (`drug` atom).

Adds the ATC (Anatomical Therapeutic Chemical) hierarchy for neuropsychiatric
drug classes (N03 anti-epileptics, N04 anti-Parkinson, N05 psycholeptics —
antipsychotics + anxiolytics + hypnotics, N06 psychoanaleptics —
antidepressants + nootropics + ADHD agents + dementia drugs, N07 other
nervous-system) plus a curated set of common INN drug names per leaf class.

Scope: ~150 nodes — manageable, but provides the structured drug backbone the
KG previously lacked. (Currently 100% of the `drug` atom comes from
claim_extraction.) Once these are merged with claim-derived drug nodes via
UMLS CUI, downstream tasks like `drug_repurposing`, `drug_response_prediction`,
and `adverse_event_prediction` get a stable anchor set.

Edges: ATC class hierarchy via `is_a` (drug → leaf-class → ... → N).

This is a curated knowledge module — no download. ATC code references
https://www.whocc.no/atc/structure_and_principles/.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)


# ── ATC class tree (level 1 → 4) ─────────────────────────────────────────────

ATC_CLASSES: dict[str, dict] = {
    # Level 1 — anatomical main group
    "ATC:N":        {"name": "Nervous system",
                     "level": 1, "parent": None,
                     "description": "ATC level-1 anatomical main group: drugs acting on the nervous system."},
    # Level 2 — therapeutic subgroups (N03–N07)
    "ATC:N03":      {"name": "Antiepileptics",
                     "level": 2, "parent": "ATC:N"},
    "ATC:N04":      {"name": "Anti-parkinson drugs",
                     "level": 2, "parent": "ATC:N"},
    "ATC:N05":      {"name": "Psycholeptics",
                     "level": 2, "parent": "ATC:N"},
    "ATC:N06":      {"name": "Psychoanaleptics",
                     "level": 2, "parent": "ATC:N"},
    "ATC:N07":      {"name": "Other nervous system drugs",
                     "level": 2, "parent": "ATC:N"},

    # Level 3 — pharmacological subgroup
    "ATC:N03A":     {"name": "Antiepileptic drugs",      "level": 3, "parent": "ATC:N03"},
    "ATC:N04A":     {"name": "Anticholinergic agents",   "level": 3, "parent": "ATC:N04"},
    "ATC:N04B":     {"name": "Dopaminergic agents",      "level": 3, "parent": "ATC:N04"},
    "ATC:N05A":     {"name": "Antipsychotics",           "level": 3, "parent": "ATC:N05"},
    "ATC:N05B":     {"name": "Anxiolytics",              "level": 3, "parent": "ATC:N05"},
    "ATC:N05C":     {"name": "Hypnotics and sedatives",  "level": 3, "parent": "ATC:N05"},
    "ATC:N06A":     {"name": "Antidepressants",          "level": 3, "parent": "ATC:N06"},
    "ATC:N06B":     {"name": "Psychostimulants and nootropics",
                     "level": 3, "parent": "ATC:N06"},
    "ATC:N06D":     {"name": "Anti-dementia drugs",      "level": 3, "parent": "ATC:N06"},
    "ATC:N07A":     {"name": "Parasympathomimetics",     "level": 3, "parent": "ATC:N07"},
    "ATC:N07B":     {"name": "Drugs used in addictive disorders",
                     "level": 3, "parent": "ATC:N07"},
    "ATC:N07X":     {"name": "Other nervous system drugs",
                     "level": 3, "parent": "ATC:N07"},

    # Level 4 — chemical subgroup (selected high-yield ones)
    "ATC:N06AA":    {"name": "Tricyclic antidepressants (TCAs)",
                     "level": 4, "parent": "ATC:N06A"},
    "ATC:N06AB":    {"name": "Selective serotonin reuptake inhibitors (SSRIs)",
                     "level": 4, "parent": "ATC:N06A"},
    "ATC:N06AX":    {"name": "Other antidepressants (SNRIs, atypical)",
                     "level": 4, "parent": "ATC:N06A"},
    "ATC:N05AH":    {"name": "Atypical antipsychotics",
                     "level": 4, "parent": "ATC:N05A"},
    "ATC:N05AA":    {"name": "Phenothiazine antipsychotics",
                     "level": 4, "parent": "ATC:N05A"},
    "ATC:N05BA":    {"name": "Benzodiazepine anxiolytics",
                     "level": 4, "parent": "ATC:N05B"},
    "ATC:N05CD":    {"name": "Benzodiazepine hypnotics",
                     "level": 4, "parent": "ATC:N05C"},
    "ATC:N05CF":    {"name": "Z-drug hypnotics",
                     "level": 4, "parent": "ATC:N05C"},
    "ATC:N06DA":    {"name": "Cholinesterase inhibitors",
                     "level": 4, "parent": "ATC:N06D"},
    "ATC:N06DX":    {"name": "Other anti-dementia (memantine, etc.)",
                     "level": 4, "parent": "ATC:N06D"},
    "ATC:N06BA":    {"name": "ADHD stimulants",
                     "level": 4, "parent": "ATC:N06B"},
    "ATC:N04BA":    {"name": "Dopa and dopa derivatives",
                     "level": 4, "parent": "ATC:N04B"},
    "ATC:N04BC":    {"name": "Dopamine agonists",
                     "level": 4, "parent": "ATC:N04B"},
    "ATC:N04BD":    {"name": "Monoamine oxidase B inhibitors",
                     "level": 4, "parent": "ATC:N04B"},
    "ATC:N07BA":    {"name": "Drugs used in nicotine dependence",
                     "level": 4, "parent": "ATC:N07B"},
    "ATC:N07BB":    {"name": "Drugs used in alcohol dependence",
                     "level": 4, "parent": "ATC:N07B"},
    "ATC:N07BC":    {"name": "Drugs used in opioid dependence",
                     "level": 4, "parent": "ATC:N07B"},
}


# ── Level 5 — INN substances (high-yield neuropsychiatric drugs) ─────────────
#
# Each entry: ATC code + parent class + canonical INN + aliases + brief
# mechanism note. Aliases include common brand names and orthographic variants
# so UMLS alignment / claim-extraction merging can connect them.

ATC_DRUGS: dict[str, dict] = {
    # SSRIs (N06AB)
    "ATC:N06AB03": {"name": "fluoxetine", "parent": "ATC:N06AB",
                    "aliases": ["Prozac"], "moa": "SSRI"},
    "ATC:N06AB04": {"name": "citalopram", "parent": "ATC:N06AB",
                    "aliases": ["Celexa"], "moa": "SSRI"},
    "ATC:N06AB05": {"name": "paroxetine", "parent": "ATC:N06AB",
                    "aliases": ["Paxil", "Seroxat"], "moa": "SSRI"},
    "ATC:N06AB06": {"name": "sertraline", "parent": "ATC:N06AB",
                    "aliases": ["Zoloft"], "moa": "SSRI"},
    "ATC:N06AB08": {"name": "fluvoxamine", "parent": "ATC:N06AB",
                    "aliases": ["Luvox"], "moa": "SSRI"},
    "ATC:N06AB10": {"name": "escitalopram", "parent": "ATC:N06AB",
                    "aliases": ["Lexapro", "Cipralex"], "moa": "SSRI"},
    # SNRIs / atypicals (N06AX)
    "ATC:N06AX16": {"name": "venlafaxine", "parent": "ATC:N06AX",
                    "aliases": ["Effexor"], "moa": "SNRI"},
    "ATC:N06AX21": {"name": "duloxetine",  "parent": "ATC:N06AX",
                    "aliases": ["Cymbalta"], "moa": "SNRI"},
    "ATC:N06AX11": {"name": "mirtazapine", "parent": "ATC:N06AX",
                    "aliases": ["Remeron"], "moa": "alpha2 antagonist + 5-HT2 block"},
    "ATC:N06AX12": {"name": "bupropion",   "parent": "ATC:N06AX",
                    "aliases": ["Wellbutrin", "Zyban"], "moa": "NDRI"},
    "ATC:N06AX26": {"name": "vortioxetine","parent": "ATC:N06AX",
                    "aliases": ["Trintellix", "Brintellix"], "moa": "multimodal serotonergic"},
    # TCAs (N06AA)
    "ATC:N06AA09": {"name": "amitriptyline", "parent": "ATC:N06AA",
                    "aliases": ["Elavil"], "moa": "TCA"},
    "ATC:N06AA10": {"name": "nortriptyline", "parent": "ATC:N06AA",
                    "aliases": ["Pamelor"], "moa": "TCA"},
    "ATC:N06AA12": {"name": "doxepin",       "parent": "ATC:N06AA",
                    "aliases": [], "moa": "TCA"},
    "ATC:N06AA04": {"name": "clomipramine",  "parent": "ATC:N06AA",
                    "aliases": ["Anafranil"], "moa": "TCA"},
    # Atypical antipsychotics (N05AH and adjacent)
    "ATC:N05AH02": {"name": "clozapine",     "parent": "ATC:N05AH",
                    "aliases": ["Clozaril"], "moa": "atypical antipsychotic"},
    "ATC:N05AH03": {"name": "olanzapine",    "parent": "ATC:N05AH",
                    "aliases": ["Zyprexa"], "moa": "atypical antipsychotic"},
    "ATC:N05AH04": {"name": "quetiapine",    "parent": "ATC:N05AH",
                    "aliases": ["Seroquel"], "moa": "atypical antipsychotic"},
    "ATC:N05AX08": {"name": "risperidone",   "parent": "ATC:N05A",
                    "aliases": ["Risperdal"], "moa": "atypical antipsychotic"},
    "ATC:N05AX12": {"name": "aripiprazole",  "parent": "ATC:N05A",
                    "aliases": ["Abilify"], "moa": "D2 partial agonist"},
    "ATC:N05AE04": {"name": "ziprasidone",   "parent": "ATC:N05A",
                    "aliases": ["Geodon"], "moa": "atypical antipsychotic"},
    "ATC:N05AX13": {"name": "paliperidone",  "parent": "ATC:N05A",
                    "aliases": ["Invega"], "moa": "atypical antipsychotic"},
    # Typical antipsychotics
    "ATC:N05AA01": {"name": "chlorpromazine","parent": "ATC:N05AA",
                    "aliases": ["Thorazine"], "moa": "typical antipsychotic"},
    "ATC:N05AD01": {"name": "haloperidol",   "parent": "ATC:N05A",
                    "aliases": ["Haldol"], "moa": "typical antipsychotic"},
    # Mood stabilisers
    "ATC:N05AN01": {"name": "lithium",       "parent": "ATC:N05A",
                    "aliases": ["lithium carbonate", "lithium salt"], "moa": "mood stabiliser"},
    # Benzodiazepines (N05BA / N05CD)
    "ATC:N05BA01": {"name": "diazepam",      "parent": "ATC:N05BA",
                    "aliases": ["Valium"], "moa": "benzodiazepine"},
    "ATC:N05BA04": {"name": "oxazepam",      "parent": "ATC:N05BA",
                    "aliases": [], "moa": "benzodiazepine"},
    "ATC:N05BA06": {"name": "lorazepam",     "parent": "ATC:N05BA",
                    "aliases": ["Ativan"], "moa": "benzodiazepine"},
    "ATC:N05BA12": {"name": "alprazolam",    "parent": "ATC:N05BA",
                    "aliases": ["Xanax"], "moa": "benzodiazepine"},
    "ATC:N05CD08": {"name": "midazolam",     "parent": "ATC:N05CD",
                    "aliases": ["Versed"], "moa": "benzodiazepine"},
    "ATC:N05CD07": {"name": "temazepam",     "parent": "ATC:N05CD",
                    "aliases": ["Restoril"], "moa": "benzodiazepine"},
    # Z-drugs (N05CF)
    "ATC:N05CF01": {"name": "zopiclone",     "parent": "ATC:N05CF",
                    "aliases": ["Imovane"], "moa": "Z-drug"},
    "ATC:N05CF02": {"name": "zolpidem",      "parent": "ATC:N05CF",
                    "aliases": ["Ambien"], "moa": "Z-drug"},
    # Cholinesterase inhibitors (N06DA)
    "ATC:N06DA02": {"name": "donepezil",     "parent": "ATC:N06DA",
                    "aliases": ["Aricept"], "moa": "AChE inhibitor"},
    "ATC:N06DA03": {"name": "rivastigmine",  "parent": "ATC:N06DA",
                    "aliases": ["Exelon"], "moa": "AChE/BuChE inhibitor"},
    "ATC:N06DA04": {"name": "galantamine",   "parent": "ATC:N06DA",
                    "aliases": ["Razadyne"], "moa": "AChE inhibitor + nicotinic modulator"},
    # NMDA antagonists (anti-dementia)
    "ATC:N06DX01": {"name": "memantine",     "parent": "ATC:N06DX",
                    "aliases": ["Namenda"], "moa": "NMDA receptor antagonist"},
    # Anti-amyloid mAbs (recent — N06D, no formal level-4 yet)
    "ATC:N06D_LECANEMAB":  {"name": "lecanemab",  "parent": "ATC:N06D",
                            "aliases": ["Leqembi"], "moa": "anti-Aβ monoclonal antibody"},
    "ATC:N06D_DONANEMAB":  {"name": "donanemab",  "parent": "ATC:N06D",
                            "aliases": ["Kisunla"], "moa": "anti-Aβ monoclonal antibody"},
    "ATC:N06D_ADUCANUMAB": {"name": "aducanumab", "parent": "ATC:N06D",
                            "aliases": ["Aduhelm"], "moa": "anti-Aβ monoclonal antibody"},
    # ADHD stimulants (N06BA)
    "ATC:N06BA04": {"name": "methylphenidate","parent": "ATC:N06BA",
                    "aliases": ["Ritalin", "Concerta"], "moa": "DAT/NET inhibitor"},
    "ATC:N06BA02": {"name": "dexamphetamine", "parent": "ATC:N06BA",
                    "aliases": ["dextroamphetamine", "Dexedrine"], "moa": "amphetamine"},
    "ATC:N06BA12": {"name": "lisdexamfetamine","parent": "ATC:N06BA",
                    "aliases": ["Vyvanse"], "moa": "amphetamine prodrug"},
    "ATC:N06BA09": {"name": "atomoxetine",    "parent": "ATC:N06BA",
                    "aliases": ["Strattera"], "moa": "NET inhibitor (non-stim ADHD)"},
    "ATC:N06BA07": {"name": "modafinil",      "parent": "ATC:N06BA",
                    "aliases": ["Provigil"], "moa": "wake-promoting"},
    # Anti-Parkinson (N04)
    "ATC:N04BA02": {"name": "levodopa",       "parent": "ATC:N04BA",
                    "aliases": ["L-dopa"], "moa": "dopamine precursor"},
    "ATC:N04BC04": {"name": "ropinirole",     "parent": "ATC:N04BC",
                    "aliases": ["Requip"], "moa": "dopamine D2/D3 agonist"},
    "ATC:N04BC05": {"name": "pramipexole",    "parent": "ATC:N04BC",
                    "aliases": ["Mirapex"], "moa": "dopamine D2/D3 agonist"},
    "ATC:N04BD01": {"name": "selegiline",     "parent": "ATC:N04BD",
                    "aliases": ["Eldepryl"], "moa": "MAO-B inhibitor"},
    "ATC:N04BD02": {"name": "rasagiline",     "parent": "ATC:N04BD",
                    "aliases": ["Azilect"], "moa": "MAO-B inhibitor"},
    # Anti-epileptics (N03A)
    "ATC:N03AE01": {"name": "clonazepam",     "parent": "ATC:N03A",
                    "aliases": ["Klonopin"], "moa": "benzodiazepine antiepileptic"},
    "ATC:N03AF01": {"name": "carbamazepine",  "parent": "ATC:N03A",
                    "aliases": ["Tegretol"], "moa": "Na+ channel blocker"},
    "ATC:N03AG01": {"name": "valproate",      "parent": "ATC:N03A",
                    "aliases": ["valproic acid", "sodium valproate", "Depakote"],
                    "moa": "GABAergic / Na+ channel"},
    "ATC:N03AX09": {"name": "lamotrigine",    "parent": "ATC:N03A",
                    "aliases": ["Lamictal"], "moa": "Na+ channel blocker"},
    "ATC:N03AX11": {"name": "topiramate",     "parent": "ATC:N03A",
                    "aliases": ["Topamax"], "moa": "multi-target AED"},
    "ATC:N03AX12": {"name": "gabapentin",     "parent": "ATC:N03A",
                    "aliases": ["Neurontin"], "moa": "α2δ Ca2+ channel ligand"},
    "ATC:N03AX14": {"name": "levetiracetam",  "parent": "ATC:N03A",
                    "aliases": ["Keppra"], "moa": "SV2A modulator"},
    "ATC:N03AX16": {"name": "pregabalin",     "parent": "ATC:N03A",
                    "aliases": ["Lyrica"], "moa": "α2δ Ca2+ channel ligand"},
    # Addiction (N07B)
    "ATC:N07BA01": {"name": "nicotine",         "parent": "ATC:N07BA",
                    "aliases": ["nicotine replacement"], "moa": "nicotinic agonist"},
    "ATC:N07BA03": {"name": "varenicline",      "parent": "ATC:N07BA",
                    "aliases": ["Chantix", "Champix"], "moa": "α4β2 nicotinic partial agonist"},
    "ATC:N07BB03": {"name": "acamprosate",      "parent": "ATC:N07BB",
                    "aliases": ["Campral"], "moa": "glutamate modulator"},
    "ATC:N07BB04": {"name": "naltrexone",       "parent": "ATC:N07BB",
                    "aliases": ["ReVia", "Vivitrol"], "moa": "opioid antagonist"},
    "ATC:N07BB01": {"name": "disulfiram",       "parent": "ATC:N07BB",
                    "aliases": ["Antabuse"], "moa": "ALDH inhibitor"},
    "ATC:N07BC02": {"name": "methadone",        "parent": "ATC:N07BC",
                    "aliases": [], "moa": "mu-opioid agonist (long acting)"},
    "ATC:N07BC01": {"name": "buprenorphine",    "parent": "ATC:N07BC",
                    "aliases": ["Suboxone", "Subutex"], "moa": "mu-opioid partial agonist"},
    # Other / emerging (N07X)
    "ATC:N07XX_KETAMINE":  {"name": "ketamine",  "parent": "ATC:N07X",
                            "aliases": ["esketamine", "Spravato"], "moa": "NMDA antagonist (rapid-acting AD)"},
    "ATC:N07XX_PSILOCYBIN":{"name": "psilocybin","parent": "ATC:N07X",
                            "aliases": [], "moa": "5-HT2A agonist (psychedelic)"},
    "ATC:N03AX_CANNABIDIOL":{"name": "cannabidiol","parent": "ATC:N03A",
                            "aliases": ["CBD", "Epidiolex"], "moa": "non-psychoactive cannabinoid"},
}


def _add_class_node(kg: KnowledgeGraph, nid: str, info: dict) -> bool:
    existed = kg.has_concept(nid)
    kg.add_concept(ConceptNode(
        id=nid,
        preferred_name=info["name"],
        domain_tags=[DomainTag.DRUG.value],
        source_vocab="ATC",
        definition=info.get("description", f"ATC level-{info['level']} class."),
        external_ids={"ATC": nid.split(":", 1)[1]},
        metadata={"atc_level": info["level"], "is_class": True},
    ))
    parent = info.get("parent")
    if parent and kg.has_concept(parent):
        kg.add_edge(Edge(
            source_id=nid,
            target_id=parent,
            relation_type="is_a",
            source="ATC",
            confidence=1.0,
        ))
    return not existed


def _add_drug_node(kg: KnowledgeGraph, nid: str, info: dict) -> bool:
    existed = kg.has_concept(nid)
    kg.add_concept(ConceptNode(
        id=nid,
        preferred_name=info["name"],
        aliases=info.get("aliases", []),
        domain_tags=[DomainTag.DRUG.value],
        source_vocab="ATC",
        definition=info.get("moa", ""),
        external_ids={"ATC": nid.split(":", 1)[1]},
        metadata={"atc_level": 5, "moa": info.get("moa", "")},
    ))
    parent = info.get("parent")
    if parent and kg.has_concept(parent):
        kg.add_edge(Edge(
            source_id=nid,
            target_id=parent,
            relation_type="is_a",
            source="ATC",
            confidence=1.0,
        ))
    return not existed


def ingest_atc_drugs(kg: KnowledgeGraph) -> dict:
    """Seed ATC neuropsychiatric drug class hierarchy + INN substances. Idempotent.

    The order matters: classes are added before drugs so the is_a edges
    can connect to existing parents.
    """
    classes_added = 0
    drugs_added = 0

    # Add classes in level order (parents first) so is_a links resolve cleanly.
    for level in (1, 2, 3, 4):
        for nid, info in ATC_CLASSES.items():
            if info["level"] != level:
                continue
            if _add_class_node(kg, nid, info):
                classes_added += 1

    for nid, info in ATC_DRUGS.items():
        if _add_drug_node(kg, nid, info):
            drugs_added += 1

    summary = {
        "classes_added": classes_added,
        "drugs_added": drugs_added,
        "total": classes_added + drugs_added,
    }
    logger.info(f"ATC drugs ingestion complete: {summary}")
    return summary


__all__ = ["ingest_atc_drugs", "ATC_CLASSES", "ATC_DRUGS"]
