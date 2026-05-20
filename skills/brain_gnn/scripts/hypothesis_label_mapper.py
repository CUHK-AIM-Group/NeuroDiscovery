"""Map hypothesis target_name to HCP-YA phenotype labels.

Expanded mapping that covers all 55 HCP label dimensions, organized by
semantic category. Each hypothesis target can map to multiple relevant
HCP labels (multi-label evaluation).

HCP Label Categories:
  - demographics: gender, age
  - cognitive_executive: flanker, cardsort, listsort, wm_0bk_acc, wm_2bk_acc, pmat24
  - cognitive_language: picvocab, readeng
  - cognitive_composite: cogfluid, cogfluidcomp, cogcrystal
  - processing_speed: procspeed
  - memory: iwrd_tot, iwrd_rtc, vsplot_tc
  - social_cognition: tom, social_tom_perc_tom, social_random_perc_random, er40_cr
  - emotion: emotion_acc, emotion_face_acc, neg_affect, pos_affect, fearsomat
  - personality: neuroticism, extraversion, openness, neofac_a, neofac_c
  - psychiatric: percstress, anghostil, angaggr, perchostil, percreject, loneliness
  - substance: ddisc_auc_40k
  - sleep: psqi
  - social_support: emotsupp, friendship, lifesatisf, meanpurp
  - motor: endurance, gaitspeed_comp, dexterity, strength
  - pain: paininterf_tscore
  - attention: scpt_sen, scpt_spec
  - task_performance: gambling_reward_perc_larger, language_story_avg_difficulty_level,
                      language_math_avg_difficulty_level, relational_acc, relational_rel_acc
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

# ─── HCP Label Category Definitions ──────────────────────────────────────────

HCP_LABEL_CATEGORIES = {
    "demographics": {
        "gender": ("hcp_gender_labels.csv", "classification"),
        "age": ("hcp_age_labels.csv", "regression"),
    },
    "cognitive_executive": {
        "flanker": ("hcp_flanker_labels.csv", "regression"),
        "cardsort": ("hcp_cardsort_labels.csv", "regression"),
        "listsort": ("hcp_listsort_labels.csv", "regression"),
        "wm_0bk_acc": ("hcp_wm_0bk_acc_labels.csv", "regression"),
        "wm_2bk_acc": ("hcp_wm_2bk_acc_labels.csv", "regression"),
        "pmat24": ("hcp_pmat24_labels.csv", "regression"),
    },
    "cognitive_language": {
        "picvocab": ("hcp_picvocab_labels.csv", "regression"),
        "readeng": ("hcp_readeng_labels.csv", "regression"),
    },
    "cognitive_composite": {
        "cogfluid": ("hcp_cogfluid_labels.csv", "regression"),
        "cogfluidcomp": ("hcp_cogfluidcomp_labels.csv", "regression"),
        "cogcrystal": ("hcp_cogcrystal_labels.csv", "regression"),
    },
    "processing_speed": {
        "procspeed": ("hcp_procspeed_labels.csv", "regression"),
    },
    "memory": {
        "iwrd_tot": ("hcp_iwrd_tot_labels.csv", "regression"),
        "iwrd_rtc": ("hcp_iwrd_rtc_labels.csv", "regression"),
        "vsplot_tc": ("hcp_vsplot_tc_labels.csv", "regression"),
    },
    "social_cognition": {
        "tom": ("hcp_tom_labels.csv", "regression"),
        "social_tom": ("hcp_social_tom_perc_tom_labels.csv", "regression"),
        "social_random": ("hcp_social_random_perc_random_labels.csv", "regression"),
        "er40_cr": ("hcp_er40_cr_labels.csv", "regression"),
    },
    "emotion": {
        "emotion_acc": ("hcp_emotion_acc_labels.csv", "regression"),
        "emotion_face_acc": ("hcp_emotion_face_acc_labels.csv", "regression"),
        "neg_affect": ("hcp_neg_affect_labels.csv", "regression"),
        "pos_affect": ("hcp_pos_affect_labels.csv", "regression"),
        "fearsomat": ("hcp_fearsomat_labels.csv", "regression"),
    },
    "personality": {
        "neuroticism": ("hcp_neuroticism_labels.csv", "regression"),
        "extraversion": ("hcp_extraversion_labels.csv", "regression"),
        "openness": ("hcp_openness_labels.csv", "regression"),
        "agreeableness": ("hcp_neofac_a_labels.csv", "regression"),
        "conscientiousness": ("hcp_neofac_c_labels.csv", "regression"),
    },
    "psychiatric_distress": {
        "percstress": ("hcp_percstress_labels.csv", "regression"),
        "anghostil": ("hcp_anghostil_labels.csv", "regression"),
        "angaggr": ("hcp_angaggr_labels.csv", "regression"),
        "perchostil": ("hcp_perchostil_labels.csv", "regression"),
        "percreject": ("hcp_percreject_labels.csv", "regression"),
        "loneliness": ("hcp_loneliness_labels.csv", "regression"),
    },
    "substance_reward": {
        "ddisc_auc_40k": ("hcp_ddisc_auc_40k_labels.csv", "regression"),
    },
    "sleep": {
        "psqi": ("hcp_psqi_labels.csv", "regression"),
    },
    "social_support": {
        "emotsupp": ("hcp_emotsupp_labels.csv", "regression"),
        "friendship": ("hcp_friendship_labels.csv", "regression"),
        "lifesatisf": ("hcp_lifesatisf_labels.csv", "regression"),
        "meanpurp": ("hcp_meanpurp_labels.csv", "regression"),
    },
    "motor": {
        "endurance": ("hcp_endurance_labels.csv", "regression"),
        "gaitspeed": ("hcp_gaitspeed_comp_labels.csv", "regression"),
        "dexterity": ("hcp_dexterity_labels.csv", "regression"),
        "strength": ("hcp_strength_labels.csv", "regression"),
    },
    "pain": {
        "paininterf": ("hcp_paininterf_tscore_labels.csv", "regression"),
    },
    "attention": {
        "scpt_sen": ("hcp_scpt_sen_labels.csv", "regression"),
        "scpt_spec": ("hcp_scpt_spec_labels.csv", "regression"),
    },
    "task_performance": {
        "gambling_reward": ("hcp_gambling_reward_perc_larger_labels.csv", "regression"),
        "language_story": ("hcp_language_story_avg_difficulty_level_labels.csv", "regression"),
        "language_math": ("hcp_language_math_avg_difficulty_level_labels.csv", "regression"),
        "relational_acc": ("hcp_relational_acc_labels.csv", "regression"),
        "relational_rel_acc": ("hcp_relational_rel_acc_labels.csv", "regression"),
    },
}

# ─── Hypothesis target → HCP label mapping ───────────────────────────────────
# Each target maps to: (primary_label_key, category, rationale)
# primary_label_key references a key in HCP_LABEL_CATEGORIES[category]

SEMANTIC_MAP: dict[str, list[tuple[str, str, str]]] = {
    # Cognitive function targets
    "impulsivity": [
        ("flanker", "cognitive_executive", "inhibitory control ~ impulsivity"),
        ("ddisc_auc_40k", "substance_reward", "delay discounting ~ impulsive choice"),
        ("cardsort", "cognitive_executive", "cognitive flexibility ~ impulse regulation"),
    ],
    "Negotiating": [
        ("social_tom", "social_cognition", "theory of mind ~ social negotiation"),
        ("er40_cr", "social_cognition", "emotion recognition ~ social skill"),
    ],
    "Social Interaction": [
        ("social_tom", "social_cognition", "ToM ~ social interaction"),
        ("friendship", "social_support", "social network ~ interaction quality"),
        ("loneliness", "psychiatric_distress", "inverse of social interaction"),
    ],
    "Social Skills": [
        ("social_tom", "social_cognition", "ToM ~ social competence"),
        ("er40_cr", "social_cognition", "emotion recognition ~ social skill"),
        ("emotsupp", "social_support", "emotional support ~ social skill outcome"),
    ],
    "Affect": [
        ("neg_affect", "emotion", "negative affect"),
        ("pos_affect", "emotion", "positive affect"),
        ("emotion_acc", "emotion", "emotion task accuracy"),
    ],
    "Psychological Distress": [
        ("percstress", "psychiatric_distress", "perceived stress"),
        ("neg_affect", "emotion", "negative affect ~ distress"),
        ("loneliness", "psychiatric_distress", "loneliness ~ distress"),
    ],
    "anxiety sensitivity": [
        ("neuroticism", "personality", "neuroticism ~ anxiety proneness"),
        ("fearsomat", "emotion", "fear/somatic anxiety"),
        ("percstress", "psychiatric_distress", "stress sensitivity"),
    ],
    "social phobia": [
        ("neuroticism", "personality", "neuroticism ~ social anxiety"),
        ("percreject", "psychiatric_distress", "perceived rejection ~ social fear"),
        ("loneliness", "psychiatric_distress", "social withdrawal"),
    ],
    "Shyness": [
        ("neuroticism", "personality", "neuroticism ~ shyness"),
        ("extraversion", "personality", "inverse: low extraversion ~ shyness"),
        ("percreject", "psychiatric_distress", "social sensitivity"),
    ],
    "substance abuse": [
        ("ddisc_auc_40k", "substance_reward", "delay discounting ~ substance risk"),
        ("angaggr", "psychiatric_distress", "aggression ~ externalizing"),
    ],
    "conduct disorder": [
        ("anghostil", "psychiatric_distress", "anger/hostility ~ conduct problems"),
        ("angaggr", "psychiatric_distress", "aggression ~ conduct"),
        ("perchostil", "psychiatric_distress", "perceived hostility"),
    ],
    "Depressive Disorder": [
        ("neg_affect", "emotion", "negative affect ~ depression"),
        ("percstress", "psychiatric_distress", "stress ~ depression risk"),
        ("lifesatisf", "social_support", "inverse: low life satisfaction"),
        ("psqi", "sleep", "sleep disturbance ~ depression"),
    ],
    "Neurodevelopment": [
        ("cogfluid", "cognitive_composite", "fluid cognition ~ neurodevelopment"),
        ("cogcrystal", "cognitive_composite", "crystallized cognition"),
        ("procspeed", "processing_speed", "processing speed ~ maturation"),
    ],
    "valence": [
        ("pos_affect", "emotion", "positive valence"),
        ("neg_affect", "emotion", "negative valence"),
        ("emotion_face_acc", "emotion", "valence discrimination"),
    ],
    "context": [
        ("wm_2bk_acc", "cognitive_executive", "working memory ~ context maintenance"),
        ("wm_0bk_acc", "cognitive_executive", "baseline WM"),
        ("relational_acc", "task_performance", "relational reasoning ~ context"),
    ],
    "suicidality": [
        ("percstress", "psychiatric_distress", "stress ~ suicidal ideation"),
        ("neg_affect", "emotion", "negative affect ~ hopelessness"),
        ("loneliness", "psychiatric_distress", "social isolation ~ risk"),
    ],
    "Suicide": [
        ("percstress", "psychiatric_distress", "stress ~ suicidal ideation"),
        ("neg_affect", "emotion", "negative affect ~ hopelessness"),
        ("loneliness", "psychiatric_distress", "social isolation ~ risk"),
    ],
    "unconscious process": [
        ("cogfluid", "cognitive_composite", "fluid cognition ~ implicit processing"),
        ("procspeed", "processing_speed", "processing speed ~ automatic processing"),
    ],
    "psychosis": [
        ("neuroticism", "personality", "neuroticism ~ psychosis proneness"),
        ("percstress", "psychiatric_distress", "stress ~ psychotic experiences"),
        ("perchostil", "psychiatric_distress", "perceived hostility ~ paranoia"),
    ],
    "Apathy": [
        ("neg_affect", "emotion", "negative affect ~ apathy"),
        ("meanpurp", "social_support", "inverse: low meaning/purpose"),
        ("lifesatisf", "social_support", "inverse: low life satisfaction"),
    ],
    "insomnia": [
        ("psqi", "sleep", "sleep quality ~ insomnia"),
        ("neg_affect", "emotion", "negative affect ~ sleep disturbance"),
        ("percstress", "psychiatric_distress", "stress ~ insomnia"),
    ],
    # Disease targets — map to closest behavioral proxy
    "Attention Deficit Disorder with Hyperactivity": [
        ("flanker", "cognitive_executive", "inhibition deficit ~ ADHD"),
        ("scpt_sen", "attention", "sustained attention ~ ADHD"),
        ("scpt_spec", "attention", "attention specificity"),
        ("ddisc_auc_40k", "substance_reward", "impulsive choice ~ ADHD"),
    ],
    "autism spectrum disorder": [
        ("social_tom", "social_cognition", "ToM deficit ~ ASD"),
        ("er40_cr", "social_cognition", "emotion recognition ~ ASD"),
        ("social_random", "social_cognition", "social perception"),
    ],
    "bipolar disorder": [
        ("neg_affect", "emotion", "mood dysregulation"),
        ("pos_affect", "emotion", "elevated mood episodes"),
        ("psqi", "sleep", "sleep disruption ~ mania"),
        ("angaggr", "psychiatric_distress", "irritability ~ bipolar"),
    ],
    "Epilepsy": [
        ("cogfluid", "cognitive_composite", "cognitive impairment ~ epilepsy"),
        ("procspeed", "processing_speed", "slowed processing"),
        ("iwrd_tot", "memory", "memory impairment"),
    ],
    "Brain Injuries": [
        ("cogfluid", "cognitive_composite", "cognitive decline ~ TBI"),
        ("procspeed", "processing_speed", "slowed processing ~ TBI"),
        ("iwrd_tot", "memory", "memory impairment ~ TBI"),
    ],
    "Brain Injuries, Traumatic": [
        ("cogfluid", "cognitive_composite", "cognitive decline ~ TBI"),
        ("procspeed", "processing_speed", "slowed processing ~ TBI"),
        ("iwrd_tot", "memory", "memory impairment ~ TBI"),
    ],
    "Hallucinations": [
        ("neuroticism", "personality", "neuroticism ~ hallucination proneness"),
        ("perchostil", "psychiatric_distress", "perceived hostility ~ paranoia"),
    ],
    "Movement Disorders": [
        ("dexterity", "motor", "fine motor ~ movement disorder"),
        ("gaitspeed", "motor", "gait ~ movement disorder"),
        ("strength", "motor", "motor strength"),
    ],
    "Capgras syndrome": [
        ("er40_cr", "social_cognition", "face/emotion recognition ~ misidentification"),
        ("neuroticism", "personality", "neuroticism ~ delusional ideation"),
    ],
    "addiction": [
        ("ddisc_auc_40k", "substance_reward", "delay discounting ~ addiction"),
        ("angaggr", "psychiatric_distress", "aggression ~ substance use"),
    ],
    "stimulant misuse": [
        ("ddisc_auc_40k", "substance_reward", "impulsive choice ~ stimulant use"),
        ("flanker", "cognitive_executive", "inhibition ~ stimulant effects"),
    ],
    "anxiety disorder": [
        ("neuroticism", "personality", "neuroticism ~ anxiety"),
        ("fearsomat", "emotion", "fear/somatic symptoms"),
        ("percstress", "psychiatric_distress", "perceived stress ~ anxiety"),
    ],
    "attention": [
        ("scpt_sen", "attention", "sustained attention sensitivity"),
        ("scpt_spec", "attention", "attention specificity"),
        ("flanker", "cognitive_executive", "attentional control"),
    ],
    "recognition": [
        ("er40_cr", "social_cognition", "emotion recognition"),
        ("iwrd_tot", "memory", "word recognition memory"),
        ("emotion_face_acc", "emotion", "face recognition accuracy"),
    ],
    "Neuronal Plasticity": [
        ("cogfluid", "cognitive_composite", "fluid cognition ~ plasticity"),
        ("pmat24", "cognitive_executive", "reasoning ~ plasticity"),
    ],
    "Arousal": [
        ("fearsomat", "emotion", "somatic arousal"),
        ("percstress", "psychiatric_distress", "stress arousal"),
    ],
    "Anxiety": [
        ("neuroticism", "personality", "neuroticism ~ anxiety"),
        ("fearsomat", "emotion", "fear/somatic anxiety"),
        ("percstress", "psychiatric_distress", "stress ~ anxiety"),
    ],
}

# Targets that truly cannot be mapped to any HCP behavioral measure
UNMAPPABLE = {
    "Neural Tube Defects",
    "neuropsychiatric lupus",
    "immune-mediated encephalitides",
    "Hydrocephalus",
    "Hypoxia, Brain",
    "Migraine Disorders",
    "Parkinson Disease",
    "Tourette Syndrome",
    "schizophrenia",
}


def get_label_info(label_key: str, category: str) -> Optional[tuple[Path, str]]:
    """Get CSV path and task type for a label key in a category."""
    cat = HCP_LABEL_CATEGORIES.get(category, {})
    entry = cat.get(label_key)
    if entry is None:
        return None
    csv_name, task = entry
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        return None
    return csv_path, task


def map_hypothesis_to_labels(target_name: str) -> list[dict]:
    """Map a hypothesis target to all relevant HCP labels.

    Returns list of dicts: {label_key, category, csv_path, task_type, rationale}
    """
    if target_name in UNMAPPABLE:
        return []
    mappings = SEMANTIC_MAP.get(target_name, [])
    if not mappings:
        return []

    results = []
    for label_key, category, rationale in mappings:
        info = get_label_info(label_key, category)
        if info is None:
            continue
        csv_path, task = info
        results.append({
            "label_key": label_key,
            "category": category,
            "csv_path": str(csv_path),
            "task_type": task,
            "rationale": rationale,
        })
    return results


# Backward-compatible: return primary (first) mapping only
def map_hypothesis_to_label(target_name: str) -> Optional[tuple[Path, str]]:
    """Return (label_csv_path, task_type) for primary mapping, or None."""
    mappings = map_hypothesis_to_labels(target_name)
    if not mappings:
        return None
    m = mappings[0]
    return Path(m["csv_path"]), m["task_type"]


def get_all_mappable_hypotheses(hypotheses: list[dict]) -> list[dict]:
    """Filter hypotheses to those with valid HCP label mappings.

    Returns list of dicts with added keys: label_csv, task_type, all_mappings.
    """
    results = []
    for h in hypotheses:
        target = h["target_name"]
        mappings = map_hypothesis_to_labels(target)
        if not mappings:
            continue
        primary = mappings[0]
        entry = {
            **h,
            "label_csv": primary["csv_path"],
            "task_type": primary["task_type"],
            "label_key": primary["label_key"],
            "label_category": primary["category"],
            "all_mappings": mappings,
        }
        results.append(entry)
    return results


def get_baseline_tasks() -> list[dict]:
    """Return sex (classification) and age (regression) baseline tasks."""
    baselines = []
    # Sex classification
    sex_path = DATA_DIR / "hcp_gender_labels.csv"
    if sex_path.exists():
        baselines.append({
            "id": "BASELINE:sex",
            "target_name": "sex",
            "label_csv": str(sex_path),
            "task_type": "classification",
            "label_key": "gender",
            "label_category": "demographics",
            "hypothesis_type": "baseline",
            "metadata": {"outcome_type": "demographics", "input_region": "whole_brain"},
            "explanation": "Baseline: sex classification from whole-brain FC (no KG prior)",
            "confidence_score": 1.0,
            "all_mappings": [{"label_key": "gender", "category": "demographics",
                              "csv_path": str(sex_path), "task_type": "classification",
                              "rationale": "biological sex from brain connectivity"}],
        })
    # Age regression
    age_path = DATA_DIR / "hcp_age_labels.csv"
    if age_path.exists():
        baselines.append({
            "id": "BASELINE:age",
            "target_name": "age",
            "label_csv": str(age_path),
            "task_type": "regression",
            "label_key": "age",
            "label_category": "demographics",
            "hypothesis_type": "baseline",
            "metadata": {"outcome_type": "demographics", "input_region": "whole_brain"},
            "explanation": "Baseline: brain age prediction from whole-brain FC (no KG prior)",
            "confidence_score": 1.0,
            "all_mappings": [{"label_key": "age", "category": "demographics",
                              "csv_path": str(age_path), "task_type": "regression",
                              "rationale": "brain age from connectivity"}],
        })
    return baselines


def get_experiment_groups(hypotheses: list[dict]) -> dict[str, list[dict]]:
    """Group hypotheses by their primary HCP label category for reporting."""
    groups: dict[str, list[dict]] = {}
    for h in hypotheses:
        cat = h.get("label_category", "unknown")
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(h)
    return groups


if __name__ == "__main__":
    import json

    hyp_path = ROOT / "neurooracle" / "data" / "quick" / "hypotheses_imaging_hcp.json"
    with open(hyp_path) as f:
        data = json.load(f)

    mappable = get_all_mappable_hypotheses(data["hypotheses"])
    baselines = get_baseline_tasks()

    print(f"Total hypotheses: {data['n_hypotheses']}")
    print(f"Mappable: {len(mappable)} (was 25, now expanded)")
    print(f"Unmappable: {data['n_hypotheses'] - len(mappable)}")
    print(f"Baselines: {len(baselines)}")
    print()

    # Show by category
    groups = get_experiment_groups(mappable)
    print("=== By HCP Label Category ===")
    for cat, hyps in sorted(groups.items()):
        print(f"\n  [{cat}] ({len(hyps)} hypotheses)")
        for h in hyps:
            n_labels = len(h["all_mappings"])
            print(f"    {h['id']} | {h['target_name']:30s} -> {h['label_key']} (+{n_labels-1} more)")

    print("\n=== Baselines ===")
    for b in baselines:
        print(f"  {b['id']} | {b['target_name']} -> {b['label_key']} ({b['task_type']})")
