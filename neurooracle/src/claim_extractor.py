"""LLM-based structured claim extraction from neuroscience paper abstracts.

Uses an adaptive LLM cascade via proxy endpoint to extract structured scientific claims
as (Subject, Predicate, Object, Evidence) triples.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import OpenAI

from .schema import Claim, Evidence, PaperRef

logger = logging.getLogger(__name__)

# ── LLM Configuration ──────────────────────────────────────────────

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "claude-sonnet-4-6")

# Multi-key pool: set OPENAI_API_KEYS env var as comma-separated keys
# e.g. export OPENAI_API_KEYS="sk-aaa,sk-bbb,sk-ccc"
# Falls back to OPENAI_API_KEY if not set
_API_KEYS_RAW = os.environ.get("OPENAI_API_KEYS", "")
API_KEY_POOL = [k.strip() for k in _API_KEYS_RAW.split(",") if k.strip()] or (
    [DEFAULT_API_KEY] if DEFAULT_API_KEY else []
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

EXTRACTION_PROMPT = """Extract ALL scientific claims from this neuroscience paper abstract as JSON array.

Each claim object fields:
- subject, subject_type, subject_canonical_hint, subject_atlas
- predicate, object, object_type, object_canonical_hint, object_atlas, negated
- effect_metric, effect_size, p_value, sample_size
- study_type, methodology, replicability, direction, raw_sentence
- conditions: list of conditions under which this claim holds (e.g. ["female only", "age > 65", "resting-state fMRI"]). Empty list [] if unconditional.
- population: object with study population info, null if not reported:
  {{"mean_age": number or null, "age_range": "e.g. 18-65" or null, "n_female": int or null, "n_male": int or null, "ethnicity": str or null, "cohort_name": str or null}}

IMPORTANT rules for numeric fields:
- p_value: output the exact number if reported (e.g. 0.003), or a comparison string like "p < 0.05" or "p > 0.01", or "not_reported" if the abstract does not mention a p-value.
- effect_size: output the number (e.g. 0.45, 1.2), or "not_reported" if not mentioned.
- sample_size: output the integer (e.g. 150, 2048), or "not_reported" if not mentioned.
- effect_metric: output the metric name (e.g. "Cohen's d", "odds ratio", "AUC", "beta"), or "not_reported" if not mentioned.
- NEVER output null for these four fields — use "not_reported" instead.

Entity types (aligned to the 7-atom alphabet — DISEASE / DRUG / IMAGING_MARKER /
GENE_TARGET / COGNITIVE_TASK / OUTCOME / INDIVIDUAL_DATA):
- disease            → DISEASE atom    (e.g. "Alzheimer disease", "schizophrenia")
- drug               → DRUG atom       (e.g. "donepezil", "ketamine", "lithium")
                       Use the INN (international non-proprietary name), NOT brand
                       names: "fluoxetine" not "Prozac"; "haloperidol" not "Haldol".
- brain_region       → IMAGING_MARKER  (e.g. "hippocampus", "dorsolateral prefrontal cortex")
- network            → IMAGING_MARKER  (e.g. "default mode network", "salience network",
                       "frontoparietal network", "ventral attention network")
- biomarker          → IMAGING_MARKER  (e.g. "amyloid SUVR", "cortical thickness", "FA",
                       "ALFF", "ReHo", "functional connectivity strength")
- gene               → GENE_TARGET     HGNC primary symbol (e.g. "APOE", "BDNF",
                       "DRD2", "COMT", "MAPT", "SNCA")
- neurotransmitter   → GENE_TARGET     For receptor / transporter measurements,
                       output the receptor GENE symbol when identifiable:
                       serotonin → "HTR1A"/"HTR1B"/"HTR2A"/"HTR4"/"SLC6A4";
                       dopamine  → "DRD1"/"DRD2"/"SLC6A3";
                       norepinephrine → "SLC6A2"; acetylcholine → "CHRM1"/"CHRNA4"/"SLC18A3";
                       GABA → "GABRA1"; glutamate (mGluR5) → "GRM5"; opioid → "OPRM1";
                       cannabinoid → "CNR1"; histamine → "HRH3"; SV2A → "SV2A".
                       Fall back to neurotransmitter name only if no receptor/transporter
                       is specified.
- protein            → GENE_TARGET     (e.g. "tau protein" → "MAPT", "alpha-synuclein"
                       → "SNCA", "amyloid-beta" → "APP")
- cognitive_function → COGNITIVE_TASK  (e.g. "working memory", "n-back", "Stroop task",
                       "MID task", "go/no-go", "emotional faces")
- rating_scale       → OUTCOME         Established clinical / behavioral rating
                       scales (e.g. "HAM-D", "MMSE", "MDS-UPDRS", "PANSS").
                       Use the OUTCOME:<scale> hint when the scale matches the
                       enum listed below.
- adverse_event      → OUTCOME         Drug / treatment side-effects (e.g.
                       "hepatotoxicity", "akathisia", "tardive dyskinesia",
                       "extrapyramidal symptoms"). LEAVE the hint EMPTY — these
                       are descriptive, not enum entries. Apply the canonical
                       naming discipline (full term, expanded abbreviations).
- clinical_event     → OUTCOME         Clinical events / symptoms / outcomes
                       that are NOT rating scales (e.g. "seizure", "stroke",
                       "encephalopathy", "cerebral edema", "all-cause mortality",
                       "post-surgical outcome", "treatment response").
                       LEAVE the hint EMPTY — descriptive, not enum.
- individual_data    → INDIVIDUAL_DATA Demographic / lifestyle / non-imaging biological
                       individual variables (e.g. "age", "sex", "APOE-ε4 status",
                       "polygenic risk score", "BMI", "blood pressure", "education",
                       "smoking", "diet", "socioeconomic status", "Big Five personality").

subject_canonical_hint / object_canonical_hint (OPTIONAL — fill in when you recognize a
standard ID; otherwise output ""). Do NOT try to guess UMLS CUIs — name-only resolution
will be performed downstream. For the entities below, when you DO know the ID, prefer
these prefix conventions:
- GENE:<HGNC_symbol>       human gene/protein (e.g. "GENE:APOE", "GENE:BDNF", "GENE:HTR2A").
                           Always uppercase HGNC primary symbol — not gene names.
- NN:<id>                  NeuroNames brain region (e.g. "NN:11" for hippocampus).
                           Only use if you are certain of the NN id; otherwise leave empty.
- ATC:<code>               drug ATC code (e.g. "ATC:N06DA02" donepezil,
                           "ATC:N06AB03" fluoxetine, "ATC:N03AX14" levetiracetam).
- OUTCOME:<scale>          ONLY for established rating scales matching this enum.
                           Do NOT use this prefix for adverse events, symptoms, or
                           clinical events — for those, leave the hint "" and rely
                           on descriptive naming. The KG currently knows:
                           OUTCOME:HAM-D / MADRS / BDI / PHQ-9 / HAM-A / GAD-7 / MMSE /
                           MoCA / CDR-SB / NPI / MDS-UPDRS / PDQ-39 / PANSS / BPRS / SANS /
                           YMRS / CGI / GAF / ADHD-RS / ADOS / SRS / ALSFRS-R / EDSS / MSFC /
                           UHDRS / NIHSS / PSQI / ESS / SF-36 / WHODAS / QOLIE-31 / BPI / AE_GI.
- INDIVIDUAL_DATA:<anchor> known anchors: aging, sex, education, body_mass_index,
                           blood_pressure, polygenic_risk_score, socioeconomic_status, diet,
                           big5_openness / big5_conscientiousness / big5_extraversion /
                           big5_agreeableness / big5_neuroticism.
- COGAT_TASK:<id>          Cognitive Atlas task/concept (use only if you recognize the id).
- COGAT_CONCEPT:<id>       Cognitive Atlas cognitive concept.
- COGAT_DISORDER:<id>      Cognitive Atlas disorder.
- ATLAS:<name>             parcellation atlas (DK / Schaefer400 / Aseg / AAL / HO / Dosenbach).

HARD RULE for hints: every hint MUST either be the empty string "" OR start with one of
these exact prefixes followed by a colon: GENE: / NN: / ATC: / OUTCOME: / INDIVIDUAL_DATA:
/ COGAT_TASK: / COGAT_CONCEPT: / COGAT_DISORDER: / ATLAS:. Anything else (e.g. raw
descriptive phrases like "tau pathology", "episodic memory", "schizophrenia") is INVALID
— in those cases leave the hint "" and put the descriptive form into `subject` / `object`
text instead.

For all other entities (especially diseases, brain regions without an NN id, networks,
biomarkers, cognitive functions, individual_data without a known anchor), the hint stays
"" and you instead make the `subject` / `object` text itself as canonical as possible:
- expand abbreviations: "AD" → "Alzheimer disease", "MDD" → "major depressive disorder",
  "DLPFC" → "dorsolateral prefrontal cortex", "DMN" → "default mode network",
  "ACC" → "anterior cingulate cortex", "OFC" → "orbitofrontal cortex".
- use the full standard term, not lab jargon: "polygenic risk score" not "PRS";
  "amyloid-beta 42" not "Aβ42".
- prefer the form a MeSH/UMLS browser would return for that concept.
This descriptive-name discipline is more useful than guessing CUI codes — downstream
ingestion will alias-match these names against the canonical KG.

subject_atlas / object_atlas (OPTIONAL, only for IMAGING_MARKER brain_region entries):
If the abstract EXPLICITLY names the parcellation/atlas the region was measured in,
output one of: "DK", "Schaefer400", "Aseg", "AAL", "HO" (Harvard-Oxford), "Brainnetome",
"Glasser", "Dosenbach", "Power", "Yeo7", "Yeo17".
HARD RULE: if the abstract does NOT explicitly name an atlas (e.g. it just says
"hippocampus" or "default mode network" with no parcellation reference), you MUST
output empty string "". DO NOT infer the atlas from modality, region type, or common
practice. Inference here is a hallucination — leave it blank.
For network-level entries: same rule. "DMN" by itself → "". "DMN as defined by Yeo's
7-network parcellation" → "Yeo7".

Predicates — CLOSED SET. The `predicate` field MUST be one of these exact strings; any
other value is invalid. Pick the MOST SPECIFIC one that fits the abstract's language:

  Causal / mechanistic:  causes, treats, inhibits, activates, increases, reduces, modulates
  Predictive:            is_biomarker_of, is_risk_factor_for, predicts, distinguishes
  Correlational:         correlates_with, mediates
  Drug-target:           binds_to               (drug → receptor / transporter gene)
  Adverse-effect:        has_adverse_effect     (drug → AE / symptom / clinical event)
  Discovery (gene maps): gene_associated_with_disease   (gene → disease, GWAS/DisGeNET)
                         gene_associated_with_anatomy   (gene → brain region)
                         gene_enriched_in_region        (gene → brain region, AHBA)
                         receptor_density_in            (receptor gene → brain region, PET)
  Vague fallback:        is_associated_with     — last-resort ONLY when the abstract
                         itself uses vague language ("X is associated with Y") and no
                         direction or mechanism is reported.

Common verb → canonical-predicate mappings (do NOT emit the left side):
- "induces", "leads to", "produces" → causes
- "ameliorates", "improves", "alleviates" → treats
- "preserves", "protects", "maintains" → treats   (when the protected outcome is a
   clinical / cognitive measure that disease would otherwise worsen)
- "facilitates", "aids" → causes (if mechanistic) OR predicts (if outcome-level)
- "reflects", "indexes" → is_biomarker_of
- "progresses toward", "predicts conversion to" → predicts

CRITICAL: Choose the most precise predicate based on the study design and language:
- RCT / intervention → "treats" or "causes"
- Longitudinal / prospective → "is_risk_factor_for" or "predicts"
- Diagnostic accuracy → "is_biomarker_of" or "distinguishes"
- Molecular mechanism → "activates", "inhibits", "increases", "reduces"
- Cross-sectional correlation → "correlates_with"
- Drug pharmacology / receptor binding → "binds_to" (drug → receptor gene)
- Drug side-effect language ("X causes/induces/leads to Y" where Y is an AE) → "has_adverse_effect"
- GWAS hit / DisGeNET-style gene-disease report → "gene_associated_with_disease"
- AHBA / spatial gene expression → "gene_enriched_in_region"
- PET receptor density mapping → "receptor_density_in"
- If the abstract says "X increases Y" or "X reduces Y", use "increases" or "reduces", NOT "is_associated_with"

Directional measurement comparisons — IMPORTANT:
If a biomarker, imaging measure, cognitive score, receptor binding measure, blood
flow measure, volume, density, activation level, or ratio is reported as increased,
decreased, reduced, lower, higher, diminished, or elevated IN a disease/patient group,
do NOT emit `MEASURE reduces DISEASE` or `MEASURE increases DISEASE`.

Those sentences mean the MEASURE differs by disease/status. Use:
- `MEASURE is_biomarker_of DISEASE` when the disease/status is the endpoint.
- `MEASURE distinguishes DISEASE_OR_GROUP` when the sentence explicitly compares
  groups or diagnostic classes.
- `MEASURE correlates_with OUTCOME` when the endpoint is severity, survival,
  score, performance, or another continuous clinical/cognitive outcome.

Bad examples — DO NOT emit:
- subject="microscopic fractional anisotropy", predicate="reduces", object="temporal lobe epilepsy"
- subject="nicotinic acetylcholine receptor binding", predicate="reduces", object="Alzheimer disease"
- subject="right hippocampal volume", predicate="reduces", object="Alzheimer disease"
- subject="regional cerebral blood flow", predicate="reduces", object="severity of dementia"

Correct examples:
- Text: "Microscopic fractional anisotropy was reduced in TLE patients."
  Claim: subject="microscopic fractional anisotropy", predicate="is_biomarker_of",
  object="temporal lobe epilepsy"
- Text: "Regional cerebral blood flow was lower and related to dementia severity."
  Claim: subject="regional cerebral blood flow", predicate="correlates_with",
  object="severity of dementia"
- Text: "Hippocampal volume was reduced in Alzheimer disease compared with controls."
  Claim: subject="hippocampal volume", predicate="distinguishes",
  object="Alzheimer disease"

Method/procedure handling — IMPORTANT:
Methods, algorithms, pipelines, classifiers, registration/segmentation procedures,
software tools, statistical models, and validation protocols are NOT biomedical
entities for `subject` or `object`. Put them in `methodology` instead.

When a sentence has the form "Using METHOD, we found MEASURE/MARKER relates to
DISEASE/OUTCOME", extract the real MEASURE/MARKER as the subject/object and store
METHOD in `methodology`.

Good examples:
- Text: "Using fluid registration, patients with Alzheimer disease showed faster
  hippocampal atrophy."
  Claim: subject="hippocampal atrophy", predicate="is_biomarker_of" or "distinguishes",
  object="Alzheimer disease", methodology="fluid registration"
- Text: "Manual segmentation showed reduced hippocampal volume in Alzheimer disease."
  Claim: subject="hippocampal volume", predicate="reduces" or "is_biomarker_of",
  object="Alzheimer disease", methodology="manual segmentation"
- Text: "A support vector machine using cortical thickness distinguished MCI converters
  from non-converters."
  Claim: subject="cortical thickness", predicate="distinguishes", object="MCI conversion",
  methodology="support vector machine"

Bad examples — DO NOT emit claims like these:
- subject="fluid registration", predicate="is_biomarker_of", object="Alzheimer disease"
- subject="manual segmentation", predicate="has_adverse_effect", object="hippocampal volume"
- subject="classifier", predicate="predicts", object="conversion"

If the abstract only evaluates a method/procedure itself (e.g. reproducibility,
accuracy, registration error, segmentation reliability) and does NOT state a biomedical
marker-disease/outcome relationship, return no claim for that sentence rather than
guessing a biomarker.

Method-validation endpoints such as "scan-rescan consistency", "volume repeatability",
"segmentation reproducibility", "registration error", "measurement accuracy", "gold
standard comparison", and "inter-rater reliability" are method-performance outcomes,
not disease biomarkers. Do NOT emit them as subject/object biomedical entities.
Do NOT turn an objective sentence like "to validate METHOD for measuring MARKER in
DISEASE" into a biomarker claim unless the abstract reports an actual disease/outcome
finding for MARKER.
The `raw_sentence` must explicitly support BOTH endpoints and the predicate. A disease
cohort mentioned only in the sample description (e.g. "15 controls and 12 Alzheimer
disease patients") is not enough. Do NOT infer "MARKER reduces in DISEASE" from a
method-comparison sentence unless the sentence says the marker is reduced, increased,
different, predictive, diagnostic, or associated with the disease/outcome.

Pure imaging modality names are methods, not biomarkers. Do NOT use CT, computed
tomography, PET, positron emission tomography, FDG-PET, amyloid PET, SPECT,
single-photon emission tomography, single photon emission tomography,
single-photon emission computed tomography, single photon emission computed
tomography, MRI, magnetic resonance imaging, structural MRI, structural magnetic
resonance imaging, fMRI, functional MRI, functional magnetic resonance imaging,
DTI, diffusion tensor imaging, diffusion MRI, diffusion magnetic resonance imaging,
EEG, electroencephalography, MEG, or magnetoencephalography as the subject of
`is_biomarker_of`, `predicts`, or `distinguishes`.

Instead, extract the concrete measurement produced by the modality when the abstract
states one. Good subjects include "FDG hypometabolism", "amyloid PET SUVR",
"entorhinal cortical thickness", "hippocampal volume", "fractional anisotropy",
"functional connectivity", "regional cerebral blood flow", or "dopamine transporter
binding". Put the modality itself in `methodology`.

Bad examples — DO NOT emit:
- subject="computed tomography", predicate="is_biomarker_of", object="reversible causes of dementia"
- subject="positron emission tomography", predicate="is_biomarker_of", object="memory function"
- subject="functional magnetic resonance imaging", predicate="predicts", object="Alzheimer disease"
- subject="single-photon emission tomography scanning", predicate="distinguishes", object="Alzheimer disease"
- subject="APOE", predicate="distinguishes", object="quantitative MRI measurements"
- subject="this technique", predicate="predicts", object="Alzheimer disease"

Real error examples from neuroimaging review abstracts — these sentences describe
modalities or literature focus, not concrete biomarker findings. Return NO CLAIM
for these sentences unless a concrete measurement is named:
- Text: "Computed tomography is still used to determine reversible causes of dementia."
  Wrong: subject="computed tomography", predicate="is_biomarker_of",
  object="reversible causes of dementia"
  Correct: no claim.
- Text: "Of the new techniques, functional magnetic resonance imaging seems the most promising."
  Wrong: subject="functional magnetic resonance imaging", predicate="predicts",
  object="Alzheimer disease"
  Correct: no claim.
- Text: "This technique can possibly play a role in predicting Alzheimer's disease in patients with mild cognitive impairment."
  Wrong: subject="functional magnetic resonance imaging", predicate="predicts",
  object="Alzheimer disease"
  Correct: no claim, because "this technique" refers to a modality and no concrete
  measurement is named.
- Text: "The use of single-photon emission computed tomography and positron emission
  tomography in early differential diagnoses seems limited."
  Wrong: subject="single-photon emission computed tomography", predicate="distinguishes",
  object="early differential diagnoses"
  Correct: no claim.
- Text: "Current and future positron emission tomography studies concentrate on memory
  function and receptor imaging."
  Wrong: subject="positron emission tomography", predicate="predicts",
  object="memory function"
  Correct: no claim.

Good examples:
- subject="FDG hypometabolism", predicate="is_biomarker_of", object="Alzheimer disease",
  methodology="FDG-PET"
- subject="amyloid PET SUVR", predicate="predicts", object="cognitive decline",
  methodology="amyloid PET"
- subject="entorhinal cortical thickness", predicate="distinguishes",
  object="MCI conversion", methodology="structural MRI"

Study types: fMRI, PET, DTI, sMRI, EEG, MEG, lesion, meta_analysis, GWAS, animal_model, clinical_trial, case_control, longitudinal, cross_sectional, review, cohort, narrative_review

Title: {title}
PMID: {pmid}
Abstract: {abstract}

Return JSON array. Empty array [] if no claims."""


@dataclass
class ExtractionResult:
    """Result of claim extraction from a single paper."""
    paper: PaperRef
    claims: list[Claim]
    raw_response: str = ""
    error: str = ""


# Map free-form predicate verbs the LLM occasionally emits despite the closed-set
# instruction back into the canonical predicate. Applied in `_item_to_claim`.
# Keep this list short and only for verbs we've actually observed in smoke runs.
PREDICATE_ALIAS = {
    "induces":               "causes",
    "leads to":              "causes",
    "produces":              "causes",
    "occurs early along":    "causes",
    "ameliorates":           "treats",
    "improves":              "treats",
    "alleviates":            "treats",
    "preserves":             "treats",
    "protects":              "treats",
    "reflects":              "is_biomarker_of",
    "indexes":               "is_biomarker_of",
    "progresses toward":     "predicts",
    "progress toward":       "predicts",
    "facilitates":           "is_associated_with",
    "aids":                  "is_associated_with",
    "provides":              "is_associated_with",
    "does not increase":     "is_associated_with",
    "precedes":              "predicts",
    "compensates_for":       "modulates",
    "compensates for":       "modulates",
}

EXTRACTION_PREDICATES = frozenset({
    "causes", "treats", "inhibits", "activates", "increases", "reduces", "modulates",
    "is_biomarker_of", "is_risk_factor_for", "predicts", "distinguishes",
    "correlates_with", "mediates",
    "binds_to", "has_adverse_effect",
    "gene_associated_with_disease", "gene_associated_with_anatomy",
    "gene_enriched_in_region", "receptor_density_in",
    "is_associated_with",
})

# Hints must start with one of these prefixes; anything else is dropped to "" by
# `_sanitize_hint`. Mirrors the HARD RULE in EXTRACTION_PROMPT.
ALLOWED_HINT_PREFIXES = (
    "GENE:", "NN:", "ATC:", "OUTCOME:", "INDIVIDUAL_DATA:",
    "COGAT_TASK:", "COGAT_CONCEPT:", "COGAT_DISORDER:", "ATLAS:",
)


def _normalize_predicate(raw: str) -> str:
    p = (raw or "").strip()
    if not p:
        return p
    return PREDICATE_ALIAS.get(p.lower(), p)


_CONTINUOUS_ENDPOINT_RE = re.compile(
    r"\b(severity|survival|score|scores|performance|function|outcome|"
    r"outcomes|decline|impairment)\b",
    re.I,
)
_ASSOCIATION_CUE_RE = re.compile(
    r"\b(related to|correlat(?:e|es|ed|ion|ions)? with|associated with|"
    r"relationship with)\b",
    re.I,
)


def _normalize_directional_comparison_predicate(
    predicate: str,
    obj: str,
    raw_sentence: str,
) -> str:
    """Avoid turning "lower X is related to severity" into "X reduces severity"."""
    if predicate not in {"reduces", "increases"}:
        return predicate
    if not _CONTINUOUS_ENDPOINT_RE.search(obj or ""):
        return predicate
    if not _ASSOCIATION_CUE_RE.search(raw_sentence or ""):
        return predicate
    return "correlates_with"


def _sanitize_hint(raw: str) -> str:
    h = (raw or "").strip()
    if not h:
        return ""
    return h if h.startswith(ALLOWED_HINT_PREFIXES) else ""


MODEL_CASCADE = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "deepseek-v3.2",
]
DOWNGRADE_THRESHOLD = 2      # consecutive failures to trigger downgrade
UPGRADE_INTERVAL = 600       # seconds before attempting upgrade back to preferred model (slow recovery)


class _WorkerCascade:
    """Per-worker adaptive model cascade state."""

    def __init__(self, worker_id: int, cascade: list[str], preferred_idx: int = 0):
        self.worker_id = worker_id
        self._cascade = cascade
        self._model_idx = preferred_idx
        self._consecutive_failures = 0
        self._last_upgrade_attempt = 0.0

    @property
    def model(self) -> str:
        import time as _time
        import random as _random
        now = _time.time()
        if self._model_idx > 0 and (now - self._last_upgrade_attempt) > UPGRADE_INTERVAL:
            self._last_upgrade_attempt = now
            prev = self._cascade[self._model_idx]
            # Randomly jump to any higher-priority model (idx in [0, current_idx))
            # rather than always stepping up one tier — explores all faster models
            # after extended downgrades.
            self._model_idx = _random.randint(0, self._model_idx - 1)
            self._consecutive_failures = 0
            logger.info(f"[worker-{self.worker_id}] model upgrade attempt: {prev} -> {self._cascade[self._model_idx]}")
        return self._cascade[self._model_idx]

    def record_success(self):
        self._consecutive_failures = 0

    def record_failure(self):
        import time as _time
        self._consecutive_failures += 1
        if self._consecutive_failures >= DOWNGRADE_THRESHOLD:
            if self._model_idx < len(self._cascade) - 1:
                prev = self._cascade[self._model_idx]
                self._model_idx += 1
                self._consecutive_failures = 0
                self._last_upgrade_attempt = _time.time()
                logger.warning(f"[worker-{self.worker_id}] model downgrade: {prev} -> {self._cascade[self._model_idx]} (after {DOWNGRADE_THRESHOLD} failures)")
            else:
                self._consecutive_failures = 0


class ClaimExtractor:
    """Extract structured claims from paper abstracts using LLM.

    Supports multi-key round-robin to bypass per-key rate limits on proxy APIs.
    Set OPENAI_API_KEYS env var as comma-separated keys.

    Each worker thread gets its own adaptive model cascade, starting from the
    strongest currently stable claim-extraction model and falling back to
    lower-cost alternatives,
    downgrading independently on repeated failures and periodically retrying higher models.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        api_keys: list[str] | None = None,
        lock_model: bool | None = None,
    ):
        self.preferred_model = model
        self.base_url = base_url
        self.lock_model = _env_flag("OPENAI_LOCK_MODEL", False) if lock_model is None else lock_model

        if self.lock_model:
            self._cascade = [model]
            self._preferred_idx = 0
        else:
            self._cascade = MODEL_CASCADE if model in MODEL_CASCADE else [model] + MODEL_CASCADE
            self._preferred_idx = self._cascade.index(model) if model in self._cascade else 0

        # Build client pool from explicit keys, env pool, or single key
        keys = api_keys or API_KEY_POOL or ([api_key] if api_key else [])
        if not keys:
            raise ValueError("No API keys provided. Set OPENAI_API_KEYS or OPENAI_API_KEY env var.")

        self._clients: list[OpenAI] = []
        for k in keys:
            self._clients.append(
                OpenAI(
                    base_url=base_url,
                    api_key=k,
                    http_client=httpx.Client(verify=False, timeout=30.0),
                    max_retries=0,  # disable internal retry; let cascade handle failures
                    timeout=30.0,
                )
            )
        self._client_idx = 0
        self._client_lock = __import__("threading").Lock()

        # Per-worker cascade state, keyed by thread id
        self._worker_cascades: dict[int, _WorkerCascade] = {}
        self._worker_lock = __import__("threading").Lock()
        self._worker_counter = 0

        logger.info(
            f"initialized {len(self._clients)} LLM client(s), model cascade: "
            f"{self._cascade} (lock_model={self.lock_model})"
        )

    def _get_worker_cascade(self) -> _WorkerCascade:
        """Get or create cascade state for the current thread."""
        import threading
        tid = threading.current_thread().ident
        with self._worker_lock:
            if tid not in self._worker_cascades:
                self._worker_counter += 1
                self._worker_cascades[tid] = _WorkerCascade(
                    self._worker_counter, self._cascade, self._preferred_idx
                )
            return self._worker_cascades[tid]

    @property
    def client(self) -> OpenAI:
        """Round-robin client selection (thread-safe)."""
        with self._client_lock:
            c = self._clients[self._client_idx % len(self._clients)]
            self._client_idx += 1
            return c

    def extract_from_abstract(
        self,
        abstract: str,
        paper: PaperRef,
        full_text: str | None = None,
        full_text_max_chars: int = 16000,
    ) -> ExtractionResult:
        """Extract claims from a single paper.

        By default operates on the abstract. Pass `full_text` to re-extract from
        the full body when the abstract under-covers the paper's claims (e.g.
        targeted refresh of high-value anchors). The body is truncated to
        `full_text_max_chars` to bound LLM cost and context.
        """
        if full_text:
            body = full_text if len(full_text) <= full_text_max_chars else full_text[:full_text_max_chars] + "..."
            source_label = "Full text"
        else:
            body = abstract if len(abstract) <= 2000 else abstract[:2000] + "..."
            source_label = "Abstract"

        prompt = EXTRACTION_PROMPT.format(
            abstract=f"[{source_label}] {body}",
            pmid=paper.pmid or "unknown",
            title=paper.title or "unknown",
            authors=paper.authors or "unknown",
            year=paper.year or "unknown",
            journal=paper.journal or "unknown",
        )

        cascade = self._get_worker_cascade()
        backoff = 5.0
        slow_response_threshold = float(os.environ.get("OPENAI_SLOW_RESPONSE_THRESHOLD", "30"))
        for attempt in range(4):
            current_model = cascade.model
            import time as _time
            req_start = _time.time()
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": "You are a precise neuroscience data extraction system. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                )

                latency = _time.time() - req_start
                raw_text = response.choices[0].message.content.strip()
                claims = self._parse_response(raw_text, paper)

                # Slow response = soft failure (no exception thrown but service degraded)
                if latency > slow_response_threshold:
                    cascade.record_failure()
                    logger.debug(f"slow response PMID {paper.pmid} model={current_model}: {latency:.1f}s")
                else:
                    cascade.record_success()

                return ExtractionResult(
                    paper=paper,
                    claims=claims,
                    raw_response=raw_text,
                )

            except Exception as e:
                err_str = str(e)
                cascade.record_failure()
                if ("429" in err_str or "rate" in err_str.lower() or "forbidden" in err_str.lower() or "timed out" in err_str.lower() or "connection" in err_str.lower()) and attempt < 3:
                    import time as _time
                    logger.warning(f"request failed PMID {paper.pmid} model={current_model} (attempt {attempt+1}), backing off {backoff:.0f}s: {err_str[:80]}")
                    _time.sleep(backoff)
                    backoff *= 2
                    continue
                logger.error(f"extraction failed for PMID {paper.pmid}: {e}")
                return ExtractionResult(
                    paper=paper,
                    claims=[],
                    error=str(e),
                )
        return ExtractionResult(paper=paper, claims=[], error="max retries exceeded")

    def extract_batch(
        self,
        papers: list[tuple[str, PaperRef]],
        max_workers: int = 1,
    ) -> list[ExtractionResult]:
        """Extract claims from multiple papers with per-worker model cascade."""
        if max_workers <= 1:
            results = []
            for i, (abstract, paper) in enumerate(papers):
                logger.info(f"extracting claims [{i+1}/{len(papers)}] PMID={paper.pmid}")
                result = self.extract_from_abstract(abstract, paper)
                logger.info(f"  extracted {len(result.claims)} claims")
                results.append(result)
            return results

        results: list[Optional[ExtractionResult]] = [None] * len(papers)

        def _extract_one(idx: int, abstract: str, paper: PaperRef) -> tuple[int, ExtractionResult]:
            logger.info(f"extracting claims [{idx+1}/{len(papers)}] PMID={paper.pmid}")
            result = self.extract_from_abstract(abstract, paper)
            logger.info(f"  extracted {len(result.claims)} claims (PMID={paper.pmid})")
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_extract_one, i, abstract, paper)
                for i, (abstract, paper) in enumerate(papers)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _parse_response(self, raw_text: str, paper: PaperRef) -> list[Claim]:
        """Parse LLM JSON response into Claim objects."""
        # try to extract JSON array from response
        json_str = self._extract_json(raw_text)
        if not json_str:
            logger.warning(f"no JSON found in response for PMID {paper.pmid}")
            return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for PMID {paper.pmid}: {e}")
            return []

        if not isinstance(data, list):
            data = [data]

        claims = []
        for item in data:
            try:
                claim = self._item_to_claim(item, paper)
                if claim:
                    claims.append(claim)
            except Exception as e:
                logger.warning(f"failed to parse claim item: {e}")
                continue

        return claims

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON array or object from LLM response text."""
        text = text.strip()

        # try direct parse first
        if text.startswith("[") or text.startswith("{"):
            return text

        # fix common LLM error: double brackets [[...]]
        if text.startswith("[["):
            text = text[1:]

        # try to find JSON in markdown code block
        match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if match:
            return match.group(1).strip()

        # try to find JSON array in text
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group(0)

        return None

    @staticmethod
    def _parse_numeric(value, target_type: str = "float"):
        """Parse a numeric field from LLM output.

        Handles: pure numbers, comparison strings ("p < 0.05", "n = 150"),
        and "not_reported". Returns (parsed_value, raw_string) tuple.
        """
        if value is None:
            return None, "not_reported"

        if isinstance(value, (int, float)):
            return value, str(value)

        s = str(value).strip().lower()
        if s in ("not_reported", "n/a", "na", "none", "", "null"):
            return None, "not_reported"

        # try direct numeric parse
        try:
            v = float(s) if target_type == "float" else int(s)
            return v, s
        except (ValueError, TypeError):
            pass

        # extract number from comparison strings like "p < 0.05", "n = 150", "β = -0.32"
        # strip commas from numbers like "2,048"
        s_clean = s.replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", s_clean)
        if match:
            try:
                v = float(match.group()) if target_type == "float" else int(float(match.group()))
                return v, s
            except (ValueError, TypeError):
                pass

        return None, s

    def _item_to_claim(self, item: dict, paper: PaperRef) -> Optional[Claim]:
        """Convert a single JSON item to a Claim object."""
        subject = item.get("subject", "").strip()
        obj = item.get("object", "").strip()
        predicate = _normalize_predicate(item.get("predicate", ""))
        raw_sentence = item.get("raw_sentence", "")
        predicate = _normalize_directional_comparison_predicate(predicate, obj, raw_sentence)

        if not subject or not obj or not predicate:
            return None
        if predicate not in EXTRACTION_PREDICATES:
            logger.debug(
                "skipped claim with invalid predicate %r: %r -> %r",
                predicate, subject, obj,
            )
            return None

        # parse numeric fields with range/comparison support
        effect_size, effect_size_raw = self._parse_numeric(item.get("effect_size"), "float")
        p_value, p_value_raw = self._parse_numeric(item.get("p_value"), "float")
        sample_size, sample_size_raw = self._parse_numeric(item.get("sample_size"), "int")

        # store raw strings in metadata for downstream use
        raw_stats = {}
        if p_value_raw != "not_reported":
            raw_stats["p_value_raw"] = p_value_raw
        if effect_size_raw != "not_reported":
            raw_stats["effect_size_raw"] = effect_size_raw
        if sample_size_raw != "not_reported":
            raw_stats["sample_size_raw"] = sample_size_raw

        effect_metric = item.get("effect_metric", "")
        if isinstance(effect_metric, str) and effect_metric.lower() in ("not_reported", "n/a", ""):
            effect_metric = ""

        evidence = Evidence(
            study_type=item.get("study_type", ""),
            methodology=item.get("methodology", ""),
            p_value=p_value,
            effect_size=effect_size,
            effect_metric=effect_metric,
            sample_size=sample_size,
            replicability=item.get("replicability", "single_study"),
            direction=item.get("direction", ""),
        )

        # generate claim ID
        claim_id = f"CLM:{uuid.uuid4().hex[:12]}"

        # parse conditions and population (contextualized triplets)
        conditions = item.get("conditions") or []
        if not isinstance(conditions, list):
            conditions = [str(conditions)]

        population = item.get("population")
        if isinstance(population, dict):
            # normalize numeric fields
            for key in ("mean_age", "n_female", "n_male"):
                if population.get(key) is not None:
                    try:
                        population[key] = float(population[key]) if key == "mean_age" else int(population[key])
                    except (ValueError, TypeError):
                        population[key] = None
        else:
            population = None

        return Claim(
            id=claim_id,
            subject_id="",  # will be resolved during ingestion
            subject_name=subject,
            predicate=predicate,
            object_id="",   # will be resolved during ingestion
            object_name=obj,
            negated=bool(item.get("negated", False)),
            confidence=self._estimate_confidence(evidence),
            evidence=evidence,
            source_paper=paper,
            raw_text=raw_sentence,
            metadata={
                "subject_type": item.get("subject_type", ""),
                "object_type": item.get("object_type", ""),
                "subject_canonical_hint": _sanitize_hint(item.get("subject_canonical_hint") or ""),
                "object_canonical_hint":  _sanitize_hint(item.get("object_canonical_hint") or ""),
                "subject_atlas": (item.get("subject_atlas") or "").strip(),
                "object_atlas":  (item.get("object_atlas") or "").strip(),
                "conditions": conditions,
                "population": population,
                "raw_stats": raw_stats,
            },
        )

    def _estimate_confidence(self, evidence: Evidence) -> float:
        """Estimate claim confidence based on evidence quality."""
        score = 0.5  # baseline

        # p-value boost
        if evidence.p_value is not None:
            if evidence.p_value < 0.001:
                score += 0.2
            elif evidence.p_value < 0.01:
                score += 0.15
            elif evidence.p_value < 0.05:
                score += 0.1

        # sample size boost
        if evidence.sample_size is not None:
            if evidence.sample_size > 1000:
                score += 0.15
            elif evidence.sample_size > 100:
                score += 0.1
            elif evidence.sample_size > 30:
                score += 0.05

        # replicability boost
        if evidence.replicability == "replicated":
            score += 0.15
        elif evidence.replicability == "controversial":
            score -= 0.1

        # meta-analysis boost
        if evidence.study_type == "meta_analysis":
            score += 0.15

        return min(max(score, 0.0), 1.0)
