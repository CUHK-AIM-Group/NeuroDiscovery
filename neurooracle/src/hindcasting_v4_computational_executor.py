"""Outcome-blind hypothesis mapping and computational feedback for hindcasting v4.

This module belongs exclusively to the discovery side of the experiment.  It
maps a frozen-KG imaging-marker/disease hypothesis to one registered TCP
measurement using public candidate metadata, then reveals the corresponding
computational result only after the caller has committed the selected batch.

It intentionally has no dependency on the retrospective literature evaluator.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MAPPER_VERSION = "hindcasting-v4-tcp-semantic-mapper.v1"
CASE1_EXECUTOR_VERSION = "hindcasting-v4-case1-tcp-feedback.v1"
BIOMARKER_EXECUTOR_VERSION = "hindcasting-v4-biomarker-tcp-feedback.v1"


DISEASE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("psychosis_SZ_SZA", re.compile(r"\b(?:schiz\w*|psychosis|schizoaffective)\b", re.I)),
    ("anxiety", re.compile(r"\banxi\w*\b", re.I)),
    ("OCD_OC_related", re.compile(r"\b(?:obsessive|compulsive|ocd)\b", re.I)),
    ("eating_disorder", re.compile(r"\b(?:anorexi\w*|bulimi\w*|eating disorder)\b", re.I)),
    ("bipolar", re.compile(r"\b(?:bipolar|mania|manic)\b", re.I)),
    ("PTSD_trauma", re.compile(r"\b(?:post traumatic|ptsd|trauma\w*)\b", re.I)),
    ("ADHD", re.compile(r"\b(?:attention deficit|hyperactiv\w*|adhd)\b", re.I)),
    ("MDD_depression", re.compile(r"\b(?:depress\w*|mdd)\b", re.I)),
    ("substance_use", re.compile(r"\b(?:substance|addict\w*|alcohol\w*|cocaine|cannabis|opioid\w*)\b", re.I)),
)


FEATURE_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("roi_falff_proxy", "amplitude", re.compile(r"\b(?:falff|fractional amplitude)\b", re.I)),
    ("roi_alff_proxy", "amplitude", re.compile(r"\b(?:alff|amplitude of low)\b", re.I)),
    ("normalized_volume_fraction", "structure", re.compile(r"\b(?:volume|volumetric|voxel based morphometr\w*|gray matter density|grey matter density)\b", re.I)),
    ("corr_node_degree_abs_top10", "correlation_fc", re.compile(r"\b(?:degree centrality|node degree|node strength)\b", re.I)),
    ("partial_mean", "partial_fc", re.compile(r"\bpartial correlation\b", re.I)),
    ("corr_mean", "correlation_fc", re.compile(r"\b(?:functional connect\w*|connectome|correlation|anticorrelation)\b", re.I)),
    ("roi_temporal_std", "temporal", re.compile(r"\b(?:variability|variance|standard deviation|temporal std)\b", re.I)),
    ("roi_temporal_mean", "temporal", re.compile(r"\b(?:mean bold|regional mean signal)\b", re.I)),
)


# The tags are deliberately anatomical/network vocabulary, not scientific
# outcome rules.  A tag must occur in both the hypothesis marker name and the
# public TCP candidate metadata before an outcome can be revealed.
ANATOMY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.I))
    for name, pattern in (
        ("hippocampus", r"hippocamp"),
        ("amygdala", r"amygdal"),
        ("thalamus", r"thalam"),
        ("putamen", r"putamen"),
        ("caudate", r"caudat"),
        ("accumbens", r"accumbens|nucleus accumbens"),
        ("striatum", r"striat"),
        ("cerebellum", r"cerebell|vermis"),
        ("ventricle", r"ventric"),
        ("white_matter", r"white matter"),
        ("corpus_callosum", r"corpus callos"),
        ("brainstem", r"brain ?stem"),
        ("insula", r"insul"),
        ("cingulate", r"cingulat|\bacc\b|\bpcc\b"),
        ("precuneus", r"precune"),
        ("fusiform", r"fusiform"),
        ("orbitofrontal", r"orbitofrontal|orbital frontal"),
        ("prefrontal", r"prefrontal|frontopolar|dlpfc|vmpfc"),
        ("frontal", r"frontal"),
        ("temporal", r"temporal|temporale"),
        ("parietal", r"parietal"),
        ("occipital", r"occipital"),
        ("motor", r"motor|precentral|postcentral|supplementary"),
        ("visual", r"visual|calcarine|cuneus"),
        ("auditory", r"auditory|heschl"),
        ("default_mode", r"default mode|\bdmn\b|default"),
        ("salience", r"salience|salventattn"),
        ("limbic", r"limbic"),
        ("control", r"frontoparietal control|executive control|\bcontrol\b"),
        ("dorsal_attention", r"dorsal attention|dorsattn"),
        ("somatomotor", r"somatomotor|sommot"),
    )
)


FORBIDDEN_PUBLIC_COLUMNS = frozenset(
    {
        "validated",
        "strict_validated",
        "statistically_validated",
        "feedback_status",
        "feedback_utility",
        "p_value",
        "q_fdr_registered_family",
        "q_fdr_disease",
        "q_fdr_global",
        "adjusted_residual_d",
        "execution_succeeded",
        "gt_rank",
        "is_gt_top",
        "is_strict_fdr",
    }
)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def disease_group(name: object) -> str | None:
    text = normalize_text(name)
    for group, pattern in DISEASE_RULES:
        if pattern.search(text):
            return group
    return None


def feature_coordinate(name: object) -> tuple[str, str] | None:
    text = normalize_text(name)
    # These modalities/measurements are not present in the registered TCP
    # executor and therefore fail closed rather than receiving a proxy result.
    if re.search(r"\b(?:pet|spect|eeg|meg|diffusion|dti|fractional anisotropy|cortical thickness)\b", text):
        return None
    for feature, family, pattern in FEATURE_RULES:
        if pattern.search(text):
            return feature, family
    return None


def anatomy_tags(name: object) -> tuple[str, ...]:
    text = normalize_text(name)
    return tuple(tag for tag, pattern in ANATOMY_RULES if pattern.search(text))


def expected_direction(name: object) -> str:
    text = normalize_text(name)
    if re.search(r"\b(?:reduc\w*|decreas\w*|lower|smaller|deficit\w*|hypo\w*)\b", text):
        return "lower"
    if re.search(r"\b(?:increas\w*|higher|larger|elevat\w*|enlarg\w*|hyper\w*)\b", text):
        return "higher"
    return ""


def _hypothesis_marker_and_disease(hypothesis: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(hypothesis.get("source_name") or ""),
        str(hypothesis.get("target_name") or ""),
    )


@dataclass(frozen=True)
class ComputationalMapping:
    status: str
    reason: str
    hypothesis_id: str
    candidate_id: str = ""
    disease: str = ""
    modality: str = ""
    feature: str = ""
    feature_family: str = ""
    anatomy_tag: str = ""
    expected_direction: str = ""

    @property
    def executable(self) -> bool:
        return self.status == "mapped"

    def factors(self) -> tuple[str, ...]:
        if not self.executable:
            return ()
        return (
            f"disease={self.disease}",
            f"modality={self.modality}",
            f"feature_family={self.feature_family}",
            f"anatomy={self.anatomy_tag}",
            f"disease_feature={self.disease}|{self.feature_family}",
        )


class PublicCandidateMapper:
    """Map frozen semantic hypotheses without reading any outcome column."""

    def __init__(self, public: pd.DataFrame) -> None:
        leaked = sorted(FORBIDDEN_PUBLIC_COLUMNS & set(public.columns))
        if leaked:
            raise ValueError(f"public candidate table contains outcome columns: {leaked}")
        required = {"candidate_id", "disease", "modality", "feature", "feature_family"}
        missing = sorted(required - set(public.columns))
        if missing:
            raise ValueError(f"public candidate table misses columns: {missing}")
        self.public = public.reset_index(drop=True).copy()
        text_columns = [
            name
            for name in (
                "roi_name",
                "anatomy",
                "anatomy_full",
                "map_group",
                "network",
                "structure_class",
            )
            if name in self.public.columns
        ]
        combined = self.public[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        row_tags: list[tuple[str, ...]] = [anatomy_tags(value) for value in combined]
        self._row_tags = row_tags
        self._groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index, row in self.public.iterrows():
            disease = str(row["disease"])
            feature = str(row["feature"])
            for tag in row_tags[index]:
                self._groups[(disease, feature, tag)].append(int(index))

    def map_one(
        self,
        hypothesis: Mapping[str, Any],
        *,
        used_candidate_ids: set[str] | None = None,
    ) -> ComputationalMapping:
        hypothesis_id = str(hypothesis.get("id") or "")
        marker, disease_name = _hypothesis_marker_and_disease(hypothesis)
        disease = disease_group(disease_name)
        if disease is None:
            return ComputationalMapping("unmapped", "disease_not_in_registered_tcp_cohort", hypothesis_id)
        coordinate = feature_coordinate(marker)
        if coordinate is None:
            return ComputationalMapping("unmapped", "imaging_measurement_not_registered", hypothesis_id, disease=disease)
        feature, family = coordinate
        tags = anatomy_tags(marker)
        if not tags:
            return ComputationalMapping(
                "unmapped",
                "anatomical_or_network_coordinate_not_resolved",
                hypothesis_id,
                disease=disease,
                feature=feature,
                feature_family=family,
            )

        candidates: dict[int, int] = defaultdict(int)
        for tag in tags:
            for index in self._groups.get((disease, feature, tag), ()):
                candidates[index] += 1
        if not candidates:
            return ComputationalMapping(
                "unmapped",
                "no_public_tcp_coordinate_matches_all_required_axes",
                hypothesis_id,
                disease=disease,
                feature=feature,
                feature_family=family,
            )

        best_overlap = max(candidates.values())
        eligible = sorted(
            index for index, overlap in candidates.items() if overlap == best_overlap
        )
        semantic_key = "\x1f".join(
            (
                str(hypothesis.get("source_id") or marker),
                str(hypothesis.get("target_id") or disease_name),
                disease,
                feature,
                "|".join(tags),
            )
        )
        offset = int.from_bytes(hashlib.sha256(semantic_key.encode("utf-8")).digest()[:8], "big")
        used = used_candidate_ids if used_candidate_ids is not None else set()
        chosen: int | None = None
        for step in range(len(eligible)):
            index = eligible[(offset + step) % len(eligible)]
            candidate_id = str(self.public.iloc[index]["candidate_id"])
            if candidate_id not in used:
                chosen = index
                break
        if chosen is None:
            return ComputationalMapping(
                "unmapped",
                "all_matching_tcp_coordinates_already_committed",
                hypothesis_id,
                disease=disease,
                feature=feature,
                feature_family=family,
            )
        row = self.public.iloc[chosen]
        candidate_id = str(row["candidate_id"])
        used.add(candidate_id)
        matching_tags = [tag for tag in tags if tag in self._row_tags[chosen]]
        return ComputationalMapping(
            "mapped",
            "strict_disease_measurement_anatomy_match",
            hypothesis_id,
            candidate_id=candidate_id,
            disease=disease,
            modality=str(row["modality"]),
            feature=feature,
            feature_family=str(row["feature_family"] or family),
            anatomy_tag=matching_tags[0],
            expected_direction=expected_direction(marker),
        )

    def map_all(self, hypotheses: Sequence[Mapping[str, Any]]) -> list[ComputationalMapping]:
        used: set[str] = set()
        return [self.map_one(row, used_candidate_ids=used) for row in hypotheses]


class ComputationalOutcomeVault:
    """Read a task-specific result table only for a committed selection."""

    def __init__(self, outcomes: pd.DataFrame, *, task_id: str) -> None:
        if "candidate_id" not in outcomes.columns:
            raise ValueError("computational outcome vault misses candidate_id")
        if outcomes["candidate_id"].astype(str).duplicated().any():
            raise ValueError("computational outcome vault contains duplicate candidate IDs")
        self.task_id = str(task_id)
        self.lookup = outcomes.copy()
        self.lookup["candidate_id"] = self.lookup["candidate_id"].astype(str)
        self.lookup = self.lookup.set_index("candidate_id", drop=False)

    def reveal(self, mapping: ComputationalMapping) -> dict[str, Any]:
        if not mapping.executable:
            return {
                "execution_status": "failed",
                "feedback_status": "execution_failed",
                "feedback_available": False,
                "feedback_utility": None,
                "statistics": {},
                "failure_reason": mapping.reason,
            }
        if mapping.candidate_id not in self.lookup.index:
            return {
                "execution_status": "failed",
                "feedback_status": "execution_failed",
                "feedback_available": False,
                "feedback_utility": None,
                "statistics": {},
                "failure_reason": "mapped_coordinate_absent_from_outcome_vault",
            }
        row = self.lookup.loc[mapping.candidate_id]
        if isinstance(row, pd.DataFrame):
            raise ValueError("computational outcome lookup is not unique")
        if self.task_id == "biomarker_discovery":
            return self._biomarker_result(row)
        return self._case1_result(row, mapping.expected_direction)

    @staticmethod
    def _case1_result(row: pd.Series, direction: str) -> dict[str, Any]:
        succeeded = bool(row.get("execution_succeeded", False))
        effect = float(pd.to_numeric(row.get("adjusted_residual_d"), errors="coerce"))
        p_value = float(pd.to_numeric(row.get("p_value"), errors="coerce"))
        finite = math.isfinite(effect) and math.isfinite(p_value)
        if not succeeded or not finite:
            return {
                "execution_status": "failed",
                "feedback_status": "execution_failed",
                "feedback_available": False,
                "feedback_utility": None,
                "statistics": {},
                "failure_reason": "tcp_analysis_or_qc_failed",
            }
        strong = p_value < 0.01 and abs(effect) >= 0.15
        opposed = strong and (
            (direction == "lower" and effect > 0)
            or (direction == "higher" and effect < 0)
        )
        status = "contradicted" if opposed else ("supported" if strong else "inconclusive")
        utility = min(1.0, abs(effect) / 0.8) * min(
            1.0,
            max(0.0, -math.log10(max(p_value, 1e-300))) / 4.0,
        )
        return {
            "execution_status": "succeeded",
            "feedback_status": status,
            "feedback_available": True,
            "feedback_utility": float(utility),
            "statistics": {
                "adjusted_residual_d": effect,
                "p_value": p_value,
                "expected_direction": direction,
            },
            "failure_reason": "",
        }

    @staticmethod
    def _biomarker_result(row: pd.Series) -> dict[str, Any]:
        succeeded = bool(row.get("execution_succeeded", False))
        status = str(row.get("feedback_status") or "").strip().casefold()
        if not succeeded or status == "execution_failed":
            return {
                "execution_status": "failed",
                "feedback_status": "execution_failed",
                "feedback_available": False,
                "feedback_utility": None,
                "statistics": {},
                "failure_reason": "biomarker_tcp_analysis_or_qc_failed",
            }
        if status not in {"supported", "contradicted", "inconclusive"}:
            status = "supported" if bool(row.get("validated", False)) else "inconclusive"
        utility = float(pd.to_numeric(row.get("feedback_utility"), errors="coerce"))
        if not math.isfinite(utility):
            utility = 1.0 if status == "supported" else 0.0
        utility = min(1.0, max(0.0, utility))
        statistics: dict[str, Any] = {}
        for field in (
            "adjusted_residual_d",
            "p_value",
            "q_fdr_registered_family",
            "cv_auc_mean",
            "cv_direction_stability",
            "direction",
        ):
            value = row.get(field)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            statistics[field] = value
        return {
            "execution_status": "succeeded",
            "feedback_status": status,
            "feedback_available": bool(row.get("feedback_available", True)),
            "feedback_utility": utility,
            "statistics": statistics,
            "failure_reason": "",
        }


class FeedbackRanker:
    """Small prespecified utility overlay for subsequent-batch ranking."""

    def __init__(
        self,
        mappings: Sequence[ComputationalMapping],
        *,
        feedback_weight: float = 0.10,
        pair_feedback_weight: float = 0.08,
        exploration_weight: float = 0.015,
        prior_strength: float = 5.0,
    ) -> None:
        self.mappings = list(mappings)
        self.feedback_weight = float(feedback_weight)
        self.pair_feedback_weight = float(pair_feedback_weight)
        self.exploration_weight = float(exploration_weight)
        self.prior_strength = float(prior_strength)
        self._sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self.utility_sum = 0.0
        self.utility_count = 0

    def update(self, mapping: ComputationalMapping, result: Mapping[str, Any]) -> None:
        utility = result.get("feedback_utility")
        if not mapping.executable or utility is None:
            return
        value = float(utility)
        if not math.isfinite(value):
            return
        value = min(1.0, max(0.0, value))
        self.utility_sum += value
        self.utility_count += 1
        for factor in mapping.factors():
            self._sum[factor] += value
            self._count[factor] += 1

    @property
    def informative_records(self) -> int:
        return self.utility_count

    def adjustments(self, *, total_observations: int) -> np.ndarray:
        out = np.zeros(len(self.mappings), dtype=float)
        if not self.utility_count:
            return out
        global_mean = self.utility_sum / self.utility_count
        for index, mapping in enumerate(self.mappings):
            factors = mapping.factors()
            if not factors:
                continue
            single = [factor for factor in factors if not factor.startswith("disease_feature=")]
            pairs = [factor for factor in factors if factor.startswith("disease_feature=")]

            def posterior(factor: str) -> float:
                count = self._count[factor]
                return (
                    self._sum[factor] + self.prior_strength * global_mean
                ) / (count + self.prior_strength)

            single_delta = np.mean([posterior(factor) - global_mean for factor in single])
            pair_delta = np.mean([posterior(factor) - global_mean for factor in pairs])
            least_count = min((self._count[factor] for factor in factors), default=0)
            exploration = math.sqrt(
                math.log(max(2, total_observations + 2)) / (least_count + 1)
            )
            out[index] = (
                self.feedback_weight * float(single_delta)
                + self.pair_feedback_weight * float(pair_delta)
                + self.exploration_weight * exploration
            )
        return out


def mapping_summary(mappings: Iterable[ComputationalMapping]) -> dict[str, Any]:
    rows = list(mappings)
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        reasons[row.reason] += 1
    return {
        "mapper_version": MAPPER_VERSION,
        "total": len(rows),
        "mapped": sum(row.executable for row in rows),
        "unmapped": sum(not row.executable for row in rows),
        "reason_counts": dict(sorted(reasons.items())),
    }


__all__ = [
    "BIOMARKER_EXECUTOR_VERSION",
    "CASE1_EXECUTOR_VERSION",
    "ComputationalMapping",
    "ComputationalOutcomeVault",
    "FeedbackRanker",
    "MAPPER_VERSION",
    "PublicCandidateMapper",
    "anatomy_tags",
    "disease_group",
    "expected_direction",
    "feature_coordinate",
    "mapping_summary",
    "normalize_text",
]
