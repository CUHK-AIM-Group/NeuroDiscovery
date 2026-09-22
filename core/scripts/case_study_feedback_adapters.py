"""Task semantics and feedback classification for closed-loop experiments.

The formal literature KG is immutable during an experiment.  These adapters
project an executable candidate into claim-like semantic assertions that can be
stored in a per-seed experimental overlay and reused by the next discovery
round without pretending that an experiment is a paper-derived claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SUPPORTED = "supported"
CONTRADICTED = "contradicted"
INCONCLUSIVE = "inconclusive"
EXECUTION_FAILED = "execution_failed"
FEEDBACK_STATUSES = frozenset(
    {SUPPORTED, CONTRADICTED, INCONCLUSIVE, EXECUTION_FAILED}
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _bool(value: object, *, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    return default


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stable_atom_id(atom_type: str, label: str) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"type": atom_type, "label": label},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"EXP_ATOM:{atom_type}:{digest}"


@dataclass(frozen=True)
class RelationTemplate:
    subject_fields: tuple[str, ...]
    subject_type: str
    predicate: str
    object_fields: tuple[str, ...]
    object_type: str
    subject_label: str = ""
    object_label: str = ""


@dataclass(frozen=True)
class CaseStudyFeedbackAdapter:
    task: str
    case_study_id: str
    factor_fields: tuple[str, ...]
    relations: tuple[RelationTemplate, ...]
    qualifier_fields: tuple[str, ...] = ()
    requires_complete_chain: bool = False

    def _atom(
        self,
        candidate: Mapping[str, Any],
        fields: Sequence[str],
        atom_type: str,
        fixed_label: str,
    ) -> dict[str, str]:
        values = [_text(candidate.get(field)) for field in fields]
        values = [value for value in values if value]
        label = fixed_label or " | ".join(values)
        if not label:
            label = "unspecified"
        return {
            "id": _stable_atom_id(atom_type, label),
            "name": label,
            "type": atom_type,
        }

    def semantic_assertions(
        self, candidate: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        qualifiers = {
            field: _text(candidate.get(field))
            for field in self.qualifier_fields
            if _text(candidate.get(field))
        }
        assertions: list[dict[str, Any]] = []
        for index, relation in enumerate(self.relations):
            subject = self._atom(
                candidate,
                relation.subject_fields,
                relation.subject_type,
                relation.subject_label,
            )
            obj = self._atom(
                candidate,
                relation.object_fields,
                relation.object_type,
                relation.object_label,
            )
            assertions.append(
                {
                    "step": index + 1,
                    "subject_id": subject["id"],
                    "subject_name": subject["name"],
                    "subject_type": subject["type"],
                    "predicate": relation.predicate,
                    "object_id": obj["id"],
                    "object_name": obj["name"],
                    "object_type": obj["type"],
                    "qualifiers": qualifiers,
                }
            )
        return assertions

    def classify_feedback(self, outcome: Mapping[str, Any]) -> str:
        explicit = _text(outcome.get("feedback_status")).casefold()
        if explicit:
            if explicit not in FEEDBACK_STATUSES:
                raise ValueError(f"unknown feedback status: {explicit!r}")
            return explicit

        error = _text(outcome.get("error"))
        if error or (
            "execution_succeeded" in outcome
            and not _bool(outcome.get("execution_succeeded"), default=True)
        ):
            return EXECUTION_FAILED
        if _bool(outcome.get("validated")):
            return SUPPORTED

        # Contradiction is only asserted when a directional hypothesis was
        # preregistered and a statistically supported opposite effect exists.
        expected = _text(outcome.get("expected_direction")).casefold()
        effect = _finite_float(
            outcome.get("effect_size", outcome.get("effect", outcome.get("adjusted_residual_d")))
        )
        q_value = _finite_float(
            outcome.get("q_value", outcome.get("q_fdr"))
        )
        if expected and effect is not None and q_value is not None and q_value < 0.05:
            expects_negative = expected in {
                "decrease",
                "decreased",
                "negative",
                "lower",
                "down",
            }
            expects_positive = expected in {
                "increase",
                "increased",
                "positive",
                "higher",
                "up",
            }
            if (expects_negative and effect > 0) or (expects_positive and effect < 0):
                return CONTRADICTED
        return INCONCLUSIVE


_ADAPTERS: dict[str, CaseStudyFeedbackAdapter] = {}


def _register(adapter: CaseStudyFeedbackAdapter, *aliases: str) -> None:
    for name in (adapter.task, *aliases):
        _ADAPTERS[name] = adapter


_register(
    CaseStudyFeedbackAdapter(
        task="case1_transdiagnostic",
        case_study_id="case1_transdiagnostic",
        factor_fields=("disease", "atlas", "anatomy", "feature_family", "feature"),
        relations=(
            RelationTemplate(
                ("disease",),
                "DISEASE",
                "has_imaging_marker",
                ("atlas", "anatomy", "feature_family", "feature"),
                "IMAGING_MARKER",
            ),
        ),
        qualifier_fields=("modality", "source", "model"),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="biomarker_discovery",
        case_study_id="biomarker_discovery",
        factor_fields=(
            "disease",
            "atlas",
            "roi_index",
            "anatomy",
            "feature_family",
            "feature",
        ),
        relations=(
            RelationTemplate(
                ("disease",),
                "DISEASE",
                "has_imaging_marker",
                ("atlas", "roi_index", "anatomy", "feature_family", "feature"),
                "IMAGING_MARKER",
            ),
        ),
        qualifier_fields=("modality", "source", "model"),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="case2_pathway_mediation",
        case_study_id="case2_pathway_mediation",
        factor_fields=("exposure", "modality", "marker", "outcome"),
        relations=(
            RelationTemplate(
                ("exposure",),
                "GENE_PATHWAY",
                "affects",
                ("modality", "marker"),
                "IMAGING_MARKER",
            ),
            RelationTemplate(
                ("modality", "marker"),
                "IMAGING_MARKER",
                "mediates",
                ("outcome",),
                "CLINICAL_OUTCOME",
            ),
        ),
        requires_complete_chain=True,
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="differential_diagnosis",
        case_study_id="differential_diagnosis",
        factor_fields=("diagnostic_contrast", "atlas", "feature_family", "model"),
        relations=(
            RelationTemplate(
                ("atlas", "feature_family"),
                "IMAGING_SIGNATURE",
                "distinguishes",
                ("diagnostic_contrast",),
                "DIAGNOSTIC_CONTRAST",
            ),
        ),
        qualifier_fields=("model",),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="disease_subtyping",
        case_study_id="disease_subtyping",
        factor_fields=("disease", "atlas", "feature_family", "model", "cluster_count"),
        relations=(
            RelationTemplate(
                ("atlas", "feature_family"),
                "IMAGING_SIGNATURE",
                "stratifies",
                ("disease",),
                "DISEASE",
            ),
        ),
        qualifier_fields=("model", "cluster_count"),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="connectome_behavior",
        case_study_id="connectome_behavior",
        factor_fields=("phenotype", "atlas", "feature_family", "model"),
        relations=(
            RelationTemplate(
                ("atlas", "feature_family"),
                "CONNECTOME_SIGNATURE",
                "predicts",
                ("phenotype",),
                "BEHAVIORAL_OUTCOME",
            ),
        ),
        qualifier_fields=("model",),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="brain_age",
        case_study_id="brain_age",
        factor_fields=("modality", "atlas", "feature_family", "model"),
        relations=(
            RelationTemplate(
                ("modality", "atlas", "feature_family"),
                "IMAGING_SIGNATURE",
                "predicts",
                (),
                "CLINICAL_OUTCOME",
                object_label="brain age",
            ),
        ),
        qualifier_fields=("model",),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="progression_prediction",
        case_study_id="progression_prediction",
        factor_fields=("outcome", "horizon", "feature_family", "model"),
        relations=(
            RelationTemplate(
                ("feature_family",),
                "IMAGING_SIGNATURE",
                "predicts_progression",
                ("outcome", "horizon"),
                "LONGITUDINAL_OUTCOME",
            ),
        ),
        qualifier_fields=("model",),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="prognosis",
        case_study_id="prognosis",
        factor_fields=("outcome", "horizon", "feature_family", "marker", "model"),
        relations=(
            RelationTemplate(
                ("feature_family", "marker"),
                "BIOMARKER",
                "predicts_prognosis",
                ("outcome", "horizon"),
                "LONGITUDINAL_OUTCOME",
            ),
        ),
        qualifier_fields=("model",),
    )
)

_register(
    CaseStudyFeedbackAdapter(
        task="imaging_genetics",
        case_study_id="imaging_genetics",
        factor_fields=("gene_pathway", "atlas", "imaging_phenotype", "model"),
        relations=(
            RelationTemplate(
                ("gene_pathway",),
                "GENE_PATHWAY",
                "associated_with_imaging_marker",
                ("imaging_phenotype",),
                "IMAGING_MARKER",
            ),
        ),
        qualifier_fields=("atlas", "model"),
    )
)


def adapter_for(
    task: str,
    *,
    factor_fields: Sequence[str] | None = None,
) -> CaseStudyFeedbackAdapter:
    adapter = _ADAPTERS.get(str(task))
    if adapter is not None:
        if factor_fields is not None and not set(factor_fields).issubset(
            set(adapter.factor_fields)
        ):
            raise ValueError(
                f"{task} contains unregistered factor fields: {tuple(factor_fields)}"
            )
        return adapter
    fields = tuple(factor_fields or ())
    if not fields:
        raise KeyError(f"no feedback adapter registered for {task!r}")
    left = fields[:1]
    right = fields[1:] or fields[:1]
    return CaseStudyFeedbackAdapter(
        task=str(task),
        case_study_id=str(task),
        factor_fields=fields,
        relations=(
            RelationTemplate(
                left,
                "EXPERIMENT_FACTOR",
                "associated_with",
                right,
                "EXPERIMENT_FACTOR",
            ),
        ),
    )


COMPLETED_CASE_STUDY_LINES = (
    "case1_transdiagnostic",
    "case2_pathway_mediation",
    "biomarker_discovery",
    "differential_diagnosis",
    "disease_subtyping",
    "connectome_behavior",
    "brain_age",
    "progression_prediction",
    "prognosis",
    "imaging_genetics",
)


__all__ = [
    "COMPLETED_CASE_STUDY_LINES",
    "CONTRADICTED",
    "EXECUTION_FAILED",
    "FEEDBACK_STATUSES",
    "INCONCLUSIVE",
    "SUPPORTED",
    "CaseStudyFeedbackAdapter",
    "RelationTemplate",
    "adapter_for",
]
