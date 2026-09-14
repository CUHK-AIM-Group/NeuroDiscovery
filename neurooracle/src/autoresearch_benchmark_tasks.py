"""Primary-experiment schedule over the formal case-study registry.

Registration, primary experiments, and validation protocols are deliberately
separate.  All seventeen scopes are formal case studies.  Only Case Study 1
and Case Study 2 are scheduled for primary experiments at present; hindcasting
support is declared in :mod:`neurooracle.src.validation_protocols`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from .case_studies import CASE_STUDIES


BenchmarkFamily = Literal["case_study"]
PrimaryExperimentProtocol = Literal[
    "transdiagnostic_exhaustive",
    "pathway_mediation",
]


@dataclass(frozen=True)
class AutoresearchBenchmarkTask:
    name: str
    label: str
    family: BenchmarkFamily
    signature: str
    description: str
    primary_protocol: PrimaryExperimentProtocol | None = None
    default_deferred: bool = False
    deferred_reason: str = ""

    @property
    def primary_experiment_scheduled(self) -> bool:
        return not self.default_deferred

    def to_dict(self) -> dict[str, str | bool | None]:
        payload = asdict(self)
        payload["primary_experiment_scheduled"] = self.primary_experiment_scheduled
        return payload


def _build_registry() -> tuple[AutoresearchBenchmarkTask, ...]:
    registry: list[AutoresearchBenchmarkTask] = []
    for case in CASE_STUDIES:
        if case.name == "case1_transdiagnostic":
            protocol: PrimaryExperimentProtocol | None = "transdiagnostic_exhaustive"
            deferred = False
            reason = ""
        elif case.name == "case2_pathway_mediation":
            protocol = "pathway_mediation"
            deferred = False
            reason = ""
        else:
            protocol = None
            deferred = True
            reason = (
                "Registered formal case study; its primary experiment is not "
                "scheduled in the current rollout."
            )

        signature = ""
        description = ""
        if case.task is not None:
            signature = case.task.signature
            description = case.task.description
        elif case.chain is not None:
            signature = case.chain.signature
            description = case.chain.description
        if case.name == "case1_transdiagnostic":
            signature = "disease x atlas/ROI x imaging feature"
            description = (
                "Recover disease-region-feature discoveries from the exhaustive "
                "cross-diagnostic experiment space."
            )
        elif case.name == "case2_pathway_mediation":
            signature = "G->IM->O[longitudinal]"
            description = (
                "Prioritise pathway-level polygenic risk to imaging marker to "
                "longitudinal outcome mediation hypotheses."
            )

        registry.append(
            AutoresearchBenchmarkTask(
                name=case.name,
                label=case.english_name,
                family="case_study",
                signature=signature,
                description=description,
                primary_protocol=protocol,
                default_deferred=deferred,
                deferred_reason=reason,
            )
        )

    names = [task.name for task in registry]
    if len(registry) != 17 or len(names) != len(set(names)):
        raise RuntimeError("the autoresearch benchmark registry must contain 17 unique case studies")
    return tuple(registry)


AUTORESEARCH_BENCHMARK_TASKS = _build_registry()
AUTORESEARCH_BENCHMARK_TASK_BY_NAME = {
    task.name: task for task in AUTORESEARCH_BENCHMARK_TASKS
}


def benchmark_tasks(
    names: Iterable[str] | None = None,
    *,
    include_deferred: bool = False,
) -> tuple[AutoresearchBenchmarkTask, ...]:
    requested = (
        list(names)
        if names is not None
        else [task.name for task in AUTORESEARCH_BENCHMARK_TASKS]
    )
    unknown = sorted(set(requested) - AUTORESEARCH_BENCHMARK_TASK_BY_NAME.keys())
    if unknown:
        raise KeyError(f"unknown autoresearch benchmark case studies: {', '.join(unknown)}")

    seen: set[str] = set()
    selected: list[AutoresearchBenchmarkTask] = []
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        task = AUTORESEARCH_BENCHMARK_TASK_BY_NAME[name]
        if task.default_deferred and not include_deferred:
            continue
        selected.append(task)
    return tuple(selected)


__all__ = [
    "AUTORESEARCH_BENCHMARK_TASKS",
    "AUTORESEARCH_BENCHMARK_TASK_BY_NAME",
    "AutoresearchBenchmarkTask",
    "BenchmarkFamily",
    "PrimaryExperimentProtocol",
    "benchmark_tasks",
]
