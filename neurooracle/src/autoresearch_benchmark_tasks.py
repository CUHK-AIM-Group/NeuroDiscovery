"""Unified task registry for the manuscript autoresearch benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from .case3_subtasks import case3_subtasks


BenchmarkFamily = Literal["ordinary", "case_study"]
EvaluationProtocol = Literal[
    "temporal_claim_hindcasting",
    "transdiagnostic_exhaustive",
    "pathway_mediation",
]


@dataclass(frozen=True)
class AutoresearchBenchmarkTask:
    name: str
    label: str
    family: BenchmarkFamily
    protocol: EvaluationProtocol
    signature: str
    description: str
    default_deferred: bool = False
    deferred_reason: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def _build_registry() -> tuple[AutoresearchBenchmarkTask, ...]:
    ordinary = tuple(
        AutoresearchBenchmarkTask(
            name=task.name,
            label=task.name.replace("_", " ").title(),
            family="ordinary",
            protocol="temporal_claim_hindcasting",
            signature=task.signature,
            description=task.description,
        )
        for task in case3_subtasks()
    )
    case_studies = (
        AutoresearchBenchmarkTask(
            name="case1_transdiagnostic",
            label="Case Study 1: Transdiagnostic brain atlas",
            family="case_study",
            protocol="transdiagnostic_exhaustive",
            signature="disease x atlas/ROI x imaging feature",
            description=(
                "Recover disease-region-feature discoveries from the exhaustive "
                "cross-diagnostic experiment space."
            ),
        ),
        AutoresearchBenchmarkTask(
            name="case2_pathway_mediation",
            label="Case Study 2: Pathway polygenic mediation",
            family="case_study",
            protocol="pathway_mediation",
            signature="G->IM->O[longitudinal]",
            description=(
                "Prioritise pathway-level polygenic risk to imaging marker to "
                "longitudinal outcome mediation hypotheses."
            ),
            default_deferred=True,
            deferred_reason="Case Study 2 currently has insufficient task-specific literature coverage.",
        ),
    )
    registry = ordinary + case_studies
    names = [task.name for task in registry]
    if len(registry) != 17 or len(names) != len(set(names)):
        raise RuntimeError("The autoresearch benchmark registry must contain 17 unique tasks")
    return registry


AUTORESEARCH_BENCHMARK_TASKS = _build_registry()
AUTORESEARCH_BENCHMARK_TASK_BY_NAME = {
    task.name: task for task in AUTORESEARCH_BENCHMARK_TASKS
}


def benchmark_tasks(
    names: Iterable[str] | None = None,
    *,
    include_deferred: bool = False,
) -> tuple[AutoresearchBenchmarkTask, ...]:
    requested = list(names) if names is not None else [task.name for task in AUTORESEARCH_BENCHMARK_TASKS]
    unknown = sorted(set(requested) - AUTORESEARCH_BENCHMARK_TASK_BY_NAME.keys())
    if unknown:
        raise KeyError(f"unknown autoresearch benchmark tasks: {', '.join(unknown)}")

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
    "EvaluationProtocol",
    "benchmark_tasks",
]
