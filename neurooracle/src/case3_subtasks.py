"""Case Study 3 subtask registry.

Case Study 3 evaluates task-general temporal hypothesis discovery. Its subtask
set is derived from the canonical flat-task registry while excluding the
transdiagnostic template owned by Case Study 1. Mechanistic TaskChain templates
remain available to the general NeuroOracle engine but are not CS3 subtasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from .atoms import CANONICAL_TASKS


SubtaskKind = Literal["task"]

CS1_TASKS = frozenset({"transdiagnostic_clustering"})


@dataclass(frozen=True)
class Case3Subtask:
    name: str
    kind: SubtaskKind
    signature: str
    description: str
    example: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _build_registry() -> tuple[Case3Subtask, ...]:
    registry = tuple(
        Case3Subtask(
            name=task.name,
            kind="task",
            signature=task.signature,
            description=task.description,
            example=task.example,
        )
        for task in CANONICAL_TASKS
        if task.name not in CS1_TASKS
    )
    names = [subtask.name for subtask in registry]
    if len(names) != len(set(names)):
        raise RuntimeError("Case Study 3 subtask names must be unique")
    return registry


CASE3_SUBTASKS = _build_registry()
CASE3_SUBTASK_BY_NAME = {subtask.name: subtask for subtask in CASE3_SUBTASKS}


def case3_subtasks(kind: SubtaskKind | None = None) -> tuple[Case3Subtask, ...]:
    if kind is None:
        return CASE3_SUBTASKS
    if kind != "task":
        raise ValueError(f"unknown Case Study 3 subtask kind: {kind}")
    return tuple(subtask for subtask in CASE3_SUBTASKS if subtask.kind == kind)


def select_case3_subtasks(
    names: Iterable[str] | None = None,
    *,
    exclude: Iterable[str] = (),
) -> tuple[Case3Subtask, ...]:
    requested = list(names) if names is not None else [s.name for s in CASE3_SUBTASKS]
    excluded = set(exclude)
    unknown = sorted((set(requested) | excluded) - CASE3_SUBTASK_BY_NAME.keys())
    if unknown:
        raise KeyError(f"unknown Case Study 3 subtasks: {', '.join(unknown)}")

    seen: set[str] = set()
    selected: list[Case3Subtask] = []
    for name in requested:
        if name in seen or name in excluded:
            continue
        seen.add(name)
        selected.append(CASE3_SUBTASK_BY_NAME[name])
    return tuple(selected)


__all__ = [
    "CASE3_SUBTASKS",
    "CASE3_SUBTASK_BY_NAME",
    "CS1_TASKS",
    "Case3Subtask",
    "SubtaskKind",
    "case3_subtasks",
    "select_case3_subtasks",
]
