from neurooracle.src.autoresearch_benchmark_tasks import (
    AUTORESEARCH_BENCHMARK_TASKS,
    benchmark_tasks,
)


def test_registry_has_15_ordinary_and_two_case_studies() -> None:
    assert len(AUTORESEARCH_BENCHMARK_TASKS) == 17
    assert sum(task.family == "ordinary" for task in AUTORESEARCH_BENCHMARK_TASKS) == 15
    assert sum(task.family == "case_study" for task in AUTORESEARCH_BENCHMARK_TASKS) == 2


def test_default_run_defers_only_case_study_2() -> None:
    selected = benchmark_tasks()
    assert len(selected) == 16
    assert "case1_transdiagnostic" in {task.name for task in selected}
    assert "case2_pathway_mediation" not in {task.name for task in selected}


def test_explicit_deferred_run_contains_all_tasks() -> None:
    selected = benchmark_tasks(include_deferred=True)
    assert len(selected) == 17
    assert selected[-1].name == "case2_pathway_mediation"
