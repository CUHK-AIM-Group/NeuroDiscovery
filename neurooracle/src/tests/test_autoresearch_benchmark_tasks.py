from neurooracle.src.autoresearch_benchmark_tasks import (
    AUTORESEARCH_BENCHMARK_TASKS,
    benchmark_tasks,
)


def test_registry_has_17_formal_case_studies() -> None:
    assert len(AUTORESEARCH_BENCHMARK_TASKS) == 17
    assert all(task.family == "case_study" for task in AUTORESEARCH_BENCHMARK_TASKS)


def test_default_primary_run_schedules_only_case_studies_1_and_2() -> None:
    selected = benchmark_tasks()
    assert [task.name for task in selected] == [
        "case1_transdiagnostic",
        "case2_pathway_mediation",
    ]


def test_explicit_deferred_run_contains_all_tasks() -> None:
    selected = benchmark_tasks(include_deferred=True)
    assert len(selected) == 17
    assert selected[-1].name == "prognosis"
