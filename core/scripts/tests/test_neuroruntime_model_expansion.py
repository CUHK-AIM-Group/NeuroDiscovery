from __future__ import annotations

from pathlib import Path

from core.scripts.run_neuroruntime_model_expansion import build_tasks


def test_full_lightweight_sweep_contains_all_registered_tasks(tmp_path: Path) -> None:
    manifest = {
        "tcp": {
            "biomarker_input": "biomarker.csv",
            "differential_input": "differential.csv",
            "subtyping_input": "subtyping.csv",
        },
        "hcpya": {
            "behavior_target": "cognition",
            "behavior_connectomes": "connectomes.npz",
            "behavior_labels": "behavior.csv",
            "brain_age_input": "brain_age.csv",
        },
        "adni": {
            "progression_input": "progression.csv",
            "survival_input": "survival.csv",
            "imaging_genetics_input": "genetics.npz",
        },
    }
    tasks = build_tasks(
        manifest=manifest,
        derived={
            "behavior_tabular": "behavior_tabular.csv",
            "imaging_genetics_multivariate": "genetics_multivariate.npz",
        },
        output_dir=tmp_path,
        cases=[
            "biomarker_discovery",
            "differential_diagnosis",
            "disease_subtyping",
            "progression_prediction",
            "connectome_behavior",
            "brain_age",
            "imaging_genetics",
            "prognosis",
        ],
        seeds=[20260805, 20260806, 20260807],
        folds=5,
        deep_epochs=50,
    )

    assert len(tasks) == 102
    by_case = {}
    for task in tasks:
        by_case.setdefault(task.case_study, set()).add(task.model)
    assert by_case["biomarker_discovery"] == {
        "logistic",
        "ridge",
        "elastic_net",
        "svm",
    }
    assert by_case["disease_subtyping"] == {
        "kmeans",
        "gmm",
        "spectral",
        "nmf",
        "consensus",
        "pca",
        "autoencoder",
    }
    assert by_case["connectome_behavior"] >= {
        "ols",
        "ridge",
        "elastic_net",
        "svm",
        "cpm_p0.01",
    }
    assert by_case["prognosis"] == {"cox", "deepsurv"}
    assert by_case["imaging_genetics"] == {"association", "pls", "cca"}

