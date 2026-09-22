from __future__ import annotations

import pandas as pd
import torch

from core.scripts import run_lifespan_multimodel_expansion as runner


def test_discover_lifespan_atlases_requires_both_cohorts(tmp_path):
    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "sub-hcpa_100.pt").touch()
    (complete / "sub-100206.pt").touch()

    hcpa_only = tmp_path / "hcpa_only"
    hcpa_only.mkdir()
    (hcpa_only / "sub-hcpa_101.pt").touch()

    assert runner.discover_lifespan_atlases(tmp_path) == ["complete"]


def test_discover_atlases_for_datasets_supports_hcpa_only(tmp_path):
    hcpa_only = tmp_path / "hcpa_only"
    hcpa_only.mkdir()
    (hcpa_only / "sub-hcpa_101.pt").touch()

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "sub-hcpa_102.pt").touch()
    (mixed / "sub-100206.pt").touch()

    assert runner.discover_atlases_for_datasets(tmp_path, ["hcpa"]) == [
        "hcpa_only",
        "mixed",
    ]
    assert runner.discover_atlases_for_datasets(
        tmp_path, ["hcpya", "hcpa"]
    ) == ["mixed"]


def test_default_lifespan_atlases_cover_all_hcpya_atlases():
    assert len(runner.DEFAULT_COMPATIBLE_ATLASES) == 19
    assert set(runner.DEFAULT_COMPATIBLE_ATLASES) == {
        "aal3_166",
        "aal_116",
        "basc_122",
        "cc200",
        "cc400",
        "destrieux_148",
        "dk_112",
        "dosenbach_160",
        "eickhoff_zilles",
        "glasser_360",
        "harvard_oxford_cort",
        "harvard_oxford_merged",
        "harvard_oxford_sub",
        "msdl_39",
        "power_264",
        "schaefer_100_7net",
        "schaefer_200_7net",
        "schaefer_400_7net",
        "talairach_tournoux",
    }
    assert runner.INCOMPATIBLE_CACHE_ATLASES == {}


def test_filter_labels_for_atlas_contract_excludes_mismatches(tmp_path):
    labels = pd.DataFrame(
        {
            "subject_id": ["1001", "HCA1", "HCA2", "HCA3"],
            "dataset": ["hcpya", "hcpa", "hcpa", "hcpa"],
            "label": [25.0, 35.0, 45.0, 55.0],
        }
    )

    def path_resolver(dataset, subject_id, atlas):
        return tmp_path / f"sub-{dataset}_{subject_id}_{atlas}.pt"

    def save(dataset, subject_id, n_roi, names):
        torch.save(
            {"fc_matrix": torch.eye(n_roi), "roi_names": names},
            path_resolver(dataset, subject_id, "atlas"),
        )

    save("hcpya", "1001", 2, ["A", "B"])
    save("hcpa", "HCA1", 2, ["A", "B"])
    save("hcpa", "HCA2", 3, ["A", "B", "C"])
    save("hcpa", "HCA3", 2, ["B", "A"])

    filtered, audit = runner.filter_labels_for_atlas_contract(
        labels, "atlas", path_resolver=path_resolver
    )

    assert filtered["subject_id"].tolist() == ["1001", "HCA1"]
    assert audit["after_by_dataset"] == {"hcpya": 1, "hcpa": 1}
    assert audit["excluded_by_reason"] == {
        "roi_dimension_mismatch": 1,
        "roi_name_or_order_mismatch": 1,
    }


def test_filter_labels_for_atlas_contract_audits_single_hcpa_cohort(tmp_path):
    labels = pd.DataFrame(
        {
            "subject_id": ["HCA1", "HCA2", "HCA3"],
            "dataset": ["hcpa", "hcpa", "hcpa"],
            "label": [35.0, 45.0, 55.0],
        }
    )

    def path_resolver(dataset, subject_id, atlas):
        return tmp_path / f"sub-{dataset}_{subject_id}_{atlas}.pt"

    def save(subject_id, n_roi, names):
        torch.save(
            {"fc_matrix": torch.eye(n_roi), "roi_names": names},
            path_resolver("hcpa", subject_id, "atlas"),
        )

    save("HCA1", 2, ["A", "B"])
    save("HCA2", 2, ["A", "B"])
    save("HCA3", 2, ["B", "A"])

    filtered, audit = runner.filter_labels_for_atlas_contract(
        labels, "atlas", path_resolver=path_resolver
    )

    assert filtered["subject_id"].tolist() == ["HCA1", "HCA2"]
    assert audit["reference_dataset"] == "hcpa"
    assert audit["after_by_dataset"] == {"hcpa": 2}
    assert audit["excluded_by_reason"] == {"roi_name_or_order_mismatch": 1}


def test_run_model_routes_ibgnn(monkeypatch):
    observed = {}

    def fake_train(**kwargs):
        observed.update(kwargs)
        return {"test_mae_yr": 1.0}

    monkeypatch.setattr(runner, "train_ibgnn_safe", fake_train)
    frame = pd.DataFrame({"subject_id": ["1"], "label": [42.0], "dataset": ["hcpa"]})
    result = runner.run_model(
        "ibgnn",
        "aal_116",
        frame,
        frame,
        frame,
        42.0,
        1.0,
        epochs=2,
        batch_size=4,
        seed=7,
    )

    assert result["test_mae_yr"] == 1.0
    assert observed["atlas"] == "aal_116"
    assert observed["n_epochs"] == 2
    assert observed["seed"] == 7
    assert observed["normalize_input"] is True
    assert observed["grad_clip"] == 1.0
    assert observed["max_edges_per_graph"] is None


def test_run_model_caps_high_resolution_ibgnn_epochs(monkeypatch):
    observed = {}

    def fake_train(**kwargs):
        observed.update(kwargs)
        return {"test_mae_yr": 1.0}

    monkeypatch.setattr(runner, "train_ibgnn_safe", fake_train)
    frame = pd.DataFrame({"subject_id": ["1"], "label": [42.0], "dataset": ["hcpa"]})
    runner.run_model(
        "ibgnn",
        "glasser_360",
        frame,
        frame,
        frame,
        42.0,
        1.0,
        epochs=12,
        batch_size=4,
        seed=7,
    )

    assert observed["n_epochs"] == 3
    assert observed["max_edges_per_graph"] == 4096


def test_power_ibgnn_uses_high_edge_count_guard(monkeypatch):
    observed = {}

    def fake_train(**kwargs):
        observed.update(kwargs)
        return {"test_mae_yr": 1.0}

    monkeypatch.setattr(runner, "train_ibgnn_safe", fake_train)
    frame = pd.DataFrame({"subject_id": ["1"], "label": [42.0], "dataset": ["hcpa"]})
    runner.run_model(
        "ibgnn",
        "power_264",
        frame,
        frame,
        frame,
        42.0,
        1.0,
        epochs=12,
        batch_size=4,
        seed=7,
    )

    assert observed["n_epochs"] == 3
    assert observed["max_edges_per_graph"] == 4096


def test_summary_aggregates_seed_mean_and_variance():
    frame = pd.DataFrame(
        [
            {
                "atlas": "aal_116",
                "model": "bnt",
                "seed": 1,
                "status": "complete",
                "test_mae_yr": 2.0,
                "test_mae_z": 0.5,
                "elapsed_sec": 3.0,
            },
            {
                "atlas": "aal_116",
                "model": "bnt",
                "seed": 2,
                "status": "complete",
                "test_mae_yr": 4.0,
                "test_mae_z": 1.0,
                "elapsed_sec": 5.0,
            },
        ]
    )

    summary = runner.summarize(frame).iloc[0]
    assert summary["test_mae_target_units_mean"] == 3.0
    assert summary["test_mae_target_units_variance"] == 2.0
    assert summary["test_mae_years_mean"] == 3.0
    assert summary["test_mae_years_variance"] == 2.0
    assert summary["n_seeds"] == 2
    assert summary["elapsed_sec"] == 8.0


def test_make_fold_splits_has_disjoint_five_fold_tests():
    frame = pd.DataFrame(
        {
            "subject_id": [str(index) for index in range(100)],
            "dataset": "hcpya",
            "label": [20.0 + index / 10 for index in range(100)],
        }
    )

    first = runner.make_fold_splits(frame, folds=5, seed=17)
    second = runner.make_fold_splits(frame, folds=5, seed=17)

    first_tests = [set(test["subject_id"]) for _, _, test in first]
    second_tests = [set(test["subject_id"]) for _, _, test in second]
    assert first_tests == second_tests
    assert set.union(*first_tests) == set(frame["subject_id"])
    assert sum(len(values) for values in first_tests) == len(frame)
    assert all(not (left & right) for i, left in enumerate(first_tests) for right in first_tests[i + 1 :])
