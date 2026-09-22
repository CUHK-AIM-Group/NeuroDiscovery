from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurooracle.scripts.build_case2_adni_experiment_table import (
    build_subject_table,
    discover_imaging_files,
    normalize_subject_id,
    write_imaging_partitions,
)


def test_normalize_subject_id_handles_adni_bids_and_plink_labels() -> None:
    assert normalize_subject_id("002_S_2010") == "002_S_2010"
    assert normalize_subject_id("sub-ADNI002S2010") == "002_S_2010"
    assert normalize_subject_id("ADNI3_129_S_6146") == "129_S_6146"
    assert normalize_subject_id("not-an-adni-id") == ""


def test_build_subject_table_inner_joins_and_checks_apoe() -> None:
    genetics = pd.DataFrame(
        {
            "subject_id": ["002_S_2010", "003_S_0001"],
            "apoe_e4_dosage": [1.0, 0.0],
            "apoe_complete": [True, True],
            "apoe_genotype": ["e3/e4", "e3/e3"],
            "batch": ["a", "b"],
        }
    )
    phenotypes = pd.DataFrame(
        {
            "subject_id": ["002_S_2010", "004_S_0002"],
            "diagnosis": ["MCI", "CN"],
            "PTGENDER": ["Male", "Female"],
            "APOE4": [1, 0],
        }
    )

    cohort, audit = build_subject_table(genetics, phenotypes)

    assert cohort["subject_id"].tolist() == ["002_S_2010"]
    assert cohort.loc[0, "diagnosis_code"] == 1
    assert cohort.loc[0, "sex_binary"] == 1
    assert bool(cohort.loc[0, "apoe4_concordant"])
    assert int(audit["included"].sum()) == 1


def _imaging_row(subject: str, source: str, roi_index: int) -> dict[str, object]:
    return {
        "PTID": subject,
        "subject_bids": f"sub-ADNI{subject.replace('_', '')}",
        "session_bids": "ses-bl",
        "VISCODE": "bl",
        "diagnosis": "MCI",
        "AGE": 72.0,
        "PTGENDER": "Male",
        "SITE": 1,
        "mean_fd": 0.12,
        "n_timepoints": 140,
        "source": source,
        "atlas": "tiny_atlas",
        "roi_index": roi_index,
        "roi_id": f"ROI:{roi_index}",
        "roi_name": f"ROI {roi_index}",
        "parcel_name": f"Parcel {roi_index}",
        "hemisphere": "L",
        "network": "Default",
        "structure_class": "cortical",
        "feature": "corr_mean",
        "value": 0.1 * roi_index,
    }


def test_write_imaging_partitions_filters_cohort(tmp_path: Path) -> None:
    imaging_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    imaging_dir.mkdir()
    input_path = imaging_dir / "adni_tiny_atlas_run_features.csv"
    pd.DataFrame(
        [
            _imaging_row("002_S_2010", "tiny_atlas_multiatlas", 1),
            _imaging_row("003_S_0001", "tiny_atlas_multiatlas", 2),
        ]
    ).to_csv(input_path, index=False)

    assert discover_imaging_files(imaging_dir) == [input_path]
    summary, qc = write_imaging_partitions(
        imaging_dir,
        output_dir,
        {"002_S_2010"},
        chunksize=1,
    )

    output = pd.read_parquet(output_dir / "tiny_atlas.parquet")
    assert output["subject_id"].tolist() == ["002_S_2010"]
    assert output["marker_id"].tolist() == [
        "tiny_atlas_multiatlas|1|corr_mean"
    ]
    assert summary.loc[0, "rows"] == 1
    assert summary.loc[0, "subjects"] == 1
    assert qc.loc[0, "imaging_source_count"] == 1


# Last Updated At: 2026-07-31 02:44 HKT
