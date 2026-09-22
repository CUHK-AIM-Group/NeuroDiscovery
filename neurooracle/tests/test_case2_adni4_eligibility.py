from neurooracle.scripts.prepare_case2_adni4_eligibility import (
    canonical_rid,
    is_valid_nonnegative,
    measurement_sets,
    merge_eligibility_rows,
)


def test_valid_measurements_reject_missing_codes() -> None:
    assert is_valid_nonnegative("0")
    assert is_valid_nonnegative(12.5)
    assert not is_valid_nonnegative("")
    assert not is_valid_nonnegative("nan")
    assert not is_valid_nonnegative(-4)


def test_measurement_sets_deduplicate_visits_and_support_five_digit_ids() -> None:
    rows = [
        {"PTID": "073_S_10054", "VISDATE": "2026-01-01", "score": 24},
        {"PTID": "073_S_10054", "VISDATE": "2026-01-01", "score": 25},
        {"PTID": "073_S_10054", "VISDATE": "2026-07-01", "score": 26},
        {"PTID": "073_S_10054", "VISDATE": "2026-09-01", "score": -4},
    ]
    visits = measurement_sets(
        rows,
        eligible_subjects={"073_S_10054"},
        value_field="score",
        date_fields=("VISDATE",),
    )
    assert len(visits["073_S_10054"]) == 2


def test_measurement_sets_resolve_rid_only_tables() -> None:
    rows = [{"RID": "10054.0", "EXAMDATE": "2026-01-01", "MEAN": "1.2"}]
    visits = measurement_sets(
        rows,
        eligible_subjects={"073_S_10054"},
        value_field="MEAN",
        date_fields=("EXAMDATE",),
        subject_field="RID",
        rid_to_subject={"10054": "073_S_10054"},
    )
    assert canonical_rid("10054.0") == "10054"
    assert len(visits["073_S_10054"]) == 1


def test_increment_replaces_same_subject_without_duplicating() -> None:
    merged = merge_eligibility_rows(
        [{"subject_id": "001_S_0001", "visits": 2}],
        [{"subject_id": "001_S_0001", "visits": 3}, {"subject_id": "002_S_0002"}],
    )
    assert [row["subject_id"] for row in merged] == ["001_S_0001", "002_S_0002"]
    assert merged[0]["visits"] == 3
