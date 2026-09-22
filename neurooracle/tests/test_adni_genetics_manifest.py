from neurooracle.src.adni_genetics_manifest import (
    build_preimputation_manifest,
    phenotype_completeness,
)


def _eligible(iid: str, visits: int, count: int) -> dict[str, object]:
    return {
        "subject_id": iid,
        "case2_core_eligible": "True",
        "visits": visits,
        "ADAS13_count": count,
        "Hippocampus_count": count,
    }


def _qc(iid: str, batch: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "FID": batch,
        "IID": iid,
        "batch": batch,
        "EUR_compatible": "True",
        "heterozygosity_outlier": "False",
        "sexcheck_problem": "False",
        "pedigree_sex_missing": "False",
    }
    row.update(overrides)
    return row


def test_phenotype_completeness_uses_visits_and_counts() -> None:
    assert phenotype_completeness(_eligible("S1", 3, 2)) == 34


def test_manifest_selects_best_array_and_resolves_related_component() -> None:
    rows = [
        _qc("S1", "old"),
        _qc("S1", "new"),
        _qc("S2", "new"),
        _qc("S3", "new", heterozygosity_outlier="True"),
    ]
    eligible = [
        _eligible("S1", 4, 3),
        _eligible("S2", 8, 5),
        _eligible("S3", 10, 5),
    ]
    manifest = build_preimputation_manifest(
        rows,
        eligible,
        {"old": 100, "new": 200},
        [("S1", "S2")],
    )
    by_record = {(row["subject_id"], row["batch"]): row for row in manifest}
    assert by_record[("S1", "old")]["duplicate_array_removed"] is True
    assert by_record[("S1", "new")]["relatedness_removed"] is True
    assert by_record[("S2", "new")]["final_preimputation_keep"] is True
    assert by_record[("S3", "new")]["base_qc_pass"] is False


def test_missing_pedigree_sex_is_not_an_exclusion() -> None:
    manifest = build_preimputation_manifest(
        [_qc("S1", "batch", pedigree_sex_missing="True")],
        [_eligible("S1", 2, 1)],
        {"batch": 100},
        [],
    )
    assert manifest[0]["final_preimputation_keep"] is True
