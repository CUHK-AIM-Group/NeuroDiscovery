from neurooracle.scripts.validate_case2_adni_imputation import genotype_dosage


def test_genotype_dosage_handles_phased_and_missing_calls() -> None:
    assert genotype_dosage("0|1:1.0") == 1
    assert genotype_dosage("1/1") == 2
    assert genotype_dosage("0|0") == 0
    assert genotype_dosage(".|.") is None
