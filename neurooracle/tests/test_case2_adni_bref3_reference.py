from pathlib import Path

from neurooracle.scripts.prepare_case2_adni_bref3_reference import (
    export_counts,
    reference_names,
)


def test_reference_names_match_imputation_lookup(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    work_root = tmp_path / "work"
    vcf_prefix, local_bref3, final_bref3 = reference_names(
        output_root,
        work_root,
        22,
    )
    assert vcf_prefix == work_root / "chr22_1kg_phase3_v5a_b37"
    assert local_bref3 == work_root / "chr22.1kg.phase3.v5a.b37.bref3"
    assert final_bref3 == (
        output_root / "chr22.1kg.phase3.v5a.b37.bref3"
    )


def test_export_counts_uses_plink_log(tmp_path: Path) -> None:
    log = tmp_path / "export.log"
    log.write_text(
        "2504 samples (1270 females) loaded from\n"
        "3429160 variants remaining after main filters.\n",
        encoding="utf-8",
    )
    assert export_counts(log, tmp_path / "unused.vcf.gz") == (2504, 3429160)
