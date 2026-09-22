from __future__ import annotations

from neurooracle.src.adni_genetics_qc import (
    classify_batch,
    detect_genome_build,
    normalize_adni_subject_id,
    overlap_matrix,
)


def test_normalize_adni_subject_id() -> None:
    assert normalize_adni_subject_id("002_S_0295") == "002_S_0295"
    assert normalize_adni_subject_id("sub-ADNI002S0295") == "002_S_0295"
    assert normalize_adni_subject_id("ADNI-002-S-0295") == "002_S_0295"
    assert normalize_adni_subject_id("073_S_10054") == "073_S_10054"
    assert normalize_adni_subject_id("ADNI4-YWM-NGY") == "ADNI4-YWM-NGY"
    assert normalize_adni_subject_id("sub-ADNI4YWMNGY") == "ADNI4-YWM-NGY"
    assert normalize_adni_subject_id("not-an-adni-id") == ""


def test_classify_adni4_gwas_batch(tmp_path) -> None:
    bed = tmp_path / "ADNI4_GWAS" / "plink" / "ADNI4_GWAS.bed"
    assert classify_batch(bed) == "ADNI4_GSA_v3"


def test_overlap_matrix() -> None:
    rows = overlap_matrix(
        {
            "batch_a": {"001_S_0001", "001_S_0002"},
            "batch_b": {"001_S_0002", "001_S_0003"},
        }
    )
    indexed = {
        (row["batch_left"], row["batch_right"]): row["shared_subjects"]
        for row in rows
    }
    assert indexed[("batch_a", "batch_a")] == 2
    assert indexed[("batch_b", "batch_b")] == 2
    assert indexed[("batch_a", "batch_b")] == 1
    assert indexed[("batch_b", "batch_a")] == 1


def test_detect_genome_build(tmp_path) -> None:
    bim = tmp_path / "batch.bim"
    bim.write_text(
        "1 rs3094315 0 752566 C T\n"
        "19 rs7412 0 45412079 T C\n",
        encoding="utf-8",
    )
    build, evidence = detect_genome_build(bim)
    assert build == "GRCh37"
    assert "rs3094315:752566" in evidence
