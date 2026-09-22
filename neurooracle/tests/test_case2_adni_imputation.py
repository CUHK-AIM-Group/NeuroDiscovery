import json
from pathlib import Path

import pytest

from neurooracle.scripts.run_case2_adni_imputation import (
    beagle_output_complete,
    beagle_chrom_arg,
    find_bref3,
    find_map,
    merge_chromosome_summaries,
    region_args,
)
from neurooracle.scripts.validate_case2_adni_imputation import (
    merge_validation_results,
)


def test_region_args_are_optional() -> None:
    assert region_args(22, None, None) == ["--chr", "22"]
    assert region_args(22, 20, 30) == [
        "--chr",
        "22",
        "--from-bp",
        "20",
        "--to-bp",
        "30",
    ]


def test_find_map_requires_one_match(tmp_path: Path) -> None:
    map_path = tmp_path / "nested" / "plink.chr22.GRCh37.map"
    map_path.parent.mkdir()
    map_path.write_text("", encoding="utf-8")
    assert find_map(tmp_path, 22) == map_path
    with pytest.raises(FileNotFoundError):
        find_map(tmp_path, 21)


def test_beagle_chrom_arg_supports_regions() -> None:
    assert beagle_chrom_arg(22, None, None) == "22"
    assert beagle_chrom_arg(22, 20, 30) == "22:20-30"
    assert beagle_chrom_arg(22, None, 30) == "22:-30"


def test_find_bref3(tmp_path: Path) -> None:
    path = tmp_path / "chr22.1kg.phase3.v5a.b37.bref3"
    path.write_bytes(b"reference")
    assert find_bref3(tmp_path, 22) == path


def test_beagle_output_requires_finished_log(tmp_path: Path) -> None:
    vcf = tmp_path / "result.vcf.gz"
    log = tmp_path / "result.log"
    vcf.write_bytes(b"x" * 1001)
    log.write_text("Terminating program.", encoding="utf-8")
    assert not beagle_output_complete(vcf, log)
    log.write_text(
        "beagle.27Feb25.75f.jar finished",
        encoding="utf-8",
    )
    assert beagle_output_complete(vcf, log)


def test_incremental_summary_keeps_existing_batch_outputs(tmp_path: Path) -> None:
    old = {"batch": "ADNI1_Human610_Quad", "postqc_variants": 100}
    old_dir = tmp_path / "ADNI1_Human610_Quad" / "chr22"
    old_dir.mkdir(parents=True)
    (old_dir / "summary.json").write_text(
        json.dumps(old), encoding="utf-8"
    )
    new = {"batch": "ADNI4_GSA_v3", "postqc_variants": 200}
    payload = merge_chromosome_summaries(
        tmp_path,
        [new],
        chrom=22,
        start_bp=None,
        end_bp=None,
    )
    assert [row["batch"] for row in payload["batches"]] == [
        "ADNI1_Human610_Quad",
        "ADNI4_GSA_v3",
    ]


def test_incremental_validation_keeps_existing_batch_outputs(tmp_path: Path) -> None:
    old = {"batch": "ADNI1_Human610_Quad", "concordance": 0.99}
    old_dir = tmp_path / "ADNI1_Human610_Quad" / "chr22"
    old_dir.mkdir(parents=True)
    (old_dir / "validation.json").write_text(json.dumps(old), encoding="utf-8")
    new = {"batch": "ADNI4_GSA_v3", "concordance": 1.0}
    payload = merge_validation_results(tmp_path, [new], chrom=22)
    assert [row["batch"] for row in payload["batches"]] == [
        "ADNI1_Human610_Quad",
        "ADNI4_GSA_v3",
    ]
