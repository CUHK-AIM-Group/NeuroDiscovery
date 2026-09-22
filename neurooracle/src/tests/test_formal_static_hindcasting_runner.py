from __future__ import annotations

import json
from pathlib import Path

from neurooracle.scripts.run_formal_static_hindcasting import (
    _commands,
    _derive_smoke_lock,
)
from neurooracle.src.experiment_source_bundle import sha256_file
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


def _write_formal_lock(root: Path) -> Path:
    matrix = root / "eligibility_matrix_locked.csv"
    matrix.write_text(
        "case_study_id,freeze_year,future_start_year,future_end_year,analysis_tier\n"
        "case1_transdiagnostic,2016,2017,2021,primary\n"
        "case1_transdiagnostic,2020,2021,2025,primary\n"
        "connectome_behavior,2020,2021,2025,primary\n",
        encoding="utf-8",
    )
    manifest = root / "eligibility_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "primary_windows": 3,
                "primary_case_study_ids": [
                    "case1_transdiagnostic",
                    "connectome_behavior",
                ],
                "locked_matrix": {
                    "path": str(matrix),
                    "sha256": sha256_file(matrix),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_smoke_lock_is_derived_from_latest_formal_case1_window(tmp_path: Path) -> None:
    formal = _write_formal_lock(tmp_path)

    smoke = _derive_smoke_lock(formal, tmp_path / "smoke")
    loaded = load_locked_hindcasting_eligibility(smoke)

    assert loaded.primary_windows == frozenset(
        {("case1_transdiagnostic", 2020, 2021, 2025)}
    )
    payload = json.loads(smoke.read_text(encoding="utf-8"))
    assert payload["source_formal_eligibility_manifest_sha256"] == sha256_file(
        formal
    )


def test_formal_commands_preserve_preregistered_static_policy(tmp_path: Path) -> None:
    formal = _write_formal_lock(tmp_path)
    commands = _commands(
        python=Path("python.exe"),
        output_root=tmp_path / "out",
        snapshot_root=tmp_path / "snapshots",
        future_claims=tmp_path / "claims.jsonl",
        eligibility_manifest=formal,
        smoke=False,
        force=False,
    )

    neuro = commands["neurodiscovery"]
    assert neuro[neuro.index("--target-per-case-study") + 1] == "1000"
    assert neuro[neuro.index("--generation-pool-size") + 1] == "1200"
    assert neuro[neuro.index("--task-scope-fraction") + 1] == "0.5"
    assert neuro[neuro.index("--evidence-frontier-fraction") + 1] == "1.0"
    assert neuro[neuro.index("--endpoint-canonical-quality-weight") + 1] == "0.0"
    assert neuro[neuro.index("--protect-general-top-k") + 1] == "0"
    assert neuro[neuro.index("--static-score-family") + 1] == "legacy"
    assert "--force" not in neuro

    baseline = commands["baselines"]
    assert baseline[baseline.index("--target-per-case-study") + 1] == "1000"
    assert baseline[baseline.index("--methods") + 1 : baseline.index("--methods") + 3] == [
        "sciagents",
        "openscholar_rag",
    ]

    evaluate = commands["evaluate"]
    top_k = evaluate[evaluate.index("--top-k") + 1 : evaluate.index("--random-trials")]
    assert top_k == ["10", "20", "50", "100", "200", "500", "1000"]
