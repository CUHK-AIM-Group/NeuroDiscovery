from __future__ import annotations

import json

import pandas as pd
import pytest

from core.scripts.run_case1_multimodel_10x3 import (
    DEFAULT_MODELS,
    EXPECTED_MODEL_FOLDS,
    EXPECTED_MODEL_FOLDS_PER_SEED,
    FORMAL_ATLASES,
    FORMAL_DISEASES,
    FORMAL_FOLDS,
    FORMAL_SEEDS,
    aggregate_attributions,
    seed_run_complete,
)


def _write_complete_seed_run(run_dir, seed: int) -> None:
    run_dir.mkdir()
    rows = [
        {
            "atlas": atlas,
            "disease": disease,
            "model": model,
            "seed": seed,
            "fold": fold,
        }
        for atlas in FORMAL_ATLASES
        for disease in FORMAL_DISEASES
        for model in DEFAULT_MODELS
        for fold in range(FORMAL_FOLDS)
    ]
    pd.DataFrame(rows).to_csv(run_dir / "performance_folds.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "atlases": list(FORMAL_ATLASES),
                "diseases": list(FORMAL_DISEASES),
                "models": list(DEFAULT_MODELS),
                "seeds": [seed],
                "folds": FORMAL_FOLDS,
                "n_model_folds": EXPECTED_MODEL_FOLDS_PER_SEED,
                "n_attribution_model_folds": EXPECTED_MODEL_FOLDS_PER_SEED,
                "n_failed_model_folds": 0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"importance_abs": 1.0}]).to_csv(
        run_dir / "heldout_attribution_by_seed.csv", index=False
    )


def test_formal_case1_protocol_has_41040_model_folds() -> None:
    assert len(FORMAL_ATLASES) == 19
    assert len(FORMAL_DISEASES) == 9
    assert len(FORMAL_SEEDS) == 10
    assert FORMAL_FOLDS == 3
    assert EXPECTED_MODEL_FOLDS == 41_040


def test_attribution_aggregation_uses_seed_level_sample_variance(tmp_path) -> None:
    shared = {
        "model": "roi_mlp",
        "atlas": "atlas",
        "disease": "disease",
        "evidence_resolution": "roi_feature",
        "roi_index": 0,
        "roi_id": 1,
        "roi_name": "ROI 1",
        "hemisphere": "L",
        "network": "Default",
        "feature": "roi_alff",
        "n_folds": 3,
    }
    for seed, value in ((11, 1.0), (12, 3.0)):
        run_dir = tmp_path / f"seed_{seed}"
        run_dir.mkdir()
        pd.DataFrame(
            [{**shared, "seed": seed, "importance_abs": value, "signed_attribution": -value}]
        ).to_csv(run_dir / "heldout_attribution_by_seed.csv", index=False)

    assert aggregate_attributions(tmp_path, (11, 12)) == 1
    summary = pd.read_csv(tmp_path / "heldout_attribution_summary.csv")
    assert summary.loc[0, "importance_abs_mean"] == pytest.approx(2.0)
    assert summary.loc[0, "importance_abs_variance"] == pytest.approx(2.0)
    assert summary.loc[0, "signed_attribution_mean"] == pytest.approx(-2.0)
    assert summary.loc[0, "signed_attribution_variance"] == pytest.approx(2.0)
    assert summary.loc[0, "n_seeds"] == 2


def test_seed_run_complete_validates_unique_expected_fold_grid(tmp_path) -> None:
    seed = FORMAL_SEEDS[0]
    run_dir = tmp_path / f"seed_{seed}"
    _write_complete_seed_run(run_dir, seed)
    assert seed_run_complete(run_dir, seed)

    performance = pd.read_csv(run_dir / "performance_folds.csv")
    performance.iloc[-1] = performance.iloc[0]
    performance.to_csv(run_dir / "performance_folds.csv", index=False)
    assert not seed_run_complete(run_dir, seed)


def test_seed_run_complete_rejects_unresolved_failures(tmp_path) -> None:
    seed = FORMAL_SEEDS[0]
    run_dir = tmp_path / f"seed_{seed}"
    _write_complete_seed_run(run_dir, seed)
    pd.DataFrame([{"atlas": FORMAL_ATLASES[0], "error": "failed"}]).to_csv(
        run_dir / "failed_model_folds.csv", index=False
    )
    assert not seed_run_complete(run_dir, seed)
