from __future__ import annotations

import numpy as np
import pandas as pd

from core.scripts.case_study_design_priors import (
    ATLAS_RESOLUTION_COLUMN,
    DESIGN_PRIOR_COLUMN,
    MODEL_CLASS_COLUMN,
    attach_brain_age_design_prior,
    attach_connectome_design_prior,
)


def test_connectome_design_prior_uses_only_public_design_metadata() -> None:
    public = pd.DataFrame(
        {
            "candidate_id": ["c0", "c1", "c2"],
            "atlas": ["atlas_100", "atlas_400", "atlas_400"],
            "model": ["svm", "ridge", "elastic_net"],
        }
    )
    manifest = {
        "provenance": {
            "protocol": {"discovery_datasets": ["discovery"]},
            "feature_cache_audits": {
                "atlas_100": {"discovery": {"expected_n_rois": 100}},
                "atlas_400": {"discovery": {"expected_n_rois": 400}},
            },
        }
    }

    scored, audit = attach_connectome_design_prior(public, manifest)

    assert np.allclose(scored[ATLAS_RESOLUTION_COLUMN], [0.0, 1.0, 1.0])
    assert np.allclose(scored[MODEL_CLASS_COLUMN], [0.5, 1.0, 1.0])
    assert np.allclose(scored[DESIGN_PRIOR_COLUMN], [0.25, 1.0, 1.0])
    assert audit["uses_internal_outcomes"] is False
    assert audit["uses_external_outcomes"] is False
    assert audit["uses_generator_identity"] is False


def test_brain_age_design_prior_uses_only_public_model_class() -> None:
    public = pd.DataFrame(
        {
            "candidate_id": ["c0", "c1", "c2"],
            "model": ["svm", "ridge", "elastic_net"],
        }
    )

    scored, audit = attach_brain_age_design_prior(public, {})

    assert np.allclose(scored[MODEL_CLASS_COLUMN], [0.5, 1.0, 1.0])
    assert np.allclose(scored[DESIGN_PRIOR_COLUMN], [0.5, 1.0, 1.0])
    assert audit["uses_internal_outcomes"] is False
    assert audit["uses_external_outcomes"] is False
    assert audit["uses_generator_identity"] is False
