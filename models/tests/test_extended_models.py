from __future__ import annotations

import json
import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import pearsonr

from models.brain_age.correction import BrainAgeBiasCorrector
from models.braingnn.net.braingnn import BrainGNN
from models.brainnetcnn.net.brainnetcnn import BrainNetCNN
from models.causal_treatment.estimators import make_cate_estimator
from models.cpm.cpm import CPM
from models.connectome_discovery.mapping import cosine_similarity_map
from models.imaging_genetics.association import (
    association_scan,
    association_scan_lmm,
    polygenic_score,
)
from models.imaging_genetics.multivariate import fit_cca, fit_pls
from models.kg_link_prediction.gnn import GNNLinkPredictor, sample_negatives
from models.neuroimaging_decoding.roi_glm import fit_roi_glm
from models.statistical_ml.estimators import cohen_d, make_estimator
from models.statistical_ml.train import main as statistical_main
from models.subtyping.estimators import fit_subtypes
from models.survival_models.estimators import make_survival_estimator
from models.survival_models.metrics import concordance_index
from models.temporal_models.net import TemporalPredictor
from models.cnn3d.net import VoxelCNN3D
from models.combraintf.scripts.data_adapter import build_community_ids


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_combraintf_community_ids_are_contiguous() -> None:
    names = ["Frontal Pole", "Occipital Pole", "Cerebellum", "Unknown"]
    communities = build_community_ids(names, "harvard_oxford_cort")
    assert sorted(set(communities)) == list(range(len(set(communities))))


def test_statistical_estimators_and_cli(tmp_path, rng):
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] + rng.normal(scale=0.2, size=60) > 0).astype(int)
    for name in (
        "logistic",
        "ridge",
        "elastic_net",
        "svm",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ):
        estimator = make_estimator(name, "classification", seed=4)
        estimator.fit(X, y)
        assert estimator.predict(X).shape == y.shape
    assert cohen_d(X[y == 1, 0], X[y == 0, 0]) > 0

    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame.insert(0, "subject_id", [f"s{i}" for i in range(len(frame))])
    frame["target"] = y
    csv_path = tmp_path / "features.csv"
    output = tmp_path / "run"
    frame.to_csv(csv_path, index=False)
    assert statistical_main(
        [
            "--features",
            str(csv_path),
            "--target",
            "target",
            "--model",
            "logistic",
            "--task",
            "classification",
            "--folds",
            "3",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert json.loads((output / "metrics.json").read_text())["auroc"] > 0.8


@pytest.mark.parametrize(
    "name", ["kmeans", "gmm", "spectral", "nmf", "consensus", "pca", "autoencoder"]
)
def test_subtyping_models(name, rng):
    X = np.r_[rng.normal(-2, 0.4, (20, 6)), rng.normal(2, 0.4, (20, 6))]
    result = fit_subtypes(
        X,
        name,
        n_clusters=2,
        seed=3,
        latent_dim=3,
        epochs=3,
        n_bootstraps=5,
    )
    assert len(np.unique(result.labels)) == 2
    assert result.embedding.shape[0] == len(X)


def test_survival_models(rng):
    X = rng.normal(size=(50, 5))
    true_risk = X[:, 0] - X[:, 1]
    duration = np.exp(-true_risk + rng.normal(scale=0.3, size=len(X)))
    event = rng.random(len(X)) > 0.2
    for name, params in (("cox", {}), ("deepsurv", {"epochs": 3, "hidden_dims": (8,)})):
        estimator = make_survival_estimator(name, seed=2, **params)
        estimator.fit(X, duration, event)
        risk = estimator.predict_risk(X)
        assert risk.shape == duration.shape
        assert 0 <= concordance_index(duration, event, risk) <= 1


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="optional scikit-survival dependency is not installed",
)
def test_random_survival_forest(rng):
    X = rng.normal(size=(40, 4))
    duration = np.exp(-X[:, 0] + rng.normal(scale=0.2, size=len(X)))
    event = rng.random(len(X)) > 0.2
    estimator = make_survival_estimator("rsf", seed=2, n_estimators=10)
    estimator.fit(X, duration, event)
    assert estimator.predict_risk(X[:4]).shape == (4,)


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="optional XGBoost dependency is not installed",
)
def test_xgboost_models(rng):
    X = rng.normal(size=(30, 4))
    y = X[:, 0] + rng.normal(scale=0.2, size=len(X))
    estimator = make_estimator("xgboost", "regression", seed=2, n_estimators=5)
    estimator.fit(X, y)
    assert estimator.predict(X[:3]).shape == (3,)


@pytest.mark.parametrize(
    "name",
    [
        "ipw",
        "s_learner",
        "t_learner",
        "x_learner",
        "doubly_robust",
        "policy_learner",
        "tarnet",
        "dragonnet",
    ],
)
def test_causal_models(name, rng):
    X = rng.normal(size=(80, 5))
    propensity = 1 / (1 + np.exp(-X[:, 0]))
    treatment = rng.binomial(1, propensity)
    effect = 1.0 + 0.5 * X[:, 1]
    outcome = X[:, 0] + treatment * effect + rng.normal(scale=0.2, size=len(X))
    params = {"epochs": 3, "hidden_dim": 8} if name in {"tarnet", "dragonnet"} else {}
    estimator = make_cate_estimator(name, seed=2, **params)
    estimator.fit(X, treatment, outcome)
    cate = estimator.predict_cate(X[:10])
    assert cate.shape == (10,)
    assert np.isfinite(cate).all()


@pytest.mark.skipif(
    importlib.util.find_spec("econml") is None,
    reason="optional econml dependency is not installed",
)
def test_causal_forest(rng):
    X = rng.normal(size=(60, 4))
    treatment = rng.binomial(1, 0.5, len(X))
    outcome = X[:, 0] + treatment * (1 + X[:, 1]) + rng.normal(size=len(X))
    estimator = make_cate_estimator(
        "causal_forest", seed=2, n_estimators=8, cv=2
    )
    estimator.fit(X, treatment, outcome)
    assert estimator.predict_cate(X[:5]).shape == (5,)


@pytest.mark.parametrize("name", ["lstm", "gru", "tcn", "transformer"])
def test_temporal_forward(name):
    model = TemporalPredictor(5, 2, name, hidden_dim=8, layers=1, nhead=2)
    output = model(torch.randn(4, 7, 5), torch.tensor([7, 6, 5, 4]))
    assert output.shape == (4, 2)


def test_imaging_genetics(rng):
    genotype = rng.binomial(2, 0.3, size=(60, 12)).astype(float)
    phenotype = genotype[:, 0] * 0.8 + rng.normal(size=60)
    associations = association_scan(genotype, phenotype)
    assert associations.iloc[0]["p_value"] < 0.05
    kinship = genotype @ genotype.T / genotype.shape[1]
    lmm = association_scan_lmm(genotype, phenotype, kinship)
    assert len(lmm) == genotype.shape[1]
    assert polygenic_score(genotype, np.ones(12)).shape == (60,)
    Y = np.column_stack([phenotype, rng.normal(size=60)])
    assert fit_pls(genotype, Y, 2).x_scores.shape == (60, 2)
    assert fit_cca(genotype, Y, 2).y_scores.shape == (60, 2)


def test_roi_glm(rng):
    group = np.r_[np.zeros(20), np.ones(20)]
    design = np.column_stack([np.ones(40), group])
    roi = rng.normal(size=(40, 4))
    roi[:, 0] += group * 2
    result = fit_roi_glm(roi, design)
    assert result.iloc[0]["q_value"] < 0.05


def test_connectome_and_brain_age(rng):
    X = rng.normal(size=(50, 10))
    y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=50)
    model = CPM("regression", p_threshold=0.05).fit(X, y)
    assert model.predict(X[:3]).shape == (3,)
    assert np.allclose(cosine_similarity_map([1, 0], [[1, 0], [0, 1]]), [1, 0])

    age = np.linspace(20, 80, 50)
    predicted = 10 + 0.7 * age
    corrector = BrainAgeBiasCorrector().fit(age, predicted)
    corrected = corrector.transform(age, predicted)
    assert abs(np.polyfit(age, corrected - age, 1)[0]) < 1e-10


def test_cpm_vectorized_correlations_match_scipy(rng):
    X = rng.normal(size=(37, 23)).astype(np.float32)
    X[:, -1] = 1.0
    y = rng.normal(size=37)
    correlations, p_values = CPM._edge_correlations(X, y, chunk_size=7)
    for column in range(X.shape[1] - 1):
        expected = pearsonr(X[:, column], y)
        assert correlations[column] == pytest.approx(expected.statistic, abs=1e-12)
        assert p_values[column] == pytest.approx(expected.pvalue, abs=1e-12)
    assert correlations[-1] == 0.0
    assert p_values[-1] == 1.0


def test_voxel_and_kg_models():
    voxel = VoxelCNN3D(1, 2, base_channels=4)
    assert voxel(torch.randn(2, 1, 16, 16, 16)).shape == (2, 2)

    connectome = BrainNetCNN(
        n_roi=20,
        nclass=2,
        e2e_channels=4,
        e2n_channels=8,
        n2g_channels=16,
        dropout=0.0,
    )
    assert connectome(torch.randn(2, 20, 20)).shape == (2, 2)

    model = GNNLinkPredictor(8, 3, "rgcn", embedding_dim=8, layers=1)
    triples = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
    edge_index = torch.cat([triples[:, [0, 2]].T, triples[:, [2, 0]].T], dim=1)
    edge_type = torch.cat([triples[:, 1], triples[:, 1]])
    assert model(edge_index, edge_type, triples).shape == (3,)
    negatives = sample_negatives(
        triples,
        8,
        {tuple(row) for row in triples.tolist()},
        2,
        np.random.default_rng(1),
    )
    assert negatives.shape == (6, 3)


def test_braingnn_preserves_single_edge_dimension_after_pooling():
    model = BrainGNN(
        indim=2,
        ratio=0.5,
        n_roi=2,
        nclass=2,
        n_communities=2,
        dim1=4,
        dim2=4,
        dim_fc1=4,
    ).eval()
    node_features = torch.tensor([[0.0, 0.4], [0.4, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[0.4], [0.4]])
    batch = torch.zeros(2, dtype=torch.long)
    position = torch.eye(2)

    with torch.no_grad():
        output = model(node_features, edge_index, batch, edge_attr, position)[0]

    assert output.shape == (1, 2)
