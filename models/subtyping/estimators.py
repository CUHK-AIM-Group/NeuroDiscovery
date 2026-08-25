"""Clustering, consensus clustering, and autoencoder subtyping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import NMF, PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class SubtypingResult:
    labels: np.ndarray
    embedding: np.ndarray
    metrics: dict[str, float]
    model: Any


def _consensus(
    X: np.ndarray,
    n_clusters: int,
    seed: int,
    n_bootstraps: int = 100,
    sample_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(X)
    together = np.zeros((n, n), dtype=float)
    sampled = np.zeros((n, n), dtype=float)
    sample_size = max(n_clusters * 2, int(np.ceil(n * sample_fraction)))
    for repeat in range(n_bootstraps):
        idx = np.sort(rng.choice(n, size=min(sample_size, n), replace=False))
        labels = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=seed + repeat,
        ).fit_predict(X[idx])
        sampled[np.ix_(idx, idx)] += 1
        together[np.ix_(idx, idx)] += labels[:, None] == labels[None, :]
    consensus = np.divide(
        together,
        sampled,
        out=np.zeros_like(together),
        where=sampled > 0,
    )
    np.fill_diagonal(consensus, 1.0)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(1.0 - consensus)
    return labels, consensus


def _autoencoder_embedding(
    X: np.ndarray,
    latent_dim: int,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x = torch.as_tensor(X, dtype=torch.float32)
    hidden = max(latent_dim * 2, min(128, X.shape[1]))
    encoder = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, latent_dim),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, X.shape[1]),
    )
    model = nn.Sequential(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), x)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        embedding = encoder(x).cpu().numpy()
    return embedding, model


def fit_subtypes(
    X: np.ndarray,
    model: str,
    n_clusters: int,
    seed: int = 123,
    latent_dim: int = 8,
    epochs: int = 100,
    **params: Any,
) -> SubtypingResult:
    X = np.asarray(X, dtype=float)
    if len(X) <= n_clusters:
        raise ValueError("Number of samples must exceed number of clusters")
    scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)
    name = model.lower().replace("-", "_")
    embedding = scaled
    fitted: Any
    if name == "kmeans":
        fitted = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = fitted.fit_predict(scaled)
    elif name in {"gmm", "gaussian_mixture"}:
        fitted = GaussianMixture(
            n_components=n_clusters,
            covariance_type=params.get("covariance_type", "full"),
            n_init=int(params.get("n_init", 10)),
            random_state=seed,
        )
        labels = fitted.fit_predict(scaled)
    elif name in {"spectral", "spectral_clustering"}:
        fitted = SpectralClustering(
            n_clusters=n_clusters,
            affinity=params.get("affinity", "nearest_neighbors"),
            assign_labels="kmeans",
            random_state=seed,
        )
        labels = fitted.fit_predict(scaled)
    elif name == "nmf":
        nonnegative = MinMaxScaler().fit_transform(X)
        reducer = NMF(
            n_components=max(n_clusters, latent_dim),
            init="nndsvda",
            max_iter=int(params.get("max_iter", 1000)),
            random_state=seed,
        )
        embedding = reducer.fit_transform(nonnegative)
        fitted = (reducer, KMeans(n_clusters=n_clusters, n_init=20, random_state=seed))
        labels = fitted[1].fit_predict(embedding)
    elif name in {"consensus", "consensus_clustering"}:
        labels, embedding = _consensus(
            scaled,
            n_clusters=n_clusters,
            seed=seed,
            n_bootstraps=int(params.get("n_bootstraps", 100)),
            sample_fraction=float(params.get("sample_fraction", 0.8)),
        )
        fitted = {"consensus_matrix": embedding}
    elif name in {"autoencoder", "ae"}:
        embedding, autoencoder = _autoencoder_embedding(
            scaled, latent_dim=latent_dim, seed=seed, epochs=epochs
        )
        clusterer = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = clusterer.fit_predict(embedding)
        fitted = (autoencoder, clusterer)
    elif name == "pca":
        reducer = PCA(n_components=min(latent_dim, X.shape[1]), random_state=seed)
        embedding = reducer.fit_transform(scaled)
        clusterer = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = clusterer.fit_predict(embedding)
        fitted = (reducer, clusterer)
    else:
        raise ValueError(f"Unknown subtyping model: {model}")
    metrics = {
        "silhouette": float(silhouette_score(scaled, labels)),
        "n_clusters": float(len(np.unique(labels))),
    }
    return SubtypingResult(labels, embedding, metrics, fitted)
