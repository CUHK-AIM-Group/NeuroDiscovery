"""Run a leakage-controlled multi-model robustness experiment.

The candidate unit remains disease x ROI x imaging feature.  Models are trained
once per disease and atlas, then held-out attributions are projected back to
those candidate units.  This avoids fitting one model per candidate while still
letting conventional and deep models contribute comparable evidence.  The
historical filename remains as a compatibility entry point; ``--case-study-id``
records the formal task that owns a run.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.scripts.case1_exhaustive_full import (  # noqa: E402
    FULL_FMRI_FEATURES,
    build_atlas_feature_matrices,
)
from core.scripts.case1_exhaustive_v1 import (  # noqa: E402
    disease_masks,
    load_metadata,
)
from core.scripts.case1_exhaustive_v2 import build_covariates  # noqa: E402
from models.bnt.net.bnt import BrainNetworkTransformer  # noqa: E402
from models.brainnetcnn.net.brainnetcnn import BrainNetCNN  # noqa: E402
from models.braingnn.net.braingnn import BrainGNN, topk_loss  # noqa: E402
from models.combraintf.net.combraintf import ComBrainTF  # noqa: E402
from models.combraintf.scripts.data_adapter import build_community_ids  # noqa: E402
from models.ibgnn.net.ibgnn import IBGNN  # noqa: E402
from models.lggnn.net.lggnn import LGGNN  # noqa: E402


DEFAULT_DATA_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\transdiag_preprocessed")
DEFAULT_OUT_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\case1_multimodel_pilot")
DEFAULT_ATLASES = ("schaefer_200_7net", "glasser_360")
DEFAULT_DISEASES = ("MDD_depression", "anxiety", "substance_use")
DEFAULT_MODELS = (
    "elasticnet",
    "roi_mlp",
    "brainnetcnn",
    "braingnn",
    "bnt",
    "ibgnn",
    "lggnn",
    "combraintf",
)
DENSE_CONNECTOME_MODELS = frozenset({"brainnetcnn", "bnt", "combraintf"})
GRAPH_MODELS = frozenset({"braingnn", "ibgnn", "lggnn"})

warnings.filterwarnings(
    "ignore",
    message="The usage of `scatter\\(reduce='max'\\)` can be accelerated.*",
    category=UserWarning,
)


@dataclass(frozen=True)
class FoldData:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class ROIMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.30):
        super().__init__()
        bottleneck = max(24, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseGraphDataset(torch.utils.data.Dataset):
    """Create PyG connectomes lazily without caching graph objects in RAM."""

    def __init__(
        self,
        matrices: np.ndarray,
        labels: np.ndarray,
        subjects: list[str],
        edge_density: float = 1.0,
    ):
        self.matrices = matrices
        self.labels = labels
        self.subjects = subjects
        self.edge_density = float(edge_density)
        if not 0.0 < self.edge_density <= 1.0:
            raise ValueError("edge_density must be in (0, 1]")
        n_roi = matrices.shape[1]
        self.n_roi = n_roi
        if self.edge_density >= 1.0:
            mask = ~torch.eye(n_roi, dtype=torch.bool)
            self.src, self.dst = torch.nonzero(mask, as_tuple=True)
            self.edge_index = torch.stack([self.src, self.dst], dim=0)
        else:
            self.src = self.dst = self.edge_index = None
        self.pos = torch.eye(n_roi, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Data:
        corr = torch.from_numpy(self.matrices[index]).float()
        if self.edge_index is None:
            k = max(1, int(math.ceil((self.n_roi - 1) * self.edge_density)))
            scores = corr.abs().clone()
            scores.fill_diagonal_(-math.inf)
            dst = torch.topk(scores, k=k, dim=1, sorted=False).indices.reshape(-1)
            src = torch.arange(self.n_roi).repeat_interleave(k)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            src, dst, edge_index = self.src, self.dst, self.edge_index
        graph = Data(
            x=corr,
            edge_index=edge_index,
            edge_attr=corr[src, dst].abs().view(-1, 1),
            pos=self.pos,
            y=torch.tensor(int(self.labels[index]), dtype=torch.long),
        )
        graph.subject_id = self.subjects[index]
        return graph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def _supports_stratification(labels: np.ndarray, n_splits: int) -> bool:
    _, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return bool(len(counts) > 1 and counts.min() >= n_splits)


def choose_stratification(
    y: np.ndarray,
    sites: np.ndarray | None,
    n_splits: int,
) -> tuple[np.ndarray, str]:
    """Use diagnosis x site only when every stratum supports every fold."""
    diagnosis = np.asarray(y).astype(str)
    if sites is not None:
        combined = np.char.add(np.char.add(diagnosis, "_site_"), np.asarray(sites).astype(str))
        if _supports_stratification(combined, n_splits):
            return combined, "diagnosis_x_site"
    return diagnosis, "diagnosis"


def make_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> list[FoldData]:
    outer_labels = np.asarray(strata) if strata is not None else np.asarray(y)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldData] = []
    for train_val, test in outer.split(np.arange(len(y)), outer_labels):
        inner_labels = outer_labels[train_val]
        # A small site x diagnosis cell can support the outer folds but not a
        # 20% validation split. In that case retain diagnosis stratification.
        if not _supports_stratification(inner_labels, 2):
            inner_labels = np.asarray(y)[train_val]
        try:
            train, val = train_test_split(
                train_val,
                test_size=0.20,
                stratify=inner_labels,
                random_state=seed + len(folds) + 1,
            )
        except ValueError:
            train, val = train_test_split(
                train_val,
                test_size=0.20,
                stratify=np.asarray(y)[train_val],
                random_state=seed + len(folds) + 1,
            )
        folds.append(FoldData(np.asarray(train), np.asarray(val), np.asarray(test)))
    return folds


def fold_adjust_standardize(
    values: np.ndarray,
    covariates: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Residualize and standardize using training subjects only."""
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    train_mean = np.nanmean(flat[train_indices], axis=0)
    train_mean = np.nan_to_num(train_mean, nan=0.0)
    flat = np.where(np.isfinite(flat), flat, train_mean[None, :])

    design = np.column_stack([np.ones(len(covariates)), covariates]).astype(np.float64)
    beta = np.linalg.pinv(design[train_indices]) @ flat[train_indices].astype(np.float64)
    residual = flat.astype(np.float64) - design @ beta
    scale = residual[train_indices].std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    adjusted = residual / scale
    return adjusted.astype(np.float32).reshape(original_shape)


def load_correlation_matrices(
    root: Path,
    atlas: str,
    subjects: Iterable[str],
) -> np.ndarray:
    matrices = []
    directory = root / "fc" / atlas / "correlation"
    for subject in subjects:
        path = directory / f"{subject}_{atlas}_correlation.npy"
        matrix = np.load(path).astype(np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    return np.stack(matrices)


def class_weights(y: np.ndarray, train: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y[train], minlength=2).astype(float)
    weights = len(train) / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _dense_output(
    model: nn.Module,
    features: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, object | None]:
    output = model(features)
    if model_name in {"bnt", "combraintf"}:
        return output[0], output[1]
    return output, None


def _graph_output(
    model: nn.Module,
    batch: Data,
    model_name: str,
) -> tuple[torch.Tensor, object | None]:
    if model_name == "braingnn":
        output = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos)
        return output[0], output[1:]
    if model_name == "lggnn":
        return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr), None


def _compact_community_ids(ids: list[int]) -> list[int]:
    remap = {value: index for index, value in enumerate(sorted(set(ids)))}
    return [remap[value] for value in ids]


def make_dense_connectome_model(
    model_name: str,
    n_roi: int,
    atlas: str,
    roi_meta: pd.DataFrame,
) -> nn.Module:
    if model_name == "brainnetcnn":
        return BrainNetCNN(
            n_roi=n_roi,
            nclass=2,
            e2e_channels=8,
            e2n_channels=16,
            n2g_channels=64,
            dropout=0.30,
        )
    if model_name == "bnt":
        return BrainNetworkTransformer(
            n_roi=n_roi,
            sizes=[max(12, n_roi // 4)],
            do_pooling=[True],
            pos_embed_dim=8,
            hidden_size=128,
            nhead=4,
            dropout=0.20,
            nclass=2,
        )
    if model_name == "combraintf":
        names = roi_meta.get(
            "parcel_name",
            roi_meta.get("roi_name", pd.Series([f"ROI_{i}" for i in range(n_roi)])),
        ).astype(str).tolist()
        communities = _compact_community_ids(build_community_ids(names, atlas, n_communities=8))
        nhead = max(head for head in (8, 4, 2, 1) if n_roi % head == 0)
        return ComBrainTF(
            n_roi=n_roi,
            nclass=2,
            community_ids=communities,
            n_communities=len(set(communities)),
            n_clusters=min(8, n_roi),
            hidden_size=128,
            nhead=nhead,
        )
    raise ValueError(f"Unsupported dense connectome model: {model_name}")


def _dense_probabilities(
    model: nn.Module,
    x: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    model_name: str,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[indices])),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities = []
    model.eval()
    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            logits, _ = _dense_output(model, features, model_name)
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def train_dense_model(
    *,
    model_name: str,
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    fold: FoldData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[nn.Module, dict[str, float], int]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, fold.train, device))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[fold.train]), torch.from_numpy(y[fold.train]).long()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(fold.train) % batch_size == 1,
    )
    best_score = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = _dense_output(model, features, model_name)
            loss = loss_fn(logits, labels)
            if model_name in {"bnt", "combraintf"}:
                loss = loss + 0.05 * model.dec_loss(auxiliary)
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        val_prob = _dense_probabilities(model, x, fold.val, device, batch_size, model_name)
        score = roc_auc_score(y[fold.val], val_prob)
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    test_prob = _dense_probabilities(model, x, fold.test, device, batch_size, model_name)
    return model, classification_metrics(y[fold.test], test_prob), epochs_run


def _graph_probabilities(
    model: nn.Module,
    dataset: DenseGraphDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    model_name: str,
) -> np.ndarray:
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = PyGDataLoader(subset, batch_size=batch_size, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output, _ = _graph_output(model, batch, model_name)
            probability = output.exp()[:, 1] if model_name == "braingnn" else torch.softmax(output, dim=1)[:, 1]
            probabilities.append(probability.cpu().numpy())
    return np.concatenate(probabilities)


def train_graph_model(
    *,
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    fold: FoldData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    edge_density: float,
) -> tuple[nn.Module, DenseGraphDataset, dict[str, float], int]:
    dataset = DenseGraphDataset(x, y, subjects, edge_density=edge_density)
    n_roi = x.shape[1]
    if model_name == "braingnn":
        model = BrainGNN(
            indim=n_roi,
            ratio=0.5,
            nclass=2,
            n_roi=n_roi,
            n_communities=8,
            dim1=16,
            dim2=16,
            dim_fc1=64,
        )
    elif model_name == "ibgnn":
        model = IBGNN(n_roi=n_roi, nclass=2, hidden_dim=32, n_gnn_layers=2)
    elif model_name == "lggnn":
        model = LGGNN(n_roi=n_roi, nclass=2, hidden_dim=32, embed_dim=16, ratio=0.5)
    else:
        raise ValueError(f"Unsupported graph model: {model_name}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = class_weights(y, fold.train, device)
    train_loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, fold.train.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(fold.train) % batch_size == 1,
    )
    best_score = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, auxiliary = _graph_output(model, batch, model_name)
            if model_name == "braingnn":
                score1, score2 = auxiliary[-2:]
                loss = F.nll_loss(output, batch.y, weight=weights)
                loss = loss + 0.05 * topk_loss(score1, 0.5) + 0.05 * topk_loss(score2, 0.5)
            else:
                loss = F.cross_entropy(output, batch.y, weight=weights)
                if model_name == "lggnn":
                    loss = loss - 0.05 * auxiliary
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        val_prob = _graph_probabilities(
            model, dataset, fold.val, device, batch_size, model_name
        )
        score = roc_auc_score(y[fold.val], val_prob)
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    test_prob = _graph_probabilities(
        model, dataset, fold.test, device, batch_size, model_name
    )
    return model, dataset, classification_metrics(y[fold.test], test_prob), epochs_run


def dense_attributions(
    model: nn.Module,
    x: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.zeros(x.shape[1:], dtype=np.float64)
    signed = np.zeros(x.shape[1:], dtype=np.float64)
    count = 0
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        features = torch.from_numpy(x[batch_indices]).to(device).requires_grad_(True)
        logits, _ = _dense_output(model, features, model_name)
        model.zero_grad(set_to_none=True)
        logits[:, 1].sum().backward()
        attribution = (features.grad * features).detach().cpu().numpy()
        absolute += np.abs(attribution).sum(axis=0)
        signed += attribution.sum(axis=0)
        count += len(batch_indices)
    return (absolute / max(count, 1)).astype(float), (signed / max(count, 1)).astype(float)


def graph_attributions(
    model: nn.Module,
    dataset: DenseGraphDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_roi = dataset.matrices.shape[1]
    absolute = np.zeros(n_roi, dtype=np.float64)
    signed = np.zeros(n_roi, dtype=np.float64)
    count = 0
    loader = PyGDataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        batch.x.requires_grad_(True)
        output, _ = _graph_output(model, batch, model_name)
        model.zero_grad(set_to_none=True)
        output[:, 1].sum().backward()
        attribution = (batch.x.grad * batch.x).detach()
        for graph_index in range(batch.num_graphs):
            lo = int(batch.ptr[graph_index])
            hi = int(batch.ptr[graph_index + 1])
            node_attr = attribution[lo:hi]
            absolute += node_attr.abs().mean(dim=1).cpu().numpy()
            signed += node_attr.mean(dim=1).cpu().numpy()
            count += 1
    return absolute / max(count, 1), signed / max(count, 1)


def attribution_rows(
    *,
    model_name: str,
    atlas: str,
    disease: str,
    seed: int,
    fold_index: int,
    roi_meta: pd.DataFrame,
    absolute: np.ndarray,
    signed: np.ndarray,
    feature_names: tuple[str, ...] | None,
) -> list[dict[str, object]]:
    rows = []
    if feature_names is None:
        absolute = np.asarray(absolute)
        signed = np.asarray(signed)
        if absolute.ndim == 2:
            absolute = (absolute.mean(axis=0) + absolute.mean(axis=1)) / 2.0
            signed = (signed.mean(axis=0) + signed.mean(axis=1)) / 2.0
        feature_names = ("correlation_profile",)
        absolute = absolute[:, None]
        signed = signed[:, None]
        resolution = "roi"
    else:
        resolution = "roi_feature"
    for roi_index in range(len(roi_meta)):
        roi = roi_meta.iloc[roi_index]
        for feature_index, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "model": model_name,
                    "atlas": atlas,
                    "disease": disease,
                    "seed": seed,
                    "fold": fold_index,
                    "evidence_resolution": resolution,
                    "roi_index": roi_index,
                    "roi_id": roi.get("roi_id", roi_index + 1),
                    "roi_name": roi.get("parcel_name", roi.get("roi_name", f"roi_{roi_index + 1}")),
                    "hemisphere": roi.get("hemisphere", ""),
                    "network": roi.get("network", ""),
                    "feature": feature_name,
                    "importance_abs": float(absolute[roi_index, feature_index]),
                    "signed_attribution": float(signed[roi_index, feature_index]),
                }
            )
    return rows


def run_elasticnet(
    x: np.ndarray,
    y: np.ndarray,
    fold: FoldData,
    seed: int,
) -> tuple[LogisticRegression, dict[str, float], int]:
    model = LogisticRegression(
        solver="saga",
        l1_ratio=0.5,
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(x[fold.train], y[fold.train])
    probability = model.predict_proba(x[fold.test])[:, 1]
    return model, classification_metrics(y[fold.test], probability), int(model.n_iter_[0])


def model_parameter_count(model: object) -> int:
    if isinstance(model, nn.Module):
        return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    return 0


def run_model_fold(
    *,
    model_name: str,
    atlas: str,
    disease: str,
    seed: int,
    fold_index: int,
    fold: FoldData,
    y: np.ndarray,
    subjects: list[str],
    adjusted_nodes: np.ndarray,
    adjusted_corr: np.ndarray | None,
    roi_meta: pd.DataFrame,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    graph_density: float,
    skip_attribution: bool,
) -> tuple[object, dict[str, float], int, list[dict[str, object]]]:
    model_evidence: list[dict[str, object]] = []
    flat_nodes = adjusted_nodes.reshape(len(y), -1)
    feature_names: tuple[str, ...] | None

    if model_name == "elasticnet":
        fitted, metrics, epochs_run = run_elasticnet(flat_nodes, y, fold, seed)
        if not skip_attribution:
            absolute = np.abs(fitted.coef_[0]).reshape(adjusted_nodes.shape[1:])
            signed = fitted.coef_[0].reshape(adjusted_nodes.shape[1:])
            feature_names = tuple(FULL_FMRI_FEATURES)
    elif model_name == "roi_mlp":
        fitted, metrics, epochs_run = train_dense_model(
            model_name=model_name,
            model=ROIMLP(flat_nodes.shape[1]),
            x=flat_nodes,
            y=y,
            fold=fold,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )
        if not skip_attribution:
            absolute, signed = dense_attributions(
                fitted, flat_nodes, fold.test, device, batch_size, model_name
            )
            absolute = absolute.reshape(adjusted_nodes.shape[1:])
            signed = signed.reshape(adjusted_nodes.shape[1:])
            feature_names = tuple(FULL_FMRI_FEATURES)
    elif model_name in DENSE_CONNECTOME_MODELS:
        if adjusted_corr is None:
            raise RuntimeError(f"{model_name} requires correlation matrices")
        n_roi = adjusted_corr.shape[1]
        dense_batch_size = max(
            1,
            min(batch_size, 2 if model_name == "combraintf" else 4),
        )
        fitted, metrics, epochs_run = train_dense_model(
            model_name=model_name,
            model=make_dense_connectome_model(model_name, n_roi, atlas, roi_meta),
            x=adjusted_corr,
            y=y,
            fold=fold,
            device=device,
            epochs=epochs,
            batch_size=dense_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )
        if not skip_attribution:
            absolute, signed = dense_attributions(
                fitted,
                adjusted_corr,
                fold.test,
                device,
                dense_batch_size,
                model_name,
            )
            feature_names = None
    elif model_name in GRAPH_MODELS:
        if adjusted_corr is None:
            raise RuntimeError(f"{model_name} requires correlation matrices")
        graph_batch_size = max(1, min(batch_size, 2 if graph_density > 0.25 else 4))
        fitted, graph_dataset, metrics, epochs_run = train_graph_model(
            model_name=model_name,
            x=adjusted_corr,
            y=y,
            subjects=subjects,
            fold=fold,
            device=device,
            epochs=epochs,
            batch_size=graph_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            edge_density=graph_density,
        )
        if not skip_attribution:
            absolute, signed = graph_attributions(
                fitted,
                graph_dataset,
                fold.test,
                device,
                graph_batch_size,
                model_name,
            )
            feature_names = None
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not skip_attribution:
        model_evidence = attribution_rows(
            model_name=model_name,
            atlas=atlas,
            disease=disease,
            seed=seed,
            fold_index=fold_index,
            roi_meta=roi_meta,
            absolute=absolute,
            signed=signed,
            feature_names=feature_names,
        )
    return fitted, metrics, epochs_run, model_evidence


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv_with_retry(
    frame: pd.DataFrame,
    path: Path,
    *,
    mode: str = "w",
    header: bool = True,
    attempts: int = 8,
) -> None:
    """Tolerate short-lived file locks on network-mounted result directories."""
    for attempt in range(attempts):
        try:
            frame.to_csv(path, mode=mode, header=header, index=False)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.25 * (2**attempt), 4.0))


def summarize_performance(
    performance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_summary = (
        performance.groupby(["atlas", "disease", "model", "seed"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
    )
    cell_summary = (
        seed_summary.groupby(["atlas", "disease", "model"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_variance=("auroc", "var"),
            auprc_mean=("auprc", "mean"),
            auprc_variance=("auprc", "var"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_variance=("balanced_accuracy", "var"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["atlas", "disease", "auroc_mean"], ascending=[True, True, False])
    )
    best_by_cell = (
        cell_summary.sort_values(
            ["atlas", "disease", "auroc_mean", "auroc_variance"],
            ascending=[True, True, False, True],
        )
        .groupby(["atlas", "disease"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_disease = (
        cell_summary.sort_values(
            ["disease", "auroc_mean", "auroc_variance"],
            ascending=[True, False, True],
        )
        .groupby("disease", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return seed_summary, cell_summary, best_by_cell, best_by_disease


def summarize_attributions(
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate folds within seed before estimating between-seed variance."""
    keys = [
        "model",
        "atlas",
        "disease",
        "evidence_resolution",
        "roi_index",
        "roi_id",
        "roi_name",
        "hemisphere",
        "network",
        "feature",
    ]
    seed_summary = (
        evidence.groupby(keys + ["seed"], dropna=False, as_index=False)
        .agg(
            importance_abs=("importance_abs", "mean"),
            signed_attribution=("signed_attribution", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
    summary = (
        seed_summary.groupby(keys, dropna=False, as_index=False)
        .agg(
            importance_abs_mean=("importance_abs", "mean"),
            importance_abs_variance=("importance_abs", "var"),
            signed_attribution_mean=("signed_attribution", "mean"),
            signed_attribution_variance=("signed_attribution", "var"),
            n_seeds=("seed", "nunique"),
        )
    )
    return seed_summary, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study-id", default="case1_transdiagnostic")
    parser.add_argument("--transdiag-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diagnosis", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--atlases", default=",".join(DEFAULT_ATLASES))
    parser.add_argument("--diseases", default=",".join(DEFAULT_DISEASES))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default="20260804")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--graph-density",
        type=float,
        default=1.0,
        help="Strongest directed edges retained per ROI for PyG models; 1 keeps all edges.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--split-stratification",
        choices=("diagnosis", "diagnosis_x_site_if_feasible"),
        default="diagnosis",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=None,
        help="Optional frozen table/build manifest recorded as provenance.",
    )
    args = parser.parse_args()

    atlases = parse_csv_list(args.atlases)
    diseases = parse_csv_list(args.diseases)
    models = parse_csv_list(args.models)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    unknown = set(models) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    if not 0.0 < args.graph_density <= 1.0:
        raise ValueError("--graph-density must be in (0, 1]")
    if args.smoke:
        atlases = atlases[:1]
        diseases = diseases[:1]
        seeds = seeds[:1]
        args.folds = 2
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)

    diagnosis = args.diagnosis or args.transdiag_root / "metadata" / "diagnosis.csv"
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    performance_path = out_dir / "performance_folds.csv"
    failure_path = out_dir / "failed_model_folds.csv"
    evidence_path = out_dir / "heldout_attributions.csv"
    split_path = out_dir / "fold_assignments.csv"
    if out_dir.exists() and performance_path.exists() and not args.resume:
        raise FileExistsError(
            f"Run already contains performance results: {out_dir}. Use --resume or a new --run-name."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.resume and performance_path.exists():
        performance_rows = pd.read_csv(performance_path).to_dict("records")
    else:
        performance_rows: list[dict[str, object]] = []
    completed = {
        (
            str(row["atlas"]),
            str(row["disease"]),
            str(row["model"]),
            int(row["seed"]),
            int(row["fold"]),
        )
        for row in performance_rows
    }
    if args.resume and failure_path.exists():
        failure_rows = pd.read_csv(failure_path).to_dict("records")
    else:
        failure_rows: list[dict[str, object]] = []
    needs_correlation = any(
        model in DENSE_CONNECTOME_MODELS or model in GRAPH_MODELS for model in models
    )
    if args.resume and split_path.exists():
        split_rows = pd.read_csv(split_path, dtype={"subject": str}).to_dict("records")
    else:
        split_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    print(
        f"{args.case_study_id} multimodel robustness device={device} "
        f"atlases={atlases} diseases={diseases} "
        f"models={models} folds={args.folds} seeds={seeds}",
        flush=True,
    )
    for atlas in atlases:
        print(f"Loading atlas={atlas}", flush=True)
        source_subjects, roi_meta, feature_matrices = build_atlas_feature_matrices(
            args.transdiag_root, atlas, requested_subjects=None
        )
        meta, available_diseases = load_metadata(diagnosis, source_subjects, min_cases=5)
        subjects = meta["subjectkey"].astype(str).tolist()
        source_order = {subject: index for index, subject in enumerate(source_subjects)}
        if len(source_order) != len(source_subjects):
            raise ValueError("Duplicate subject identifiers in metadata")
        aligned_indices = [source_order[subject] for subject in subjects]
        node_features = np.stack(
            [feature_matrices[name][aligned_indices] for name in FULL_FMRI_FEATURES], axis=-1
        ).astype(np.float32)
        correlations = (
            load_correlation_matrices(args.transdiag_root, atlas, subjects)
            if needs_correlation
            else None
        )
        covariates = build_covariates(meta).to_numpy(np.float32)
        available_names = {str(row["disease"]) for row in available_diseases}

        for disease in diseases:
            if disease not in available_names:
                print(f"Skipping unavailable disease={disease} atlas={atlas}", flush=True)
                continue
            case_mask, control_mask = disease_masks(meta, disease)
            subset = np.flatnonzero(case_mask | control_mask)
            y = case_mask[subset].astype(np.int64)
            disease_subjects = [subjects[index] for index in subset]
            raw_nodes = node_features[subset]
            raw_corr = correlations[subset] if correlations is not None else None
            disease_covariates = covariates[subset]
            site_column = next(
                (column for column in meta.columns if str(column).strip().lower() == "site"),
                None,
            )
            disease_sites = (
                meta.iloc[subset][site_column].fillna("unknown").astype(str).to_numpy()
                if site_column is not None
                else None
            )
            strata, split_stratification = choose_stratification(
                y,
                disease_sites if args.split_stratification == "diagnosis_x_site_if_feasible" else None,
                args.folds,
            )
            print(
                f"atlas={atlas} disease={disease} n={len(y)} "
                f"case={int(y.sum())} control={int((y == 0).sum())}",
                flush=True,
            )
            for seed in seeds:
                set_seed(seed)
                folds = make_folds(y, args.folds, seed, strata=strata)
                for fold_index, fold in enumerate(folds):
                    for split_name, indices in (
                        ("train", fold.train), ("validation", fold.val), ("test", fold.test)
                    ):
                        split_rows.extend(
                            {
                                "atlas": atlas,
                                "disease": disease,
                                "seed": seed,
                                "fold": fold_index,
                                "split": split_name,
                                "subject": disease_subjects[index],
                                "label": int(y[index]),
                                "site": (
                                    str(disease_sites[index])
                                    if disease_sites is not None
                                    else ""
                                ),
                                "stratification": split_stratification,
                            }
                            for index in indices
                        )

                    adjusted_nodes = fold_adjust_standardize(raw_nodes, disease_covariates, fold.train)
                    adjusted_corr = (
                        fold_adjust_standardize(raw_corr, disease_covariates, fold.train)
                        if raw_corr is not None
                        else None
                    )
                    if adjusted_corr is not None:
                        for matrix in adjusted_corr:
                            np.fill_diagonal(matrix, 0.0)
                    flat_nodes = adjusted_nodes.reshape(len(y), -1)

                    for model_index, model_name in enumerate(models):
                        key = (atlas, disease, model_name, seed, fold_index)
                        if key in completed:
                            print(
                                f"  fold={fold_index} model={model_name} already complete; skipping",
                                flush=True,
                            )
                            continue
                        set_seed(seed + fold_index * 101 + model_index * 1009)
                        fit_start = time.perf_counter()
                        try:
                            fitted, metrics, epochs_run, model_evidence = run_model_fold(
                                model_name=model_name,
                                atlas=atlas,
                                disease=disease,
                                seed=seed,
                                fold_index=fold_index,
                                fold=fold,
                                y=y,
                                subjects=disease_subjects,
                                adjusted_nodes=adjusted_nodes,
                                adjusted_corr=adjusted_corr,
                                roi_meta=roi_meta,
                                device=device,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.lr,
                                weight_decay=args.weight_decay,
                                patience=args.patience,
                                graph_density=args.graph_density,
                                skip_attribution=args.skip_attribution,
                            )
                        except Exception as exc:
                            elapsed = time.perf_counter() - fit_start
                            failure = {
                                "atlas": atlas,
                                "disease": disease,
                                "model": model_name,
                                "seed": seed,
                                "fold": fold_index,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "elapsed_sec": round(elapsed, 3),
                            }
                            failure_rows.append(failure)
                            write_csv_with_retry(
                                pd.DataFrame([failure]),
                                failure_path,
                                mode="a",
                                header=not failure_path.exists(),
                            )
                            print(
                                f"  fold={fold_index} model={model_name} FAILED "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if args.fail_fast:
                                raise
                            continue

                        elapsed = time.perf_counter() - fit_start
                        row = {
                            "atlas": atlas,
                            "disease": disease,
                            "model": model_name,
                            "seed": seed,
                            "fold": fold_index,
                            "n_case": int(y.sum()),
                            "n_control": int((y == 0).sum()),
                            "n_train": len(fold.train),
                            "n_validation": len(fold.val),
                            "n_test": len(fold.test),
                            "epochs_or_iterations": epochs_run,
                            "parameters": model_parameter_count(fitted),
                            "elapsed_sec": round(elapsed, 3),
                            **metrics,
                        }
                        # Attribution is committed before the performance row, which is the
                        # resume marker. A crash can then cause harmless duplicate evidence,
                        # but cannot mark a fold complete while losing its attribution.
                        if model_evidence:
                            write_csv_with_retry(
                                pd.DataFrame(model_evidence),
                                evidence_path,
                                mode="a",
                                header=not evidence_path.exists(),
                            )
                        performance_rows.append(row)
                        write_csv_with_retry(
                            pd.DataFrame([row]),
                            performance_path,
                            mode="a",
                            header=not performance_path.exists(),
                        )
                        completed.add(key)
                        print(
                            f"  fold={fold_index} model={model_name} "
                            f"AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f} "
                            f"BAcc={metrics['balanced_accuracy']:.3f} time={elapsed:.1f}s",
                            flush=True,
                        )
                        del fitted
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    performance = pd.DataFrame(performance_rows)
    performance = performance.drop_duplicates(
        ["atlas", "disease", "model", "seed", "fold"], keep="last"
    )
    write_csv_with_retry(performance, performance_path)
    failures = pd.DataFrame(failure_rows)
    if not failures.empty:
        failures = failures.drop_duplicates(
            ["atlas", "disease", "model", "seed", "fold"], keep="last"
        )
        unresolved = [
            (
                str(row.atlas),
                str(row.disease),
                str(row.model),
                int(row.seed),
                int(row.fold),
            )
            not in completed
            for row in failures.itertuples(index=False)
        ]
        failures = failures.loc[unresolved].reset_index(drop=True)
    if failures.empty:
        failure_path.unlink(missing_ok=True)
    else:
        write_csv_with_retry(failures, failure_path)
    seed_summary, summary, best_by_cell, best_by_disease = summarize_performance(performance)
    write_csv_with_retry(seed_summary, out_dir / "performance_by_seed.csv")
    write_csv_with_retry(summary, out_dir / "performance_summary.csv")
    write_csv_with_retry(best_by_cell, out_dir / "best_model_by_atlas_disease.csv")
    write_csv_with_retry(best_by_disease, out_dir / "best_atlas_model_by_disease.csv")
    splits = pd.DataFrame(split_rows).drop_duplicates(
        ["atlas", "disease", "seed", "fold", "split", "subject"], keep="last"
    )
    write_csv_with_retry(splits, split_path)
    n_attribution_model_folds = 0
    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path, low_memory=False)
        evidence = evidence.drop_duplicates(
            ["model", "atlas", "disease", "seed", "fold", "roi_index", "feature"],
            keep="last",
        )
        n_attribution_model_folds = len(
            evidence[["model", "atlas", "disease", "seed", "fold"]].drop_duplicates()
        )
        write_csv_with_retry(evidence, evidence_path)
        evidence_by_seed, evidence_summary = summarize_attributions(evidence)
        write_csv_with_retry(
            evidence_by_seed,
            out_dir / "heldout_attribution_by_seed.csv",
        )
        write_csv_with_retry(evidence_summary, out_dir / "heldout_attribution_summary.csv")

    manifest = {
        "schema_version": "case-study-multimodel-robustness.v2",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_study_id": args.case_study_id,
        "run_name": run_name,
        "transdiag_root": str(args.transdiag_root),
        "diagnosis": str(diagnosis),
        "out_dir": str(out_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "atlases": atlases,
        "diseases": diseases,
        "models": models,
        "folds": args.folds,
        "seeds": seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "graph_density": args.graph_density,
        "candidate_unit": "disease_x_roi_x_imaging_feature",
        "split_stratification_requested": args.split_stratification,
        "split_stratification_policy": (
            "diagnosis x site when every stratum supports all outer folds; diagnosis otherwise"
        ),
        "covariate_policy": "fit training fold only; age, sex, site, and mean_fd when available",
        "classification_unit": "one binary disease-vs-control task per atlas",
        "attribution_enabled": not args.skip_attribution,
        "evidence_policy": "held-out gradient-times-input; elastic-net standardized coefficients",
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "n_model_folds": len(performance),
        "n_attribution_model_folds": n_attribution_model_folds,
        "n_failed_model_folds": len(failures),
        "protocol_manifest": str(args.protocol_manifest) if args.protocol_manifest else None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
