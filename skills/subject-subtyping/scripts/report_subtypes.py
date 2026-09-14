"""Export patient-subtype profiles, subject-level QA and descriptive figures.

Consumes frozen assignments; does not retrain, select k using outcomes, or turn
clusters into diagnoses. Optional replicate assignments assess actual stability.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from models.common.research_outputs import OutputBundle, read_json, subject_ids, write_json


def read_table(path, id_column="subject_id"):
    frame = pd.read_csv(path, dtype={id_column: str}, keep_default_na=False)
    if id_column not in frame:
        raise ValueError(f"Missing subject column: {id_column}")
    subject_ids(frame[id_column].tolist(), len(frame))
    return frame.rename(columns={id_column: "subject_id"})


def align_assignments(path, ids, subtype_col):
    frame = read_table(path)
    if subtype_col not in frame or frame[subtype_col].isna().any():
        raise ValueError("Assignments require a subtype for every subject")
    if set(frame.subject_id) != set(ids):
        raise ValueError("Assignment subject set differs from feature subject set")
    frame = frame.set_index("subject_id").loc[ids].reset_index()
    labels = frame[subtype_col].astype(str)
    if (labels.str.strip() == "").any():
        raise ValueError("Empty subtype label")
    return frame, labels.to_numpy()


def stability_rows(reference, candidate, name):
    """Label-permutation-invariant agreement; no arbitrary cluster-ID matching."""
    a, b = np.unique(reference), np.unique(candidate)
    scores = np.empty((len(a), len(b)))
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            scores[i, j] = np.sum((reference == left) & (candidate == right)) / np.sum(
                (reference == left) | (candidate == right))
    ii, jj = linear_sum_assignment(-scores)
    matched = {int(i): int(j) for i, j in zip(ii, jj)}
    ari = float(adjusted_rand_score(reference, candidate))
    return [{"replicate": name, "reference_subtype": str(label),
             "matched_subtype": str(b[matched[i]]) if i in matched else "unmatched",
             "jaccard": float(scores[i, matched[i]]) if i in matched else 0.0,
             "adjusted_rand_index": ari} for i, label in enumerate(a)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-columns", required=True, type=Path, help="JSON array of prespecified feature names")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--assignments", required=True, type=Path, help="Frozen CSV with subject_id, subtype")
    parser.add_argument("--subtype-col", default="subtype")
    parser.add_argument("--replicate", action="append", type=Path, default=[], help="Independent completed clustering assignments; repeat flag")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    frame = read_table(args.features, args.subject_col)
    names = read_json(args.feature_columns)
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or not set(names).issubset(frame.columns)
            or "subject_id" in names):
        raise ValueError("Provide a nonempty unique feature-name list excluding identifiers")
    # Blank cells are missing; strings that are not numerical are not silently coerced.
    features = frame[names].replace("", np.nan).apply(pd.to_numeric, errors="raise")
    if np.isinf(features.to_numpy()).any() or features.isna().all().any():
        raise ValueError("Infinite values or all-missing features")
    ids = frame.subject_id.tolist()
    assignments, labels = align_assignments(args.assignments, ids, args.subtype_col)
    unique = np.unique(labels)
    if not 2 <= len(unique) < len(frame):
        raise ValueError("Reporting requires between 2 and N-1 observed subtypes")
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    filled = imputer.fit_transform(features)
    scaled = scaler.fit_transform(filled)
    if np.unique(scaled, axis=0).shape[0] < 2:
        raise ValueError("No feature variation for descriptive diagnostics")
    silhouette = silhouette_samples(scaled, labels)
    embedding_columns = [c for c in assignments if c.startswith("embedding_")]
    if len(embedding_columns) >= 2:
        embedding = assignments[embedding_columns[:2]].to_numpy(dtype=float)
        embedding_source = "provided clustering embedding (first two columns)"
    else:
        embedding = PCA(n_components=min(2, len(names)), svd_solver="full").fit_transform(scaled)
        embedding_source = "descriptive PCA of median-imputed standardized features; not the clustering model"
        if embedding.shape[1] == 1:
            embedding = np.column_stack([embedding[:, 0], np.zeros(len(frame))])
    if not np.isfinite(embedding).all():
        raise ValueError("Nonfinite embedding")
    output = pd.DataFrame({"subject_id": ids, "subtype": labels,
                           "silhouette": silhouette,
                           "missing_feature_count": features.isna().sum(axis=1).to_numpy()})
    embed_frame = pd.DataFrame({"subject_id": ids, "subtype": labels,
                                "display_1": embedding[:, 0], "display_2": embedding[:, 1]})
    profiles, counts = [], []
    heat = []
    for label in unique:
        group = labels == label
        counts.append({"subtype": label, "n": int(group.sum()), "fraction": float(group.mean()),
                       "mean_silhouette": float(silhouette[group].mean()), "singleton": bool(group.sum() == 1)})
        heat.append(scaled[group].mean(axis=0))
        for index, name in enumerate(names):
            observed = features.loc[group, name].dropna()
            profiles.append({"subtype": label, "feature": name, "n_observed": len(observed),
                             "n_missing": int(group.sum()) - len(observed),
                             "mean": float(observed.mean()) if len(observed) else None,
                             "median": float(observed.median()) if len(observed) else None,
                             "std": float(observed.std(ddof=1)) if len(observed) > 1 else None,
                             "descriptive_standardized_mean": float(scaled[group, index].mean())})
    stability = []
    for index, path in enumerate(args.replicate):
        _, candidate = align_assignments(path, ids, args.subtype_col)
        stability.extend(stability_rows(labels, candidate, f"replicate_{index + 1}:{path.name}"))
    config = {"feature_columns": names, "subtype_column": args.subtype_col,
              "embedding_source": embedding_source, "assignment_source": "frozen input; no retraining"}
    bundle = OutputBundle(args.output_dir, "patient-subtype-report", config,
                          [args.features, args.feature_columns, args.assignments, *args.replicate])
    tables = [("subtype_assignments.csv", output, "subtype_assignments"),
              ("embedding.csv", embed_frame, "embedding"),
              ("subtype_profiles.csv", pd.DataFrame(profiles), "subtype_profiles"),
              ("subtype_counts.csv", pd.DataFrame(counts), "subtype_counts")]
    for filename, table, role in tables:
        table.to_csv(bundle.root / filename, index=False)
        spec = {"subject_ids": ids} if "subject_id" in table else {}
        bundle.add(filename, role, "csv", rows=len(table), columns=list(table.columns), **spec)
    if stability:
        pd.DataFrame(stability).to_csv(bundle.root / "stability.csv", index=False)
        bundle.add("stability.csv", "stability", "csv", columns=["replicate", "jaccard", "adjusted_rand_index"])
    write_json(bundle.root / "descriptive_preprocessing.json", {
        "purpose": "descriptive reporting only; not a replacement for the training pipeline",
        "feature_columns": names, "imputation_medians": imputer.statistics_,
        "means_after_imputation": scaler.mean_, "scales_after_imputation": scaler.scale_,
    })
    qc = {"n_subjects": len(frame), "n_subtypes": len(unique), "n_features": len(names),
          "mean_silhouette": float(silhouette.mean()), "negative_silhouette_count": int((silhouette < 0).sum()),
          "stability_status": "measured_from_supplied_replicates" if stability else "not_evaluated",
          "replicates": len(args.replicate), "clinical_associations": "not_tested",
          "warnings": ["Subtype IDs are arbitrary clusters, not validated diagnoses.",
                       "Descriptive separation is not external replication or clinical utility.",
                       "Replicate agreement only measures the perturbations represented in supplied runs."]}
    write_json(bundle.root / "qc.json", qc)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label in unique:
        idx = labels == label
        axes[0].scatter(embedding[idx, 0], embedding[idx, 1], s=24, label=str(label), alpha=.75)
    axes[0].set(xlabel="display dimension 1", ylabel="display dimension 2", title="Descriptive subtype embedding")
    axes[0].legend(title="Subtype")
    # The complete table remains the authoritative artifact when features are numerous.
    selected = np.argsort(np.var(heat, axis=0))[-min(len(names), 30):]
    handle = axes[1].imshow(np.asarray(heat)[:, selected], aspect="auto", cmap="coolwarm")
    axes[1].set_xticks(range(len(selected)), [names[i] for i in selected], rotation=90)
    axes[1].set_yticks(range(len(unique)), unique)
    axes[1].set_title("Descriptive standardized means (up to 30 features)")
    fig.colorbar(handle, ax=axes[1], shrink=.7)
    fig.tight_layout()
    fig.savefig(bundle.root / "subtype_report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    report = ("# Research patient-subtyping report\n\n"
              f"Subjects: {len(frame)}; subtypes: {len(unique)}; descriptive mean silhouette: {silhouette.mean():.4f}.\n\n"
              f"Stability: {qc['stability_status']} ({len(args.replicate)} supplied replicates).\n\n"
              f"Embedding: {embedding_source}. Profiles use observed values; standardized display values use median imputation.\n\n"
              "No clinical outcome association, external validation or diagnostic interpretation was performed. "
              "Freeze feature selection, preprocessing and subtype definitions before testing endpoints. "
              "For future patients use a separately validated, training-fitted preprocessing and assignment pipeline; "
              "this report does not fit or provide one.\n")
    (bundle.root / "report.md").write_text(report, encoding="utf-8")
    bundle.add("descriptive_preprocessing.json", "descriptive_preprocessing", "json")
    bundle.add("qc.json", "qc", "json")
    bundle.add("subtype_report.png", "subtype_figure")
    bundle.add("report.md", "report")
    required = ["subtype_assignments", "embedding", "subtype_profiles", "subtype_counts",
                "descriptive_preprocessing", "qc", "subtype_figure", "report"]
    if stability:
        required.append("stability")
    print(bundle.finish(required))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
