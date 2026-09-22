"""Run execution pilots for the Case Studies supported by current local data.

The suite is deliberately split across public and controlled NAS roots. TCP and
HCP-YA artifacts are written under ``Public Dataset``. ADNI subject-level
inputs, predictions, and checkpoints stay beside the controlled ADNI data; the
public run only receives aggregate metrics and paths to the controlled run.

This is an execution-readiness pilot, not a replacement for each Case Study's
full preregistered benchmark. Case 1 and Case 2 keep their specialised runners;
the remaining tasks use canonical NeuroRuntime model entry points.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
NAS_PUBLIC = Path(r"\\192.168.3.61\data\Public Dataset")
ADNI_CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_PUBLIC_RUN_ROOT = NAS_PUBLIC / "case_study_pilots"
DEFAULT_CONTROLLED_RUN_ROOT = ADNI_CASE2_ROOT / "experiments" / "direct_case_study_pilots"
DEFAULT_GRAPH = ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
SCHAEFER_LABELS = (
    Path(r"C:\Users\45846\Documents\Code\NeuroSTORM\datasets\atlas")
    / "schaefer_100_7net"
    / "labels.csv"
)

DIRECT_CASE_STUDIES = (
    "case1_transdiagnostic",
    "case2_pathway_mediation",
    "biomarker_discovery",
    "disease_subtyping",
    "progression_prediction",
    "imaging_genetics",
    "differential_diagnosis",
    "connectome_behavior",
    "brain_age",
    "prognosis",
)


@dataclass
class TaskRecord:
    case_study: str
    stage: str
    status: str
    started_at: str
    elapsed_seconds: float
    command: list[str] | None = None
    output_dir: str | None = None
    log_file: str | None = None
    reason: str | None = None
    metrics: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    if include_hash:
        payload["sha256"] = sha256(path)
    return payload


def check_required_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def run_command(
    *,
    case_study: str,
    stage: str,
    command: list[str],
    output_dir: Path,
    log_dir: Path,
) -> TaskRecord:
    started = utc_now()
    start = time.perf_counter()
    log_path = log_dir / f"{case_study}_{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout or "", encoding="utf-8")
    elapsed = time.perf_counter() - start
    metrics_path = output_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else None
    return TaskRecord(
        case_study=case_study,
        stage=stage,
        status="completed" if result.returncode == 0 else "failed",
        started_at=started,
        elapsed_seconds=round(elapsed, 3),
        command=command,
        output_dir=str(output_dir),
        log_file=str(log_path),
        reason=None if result.returncode == 0 else f"exit code {result.returncode}",
        metrics=metrics,
    )


def skipped(case_study: str, stage: str, reason: str) -> TaskRecord:
    return TaskRecord(
        case_study=case_study,
        stage=stage,
        status="skipped",
        started_at=utc_now(),
        elapsed_seconds=0.0,
        reason=reason,
    )


def failed(case_study: str, stage: str, reason: str, elapsed: float = 0.0) -> TaskRecord:
    return TaskRecord(
        case_study=case_study,
        stage=stage,
        status="failed",
        started_at=utc_now(),
        elapsed_seconds=round(elapsed, 3),
        reason=reason,
    )


def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "torch",
    ):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - environment dependent
            packages[name] = f"unavailable: {exc}"
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        cuda = {"available": False, "error": str(exc)}
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda": cuda,
        "repository": str(ROOT),
    }


def schaefer_networks(labels_path: Path, n_rois: int) -> tuple[list[str], list[str]]:
    labels = pd.read_csv(labels_path)
    names = [
        str(name)
        for name in labels["name"]
        if str(name).strip().casefold() != "background"
    ]
    if len(names) < n_rois:
        raise ValueError(
            f"Atlas label table has {len(names)} regions, expected at least {n_rois}"
        )
    names = names[:n_rois]
    networks = []
    for name in names:
        match = re.search(r"7Networks_[LR]H_([^_]+)_", name)
        networks.append(match.group(1) if match else "Unknown")
    return names, networks


def fc_summary(
    matrix: np.ndarray,
    networks: list[str],
) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"FC matrix must be square, got {matrix.shape}")
    if matrix.shape[0] != len(networks):
        raise ValueError(
            f"FC matrix has {matrix.shape[0]} ROIs but {len(networks)} labels"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("FC matrix contains NaN or Inf")
    upper = matrix[np.triu_indices(matrix.shape[0], 1)]
    payload = {
        "fc_global_mean": float(np.mean(upper)),
        "fc_global_mean_abs": float(np.mean(np.abs(upper))),
        "fc_global_positive_mean": float(np.mean(upper[upper > 0]))
        if np.any(upper > 0)
        else 0.0,
        "fc_global_negative_mean": float(np.mean(upper[upper < 0]))
        if np.any(upper < 0)
        else 0.0,
    }
    ordered = list(dict.fromkeys(networks))
    network_array = np.asarray(networks)
    for left_index, left in enumerate(ordered):
        left_roi = np.flatnonzero(network_array == left)
        for right in ordered[left_index:]:
            right_roi = np.flatnonzero(network_array == right)
            if left == right:
                block = matrix[np.ix_(left_roi, right_roi)]
                values = block[np.triu_indices(len(left_roi), 1)]
            else:
                values = matrix[np.ix_(left_roi, right_roi)].ravel()
            key = f"fc_{left}_{right}"
            payload[key] = float(np.mean(values)) if len(values) else 0.0
    return payload


def load_fc_dataset(
    fc_dir: Path,
    suffix: str,
    labels_path: Path,
    *,
    export_edges: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    files = sorted(fc_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No FC matrices in {fc_dir}")
    first = np.load(files[0], mmap_mode="r")
    _, networks = schaefer_networks(labels_path, int(first.shape[0]))
    rows: list[dict[str, Any]] = []
    edges: list[np.ndarray] = []
    skipped_files: list[str] = []
    triu = np.triu_indices(int(first.shape[0]), 1)
    for path in files:
        try:
            matrix = np.load(path)
            subject_id = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
            rows.append({"subject_id": subject_id, **fc_summary(matrix, networks)})
            if export_edges:
                edges.append(np.asarray(matrix[triu], dtype=np.float32))
        except Exception as exc:
            skipped_files.append(f"{path.name}: {exc}")
    frame = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    edge_matrix = np.stack(edges) if export_edges and edges else None
    return frame, edge_matrix, {
        "fc_dir": str(fc_dir),
        "files_seen": len(files),
        "subjects_loaded": len(frame),
        "skipped_files": skipped_files,
        "n_rois": int(first.shape[0]),
        "n_network_features": int(frame.shape[1] - 1),
        "n_edges": int(edge_matrix.shape[1]) if edge_matrix is not None else None,
    }


def fill_numeric_medians(frame: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce")
            median = values.median()
            frame[column] = values.fillna(0.0 if pd.isna(median) else median)
    return frame


def prepare_tcp(public_run: Path) -> dict[str, Any]:
    input_dir = public_run / "inputs" / "tcp"
    input_dir.mkdir(parents=True, exist_ok=True)
    transdiag = NAS_PUBLIC / "transdiag_preprocessed"
    fc_dir = transdiag / "fc" / "schaefer_100_7net" / "correlation"
    diagnosis_path = transdiag / "metadata" / "diagnosis.csv"
    clinical_path = transdiag / "metadata" / "clinical_variables.csv"
    missing = check_required_paths([fc_dir, diagnosis_path, clinical_path, SCHAEFER_LABELS])
    if missing:
        raise FileNotFoundError(f"Missing TCP inputs: {missing}")

    fc, _, qc = load_fc_dataset(
        fc_dir,
        "_schaefer_100_7net_correlation.npy",
        SCHAEFER_LABELS,
        export_edges=False,
    )
    diagnosis = pd.read_csv(diagnosis_path, low_memory=False)
    merged = fc.merge(
        diagnosis,
        left_on="subject_id",
        right_on="subjectkey",
        how="inner",
        validate="one_to_one",
    )
    feature_columns = [column for column in fc.columns if column != "subject_id"]

    biomarker = merged[["subject_id", "is_patient", *feature_columns]].copy()
    biomarker["is_patient"] = pd.to_numeric(
        biomarker["is_patient"], errors="coerce"
    )
    biomarker = biomarker.dropna(subset=["is_patient"]).reset_index(drop=True)
    biomarker["is_patient"] = biomarker["is_patient"].astype(int)
    biomarker_path = input_dir / "tcp_biomarker_discovery.csv"
    biomarker.to_csv(biomarker_path, index=False)

    effects = []
    patient = biomarker["is_patient"] == 1
    for feature in feature_columns:
        case_values = biomarker.loc[patient, feature].to_numpy(dtype=float)
        control_values = biomarker.loc[~patient, feature].to_numpy(dtype=float)
        pooled = np.sqrt(
            (
                (len(case_values) - 1) * np.var(case_values, ddof=1)
                + (len(control_values) - 1) * np.var(control_values, ddof=1)
            )
            / max(1, len(case_values) + len(control_values) - 2)
        )
        effect = (
            float((np.mean(case_values) - np.mean(control_values)) / pooled)
            if pooled > 0
            else 0.0
        )
        test = stats.ttest_ind(case_values, control_values, equal_var=False)
        effects.append(
            {
                "feature": feature,
                "n_patient": len(case_values),
                "n_control": len(control_values),
                "cohen_d": effect,
                "p_value": float(test.pvalue),
            }
        )
    effect_frame = pd.DataFrame(effects)
    effect_frame["q_value"] = multipletests(
        effect_frame["p_value"].to_numpy(), method="fdr_bh"
    )[1]
    effect_frame = effect_frame.sort_values(
        ["q_value", "p_value", "cohen_d"], ascending=[True, True, False]
    )
    effect_path = public_run / "biomarker_discovery" / "univariate_effects.csv"
    effect_path.parent.mkdir(parents=True, exist_ok=True)
    effect_frame.to_csv(effect_path, index=False)

    subtyping = merged.loc[
        pd.to_numeric(merged["is_patient"], errors="coerce") == 1,
        ["subject_id", *feature_columns],
    ].reset_index(drop=True)
    subtyping_path = input_dir / "tcp_disease_subtyping.csv"
    subtyping.to_csv(subtyping_path, index=False)

    broad = merged["diagnosis_broad_any"].fillna("").astype(str)
    psychosis = broad.str.contains("psychosis_SZ_SZA", regex=False)
    depression = broad.str.contains("MDD_depression", regex=False)
    differential = merged.loc[
        psychosis ^ depression, ["subject_id", *feature_columns]
    ].copy()
    differential["psychosis_vs_mdd"] = psychosis.loc[psychosis ^ depression].astype(int).to_numpy()
    differential = differential[
        ["subject_id", "psychosis_vs_mdd", *feature_columns]
    ].reset_index(drop=True)
    differential_path = input_dir / "tcp_differential_diagnosis.csv"
    differential.to_csv(differential_path, index=False)

    clinical = pd.read_csv(clinical_path, low_memory=False)
    clinical_path_out = input_dir / "tcp_clinical_for_subtype_validation.csv"
    clinical.to_csv(clinical_path_out, index=False)
    return {
        **qc,
        "diagnosis_overlap": len(merged),
        "biomarker_samples": len(biomarker),
        "subtyping_samples": len(subtyping),
        "differential_samples": len(differential),
        "differential_class_counts": differential["psychosis_vs_mdd"].value_counts().to_dict(),
        "biomarker_input": str(biomarker_path),
        "subtyping_input": str(subtyping_path),
        "differential_input": str(differential_path),
        "clinical_input": str(clinical_path_out),
        "biomarker_effects": str(effect_path),
    }


def prepare_hcp(public_run: Path) -> dict[str, Any]:
    input_dir = public_run / "inputs" / "hcpya"
    input_dir.mkdir(parents=True, exist_ok=True)
    hcp_root = NAS_PUBLIC / "hcpya_preprocessed"
    fc_dir = hcp_root / "fc" / "schaefer_100_7net" / "correlation"
    phenotype_path = NAS_PUBLIC / "csv" / "hcp.csv"
    age_path = NAS_PUBLIC / "csv" / "HCP_1200_precise_age.csv"
    missing = check_required_paths([fc_dir, phenotype_path, age_path, SCHAEFER_LABELS])
    if missing:
        raise FileNotFoundError(f"Missing HCP-YA inputs: {missing}")

    fc, edges, qc = load_fc_dataset(
        fc_dir,
        "_schaefer_100_7net_correlation.npy",
        SCHAEFER_LABELS,
        export_edges=True,
    )
    assert edges is not None
    phenotype = pd.read_csv(phenotype_path, low_memory=False)
    phenotype["subject_id"] = phenotype["Subject"].astype(str)
    behavior = fc.merge(
        phenotype[["subject_id", "CogTotalComp_Unadj"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )
    behavior["CogTotalComp_Unadj"] = pd.to_numeric(
        behavior["CogTotalComp_Unadj"], errors="coerce"
    )
    behavior = behavior.dropna(subset=["CogTotalComp_Unadj"]).reset_index(drop=True)
    behavior_labels = input_dir / "hcpya_connectome_behavior_labels.csv"
    behavior[["subject_id", "CogTotalComp_Unadj"]].to_csv(
        behavior_labels, index=False
    )
    id_to_index = {sid: index for index, sid in enumerate(fc["subject_id"].astype(str))}
    behavior_edges = np.stack(
        [edges[id_to_index[sid]] for sid in behavior["subject_id"].astype(str)]
    )
    behavior_npz = input_dir / "hcpya_schaefer100_connectomes.npz"
    np.savez_compressed(
        behavior_npz,
        X=behavior_edges,
        subject_id=behavior["subject_id"].astype(str).to_numpy(),
    )

    age = pd.read_csv(age_path, low_memory=False)
    age["subject_id"] = age["subject"].astype(str)
    age["age"] = pd.to_numeric(age["age"], errors="coerce")
    brain_age = fc.merge(
        age[["subject_id", "age"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["age"])
    brain_age = brain_age[["subject_id", "age", *[c for c in fc if c != "subject_id"]]]
    brain_age_path = input_dir / "hcpya_brain_age.csv"
    brain_age.to_csv(brain_age_path, index=False)
    return {
        **qc,
        "behavior_samples": len(behavior),
        "behavior_target": "CogTotalComp_Unadj",
        "behavior_connectomes": str(behavior_npz),
        "behavior_labels": str(behavior_labels),
        "brain_age_samples": len(brain_age),
        "brain_age_range": [float(brain_age["age"].min()), float(brain_age["age"].max())],
        "brain_age_input": str(brain_age_path),
    }


def _normalized_diagnosis(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if "dement" in text or text in {"ad", "alzheimer's disease"}:
        return "Dementia"
    if "mci" in text:
        return "MCI"
    if text in {"cn", "normal", "cognitively normal"}:
        return "CN"
    return str(value or "")


def prepare_adni(controlled_run: Path) -> dict[str, Any]:
    input_dir = controlled_run / "inputs" / "adni"
    input_dir.mkdir(parents=True, exist_ok=True)
    table_root = ADNI_CASE2_ROOT / "experiment_tables" / "case2_adni_multimodal_v1"
    genetics_path = table_root / "case2_adni_subject_genetics_covariates.parquet"
    visits_path = table_root / "case2_adni_clinical_visits.parquet"
    missing = check_required_paths([genetics_path, visits_path])
    if missing:
        raise FileNotFoundError(f"Missing ADNI inputs: {missing}")

    genetics = pd.read_parquet(genetics_path)
    visits = pd.read_parquet(visits_path)
    visits["clinical_date"] = pd.to_datetime(visits["clinical_date"], errors="coerce")
    visits = visits.dropna(subset=["subject_id", "clinical_date"]).sort_values(
        ["subject_id", "clinical_date"]
    )
    # Keep one real baseline row per subject. GroupBy.first() would select the
    # first non-null value independently for each column and could therefore
    # leak measurements from later visits into a synthetic baseline row.
    baseline = visits.drop_duplicates("subject_id", keep="first").copy()
    baseline = baseline[
        [
            "subject_id",
            "Hippocampus",
            "Entorhinal",
            "WholeBrain",
            "Ventricles",
            "MidTemp",
            "Fusiform",
            "ICV",
            "MMSE",
            "ADAS13",
            "CDRSB",
        ]
    ]
    cohort = genetics.merge(baseline, on="subject_id", how="left", validate="one_to_one")
    cohort["baseline_dx"] = cohort["DX_bl"].map(_normalized_diagnosis)
    for marker in (
        "Hippocampus",
        "Entorhinal",
        "WholeBrain",
        "Ventricles",
        "MidTemp",
        "Fusiform",
    ):
        cohort[f"{marker}_icv"] = pd.to_numeric(
            cohort[marker], errors="coerce"
        ) / pd.to_numeric(cohort["ICV"], errors="coerce")

    survival_rows = []
    for subject_id, subject_visits in visits.groupby("subject_id"):
        subject_genetics = genetics.loc[genetics["subject_id"] == subject_id]
        if subject_genetics.empty:
            continue
        if _normalized_diagnosis(subject_genetics.iloc[0]["DX_bl"]) != "MCI":
            continue
        ordered = subject_visits.sort_values("clinical_date")
        baseline_date = ordered.iloc[0]["clinical_date"]
        last_date = ordered.iloc[-1]["clinical_date"]
        diagnoses = ordered["DX"].map(_normalized_diagnosis)
        event_dates = ordered.loc[
            (diagnoses == "Dementia") & (ordered["clinical_date"] > baseline_date),
            "clinical_date",
        ]
        event = int(not event_dates.empty)
        endpoint = event_dates.iloc[0] if event else last_date
        duration_years = float((endpoint - baseline_date).days / 365.25)
        followup_years = float((last_date - baseline_date).days / 365.25)
        if duration_years <= 0:
            continue
        survival_rows.append(
            {
                "subject_id": str(subject_id),
                "duration_years": duration_years,
                "event": event,
                "followup_years": followup_years,
                "progression_3y": int(event and duration_years <= 3.0),
                "eligible_3y": bool((event and duration_years <= 3.0) or followup_years >= 3.0),
            }
        )
    survival = pd.DataFrame(survival_rows)
    feature_columns = [
        "AGE",
        "sex_binary",
        "PTEDUCAT",
        "APOE4",
        "Hippocampus_icv",
        "Entorhinal_icv",
        "WholeBrain_icv",
        "Ventricles_icv",
        "MidTemp_icv",
        "Fusiform_icv",
    ]
    survival = survival.merge(
        cohort[["subject_id", *feature_columns]],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    survival = fill_numeric_medians(
        survival,
        excluded={"subject_id", "duration_years", "event", "progression_3y", "eligible_3y"},
    )
    survival_path = input_dir / "adni_mci_prognosis_survival.csv"
    survival[
        ["subject_id", "duration_years", "event", *feature_columns]
    ].to_csv(survival_path, index=False)
    progression = survival.loc[
        survival["eligible_3y"],
        ["subject_id", "progression_3y", *feature_columns],
    ].reset_index(drop=True)
    progression_path = input_dir / "adni_mci_progression_3y.csv"
    progression.to_csv(progression_path, index=False)

    pathway_columns = [
        column for column in genetics if str(column).startswith("pathway_prs__")
    ]
    imaging_genetics = cohort[
        [
            "subject_id",
            *pathway_columns,
            "Hippocampus_icv",
            "AGE",
            "sex_binary",
            "PTEDUCAT",
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PC5",
        ]
    ].copy()
    numeric_columns = [column for column in imaging_genetics if column != "subject_id"]
    for column in numeric_columns:
        imaging_genetics[column] = pd.to_numeric(
            imaging_genetics[column], errors="coerce"
        )
    imaging_genetics = imaging_genetics.dropna(subset=["Hippocampus_icv"])
    imaging_genetics = fill_numeric_medians(
        imaging_genetics, excluded={"subject_id", "Hippocampus_icv"}
    )
    imaging_genetics_path = input_dir / "adni_pathway_prs_hippocampus.npz"
    covariate_columns = ["AGE", "sex_binary", "PTEDUCAT", "PC1", "PC2", "PC3", "PC4", "PC5"]
    np.savez_compressed(
        imaging_genetics_path,
        genotype=imaging_genetics[pathway_columns].to_numpy(dtype=np.float32),
        phenotype=imaging_genetics["Hippocampus_icv"].to_numpy(dtype=np.float32),
        variant_id=np.asarray(pathway_columns, dtype=object),
        covariates=imaging_genetics[covariate_columns].to_numpy(dtype=np.float32),
        subject_id=imaging_genetics["subject_id"].astype(str).to_numpy(),
    )
    return {
        "genetic_subjects": len(genetics),
        "clinical_visits": len(visits),
        "mci_survival_samples": len(survival),
        "mci_survival_events": int(survival["event"].sum()),
        "progression_3y_samples": len(progression),
        "progression_3y_events": int(progression["progression_3y"].sum()),
        "imaging_genetics_samples": len(imaging_genetics),
        "pathway_prs_features": len(pathway_columns),
        "survival_input": str(survival_path),
        "progression_input": str(progression_path),
        "imaging_genetics_input": str(imaging_genetics_path),
    }


def validate_subtypes(public_run: Path, tcp_manifest: dict[str, Any]) -> dict[str, Any]:
    prediction_path = public_run / "disease_subtyping" / "predictions.csv"
    if not prediction_path.is_file():
        return {"status": "missing predictions"}
    predictions = pd.read_csv(prediction_path)
    clinical = pd.read_csv(tcp_manifest["clinical_input"], low_memory=False)
    merged = predictions.merge(
        clinical,
        left_on="subject_id",
        right_on="subjectkey",
        how="inner",
    )
    rows = []
    for column in clinical.columns:
        if column == "subjectkey":
            continue
        values = pd.to_numeric(merged[column], errors="coerce")
        groups = [
            values[merged["subtype"] == subtype].dropna().to_numpy()
            for subtype in sorted(merged["subtype"].unique())
        ]
        groups = [group for group in groups if len(group) >= 5]
        if len(groups) < 2:
            continue
        test = stats.kruskal(*groups)
        rows.append(
            {
                "clinical_variable": column,
                "n": int(values.notna().sum()),
                "kruskal_h": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return {"status": "no testable clinical variables"}
    result["q_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result = result.sort_values(["q_value", "p_value"])
    output = public_run / "disease_subtyping" / "clinical_separation.csv"
    result.to_csv(output, index=False)
    return {
        "status": "completed",
        "variables_tested": len(result),
        "fdr_significant_variables": int((result["q_value"] < 0.05).sum()),
        "best_variable": str(result.iloc[0]["clinical_variable"]),
        "best_q_value": float(result.iloc[0]["q_value"]),
        "output": str(output),
    }


def generate_hypotheses_once(
    graph_path: Path,
    output_root: Path,
    records: list[TaskRecord],
    case_studies: Iterable[str] | None = None,
) -> dict[str, Any]:
    from neurooracle.src.case_studies import (
        GENERATOR_CASE1_CANDIDATE,
        GENERATOR_CHAIN,
        GENERATOR_TASK,
        case_study_by_name,
    )
    from neurooracle.src.hypothesis_cli import cmd_batch, load_graph
    from neurooracle.src.hypothesis_engine import HypothesisEngine

    generation_root = output_root / "hypothesis_generation"
    generation_root.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    kg = load_graph(graph_path)
    engine = HypothesisEngine(kg)
    load_seconds = time.perf_counter() - load_started
    counts: dict[str, Any] = {"graph_load_seconds": round(load_seconds, 3)}

    selected_cases = list(case_studies or DIRECT_CASE_STUDIES)
    unknown_cases = sorted(set(selected_cases) - set(DIRECT_CASE_STUDIES))
    if unknown_cases:
        raise ValueError(f"Unknown direct Case Studies: {', '.join(unknown_cases)}")
    ordered_cases = [case for case in selected_cases if case != "case2_pathway_mediation"]
    if "case2_pathway_mediation" in selected_cases:
        ordered_cases.append("case2_pathway_mediation")
    for case_name in ordered_cases:
        start = time.perf_counter()
        started_at = utc_now()
        case = case_study_by_name(case_name)
        case_dir = generation_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        output_path = case_dir / "hypotheses_raw.json"
        console_output = io.StringIO()
        try:
            for hook in case.pre_hooks:
                hook(engine, case)
            batch = case.stage_params.batch
            if case.generator == GENERATOR_CASE1_CANDIDATE:
                extras = case.extras or {}
                hypotheses = engine.generate_case1_hypotheses(
                    methods=tuple(extras.get("generation_methods", ())),
                    disease_names=tuple(extras.get("disease_include_names", ())),
                    atlas_rois=tuple(extras.get("atlas_rois", ())),
                    atlas_label_names=tuple(extras.get("atlas_label_names", ())),
                    atlas_label_sources=dict(extras.get("atlas_label_sources", {})),
                    feature_space=tuple(extras.get("feature_space", ())),
                    max_per_method=extras.get(
                        "max_hypotheses_per_method", batch.target_per_task
                    ),
                    random_seed=int(extras.get("random_seed", 0)) or None,
                )
                for hypothesis in hypotheses:
                    metadata = hypothesis.metadata or {}
                    metadata["case_study_id"] = case.name
                    hypothesis.metadata = metadata
                engine.save_hypotheses(hypotheses, output_path)
            else:
                task_filter = case.task.name if case.generator == GENERATOR_TASK else ""
                chain_filter = case.chain.name if case.generator == GENERATOR_CHAIN else ""
                with contextlib.redirect_stdout(console_output):
                    cmd_batch(
                        engine,
                        str(output_path),
                        max_hops=batch.max_hops,
                        min_hops=batch.min_hops,
                        metapath_min_domains=batch.metapath_min_domains,
                        max_paths=batch.max_paths,
                        max_seeds=batch.max_seeds,
                        target_per_task=batch.target_per_task,
                        max_retries=batch.max_retries,
                        retry_scale=batch.retry_scale,
                        prefer_longer_paths=batch.prefer_longer_paths,
                        task_filter=task_filter,
                        chain_filter=chain_filter,
                        as_json=False,
                    )
                hypotheses = engine.load_hypotheses(output_path)
                for hypothesis in hypotheses:
                    metadata = hypothesis.metadata or {}
                    metadata["case_study_id"] = case.name
                    hypothesis.metadata = metadata
                engine.save_hypotheses(hypotheses, output_path)
            for hook in case.post_hooks:
                hook(engine, case, output_path)
            hypothesis_count = len(engine.load_hypotheses(output_path))
            counts[case_name] = hypothesis_count
            records.append(
                TaskRecord(
                    case_study=case_name,
                    stage="hypothesis_generation",
                    status="completed",
                    started_at=started_at,
                    elapsed_seconds=round(time.perf_counter() - start, 3),
                    output_dir=str(case_dir),
                    metrics={"hypotheses_generated": hypothesis_count},
                )
            )
        except Exception as exc:
            error_path = case_dir / "error.log"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            records.append(
                TaskRecord(
                    case_study=case_name,
                    stage="hypothesis_generation",
                    status="failed",
                    started_at=started_at,
                    elapsed_seconds=round(time.perf_counter() - start, 3),
                    output_dir=str(case_dir),
                    log_file=str(error_path),
                    reason=str(exc),
                )
            )
        finally:
            if console_output.tell():
                (case_dir / "generation.log").write_text(
                    console_output.getvalue(), encoding="utf-8"
                )
    return counts


def collect_metrics(public_run: Path, controlled_run: Path) -> dict[str, Any]:
    locations = {
        "biomarker_discovery": public_run / "biomarker_discovery" / "metrics.json",
        "disease_subtyping": public_run / "disease_subtyping" / "metrics.json",
        "differential_diagnosis": public_run / "differential_diagnosis" / "metrics.json",
        "connectome_behavior": public_run / "connectome_behavior" / "metrics.json",
        "brain_age": public_run / "brain_age" / "metrics.json",
        "progression_prediction": controlled_run / "progression_prediction" / "metrics.json",
        "imaging_genetics": controlled_run / "imaging_genetics" / "metrics.json",
        "prognosis": controlled_run / "prognosis" / "metrics.json",
    }
    result = {}
    for case_study, path in locations.items():
        result[case_study] = read_json(path) if path.is_file() else None
    case2_manifest = controlled_run / "case2_pathway_mediation" / "manifest.json"
    if case2_manifest.is_file():
        result["case2_pathway_mediation"] = read_json(case2_manifest)
    case1_manifests = sorted(
        (public_run / "case1_transdiagnostic").rglob("*manifest.json")
    )
    if case1_manifests:
        result["case1_transdiagnostic"] = read_json(case1_manifests[-1])
    return result


def output_hashes(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "output_hashes.json":
                continue
            rows.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def write_experiment_report(
    public_run: Path,
    controlled_run: Path,
    records: list[TaskRecord],
    data_manifest: dict[str, Any],
    metrics: dict[str, Any],
    subtype_validation: dict[str, Any],
) -> None:
    rows = []
    for case_study in DIRECT_CASE_STUDIES:
        case_records = [record for record in records if record.case_study == case_study]
        status = "completed"
        if any(record.status == "failed" for record in case_records):
            status = "failed"
        elif any(record.status == "skipped" for record in case_records):
            status = "skipped"
        rows.append(
            {
                "case_study": case_study,
                "status": status,
                "stages_completed": ", ".join(
                    record.stage for record in case_records if record.status == "completed"
                ),
                "failed_or_skipped_reason": "; ".join(
                    record.reason or ""
                    for record in case_records
                    if record.status != "completed"
                ),
                "aggregate_metrics": json.dumps(
                    metrics.get(case_study), ensure_ascii=False, default=str
                ),
            }
        )
    pd.DataFrame(rows).to_csv(public_run / "case_study_summary.csv", index=False)
    report_lines = [
        "# Direct Case Study pilot experiment",
        "",
        f"- Created: {utc_now()}",
        f"- Public run: `{public_run}`",
        f"- Controlled ADNI run: `{controlled_run}`",
        f"- Python: `{sys.executable}`",
        "",
        "## Scope",
        "",
        "This run is an execution-readiness pilot using only currently available data. ",
        "It does not replace the full preregistered baseline comparison for any Case Study.",
        "",
        "## Results",
        "",
        "| Case Study | Status | Completed stages |",
        "|---|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['case_study']} | {row['status']} | {row['stages_completed']} |"
        )
    report_lines.extend(
        [
            "",
            "## Data manifest",
            "",
            "```json",
            json.dumps(data_manifest, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## Aggregate metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## Subtype clinical separation",
            "",
            "```json",
            json.dumps(subtype_validation, indent=2, ensure_ascii=False, default=str),
            "```",
        ]
    )
    (public_run / "EXPERIMENT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_RUN_ROOT)
    parser.add_argument("--controlled-root", type=Path, default=DEFAULT_CONTROLLED_RUN_ROOT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-specialized", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    public_run = args.public_root / args.run_id
    controlled_run = args.controlled_root / args.run_id
    public_run.mkdir(parents=True, exist_ok=False)
    controlled_run.mkdir(parents=True, exist_ok=False)
    log_dir = public_run / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    records: list[TaskRecord] = []

    write_json(public_run / "environment_manifest.json", environment_manifest())
    graph_fingerprint = file_fingerprint(args.graph, include_hash=False)
    preflight = {
        "created_at": utc_now(),
        "public_nas": str(NAS_PUBLIC),
        "public_nas_accessible": NAS_PUBLIC.exists(),
        "controlled_adni_root": str(ADNI_CASE2_ROOT),
        "controlled_adni_accessible": ADNI_CASE2_ROOT.exists(),
        "graph": graph_fingerprint,
    }
    write_json(public_run / "preflight.json", preflight)

    data_manifest: dict[str, Any] = {}
    for name, function, target in (
        ("tcp", prepare_tcp, public_run),
        ("hcpya", prepare_hcp, public_run),
        ("adni", prepare_adni, controlled_run),
    ):
        start = time.perf_counter()
        try:
            data_manifest[name] = function(target)
            records.append(
                TaskRecord(
                    case_study="shared",
                    stage=f"prepare_{name}",
                    status="completed",
                    started_at=utc_now(),
                    elapsed_seconds=round(time.perf_counter() - start, 3),
                )
            )
        except Exception as exc:
            data_manifest[name] = {"error": str(exc)}
            error_path = log_dir / f"prepare_{name}.log"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            records.append(
                TaskRecord(
                    case_study="shared",
                    stage=f"prepare_{name}",
                    status="failed",
                    started_at=utc_now(),
                    elapsed_seconds=round(time.perf_counter() - start, 3),
                    log_file=str(error_path),
                    reason=str(exc),
                )
            )
    write_json(public_run / "data_manifest.json", data_manifest)

    if args.prepare_only:
        write_json(
            public_run / "experiment_task_manifest.json",
            {"tasks": [asdict(record) for record in records]},
        )
        print(json.dumps({"public_run": str(public_run), "data": data_manifest}, indent=2))
        return 0

    py = sys.executable
    if "tcp" not in data_manifest or "error" in data_manifest["tcp"]:
        for case_study in (
            "case1_transdiagnostic",
            "biomarker_discovery",
            "disease_subtyping",
            "differential_diagnosis",
        ):
            records.append(skipped(case_study, "experiment", "TCP preparation failed"))
    else:
        tcp = data_manifest["tcp"]
        if not args.skip_specialized:
            case1_output = public_run / "case1_transdiagnostic"
            command = [
                py,
                str(ROOT / "core" / "scripts" / "case1_exhaustive_full.py"),
                "--transdiag-root",
                str(NAS_PUBLIC / "transdiag_preprocessed"),
                "--diagnosis",
                str(NAS_PUBLIC / "transdiag_preprocessed" / "metadata" / "diagnosis.csv"),
                "--out-root",
                str(case1_output),
                "--atlas-root",
                str(SCHAEFER_LABELS.parents[1]),
                "--run-name",
                "schaefer100_pilot",
                "--min-cases",
                "8",
                "--n-boot",
                "0",
                "--atlas-pattern",
                "schaefer_100_7net",
            ]
            records.append(
                run_command(
                    case_study="case1_transdiagnostic",
                    stage="specialized_experiment",
                    command=command,
                    output_dir=case1_output / "schaefer100_pilot",
                    log_dir=log_dir,
                )
            )

        for case_study, input_key, target, model, task in (
            (
                "biomarker_discovery",
                "biomarker_input",
                "is_patient",
                "logistic",
                "classification",
            ),
            (
                "differential_diagnosis",
                "differential_input",
                "psychosis_vs_mdd",
                "logistic",
                "classification",
            ),
        ):
            output = public_run / case_study
            command = [
                py,
                "-m",
                "models.statistical_ml.train",
                "--features",
                tcp[input_key],
                "--target",
                target,
                "--model",
                model,
                "--task",
                task,
                "--folds",
                "5",
                "--seed",
                "20260805",
                "--output-dir",
                str(output),
            ]
            records.append(
                run_command(
                    case_study=case_study,
                    stage="neuro_runtime_model",
                    command=command,
                    output_dir=output,
                    log_dir=log_dir,
                )
            )

        subtyping_output = public_run / "disease_subtyping"
        subtyping_command = [
            py,
            "-m",
            "models.subtyping.train",
            "--features",
            tcp["subtyping_input"],
            "--model",
            "consensus",
            "--n-clusters",
            "3",
            "--seed",
            "20260805",
            "--output-dir",
            str(subtyping_output),
        ]
        records.append(
            run_command(
                case_study="disease_subtyping",
                stage="neuro_runtime_model",
                command=subtyping_command,
                output_dir=subtyping_output,
                log_dir=log_dir,
            )
        )

    if "hcpya" not in data_manifest or "error" in data_manifest["hcpya"]:
        records.append(skipped("connectome_behavior", "experiment", "HCP-YA preparation failed"))
        records.append(skipped("brain_age", "experiment", "HCP-YA preparation failed"))
    else:
        hcp = data_manifest["hcpya"]
        connectome_output = public_run / "connectome_behavior"
        connectome_command = [
            py,
            "-m",
            "models.cpm.train",
            "--connectomes",
            hcp["behavior_connectomes"],
            "--labels",
            hcp["behavior_labels"],
            "--target",
            hcp["behavior_target"],
            "--task",
            "regression",
            "--p-threshold",
            "0.01",
            "--folds",
            "5",
            "--seed",
            "20260805",
            "--output-dir",
            str(connectome_output),
        ]
        records.append(
            run_command(
                case_study="connectome_behavior",
                stage="neuro_runtime_model",
                command=connectome_command,
                output_dir=connectome_output,
                log_dir=log_dir,
            )
        )
        brain_age_output = public_run / "brain_age"
        brain_age_command = [
            py,
            "-m",
            "models.brain_age.train",
            "--features",
            hcp["brain_age_input"],
            "--age-col",
            "age",
            "--model",
            "ridge",
            "--folds",
            "5",
            "--seed",
            "20260805",
            "--output-dir",
            str(brain_age_output),
        ]
        records.append(
            run_command(
                case_study="brain_age",
                stage="neuro_runtime_model",
                command=brain_age_command,
                output_dir=brain_age_output,
                log_dir=log_dir,
            )
        )

    if "adni" not in data_manifest or "error" in data_manifest["adni"]:
        for case_study in (
            "case2_pathway_mediation",
            "progression_prediction",
            "imaging_genetics",
            "prognosis",
        ):
            records.append(skipped(case_study, "experiment", "ADNI preparation failed"))
    else:
        adni = data_manifest["adni"]
        if not args.skip_specialized:
            case2_output = controlled_run / "case2_pathway_mediation"
            pathway_exposures = [
                "pathway_prs__curated__microglia_immune__p5em02",
                "pathway_prs__curated__myelin__p5em02",
                "pathway_prs__curated__synaptic__p5em02",
            ]
            case2_command = [
                py,
                str(ROOT / "neurooracle" / "scripts" / "run_case2_adni_mediation_smoke.py"),
                "--output-root",
                str(case2_output),
                "--exposure-set",
                "pathway",
                "--outcome",
                "ADAS13",
                "--outcome",
                "MMSE",
                "--top-n",
                "50",
                "--force",
            ]
            for exposure in pathway_exposures:
                case2_command.extend(["--exposure", exposure])
            records.append(
                run_command(
                    case_study="case2_pathway_mediation",
                    stage="specialized_experiment",
                    command=case2_command,
                    output_dir=case2_output,
                    log_dir=log_dir,
                )
            )

        progression_output = controlled_run / "progression_prediction"
        progression_command = [
            py,
            "-m",
            "models.statistical_ml.train",
            "--features",
            adni["progression_input"],
            "--target",
            "progression_3y",
            "--model",
            "logistic",
            "--task",
            "classification",
            "--folds",
            "5",
            "--seed",
            "20260805",
            "--output-dir",
            str(progression_output),
        ]
        records.append(
            run_command(
                case_study="progression_prediction",
                stage="neuro_runtime_model",
                command=progression_command,
                output_dir=progression_output,
                log_dir=log_dir,
            )
        )

        genetics_output = controlled_run / "imaging_genetics"
        genetics_command = [
            py,
            "-m",
            "models.imaging_genetics.train",
            "--input",
            adni["imaging_genetics_input"],
            "--model",
            "association",
            "--output-dir",
            str(genetics_output),
        ]
        records.append(
            run_command(
                case_study="imaging_genetics",
                stage="neuro_runtime_model",
                command=genetics_command,
                output_dir=genetics_output,
                log_dir=log_dir,
            )
        )

        prognosis_output = controlled_run / "prognosis"
        prognosis_command = [
            py,
            "-m",
            "models.survival_models.train",
            "--features",
            adni["survival_input"],
            "--duration-col",
            "duration_years",
            "--event-col",
            "event",
            "--model",
            "cox",
            "--folds",
            "5",
            "--seed",
            "20260805",
            "--output-dir",
            str(prognosis_output),
        ]
        records.append(
            run_command(
                case_study="prognosis",
                stage="neuro_runtime_model",
                command=prognosis_command,
                output_dir=prognosis_output,
                log_dir=log_dir,
            )
        )

    if not args.skip_generation:
        try:
            generation_counts = generate_hypotheses_once(
                args.graph, public_run, records
            )
            write_json(public_run / "hypothesis_generation_summary.json", generation_counts)
        except Exception as exc:
            error_path = log_dir / "hypothesis_generation_global.log"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            for case_study in DIRECT_CASE_STUDIES:
                if not any(
                    record.case_study == case_study
                    and record.stage == "hypothesis_generation"
                    for record in records
                ):
                    records.append(
                        TaskRecord(
                            case_study=case_study,
                            stage="hypothesis_generation",
                            status="failed",
                            started_at=utc_now(),
                            elapsed_seconds=0.0,
                            log_file=str(error_path),
                            reason=str(exc),
                        )
                    )

    subtype_validation = (
        validate_subtypes(public_run, data_manifest["tcp"])
        if "tcp" in data_manifest and "error" not in data_manifest["tcp"]
        else {"status": "skipped"}
    )
    metrics = collect_metrics(public_run, controlled_run)
    write_json(public_run / "aggregate_metrics.json", metrics)
    write_json(
        public_run / "experiment_task_manifest.json",
        {
            "created_at": utc_now(),
            "public_run": str(public_run),
            "controlled_run": str(controlled_run),
            "tasks": [asdict(record) for record in records],
        },
    )
    write_experiment_report(
        public_run,
        controlled_run,
        records,
        data_manifest,
        metrics,
        subtype_validation,
    )

    if not args.skip_hashes:
        hashes = output_hashes([public_run, controlled_run])
        write_json(public_run / "output_hashes.json", hashes)
    completed = sum(record.status == "completed" for record in records)
    failed_count = sum(record.status == "failed" for record in records)
    skipped_count = sum(record.status == "skipped" for record in records)
    audit = {
        "created_at": utc_now(),
        "cases_requested": list(DIRECT_CASE_STUDIES),
        "task_records_completed": completed,
        "task_records_failed": failed_count,
        "task_records_skipped": skipped_count,
        "all_case_studies_have_records": all(
            any(record.case_study == case for record in records)
            for case in DIRECT_CASE_STUDIES
        ),
        "public_run": str(public_run),
        "controlled_run": str(controlled_run),
    }
    write_json(public_run / "experiment_audit_report.json", audit)
    print(json.dumps(audit, indent=2))
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
