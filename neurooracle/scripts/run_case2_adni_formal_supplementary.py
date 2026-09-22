"""Run the locked formal supplementary Case Study 2 analysis.

This entry point is fail-closed.  It verifies the outcome-blind readiness lock
and requires an explicit outcome-access acknowledgement before loading the
observed outcome-value column or fitting any association model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy
import statsmodels
from threadpoolctl import threadpool_limits

from neurooracle.scripts.case2_formal_supplementary import (
    CATEGORICAL_COMMON_COVARIATES,
    bh_adjust_fixed,
    canonical_sha256,
    complete_candidate_frame,
    file_sha256,
    fit_candidate_model,
    load_protocol,
    orient_outcome,
)
from neurooracle.scripts.prepare_case2_adni_formal_supplementary import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_READINESS_ROOT,
    HKT,
)


def _verify_pin(path: Path, pin: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = int(path.stat().st_size)
    observed_hash = file_sha256(path)
    if observed_size != int(pin["bytes"]) or observed_hash != pin["sha256"]:
        raise ValueError(f"Pinned {label} changed after the readiness freeze")


def verify_readiness_lock(lock_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the lock, all readiness artifacts, and every frozen input."""

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_schema = "neurooracle.case2_formal_supplementary_readiness_lock.v1"
    if lock.get("schema_version") != expected_schema:
        raise ValueError(
            f"Unexpected readiness lock schema: {lock.get('schema_version')}"
        )
    if lock.get("status") != "locked_ready_for_formal_execution":
        raise ValueError("Readiness lock is not executable")
    material = lock["lock_material"]
    if canonical_sha256(material) != lock.get("freeze_id"):
        raise ValueError("Readiness freeze ID does not match its lock material")
    if material.get("outcome_value_column_read") is not False:
        raise ValueError("Readiness lock does not certify outcome blindness")
    if material.get("association_model_fit") is not False:
        raise ValueError("Models were already fit during the readiness audit")

    readiness_root = lock_path.parent
    for label, pin in material["artifact_pins"].items():
        _verify_pin(
            readiness_root / pin["relative_path"],
            pin,
            label=f"readiness artifact {label}",
        )
    for label, pin in material["input_pins"].items():
        _verify_pin(Path(pin["path"]), pin, label=f"input {label}")
    protocol_pin = material["artifact_pins"]["protocol_snapshot"]
    protocol_path = readiness_root / protocol_pin["relative_path"]
    protocol = load_protocol(protocol_path)
    if protocol["protocol_id"] != lock["protocol_id"]:
        raise ValueError("Readiness lock and protocol IDs differ")
    return lock, protocol


def _unique_value_lookup(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(f"{label} contains duplicate frozen keys")
    return frame


def assemble_candidate_frame(
    candidate: Mapping[str, Any],
    row_keys: pd.DataFrame,
    markers: pd.DataFrame,
    outcomes: pd.DataFrame,
    subjects: pd.DataFrame,
    cohort_baseline_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach frozen values to one candidate's locked date and visit keys."""

    family = row_keys[
        row_keys["readiness_family_id"].eq(candidate["readiness_family_id"])
    ].copy()
    marker_values = markers[
        markers["modality"].eq(candidate["modality"])
        & markers["marker"].eq(candidate["marker"])
        & markers["qc_primary_eligible"].fillna(False).astype(bool)
    ][["visit_id", "subject_id", "modality", "marker", "imaging_date", "value"]].rename(
        columns={"value": "mediator"}
    )
    marker_values = _unique_value_lookup(
        marker_values,
        ["visit_id", "subject_id", "modality", "marker", "imaging_date"],
        label="Imaging marker table",
    )
    frame = family.merge(
        marker_values,
        on=["visit_id", "subject_id", "modality", "marker", "imaging_date"],
        how="left",
        validate="one_to_one",
    )

    outcome_values = outcomes[outcomes["outcome"].eq(candidate["outcome"])][
        ["subject_id", "outcome", "outcome_date", "value"]
    ]
    outcome_values = _unique_value_lookup(
        outcome_values,
        ["subject_id", "outcome", "outcome_date"],
        label="Outcome table",
    )
    baseline = outcome_values.rename(
        columns={
            "outcome_date": "baseline_outcome_date",
            "value": "baseline_outcome_raw",
        }
    )
    future = outcome_values.rename(
        columns={
            "outcome_date": "future_outcome_date",
            "value": "future_outcome_raw",
        }
    )
    frame = frame.merge(
        baseline,
        on=["subject_id", "outcome", "baseline_outcome_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        future,
        on=["subject_id", "outcome", "future_outcome_date"],
        how="left",
        validate="one_to_one",
    )

    subject_columns = [
        "subject_id",
        candidate["score_name"],
        "AGE",
        "sex_binary",
        "PTEDUCAT",
        *[f"PC{index}" for index in range(1, 11)],
        *CATEGORICAL_COMMON_COVARIATES,
    ]
    frame = frame.merge(
        subjects.loc[:, subject_columns],
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).merge(
        cohort_baseline_dates,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    frame["age_at_imaging"] = (
        pd.to_numeric(frame["AGE"], errors="coerce")
        + (frame["imaging_date"] - frame["cohort_baseline_date"]).dt.days / 365.25
    )
    frame["exposure"] = pd.to_numeric(frame[candidate["score_name"]], errors="coerce")
    multiplier = float(candidate["higher_is_worse_multiplier"])
    frame["baseline_outcome"] = orient_outcome(
        frame["baseline_outcome_raw"], multiplier
    )
    frame["future_outcome"] = orient_outcome(frame["future_outcome_raw"], multiplier)

    if candidate["modality"] == "smri_adnimerge":
        icv = markers[
            markers["modality"].eq("smri_adnimerge")
            & markers["marker"].eq("ICV")
            & markers["qc_primary_eligible"].fillna(False).astype(bool)
        ][["visit_id", "value"]].rename(columns={"value": "icv"})
        icv = _unique_value_lookup(icv, ["visit_id"], label="ICV marker table")
        frame = frame.merge(icv, on="visit_id", how="left", validate="many_to_one")
    return frame


def _load_frozen_inputs(
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
    readiness_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    material = lock["lock_material"]
    artifact_pins = material["artifact_pins"]
    input_pins = material["input_pins"]
    master = pd.read_csv(
        readiness_root / artifact_pins["master_candidate_registry"]["relative_path"]
    )
    executable = pd.read_csv(
        readiness_root / artifact_pins["executable_candidate_registry"]["relative_path"]
    )
    row_keys = pd.read_parquet(
        readiness_root / artifact_pins["executable_analysis_row_keys"]["relative_path"]
    )
    exposure_names = [row["score_name"] for row in protocol["pathway_exposures"]]
    subject_columns = [
        "subject_id",
        "AGE",
        "sex_binary",
        "PTEDUCAT",
        *[f"PC{index}" for index in range(1, 11)],
        *CATEGORICAL_COMMON_COVARIATES,
        *exposure_names,
    ]
    subjects = pd.read_parquet(
        Path(input_pins["subjects"]["path"]), columns=subject_columns
    )
    markers = pd.read_parquet(
        Path(input_pins["imaging_markers"]["path"]),
        columns=[
            "visit_id",
            "subject_id",
            "modality",
            "imaging_date",
            "qc_primary_eligible",
            "marker",
            "value",
        ],
    )
    outcomes = pd.read_parquet(
        Path(input_pins["outcome_dates"]["path"]),
        columns=["subject_id", "outcome", "outcome_date", "value"],
    )
    clinical = pd.read_parquet(
        Path(input_pins["clinical_dates"]["path"]),
        columns=["subject_id", "clinical_date"],
    )
    clinical["clinical_date"] = pd.to_datetime(
        clinical["clinical_date"], errors="coerce"
    )
    baseline_dates = (
        clinical.dropna(subset=["clinical_date"])
        .groupby("subject_id", as_index=False)["clinical_date"]
        .min()
        .rename(columns={"clinical_date": "cohort_baseline_date"})
    )
    return master, executable, row_keys, subjects, markers, outcomes, baseline_dates


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _candidate_checkpoint_name(candidate_id: str) -> str:
    digest = canonical_sha256({"candidate_id": candidate_id})
    return f"c_{digest[:12]}.json"


def _fit_candidate_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fit one candidate in a process with BLAS oversubscription disabled."""

    candidate = payload["candidate"]
    frame = payload["frame"]
    model_spec = payload["model_spec"]
    include_icv = bool(payload["include_icv"])
    with threadpool_limits(limits=1):
        fitted = fit_candidate_model(
            frame,
            include_icv=include_icv,
            bootstrap_replicates=int(model_spec["bootstrap_replicates"]),
            bootstrap_seed=int(model_spec["bootstrap_seed"]),
            candidate_id=str(candidate["candidate_id"]),
            confidence_level=float(model_spec["bootstrap_confidence_interval"]),
            minimum_category_n=int(model_spec["minimum_categorical_level_n"]),
        )
    fitted["path_a_design_columns"] = json.dumps(
        fitted["path_a_design_columns"], separators=(",", ":")
    )
    fitted["path_b_design_columns"] = json.dumps(
        fitted["path_b_design_columns"], separators=(",", ":")
    )
    return {**candidate, **fitted, "analysis_status": "estimated"}


def run_formal_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_outcome_access_after_lock:
        raise PermissionError(
            "Formal outcome access is fail-closed. Re-run with "
            "--confirm-outcome-access-after-lock after reviewing READINESS.lock.json."
        )
    lock, protocol = verify_readiness_lock(args.readiness_lock)
    readiness_root = args.readiness_lock.parent
    output_root = args.output_root or (
        readiness_root.parent / "formal_results" / f"run_{lock['freeze_id'][:12]}"
    )
    if output_root.exists():
        raise FileExistsError(f"Formal result output already exists: {output_root}")

    (
        master,
        executable,
        row_keys,
        subjects,
        markers,
        outcomes,
        baseline_dates,
    ) = _load_frozen_inputs(lock, protocol, readiness_root)
    model_spec = protocol["model"]
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers must be at least one")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    work_root = output_root.parent / f".{output_root.name}.work"
    checkpoint_root = work_root / "candidate_checkpoints"
    run_context = {
        "schema_version": "neurooracle.case2_formal_supplementary_work.v1",
        "protocol_id": protocol["protocol_id"],
        "readiness_freeze_id": lock["freeze_id"],
        "master_candidate_count": int(len(master)),
        "executable_candidate_count": int(len(executable)),
        "bootstrap_replicates": int(model_spec["bootstrap_replicates"]),
        "bootstrap_seed": int(model_spec["bootstrap_seed"]),
        "master_seed_count": int(model_spec["master_seed_count"]),
    }
    context_path = work_root / "RUN_CONTEXT.json"
    if work_root.exists():
        if not context_path.is_file():
            raise ValueError(
                f"Incomplete formal work directory lacks context: {work_root}"
            )
        observed_context = json.loads(context_path.read_text(encoding="utf-8"))
        if observed_context != run_context:
            raise ValueError(
                "Existing formal checkpoints belong to a different locked run"
            )
    else:
        work_root.mkdir()
        context_path.write_text(
            json.dumps(run_context, indent=2) + "\n", encoding="utf-8"
        )
    checkpoint_root.mkdir(exist_ok=True)
    checkpoint_names = {
        _candidate_checkpoint_name(str(candidate_id))
        for candidate_id in executable["candidate_id"]
    }
    if len(checkpoint_names) != len(executable):
        raise ValueError(
            "Short checkpoint hashes collide within the candidate registry"
        )
    longest_checkpoint = max(checkpoint_names, key=len)
    if len(str(checkpoint_root / longest_checkpoint)) >= 248:
        raise ValueError("Formal checkpoint path remains unsafe for Windows")

    results: list[dict[str, Any]] = []
    pending_payloads: list[dict[str, Any]] = []
    for candidate in executable.to_dict(orient="records"):
        checkpoint_path = checkpoint_root / _candidate_checkpoint_name(
            str(candidate["candidate_id"])
        )
        if checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("candidate_id") != candidate["candidate_id"]:
                raise ValueError(f"Checkpoint ID mismatch: {checkpoint_path}")
            if int(checkpoint.get("n", -1)) != int(candidate["complete_case_n"]):
                raise ValueError(f"Checkpoint N mismatch: {checkpoint_path}")
            results.append(checkpoint)
            continue
        frame = assemble_candidate_frame(
            candidate,
            row_keys,
            markers,
            outcomes,
            subjects,
            baseline_dates,
        )
        include_icv = candidate["modality"] == "smri_adnimerge"
        observed_n = len(complete_candidate_frame(frame, include_icv=include_icv))
        if observed_n != int(candidate["complete_case_n"]):
            raise ValueError(
                f"Locked N changed for {candidate['candidate_id']}: "
                f"expected {candidate['complete_case_n']}, observed {observed_n}"
            )
        pending_payloads.append(
            {
                "candidate": candidate,
                "frame": frame,
                "model_spec": model_spec,
                "include_icv": include_icv,
            }
        )

    completed = len(results)
    total = int(len(executable))
    if completed:
        print(f"formal_progress={completed}/{total} resumed", flush=True)
    if pending_payloads:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fit_candidate_task, payload): str(
                    payload["candidate"]["candidate_id"]
                )
                for payload in pending_payloads
            }
            for future in as_completed(futures):
                result = future.result()
                checkpoint_path = checkpoint_root / _candidate_checkpoint_name(
                    str(result["candidate_id"])
                )
                temporary_path = checkpoint_path.with_suffix(".tmp")
                temporary_path.write_text(
                    json.dumps(result, indent=2, default=_json_default) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(checkpoint_path)
                results.append(result)
                completed += 1
                print(f"formal_progress={completed}/{total}", flush=True)

    if len(results) != total:
        raise ValueError(f"Expected {total} fitted candidates, found {len(results)}")

    estimated = pd.DataFrame(results)
    estimated_payload = estimated.drop(
        columns=[
            column
            for column in master.columns
            if column != "candidate_id" and column in estimated
        ]
    )
    all_results = master.merge(
        estimated_payload,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    all_results["analysis_status"] = all_results["analysis_status"].fillna(
        "excluded_by_locked_readiness_gate"
    )
    all_results["indirect_bootstrap_q_global_168"] = bh_adjust_fixed(
        all_results["indirect_bootstrap_p"], family_size=168
    )
    all_results.loc[
        all_results["analysis_status"].ne("estimated"),
        "indirect_bootstrap_q_global_168",
    ] = 1.0
    all_results["indirect_bootstrap_q_family_8"] = np.nan
    for _, index in all_results.groupby(
        "supplemental_fdr_family_id", sort=False
    ).groups.items():
        adjusted = bh_adjust_fixed(
            all_results.loc[index, "indirect_bootstrap_p"], family_size=8
        )
        all_results.loc[index, "indirect_bootstrap_q_family_8"] = adjusted
    all_results.loc[
        all_results["analysis_status"].ne("estimated"),
        "indirect_bootstrap_q_family_8",
    ] = 1.0

    results_path = work_root / "FORMAL_RESULTS.csv"
    all_results.to_csv(results_path, index=False)
    checkpoint_manifest = {
        path.name: {
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted(checkpoint_root.glob("c_*.json"))
    }
    checkpoint_manifest_path = work_root / "CHECKPOINT_MANIFEST.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "schema_version": "neurooracle.case2_formal_supplementary_run.v1",
        "protocol_id": protocol["protocol_id"],
        "readiness_freeze_id": lock["freeze_id"],
        "generated_at_hkt": datetime.now(HKT).isoformat(),
        "interpretation": protocol["model"]["interpretation"],
        "master_candidate_count": int(len(master)),
        "estimated_candidate_count": int(len(estimated)),
        "excluded_candidate_count": int(len(master) - len(estimated)),
        "bootstrap_replicates_per_candidate": int(model_spec["bootstrap_replicates"]),
        "master_seed_count": int(model_spec["master_seed_count"]),
        "master_bootstrap_seed": int(model_spec["bootstrap_seed"]),
        "parallel_workers": workers,
        "checkpoint_count": int(len(checkpoint_manifest)),
        "multiplicity": protocol["multiplicity"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    metadata_path = work_root / "RUN_METADATA.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    run_lock = {
        "schema_version": "neurooracle.case2_formal_supplementary_result_lock.v1",
        "protocol_id": protocol["protocol_id"],
        "readiness_freeze_id": lock["freeze_id"],
        "formal_results_sha256": file_sha256(results_path),
        "run_metadata_sha256": file_sha256(metadata_path),
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest_path),
        "run_context_sha256": file_sha256(context_path),
    }
    run_lock["run_id"] = canonical_sha256(run_lock)
    (work_root / "RESULTS.lock.json").write_text(
        json.dumps(run_lock, indent=2) + "\n", encoding="utf-8"
    )
    work_root.rename(output_root)
    return {
        "output_root": str(output_root),
        "run_id": run_lock["run_id"],
        "master_candidates": int(len(master)),
        "estimated_candidates": int(len(estimated)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness-lock",
        type=Path,
        default=DEFAULT_READINESS_ROOT / "READINESS.lock.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) // 2)),
        help="Candidate-level worker processes; seeds and estimates are worker-count invariant.",
    )
    parser.add_argument(
        "--confirm-outcome-access-after-lock",
        action="store_true",
        help="Acknowledge that the verified formal run may load observed outcomes.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    summary = run_formal_analysis(parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
