"""Export the locked Case Study 2 biological-results tables.

This reporter is deliberately downstream of the formal result lock.  It
verifies both the outcome-blind readiness freeze and the completed formal run,
then reattaches the observed ADAS13, FAQ, and LDELTOTAL values to the exact
locked candidate rows.  It does not refit or modify the formal analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from neurooracle.scripts.case2_formal_supplementary import (
    canonical_sha256,
    complete_candidate_frame,
    file_sha256,
)
from neurooracle.scripts.prepare_case2_adni_formal_supplementary import HKT
from neurooracle.scripts.run_case2_adni_formal_supplementary import (
    _load_frozen_inputs,
    assemble_candidate_frame,
    verify_readiness_lock,
)


RESULT_LOCK_SCHEMA = "neurooracle.case2_formal_supplementary_result_lock.v1"
REPORT_LOCK_SCHEMA = "neurooracle.case2_biological_results_lock.v1"
TARGET_OUTCOMES = ("ADAS13", "FAQ", "LDELTOTAL")
MARKER_LABELS = {
    "amyloid_pet::CENTILOIDS": "Amyloid PET Centiloids",
    "fdg_pet::FDG_META_ROI_SUVR": "FDG meta-ROI SUVR",
    "tau_pet::META_TEMPORAL_SUVR": "Meta-temporal tau SUVR",
    "tau_pet::CTX_ENTORHINAL_SUVR": "Entorhinal tau SUVR",
    "smri_adnimerge::Hippocampus": "Hippocampal volume",
    "smri_adnimerge::Entorhinal": "Entorhinal volume",
    "smri_adnimerge::WholeBrain": "Whole-brain volume",
    "smri_adnimerge::Ventricles": "Ventricular volume",
}


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, label: str
) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def verify_formal_result_lock(
    result_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the formal result lock and every artifact pinned by it."""

    lock_path = result_root / "RESULTS.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != RESULT_LOCK_SCHEMA:
        raise ValueError(f"Unexpected result lock schema: {lock.get('schema_version')}")
    run_id = lock.get("run_id")
    material = {key: value for key, value in lock.items() if key != "run_id"}
    if canonical_sha256(material) != run_id:
        raise ValueError("Formal run ID does not match its lock material")

    pins = {
        "FORMAL_RESULTS.csv": "formal_results_sha256",
        "RUN_METADATA.json": "run_metadata_sha256",
        "CHECKPOINT_MANIFEST.json": "checkpoint_manifest_sha256",
        "RUN_CONTEXT.json": "run_context_sha256",
    }
    for filename, hash_field in pins.items():
        path = result_root / filename
        if not path.is_file() or file_sha256(path) != lock.get(hash_field):
            raise ValueError(f"Pinned formal artifact changed: {filename}")

    metadata = json.loads(
        (result_root / "RUN_METADATA.json").read_text(encoding="utf-8")
    )
    if metadata.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("Formal metadata and result lock protocol IDs differ")
    if metadata.get("readiness_freeze_id") != lock.get("readiness_freeze_id"):
        raise ValueError("Formal metadata and result lock readiness freezes differ")
    return lock, metadata


def classify_results(results: pd.DataFrame) -> pd.Series:
    """Assign the prespecified result tier without changing any P or q value."""

    _require_columns(
        results,
        (
            "indirect_bootstrap_p",
            "indirect_bootstrap_q_global_168",
            "indirect_bootstrap_q_family_8",
        ),
        label="Formal results",
    )
    p_value = pd.to_numeric(results["indirect_bootstrap_p"], errors="coerce")
    q_global = pd.to_numeric(
        results["indirect_bootstrap_q_global_168"], errors="coerce"
    )
    q_family = pd.to_numeric(results["indirect_bootstrap_q_family_8"], errors="coerce")
    tier = pd.Series("not_nominal", index=results.index, dtype="object")
    tier.loc[p_value.lt(0.05)] = "nominal_only"
    tier.loc[q_family.lt(0.05)] = "supplemental_family_fdr"
    tier.loc[q_global.lt(0.05)] = "primary_global_fdr"
    return tier


def build_source_outcome_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize the observed raw scores in the frozen 691-person source table."""

    _require_columns(outcomes, ("subject_id", "outcome", "value"), label="Outcomes")
    selected = outcomes[outcomes["outcome"].isin(TARGET_OUTCOMES)].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for outcome in TARGET_OUTCOMES:
        group = selected[selected["outcome"].eq(outcome)]
        values = group["value"].dropna()
        rows.append(
            {
                "outcome": outcome,
                "raw_score_direction": (
                    "higher_is_better; multiplied_by_-1_for_model"
                    if outcome == "LDELTOTAL"
                    else "higher_is_worse"
                ),
                "source_observation_n": int(len(group)),
                "source_subject_n": int(group["subject_id"].nunique()),
                "missing_score_n": int(group["value"].isna().sum()),
                "raw_mean": float(values.mean()),
                "raw_sd": float(values.std(ddof=1)),
                "raw_median": float(values.median()),
                "raw_min": float(values.min()),
                "raw_max": float(values.max()),
            }
        )
    summary = pd.DataFrame(rows)
    if set(summary["outcome"]) != set(TARGET_OUTCOMES):
        raise ValueError("Frozen outcome table does not contain all target outcomes")
    return summary


def _describe(values: pd.Series, prefix: str) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"No observed values available for {prefix}")
    return {
        f"{prefix}_n": int(len(numeric)),
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_sd": float(numeric.std(ddof=1)),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_max": float(numeric.max()),
    }


def build_candidate_score_summary(
    executable: pd.DataFrame,
    formal_results: pd.DataFrame,
    row_keys: pd.DataFrame,
    markers: pd.DataFrame,
    outcomes: pd.DataFrame,
    subjects: pd.DataFrame,
    baseline_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Reattach actual outcome scores to each locked complete-case sample."""

    formal_n = formal_results.set_index("candidate_id")["n"].to_dict()
    rows: list[dict[str, Any]] = []
    for candidate in executable.to_dict(orient="records"):
        frame = assemble_candidate_frame(
            candidate,
            row_keys,
            markers,
            outcomes,
            subjects,
            baseline_dates,
        )
        complete = complete_candidate_frame(
            frame,
            include_icv=candidate["modality"] == "smri_adnimerge",
        )
        expected_n = int(formal_n[candidate["candidate_id"]])
        if len(complete) != expected_n:
            raise ValueError(
                f"Complete-case N changed for {candidate['candidate_id']}: "
                f"expected {expected_n}, observed {len(complete)}"
            )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                **_describe(complete["baseline_outcome_raw"], "baseline_raw"),
                **_describe(complete["future_outcome_raw"], "future_raw"),
                **_describe(complete["exact_followup_days"], "followup_days"),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(executable) or not result["candidate_id"].is_unique:
        raise ValueError("Candidate score summary is not one-to-one with execution")
    return result


def build_biological_results(
    formal_results: pd.DataFrame,
    score_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the readable all-candidate biological-results table."""

    results = formal_results.merge(
        score_summary,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    results["result_tier"] = classify_results(results)
    results["marker_label"] = results["marker_spec"].map(MARKER_LABELS)
    if results["marker_label"].isna().any():
        missing = sorted(
            results.loc[results["marker_label"].isna(), "marker_spec"].unique()
        )
        raise ValueError(f"Missing biological marker labels: {missing}")
    results["outcome_modeling_note"] = np.where(
        results["outcome"].eq("LDELTOTAL"),
        "raw higher=better; modeled x -1 so higher=worse",
        "raw and modeled higher=worse",
    )
    results["association_scope"] = (
        "standardized association-based indirect effect; not causal mediation"
    )

    columns = [
        "candidate_id",
        "pathway_id",
        "pathway_name",
        "threshold_label",
        "modality",
        "marker",
        "marker_spec",
        "marker_label",
        "outcome",
        "outcome_domain",
        "outcome_modeling_note",
        "n",
        "baseline_raw_mean",
        "baseline_raw_sd",
        "baseline_raw_median",
        "baseline_raw_min",
        "baseline_raw_max",
        "future_raw_mean",
        "future_raw_sd",
        "future_raw_median",
        "future_raw_min",
        "future_raw_max",
        "followup_days_mean",
        "followup_days_sd",
        "followup_days_median",
        "followup_days_min",
        "followup_days_max",
        "a_path_std",
        "a_path_hc3_se",
        "a_path_hc3_p",
        "b_path_std",
        "b_path_hc3_se",
        "b_path_hc3_p",
        "direct_effect_std",
        "direct_effect_hc3_se",
        "direct_effect_hc3_p",
        "indirect_effect_std",
        "indirect_bootstrap_ci_lower",
        "indirect_bootstrap_ci_upper",
        "indirect_bootstrap_p",
        "indirect_bootstrap_q_global_168",
        "indirect_bootstrap_q_family_8",
        "bootstrap_valid_replicates",
        "master_seed_count",
        "master_bootstrap_seed",
        "candidate_bootstrap_seed",
        "result_tier",
        "association_scope",
    ]
    _require_columns(results, columns, label="Merged biological results")
    return (
        results.loc[:, columns]
        .sort_values(
            [
                "indirect_bootstrap_q_global_168",
                "indirect_bootstrap_q_family_8",
                "indirect_bootstrap_p",
                "candidate_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_report(
    *,
    lock: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_summary: pd.DataFrame,
    biological: pd.DataFrame,
) -> str:
    supplemental = biological[
        biological["result_tier"].eq("supplemental_family_fdr")
    ].copy()
    nominal_n = int(pd.to_numeric(biological["indirect_bootstrap_p"]).lt(0.05).sum())
    family_n = int(
        pd.to_numeric(biological["indirect_bootstrap_q_family_8"]).lt(0.05).sum()
    )
    global_n = int(
        pd.to_numeric(biological["indirect_bootstrap_q_global_168"]).lt(0.05).sum()
    )
    lines = [
        "# Case Study 2 biological results",
        "",
        f"- Protocol: `{metadata['protocol_id']}`",
        f"- Formal run ID: `{lock['run_id']}`",
        f"- Readiness freeze ID: `{lock['readiness_freeze_id']}`",
        f"- Candidates fitted: {len(biological)}/168",
        f"- Master seeds: {metadata['master_seed_count']} (`{metadata['master_bootstrap_seed']}`)",
        f"- Bootstrap: {metadata['bootstrap_replicates_per_candidate']} valid replicates per candidate",
        "- Interpretation: association-based indirect effects; not causal mediation",
        "",
        "## Observed source scores",
        "",
        "| Outcome | Observations | Subjects | Mean | SD | Median | Range | Raw direction |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in source_summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.outcome,
                    str(row.source_observation_n),
                    str(row.source_subject_n),
                    _fmt(row.raw_mean),
                    _fmt(row.raw_sd),
                    _fmt(row.raw_median),
                    f"{_fmt(row.raw_min)}-{_fmt(row.raw_max)}",
                    row.raw_score_direction,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Multiplicity summary",
            "",
            f"- Nominal bootstrap P < 0.05: {nominal_n}",
            f"- Supplemental family-FDR q < 0.05: {family_n}",
            f"- Primary global-FDR q < 0.05: {global_n}",
            "",
            "## Supplemental family-FDR results",
            "",
            "| Pathway PRS | Imaging marker | Outcome | N | Path a | Path b | Indirect effect (95% CI) | P | Family q | Global q |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    supplemental = supplemental.sort_values(
        ["indirect_bootstrap_q_family_8", "indirect_bootstrap_p"], kind="stable"
    )
    for row in supplemental.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.pathway_name,
                    row.marker_label,
                    row.outcome,
                    str(int(row.n)),
                    _fmt(row.a_path_std),
                    _fmt(row.b_path_std),
                    (
                        f"{_fmt(row.indirect_effect_std)} "
                        f"[{_fmt(row.indirect_bootstrap_ci_lower)}, "
                        f"{_fmt(row.indirect_bootstrap_ci_upper)}]"
                    ),
                    _fmt(row.indirect_bootstrap_p),
                    _fmt(row.indirect_bootstrap_q_family_8),
                    _fmt(row.indirect_bootstrap_q_global_168),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The global 168-test endpoint has no FDR-controlled discovery. The table above is supplemental and must not be presented as confirmatory or causal.",
            "",
        ]
    )
    return "\n".join(lines)


def export_biological_results(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_outcome_access_after_result_lock:
        raise PermissionError(
            "Observed outcome access is fail-closed. Re-run with "
            "--confirm-outcome-access-after-result-lock after verifying the formal lock."
        )
    result_root = args.result_root.resolve()
    result_lock, metadata = verify_formal_result_lock(result_root)
    readiness_lock, protocol = verify_readiness_lock(args.readiness_lock)
    if result_lock["readiness_freeze_id"] != readiness_lock["freeze_id"]:
        raise ValueError("Formal result and readiness lock freeze IDs differ")
    if result_lock["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Formal result and readiness protocol IDs differ")

    formal_results = pd.read_csv(result_root / "FORMAL_RESULTS.csv")
    if len(formal_results) != 168 or not formal_results["candidate_id"].is_unique:
        raise ValueError("Formal result table is not the fixed 168-candidate universe")
    if not formal_results["analysis_status"].eq("estimated").all():
        raise ValueError("Not all fixed formal candidates were estimated")
    if not formal_results["master_seed_count"].eq(1).all():
        raise ValueError("Current biological report requires the locked one-seed run")

    (
        _,
        executable,
        row_keys,
        subjects,
        markers,
        outcomes,
        baseline_dates,
    ) = _load_frozen_inputs(readiness_lock, protocol, args.readiness_lock.parent)
    source_summary = build_source_outcome_summary(outcomes)
    score_summary = build_candidate_score_summary(
        executable,
        formal_results,
        row_keys,
        markers,
        outcomes,
        subjects,
        baseline_dates,
    )
    biological = build_biological_results(formal_results, score_summary)
    supplemental = biological[
        biological["result_tier"].eq("supplemental_family_fdr")
    ].copy()

    output_root = args.output_root or (
        result_root.parent.parent
        / "biological_results"
        / f"run_{result_lock['run_id'][:12]}"
    )
    if output_root.exists():
        raise FileExistsError(f"Biological result output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    work_root = output_root.parent / f".{output_root.name}.work"
    if work_root.exists():
        raise FileExistsError(f"Biological result work directory exists: {work_root}")
    work_root.mkdir()

    output_files = {
        "source_score_summary": work_root / "OUTCOME_SOURCE_SCORE_SUMMARY.csv",
        "candidate_score_summary": work_root / "CANDIDATE_OUTCOME_SCORE_SUMMARY.csv",
        "all_results": work_root / "BIOLOGICAL_RESULTS_ALL.csv",
        "supplemental_results": work_root / "BIOLOGICAL_RESULTS_SUPPLEMENTAL_FDR.csv",
        "report": work_root / "BIOLOGICAL_RESULTS.md",
    }
    source_summary.to_csv(output_files["source_score_summary"], index=False)
    score_summary.to_csv(output_files["candidate_score_summary"], index=False)
    biological.to_csv(output_files["all_results"], index=False)
    supplemental.to_csv(output_files["supplemental_results"], index=False)
    output_files["report"].write_text(
        _markdown_report(
            lock=result_lock,
            metadata=metadata,
            source_summary=source_summary,
            biological=biological,
        ),
        encoding="utf-8",
    )

    report_material = {
        "schema_version": REPORT_LOCK_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "formal_run_id": result_lock["run_id"],
        "readiness_freeze_id": readiness_lock["freeze_id"],
        "generated_at_hkt": datetime.now(HKT).isoformat(),
        "source_subject_count": int(source_summary["source_subject_n"].max()),
        "candidate_count": int(len(biological)),
        "nominal_p_lt_0_05_count": int(
            pd.to_numeric(biological["indirect_bootstrap_p"]).lt(0.05).sum()
        ),
        "supplemental_family_q_lt_0_05_count": int(len(supplemental)),
        "primary_global_q_lt_0_05_count": int(
            pd.to_numeric(biological["indirect_bootstrap_q_global_168"]).lt(0.05).sum()
        ),
        "files": {
            label: {
                "filename": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
            for label, path in output_files.items()
        },
    }
    report_lock = {
        **report_material,
        "report_id": canonical_sha256(report_material),
    }
    (work_root / "BIOLOGICAL_RESULTS.lock.json").write_text(
        json.dumps(report_lock, indent=2) + "\n",
        encoding="utf-8",
    )
    work_root.rename(output_root)
    return {
        "output_root": str(output_root),
        "report_id": report_lock["report_id"],
        "candidate_count": int(len(biological)),
        "supplemental_family_fdr_count": int(len(supplemental)),
        "primary_global_fdr_count": int(
            pd.to_numeric(biological["indirect_bootstrap_q_global_168"]).lt(0.05).sum()
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--readiness-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--confirm-outcome-access-after-result-lock",
        action="store_true",
        help="Acknowledge that observed ADAS13, FAQ, and LDELTOTAL values will be read.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    result = export_biological_results(parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
