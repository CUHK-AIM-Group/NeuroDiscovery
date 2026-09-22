"""Independently verify and seal the completed Case Study 2 v3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from neurooracle.scripts.case2_confirmatory_generator_metrics import (
    aggregate_trials,
    paired_primary_comparisons,
)
from neurooracle.scripts.case2_confirmatory_statistics import (
    global_chain_labels,
    pathway_outcome_family_labels,
)
from neurooracle.scripts.evaluate_case2_adni_confirmatory_generators import (
    BASELINE_METHODS,
    NEURODISCOVERY,
    _primary_metric_names,
)
from neurooracle.scripts.freeze_case2_adni_confirmatory_protocol import CASE2_ROOT
from neurooracle.scripts.run_case2_adni_confirmatory_mediation import (
    _holm_adjust_fixed_family,
    _one_sided_p,
    _weakest_link_evidence,
)


DEFAULT_PROTOCOL_ROOT = CASE2_ROOT / "protocols" / "case2_adni_phase_holdout_v3"
EXPECTED_METHODS = {*BASELINE_METHODS, NEURODISCOVERY}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_sha(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _pin(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bool_array(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(bool)
    normalized = values.fillna(False).astype(str).str.strip().str.casefold()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
    if unknown:
        raise ValueError(f"cannot parse boolean values: {unknown[:5]}")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def _verify_result_table(
    result_root: Path,
    *,
    freeze_id: str,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    result_lock_path = result_root / "PHASE_HELDOUT_RESULTS.lock.json"
    manifest_path = result_root / "manifest.json"
    result_lock = _load_json(result_lock_path)
    manifest = _load_json(manifest_path)
    if result_lock.get("protocol_freeze_id") != freeze_id:
        raise ValueError("held-out result lock belongs to another protocol")
    if result_lock.get("holdout_axis") != "cohort_phase":
        raise ValueError("v3 result lock is not cohort-phase held out")
    if result_lock.get("do_not_retune_after_this_lock") is not True:
        raise ValueError("v3 result lock does not prohibit post-outcome retuning")

    result_path = Path(manifest["outputs"]["results"])
    rows_path = Path(manifest["outputs"]["analysis_rows"])
    csv_path = result_path.with_suffix(".csv")
    for path, key, label in (
        (result_path, "results_sha256", "phase-held-out result parquet"),
        (csv_path, "results_csv_sha256", "phase-held-out result CSV"),
        (rows_path, "analysis_rows_sha256", "phase-held-out analysis rows"),
        (manifest_path, "manifest_sha256", "phase-held-out result manifest"),
    ):
        _require_sha(path, str(result_lock[key]), label)

    results = pd.read_parquet(result_path)
    expected_count = int(protocol["candidate_universe"]["expected_count"])
    if len(results) != expected_count or not results["candidate_id"].is_unique:
        raise ValueError("v3 result table does not preserve the public universe")
    alpha = float(protocol["statistics"]["fdr_alpha"])
    family_labels, family_q = pathway_outcome_family_labels(results, alpha)
    global_labels, global_q = global_chain_labels(results, alpha)
    nominal = (
        pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["sobel_p"], errors="coerce").lt(alpha).to_numpy()
    )
    checks = (
        (family_labels, "family_fdr_chain_hit"),
        (global_labels, "global_fdr_chain_hit"),
        (nominal, "nominal_chain_hit"),
    )
    for expected, column in checks:
        if not np.array_equal(expected, _bool_array(results[column])):
            raise ValueError(f"stored {column} labels do not reproduce")
    if not np.allclose(
        family_q,
        pd.to_numeric(results["sobel_q_family"], errors="coerce"),
        equal_nan=True,
    ):
        raise ValueError("stored family-FDR q values do not reproduce")
    if not np.allclose(
        global_q,
        pd.to_numeric(results["sobel_q_global"], errors="coerce"),
        equal_nan=True,
    ):
        raise ValueError("stored global-FDR q values do not reproduce")

    evidence = _weakest_link_evidence(
        results,
        cap=float(protocol.get("generator_evaluation", {}).get("evidence_cap", 50.0)),
    )
    if not np.allclose(
        evidence,
        pd.to_numeric(results["heldout_chain_evidence_score"], errors="coerce"),
        equal_nan=True,
    ):
        raise ValueError("stored held-out chain evidence does not reproduce")

    registered = _bool_array(results["registered_replication_candidate"])
    expected_family_size = int(
        protocol["statistics"]["replication_family_size_expected"]
    )
    if int(registered.sum()) != expected_family_size:
        raise ValueError("registered replication family size changed")
    positions = np.flatnonzero(registered)
    direction_matches: dict[str, np.ndarray] = {}
    for effect, two_sided, sign, one_sided, direction_column in (
        (
            "a_path_std",
            "a_path_p",
            "expected_a_sign",
            "a_path_p_one_sided",
            "a_direction_match",
        ),
        (
            "b_path_std",
            "b_path_p",
            "expected_b_sign",
            "b_path_p_one_sided",
            "b_direction_match",
        ),
        (
            "indirect_effect_std",
            "sobel_p",
            "expected_indirect_sign",
            "sobel_p_one_sided",
            "indirect_direction_match",
        ),
    ):
        recomputed = np.asarray(
            [
                _one_sided_p(
                    results.at[position, two_sided],
                    results.at[position, effect],
                    results.at[position, sign],
                )
                for position in positions
            ],
            dtype=float,
        )
        if not np.allclose(
            recomputed,
            pd.to_numeric(results.loc[positions, one_sided], errors="coerce"),
            equal_nan=True,
        ):
            raise ValueError(f"stored {one_sided} values do not reproduce")
        matches = (
            pd.to_numeric(results.loc[positions, effect], errors="coerce").to_numpy()
            * pd.to_numeric(results.loc[positions, sign], errors="coerce").to_numpy()
            > 0
        )
        if not np.array_equal(
            matches, _bool_array(results.loc[positions, direction_column])
        ):
            raise ValueError(f"stored {direction_column} values do not reproduce")
        direction_matches[direction_column] = matches
    holm = _holm_adjust_fixed_family(
        results.loc[positions, "sobel_p_one_sided"].to_numpy(float)
    )
    if not np.allclose(
        holm,
        pd.to_numeric(
            results.loc[positions, "sobel_p_one_sided_holm"], errors="coerce"
        ),
        equal_nan=True,
    ):
        raise ValueError("stored directional Holm values do not reproduce")
    replication_alpha = float(protocol["statistics"]["replication_alpha"])
    replicated = np.zeros(len(results), dtype=bool)
    replicated[positions] = (
        _bool_array(results.loc[positions, "executable"])
        & direction_matches["a_direction_match"]
        & direction_matches["b_direction_match"]
        & direction_matches["indirect_direction_match"]
        & pd.to_numeric(
            results.loc[positions, "a_path_p_one_sided"], errors="coerce"
        ).lt(replication_alpha).to_numpy()
        & pd.to_numeric(
            results.loc[positions, "b_path_p_one_sided"], errors="coerce"
        ).lt(replication_alpha).to_numpy()
        & pd.to_numeric(
            results.loc[positions, "sobel_p_one_sided"], errors="coerce"
        ).lt(replication_alpha).to_numpy()
        & (holm < replication_alpha)
    )
    if not np.array_equal(
        replicated, _bool_array(results["directional_replication_hit"])
    ):
        raise ValueError("stored directional replication labels do not reproduce")

    counts = {
        "candidate_count": int(len(results)),
        "estimated_candidates": int(_bool_array(results["executable"]).sum()),
        "nominal_chain_hits": int(nominal.sum()),
        "family_fdr_chain_hits": int(family_labels.sum()),
        "global_fdr_chain_hits": int(global_labels.sum()),
        "registered_replication_candidates": int(registered.sum()),
        "directionally_replicated_registered_candidates": int(replicated.sum()),
    }
    return results, counts, {
        "result_lock": _pin(result_lock_path),
        "result_manifest": _pin(manifest_path),
        "results": _pin(result_path),
        "results_csv": _pin(csv_path),
        "analysis_rows": _pin(rows_path),
    }


def _verify_generator_evaluation(
    evaluation_root: Path,
    *,
    freeze_id: str,
    result_lock_sha256: str,
    replication_positive_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = evaluation_root / "manifest.json"
    verdict_path = evaluation_root / "case2_sota_verdict.json"
    plan_lock_path = evaluation_root / "EVALUATION_PLAN.lock.json"
    manifest = _load_json(manifest_path)
    verdict = _load_json(verdict_path)
    plan_lock = _load_json(plan_lock_path)
    plan_path = Path(plan_lock["evaluation_plan_path"])
    _require_sha(plan_path, str(plan_lock["evaluation_plan_sha256"]), "evaluation plan")
    plan = _load_json(plan_path)
    if manifest.get("protocol_freeze_id") != freeze_id:
        raise ValueError("generator evaluation belongs to another protocol")
    if manifest.get("confirmation_result_lock_sha256") != result_lock_sha256:
        raise ValueError("generator evaluation is not bound to the v3 result lock")
    if set(manifest.get("methods", [])) != EXPECTED_METHODS:
        raise ValueError("generator evaluation does not contain seven methods")
    if any(int(value) != 10 for value in manifest["trials_per_method"].values()):
        raise ValueError("generator evaluation does not contain ten runs per method")
    if manifest.get("verdict") != verdict:
        raise ValueError("standalone and embedded SOTA verdicts differ")

    trial_path = Path(manifest["outputs"]["trial_summary"])
    comparison_path = Path(manifest["outputs"]["primary_comparisons"])
    trial_summary = pd.read_csv(trial_path)
    stored_comparisons = pd.read_csv(comparison_path)
    if len(trial_summary) != 70:
        raise ValueError("generator trial table must contain 7 x 10 rows")
    if set(trial_summary["method"]) != EXPECTED_METHODS:
        raise ValueError("generator trial table methods changed")

    comparison_config = plan["comparison"]
    recomputed_frames: list[pd.DataFrame] = []
    for metric_index, metric in enumerate(_primary_metric_names(plan)):
        recomputed_frames.append(
            paired_primary_comparisons(
                trial_summary,
                metric=metric,
                target_method=NEURODISCOVERY,
                baseline_methods=BASELINE_METHODS,
                bootstrap_resamples=int(comparison_config["bootstrap_resamples"]),
                bootstrap_seed=(
                    int(comparison_config["bootstrap_seed"])
                    + 100_000 * metric_index
                ),
            )
        )
    recomputed = pd.concat(recomputed_frames, ignore_index=True).sort_values(
        ["metric", "baseline"], kind="stable"
    )
    stored = stored_comparisons.sort_values(
        ["metric", "baseline"], kind="stable"
    )
    if len(recomputed) != 12 or len(stored) != 12:
        raise ValueError("primary comparison table must contain 2 x 6 rows")
    for column in recomputed.columns:
        if pd.api.types.is_numeric_dtype(recomputed[column]):
            if not np.allclose(
                pd.to_numeric(recomputed[column], errors="coerce"),
                pd.to_numeric(stored[column], errors="coerce"),
                equal_nan=True,
            ):
                raise ValueError(f"primary comparison column does not reproduce: {column}")
        elif recomputed[column].astype(str).tolist() != stored[column].astype(str).tolist():
            raise ValueError(f"primary comparison column does not reproduce: {column}")

    aggregate = aggregate_trials(trial_summary)
    metric_passes: list[bool] = []
    for metric in _primary_metric_names(plan):
        mean_column = f"{metric}_mean"
        target = pd.to_numeric(
            aggregate.loc[aggregate["method"].eq(NEURODISCOVERY), mean_column],
            errors="coerce",
        ).iloc[0]
        baselines = pd.to_numeric(
            aggregate.loc[aggregate["method"].isin(BASELINE_METHODS), mean_column],
            errors="coerce",
        )
        rows = recomputed.loc[recomputed["metric"].eq(metric)]
        metric_passes.append(
            bool(
                np.isfinite(target)
                and baselines.notna().all()
                and target > baselines.max()
                and _bool_array(rows["target_superior"]).all()
            )
        )
    expected_evaluable = replication_positive_count > 0 and all(
        np.isfinite(
            pd.to_numeric(
                aggregate[f"{metric}_mean"], errors="coerce"
            )
        ).all()
        for metric in _primary_metric_names(plan)
    )
    expected_clear_sota = bool(expected_evaluable and all(metric_passes))
    if bool(verdict.get("sota_evaluable")) != expected_evaluable:
        raise ValueError("stored SOTA evaluability does not reproduce")
    if bool(verdict.get("clear_sota")) != expected_clear_sota:
        raise ValueError("stored clear-SOTA verdict does not reproduce")

    output_pins: dict[str, Any] = {}
    for name, raw_path in manifest["outputs"].items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing generator output: {path}")
        output_pins[name] = _pin(path)
    for record in manifest["neurodiscovery_batch_commit_manifests"]:
        _require_sha(Path(record["path"]), record["sha256"], "ND batch commits")
        _require_sha(Path(record["trace_path"]), record["trace_sha256"], "ND trace")
    for record in manifest["neurodiscovery_overlay_manifests"]:
        _require_sha(Path(record["path"]), record["sha256"], "ND overlay")
    return verdict, {
        "evaluation_plan_lock": _pin(plan_lock_path),
        "evaluation_plan": _pin(plan_path),
        "evaluation_manifest": _pin(manifest_path),
        "sota_verdict": _pin(verdict_path),
        "outputs": output_pins,
    }


def build_seal(protocol_root: Path, run_root: Path) -> dict[str, Any]:
    protocol_path = protocol_root / "protocol.json"
    protocol_manifest_path = protocol_root / "protocol_freeze_manifest.json"
    protocol_lock_path = protocol_root / "FREEZE.lock.json"
    protocol = _load_json(protocol_path)
    protocol_manifest = _load_json(protocol_manifest_path)
    protocol_lock = _load_json(protocol_lock_path)
    freeze_id = str(protocol_manifest["freeze_id"])
    if protocol_lock.get("freeze_id") != freeze_id:
        raise ValueError("protocol lock freeze ID mismatch")
    _require_sha(
        protocol_manifest_path,
        str(protocol_lock["manifest_sha256"]),
        "protocol freeze manifest",
    )
    if protocol.get("cohort", {}).get("holdout_axis") != "cohort_phase":
        raise ValueError("this sealer only accepts the v3 cohort-phase protocol")
    if run_root.name != freeze_id[:12]:
        raise ValueError("run root does not match the protocol freeze ID")

    results, counts, result_pins = _verify_result_table(
        run_root / "hidden_phase_heldout_results",
        freeze_id=freeze_id,
        protocol=protocol,
    )
    verdict, evaluation_pins = _verify_generator_evaluation(
        run_root / "generator_evaluation",
        freeze_id=freeze_id,
        result_lock_sha256=result_pins["result_lock"]["sha256"],
        replication_positive_count=int(
            _bool_array(results["directional_replication_hit"]).sum()
        ),
    )

    ranking_locks = {
        "baseline": run_root
        / "official_baselines_gpt55_high"
        / "BASELINE_RANKINGS.lock.json",
        "neurodiscovery": run_root
        / "neurodiscovery_policy"
        / "INITIAL_RANKING.lock.json",
    }
    seal = {
        "schema_version": "neurooracle.case2_phase_holdout_final_seal.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": (
            "sealed_clear_sota"
            if verdict.get("clear_sota")
            else "sealed_without_clear_sota"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_freeze_id": freeze_id,
        "holdout_axis": "cohort_phase",
        "temporal_hindcasting": False,
        "run_root": str(run_root),
        "result_summary": counts,
        "formal_verdict": verdict,
        "immutability_policy": {
            "modify_v3_protocol": False,
            "retune_v3_after_phase_heldout_access": False,
            "relax_v3_thresholds": False,
            "future_changes_require_a_new_protocol_id_and_new_untouched_data": True,
        },
        "pins": {
            "protocol": _pin(protocol_path),
            "protocol_manifest": _pin(protocol_manifest_path),
            "protocol_lock": _pin(protocol_lock_path),
            "ranking_locks": {
                name: _pin(path) for name, path in ranking_locks.items()
            },
            "phase_heldout_results": result_pins,
            "generator_evaluation": evaluation_pins,
        },
    }
    seal["content_sha256"] = _canonical_sha256(seal)
    return seal


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_manifest = _load_json(
        args.protocol_root / "protocol_freeze_manifest.json"
    )
    run_root = args.run_root or (
        CASE2_ROOT
        / "experiments"
        / "case2_adni_phase_holdout_v3"
        / str(protocol_manifest["freeze_id"])[:12]
    )
    output_path = args.output or run_root / "FINAL_RESULT_SEAL.json"
    seal = build_seal(args.protocol_root, run_root)
    if output_path.exists() and not args.force:
        existing = _load_json(output_path)
        existing_without_hash = dict(existing)
        existing_hash = existing_without_hash.pop("content_sha256", None)
        if existing_hash != _canonical_sha256(existing_without_hash):
            raise ValueError("existing final seal is internally inconsistent")
        stable = dict(seal)
        stable.pop("created_at_utc", None)
        stable.pop("content_sha256", None)
        previous = dict(existing)
        previous.pop("created_at_utc", None)
        previous.pop("content_sha256", None)
        if previous != stable:
            raise ValueError("current artifacts differ from the existing final seal")
        return existing
    output_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), **seal}, indent=2))
    return seal


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 14:08 HKT
