"""Evaluate frozen Case Study 1 TCP rankings on independent cohorts.

The generator rankings are reconstructed exclusively from TCP data and the
frozen knowledge graph. External outcomes are joined only after every ranking
has been fixed, preventing leakage from UCLA, COBRE, HCP-EP, or ADHD200 into
hypothesis generation.
"""

from __future__ import annotations

import argparse
import gzip
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

try:
    from core.scripts import build_case1_external_labels as external_labels
    from core.scripts import case1_method_comparison as comparison
    from core.scripts.canonical_kg_release import (
        validate_canonical_kg_release,
        write_release_manifest,
    )
    from core.scripts.run_case1_neurodiscovery_formal import (
        EXTERNAL_SOTA_DEFINITION,
        audit_seed as audit_formal_neurodiscovery_seed,
    )
except ModuleNotFoundError:
    import build_case1_external_labels as external_labels
    import case1_method_comparison as comparison
    from canonical_kg_release import (
        validate_canonical_kg_release,
        write_release_manifest,
    )
    from run_case1_neurodiscovery_formal import (
        EXTERNAL_SOTA_DEFINITION,
        audit_seed as audit_formal_neurodiscovery_seed,
    )


DEFAULT_BUDGETS = (
    5_000,
    10_000,
    50_000,
    100_000,
    200_000,
)
DEFAULT_RECALL_TARGETS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
OUTCOME_NAMES = ("confirmed", "falsified", "conflicting", "not_confirmed")
NATIVE_METHOD_LABELS = {
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
}
PRIMARY_METHODS = {
    *comparison.PRIMARY_BASELINE_METHODS,
    *NATIVE_METHOD_LABELS,
    "neurodiscovery",
}


def method_label(method: str) -> str:
    return comparison.METHOD_LABELS.get(
        method, NATIVE_METHOD_LABELS.get(method, method)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_internal_sota_evidence(formal_root: Path) -> dict[str, Any]:
    evaluation_dir = formal_root / "evaluation"
    manifest_path = evaluation_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "case1-neurodiscovery-formal-evaluation.v1"
        or manifest.get("status") != "complete_sota"
        or manifest.get("seeds") != [0, 1, 2]
        or manifest.get("ranking_freeze_preceded_gt_load") is not True
    ):
        raise RuntimeError("internal formal SOTA evaluation is incomplete")
    gate_record = (manifest.get("artifacts") or {}).get("sota_gate.json") or {}
    gate_path = Path(str(gate_record.get("path") or ""))
    if (
        not gate_path.is_file()
        or sha256_file(gate_path) != gate_record.get("sha256")
    ):
        raise RuntimeError("internal SOTA gate artifact hash mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("sota_achieved") is not True:
        raise RuntimeError("internal SOTA gate did not pass")
    return {
        "evaluation_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "internal_sota_gate": {
            "path": str(gate_path.resolve()),
            "sha256": sha256_file(gate_path),
            **gate,
        },
    }


def neurodiscovery_tail_static_score(
    scored: pd.DataFrame,
    config: comparison.Case1NeuroDiscoveryConfig,
    *,
    selected_total: int,
) -> np.ndarray:
    """Reproduce the outcome-blind score used after the feedback horizon."""

    base_initial = scored["score_neurodiscovery"].to_numpy(float).copy()
    if config.feature_support_decay_budget <= 0:
        return base_initial
    support_weight = config.global_support_weight + config.scoped_support_weight
    if support_weight <= 0:
        raise ValueError("NeuroDiscovery support weights must sum to a positive value")
    base_floor = (
        config.global_support_weight
        * scored["score_neurodiscovery_global_base"].to_numpy(float)
        + config.scoped_support_weight
        * scored["score_case_study_support_base"].to_numpy(float)
    ) / support_weight
    base_floor += 0.01 * scored["score_random"].to_numpy(float)
    remaining_prior = max(
        0.0,
        1.0
        - float(selected_total) / max(config.feature_support_decay_budget, 1),
    )
    return base_floor + remaining_prior * (base_initial - base_floor)


def load_neurodiscovery_overlay_order(
    scored: pd.DataFrame,
    overlay_dir: Path,
    *,
    seed: int,
    trial: int,
    rng: np.random.Generator,
    config: comparison.Case1NeuroDiscoveryConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover a full TCP ranking from its verified feedback-prefix hash chain."""

    base = overlay_dir / f"seed_{seed}_trial_{trial:02d}.jsonl"
    candidates = (base.with_suffix(base.suffix + ".gz"), base)
    overlay_path = next((path for path in candidates if path.is_file()), None)
    if overlay_path is None:
        raise FileNotFoundError(base)

    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    id_to_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    seen = np.zeros(len(candidate_ids), dtype=bool)
    order: list[int] = []
    previous_hash = "0" * 64
    status_counts: dict[str, int] = {}
    round_counts: dict[int, int] = {}
    previous_round = -1
    opener = gzip.open if overlay_path.suffix == ".gz" else open
    with opener(overlay_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "experimental-claim.v2":
                raise ValueError(
                    f"Unexpected overlay schema at {overlay_path}:{line_number}"
                )
            if record.get("case_study_id") != comparison.CASE1_CASE_STUDY_ID:
                raise ValueError(
                    f"Overlay case-study mismatch at {overlay_path}:{line_number}"
                )
            if int(record.get("seed", -1)) != seed or int(
                record.get("trial", -1)
            ) != trial:
                raise ValueError(
                    f"Overlay seed/trial mismatch at {overlay_path}:{line_number}"
                )
            if record.get("previous_hash") != previous_hash:
                raise ValueError(
                    f"Overlay hash-chain break at {overlay_path}:{line_number}"
                )
            record_hash = str(record.get("record_hash") or "")
            canonical_record = dict(record)
            canonical_record.pop("record_hash", None)
            calculated_hash = hashlib.sha256(
                json.dumps(
                    canonical_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if not record_hash or calculated_hash != record_hash:
                raise ValueError(
                    f"Overlay record hash mismatch at {overlay_path}:{line_number}"
                )
            previous_hash = record_hash

            round_index = int(record.get("round", -1))
            if round_index < 0 or (
                previous_round >= 0
                and round_index not in (previous_round, previous_round + 1)
            ):
                raise ValueError(
                    f"Overlay round sequence is invalid at {overlay_path}:{line_number}"
                )
            if previous_round < 0 and round_index != 0:
                raise ValueError(
                    f"Overlay round sequence must start at zero: {overlay_path}"
                )
            previous_round = round_index
            round_counts[round_index] = round_counts.get(round_index, 0) + 1

            candidate_id = str(record.get("candidate_id") or "")
            candidate_index = id_to_index.get(candidate_id)
            if candidate_index is None:
                raise ValueError(
                    f"Overlay contains an unknown candidate at {overlay_path}:{line_number}"
                )
            if seen[candidate_index]:
                raise ValueError(
                    f"Overlay repeats candidate {candidate_id!r} at line {line_number}"
                )
            seen[candidate_index] = True
            order.append(candidate_index)
            status = str(record.get("status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1

    expected_prefix = min(len(candidate_ids), int(config.max_closed_loop_budget))
    if len(order) != expected_prefix:
        raise ValueError(
            "NeuroDiscovery overlay does not contain the expected TCP feedback prefix: "
            f"{len(order):,}/{expected_prefix:,} candidates"
        )
    if any(count > int(config.batch_size) for count in round_counts.values()):
        raise ValueError("NeuroDiscovery overlay round exceeds the frozen batch size")

    prefix = np.asarray(order, dtype=np.int32)
    if len(prefix) < len(candidate_ids):
        # The internal generator draws one full-length normal vector per feedback
        # round before drawing the frozen tail perturbation. Advancing the same RNG
        # reproduces that tail without observing any external-cohort outcome.
        for _ in range(len(round_counts)):
            rng.normal(0.0, 1.0, size=len(candidate_ids))
        tail_score = neurodiscovery_tail_static_score(
            scored,
            config,
            selected_total=len(prefix),
        ) + rng.normal(0.0, 0.005, size=len(candidate_ids))
        remaining = np.flatnonzero(~seen)
        tail = remaining[
            np.lexsort((candidate_ids[remaining], -tail_score[remaining]))
        ].astype(np.int32)
        compact = np.concatenate((prefix, tail))
    else:
        compact = prefix
    if len(compact) != len(candidate_ids) or len(np.unique(compact)) != len(
        candidate_ids
    ):
        raise RuntimeError("Verified NeuroDiscovery ranking is not a full permutation")
    return compact, {
        "schema_version": "case1-verified-overlay-order.v2",
        "path": str(overlay_path.resolve()),
        "sha256": sha256_file(overlay_path),
        "seed": seed,
        "trial": trial,
        "records": len(prefix),
        "feedback_prefix_records": len(prefix),
        "tail_records": len(compact) - len(prefix),
        "feedback_rounds": len(round_counts),
        "records_by_status": dict(sorted(status_counts.items())),
        "final_chain_hash": previous_hash,
        "candidate_order_sha256": hashlib.sha256(compact.tobytes()).hexdigest(),
    }


def load_formal_neurodiscovery_order(
    scored: pd.DataFrame,
    formal_root: Path,
    *,
    trial: int,
    config: comparison.Case1NeuroDiscoveryConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one commitment-audited ranking frozen before GT/external access."""

    input_manifest_path = formal_root / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    candidate_order_sha256 = hashlib.sha256(
        "\n".join(candidate_ids).encode("utf-8")
    ).hexdigest()
    if input_manifest.get("candidate_order_sha256") != candidate_order_sha256:
        raise ValueError("formal NeuroDiscovery candidate order does not match TCP")

    seed_dir = formal_root / "neurodiscovery" / f"seed_{trial:02d}"
    seed_manifest_path = seed_dir / "seed_manifest.json"
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    if (
        seed_manifest.get("schema_version")
        != "case1-neurodiscovery-formal-seed.v1"
        or seed_manifest.get("status") != "complete"
        or int(seed_manifest.get("seed", -1)) != trial
    ):
        raise ValueError(f"formal NeuroDiscovery seed {trial} is not complete")
    if seed_manifest.get("config") != config.to_dict():
        raise ValueError("formal NeuroDiscovery config differs from external run")
    integrity = seed_manifest.get("closed_loop_integrity") or {}
    integrity_checks = {
        "outcome_blind_scoring": bool(
            integrity.get("public_scoring_frame_outcome_blind")
        ),
        "gt_not_opened": integrity.get("gt_labels_opened_by_seed_process") is False,
        "feedback_consumed": bool(
            integrity.get("feedback_consumed_during_ranking")
        ),
        "nonzero_feedback": int(integrity.get("nonzero_feedback_reads") or 0) > 0,
        "selection_changed": int(
            integrity.get("selection_changed_batches") or 0
        )
        > 0,
        "commit_before_reveal": bool(integrity.get("commit_before_reveal")),
        "ranking_frozen_before_gt": bool(
            seed_manifest.get("ranking_frozen_before_gt_evaluation")
        ),
    }
    if not all(integrity_checks.values()):
        raise ValueError(
            f"formal NeuroDiscovery seed {trial} failed integrity: {integrity_checks}"
        )
    formal_seed_audit = audit_formal_neurodiscovery_seed(formal_root, trial)
    if formal_seed_audit.get("status") != "passed":
        raise ValueError(
            f"formal NeuroDiscovery seed {trial} failed artifact audit: "
            f"{formal_seed_audit}"
        )
    ranking = seed_manifest.get("frozen_ranking") or {}
    ranking_path = Path(str(ranking.get("path") or ""))
    if not ranking_path.is_file() or sha256_file(ranking_path) != ranking.get(
        "sha256"
    ):
        raise ValueError("formal NeuroDiscovery frozen ranking hash mismatch")
    with ranking_path.open("rb") as handle:
        order = np.load(handle, allow_pickle=False).astype(np.int64)
    if (
        len(order) != len(candidate_ids)
        or np.any(order < 0)
        or np.any(order >= len(candidate_ids))
        or len(np.unique(order)) != len(candidate_ids)
    ):
        raise ValueError("formal NeuroDiscovery ranking is not a full permutation")
    return order, {
        "schema_version": "case1-formal-neurodiscovery-order.v1",
        "formal_root": str(formal_root.resolve()),
        "input_manifest": str(input_manifest_path.resolve()),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "seed_manifest": str(seed_manifest_path.resolve()),
        "seed_manifest_sha256": sha256_file(seed_manifest_path),
        "trial": trial,
        "ranking_path": str(ranking_path.resolve()),
        "ranking_sha256": ranking["sha256"],
        "closed_loop_integrity": integrity_checks,
        "formal_seed_audit": formal_seed_audit,
    }


def reconstruct_orders(
    scored: pd.DataFrame,
    generation_policies: dict[tuple[str, int], Any],
    *,
    n_trials: int,
    seed: int,
    neurodiscovery_config: comparison.Case1NeuroDiscoveryConfig,
    neurodiscovery_overlay_dir: Path | None = None,
    neurodiscovery_formal_root: Path | None = None,
    overlay_audits: list[dict[str, Any]] | None = None,
):
    for trial in range(n_trials):
        for method in comparison.GENERATOR_METHODS:
            rng = np.random.default_rng(
                seed
                + 1009 * trial
                + 7919 * (comparison.GENERATOR_METHODS.index(method) + 1)
            )
            if method == "neurodiscovery":
                if neurodiscovery_formal_root is not None:
                    order, audit = load_formal_neurodiscovery_order(
                        scored,
                        neurodiscovery_formal_root,
                        trial=trial,
                        config=neurodiscovery_config,
                    )
                    if overlay_audits is not None:
                        overlay_audits.append(audit)
                elif neurodiscovery_overlay_dir is None:
                    order = comparison.closed_loop_neurodiscovery_order(
                        scored,
                        rng,
                        config=neurodiscovery_config,
                    )
                else:
                    order, audit = load_neurodiscovery_overlay_order(
                        scored,
                        neurodiscovery_overlay_dir,
                        seed=seed,
                        trial=trial,
                        rng=rng,
                        config=neurodiscovery_config,
                    )
                    if overlay_audits is not None:
                        overlay_audits.append(audit)
            elif (method, trial) in generation_policies:
                order = comparison.compile_policy_order(
                    scored,
                    generation_policies[(method, trial)],
                )
            else:
                raise ValueError(
                    f"Missing frozen SearchPolicy for {method} trial {trial}; "
                    "external validation cannot substitute an emulated or random ranking"
                )
            yield trial, method, order


def freeze_tcp_rankings(
    scored: pd.DataFrame,
    generation_policies: dict[tuple[str, int], Any],
    *,
    out_dir: Path,
    all_tests: Path,
    kg: Path,
    search_policy_path: Path,
    n_trials: int,
    seed: int,
    neurodiscovery_config: comparison.Case1NeuroDiscoveryConfig,
    ranking_provenance: dict[str, Any],
    neurodiscovery_overlay_dir: Path | None = None,
    neurodiscovery_formal_root: Path | None = None,
) -> tuple[list[tuple[int, str, np.ndarray]], dict[str, Any]]:
    """Materialize every TCP-only order before external outcomes are loaded."""

    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    overlay_audits: list[dict[str, Any]] = []
    frozen_orders = list(
        reconstruct_orders(
            scored,
            generation_policies,
            n_trials=n_trials,
            seed=seed,
            neurodiscovery_config=neurodiscovery_config,
            neurodiscovery_overlay_dir=neurodiscovery_overlay_dir,
            neurodiscovery_formal_root=neurodiscovery_formal_root,
            overlay_audits=overlay_audits,
        )
    )

    frozen_dir = out_dir / "frozen_tcp_rankings"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    candidate_columns = [
        column
        for column in (
            "candidate_id",
            "modality",
            "source",
            "disease",
            "feature",
            "feature_family",
            "roi_index",
            "roi_id",
            "roi_name",
            "anatomy_full",
            "network",
            "adjusted_residual_d",
            "score_neurodiscovery_global",
            "score_case_study_support",
            "score_neurodiscovery",
            "kg_scoped_disease_degree",
            "kg_scoped_region_degree",
            "kg_scoped_pair_support",
        )
        if column in scored.columns
    ]
    candidates_path = frozen_dir / "frozen_tcp_candidates.csv.gz"
    scored[candidate_columns].to_csv(
        candidates_path,
        index=False,
        compression="gzip",
    )

    payload: dict[str, np.ndarray] = {}
    order_records: list[dict[str, Any]] = []
    for index, (trial, method, order) in enumerate(frozen_orders):
        key = f"order_{index:03d}"
        compact = np.asarray(order, dtype=np.int32)
        payload[key] = compact
        order_records.append(
            {
                "array_key": key,
                "method": method,
                "trial": int(trial),
                "n_slots": int(len(compact)),
                "order_sha256": hashlib.sha256(compact.tobytes()).hexdigest(),
            }
        )
    orders_path = frozen_dir / "frozen_tcp_orders.npz"
    np.savez_compressed(orders_path, **payload)

    method_trial_counts = (
        pd.DataFrame(order_records)
        .groupby("method", sort=True)["trial"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "schema_version": "case1-frozen-tcp-rankings-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "discovery_dataset": "TCP",
        "case_study_membership": {
            "schema_version": "case_study_membership.v2",
            "case_study_id": comparison.CASE1_CASE_STUDY_ID,
            "global_support_weight": comparison.CASE1_GLOBAL_SUPPORT_WEIGHT,
            "case_study_support_weight": comparison.CASE1_SCOPED_SUPPORT_WEIGHT,
        },
        "external_data_read_before_freeze": False,
        "candidate_count": int(len(candidate_ids)),
        "candidate_id_sha256": hashlib.sha256(
            "\n".join(candidate_ids).encode("utf-8")
        ).hexdigest(),
        "candidate_table": {
            "path": str(candidates_path),
            "sha256": sha256_file(candidates_path),
        },
        "orders": {
            "path": str(orders_path),
            "sha256": sha256_file(orders_path),
            "n_orders": len(order_records),
            "n_trials_by_method": method_trial_counts,
            "records": order_records,
        },
        "inputs": {
            "tcp_all_tests": {
                "path": str(all_tests),
                "sha256": sha256_file(all_tests),
            },
            "kg": {"path": str(kg), "sha256": sha256_file(kg)},
            "search_policies": {
                "path": str(search_policy_path),
                "sha256": sha256_file(search_policy_path),
            },
        },
        "seed_base": seed,
        "n_trials": n_trials,
        "ranking_provenance": ranking_provenance,
        "neurodiscovery_order_materialization": (
            "formal_commitment_audited_frozen_rankings"
            if neurodiscovery_formal_root is not None
            else "verified_experimental_overlays"
            if neurodiscovery_overlay_dir is not None
            else "recomputed_closed_loop"
        ),
        "neurodiscovery_overlay_audits": overlay_audits,
    }
    manifest_path = frozen_dir / "frozen_tcp_rankings_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return frozen_orders, manifest


def load_frozen_tcp_rankings(
    frozen_dir: Path,
    candidate_ids: np.ndarray,
    *,
    expected_ranking_provenance: dict[str, Any],
) -> tuple[list[tuple[int, str, np.ndarray]], dict[str, Any]]:
    """Load and verify previously materialized TCP-only rankings."""

    manifest_path = frozen_dir / "frozen_tcp_rankings_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_candidate_hash = hashlib.sha256(
        "\n".join(candidate_ids.astype(str)).encode("utf-8")
    ).hexdigest()
    if manifest["candidate_id_sha256"] != expected_candidate_hash:
        raise ValueError("Frozen TCP candidate identifiers do not match this run")
    if manifest.get("ranking_provenance") != expected_ranking_provenance:
        raise ValueError(
            "Frozen TCP ranking provenance does not match the canonical KG, "
            "SearchPolicy, NeuroDiscovery configuration, or score bundle"
        )

    orders_path = Path(manifest["orders"]["path"])
    if sha256_file(orders_path) != manifest["orders"]["sha256"]:
        raise ValueError("Frozen TCP order archive failed SHA-256 verification")
    archive = np.load(orders_path, allow_pickle=False)
    frozen_orders: list[tuple[int, str, np.ndarray]] = []
    for record in manifest["orders"]["records"]:
        order = np.asarray(archive[record["array_key"]], dtype=np.int32)
        if hashlib.sha256(order.tobytes()).hexdigest() != record["order_sha256"]:
            raise ValueError(
                f"Frozen order failed verification: {record['array_key']}"
            )
        frozen_orders.append(
            (int(record["trial"]), str(record["method"]), order)
        )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    manifest["reused"] = True
    return frozen_orders, manifest


def filter_external_compatibility(
    scored: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only exact TCP/external source-level ROI and feature intersections."""

    tcp_sources = {
        str(source): {
            "roi_indices": frozenset(
                pd.to_numeric(group["roi_index"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            ),
            "features": frozenset(group["feature"].astype(str).unique()),
        }
        for source, group in scored.groupby("source", sort=False)
    }
    audit_rows: list[dict[str, Any]] = []
    keep_indices: list[int] = []
    for (dataset, source), group in external.groupby(
        ["dataset", "source"],
        sort=False,
    ):
        tcp = tcp_sources.get(str(source))
        external_rois = frozenset(
            pd.to_numeric(group["roi_index"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        external_features = frozenset(group["feature"].astype(str).unique())
        roi_indices_match = tcp is not None and external_rois == tcp["roi_indices"]
        features_match = tcp is not None and external_features == tcp["features"]
        included = bool(roi_indices_match and features_match)
        if included:
            keep_indices.extend(group.index.tolist())
        audit_rows.append(
            {
                "dataset": str(dataset),
                "source": str(source),
                "tcp_roi_count": len(tcp["roi_indices"]) if tcp else 0,
                "external_roi_count": len(external_rois),
                "roi_indices_match": bool(roi_indices_match),
                "tcp_feature_count": len(tcp["features"]) if tcp else 0,
                "external_feature_count": len(external_features),
                "features_match": bool(features_match),
                "included": included,
                "exclusion_reason": (
                    ""
                    if included
                    else "ROI index set differs from TCP"
                    if not roi_indices_match
                    else "feature set differs from TCP"
                ),
            }
        )
    compatible = external.loc[sorted(keep_indices)].copy()
    return compatible, pd.DataFrame(audit_rows)


def external_run_audit(
    external_root: Path,
    run_name: str,
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for dataset in external_labels.PRIMARY_DISEASES_BY_DATASET:
        manifest_path = (
            external_root
            / dataset
            / run_name
            / "case1_exhaustive_full_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        diagnosis_path = Path(manifest["diagnosis"])
        diagnosis = pd.read_csv(diagnosis_path, low_memory=False)
        mean_fd_available = bool(
            pd.to_numeric(
                diagnosis.get(
                    "mean_fd",
                    pd.Series(np.nan, index=diagnosis.index),
                ),
                errors="coerce",
            ).notna().any()
        )
        covariates = list(manifest.get("covariates", []))
        if mean_fd_available and "mean_fd_z" not in covariates:
            raise ValueError(
                f"{dataset} has mean_fd values but its external model omitted mean_fd_z"
            )
        audits[dataset] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "subjects": int(manifest["subjects"]),
            "controls": int(manifest["controls"]),
            "diseases_computed": manifest["diseases"],
            "covariates": covariates,
            "mean_fd_available": mean_fd_available,
            "atlases": manifest["atlases"],
            "features": manifest["fmri_features"],
            "n_tests": int(manifest["n_tests"]),
            "qc": (
                "Successful preprocessing plus complete ROI, correlation, and "
                "partial-correlation derivatives and finite model inputs."
            ),
        }
    return audits


def build_external_outcomes(
    scored: pd.DataFrame,
    external: pd.DataFrame,
    *,
    q_threshold: float,
    min_abs_d: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = scored.copy()
    frozen["rank"] = np.arange(1, len(frozen) + 1)
    long, labels = external_labels.classify_results(
        frozen,
        external,
        q_threshold=q_threshold,
        min_abs_d=min_abs_d,
    )
    return long, labels


def outcome_arrays(
    candidate_ids: np.ndarray,
    labels: pd.DataFrame,
) -> dict[str, np.ndarray]:
    label_by_id = labels.set_index("candidate_id")["external_label"].astype(str)
    aligned = pd.Series(candidate_ids).map(label_by_id).fillna("not_executable")
    arrays = {
        name: aligned.eq(name).to_numpy(dtype=bool) for name in OUTCOME_NAMES
    }
    arrays["executable"] = aligned.ne("not_executable").to_numpy(dtype=bool)
    return arrays


def metrics_for_order(
    method: str,
    trial: int,
    order: np.ndarray,
    arrays: dict[str, np.ndarray],
    budgets: list[int],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    ordered = {name: values[order] for name, values in arrays.items()}
    cumulative = {
        name: np.cumsum(values.astype(np.int64)) for name, values in ordered.items()
    }
    total_confirmed = int(arrays["confirmed"].sum())
    rows: list[dict[str, Any]] = []

    for scope in ("tcp_budget", "executable_budget"):
        if scope == "tcp_budget":
            scoped_indices = np.arange(len(order), dtype=int)
        else:
            scoped_indices = np.flatnonzero(ordered["executable"])
        for requested_budget in budgets:
            if requested_budget > len(scoped_indices):
                continue
            selected = scoped_indices[:requested_budget]
            if len(selected) == 0:
                continue
            last = int(selected[-1])
            counts = {
                name: int(cumulative[name][last])
                for name in (*OUTCOME_NAMES, "executable")
            }
            confirmed = counts["confirmed"]
            executable = counts["executable"]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "label": method_label(method),
                    "trial": trial,
                    "scope": scope,
                    "budget": int(requested_budget),
                    "tcp_rank_reached": last + 1,
                    **{f"n_{name}": value for name, value in counts.items()},
                    "external_precision": (
                        confirmed / executable if executable else np.nan
                    ),
                    "external_recall": (
                        confirmed / total_confirmed if total_confirmed else np.nan
                    ),
                    "external_gt_total": total_confirmed,
                    "executable_fraction": (
                        executable / requested_budget
                        if scope == "tcp_budget"
                        else 1.0
                    ),
                }
            )
    return rows


def recall_cost_rows(
    method: str,
    trial: int,
    order: np.ndarray,
    confirmed: np.ndarray,
    executable: np.ndarray,
    targets: tuple[float, ...],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    ordered_confirmed = confirmed[order]
    ordered_executable = executable[order]
    confirmed_positions = np.flatnonzero(ordered_confirmed) + 1
    executable_order = np.flatnonzero(ordered_executable)
    executable_confirmed_positions = (
        np.flatnonzero(ordered_confirmed[executable_order]) + 1
    )
    total = int(confirmed.sum())
    rows: list[dict[str, Any]] = []
    for target in targets:
        need = int(math.ceil(total * target))
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "label": method_label(method),
                "trial": trial,
                "recall_target": target,
                "confirmed_needed": need,
                "external_gt_total": total,
                "tcp_experiments_required": (
                    int(confirmed_positions[need - 1])
                    if need > 0 and len(confirmed_positions) >= need
                    else np.nan
                ),
                "executable_experiments_required": (
                    int(executable_confirmed_positions[need - 1])
                    if need > 0 and len(executable_confirmed_positions) >= need
                    else np.nan
                ),
            }
        )
    return rows


def aggregate_with_variance(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    excluded = set(group_columns) | {"trial", "label"}
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        method = row.get("method")
        if method:
            row["label"] = method_label(str(method))
        row["n_trials"] = int(group["trial"].nunique())
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_variance"] = variance
            row[f"{column}_sd"] = math.sqrt(variance)
        rows.append(row)
    return pd.DataFrame(rows)


def build_external_sota_gate(
    metric_summary: pd.DataFrame,
    recall_cost_summary: pd.DataFrame,
    *,
    budgets: list[int],
    recall_targets: list[float],
) -> dict[str, Any]:
    expected_methods = set(PRIMARY_METHODS)
    pooled_yield = metric_summary[
        (metric_summary["dataset"] == "pooled")
        & (metric_summary["scope"] == "tcp_budget")
        & metric_summary["budget"].isin(budgets)
    ].copy()
    pooled_cost = recall_cost_summary[
        (recall_cost_summary["dataset"] == "pooled")
        & recall_cost_summary["recall_target"].isin(recall_targets)
    ].copy()
    for frame, endpoint in (
        (pooled_yield, "external yield"),
        (pooled_cost, "external recall cost"),
    ):
        if set(frame["method"].astype(str)) != expected_methods:
            raise RuntimeError(f"{endpoint} SOTA gate has an incomplete method set")
        if not (pd.to_numeric(frame["n_trials"], errors="coerce") == 3).all():
            raise RuntimeError(f"{endpoint} SOTA gate requires three trials")

    yield_gates: dict[str, Any] = {}
    for budget in budgets:
        endpoint = pooled_yield[pooled_yield["budget"] == budget]
        if len(endpoint) != len(expected_methods):
            raise RuntimeError(f"external yield endpoint {budget} is incomplete")
        nd = float(
            endpoint.loc[
                endpoint["method"] == "neurodiscovery", "n_confirmed_mean"
            ].iloc[0]
        )
        best_baseline = float(
            endpoint.loc[
                endpoint["method"] != "neurodiscovery", "n_confirmed_mean"
            ].max()
        )
        yield_gates[str(budget)] = {
            "neurodiscovery_mean": nd,
            "best_baseline_mean": best_baseline,
            "passed": nd > best_baseline,
        }

    recall_gates: dict[str, Any] = {}
    for target in recall_targets:
        endpoint = pooled_cost[
            np.isclose(pooled_cost["recall_target"].astype(float), target)
        ]
        if len(endpoint) != len(expected_methods):
            raise RuntimeError(f"external recall endpoint {target} is incomplete")
        nd = float(
            endpoint.loc[
                endpoint["method"] == "neurodiscovery",
                "tcp_experiments_required_mean",
            ].iloc[0]
        )
        best_baseline = float(
            endpoint.loc[
                endpoint["method"] != "neurodiscovery",
                "tcp_experiments_required_mean",
            ].min()
        )
        recall_gates[f"{target:.2f}"] = {
            "neurodiscovery_mean": nd,
            "best_baseline_mean": best_baseline,
            "passed": nd < best_baseline,
        }

    achieved = all(
        record["passed"]
        for record in (*yield_gates.values(), *recall_gates.values())
    )
    return {
        "schema_version": "case1-neurodiscovery-external-sota-gate.v1",
        "definition": EXTERNAL_SOTA_DEFINITION,
        "dataset": "pooled",
        "yield_gates": yield_gates,
        "recall_cost_gates": recall_gates,
        "external_sota_achieved": achieved,
    }


def paired_p_values(
    metrics: pd.DataFrame,
    recall_costs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    neuro = "neurodiscovery"
    available_methods = set(metrics["method"].astype(str).unique())
    baselines = list(
        dict.fromkeys(
            method
            for method in (*comparison.PRIMARY_BASELINE_METHODS, *NATIVE_METHOD_LABELS)
            if method in available_methods
        )
    )
    for (dataset, scope, budget), group in metrics.groupby(
        ["dataset", "scope", "budget"], sort=False
    ):
        nd = group[group["method"] == neuro].set_index("trial")
        for baseline in baselines:
            base = group[group["method"] == baseline].set_index("trial")
            joined = nd[["n_confirmed"]].join(
                base[["n_confirmed"]],
                how="inner",
                lsuffix="_neuro",
                rsuffix="_baseline",
            )
            rows.append(
                paired_test_row(
                    joined["n_confirmed_neuro"].to_numpy(float),
                    joined["n_confirmed_baseline"].to_numpy(float),
                    alternative="greater",
                    dataset=dataset,
                    comparison_type="same_experiments_confirmed",
                    scope=scope,
                    value=float(budget),
                    baseline=baseline,
                )
            )

    for (dataset, target), group in recall_costs.groupby(
        ["dataset", "recall_target"], sort=False
    ):
        nd = group[group["method"] == neuro].set_index("trial")
        for baseline in baselines:
            base = group[group["method"] == baseline].set_index("trial")
            joined = nd[["tcp_experiments_required"]].join(
                base[["tcp_experiments_required"]],
                how="inner",
                lsuffix="_neuro",
                rsuffix="_baseline",
            ).dropna()
            rows.append(
                paired_test_row(
                    joined["tcp_experiments_required_neuro"].to_numpy(float),
                    joined["tcp_experiments_required_baseline"].to_numpy(float),
                    alternative="less",
                    dataset=dataset,
                    comparison_type="same_recall_tcp_experiments",
                    scope="tcp_budget",
                    value=float(target),
                    baseline=baseline,
                )
            )
    return pd.DataFrame(rows)


def paired_test_row(
    neuro: np.ndarray,
    baseline_values: np.ndarray,
    *,
    alternative: str,
    dataset: str,
    comparison_type: str,
    scope: str,
    value: float,
    baseline: str,
) -> dict[str, Any]:
    differences = neuro - baseline_values
    if len(differences) == 0:
        statistic = np.nan
        p_value = np.nan
    elif np.allclose(differences, 0.0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(neuro, baseline_values, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    return {
        "dataset": dataset,
        "comparison_type": comparison_type,
        "scope": scope,
        "budget_or_recall_target": value,
        "neurodiscovery_method": "neurodiscovery",
        "baseline_method": baseline,
        "n_paired_seeds": int(len(differences)),
        "neurodiscovery_mean": (
            float(np.mean(neuro)) if len(neuro) else np.nan
        ),
        "baseline_mean": (
            float(np.mean(baseline_values)) if len(baseline_values) else np.nan
        ),
        "mean_paired_difference": (
            float(np.mean(differences)) if len(differences) else np.nan
        ),
        "wilcoxon_statistic": statistic,
        "p_value_one_sided": p_value,
        "alternative": alternative,
    }


def dataset_outcome_arrays(
    candidate_ids: np.ndarray,
    long: pd.DataFrame,
    dataset: str,
) -> dict[str, np.ndarray]:
    subset = long[long["dataset"] == dataset]
    outcome_by_id = subset.set_index("candidate_id")["external_outcome"].astype(str)
    aligned = pd.Series(candidate_ids).map(outcome_by_id).fillna("not_executable")
    arrays = {
        name: aligned.eq(name).to_numpy(dtype=bool)
        for name in ("confirmed", "falsified", "not_confirmed")
    }
    arrays["conflicting"] = np.zeros(len(candidate_ids), dtype=bool)
    arrays["executable"] = aligned.ne("not_executable").to_numpy(dtype=bool)
    return arrays


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(values[valid], kind="mergesort")]
    ranks = np.arange(1, len(order) + 1, dtype=float)
    raw = values[order] * len(order) / ranks
    corrected = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted[order] = np.clip(corrected, 0.0, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def fixed_effect_meta_analysis(
    long: pd.DataFrame,
    *,
    q_threshold: float,
    min_abs_d: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long.groupby("candidate_id", sort=False):
        effects = pd.to_numeric(
            group["external_adjusted_residual_d"], errors="coerce"
        ).to_numpy(float)
        n_case = pd.to_numeric(
            group["external_n_case"], errors="coerce"
        ).to_numpy(float)
        n_control = pd.to_numeric(
            group["external_n_control"], errors="coerce"
        ).to_numpy(float)
        valid = (
            np.isfinite(effects)
            & np.isfinite(n_case)
            & np.isfinite(n_control)
            & (n_case > 1)
            & (n_control > 1)
        )
        effects = effects[valid]
        n_case = n_case[valid]
        n_control = n_control[valid]
        datasets = group.loc[group.index[valid], "dataset"].astype(str).tolist()
        if not len(effects):
            continue
        variances = (
            (n_case + n_control) / (n_case * n_control)
            + effects**2 / (2.0 * np.maximum(n_case + n_control - 2.0, 1.0))
        )
        weights = 1.0 / np.maximum(variances, np.finfo(float).eps)
        pooled_effect = float(np.sum(weights * effects) / np.sum(weights))
        pooled_se = float(math.sqrt(1.0 / np.sum(weights)))
        z_value = pooled_effect / pooled_se
        p_value = float(2.0 * norm.sf(abs(z_value)))
        q_statistic = float(np.sum(weights * (effects - pooled_effect) ** 2))
        degrees_freedom = max(0, len(effects) - 1)
        i_squared = (
            max(0.0, (q_statistic - degrees_freedom) / q_statistic)
            if q_statistic > 0 and degrees_freedom > 0
            else 0.0
        )
        base = group.iloc[0]
        tcp_effect = float(base["tcp_adjusted_residual_d"])
        rows.append(
            {
                "candidate_id": candidate,
                "disease": str(base["disease"]),
                "modality": str(base["modality"]),
                "source": str(base["source"]),
                "feature": str(base["feature"]),
                "roi_index": int(base["roi_index"]),
                "roi_name": str(base.get("roi_name") or ""),
                "datasets": json.dumps(datasets),
                "n_datasets": len(effects),
                "meta_eligible": len(effects) >= 2,
                "tcp_adjusted_residual_d": tcp_effect,
                "fixed_effect_d": pooled_effect,
                "fixed_effect_se": pooled_se,
                "z_value": z_value,
                "p_value": p_value,
                "q_statistic": q_statistic,
                "i_squared": i_squared,
                "direction_concordant_with_tcp": (
                    np.sign(pooled_effect) == np.sign(tcp_effect)
                ),
            }
        )
    meta = pd.DataFrame(rows)
    if meta.empty:
        return meta
    meta["q_fdr_disease"] = np.nan
    eligible = meta["meta_eligible"].astype(bool)
    for _disease, indices in meta[eligible].groupby("disease").groups.items():
        meta.loc[indices, "q_fdr_disease"] = bh_adjust(
            meta.loc[indices, "p_value"]
        )
    meta["meta_confirmed"] = (
        meta["meta_eligible"].astype(bool)
        & (pd.to_numeric(meta["q_fdr_disease"], errors="coerce") < q_threshold)
        & (meta["fixed_effect_d"].abs() > min_abs_d)
        & meta["direction_concordant_with_tcp"].astype(bool)
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--current-state", type=Path, required=True)
    parser.add_argument("--generation-first-dir", type=Path, required=True)
    parser.add_argument(
        "--neurodiscovery-config",
        type=Path,
        default=None,
        help=(
            "Optional frozen NeuroDiscovery JSON selected without external outcomes. "
            "When omitted, use the audited default configuration."
        ),
    )
    parser.add_argument(
        "--score-components",
        type=Path,
        default=None,
        help=(
            "Optional frozen candidate score table used by the TCP ranking stage. "
            "Must be paired with --score-components-manifest."
        ),
    )
    parser.add_argument(
        "--score-components-manifest",
        type=Path,
        default=None,
        help=(
            "Optional audit manifest paired with --score-components. When both "
            "score-component arguments are omitted, no auxiliary score is activated."
        ),
    )
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--external-run-name", default="primary")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-frozen-rankings",
        type=Path,
        default=None,
        help=(
            "Optional frozen_tcp_rankings directory from a prior TCP-only stage. "
            "All hashes are verified before external results are opened."
        ),
    )
    parser.add_argument(
        "--neurodiscovery-overlay-dir",
        type=Path,
        default=None,
        help=(
            "Optional internal-validation experimental_overlays directory. Each "
            "TCP-only NeuroDiscovery order is recovered only after its hash chain, "
            "seed/trial identity, uniqueness, and full candidate coverage pass."
        ),
    )
    parser.add_argument(
        "--neurodiscovery-formal-root",
        type=Path,
        default=None,
        help=(
            "Optional Case 1 formal NeuroDiscovery root containing commitment-"
            "audited seed manifests and frozen_order_indices.npy files."
        ),
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260616)
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--min-abs-d", type=float, default=0.15)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        default=DEFAULT_RECALL_TARGETS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking_sources = sum(
        value is not None
        for value in (
            args.reuse_frozen_rankings,
            args.neurodiscovery_overlay_dir,
            args.neurodiscovery_formal_root,
        )
    )
    if ranking_sources > 1:
        raise ValueError(
            "reuse, overlay, and formal NeuroDiscovery ranking sources are "
            "mutually exclusive"
        )
    canonical_release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id=comparison.CASE1_CASE_STUDY_ID,
    )
    policy_path = args.generation_first_dir / "case1_search_policies.jsonl"
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)

    tcp_results = comparison.load_results(args.all_tests, args.gt_top_frac)
    if args.neurodiscovery_config is None:
        neurodiscovery_config = comparison.Case1NeuroDiscoveryConfig()
        neurodiscovery_config_path = None
        neurodiscovery_config_file_sha256 = None
    else:
        neurodiscovery_config = comparison.Case1NeuroDiscoveryConfig.from_json(
            args.neurodiscovery_config
        )
        neurodiscovery_config_path = str(args.neurodiscovery_config)
        neurodiscovery_config_file_sha256 = sha256_file(args.neurodiscovery_config)
    kg_index = comparison.load_kg_index(
        args.kg,
        comparison.kg_query_terms_for_candidates(tcp_results),
    )
    scored = comparison.add_generator_scores(
        tcp_results,
        kg_index,
        args.seed,
        config=neurodiscovery_config,
    )
    if (args.score_components is None) != (args.score_components_manifest is None):
        raise ValueError(
            "--score-components and --score-components-manifest must be provided together"
        )
    if args.score_components is None:
        score_component_audit = comparison.embedded_score_component_audit(scored)
    else:
        scored, score_component_audit = comparison.load_score_component_bundle(
            scored,
            table_path=args.score_components,
            manifest_path=args.score_components_manifest,
        )
    generation_policies = comparison.load_search_policies(policy_path)
    ranking_provenance = {
        "canonical_release_sha256": sha256_payload(canonical_release),
        "search_policies_sha256": sha256_file(policy_path),
        "neurodiscovery_config_sha256": sha256_payload(
            neurodiscovery_config.to_dict()
        ),
        "score_component_audit_sha256": sha256_payload(score_component_audit),
    }
    if args.neurodiscovery_formal_root is not None:
        formal_root = args.neurodiscovery_formal_root.resolve()
        formal_records = {
            "input_manifest": sha256_file(formal_root / "input_manifest.json"),
            "seed_manifests": {
                str(trial): sha256_file(
                    formal_root
                    / "neurodiscovery"
                    / f"seed_{trial:02d}"
                    / "seed_manifest.json"
                )
                for trial in range(args.trials)
            },
        }
        ranking_provenance["formal_neurodiscovery"] = formal_records

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_release_manifest(
        args.out_dir / "canonical_kg_release.json",
        canonical_release,
    )
    if args.reuse_frozen_rankings is not None:
        frozen_orders, frozen_manifest = load_frozen_tcp_rankings(
            args.reuse_frozen_rankings,
            scored["candidate_id"].astype(str).to_numpy(),
            expected_ranking_provenance=ranking_provenance,
        )
    else:
        frozen_orders, frozen_manifest = freeze_tcp_rankings(
            scored,
            generation_policies,
            out_dir=args.out_dir,
            all_tests=args.all_tests,
            kg=args.kg,
            search_policy_path=policy_path,
            n_trials=args.trials,
            seed=args.seed,
            neurodiscovery_config=neurodiscovery_config,
            ranking_provenance=ranking_provenance,
            neurodiscovery_overlay_dir=args.neurodiscovery_overlay_dir,
            neurodiscovery_formal_root=args.neurodiscovery_formal_root,
        )
    frozen_orders = [
        item for item in frozen_orders if item[1] in PRIMARY_METHODS
    ]
    internal_sota_evidence = (
        load_internal_sota_evidence(args.neurodiscovery_formal_root.resolve())
        if args.neurodiscovery_formal_root is not None
        else None
    )

    # External files are first touched only after every TCP ranking has been
    # materialized and hashed above.
    external = external_labels.load_external_results(
        args.external_root,
        args.external_run_name,
    )
    external, compatibility_audit = filter_external_compatibility(scored, external)
    run_audit = external_run_audit(
        args.external_root,
        args.external_run_name,
    )
    long, pooled_labels = build_external_outcomes(
        scored,
        external,
        q_threshold=args.q_threshold,
        min_abs_d=args.min_abs_d,
    )
    meta_effects = fixed_effect_meta_analysis(
        long,
        q_threshold=args.q_threshold,
        min_abs_d=args.min_abs_d,
    )
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    arrays_by_dataset = {
        "pooled": outcome_arrays(candidate_ids, pooled_labels),
        **{
            dataset: dataset_outcome_arrays(candidate_ids, long, dataset)
            for dataset in sorted(long["dataset"].unique())
        },
    }
    budgets = sorted(
        {int(value) for value in args.budgets if 0 < int(value) <= len(scored)}
    )
    recall_targets = sorted(
        {float(value) for value in args.recall_targets if 0 < float(value) <= 1}
    )
    if not recall_targets:
        raise ValueError("at least one external recall target in (0, 1] is required")
    metric_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for trial, method, order in frozen_orders:
        for dataset, arrays in arrays_by_dataset.items():
            metric_rows.extend(
                metrics_for_order(
                    method,
                    trial,
                    order,
                    arrays,
                    budgets,
                    dataset=dataset,
                )
            )
            cost_rows.extend(
                recall_cost_rows(
                    method,
                    trial,
                    order,
                    arrays["confirmed"],
                    arrays["executable"],
                    recall_targets,
                    dataset=dataset,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    recall_costs = pd.DataFrame(cost_rows)
    metric_summary = aggregate_with_variance(
        metrics,
        ["dataset", "method", "scope", "budget"],
    )
    recall_cost_summary = aggregate_with_variance(
        recall_costs,
        ["dataset", "method", "recall_target"],
    )
    p_values = paired_p_values(metrics, recall_costs)
    external_sota_gate = build_external_sota_gate(
        metric_summary,
        recall_cost_summary,
        budgets=budgets,
        recall_targets=recall_targets,
    )

    compatibility_audit.to_csv(
        args.out_dir / "case1_external_compatibility_audit.csv",
        index=False,
    )
    long.to_csv(args.out_dir / "case1_external_results_long.csv", index=False)
    pooled_labels.to_csv(
        args.out_dir / "case1_external_candidate_labels.csv", index=False
    )
    meta_effects.to_csv(
        args.out_dir / "case1_external_fixed_effect_meta_analysis.csv",
        index=False,
    )
    metrics.to_csv(
        args.out_dir / "case1_external_metrics_by_trial.csv", index=False
    )
    metric_summary.to_csv(
        args.out_dir / "case1_external_metrics_summary.csv", index=False
    )
    recall_costs.to_csv(
        args.out_dir / "case1_external_recall_cost_by_trial.csv", index=False
    )
    recall_cost_summary.to_csv(
        args.out_dir / "case1_external_recall_cost_summary.csv", index=False
    )
    p_values.to_csv(
        args.out_dir / "case1_external_comparison_p_values.csv", index=False
    )
    external_sota_gate_path = args.out_dir / "external_sota_gate.json"
    external_sota_gate_path.write_text(
        json.dumps(external_sota_gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    overall_sota_path: Path | None = None
    overall_sota_gate: dict[str, Any] | None = None
    if internal_sota_evidence is not None:
        overall_sota_gate = {
            "schema_version": "case1-neurodiscovery-overall-sota-gate.v1",
            "definition": (
                "Both the preregistered internal and pooled independent-cohort "
                "SOTA gates must pass using the same three frozen rankings."
            ),
            "internal_sota_achieved": True,
            "external_sota_achieved": bool(
                external_sota_gate["external_sota_achieved"]
            ),
            "overall_sota_achieved": bool(
                external_sota_gate["external_sota_achieved"]
            ),
            "internal_evidence": internal_sota_evidence,
            "external_gate": external_sota_gate,
        }
        overall_sota_path = args.out_dir / "overall_sota_gate.json"
        overall_sota_path.write_text(
            json.dumps(overall_sota_gate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    external_files = {
        dataset: (
            args.external_root
            / dataset
            / args.external_run_name
            / external_labels.DEFAULT_RESULT_NAME
        )
        for dataset in external_labels.PRIMARY_DISEASES_BY_DATASET
    }
    manifest = {
        "schema_version": "case1-external-method-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tcp_all_tests": str(args.all_tests),
        "tcp_all_tests_sha256": sha256_file(args.all_tests),
        "kg": str(args.kg),
        "kg_sha256": sha256_file(args.kg),
        "canonical_kg_release": canonical_release,
        "case_study_membership": {
            "schema_version": "case_study_membership.v2",
            "case_study_id": comparison.CASE1_CASE_STUDY_ID,
            "canonical_claim_field": "claim_case_study_ids",
            "global_support_weight": comparison.CASE1_GLOBAL_SUPPORT_WEIGHT,
            "case_study_support_weight": comparison.CASE1_SCOPED_SUPPORT_WEIGHT,
            "index_stats": kg_index.stats,
        },
        "search_policies": str(policy_path),
        "search_policies_sha256": sha256_file(policy_path),
        "neurodiscovery_config": neurodiscovery_config.to_dict(),
        "neurodiscovery_config_path": neurodiscovery_config_path,
        "neurodiscovery_config_sha256": sha256_payload(
            neurodiscovery_config.to_dict()
        ),
        "neurodiscovery_config_file_sha256": neurodiscovery_config_file_sha256,
        "score_component_bundle": score_component_audit,
        "external_sota_gate": {
            "path": str(external_sota_gate_path),
            "sha256": sha256_file(external_sota_gate_path),
            **external_sota_gate,
        },
        "overall_sota_gate": (
            {
                "path": str(overall_sota_path),
                "sha256": sha256_file(overall_sota_path),
                **overall_sota_gate,
            }
            if overall_sota_path is not None and overall_sota_gate is not None
            else None
        ),
        "external_results": {
            dataset: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for dataset, path in external_files.items()
        },
        "ranking_policy": (
            "Every method is ranked using TCP and the frozen KG only. All TCP "
            "orders are materialized and hashed before any external result file "
            "is opened. External labels never influence generation, ranking, "
            "parameter tuning, or NeuroDiscovery score updates."
        ),
        "frozen_tcp_rankings": frozen_manifest,
        "matching_key": list(external_labels.KEY_COLUMNS),
        "compatibility_policy": (
            "An external atlas source is executable only when its complete ROI "
            "index set and imaging-feature set exactly match TCP."
        ),
        "compatibility_audit": {
            "path": str(args.out_dir / "case1_external_compatibility_audit.csv"),
            "sha256": sha256_file(
                args.out_dir / "case1_external_compatibility_audit.csv"
            ),
            "excluded_sources_by_dataset": {
                dataset: sorted(group.loc[~group["included"], "source"].astype(str))
                for dataset, group in compatibility_audit.groupby(
                    "dataset",
                    sort=True,
                )
            },
        },
        "external_run_audit": run_audit,
        "protocol": {
            "discovery_dataset": "TCP",
            "external_datasets": ["ucla", "cobre", "hcpep", "adhd200"],
            "external_validation_only": True,
            "external_feedback_to_tcp_ranking": False,
            "q_fdr_threshold": args.q_threshold,
            "minimum_absolute_cohens_d": args.min_abs_d,
            "direction_must_match_tcp": True,
            "fdr_scope": "within each external dataset and disease",
            "primary_cost_axis": "TCP experiments evaluated",
            "secondary_fairness_axis": "equal externally executable hypotheses",
        },
        "primary_diseases_by_dataset": {
            dataset: sorted(diseases)
            for dataset, diseases in external_labels.PRIMARY_DISEASES_BY_DATASET.items()
        },
        "confirmation_definition": {
            "q_fdr_disease_less_than": args.q_threshold,
            "minimum_absolute_cohens_d": args.min_abs_d,
            "direction_must_match_tcp": True,
        },
        "external_gt_definition": (
            "Candidates confirmed in at least one primary external dataset and "
            "not significantly direction-discordant in any primary dataset."
        ),
        "methods": sorted({method for _, method, _ in frozen_orders}),
        "supplementary_methods_excluded": ["openscholar_rag"],
        "trials": args.trials,
        "seed_base": args.seed,
        "budgets": budgets,
        "recall_targets": recall_targets,
        "dispersion": (
            f"sample variance across {args.trials} independent ranking seeds"
        ),
        "external_gt_totals": {
            dataset: int(arrays["confirmed"].sum())
            for dataset, arrays in arrays_by_dataset.items()
        },
        "meta_analysis": {
            "model": "inverse-variance fixed effect",
            "minimum_datasets": 2,
            "fdr_scope": "within disease across meta-eligible candidates",
            "n_meta_eligible": int(
                meta_effects.get(
                    "meta_eligible", pd.Series(dtype=bool)
                ).sum()
            ),
            "n_meta_confirmed": int(
                meta_effects.get(
                    "meta_confirmed", pd.Series(dtype=bool)
                ).sum()
            ),
        },
    }
    (args.out_dir / "case1_external_method_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
