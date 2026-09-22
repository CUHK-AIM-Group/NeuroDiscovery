"""Run audited, full-space NeuroDiscovery for Case Study 1.

The workflow is intentionally split into separate commands:

1. ``prepare`` writes an outcome-blind scoring table, a feedback-only vault,
   and GT labels into separate hashed files.
2. ``run-seed`` reads only the public table and feedback vault. Every batch is
   hash-committed before the vault returns its outcomes.
3. ``evaluate`` opens GT labels only after all requested seed rankings have
   been frozen, then compares NeuroDiscovery with the canonical baselines.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    validate_canonical_kg_release,
)
from core.scripts.case1_method_comparison import (
    add_generator_scores,
    closed_loop_neurodiscovery_order,
    kg_query_terms_for_candidates,
    load_kg_index,
    load_results,
    neurodiscovery_score_arrays,
)
from core.scripts.case1_neurodiscovery_config import Case1NeuroDiscoveryConfig
from core.scripts.case1_search_policy import (
    classify_observed_feedback,
    compile_policy_order,
    policy_from_payload,
)
from core.scripts.case_study_score_components import (
    candidate_id_sha256,
    load_score_component_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SOURCE_RELATIVE_PATHS = (
    "core/scripts/run_case1_neurodiscovery_formal.py",
    "core/scripts/case1_method_comparison.py",
    "core/scripts/case1_neurodiscovery_config.py",
    "core/scripts/case1_search_policy.py",
    "core/scripts/case_study_closed_loop_engine.py",
)
CASE_STUDY_ID = "case1_transdiagnostic"
DEFAULT_ALL_TESTS = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260616_full_main_noboot\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_KG = ROOT / "neurooracle/data/full_v2/knowledge_graph.json"
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
DEFAULT_STATE = ROOT / "neurooracle/data/full_v2/CURRENT_STATE.json"
DEFAULT_SCORE_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_autoresearch_comparison"
    r"\20260821_kg89e40d_direct_closed_loop\score_components"
)
DEFAULT_POLICY_FILE = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_autoresearch_comparison"
    r"\20260823_kg89e40d_fullspace_freshanchors_3seeds_v1"
    r"\case1_search_policies.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_autoresearch_comparison"
    r"\20260824_kg89e40d_neurodiscovery_true_closed_loop_3seeds_v1"
)
DEFAULT_SCORE_TABLE = DEFAULT_SCORE_ROOT / "score_components.csv"
DEFAULT_SCORE_MANIFEST = DEFAULT_SCORE_ROOT / "score_components.manifest.json"

BUDGETS = (5_000, 10_000, 50_000, 100_000, 200_000)
RECALL_TARGETS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
STATIC_SCORE_SEED = 260810
FORMAL_RNG_SEED_BASE = 20260826
FORMAL_SEEDS = (0, 1, 2)
BASELINE_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)
SOTA_DEFINITION = (
    "NeuroDiscovery 3-seed mean must exceed every baseline mean at all five "
    "discovery-yield budgets and require fewer experiments at all six recall "
    "targets."
)
EXTERNAL_SOTA_DEFINITION = (
    "On pooled independent-cohort confirmation labels, NeuroDiscovery's "
    "3-seed mean must exceed every baseline mean at all five frozen TCP "
    "budgets and require fewer frozen TCP experiments at all six external "
    "recall targets."
)
INPUT_SCHEMA = "case1-neurodiscovery-formal-inputs.v1"
SEED_SCHEMA = "case1-neurodiscovery-formal-seed.v1"
EVALUATION_SCHEMA = "case1-neurodiscovery-formal-evaluation.v1"

HIDDEN_SCORING_COLUMNS = frozenset(
    {
        "adjusted_residual_d",
        "abs_adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "q_fdr_disease",
        "q_fdr_modality",
        "execution_succeeded",
        "gt_rank",
        "is_gt_top",
        "is_strict_fdr",
    }
)
PUBLIC_COLUMNS = (
    "candidate_id",
    "disease",
    "modality",
    "source",
    "roi_index",
    "roi_name",
    "anatomy_key",
    "anatomy_full",
    "hemisphere",
    "network",
    "structure_class",
    "feature",
    "feature_family",
    "map_group",
    "roi_key",
    "score_random",
    "score_neurodiscovery_global_base",
    "score_case_study_support_base",
    "score_feature_support_global",
    "score_feature_support_scoped",
    "score_kge",
    "score_novelty",
    "score_critic",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_order_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verify_artifact(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed.casefold() != str(record["sha256"]).casefold():
        raise RuntimeError(f"artifact hash mismatch: {path}")
    return path


def freeze_protocol_snapshot(root: Path) -> dict[str, Any]:
    """Freeze the exact closed-loop implementation before the first formal seed."""

    snapshot_dir = root / "protocol_snapshot"
    manifest_path = root / "protocol_snapshot_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("schema_version") != "case1-neurodiscovery-protocol.v1":
            raise RuntimeError("formal protocol snapshot schema is incompatible")
        sources = manifest.get("sources") or {}
        if set(sources) != set(PROTOCOL_SOURCE_RELATIVE_PATHS):
            raise RuntimeError("formal protocol snapshot source set is incomplete")
        for relative_path, record in sources.items():
            verify_artifact(record["snapshot"])
            current_path = ROOT / relative_path
            if sha256_file(current_path) != record["source_sha256"]:
                raise RuntimeError(
                    "closed-loop source changed after protocol freeze: "
                    f"{relative_path}"
                )
        return artifact(manifest_path)

    if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
        raise RuntimeError("partial formal protocol snapshot requires a new root")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    for relative_path in PROTOCOL_SOURCE_RELATIVE_PATHS:
        source = ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = snapshot_dir / source.name
        shutil.copy2(source, destination)
        sources[relative_path] = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "snapshot": artifact(destination),
        }
    write_json(
        manifest_path,
        {
            "schema_version": "case1-neurodiscovery-protocol.v1",
            "created_at": utc_now(),
            "sources": sources,
        },
    )
    return artifact(manifest_path)


def freeze_evaluation_design(
    root: Path,
    *,
    input_manifest_sha256: str,
    candidate_id_sha256_value: str,
    config_artifact: Mapping[str, Any],
    source_config_artifact: Mapping[str, Any],
    tuning_manifest_artifact: Mapping[str, Any],
    protocol_snapshot_manifest: Mapping[str, Any],
    baseline_policies: Path,
) -> dict[str, Any]:
    """Preregister seeds, endpoints, baselines, and the SOTA gate before ranking."""

    design_path = root / "formal_design.json"
    frozen_fields = {
        "schema_version": "case1-neurodiscovery-formal-design.v1",
        "input_manifest_sha256": str(input_manifest_sha256),
        "candidate_id_sha256": str(candidate_id_sha256_value),
        "formal_seeds": list(FORMAL_SEEDS),
        "rng_seeds": {
            str(seed): FORMAL_RNG_SEED_BASE + 1009 * seed
            for seed in FORMAL_SEEDS
        },
        "budgets": list(BUDGETS),
        "recall_targets": list(RECALL_TARGETS),
        "config_artifact": dict(config_artifact),
        "source_config_artifact": dict(source_config_artifact),
        "tuning_manifest": dict(tuning_manifest_artifact),
        "protocol_snapshot_manifest": dict(protocol_snapshot_manifest),
        "baseline_policies": artifact(baseline_policies),
        "sota_definition": SOTA_DEFINITION,
        "external_sota_definition": EXTERNAL_SOTA_DEFINITION,
        "external_outcomes_available_to_ranking": False,
        "gt_available_to_seed_process": False,
    }
    if design_path.exists():
        observed = load_json(design_path)
        comparable = dict(observed)
        comparable.pop("created_at", None)
        if comparable != frozen_fields:
            raise RuntimeError("formal evaluation design changed after preregistration")
    else:
        write_json(
            design_path,
            {**frozen_fields, "created_at": utc_now()},
        )
    design = load_json(design_path)
    verify_artifact(design["config_artifact"])
    verify_artifact(design["source_config_artifact"])
    verify_artifact(design["tuning_manifest"])
    verify_artifact(design["protocol_snapshot_manifest"])
    verify_artifact(design["baseline_policies"])
    return artifact(design_path)


def validate_tuning_evidence(
    *,
    config: Case1NeuroDiscoveryConfig,
    config_path: Path,
    tuning_manifest_path: Path,
    input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_json(tuning_manifest_path)
    masking = manifest.get("feedback_masking_contract") or {}
    artifacts = manifest.get("artifacts") or {}
    best_config_record = artifacts.get("best_config.json") or {}
    if (
        manifest.get("schema_version") != "case1-neurodiscovery-tuning.v1"
        or manifest.get("selected_formal_config") != config.to_dict()
        or manifest.get("external_validation_used_for_tuning") is not False
        or manifest.get("outer_holdout_effects_hidden_until_config_freeze")
        is not True
        or masking.get("hidden_feedback_available") is not False
        or masking.get("hidden_rows_update_factor_or_pair_feedback") is not False
        or manifest.get("all_tests_sha256")
        != input_manifest["source_files"]["all_tests"]["sha256"]
        or (manifest.get("score_components") or {}).get("candidate_id_sha256")
        != input_manifest["candidate_id_sha256"]
        or best_config_record.get("sha256") != sha256_file(config_path)
    ):
        raise RuntimeError("tuning evidence does not prove the frozen formal config")
    for record in (manifest.get("code_provenance") or {}).values():
        verify_artifact(record)
    for record in artifacts.values():
        verify_artifact(record)
    return artifact(tuning_manifest_path)


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    manifest_path = root / "input_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("schema_version") != INPUT_SCHEMA:
            raise RuntimeError("existing output root has an incompatible input manifest")
        for record in manifest["artifacts"].values():
            verify_artifact(record)
        if "candidate_order_sha256" not in manifest:
            public_path = Path(manifest["artifacts"]["public_candidates"]["path"])
            public_ids = pd.read_csv(
                public_path, usecols=["candidate_id"], low_memory=False
            )["candidate_id"].astype(str)
            manifest["candidate_id_sha256"] = candidate_id_sha256(public_ids)
            manifest["candidate_order_sha256"] = candidate_order_sha256(
                public_ids.tolist()
            )
            write_json(manifest_path, manifest)
        return manifest
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            "refusing to overwrite a partial formal root without an input manifest"
        )
    root.mkdir(parents=True, exist_ok=True)

    release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id=CASE_STUDY_ID,
        expected_sha256=CURRENT_CANONICAL_SHA256,
        allow_relocated_artifacts=args.allow_relocated_kg,
    )
    scored = load_results(args.all_tests, args.gt_top_fraction)
    kg = load_kg_index(args.kg, kg_query_terms_for_candidates(scored))
    scored = add_generator_scores(
        scored,
        kg,
        STATIC_SCORE_SEED,
        config=Case1NeuroDiscoveryConfig(),
    )
    scored, component_audit = load_score_component_bundle(
        scored,
        table_path=args.score_components,
        manifest_path=args.score_components_manifest,
    )

    missing_public = sorted(set(PUBLIC_COLUMNS) - set(scored.columns))
    if missing_public:
        raise ValueError(f"prepared score table lacks public columns: {missing_public}")
    public = scored.loc[:, PUBLIC_COLUMNS].copy()
    leaked = sorted(HIDDEN_SCORING_COLUMNS & set(public.columns))
    if leaked:
        raise AssertionError(f"public scoring table leaked outcomes: {leaked}")
    if public["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate IDs must be unique")

    feedback = scored.loc[
        :, ["candidate_id", "execution_succeeded", "adjusted_residual_d", "p_value"]
    ].copy()
    feedback["expected_direction"] = (
        scored["expected_direction"].fillna("").astype(str)
        if "expected_direction" in scored
        else ""
    )
    gt = scored.loc[:, ["candidate_id", "is_gt_top", "is_strict_fdr"]].copy()

    inputs = root / "inputs"
    public_path = inputs / "public_candidates.csv.gz"
    feedback_path = inputs / "feedback_outcomes.csv.gz"
    gt_path = inputs / "gt_labels.csv.gz"
    write_csv_gzip(public, public_path)
    write_csv_gzip(feedback, feedback_path)
    write_csv_gzip(gt, gt_path)

    candidate_ids = public["candidate_id"].astype(str).tolist()
    registry_sha256 = candidate_id_sha256(public["candidate_id"])
    if registry_sha256 != str(component_audit["candidate_id_sha256"]):
        raise RuntimeError("prepared registry does not match the score-component bundle")
    manifest = {
        "schema_version": INPUT_SCHEMA,
        "created_at": utc_now(),
        "case_study_id": CASE_STUDY_ID,
        "status": "prepared",
        "candidate_count": int(len(public)),
        "candidate_id_sha256": registry_sha256,
        "candidate_order_sha256": candidate_order_sha256(candidate_ids),
        "gt_top_fraction": float(args.gt_top_fraction),
        "gt_total": int(gt["is_gt_top"].astype(bool).sum()),
        "strict_fdr_total": int(gt["is_strict_fdr"].astype(bool).sum()),
        "canonical_kg_release": release,
        "kg_index_stats": kg.stats,
        "static_score_seed": STATIC_SCORE_SEED,
        "score_component_bundle": component_audit,
        "separation_contract": {
            "public_columns": list(public.columns),
            "hidden_outcome_columns_in_public": leaked,
            "feedback_columns": list(feedback.columns),
            "gt_columns": list(gt.columns),
            "feedback_file_contains_gt": bool(
                {"is_gt_top", "is_strict_fdr", "gt_rank"} & set(feedback.columns)
            ),
            "formal_seed_process_must_not_open_gt_labels": True,
        },
        "source_files": {
            "all_tests": artifact(args.all_tests),
            "score_components": artifact(args.score_components),
            "score_components_manifest": artifact(args.score_components_manifest),
        },
        "artifacts": {
            "public_candidates": artifact(public_path),
            "feedback_outcomes": artifact(feedback_path),
            "gt_labels": artifact(gt_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


class FormalOutcomeVault:
    """Reveal exact committed batches and persist a tamper-evident audit chain."""

    def __init__(
        self,
        feedback: pd.DataFrame,
        *,
        seed_dir: Path,
        seed: int,
        public_sha256: str,
        feedback_sha256: str,
        config_sha256: str,
    ) -> None:
        if feedback["candidate_id"].astype(str).duplicated().any():
            raise ValueError("feedback vault candidate IDs must be unique")
        self.lookup = feedback.assign(
            candidate_id=feedback["candidate_id"].astype(str)
        ).set_index("candidate_id", drop=False)
        self.seed_dir = seed_dir
        self.seed = int(seed)
        self.public_sha256 = str(public_sha256)
        self.feedback_sha256 = str(feedback_sha256)
        self.config_sha256 = str(config_sha256)
        self.previous_commit_sha256 = "0" * 64
        self.active_commit: dict[str, Any] | None = None
        self.active_commit_path: Path | None = None
        self.commit_count = 0
        self.reveal_count = 0

    def commit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.active_commit is not None:
            raise RuntimeError("previous committed batch has not been revealed")
        batch = int(payload["batch"])
        if batch != self.commit_count:
            raise RuntimeError("batch commitment sequence is not contiguous")
        batch_dir = self.seed_dir / "batches" / f"batch_{batch:04d}"
        if batch_dir.exists() and any(batch_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite batch artifacts: {batch_dir}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        committed = {
            **dict(payload),
            "committed_at": utc_now(),
            "public_candidates_sha256": self.public_sha256,
            "feedback_vault_sha256": self.feedback_sha256,
            "config_sha256": self.config_sha256,
            "previous_commit_sha256": self.previous_commit_sha256,
        }
        committed["commit_sha256"] = canonical_sha256(committed)
        path = batch_dir / "selection_commitment.json"
        write_json(path, committed)
        self.active_commit = committed
        self.active_commit_path = path
        self.commit_count += 1
        return {
            "path": str(path.resolve()),
            "commit_sha256": committed["commit_sha256"],
            "previous_commit_sha256": committed["previous_commit_sha256"],
            "candidate_ids": list(committed["candidate_ids"]),
            "batch": batch,
        }

    def reveal(
        self,
        candidate_ids: Sequence[str],
        selection_commit: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if self.active_commit is None or self.active_commit_path is None:
            raise RuntimeError("outcomes cannot be read before selection commitment")
        committed = load_json(self.active_commit_path)
        observed_hash = str(committed.pop("commit_sha256"))
        if canonical_sha256(committed) != observed_hash:
            raise RuntimeError("selection commitment hash verification failed")
        committed["commit_sha256"] = observed_hash
        requested = [str(value) for value in candidate_ids]
        if requested != [str(value) for value in committed["candidate_ids"]]:
            raise RuntimeError("outcome request does not match committed candidates")
        if str(selection_commit.get("commit_sha256")) != observed_hash:
            raise RuntimeError("reveal callback received the wrong commitment")
        if committed.get("outcomes_read_before_commit") is not False:
            raise RuntimeError("invalid commitment outcome-read guard")

        rows = self.lookup.loc[requested].to_dict(orient="records")
        revealed_rows: list[dict[str, Any]] = []
        for row in rows:
            succeeded = bool(row.get("execution_succeeded", False))
            effect = pd.to_numeric(row.get("adjusted_residual_d"), errors="coerce")
            p_value = pd.to_numeric(row.get("p_value"), errors="coerce")
            finite = bool(np.isfinite(effect) and np.isfinite(p_value))
            status = (
                classify_observed_feedback(
                    float(effect),
                    float(p_value),
                    expected_direction=str(row.get("expected_direction") or ""),
                    alpha=0.01,
                    min_abs_d=0.15,
                )
                if succeeded and finite
                else "execution_failed"
            )
            revealed_rows.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "execution_succeeded": bool(succeeded and finite),
                    "adjusted_residual_d": float(effect) if finite else None,
                    "p_value": float(p_value) if finite else None,
                    "expected_direction": str(row.get("expected_direction") or ""),
                    "feedback_status": status,
                    "feedback_available": True,
                }
            )
        batch_dir = self.active_commit_path.parent
        feedback_payload = {
            "schema_version": "case1-neurodiscovery-batch-feedback.v1",
            "seed": self.seed,
            "batch": int(committed["batch"]),
            "selection_commit_sha256": observed_hash,
            "revealed_at": utc_now(),
            "gt_fields_read": False,
            "rows": revealed_rows,
        }
        write_json(batch_dir / "experimental_feedback.json", feedback_payload)
        reveal_audit = {
            "schema_version": "case1-neurodiscovery-batch-reveal-audit.v1",
            "selection_commit_sha256": observed_hash,
            "selected_candidate_count": len(requested),
            "returned_candidate_count": len(revealed_rows),
            "candidate_order_exact": requested
            == [row["candidate_id"] for row in revealed_rows],
            "outcomes_revealed_after_commitment": True,
            "feedback_fields_read": [
                "execution_succeeded",
                "adjusted_residual_d",
                "p_value",
                "expected_direction",
                "feedback_available",
            ],
            "gt_fields_read": False,
        }
        write_json(batch_dir / "reveal_audit.json", reveal_audit)
        self.previous_commit_sha256 = observed_hash
        self.active_commit = None
        self.active_commit_path = None
        self.reveal_count += 1
        return revealed_rows

    def manifest(self) -> dict[str, Any]:
        return {
            "commit_count": int(self.commit_count),
            "reveal_count": int(self.reveal_count),
            "all_commits_revealed": bool(
                self.active_commit is None and self.commit_count == self.reveal_count
            ),
            "final_commit_chain_hash": self.previous_commit_sha256,
            "gt_fields_read": False,
        }


def run_seed(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    input_manifest = load_json(root / "input_manifest.json")
    if input_manifest.get("schema_version") != INPUT_SCHEMA:
        raise RuntimeError("formal input manifest is missing or incompatible")
    public_path = verify_artifact(input_manifest["artifacts"]["public_candidates"])
    feedback_path = verify_artifact(input_manifest["artifacts"]["feedback_outcomes"])
    config = Case1NeuroDiscoveryConfig.from_json(args.config)
    if config.max_closed_loop_budget < max(BUDGETS):
        raise ValueError("formal config must consume feedback through the 200k endpoint")
    source_config_artifact = artifact(args.config)
    tuning_manifest_path = (
        args.tuning_manifest
        if args.tuning_manifest is not None
        else args.config.with_name("tuning_manifest.json")
    )
    tuning_manifest_artifact = validate_tuning_evidence(
        config=config,
        config_path=args.config,
        tuning_manifest_path=tuning_manifest_path,
        input_manifest=input_manifest,
    )
    frozen_config_path = root / "frozen_config.json"
    if frozen_config_path.exists():
        frozen_config = Case1NeuroDiscoveryConfig.from_json(frozen_config_path)
        if frozen_config.to_dict() != config.to_dict():
            raise RuntimeError(
                "formal root is already bound to a different NeuroDiscovery config"
            )
    else:
        existing_seed_dirs = list((root / "neurodiscovery").glob("seed_*"))
        if existing_seed_dirs:
            raise RuntimeError(
                "formal seed artifacts exist without a root-level frozen config"
            )
        config.write_json(frozen_config_path)
    config_sha = sha256_file(frozen_config_path)
    config_artifact_record = artifact(frozen_config_path)
    protocol_snapshot = freeze_protocol_snapshot(root)
    formal_design = freeze_evaluation_design(
        root,
        input_manifest_sha256=sha256_file(root / "input_manifest.json"),
        candidate_id_sha256_value=str(input_manifest["candidate_id_sha256"]),
        config_artifact=config_artifact_record,
        source_config_artifact=source_config_artifact,
        tuning_manifest_artifact=tuning_manifest_artifact,
        protocol_snapshot_manifest=protocol_snapshot,
        baseline_policies=args.baseline_policies,
    )

    seed = int(args.seed)
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"formal seed must be one of {FORMAL_SEEDS}: {seed}")
    seed_dir = root / "neurodiscovery" / f"seed_{seed:02d}"
    manifest_path = seed_dir / "seed_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != SEED_SCHEMA
            or manifest.get("status") != "complete"
            or int(manifest.get("seed", -1)) != seed
            or manifest.get("config") != config.to_dict()
            or manifest.get("input_manifest_sha256")
            != sha256_file(root / "input_manifest.json")
            or str((manifest.get("config_artifact") or {}).get("sha256", ""))
            != config_sha
        ):
            raise RuntimeError(
                "existing seed manifest differs from the requested formal run"
            )
        seed_audit = audit_seed(root, seed)
        if seed_audit["status"] != "passed":
            raise RuntimeError(f"existing seed failed formal audit: {seed_audit}")
        return manifest
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise RuntimeError(
            f"partial seed artifacts require a new output root: {seed_dir}"
        )
    seed_dir.mkdir(parents=True, exist_ok=True)

    public = pd.read_csv(public_path, low_memory=False)
    leaked = sorted(HIDDEN_SCORING_COLUMNS & set(public.columns))
    if leaked:
        raise RuntimeError(f"formal scoring table contains hidden outcomes: {leaked}")
    if candidate_order_sha256(
        public["candidate_id"].astype(str).tolist()
    ) != input_manifest[
        "candidate_order_sha256"
    ]:
        raise RuntimeError("public candidate order no longer matches the input manifest")
    _global, _scoped, combined = neurodiscovery_score_arrays(public, config)
    public["score_neurodiscovery"] = combined
    feedback = pd.read_csv(feedback_path, low_memory=False)
    if {"is_gt_top", "is_strict_fdr", "gt_rank"} & set(feedback.columns):
        raise RuntimeError("feedback vault contains forbidden GT fields")
    if feedback["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("feedback vault contains duplicate candidate IDs")
    if candidate_id_sha256(feedback["candidate_id"].astype(str)) != str(
        input_manifest["candidate_id_sha256"]
    ):
        raise RuntimeError("feedback vault candidate registry no longer matches public")

    vault = FormalOutcomeVault(
        feedback,
        seed_dir=seed_dir,
        seed=seed,
        public_sha256=str(input_manifest["artifacts"]["public_candidates"]["sha256"]),
        feedback_sha256=str(
            input_manifest["artifacts"]["feedback_outcomes"]["sha256"]
        ),
        config_sha256=config_sha,
    )
    rng_seed = FORMAL_RNG_SEED_BASE + 1009 * seed
    order, batch_audit, overlay = closed_loop_neurodiscovery_order(
        public,
        np.random.default_rng(rng_seed),
        seed=seed,
        trial=seed,
        overlay_path=seed_dir / "experimental_kg_overlay.jsonl.gz",
        return_audit=True,
        return_overlay_manifest=True,
        config=config,
        batch_commit_callback=vault.commit,
        outcome_reveal_callback=vault.reveal,
    )
    if len(order) != len(public) or len(np.unique(order)) != len(public):
        raise RuntimeError("formal ranking is not a full candidate permutation")
    vault_audit = vault.manifest()
    if not vault_audit["all_commits_revealed"]:
        raise RuntimeError("formal outcome vault did not reveal every committed batch")

    ranking_path = seed_dir / "frozen_order_indices.npy"
    with ranking_path.open("wb") as handle:
        np.save(handle, order.astype(np.int32), allow_pickle=False)
    batch_audit_path = seed_dir / "batch_audit.csv"
    batch_audit.to_csv(batch_audit_path, index=False)
    overlay_manifest_path = seed_dir / "experimental_kg_overlay.manifest.json"
    write_json(overlay_manifest_path, overlay)

    integrity = {
        "public_scoring_frame_outcome_blind": not leaked,
        "gt_labels_opened_by_seed_process": False,
        "feedback_consumed_during_ranking": bool(
            overlay.get("feedback_consumed_during_ranking")
        ),
        "nonzero_feedback_reads": int(overlay.get("nonzero_feedback_reads") or 0),
        "selection_changed_batches": int(
            overlay.get("selection_changed_batches") or 0
        ),
        "commit_before_reveal": bool(
            overlay.get("batch_selection_commits", {}).get(
                "committed_before_selected_outcome_lookup"
            )
        ),
        "formal_outcome_vault": overlay.get("formal_outcome_vault"),
        "per_seed_overlay": True,
    }
    required_true = (
        integrity["public_scoring_frame_outcome_blind"],
        not integrity["gt_labels_opened_by_seed_process"],
        integrity["feedback_consumed_during_ranking"],
        integrity["commit_before_reveal"],
        integrity["nonzero_feedback_reads"] > 0,
        integrity["selection_changed_batches"] > 0,
    )
    if not all(required_true):
        raise RuntimeError(f"formal closed-loop integrity gate failed: {integrity}")

    manifest = {
        "schema_version": SEED_SCHEMA,
        "created_at": utc_now(),
        "status": "complete",
        "method": "neurodiscovery",
        "seed": seed,
        "trial": seed,
        "rng_seed": rng_seed,
        "candidate_count": int(len(public)),
        "candidate_id_sha256": input_manifest["candidate_id_sha256"],
        "input_manifest_sha256": sha256_file(root / "input_manifest.json"),
        "config": config.to_dict(),
        "config_artifact": config_artifact_record,
        "protocol_snapshot_manifest": protocol_snapshot,
        "formal_design": formal_design,
        "feedback_horizon": int(config.max_closed_loop_budget),
        "vault_audit": vault_audit,
        "closed_loop_integrity": integrity,
        "frozen_ranking": artifact(ranking_path),
        "batch_audit": artifact(batch_audit_path),
        "overlay_manifest": artifact(overlay_manifest_path),
        "experimental_overlay": artifact(
            Path(str(overlay["path"]))
        ),
        "ranking_frozen_before_gt_evaluation": True,
    }
    write_json(manifest_path, manifest)
    return manifest


def metrics_from_order(
    order: np.ndarray,
    gt: np.ndarray,
    strict: np.ndarray,
    *,
    method: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_gt = gt[order]
    ordered_strict = strict[order]
    cumulative_gt = np.cumsum(ordered_gt)
    cumulative_strict = np.cumsum(ordered_strict)
    gt_positions = np.flatnonzero(ordered_gt) + 1
    n_gt = int(gt.sum())
    curves = []
    for budget in BUDGETS:
        curves.append(
            {
                "method": method,
                "seed": int(seed),
                "budget": budget,
                "gt_hits": int(cumulative_gt[budget - 1]),
                "gt_recall": float(cumulative_gt[budget - 1] / n_gt),
                "strict_fdr_hits": int(cumulative_strict[budget - 1]),
            }
        )
    costs = []
    for target in RECALL_TARGETS:
        needed = int(math.ceil(n_gt * target))
        costs.append(
            {
                "method": method,
                "seed": int(seed),
                "recall_target": target,
                "experiments_required": int(gt_positions[needed - 1]),
            }
        )
    return curves, costs


def aggregate_metric(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, keys, strict=True))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def exact_positive_sign_test(differences: np.ndarray) -> float:
    """One-sided exact sign-test P value; positive always favors NeuroDiscovery."""

    finite = np.asarray(differences, dtype=float)
    finite = finite[np.isfinite(finite) & ~np.isclose(finite, 0.0)]
    n = int(len(finite))
    if n == 0:
        return 1.0
    wins = int(np.sum(finite > 0))
    return float(
        sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_endpoint_tests(
    curves: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline in BASELINE_METHODS:
        for budget in BUDGETS:
            nd = curves[
                (curves["method"] == "neurodiscovery")
                & (curves["budget"] == budget)
            ].set_index("seed")
            other = curves[
                (curves["method"] == baseline) & (curves["budget"] == budget)
            ].set_index("seed")
            joined = nd[["gt_hits"]].join(
                other[["gt_hits"]],
                how="inner",
                lsuffix="_nd",
                rsuffix="_baseline",
            )
            differences = (
                joined["gt_hits_nd"] - joined["gt_hits_baseline"]
            ).to_numpy(float)
            rows.append(
                {
                    "comparison_type": "same_experiments_gt_hits",
                    "baseline_method": baseline,
                    "endpoint": float(budget),
                    "n_pairs": int(len(differences)),
                    "mean_difference_favoring_nd": float(np.mean(differences)),
                    "sd_difference": float(np.std(differences, ddof=1)),
                    "nd_wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(np.isclose(differences, 0.0))),
                    "p_sign_one_sided": exact_positive_sign_test(differences),
                }
            )
        for target in RECALL_TARGETS:
            nd = costs[
                (costs["method"] == "neurodiscovery")
                & np.isclose(costs["recall_target"], target)
            ].set_index("seed")
            other = costs[
                (costs["method"] == baseline)
                & np.isclose(costs["recall_target"], target)
            ].set_index("seed")
            joined = nd[["experiments_required"]].join(
                other[["experiments_required"]],
                how="inner",
                lsuffix="_nd",
                rsuffix="_baseline",
            )
            # Positive means the baseline needed more experiments, so every
            # reported difference has the same "positive favors ND" direction.
            differences = (
                joined["experiments_required_baseline"]
                - joined["experiments_required_nd"]
            ).to_numpy(float)
            rows.append(
                {
                    "comparison_type": "same_recall_experiments",
                    "baseline_method": baseline,
                    "endpoint": float(target),
                    "n_pairs": int(len(differences)),
                    "mean_difference_favoring_nd": float(np.mean(differences)),
                    "sd_difference": float(np.std(differences, ddof=1)),
                    "nd_wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(np.isclose(differences, 0.0))),
                    "p_sign_one_sided": exact_positive_sign_test(differences),
                }
            )
    result = pd.DataFrame(rows)
    if not (result["n_pairs"] == len(FORMAL_SEEDS)).all():
        raise RuntimeError("paired endpoint tests do not contain all formal seeds")
    result["p_sign_holm"] = holm_adjust(result["p_sign_one_sided"])
    return result


def audit_seed(root: Path, seed: int) -> dict[str, Any]:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"formal seed must be one of {FORMAL_SEEDS}: {seed}")
    input_manifest_path = root / "input_manifest.json"
    input_manifest = load_json(input_manifest_path)
    if input_manifest.get("schema_version") != INPUT_SCHEMA:
        raise RuntimeError("formal input manifest is incompatible")
    public_path = verify_artifact(input_manifest["artifacts"]["public_candidates"])
    public_ids = pd.read_csv(
        public_path,
        usecols=["candidate_id"],
        low_memory=False,
    )["candidate_id"].astype(str).to_numpy()
    if len(public_ids) != int(input_manifest["candidate_count"]):
        raise RuntimeError("formal public candidate count changed")
    if candidate_order_sha256(public_ids.tolist()) != str(
        input_manifest["candidate_order_sha256"]
    ):
        raise RuntimeError("formal public candidate order changed")
    if candidate_id_sha256(pd.Series(public_ids)) != str(
        input_manifest["candidate_id_sha256"]
    ):
        raise RuntimeError("formal public candidate registry changed")

    seed_dir = root / "neurodiscovery" / f"seed_{seed:02d}"
    seed_manifest_path = seed_dir / "seed_manifest.json"
    manifest = load_json(seed_manifest_path)
    if (
        manifest.get("schema_version") != SEED_SCHEMA
        or manifest.get("status") != "complete"
        or int(manifest.get("seed", -1)) != seed
        or int(manifest.get("trial", -1)) != seed
    ):
        raise RuntimeError(f"seed {seed} is not complete")
    if manifest.get("input_manifest_sha256") != sha256_file(input_manifest_path):
        raise RuntimeError("seed is not bound to the current formal input manifest")
    if int(manifest.get("candidate_count", -1)) != len(public_ids):
        raise RuntimeError("seed candidate count differs from formal input")
    if manifest.get("candidate_id_sha256") != input_manifest["candidate_id_sha256"]:
        raise RuntimeError("seed candidate registry differs from formal input")

    config_path = verify_artifact(manifest["config_artifact"])
    config = Case1NeuroDiscoveryConfig.from_json(config_path)
    if config.to_dict() != manifest.get("config"):
        raise RuntimeError("seed config payload differs from its frozen artifact")
    config_sha256 = str(manifest["config_artifact"]["sha256"])
    protocol_manifest_path = verify_artifact(manifest["protocol_snapshot_manifest"])
    protocol_manifest = load_json(protocol_manifest_path)
    protocol_sources = protocol_manifest.get("sources") or {}
    if (
        protocol_manifest.get("schema_version")
        != "case1-neurodiscovery-protocol.v1"
        or set(protocol_sources) != set(PROTOCOL_SOURCE_RELATIVE_PATHS)
    ):
        raise RuntimeError("seed protocol snapshot is incomplete")
    for record in protocol_sources.values():
        snapshot_path = verify_artifact(record["snapshot"])
        if sha256_file(snapshot_path) != str(record["source_sha256"]):
            raise RuntimeError("protocol snapshot differs from its source hash")
    formal_design_path = verify_artifact(manifest["formal_design"])
    formal_design = load_json(formal_design_path)
    if (
        formal_design.get("schema_version")
        != "case1-neurodiscovery-formal-design.v1"
        or formal_design.get("formal_seeds") != list(FORMAL_SEEDS)
        or formal_design.get("budgets") != list(BUDGETS)
        or formal_design.get("recall_targets") != list(RECALL_TARGETS)
        or formal_design.get("sota_definition") != SOTA_DEFINITION
        or formal_design.get("external_sota_definition")
        != EXTERNAL_SOTA_DEFINITION
        or formal_design.get("input_manifest_sha256")
        != sha256_file(input_manifest_path)
        or formal_design.get("candidate_id_sha256")
        != input_manifest["candidate_id_sha256"]
        or formal_design.get("rng_seeds", {}).get(str(seed))
        != FORMAL_RNG_SEED_BASE + 1009 * seed
        or formal_design.get("external_outcomes_available_to_ranking") is not False
        or formal_design.get("gt_available_to_seed_process") is not False
    ):
        raise RuntimeError("seed differs from the preregistered formal design")
    frozen_design_config = verify_artifact(formal_design["config_artifact"])
    frozen_source_config = verify_artifact(
        formal_design["source_config_artifact"]
    )
    tuning_manifest_artifact = validate_tuning_evidence(
        config=config,
        config_path=frozen_source_config,
        tuning_manifest_path=verify_artifact(formal_design["tuning_manifest"]),
        input_manifest=input_manifest,
    )
    frozen_design_protocol = verify_artifact(
        formal_design["protocol_snapshot_manifest"]
    )
    verify_artifact(formal_design["baseline_policies"])
    if (
        sha256_file(frozen_design_config) != config_sha256
        or Case1NeuroDiscoveryConfig.from_json(frozen_source_config).to_dict()
        != config.to_dict()
        or tuning_manifest_artifact["sha256"]
        != formal_design["tuning_manifest"]["sha256"]
        or sha256_file(frozen_design_protocol)
        != manifest["protocol_snapshot_manifest"]["sha256"]
    ):
        raise RuntimeError("formal design is not bound to this seed implementation")
    expected_prefix = min(len(public_ids), int(manifest["feedback_horizon"]))
    if expected_prefix < max(BUDGETS):
        raise RuntimeError("formal seed feedback horizon does not cover 200k")

    ranking_path = verify_artifact(manifest["frozen_ranking"])
    batch_audit_path = verify_artifact(manifest["batch_audit"])
    overlay_manifest_path = verify_artifact(manifest["overlay_manifest"])
    overlay_path = verify_artifact(manifest["experimental_overlay"])
    with ranking_path.open("rb") as handle:
        order = np.load(handle, allow_pickle=False).astype(np.int64)
    if (
        len(order) != len(public_ids)
        or np.any(order < 0)
        or np.any(order >= len(public_ids))
        or len(np.unique(order)) != len(public_ids)
    ):
        raise RuntimeError("frozen NeuroDiscovery ranking is not a full permutation")

    overlay = load_json(overlay_manifest_path)
    batch_dirs = sorted((seed_dir / "batches").glob("batch_*"))
    expected_batch_count = math.ceil(expected_prefix / int(config.batch_size))
    if len(batch_dirs) != expected_batch_count:
        raise RuntimeError(
            "formal batch artifact count differs from the feedback horizon: "
            f"{len(batch_dirs)}/{expected_batch_count}"
        )
    previous = "0" * 64
    committed_ids: list[str] = []
    committed_statuses: list[str] = []
    commit_hashes: list[str] = []
    seen_ids: set[str] = set()
    for expected_batch, batch_dir in enumerate(batch_dirs):
        commitment = load_json(batch_dir / "selection_commitment.json")
        stored_hash = str(commitment.pop("commit_sha256"))
        if canonical_sha256(commitment) != stored_hash:
            raise RuntimeError(f"invalid commitment hash: {batch_dir}")
        if (
            commitment.get("schema_version")
            != "case1-neurodiscovery-batch-selection.v1"
            or int(commitment.get("seed", -1)) != seed
            or int(commitment.get("trial", -1)) != seed
            or int(commitment.get("batch", -1)) != expected_batch
        ):
            raise RuntimeError("commitment batches are not contiguous")
        if commitment["previous_commit_sha256"] != previous:
            raise RuntimeError("commitment hash chain is broken")
        if commitment.get("public_candidates_sha256") != input_manifest[
            "artifacts"
        ]["public_candidates"]["sha256"]:
            raise RuntimeError("commitment is not bound to the public registry")
        if commitment.get("feedback_vault_sha256") != input_manifest[
            "artifacts"
        ]["feedback_outcomes"]["sha256"]:
            raise RuntimeError("commitment is not bound to the feedback vault")
        if commitment.get("config_sha256") != config_sha256:
            raise RuntimeError("commitment is not bound to the frozen config")

        candidate_ids = [str(value) for value in commitment.get("candidate_ids", [])]
        if not candidate_ids or len(candidate_ids) > int(config.batch_size):
            raise RuntimeError("commitment has an invalid candidate batch size")
        expected_start = len(committed_ids) + 1
        expected_end = len(committed_ids) + len(candidate_ids)
        if (
            int(commitment.get("start_rank", -1)) != expected_start
            or int(commitment.get("end_rank", -1)) != expected_end
        ):
            raise RuntimeError("commitment ranks are not contiguous")
        if len(set(candidate_ids)) != len(candidate_ids) or seen_ids.intersection(
            candidate_ids
        ):
            raise RuntimeError("commitment repeats a candidate")

        feedback = load_json(batch_dir / "experimental_feedback.json")
        reveal = load_json(batch_dir / "reveal_audit.json")
        if (
            feedback.get("schema_version")
            != "case1-neurodiscovery-batch-feedback.v1"
            or int(feedback.get("seed", -1)) != seed
            or int(feedback.get("batch", -1)) != expected_batch
            or feedback.get("selection_commit_sha256") != stored_hash
        ):
            raise RuntimeError("feedback is not bound to its selection commitment")
        feedback_rows = feedback.get("rows") or []
        feedback_ids = [str(row.get("candidate_id") or "") for row in feedback_rows]
        if feedback_ids != candidate_ids:
            raise RuntimeError("revealed feedback candidate order differs from commitment")
        if feedback.get("gt_fields_read") is not False:
            raise RuntimeError("feedback artifact reports GT access")
        batch_statuses: list[str] = []
        for row in feedback_rows:
            if {"is_gt_top", "is_strict_fdr", "gt_rank"} & set(row):
                raise RuntimeError("revealed feedback contains forbidden GT fields")
            effect = pd.to_numeric(row.get("adjusted_residual_d"), errors="coerce")
            p_value = pd.to_numeric(row.get("p_value"), errors="coerce")
            finite = bool(np.isfinite(effect) and np.isfinite(p_value))
            expected_status = (
                classify_observed_feedback(
                    float(effect),
                    float(p_value),
                    expected_direction=str(row.get("expected_direction") or ""),
                    alpha=0.01,
                    min_abs_d=0.15,
                )
                if bool(row.get("execution_succeeded")) and finite
                else "execution_failed"
            )
            if str(row.get("feedback_status") or "") != expected_status:
                raise RuntimeError("revealed feedback status is not reproducible")
            if row.get("feedback_available") is not True:
                raise RuntimeError("formal feedback was marked unavailable")
            batch_statuses.append(expected_status)
        if reveal.get("outcomes_revealed_after_commitment") is not True:
            raise RuntimeError("reveal audit does not prove commitment-first order")
        if (
            reveal.get("selection_commit_sha256") != stored_hash
            or int(reveal.get("selected_candidate_count", -1)) != len(candidate_ids)
            or int(reveal.get("returned_candidate_count", -1)) != len(candidate_ids)
            or reveal.get("candidate_order_exact") is not True
            or reveal.get("gt_fields_read") is not False
        ):
            raise RuntimeError("GT was accessed during closed-loop selection")
        committed_ids.extend(candidate_ids)
        committed_statuses.extend(batch_statuses)
        seen_ids.update(candidate_ids)
        commit_hashes.append(stored_hash)
        previous = stored_hash

    if len(committed_ids) != expected_prefix:
        raise RuntimeError("committed candidate prefix does not cover feedback horizon")
    frozen_prefix_ids = public_ids[order[:expected_prefix]].tolist()
    if committed_ids != frozen_prefix_ids:
        raise RuntimeError("frozen ranking prefix differs from committed selections")

    batch_audit = pd.read_csv(batch_audit_path, low_memory=False)
    if len(batch_audit) != len(batch_dirs):
        raise RuntimeError("batch audit row count differs from commitment count")
    if batch_audit["selection_commit_sha256"].astype(str).tolist() != commit_hashes:
        raise RuntimeError("batch audit commitment hashes differ from batch artifacts")
    if not batch_audit["outcomes_revealed_after_commitment"].astype(bool).all():
        raise RuntimeError("batch audit contains a reveal-before-commit row")
    if "batch_gt_hits" in batch_audit and not batch_audit["batch_gt_hits"].isna().all():
        raise RuntimeError("formal batch audit leaked GT hit counts")

    overlay_previous = "0" * 64
    overlay_count = 0
    overlay_statuses: list[str] = []
    opener = gzip.open if overlay_path.suffix == ".gz" else open
    with opener(overlay_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if overlay_count >= expected_prefix:
                raise RuntimeError("experimental overlay has excess records")
            record = json.loads(line)
            record_hash = str(record.pop("record_hash", ""))
            calculated_record_hash = hashlib.sha256(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                record.get("schema_version") != "experimental-claim.v2"
                or record.get("case_study_id") != CASE_STUDY_ID
                or int(record.get("seed", -1)) != seed
                or int(record.get("trial", -1)) != seed
                or record.get("previous_hash") != overlay_previous
                or calculated_record_hash != record_hash
            ):
                raise RuntimeError(
                    f"invalid experimental overlay record at line {line_number}"
                )
            if str(record.get("candidate_id") or "") != committed_ids[overlay_count]:
                raise RuntimeError("experimental overlay order differs from commitments")
            if str(record.get("status") or "") != committed_statuses[overlay_count]:
                raise RuntimeError("experimental overlay status differs from reveal")
            overlay_previous = record_hash
            overlay_statuses.append(str(record["status"]))
            overlay_count += 1
    if overlay_count != expected_prefix:
        raise RuntimeError("experimental overlay does not cover committed feedback")

    integrity = manifest["closed_loop_integrity"]
    commits = overlay.get("batch_selection_commits") or {}
    formal_vault = overlay.get("formal_outcome_vault") or {}
    expected_status_counts = dict(sorted(Counter(committed_statuses).items()))
    checks = {
        "commitment_chain_valid": previous
        == manifest["vault_audit"]["final_commit_chain_hash"],
        "commit_count_matches": len(batch_dirs)
        == int(manifest["vault_audit"]["commit_count"]),
        "all_commits_revealed": bool(
            manifest["vault_audit"]["all_commits_revealed"]
        ),
        "feedback_consumed": bool(integrity["feedback_consumed_during_ranking"]),
        "nonzero_feedback": int(integrity["nonzero_feedback_reads"]) > 0,
        "selection_changed": int(integrity["selection_changed_batches"]) > 0,
        "commit_before_reveal": bool(integrity["commit_before_reveal"]),
        "outcome_blind_scoring": bool(
            integrity["public_scoring_frame_outcome_blind"]
        ),
        "gt_not_opened": not bool(integrity["gt_labels_opened_by_seed_process"]),
        "ranking_prefix_matches_commitments": committed_ids == frozen_prefix_ids,
        "feedback_vault_bound": all(
            load_json(batch_dir / "selection_commitment.json").get(
                "feedback_vault_sha256"
            )
            == input_manifest["artifacts"]["feedback_outcomes"]["sha256"]
            for batch_dir in batch_dirs
        ),
        "overlay_feedback_consumed": bool(
            overlay["feedback_consumed_during_ranking"]
        ),
        "overlay_seed_isolated": int(overlay["seed"]) == seed
        and int(overlay["trial"]) == seed,
        "overlay_artifact_bound": Path(str(overlay.get("path"))).resolve()
        == overlay_path.resolve()
        and str(overlay.get("sha256", "")).casefold()
        == str(manifest["experimental_overlay"]["sha256"]).casefold(),
        "overlay_hash_chain_valid": overlay_previous
        == str(overlay.get("final_chain_hash")),
        "overlay_records_match": overlay_count == int(overlay.get("records", -1))
        and expected_status_counts
        == dict(sorted((overlay.get("records_by_status") or {}).items()))
        and overlay_statuses == committed_statuses,
        "overlay_commit_counts_match": int(commits.get("count", -1))
        == len(batch_dirs)
        and int(commits.get("outcome_reveal_count", -1)) == len(batch_dirs)
        and commits.get("committed_before_selected_outcome_lookup") is True,
        "formal_vault_enabled": formal_vault.get("enabled") is True
        and formal_vault.get("scoring_frame_outcome_blind") is True
        and formal_vault.get("outcomes_revealed_only_after_commitment") is True,
        "formal_kg_immutable": overlay.get("mutates_formal_kg") is False,
    }
    return {
        "seed": seed,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "commit_count": len(batch_dirs),
        "nonzero_feedback_reads": int(integrity["nonzero_feedback_reads"]),
        "selection_changed_batches": int(integrity["selection_changed_batches"]),
        "frozen_ranking_sha256": manifest["frozen_ranking"]["sha256"],
        "config_sha256": config_sha256,
        "protocol_snapshot_manifest_sha256": manifest[
            "protocol_snapshot_manifest"
        ]["sha256"],
        "formal_design_sha256": manifest["formal_design"]["sha256"],
        "committed_prefix_records": len(committed_ids),
        "committed_prefix_sha256": candidate_order_sha256(committed_ids),
        "experimental_overlay_sha256": manifest["experimental_overlay"]["sha256"],
    }


def audit_seeds(args: argparse.Namespace) -> dict[str, Any]:
    """Audit all three frozen rankings without resolving or opening GT labels."""

    seeds = tuple(int(seed) for seed in args.seeds)
    if seeds != FORMAL_SEEDS:
        raise ValueError(
            "three-seed closed-loop audit requires exactly seeds 0, 1, and 2 "
            f"in that order; received {seeds}"
        )
    if args.audit_classification not in {"diagnostic", "formal"}:
        raise ValueError(
            "audit-seeds requires --audit-classification diagnostic|formal"
        )

    root = args.output_root.resolve()
    input_manifest_path = root / "input_manifest.json"
    formal_design_path = root / "formal_design.json"
    input_manifest = load_json(input_manifest_path)
    formal_design = load_json(formal_design_path)

    # audit_seed verifies every per-batch commitment and overlay record.  Keep
    # this entire command GT-blind: only manifests and already-frozen rankings
    # are resolved here.
    seed_audits = [audit_seed(root, seed) for seed in seeds]
    order_hashes = [record["frozen_ranking_sha256"] for record in seed_audits]
    config_hashes = [record["config_sha256"] for record in seed_audits]
    protocol_hashes = [
        record["protocol_snapshot_manifest_sha256"] for record in seed_audits
    ]
    design_hashes = [record["formal_design_sha256"] for record in seed_audits]
    checks = {
        "all_seed_audits_passed": all(
            record["status"] == "passed" for record in seed_audits
        ),
        "frozen_rankings_distinct": len(set(order_hashes)) == len(order_hashes),
        "one_frozen_configuration": len(set(config_hashes)) == 1,
        "one_frozen_implementation": len(set(protocol_hashes)) == 1,
        "one_preregistered_design": len(set(design_hashes)) == 1,
        "gt_not_opened_by_any_seed": all(
            record["checks"]["gt_not_opened"] for record in seed_audits
        ),
    }
    manifest = {
        "schema_version": "case1-neurodiscovery-three-seed-structural-audit.v1",
        "created_at": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "classification": args.audit_classification,
        "diagnostic_only": args.audit_classification == "diagnostic",
        "performance_evaluated": False,
        "gt_artifact_resolved_or_opened_by_audit": False,
        "seeds": list(seeds),
        "cross_seed_checks": checks,
        "seed_audits": seed_audits,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "formal_design_sha256": sha256_file(formal_design_path),
        "canonical_kg_release": input_manifest.get("canonical_kg_release"),
        "baseline_policies": formal_design.get("baseline_policies"),
    }
    output_path = args.audit_output or (
        root / "structural_audit" / "three_seed_closed_loop_audit.json"
    )
    write_json(output_path, manifest)
    if manifest["status"] != "passed":
        raise RuntimeError(f"three-seed closed-loop audit failed: {checks}")
    return manifest


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in args.seeds)
    if seeds != FORMAL_SEEDS:
        raise ValueError(
            "formal SOTA evaluation requires exactly seeds 0, 1, and 2 "
            f"in that order; received {seeds}"
        )
    root = args.output_root.resolve()
    input_manifest = load_json(root / "input_manifest.json")
    public_path = verify_artifact(input_manifest["artifacts"]["public_candidates"])
    public = pd.read_csv(public_path, low_memory=False)

    # Audit every frozen ranking before the evaluation process even resolves or
    # opens the GT artifact. This ordering is part of the formal blinding proof.
    seed_audits = [audit_seed(root, seed) for seed in seeds]
    if not all(record["status"] == "passed" for record in seed_audits):
        raise RuntimeError(f"closed-loop audit failed: {seed_audits}")
    order_hashes = [record["frozen_ranking_sha256"] for record in seed_audits]
    if len(set(order_hashes)) != len(order_hashes):
        raise RuntimeError("formal seeds produced identical frozen rankings")
    config_hashes = [record["config_sha256"] for record in seed_audits]
    if len(set(config_hashes)) != 1:
        raise RuntimeError("formal seeds did not use one frozen configuration")
    protocol_hashes = [
        record["protocol_snapshot_manifest_sha256"] for record in seed_audits
    ]
    if len(set(protocol_hashes)) != 1:
        raise RuntimeError("formal seeds did not use one frozen implementation")
    design_hashes = [record["formal_design_sha256"] for record in seed_audits]
    if len(set(design_hashes)) != 1:
        raise RuntimeError("formal seeds did not use one preregistered design")
    seed_zero_manifest = load_json(
        root / "neurodiscovery" / "seed_00" / "seed_manifest.json"
    )
    formal_design_path = verify_artifact(seed_zero_manifest["formal_design"])
    formal_design = load_json(formal_design_path)
    requested_baselines = artifact(args.baseline_policies)
    if requested_baselines["sha256"] != formal_design["baseline_policies"][
        "sha256"
    ]:
        raise RuntimeError(
            "evaluation baseline policies differ from the preregistered artifact"
        )

    gt_path = verify_artifact(input_manifest["artifacts"]["gt_labels"])
    gt_table = pd.read_csv(gt_path, low_memory=False)
    if public["candidate_id"].astype(str).tolist() != gt_table[
        "candidate_id"
    ].astype(str).tolist():
        raise RuntimeError("GT labels do not align with the frozen public registry")
    gt = gt_table["is_gt_top"].astype(bool).to_numpy()
    strict = gt_table["is_strict_fdr"].astype(bool).to_numpy()
    if int(gt.sum()) != int(input_manifest["gt_total"]):
        raise RuntimeError("GT count differs from the frozen input manifest")
    if int(strict.sum()) != int(input_manifest["strict_fdr_total"]):
        raise RuntimeError("strict-FDR count differs from the frozen input manifest")

    curve_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for seed in seeds:
        manifest = load_json(
            root / "neurodiscovery" / f"seed_{seed:02d}" / "seed_manifest.json"
        )
        order_path = verify_artifact(manifest["frozen_ranking"])
        with order_path.open("rb") as handle:
            order = np.load(handle, allow_pickle=False).astype(np.int64)
        curves, costs = metrics_from_order(
            order, gt, strict, method="neurodiscovery", seed=seed
        )
        curve_rows.extend(curves)
        cost_rows.extend(costs)

    policies: dict[tuple[str, int], Any] = {}
    with args.baseline_policies.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                policy = policy_from_payload(json.loads(line))
                key = (str(policy.method), int(policy.trial))
                if key in policies:
                    raise RuntimeError(f"duplicate baseline SearchPolicy: {key}")
                policies[key] = policy
    expected_policy_keys = {
        (method, seed) for method in BASELINE_METHODS for seed in seeds
    }
    if set(policies) != expected_policy_keys:
        missing = sorted(expected_policy_keys - set(policies))
        unexpected = sorted(set(policies) - expected_policy_keys)
        raise RuntimeError(
            "baseline policy matrix is not the frozen six-method x three-seed "
            f"design; missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(expected_policy_keys):
        policy = policies[key]
        order = compile_policy_order(public, policy)
        curves, costs = metrics_from_order(
            order,
            gt,
            strict,
            method=str(policy.method),
            seed=int(policy.trial),
        )
        curve_rows.extend(curves)
        cost_rows.extend(costs)

    curves = pd.DataFrame(curve_rows)
    costs = pd.DataFrame(cost_rows)
    expected_methods = {"neurodiscovery", *BASELINE_METHODS}
    method_seed_counts = curves.groupby("method")["seed"].nunique()
    if set(method_seed_counts.index) != expected_methods or not (
        method_seed_counts == len(seeds)
    ).all():
        raise RuntimeError(
            f"seven-method seed matrix is incomplete: {method_seed_counts.to_dict()}"
        )
    curve_summary = aggregate_metric(
        curves,
        group_columns=("method", "budget"),
        metric_columns=("gt_hits", "gt_recall", "strict_fdr_hits"),
    )
    cost_summary = aggregate_metric(
        costs,
        group_columns=("method", "recall_target"),
        metric_columns=("experiments_required",),
    )
    endpoint_tests = paired_endpoint_tests(curves, costs)

    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(evaluation_dir / "internal_curves_by_seed.csv", index=False)
    costs.to_csv(evaluation_dir / "internal_recall_cost_by_seed.csv", index=False)
    curve_summary.to_csv(
        evaluation_dir / "internal_curves_mean_sample_sd.csv", index=False
    )
    cost_summary.to_csv(
        evaluation_dir / "internal_recall_cost_mean_sample_sd.csv", index=False
    )
    endpoint_tests.to_csv(
        evaluation_dir / "paired_endpoint_tests.csv",
        index=False,
    )

    nd_curve = curve_summary[curve_summary["method"] == "neurodiscovery"].set_index(
        "budget"
    )
    baseline_curve = curve_summary[
        curve_summary["method"] != "neurodiscovery"
    ]
    yield_gates = {}
    for budget in BUDGETS:
        nd_value = float(nd_curve.loc[budget, "gt_hits_mean"])
        best_baseline = float(
            baseline_curve.loc[
                baseline_curve["budget"] == budget, "gt_hits_mean"
            ].max()
        )
        yield_gates[str(budget)] = {
            "neurodiscovery_mean": nd_value,
            "best_baseline_mean": best_baseline,
            "passed": nd_value > best_baseline,
        }
    nd_cost = cost_summary[
        cost_summary["method"] == "neurodiscovery"
    ].set_index("recall_target")
    baseline_cost = cost_summary[cost_summary["method"] != "neurodiscovery"]
    recall_gates = {}
    for target in RECALL_TARGETS:
        nd_value = float(nd_cost.loc[target, "experiments_required_mean"])
        best_baseline = float(
            baseline_cost.loc[
                np.isclose(baseline_cost["recall_target"], target),
                "experiments_required_mean",
            ].min()
        )
        recall_gates[f"{target:.2f}"] = {
            "neurodiscovery_mean": nd_value,
            "best_baseline_mean": best_baseline,
            "passed": nd_value < best_baseline,
        }
    sota_achieved = all(
        record["passed"] for record in (*yield_gates.values(), *recall_gates.values())
    )
    sota_gate = {
        "schema_version": "case1-neurodiscovery-sota-gate.v1",
        "definition": SOTA_DEFINITION,
        "yield_gates": yield_gates,
        "recall_cost_gates": recall_gates,
        "sota_achieved": sota_achieved,
    }
    write_json(evaluation_dir / "sota_gate.json", sota_gate)
    audit = {
        "schema_version": "case1-neurodiscovery-closed-loop-audit.v1",
        "status": "passed",
        "seed_audits": seed_audits,
        "per_seed_isolation": len(set(order_hashes)) == len(order_hashes),
        "gt_opened_only_after_rankings_frozen": True,
        "external_outcomes_used_during_ranking": False,
    }
    write_json(evaluation_dir / "closed_loop_audit.json", audit)
    manifest = {
        "schema_version": EVALUATION_SCHEMA,
        "created_at": utc_now(),
        "status": "complete_sota" if sota_achieved else "complete_not_sota",
        "seeds": list(seeds),
        "methods": sorted(curves["method"].unique().tolist()),
        "candidate_count": int(len(public)),
        "gt_total": int(gt.sum()),
        "budgets": list(BUDGETS),
        "recall_targets": list(RECALL_TARGETS),
        "ranking_freeze_preceded_gt_load": True,
        "input_manifest_sha256": sha256_file(root / "input_manifest.json"),
        "baseline_policies": artifact(args.baseline_policies),
        "gt_labels": artifact(gt_path),
        "formal_design": artifact(formal_design_path),
        "closed_loop_audit": audit,
        "sota_gate": sota_gate,
        "artifacts": {
            path.name: artifact(path)
            for path in sorted(evaluation_dir.glob("*"))
            if path.is_file() and path.name != "evaluation_manifest.json"
        },
    }
    write_json(evaluation_dir / "evaluation_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run-seed", "audit-seeds", "evaluate")
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--allow-relocated-kg", action="store_true")
    parser.add_argument("--score-components", type=Path, default=DEFAULT_SCORE_TABLE)
    parser.add_argument(
        "--score-components-manifest", type=Path, default=DEFAULT_SCORE_MANIFEST
    )
    parser.add_argument("--gt-top-fraction", type=float, default=0.01)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tuning-manifest", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--audit-classification", choices=("diagnostic", "formal")
    )
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--baseline-policies", type=Path, default=DEFAULT_POLICY_FILE
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_inputs(args)
    elif args.command == "run-seed":
        if args.config is None or args.seed is None:
            raise ValueError("run-seed requires --config and --seed")
        result = run_seed(args)
    elif args.command == "audit-seeds":
        result = audit_seeds(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
