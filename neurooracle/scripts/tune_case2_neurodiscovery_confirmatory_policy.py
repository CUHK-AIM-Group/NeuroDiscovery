"""Tune and lock the CS2 NeuroDiscovery policy using development outcomes only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.scripts.case2_search_policy import (
    build_public_registry,
    candidate_id_from_fields,
)
from core.scripts.case_study_closed_loop import ClosedLoopConfig
from core.scripts.case_study_closed_loop_engine import run_closed_loop_order
from core.scripts.case_study_feedback_adapters import adapter_for
from neurooracle.scripts.freeze_case2_adni_confirmatory_protocol import CASE2_ROOT
from neurooracle.scripts.case2_confirmatory_statistics import (
    pathway_outcome_family_labels,
)
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import load_hypotheses
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni_longitudinal import (
    build_ranked_candidates,
)
from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    DEFAULT_PATHWAY_ROOT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = CASE2_ROOT / "protocols" / "case2_adni_endpoint_holdout_v2"
DEFAULT_DEV_RESULTS = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_kg_guided_longitudinal_v1"
    / "20260812_kgA8C354B5_rerun"
    / "kg_ranked_longitudinal_results.parquet"
)
DEFAULT_HYPOTHESES = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "experiments"
    / "case2"
    / "pathway_mediation_20260816_kg89E40DD8_confirmatory_v1"
    / "hypotheses_raw.json"
)
DEFAULT_BUDGETS = (5, 10, 20, 50, 100, 210)
DEFAULT_TRIALS = 10
DEFAULT_SEED = 20260816


PRIOR_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "mapping_only",
        "weights": {"mapping": 1.00, "support": 0.00, "chain": 0.00, "kg": 0.00},
    },
    {
        "name": "mapping_consensus_75_25",
        "weights": {"mapping": 0.75, "support": 0.25, "chain": 0.00, "kg": 0.00},
    },
    {
        "name": "balanced_60_30_10",
        "weights": {"mapping": 0.60, "support": 0.30, "chain": 0.10, "kg": 0.00},
    },
    {
        "name": "consensus_50_40_10",
        "weights": {"mapping": 0.50, "support": 0.40, "chain": 0.10, "kg": 0.00},
    },
    {
        "name": "balanced_with_kg_45_35_15_05",
        "weights": {"mapping": 0.45, "support": 0.35, "chain": 0.15, "kg": 0.05},
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Iterable[str]) -> str:
    payload = "\n".join(map(str, values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if not np.isfinite(array).all():
        raise ValueError("NeuroDiscovery prior components must all be finite")
    lower = float(array.min())
    upper = float(array.max())
    if math.isclose(lower, upper):
        return np.full(len(array), 0.5, dtype=float)
    return (array - lower) / (upper - lower)


def prior_score(components: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Compose the four prespecified outcome-blind prior components."""

    expected = {"mapping", "support", "chain", "kg"}
    if set(weights) != expected:
        raise ValueError(f"prior weights must be exactly {sorted(expected)}")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("prior weights cannot be negative")
    if not math.isclose(sum(map(float, weights.values())), 1.0, abs_tol=1e-9):
        raise ValueError("prior weights must sum to one")
    arrays = {
        "mapping": _normalise(components["mapping_score"]),
        "support": _normalise(
            np.log1p(
                pd.to_numeric(
                    components["meaningful_support_count"], errors="raise"
                ).to_numpy(float)
            )
        ),
        "chain": _normalise(components["primary_chain_alignment"]),
        "kg": _normalise(components["primary_kg_score"]),
    }
    return sum(float(weights[name]) * arrays[name] for name in sorted(expected))


def closed_loop_profiles(n_candidates: int) -> tuple[tuple[str, ClosedLoopConfig], ...]:
    rounds_5 = int(math.ceil(n_candidates / 5))
    rounds_10 = int(math.ceil(n_candidates / 10))
    return (
        (
            "static_control",
            ClosedLoopConfig(
                batch_size=1,
                warmup_batches=1,
                sampling_temperature=0.0,
                feedback_weight=0.0,
                pair_feedback_weight=0.0,
                exploration_weight=0.0,
                diversity_penalty=0.0,
                max_feedback_rounds=1,
                feedback_horizon=1,
                feedback_projection="generalizable_factors",
                preserve_static_until_informative_feedback=True,
            ),
        ),
        (
            "conservative_batch5",
            ClosedLoopConfig(
                batch_size=5,
                warmup_batches=1,
                sampling_temperature=0.002,
                feedback_weight=0.45,
                pair_feedback_weight=0.25,
                exploration_weight=0.05,
                diversity_penalty=0.50,
                warmup_diversity_penalty=1.0,
                max_feedback_rounds=rounds_5,
                feedback_horizon=n_candidates,
                feedback_projection="generalizable_factors",
                preserve_static_until_informative_feedback=True,
            ),
        ),
        (
            "strong_factor_batch5",
            ClosedLoopConfig(
                batch_size=5,
                warmup_batches=1,
                sampling_temperature=0.002,
                feedback_weight=0.80,
                pair_feedback_weight=0.30,
                exploration_weight=0.05,
                diversity_penalty=0.50,
                warmup_diversity_penalty=1.0,
                max_feedback_rounds=rounds_5,
                feedback_horizon=n_candidates,
                feedback_projection="generalizable_factors",
                preserve_static_until_informative_feedback=True,
            ),
        ),
        (
            "relation_batch5",
            ClosedLoopConfig(
                batch_size=5,
                warmup_batches=1,
                sampling_temperature=0.002,
                feedback_weight=0.60,
                pair_feedback_weight=0.35,
                exploration_weight=0.08,
                diversity_penalty=0.50,
                warmup_diversity_penalty=1.0,
                max_feedback_rounds=rounds_5,
                feedback_horizon=n_candidates,
                feedback_projection="relation_endpoints",
                preserve_static_until_informative_feedback=True,
            ),
        ),
        (
            "relation_batch10",
            ClosedLoopConfig(
                batch_size=10,
                warmup_batches=1,
                sampling_temperature=0.002,
                feedback_weight=0.60,
                pair_feedback_weight=0.35,
                exploration_weight=0.08,
                diversity_penalty=0.50,
                warmup_diversity_penalty=1.0,
                max_feedback_rounds=rounds_10,
                feedback_horizon=n_candidates,
                feedback_projection="relation_endpoints",
                preserve_static_until_informative_feedback=True,
            ),
        ),
    )


def heldout_feedback_outcomes(
    public: pd.DataFrame,
    labels: np.ndarray,
    *,
    heldout_outcome: str,
) -> pd.DataFrame:
    """Hide one outcome family's labels while retaining the other two for feedback."""

    is_heldout = public["outcome"].astype(str).eq(str(heldout_outcome)).to_numpy()
    feedback_labels = np.asarray(labels, dtype=bool) & ~is_heldout
    return pd.DataFrame(
        {
            "candidate_id": public["candidate_id"].astype(str),
            "validated": feedback_labels,
            "feedback_status": np.where(
                feedback_labels, "supported", "inconclusive"
            ),
        }
    )


def _recall_objective(
    order: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    budgets: tuple[int, ...],
) -> tuple[float, dict[str, float | int]]:
    target = np.asarray(labels, dtype=bool) & np.asarray(mask, dtype=bool)
    total = int(target.sum())
    cumulative = np.cumsum(target[np.asarray(order, dtype=int)].astype(np.int64))
    recalls: list[float] = []
    row: dict[str, float | int] = {"target_total": total}
    for budget in budgets:
        effective = min(int(budget), len(cumulative))
        hits = int(cumulative[effective - 1]) if effective else 0
        recall = float(hits / total) if total else 0.0
        row[f"hits_at_{budget}"] = hits
        row[f"recall_at_{budget}"] = recall
        recalls.append(recall)
    weights = 1.0 / np.sqrt(np.asarray(budgets, dtype=float))
    weights /= weights.sum()
    objective = float(np.dot(weights, np.asarray(recalls, dtype=float)))
    row["objective"] = objective
    return objective, row


def _load_development_bundle(
    *,
    dev_results_path: Path,
    exposures_path: Path,
    pathway_catalog_path: Path,
    hypotheses_path: Path,
    development_outcomes: tuple[str, ...],
    confirmation_outcomes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    results = pd.read_parquet(dev_results_path)
    observed = set(results["outcome"].astype(str))
    if observed != set(development_outcomes):
        raise ValueError(
            f"development result outcomes differ from protocol: {sorted(observed)}"
        )
    if observed & set(confirmation_outcomes):
        raise ValueError("confirmation outcomes leaked into development results")
    universe_columns = [
        "exposure",
        "pathway_id",
        "pathway_name",
        "threshold_label",
        "gene_count",
        "modality",
        "marker",
        "outcome",
    ]
    universe = results.loc[:, universe_columns].drop_duplicates().reset_index(drop=True)
    ranked, _, mapping_audit = build_ranked_candidates(
        load_hypotheses(hypotheses_path),
        pd.read_csv(exposures_path),
        pd.read_csv(pathway_catalog_path),
        universe,
    )
    ranked["candidate_id"] = [
        candidate_id_from_fields(*values)
        for values in ranked.loc[
            :, ["exposure", "modality", "marker", "outcome"]
        ].itertuples(index=False, name=None)
    ]
    public = build_public_registry(ranked)
    components = public.loc[:, ["candidate_id"]].merge(
        ranked,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    labelled = results.copy()
    labelled["candidate_id"] = [
        candidate_id_from_fields(*values)
        for values in labelled.loc[
            :, ["exposure", "modality", "marker", "outcome"]
        ].itertuples(index=False, name=None)
    ]
    labelled = public.loc[:, ["candidate_id"]].merge(
        labelled, on="candidate_id", how="left", validate="one_to_one"
    )
    labels, _ = pathway_outcome_family_labels(labelled, 0.05)
    return public, components, labels, mapping_audit


def _verify_protocol(protocol_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    manifest_path = protocol_root / "protocol_freeze_manifest.json"
    lock_path = protocol_root / "FREEZE.lock.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != lock.get("freeze_id"):
        raise ValueError("protocol freeze manifest and lock disagree")
    if _sha256(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("protocol freeze manifest changed after locking")
    return protocol, manifest


def generator_tuning_outcomes(protocol: dict[str, Any]) -> tuple[str, ...]:
    """Return previously accessed non-target outcomes allowed for tuning."""

    previously_accessed = tuple(
        map(str, protocol["development_data"]["outcomes_previously_accessed"])
    )
    confirmation = set(map(str, protocol["confirmation_data"]["outcomes"]))
    allowed = tuple(outcome for outcome in previously_accessed if outcome not in confirmation)
    if not allowed:
        raise ValueError("No non-target development outcomes remain for generator tuning")
    if set(allowed) & confirmation:
        raise AssertionError("Target outcomes leaked into generator tuning outcomes")
    return allowed


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, freeze = _verify_protocol(args.protocol_root)
    freeze_id = str(freeze["freeze_id"])
    output_root = args.output_root or (
        CASE2_ROOT
        / "experiments"
        / "case2_adni_confirmatory_v1"
        / freeze_id[:12]
        / "neurodiscovery_policy"
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    exposures_path = args.protocol_root / "selected_pathway_exposures.csv"
    pathway_catalog_path = args.pathway_root / "pathway_catalog.csv"
    confirmation_outcomes = tuple(protocol["confirmation_data"]["outcomes"])
    dev_outcomes = generator_tuning_outcomes(protocol)
    public, components, labels, mapping_audit = _load_development_bundle(
        dev_results_path=args.dev_results,
        exposures_path=exposures_path,
        pathway_catalog_path=pathway_catalog_path,
        hypotheses_path=args.hypotheses,
        development_outcomes=dev_outcomes,
        confirmation_outcomes=confirmation_outcomes,
    )
    adapter = adapter_for("case2_pathway_mediation")
    factor_fields = tuple(adapter.factor_fields)
    budgets = tuple(sorted({min(int(value), len(public)) for value in args.budgets}))
    rows: list[dict[str, Any]] = []
    for prior in PRIOR_PROFILES:
        scored = public.copy()
        scored["score_neurodiscovery"] = prior_score(
            components, dict(prior["weights"])
        )
        for loop_name, loop_config in closed_loop_profiles(len(scored)):
            for heldout_outcome in dev_outcomes:
                target_mask = scored["outcome"].astype(str).eq(heldout_outcome).to_numpy()
                feedback = heldout_feedback_outcomes(
                    scored,
                    labels,
                    heldout_outcome=heldout_outcome,
                )
                for trial in range(args.trials):
                    trial_seed = int(args.seed + 1009 * trial)
                    order, _, _ = run_closed_loop_order(
                        scored,
                        feedback,
                        adapter=adapter,
                        factor_fields=factor_fields,
                        rng=np.random.default_rng(trial_seed),
                        config=loop_config,
                        seed=args.seed,
                        trial=trial,
                        audit_records=False,
                        collect_trace=False,
                    )
                    objective, metrics = _recall_objective(
                        order,
                        labels,
                        target_mask,
                        budgets,
                    )
                    rows.append(
                        {
                            "prior_profile": prior["name"],
                            "loop_profile": loop_name,
                            "dynamic": loop_name != "static_control",
                            "heldout_outcome": heldout_outcome,
                            "trial": trial,
                            "trial_seed": trial_seed,
                            "objective": objective,
                            **metrics,
                        }
                    )

    trials = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (prior_name, loop_name, dynamic), group in trials.groupby(
        ["prior_profile", "loop_profile", "dynamic"], sort=False
    ):
        outcome_means = group.groupby("heldout_outcome")["objective"].mean()
        summaries.append(
            {
                "prior_profile": prior_name,
                "loop_profile": loop_name,
                "dynamic": bool(dynamic),
                "objective_mean": float(group["objective"].mean()),
                "objective_sd": float(group["objective"].std(ddof=1)),
                "worst_outcome_mean": float(outcome_means.min()),
                "hits_at_20_mean": float(group["hits_at_20"].mean()),
                "hits_at_50_mean": float(group["hits_at_50"].mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    dynamic = summary.loc[summary["dynamic"]].sort_values(
        ["objective_mean", "worst_outcome_mean", "hits_at_20_mean", "loop_profile"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    if dynamic.empty:
        raise RuntimeError("no dynamic NeuroDiscovery policy was evaluated")
    selected = dynamic.iloc[0]
    selected_prior = next(
        item for item in PRIOR_PROFILES if item["name"] == selected["prior_profile"]
    )
    selected_loop = dict(closed_loop_profiles(len(public)))[selected["loop_profile"]]

    prior_root = args.prior_root or (
        CASE2_ROOT
        / "experiments"
        / "case2_adni_confirmatory_v1"
        / freeze_id[:12]
        / "outcome_blind_generator_inputs"
    )
    confirmation_manifest_path = prior_root / "prior_freeze_manifest.json"
    confirmation_manifest = json.loads(
        confirmation_manifest_path.read_text(encoding="utf-8")
    )
    if confirmation_manifest.get("experimental_statistics_accessed"):
        raise ValueError("confirmation prior manifest reports experimental access")
    confirmation_components_path = Path(
        confirmation_manifest["artifacts"]["prior_components"]
    )
    confirmation_components = pd.read_parquet(confirmation_components_path)
    confirmation_components["score_neurodiscovery"] = prior_score(
        confirmation_components, dict(selected_prior["weights"])
    )
    initial = confirmation_components.sort_values(
        ["score_neurodiscovery", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    initial.insert(0, "initial_rank", np.arange(1, len(initial) + 1))
    initial_path = output_root / "confirmation_initial_ranking.csv"
    initial.loc[
        :,
        [
            "initial_rank",
            "candidate_id",
            "exposure",
            "modality",
            "marker",
            "outcome",
            "score_neurodiscovery",
        ],
    ].to_csv(initial_path, index=False)

    trials_path = output_root / "development_tuning_trials.csv"
    summary_path = output_root / "development_tuning_summary.csv"
    mapping_path = output_root / "development_hypothesis_mapping_audit.csv"
    trials.to_csv(trials_path, index=False)
    summary.sort_values(
        ["dynamic", "objective_mean"], ascending=[False, False]
    ).to_csv(summary_path, index=False)
    mapping_audit.to_csv(mapping_path, index=False)

    ranking_sha = _sha256_lines(initial["candidate_id"].astype(str))
    phase_heldout = (
        freeze.get("status") == "frozen_before_phase_heldout_association_access"
    )
    ranking_status = (
        "locked_before_phase_heldout_association_access"
        if phase_heldout
        else "locked_before_confirmation_association_access"
    )
    policy = {
        "schema_version": "neurooracle.case2_neurodiscovery_policy.v1",
        "status": ranking_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_freeze_id": freeze_id,
        "selection_scope": "non-target development outcomes only",
        "development_outcomes_previously_accessed": list(
            protocol["development_data"]["outcomes_previously_accessed"]
        ),
        "development_outcomes_used_for_tuning": list(dev_outcomes),
        "confirmation_outcomes_accessed": [],
        "selection_design": (
            "Outcome-family cross-validation: each development outcome is hidden "
            "from feedback and scored after adapting only to the other outcomes."
        ),
        "selected_prior_profile": selected_prior,
        "selected_closed_loop_profile": {
            "name": str(selected["loop_profile"]),
            "config": asdict(selected_loop),
        },
        "selection_metrics": {
            "objective_mean": float(selected["objective_mean"]),
            "objective_sd": float(selected["objective_sd"]),
            "worst_outcome_mean": float(selected["worst_outcome_mean"]),
            "hits_at_20_mean": float(selected["hits_at_20_mean"]),
            "hits_at_50_mean": float(selected["hits_at_50_mean"]),
        },
        "freeze_semantics": {
            "temporal_kg_freeze": False,
            "temporal_freeze_year": None,
            "kg_snapshot_pinning": True,
            "ranking_freeze": True,
            "sequential_batch_commit_required": True,
        },
        "kg_snapshot_sha256": freeze["lock_material"]["kg_snapshot"][
            "knowledge_graph"
        ]["sha256"],
        "claim_store_snapshot_sha256": freeze["lock_material"]["kg_snapshot"][
            "extracted_claims"
        ]["sha256"],
        "initial_ranking_commit_sha256": ranking_sha,
        "candidate_count": int(len(initial)),
        "budgets": list(budgets),
        "trials": int(args.trials),
        "inputs": {
            "development_results": {
                "path": str(args.dev_results),
                "sha256": _sha256(args.dev_results),
            },
            "hypotheses": {
                "path": str(args.hypotheses),
                "sha256": _sha256(args.hypotheses),
            },
            "confirmation_prior_manifest": {
                "path": str(confirmation_manifest_path),
                "sha256": _sha256(confirmation_manifest_path),
            },
        },
        "artifacts": {
            "development_trials": str(trials_path),
            "development_summary": str(summary_path),
            "initial_ranking": str(initial_path),
        },
    }
    policy_path = output_root / "LOCKED_NEURODISCOVERY_POLICY.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lock = {
        "schema_version": "neurooracle.case2_ranking_commit.v1",
        "status": ranking_status,
        "protocol_freeze_id": freeze_id,
        "candidate_registry_sha256": freeze["lock_material"]["registry_sha256"],
        "candidate_count": int(len(initial)),
        "association_results_accessed": False,
        "confirmation_holdout_axis": (
            "cohort_phase" if phase_heldout else "endpoint"
        ),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "initial_ranking_path": str(initial_path),
        "initial_ranking_file_sha256": _sha256(initial_path),
        "ranking_commit_sha256": ranking_sha,
        "kg_snapshot_sha256": policy["kg_snapshot_sha256"],
        "claim_store_snapshot_sha256": policy["claim_store_snapshot_sha256"],
        "sequential_batch_commit_required": True,
        "temporal_freeze_year": None,
    }
    lock_path = output_root / "INITIAL_RANKING.lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), **lock}, indent=2))
    return {"policy": policy, "lock": lock}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--dev-results", type=Path, default=DEFAULT_DEV_RESULTS)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 13:06 HKT
