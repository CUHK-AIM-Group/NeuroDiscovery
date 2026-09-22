"""Run six native autoresearch frameworks on the blinded Case Study 2 panel.

AI Scientist-v2, Open Co-Scientist, SciAgents, and Virtual Lab use the shared
official adapter client. BrainPilot and Biomni use their native service/agent
clients in ranked batches. Every method sees only the exact 210-candidate
public registry; invalid or missing slots are retained as failures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from core.scripts.case2_search_policy import (
    PUBLIC_COLUMNS,
    SCHEMA_VERSION,
    SearchPolicy,
    build_public_registry,
    compile_policy_order,
    policy_from_payload,
    policy_to_payload,
    validate_policy,
)
from core.scripts.case_study_native_output import (
    anchors_from_validated,
    extract_json,
    validate_ranked_hypotheses,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
CASE2_ROOT = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc")
    / "case2_adni_genetics_v1"
)
DEFAULT_PROTOCOL_ROOT = CASE2_ROOT / "protocols" / "case2_adni_endpoint_holdout_v2"
DEFAULT_REGISTRY = DEFAULT_PROTOCOL_ROOT / "public_candidate_registry.csv"
DEFAULT_PROTOCOL_MANIFEST = DEFAULT_PROTOCOL_ROOT / "protocol_freeze_manifest.json"
PRE_OUTCOME_PROTOCOL_STATUSES = {
    "frozen_before_confirmation_association_access",
    "frozen_before_phase_heldout_association_access",
}
OFFICIAL_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
)
NATIVE_METHODS = ("brainpilot_native", "biomni_native")
METHODS = (*OFFICIAL_METHODS, *NATIVE_METHODS)
FORBIDDEN_GENERATOR_TOKENS = (
    "private_replication_registry",
    "expected_a_sign",
    "expected_b_sign",
    "expected_indirect_sign",
    "a_path_p",
    "b_path_p",
    "sobel_p",
    "family_fdr",
    "global_fdr",
    "heldout_association",
)
REPOSITORIES = {
    "ai_scientist_v2": "AI-Scientist-v2",
    "open_coscientist": "open-coscientist",
    "sciagents": "SciAgentsDiscovery",
    "virtual_lab": "virtual-lab",
}
IMPORT_PROBES = {
    "ai_scientist_v2": (
        "from ai_scientist.llm import create_client; "
        "from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea"
    ),
    "open_coscientist": (
        "import sys; sys.path.insert(0, 'src'); "
        "from open_coscientist import HypothesisGenerator"
    ),
    "sciagents": "import autogen",
    "virtual_lab": (
        "import sys; sys.path.insert(0, 'src'); "
        "from virtual_lab import Agent, run_meeting"
    ),
}


class InfrastructureFailure(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--protocol-freeze-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_MANIFEST,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-anchors", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--brainpilot-url", default="http://127.0.0.1:9460/api")
    parser.add_argument("--brainpilot-runtime-manifest", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--enable-native-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_frozen_registry(
    registry_path: Path,
    protocol_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the hash-locked public registry, never an association table."""

    manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in PRE_OUTCOME_PROTOCOL_STATUSES:
        raise ValueError("Case 2 protocol is not frozen before outcome access")
    expected = manifest["lock_material"]["registry_sha256"]
    if _sha256(registry_path) != expected:
        raise ValueError("Case 2 public registry hash differs from the protocol lock")
    raw = pd.read_csv(registry_path)
    if tuple(raw.columns) != tuple(PUBLIC_COLUMNS):
        raise ValueError("Case 2 public registry does not have the exact public schema")
    registry = build_public_registry(raw)
    if registry["candidate_id"].tolist() != raw["candidate_id"].astype(str).tolist():
        raise ValueError("Case 2 public registry order changed during validation")
    return registry, manifest


def freeze_baseline_rankings(
    registry: pd.DataFrame,
    policies: list[SearchPolicy],
    *,
    out_dir: Path,
    protocol_manifest: dict[str, Any],
    registry_path: Path,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Commit each static baseline's complete order before outcome access."""

    payload: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, policy in enumerate(
        sorted(policies, key=lambda item: (item.method, item.trial))
    ):
        order = np.asarray(compile_policy_order(registry, policy), dtype=np.int32)
        key = f"order_{index:04d}"
        payload[key] = order
        candidate_ids = registry.iloc[order]["candidate_id"].astype(str).tolist()
        records.append(
            {
                "array_key": key,
                "method": policy.method,
                "trial": int(policy.trial),
                "candidate_count": int(len(order)),
                "order_index_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                "ranking_commit_sha256": _sha256_lines(candidate_ids),
            }
        )
    archive_path = out_dir / "frozen_baseline_orders.npz"
    np.savez_compressed(archive_path, **payload)
    aggregate_commit = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    snapshot = protocol_manifest["lock_material"]["kg_snapshot"]
    phase_heldout = (
        protocol_manifest.get("status")
        == "frozen_before_phase_heldout_association_access"
    )
    lock = {
        "schema_version": "neurooracle.case2_static_baseline_ranking_lock.v1",
        "status": (
            "locked_before_phase_heldout_association_access"
            if phase_heldout
            else "locked_before_confirmation_association_access"
        ),
        "protocol_freeze_id": protocol_manifest["freeze_id"],
        "candidate_registry_path": str(registry_path),
        "candidate_registry_sha256": _sha256(registry_path),
        "association_results_accessed": False,
        "confirmation_holdout_axis": (
            "cohort_phase" if phase_heldout else "endpoint"
        ),
        "temporal_kg_freeze": False,
        "temporal_freeze_year": None,
        "kg_snapshot_pinning": True,
        "kg_snapshot_sha256": snapshot["knowledge_graph"]["sha256"],
        "claim_store_snapshot_sha256": snapshot["extracted_claims"]["sha256"],
        "ranking_freeze": "complete static order per method and trial",
        "ranking_commit_sha256": aggregate_commit,
        "orders_path": str(archive_path),
        "orders_file_sha256": _sha256(archive_path),
        "orders": records,
    }
    if audit_path is not None:
        lock["ranking_audit_path"] = str(audit_path)
        lock["ranking_audit_sha256"] = _sha256(audit_path)
    lock_path = out_dir / "BASELINE_RANKINGS.lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lock["lock_path"] = str(lock_path)
    lock["lock_file_sha256"] = _sha256(lock_path)
    return lock


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def audit_baseline_rankings(
    registry: pd.DataFrame,
    policies: list[SearchPolicy],
    *,
    out_dir: Path,
    expected_methods: list[str],
    expected_trials: list[int],
    n_anchors: int,
) -> dict[str, Any]:
    """Audit completeness, blinding, exact IDs, and cross-method independence."""

    candidate_ids = set(registry["candidate_id"].astype(str))
    expected = {
        (method, int(trial))
        for method in expected_methods
        for trial in expected_trials
    }
    observed: dict[tuple[str, int], SearchPolicy] = {}
    critical: list[str] = []
    policy_rows: list[dict[str, Any]] = []
    orders: dict[tuple[str, int], list[str]] = {}

    for policy in policies:
        key = (policy.method, int(policy.trial))
        if key in observed:
            critical.append(f"duplicate_policy:{key[0]}:trial_{key[1]:02d}")
            continue
        observed[key] = policy
        try:
            validate_policy(policy, registry)
            order_index = compile_policy_order(registry, policy)
            if sorted(order_index) != list(range(len(registry))):
                raise ValueError("compiled order is not a complete candidate permutation")
            orders[key] = registry.iloc[order_index]["candidate_id"].astype(str).tolist()
            anchor_ids = [anchor.candidate_id for anchor in policy.anchors]
            unknown = sorted(set(anchor_ids) - candidate_ids)
            if unknown:
                raise ValueError(f"unknown candidate IDs: {unknown[:3]}")
            policy_rows.append(
                {
                    "method": key[0],
                    "trial": key[1],
                    "requested_slots": int(n_anchors),
                    "valid_unique_anchors": len(anchor_ids),
                    "failed_slots": int(n_anchors - len(anchor_ids)),
                    "anchor_sequence_sha256": _sha256_lines(anchor_ids),
                    "compiled_order_sha256": _sha256_lines(orders[key]),
                }
            )
        except Exception as exc:
            critical.append(
                f"invalid_policy:{key[0]}:trial_{key[1]:02d}:{type(exc).__name__}:{exc}"
            )

    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    critical.extend(f"missing_policy:{method}:trial_{trial:02d}" for method, trial in missing)
    critical.extend(
        f"unexpected_policy:{method}:trial_{trial:02d}" for method, trial in unexpected
    )

    leakage_hits: list[dict[str, Any]] = []
    for method, trial in sorted(expected):
        trial_dir = out_dir / method / f"trial_{trial:02d}"
        for name in ("task.json", "native_result.json", "search_policy.json"):
            path = trial_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for token in FORBIDDEN_GENERATOR_TOKENS:
                if token in text:
                    leakage_hits.append(
                        {"method": method, "trial": trial, "file": str(path), "token": token}
                    )
    critical.extend(
        f"private_field_leakage:{row['method']}:trial_{row['trial']:02d}:{row['token']}"
        for row in leakage_hits
    )

    maximum_prefix = max(1, len(registry) // 2)
    comparison_ks = sorted({min(k, maximum_prefix) for k in (20, 50, 80)})
    similarity_rows: list[dict[str, Any]] = []
    for left_key, right_key in combinations(sorted(orders), 2):
        scope = "within_method" if left_key[0] == right_key[0] else "cross_method"
        for k in comparison_ks:
            value = _jaccard(orders[left_key][:k], orders[right_key][:k])
            similarity_rows.append(
                {
                    "scope": scope,
                    "k": k,
                    "left_method": left_key[0],
                    "left_trial": left_key[1],
                    "right_method": right_key[0],
                    "right_trial": right_key[1],
                    "jaccard": value,
                }
            )
            if scope == "cross_method" and k == comparison_ks[-1] and value >= 0.95:
                critical.append(
                    "cross_method_collapse:"
                    f"{left_key[0]}:{left_key[1]}:{right_key[0]}:{right_key[1]}:"
                    f"top_{k}_jaccard={value:.6f}"
                )

    similarity_path = out_dir / "ranking_similarity.csv"
    pd.DataFrame(similarity_rows).to_csv(similarity_path, index=False)
    similarity_summary: list[dict[str, Any]] = []
    similarity_frame = pd.DataFrame(similarity_rows)
    if not similarity_frame.empty:
        for (scope, k), group in similarity_frame.groupby(["scope", "k"]):
            maximum = group.sort_values("jaccard", ascending=False).iloc[0]
            similarity_summary.append(
                {
                    "scope": str(scope),
                    "k": int(k),
                    "pair_count": int(len(group)),
                    "mean_jaccard": float(group["jaccard"].mean()),
                    "max_jaccard": float(maximum["jaccard"]),
                    "max_pair": [
                        str(maximum["left_method"]),
                        int(maximum["left_trial"]),
                        str(maximum["right_method"]),
                        int(maximum["right_trial"]),
                    ],
                }
            )

    audit = {
        "schema_version": "neurooracle.case2_static_ranking_audit.v1",
        "status": "passed" if not critical else "failed",
        "association_results_accessed": False,
        "expected_policy_count": len(expected),
        "observed_policy_count": len(observed),
        "candidate_count": len(registry),
        "requested_anchors_per_policy": int(n_anchors),
        "policy_rows": sorted(policy_rows, key=lambda row: (row["method"], row["trial"])),
        "forbidden_tokens": list(FORBIDDEN_GENERATOR_TOKENS),
        "private_field_leakage": leakage_hits,
        "collapse_threshold": {"scope": "cross_method", "jaccard": 0.95},
        "similarity_summary": similarity_summary,
        "similarity_rows_path": str(similarity_path),
        "similarity_rows_sha256": _sha256(similarity_path),
        "critical_issues": sorted(set(critical)),
    }
    audit_path = out_dir / "RANKING_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    audit["audit_path"] = str(audit_path)
    audit["audit_sha256"] = _sha256(audit_path)
    return audit


def _repository_python(method: str, repo: Path) -> Path:
    generic_name = "CASE_STUDY_" + method.upper() + "_PYTHON"
    legacy_name = "CS1_" + method.upper() + "_PYTHON"
    configured = os.environ.get(generic_name) or os.environ.get(legacy_name)
    candidates = [
        Path(configured) if configured else None,
        repo / ".venv" / "Scripts" / "python.exe",
        repo / "venv" / "Scripts" / "python.exe",
        Path(os.environ["CONDA_PYTHON"]) if os.environ.get("CONDA_PYTHON") else None,
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise InfrastructureFailure(f"No Python runtime found for {method}")


def _preflight(method: str, args: argparse.Namespace) -> dict[str, Any]:
    if method in OFFICIAL_METHODS:
        repo = BASELINES_ROOT / REPOSITORIES[method]
        if not repo.exists():
            return {"method": method, "status": "failed", "error": "repository missing"}
        try:
            runtime = _repository_python(method, repo)
            result = subprocess.run(
                [str(runtime), "-c", IMPORT_PROBES[method]],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:
            return {"method": method, "status": "failed", "error": str(exc)}
        return {
            "method": method,
            "status": "ok" if result.returncode == 0 else "failed",
            "runtime": str(runtime),
            "repository": str(repo),
            "error": result.stderr[-2000:] if result.returncode else "",
        }
    if method == "biomni_native":
        runtime = BASELINES_ROOT / "Biomni" / ".venv" / "Scripts" / "python.exe"
        return {
            "method": method,
            "status": "ok" if runtime.exists() else "failed",
            "runtime": str(runtime),
        }
    return {
        "method": method,
        "status": "not_checked" if args.prepare_only else "runtime_manifest_required",
        "url": args.brainpilot_url,
    }


def _registry_lines(registry: pd.DataFrame) -> str:
    return "\n".join(
        f"- {row.candidate_id} :: {row.pathway_name}; {row.modality} {row.marker}; {row.outcome}"
        for row in registry.itertuples(index=False)
    )


def build_task(
    registry: pd.DataFrame,
    *,
    method: str,
    trial: int,
    n_anchors: int,
    registry_path: Path,
) -> dict[str, Any]:
    research_goal = f"""Blinded Case Study 2 hypothesis prioritization.

Across the complete trial, prioritize exactly {n_anchors} executable pathway-PRS -> imaging-marker -> future-clinical-decline hypotheses. The scientific question is which inherited pathway burdens may be associated with subsequent clinical worsening through a measured imaging marker. This is an association screen, not a causal claim.

You may use the native {method} planning, reflection, discussion, and retrieval workflow. You are not given and must not infer local experimental coefficients, P values, FDR values, KG ranks, NeuroDiscovery scores, or hidden labels.

Only the following exact candidate_id values are executable. Copy IDs verbatim; an invalid, duplicate, or missing proposal consumes its slot and receives zero credit.
{_registry_lines(registry)}

Rank candidates by biological plausibility, specificity of the pathway-marker-outcome chain, likely longitudinal testability, and portfolio diversity. Every final item must contain one exact candidate_id and a concise rationale."""
    return {
        "case_study_id": "case2",
        "policy_schema_version": SCHEMA_VERSION,
        "candidate_id_template": "exposure|modality|marker|outcome",
        "candidate_id_fields": ["exposure", "modality", "marker", "outcome"],
        "coordinate_constraint": (
            "Use only registered pathway PRS, imaging marker, and future outcome coordinates."
        ),
        "ontology_path_fields": [
            "pathway_name",
            "modality",
            "marker",
            "outcome",
        ],
        "lead_expertise": "imaging genetics and longitudinal research prioritization",
        "imaging_expertise": "amyloid, tau, FDG PET and structural MRI biomarkers",
        "biological_expertise": "Alzheimer disease genetics and clinical progression",
        "sciagents_planner_prompt": (
            "Plan a blinded graph search over pathway PRS, imaging marker, and future "
            "clinical outcome coordinates. Do not propose final answers."
        ),
        "sciagents_ontologist_prompt": (
            "Organize pathway, molecular process, imaging biomarker, and longitudinal "
            "clinical outcome relations without hidden experiment results."
        ),
        "method": method,
        "trial": trial,
        "n_anchors": n_anchors,
        "public_registry_path": str(registry_path.resolve()),
        "research_goal": research_goal,
    }


def _native_batch_prompt(
    task: dict[str, Any],
    *,
    method: str,
    start_rank: int,
    end_rank: int,
    previous_ids: list[str],
) -> str:
    final_rule = (
        "Deliver the final JSON through BrainPilot's result_deliver tool."
        if method == "brainpilot_native"
        else "Put the final JSON inside one <solution>...</solution> tag."
    )
    previous = "\n".join(f"- {value}" for value in previous_ids) or "- none"
    return f"""{task['research_goal']}

Generate exactly ranks {start_rank}-{end_rank}, each once. Do not repeat these earlier valid IDs:
{previous}

Return only this object:
{{
  "method": "{method}",
  "hypotheses": [
    {{"rank": {start_rank}, "candidate_id": "exact registered ID", "rationale": "one sentence", "confidence": 0.75}}
  ]
}}
{final_rule}
"""


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    out_dir: Path,
    timeout: int,
    secret: str,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "launcher.stdout.log").write_text(
        result.stdout.replace(secret, "<redacted>"), encoding="utf-8"
    )
    (out_dir / "launcher.stderr.log").write_text(
        result.stderr.replace(secret, "<redacted>"), encoding="utf-8"
    )
    if result.returncode != 0:
        raise InfrastructureFailure(f"child process exited {result.returncode}")


def _run_official(
    method: str,
    task_path: Path,
    trial_dir: Path,
    args: argparse.Namespace,
    secret: str,
) -> SearchPolicy:
    policy_path = trial_dir / "search_policy.json"
    if policy_path.exists() and not args.force:
        return policy_from_payload(json.loads(policy_path.read_text(encoding="utf-8")))
    repo = BASELINES_ROOT / REPOSITORIES[method]
    command = [
        str(_repository_python(method, repo)),
        str(ROOT / "core" / "scripts" / "case_study_official_adapter_client.py"),
        "--method",
        method,
        "--task",
        str(task_path),
        "--out",
        str(trial_dir),
        "--repo",
        str(repo),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if args.enable_native_retrieval:
        command.append("--enable-native-retrieval")
    env = os.environ.copy()
    env["CASE_STUDY_LOCAL_API_KEY"] = secret
    env["CS1_LOCAL_API_KEY"] = secret
    env["OPENAI_API_KEY"] = secret
    env["OPENAI_BASE_URL"] = args.base_url
    env["PYTHONUTF8"] = "1"
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            _run_process(
                command,
                cwd=repo,
                env=env,
                out_dir=trial_dir,
                timeout=args.timeout_seconds,
                secret=secret,
            )
            last_error = None
            break
        except (InfrastructureFailure, subprocess.TimeoutExpired) as exc:
            last_error = exc
            # A native artifact is a completed scientific output. Do not rerun it
            # merely because it contains no valid exact IDs.
            if (trial_dir / "native_result.json").exists():
                raise
            if attempt < args.max_retries:
                time.sleep(min(30.0, 2.0**attempt))
    if last_error is not None:
        raise InfrastructureFailure(
            f"{method} failed after {args.max_retries} attempts: {last_error}"
        )
    policy = policy_from_payload(json.loads(policy_path.read_text(encoding="utf-8")))
    return policy


def _run_native_batch(
    method: str,
    prompt_path: Path,
    batch_dir: Path,
    args: argparse.Namespace,
    secret: str,
) -> str:
    final_path = batch_dir / "final.txt"
    if final_path.exists() and not args.force:
        return final_path.read_text(encoding="utf-8")
    env = os.environ.copy()
    if method == "brainpilot_native":
        env["ANTHROPIC_API_KEY"] = secret
        env["ANTHROPIC_MODEL"] = args.model
        env["BP_THINKING_LEVEL"] = args.reasoning_effort
        command = [
            "node",
            str(ROOT / "core/scripts/brainpilot_case_study_batch_client.mjs"),
            "--prompt",
            str(prompt_path),
            "--out",
            str(batch_dir),
            "--client-dist",
            str(BASELINES_ROOT / "BrainPilot/packages/client-cli/dist/index.js"),
            "--base-url",
            args.brainpilot_url,
            "--max-events",
            "1000",
        ]
        cwd = BASELINES_ROOT / "BrainPilot"
    else:
        env["BIOMNI_API_KEY"] = secret
        env["PYTHONUTF8"] = "1"
        command = [
            str(BASELINES_ROOT / "Biomni/.venv/Scripts/python.exe"),
            str(ROOT / "core/scripts/biomni_case_study_batch_client.py"),
            "--prompt",
            str(prompt_path),
            "--out",
            str(batch_dir),
            "--workspace",
            str(batch_dir / "workspace"),
            "--biomni-root",
            str(BASELINES_ROOT / "Biomni"),
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--reasoning-effort",
            args.reasoning_effort,
        ]
        cwd = BASELINES_ROOT / "Biomni"
    _run_process(
        command,
        cwd=cwd,
        env=env,
        out_dir=batch_dir,
        timeout=args.timeout_seconds,
        secret=secret,
    )
    if not final_path.exists():
        raise InfrastructureFailure(f"{method} produced no final.txt")
    return final_path.read_text(encoding="utf-8")


def _run_native(
    method: str,
    task: dict[str, Any],
    trial_dir: Path,
    registry: pd.DataFrame,
    args: argparse.Namespace,
    secret: str,
    lock: threading.Lock,
) -> SearchPolicy:
    policy_path = trial_dir / "search_policy.json"
    if policy_path.exists() and not args.force:
        return policy_from_payload(json.loads(policy_path.read_text(encoding="utf-8")))
    payloads: list[tuple[int, int, dict[str, Any] | None, str | None]] = []
    previous_ids: list[str] = []
    for start in range(1, args.n_anchors + 1, args.batch_size):
        end = min(args.n_anchors, start + args.batch_size - 1)
        batch_dir = trial_dir / f"batch_{start:03d}_{end:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = batch_dir / "prompt.txt"
        prompt_path.write_text(
            _native_batch_prompt(
                task,
                method=method,
                start_rank=start,
                end_rank=end,
                previous_ids=previous_ids,
            ),
            encoding="utf-8",
        )
        if args.prepare_only:
            payloads.append((start, end, None, None))
            continue
        final_text: str | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                with lock:
                    final_text = _run_native_batch(
                        method, prompt_path, batch_dir, args, secret
                    )
                break
            except (InfrastructureFailure, subprocess.TimeoutExpired):
                if attempt == args.max_retries:
                    raise
                time.sleep(min(30.0, 2.0**attempt))
        assert final_text is not None
        try:
            payload = extract_json(final_text)
            error = None
            for item in payload.get("hypotheses") or []:
                if isinstance(item, dict):
                    candidate_id = str(item.get("candidate_id") or "")
                    if candidate_id in set(registry["candidate_id"]):
                        previous_ids.append(candidate_id)
        except Exception as exc:
            payload = None
            error = f"invalid_native_output:{type(exc).__name__}"
            (batch_dir / "parse_error.txt").write_text(str(exc), encoding="utf-8")
        payloads.append((start, end, payload, error))
    if args.prepare_only:
        return SearchPolicy(
            method=method, trial=int(task["trial"]), schema_version=SCHEMA_VERSION
        )
    validated = validate_ranked_hypotheses(
        method=method,
        trial=int(task["trial"]),
        payloads=payloads,
        candidate_ids=set(registry["candidate_id"]),
    )
    validated.to_csv(trial_dir / "native_validation.csv", index=False)
    anchors = anchors_from_validated(validated)
    policy = SearchPolicy(
        method=method,
        trial=int(task["trial"]),
        schema_version=SCHEMA_VERSION,
        anchors=anchors,
        metadata={
            "adapter": "native",
            "requested_slots": args.n_anchors,
            "valid_anchor_count": len(anchors),
            "failed_slots": int((~validated["valid"]).sum()),
        },
    )
    policy_path.write_text(
        json.dumps(policy_to_payload(policy), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return policy


def _verify_brainpilot(args: argparse.Namespace) -> None:
    if args.prepare_only or "brainpilot_native" not in args.methods:
        return
    if args.brainpilot_runtime_manifest is None:
        raise InfrastructureFailure("BrainPilot requires --brainpilot-runtime-manifest")
    manifest = json.loads(args.brainpilot_runtime_manifest.read_text(encoding="utf-8"))
    if manifest.get("model") != args.model:
        raise InfrastructureFailure("BrainPilot runtime model does not match")
    if manifest.get("reasoning_effort") != args.reasoning_effort:
        raise InfrastructureFailure("BrainPilot reasoning effort does not match")
    declared_urls = set(manifest.get("base_urls") or [])
    if declared_urls and args.brainpilot_url not in declared_urls:
        raise InfrastructureFailure("BrainPilot runtime URL is not declared")
    with urlopen(args.brainpilot_url.rstrip("/") + "/health", timeout=10) as response:
        if response.status >= 400:
            raise InfrastructureFailure(f"BrainPilot health HTTP {response.status}")


def main() -> int:
    args = parse_args()
    registry, protocol_manifest = load_frozen_registry(
        args.registry, args.protocol_freeze_manifest
    )
    if args.out_dir is None:
        experiment_name = (
            str(protocol_manifest["protocol_id"])
            if protocol_manifest.get("status")
            == "frozen_before_phase_heldout_association_access"
            else "case2_adni_confirmatory_v1"
        )
        args.out_dir = (
            CASE2_ROOT
            / "experiments"
            / experiment_name
            / str(protocol_manifest["freeze_id"])[:12]
            / "official_baselines_gpt55_high"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.out_dir / "case2_public_registry.jsonl"
    registry.to_json(registry_path, orient="records", lines=True, force_ascii=False)
    preflight = [_preflight(method, args) for method in args.methods]
    (args.out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2),
        encoding="utf-8",
    )
    failed_preflight = [row for row in preflight if row["status"] == "failed"]
    if failed_preflight and not args.prepare_only:
        failed_methods = ", ".join(row["method"] for row in failed_preflight)
        raise InfrastructureFailure(f"runtime preflight failed: {failed_methods}")
    _verify_brainpilot(args)
    secret = os.environ.get("CASE_STUDY_LOCAL_API_KEY") or os.environ.get(
        "CS1_LOCAL_API_KEY"
    )
    if not args.prepare_only and not secret:
        raise RuntimeError("CASE_STUDY_LOCAL_API_KEY is required")

    locks = {method: threading.Lock() for method in NATIVE_METHODS}

    def run_one(method: str, trial: int) -> SearchPolicy:
        trial_dir = args.out_dir / method / f"trial_{trial:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        task = build_task(
            registry,
            method=method,
            trial=trial,
            n_anchors=args.n_anchors,
            registry_path=registry_path,
        )
        task_path = trial_dir / "task.json"
        task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
        if method in NATIVE_METHODS:
            policy = _run_native(
                method,
                task,
                trial_dir,
                registry,
                args,
                secret or "",
                locks[method],
            )
        elif args.prepare_only:
            policy = SearchPolicy(
                method=method, trial=trial, schema_version=SCHEMA_VERSION
            )
        else:
            policy = _run_official(method, task_path, trial_dir, args, secret or "")
        if not args.prepare_only:
            validate_policy(policy, registry)
        return policy

    policies: list[SearchPolicy] = []
    failures: list[dict[str, Any]] = []

    def run_native_series(
        method: str,
    ) -> tuple[list[SearchPolicy], list[dict[str, Any]]]:
        method_policies: list[SearchPolicy] = []
        method_failures: list[dict[str, Any]] = []
        for trial in args.trials:
            try:
                policy = run_one(method, trial)
            except Exception as exc:
                method_failures.append(
                    {
                        "method": method,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not args.prepare_only:
                method_policies.append(policy)
        return method_policies, method_failures

    official_jobs = [
        (method, trial)
        for trial in args.trials
        for method in args.methods
        if method in OFFICIAL_METHODS
    ]
    native_methods = [method for method in args.methods if method in NATIVE_METHODS]
    job_count = len(official_jobs) + len(native_methods)
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.max_workers, job_count))
    ) as pool:
        # One serial worker per native framework avoids filling the shared pool
        # with lock waiters while still allowing BrainPilot and Biomni to run
        # concurrently. Official framework trials remain ordinary parallel jobs.
        futures: dict[Any, tuple[str, str, int | None]] = {}
        for method in native_methods:
            futures[pool.submit(run_native_series, method)] = (
                "native_series",
                method,
                None,
            )
        for method, trial in official_jobs:
            futures[pool.submit(run_one, method, trial)] = (
                "official_trial",
                method,
                trial,
            )
        for future in as_completed(futures):
            kind, method, trial = futures[future]
            if kind == "native_series":
                try:
                    method_policies, method_failures = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "method": method,
                            "trial": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                policies.extend(method_policies)
                failures.extend(method_failures)
                continue
            assert trial is not None
            try:
                policy = future.result()
            except Exception as exc:
                failures.append(
                    {
                        "method": method,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not args.prepare_only:
                policies.append(policy)

    (args.out_dir / "infrastructure_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    if failures and not args.allow_incomplete:
        raise InfrastructureFailure(f"{len(failures)} trials failed")
    if not args.prepare_only:
        with (args.out_dir / "case2_search_policies.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for policy in sorted(policies, key=lambda item: (item.method, item.trial)):
                handle.write(
                    json.dumps(policy_to_payload(policy), ensure_ascii=False) + "\n"
                )
        proposal_rows = []
        for policy in sorted(policies, key=lambda item: (item.method, item.trial)):
            valid = len(policy.anchors)
            proposal_rows.append(
                {
                    "method": policy.method,
                    "trial": policy.trial,
                    "requested_slots": args.n_anchors,
                    "valid_unique_anchors": valid,
                    "failed_slots": args.n_anchors - valid,
                    "valid_fraction": valid / args.n_anchors,
                }
            )
        pd.DataFrame(proposal_rows).to_csv(
            args.out_dir / "official_native_proposal_summary.csv", index=False
        )
        ranking_audit = audit_baseline_rankings(
            registry,
            policies,
            out_dir=args.out_dir,
            expected_methods=list(args.methods),
            expected_trials=list(args.trials),
            n_anchors=args.n_anchors,
        )
        if ranking_audit["status"] != "passed":
            raise InfrastructureFailure(
                f"ranking audit failed with {len(ranking_audit['critical_issues'])} issue(s)"
            )
        ranking_lock = freeze_baseline_rankings(
            registry,
            policies,
            out_dir=args.out_dir,
            protocol_manifest=protocol_manifest,
            registry_path=args.registry,
            audit_path=Path(ranking_audit["audit_path"]),
        )
    else:
        ranking_lock = None
    manifest = {
        "case_study": "case2",
        "protocol_freeze_id": protocol_manifest["freeze_id"],
        "public_registry_source": str(args.registry),
        "public_registry_sha256": _sha256(args.registry),
        "association_results_accessed": False,
        "candidate_count": len(registry),
        "methods": args.methods,
        "trials": args.trials,
        "n_anchors": args.n_anchors,
        "prepare_only": args.prepare_only,
        "blinding": (
            "public registry excludes coefficients, P/FDR values, KG rank, "
            "NeuroDiscovery score, and experimental labels"
        ),
        "freeze_semantics": {
            "temporal_kg_freeze": False,
            "temporal_freeze_year": None,
            "kg_snapshot_pinning": True,
            "static_baseline_ranking_freeze": not args.prepare_only,
        },
        "ranking_lock": ranking_lock,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 15:29 HKT
