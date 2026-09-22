"""Run and score native BrainPilot and Biomni baselines for Case Study 1.

The native agents see only the registered disease, feature, and anatomy spaces.
They never receive exhaustive outcomes, effect sizes, FDR values, GT labels,
NeuroDiscovery scores, or closed-loop feedback.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    from core.scripts.case1_generation_baseline_experiment import (
        DISEASE_DESCRIPTIONS,
        FEATURE_DESCRIPTIONS,
    )
    from core.scripts.case1_method_comparison import DEFAULT_ALL_TESTS, load_results
    from core.scripts.case1_search_policy import (
        PolicyAnchor,
        SearchPolicy,
        build_public_registry,
        compile_policy_order,
        policy_to_payload,
    )
except ModuleNotFoundError:
    from case1_generation_baseline_experiment import (
        DISEASE_DESCRIPTIONS,
        FEATURE_DESCRIPTIONS,
    )
    from case1_method_comparison import DEFAULT_ALL_TESTS, load_results
    from case1_search_policy import (
        PolicyAnchor,
        SearchPolicy,
        build_public_registry,
        compile_policy_order,
        policy_to_payload,
    )


ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
DEFAULT_OUT_DIR = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260707_fullv2_kg_rerun\native_baselines_gpt55_high"
)
METHODS = ("brainpilot_native", "biomni_native")
METHOD_LABELS = {"brainpilot_native": "BrainPilot", "biomni_native": "Biomni"}


class InfrastructureFailure(RuntimeError):
    """A launch, service, or transport failure that must not become a zero-hit."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-hypotheses", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--brainpilot-urls",
        nargs="+",
        default=["http://127.0.0.1:9460/api"],
        help="Independent BrainPilot backends; seeds are assigned round-robin.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel independent method/seed runs.",
    )
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export completed trials when an infrastructure failure remains after retries.",
    )
    parser.add_argument(
        "--brainpilot-runtime-manifest",
        type=Path,
        default=None,
        help=(
            "JSON written when the BrainPilot service starts; must declare model, "
            "reasoning_effort, and base_urls for a primary run."
        ),
    )
    return parser.parse_args()


def normalize_hemisphere(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"left", "lh"}:
        return "left"
    if text in {"right", "rh"}:
        return "right"
    if text in {"bilateral", "both"}:
        return "bilateral"
    return "unspecified"


def build_registry(scored: pd.DataFrame) -> dict[str, Any]:
    public = build_public_registry(scored)
    diseases = sorted(public["disease"].unique())
    features = sorted(public["feature"].unique())
    anatomy_columns = [
        "anatomy_id",
        "modality",
        "source",
        "roi_index",
        "roi_name",
        "anatomy_full",
        "hemisphere",
        "map_group",
        "network",
        "structure_class",
    ]
    anatomy = public[anatomy_columns].drop_duplicates("anatomy_id").copy()
    anatomy = anatomy.sort_values("anatomy_id", kind="mergesort")
    return {
        "diseases": diseases,
        "features": features,
        "anatomy": anatomy.to_dict(orient="records"),
        "candidate_count": len(public),
    }


def registry_prompt(
    method: str,
    registry: dict[str, Any],
    seed: int,
    start_rank: int,
    batch_size: int,
    previous: list[dict[str, Any]],
) -> str:
    end_rank = start_rank + batch_size - 1
    disease_lines = "\n".join(
        f"- {code}: {DISEASE_DESCRIPTIONS.get(code, code)}"
        for code in registry["diseases"]
    )
    feature_lines = "\n".join(
        f"- {code}: {FEATURE_DESCRIPTIONS.get(code, code)}"
        for code in registry["features"]
    )
    anatomy_lines = "\n".join(
        f"- {row['anatomy_id']} = {row['anatomy_full']}" for row in registry["anatomy"]
    )
    previous_lines = (
        "\n".join(
            f"- {item.get('candidate_id')}"
            for item in previous
            if isinstance(item, dict)
        )
        or "- none"
    )
    framework = METHOD_LABELS[method]
    solution_rule = (
        "Deliver the final JSON through BrainPilot's result_deliver tool."
        if method == "brainpilot_native"
        else "Put the final JSON inside one <solution>...</solution> tag."
    )
    return f"""Blinded Case Study 1 native-agent hypothesis generation.

Framework: {framework}
Seed: {seed}
Batch ranks: {start_rank}-{end_rank}

Generate exactly {batch_size} ranked, testable hypotheses. Each hypothesis is one
disease x feature x anatomy combination. Use your native research-agent workflow.
You may use the framework's normal reasoning and literature tools, but do not inspect
local experiment-result files.
You are not given GT, effect sizes, p-values, FDR values, experiment labels,
NeuroDiscovery scores, or closed-loop feedback.

Allowed disease codes (use the code exactly):
{disease_lines}

Allowed feature codes (use the code exactly):
{feature_lines}

Allowed anatomy registry entries. Select one anatomy_id exactly as written:
{anatomy_lines}

Already generated combinations for this seed; do not repeat them:
{previous_lines}

Constraints:
- Return exactly ranks {start_rank} through {end_rank}, each once.
- candidate_id must be an exact executable ID composed as:
  modality|source|disease|feature|roi_index
  using one allowed disease, feature, and anatomy_id. Do not paraphrase any component.
- Keep the set diverse across diseases, features, atlases, and anatomy.
- rationale is one concise sentence and confidence is a number from 0 to 1.
- Invalid, duplicate, missing, or unmappable outputs are failures and will not be repaired.

Return this JSON object and no explanatory prose:
{{
  "method": "{method}",
  "hypotheses": [
    {{
      "rank": {start_rank},
      "candidate_id": "one exact executable candidate_id",
      "rationale": "one concise sentence",
      "confidence": 0.75
    }}
  ]
}}

{solution_rule}
"""


def extract_json(text: str) -> dict[str, Any]:
    tagged = re.findall(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.S | re.I)
    candidates = [*reversed(tagged), text.strip()]
    decoder = json.JSONDecoder()
    payload: Any = None
    last_error: json.JSONDecodeError | None = None

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc

        decoded: list[tuple[int, int, Any]] = []
        for start, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, relative_end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                decoded.append((start + relative_end, relative_end, value))
        if decoded:
            # Prefer the complete trailing artifact over nested objects within it.
            payload = max(decoded, key=lambda item: (item[0], item[1]))[2]
            break

    if payload is None:
        if last_error is not None:
            raise last_error
        raise ValueError("Native output contains no JSON object or array")
    if isinstance(payload, list):
        return {"hypotheses": payload}
    if not isinstance(payload, dict):
        raise ValueError("Native output must be a JSON object or array")
    return payload


def sanitize_log(text: str, secret: str) -> str:
    out = text.replace(secret, "<redacted>")
    if len(secret) >= 4:
        out = out.replace(secret[-4:], "<redacted-suffix>")
    return out


def verify_brainpilot_runtime(args: argparse.Namespace) -> dict[str, Any] | None:
    if "brainpilot_native" not in args.methods or args.prepare_only:
        return None
    if args.brainpilot_runtime_manifest is None:
        raise RuntimeError(
            "--brainpilot-runtime-manifest is required for BrainPilot primary runs "
            "because client-side environment variables cannot change an already-running server"
        )
    payload = json.loads(args.brainpilot_runtime_manifest.read_text(encoding="utf-8"))
    if payload.get("model") != args.model:
        raise RuntimeError(
            f"BrainPilot runtime model is {payload.get('model')!r}, expected {args.model!r}"
        )
    if payload.get("reasoning_effort") != args.reasoning_effort:
        raise RuntimeError(
            "BrainPilot runtime reasoning_effort does not match the requested setting"
        )
    declared_urls = set(payload.get("base_urls") or [])
    missing_urls = sorted(set(args.brainpilot_urls) - declared_urls)
    if missing_urls:
        raise RuntimeError(
            f"BrainPilot runtime manifest does not declare: {missing_urls}"
        )
    for base_url in args.brainpilot_urls:
        health_url = base_url.rstrip("/") + "/health"
        try:
            with urlopen(health_url, timeout=10) as response:
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"BrainPilot health check failed at {health_url}: {exc}"
            ) from exc
    return payload


def run_command(
    command: list[str], env: dict[str, str], out_dir: Path, timeout: int, secret: str
) -> None:
    started = time.time()
    try:
        result = subprocess.run(
            command,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = sanitize_log(result.stdout, secret)
        stderr = sanitize_log(result.stderr, secret)
        (out_dir / "launcher.stdout.log").write_text(stdout, encoding="utf-8")
        (out_dir / "launcher.stderr.log").write_text(stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"Native client exited {result.returncode}; see {out_dir}"
            )
    except subprocess.TimeoutExpired as exc:
        (out_dir / "launcher.stderr.log").write_text(
            f"Timed out after {timeout} seconds\n{sanitize_log(str(exc), secret)}",
            encoding="utf-8",
        )
        raise
    finally:
        (out_dir / "launcher_duration_seconds.txt").write_text(
            f"{time.time() - started:.3f}\n", encoding="utf-8"
        )


def run_native_batch(
    method: str,
    prompt_path: Path,
    batch_dir: Path,
    args: argparse.Namespace,
    secret: str,
    brainpilot_url: str | None = None,
    force_run: bool = False,
) -> str:
    final_path = batch_dir / "final.txt"
    if final_path.exists() and not args.force and not force_run:
        return final_path.read_text(encoding="utf-8")
    batch_dir.mkdir(parents=True, exist_ok=True)
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
            brainpilot_url or args.brainpilot_urls[0],
            "--max-events",
            str(getattr(args, "max_events", 1000)),
        ]
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
    try:
        run_command(command, env, batch_dir, args.timeout_seconds, secret)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        raise InfrastructureFailure(str(exc)) from exc
    if not final_path.exists():
        raise InfrastructureFailure(
            f"Native client produced no final.txt in {batch_dir}"
        )
    return final_path.read_text(encoding="utf-8")


def validate_seed_outputs(
    method: str,
    seed: int,
    batch_payloads: list[tuple[int, int, dict[str, Any] | None, str | None]],
    public_registry: pd.DataFrame,
) -> pd.DataFrame:
    candidate_lookup = public_registry.set_index("candidate_id", drop=False)
    candidate_set = set(candidate_lookup.index)
    rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for start_rank, end_rank, payload, batch_error in batch_payloads:
        items = payload.get("hypotheses") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []
        by_rank: dict[int, list[Any]] = {}
        for item in items:
            try:
                rank = int(item.get("rank")) if isinstance(item, dict) else -1
            except (TypeError, ValueError):
                rank = -1
            by_rank.setdefault(rank, []).append(item)
        for rank in range(start_rank, end_rank + 1):
            candidates = by_rank.get(rank, [])
            item = (
                candidates[0]
                if len(candidates) == 1 and isinstance(candidates[0], dict)
                else None
            )
            errors: list[str] = []
            if batch_error:
                errors.append(batch_error)
            if not candidates:
                errors.append("missing_rank")
            elif len(candidates) > 1:
                errors.append("duplicate_rank")
            elif item is None:
                errors.append("non_object_hypothesis")
            item = item or {}
            candidate_id = str(item.get("candidate_id") or "").strip()
            rationale = item.get("rationale")
            confidence = item.get("confidence")
            if candidate_id not in candidate_set:
                errors.append("invalid_candidate_id")
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append("missing_rationale")
            try:
                confidence_value = float(confidence)
                if not 0.0 <= confidence_value <= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                confidence_value = np.nan
                errors.append("invalid_confidence")
            if not errors and candidate_id in seen_candidates:
                errors.append("duplicate_hypothesis")
            if not errors:
                seen_candidates.add(candidate_id)
            candidate = (
                candidate_lookup.loc[candidate_id]
                if candidate_id in candidate_set
                else pd.Series(dtype=object)
            )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "generated_rank": rank,
                    "generated_candidate_id": candidate_id,
                    "generated_disease": candidate.get("disease"),
                    "generated_feature": candidate.get("feature"),
                    "generated_anatomy_query": candidate.get("anatomy_full"),
                    "generated_hemisphere": candidate.get("hemisphere"),
                    "generated_rationale": rationale,
                    "generated_confidence": confidence_value,
                    "native_validation_status": "valid"
                    if not errors
                    else ";".join(dict.fromkeys(errors)),
                    "native_schema_valid": not errors,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("generated_rank", kind="mergesort")
        .reset_index(drop=True)
    )


def map_validated(validated: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Attach result columns by exact candidate_id after blinded generation."""

    scored = scored.reset_index(drop=True)
    lookup = scored.set_index("candidate_id", drop=False)
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for record in validated.to_dict(orient="records"):
        row = dict(record)
        if not bool(row["native_schema_valid"]):
            row["mapping_status"] = "native_validation_failed"
            row["mapping_score"] = 0.0
            rows.append(row)
            continue
        candidate_id = str(row["generated_candidate_id"])
        if candidate_id in used:
            row["mapping_status"] = "duplicate_candidate_id"
            row["mapping_score"] = 0.0
        elif candidate_id not in lookup.index:
            row["mapping_status"] = "unknown_candidate_id"
            row["mapping_score"] = 0.0
        else:
            mapped = lookup.loc[candidate_id]
            row["mapping_status"] = "mapped"
            row["mapping_score"] = 1.0
            used.add(candidate_id)
            for col in [
                "candidate_id",
                "disease",
                "feature",
                "modality",
                "source",
                "roi_index",
                "roi_name",
                "anatomy_full",
                "hemisphere",
                "map_group",
                "is_gt_top",
                "is_strict_fdr",
                "abs_adjusted_residual_d",
            ]:
                row[f"mapped_{col}"] = mapped.get(col)
        rows.append(row)
    return pd.DataFrame(rows)


def direct_seed_summary(
    mapped: pd.DataFrame, n_gt: int, budgets: list[int]
) -> pd.DataFrame:
    rows = []
    for (method, seed), sub in mapped.groupby(["method", "seed"], sort=False):
        sub = sub.sort_values("generated_rank", kind="mergesort")
        for budget in budgets:
            head = sub.head(budget)
            is_mapped = head["mapping_status"].eq("mapped")
            gt = (
                head.get("mapped_is_gt_top", pd.Series(False, index=head.index))
                .fillna(False)
                .astype(bool)
            )
            strict = (
                head.get("mapped_is_strict_fdr", pd.Series(False, index=head.index))
                .fillna(False)
                .astype(bool)
            )
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "budget": budget,
                    "schema_valid": int(head["native_schema_valid"].sum()),
                    "mapped": int(is_mapped.sum()),
                    "mapping_rate": float(is_mapped.mean()),
                    "gt_hits": int(gt.sum()),
                    "gt_recall": float(gt.sum() / n_gt),
                    "precision_per_generated_slot": float(gt.sum() / budget),
                    "strict_fdr_hits": int(strict.sum()),
                }
            )
    return pd.DataFrame(rows)


def full_curve(
    mapped: pd.DataFrame, scored: pd.DataFrame, n_gt: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_all = scored["is_gt_top"].to_numpy(bool)
    strict_all = scored["is_strict_fdr"].to_numpy(bool)
    budgets = [
        budget
        for budget in [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
        if budget <= len(scored)
    ]
    targets = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    curve_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for (method, seed), sub in mapped.groupby(["method", "seed"], sort=False):
        sub = sub.sort_values("generated_rank", kind="mergesort")
        anchors: list[PolicyAnchor] = []
        used_ids: set[str] = set()
        for row in sub.to_dict(orient="records"):
            cid = (
                row.get("mapped_candidate_id")
                if row.get("mapping_status") == "mapped"
                else None
            )
            candidate_id = str(cid) if cid is not None else ""
            if not candidate_id or candidate_id in used_ids:
                continue
            confidence = pd.to_numeric(row.get("generated_confidence"), errors="coerce")
            anchors.append(
                PolicyAnchor(
                    candidate_id=candidate_id,
                    score=float(confidence) if pd.notna(confidence) else 1.0,
                    rationale=str(row.get("generated_rationale") or ""),
                )
            )
            used_ids.add(candidate_id)
        if not anchors:
            raise ValueError(
                f"{method} trial {seed} produced no valid exact candidate anchors"
            )
        policy = SearchPolicy(
            method=str(method), trial=int(seed), anchors=tuple(anchors)
        )
        order = compile_policy_order(scored, policy)
        ordered_gt = gt_all[order]
        ordered_strict = strict_all[order]
        cum_gt = np.cumsum(ordered_gt)
        cum_strict = np.cumsum(ordered_strict)
        for budget in budgets:
            curve_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "budget": budget,
                    "gt_hits": int(cum_gt[budget - 1]),
                    "gt_recall": float(cum_gt[budget - 1] / n_gt),
                    "strict_fdr_hits": int(cum_strict[budget - 1]),
                }
            )
        positions = np.flatnonzero(ordered_gt) + 1
        target_row: dict[str, Any] = {"method": method, "seed": int(seed)}
        for target in targets:
            need = int(math.ceil(n_gt * target))
            target_row[f"experiments_for_recall_{int(target * 100)}pct"] = (
                int(positions[need - 1]) if len(positions) >= need else np.nan
            )
        target_rows.append(target_row)
    return pd.DataFrame(curve_rows), pd.DataFrame(target_rows)


def aggregate_numeric(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols)
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        col
        for col in frame.select_dtypes(include=[np.number]).columns
        if col not in {*group_cols, "seed"}
    ]
    for keys, sub in frame.groupby(group_cols, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub else len(sub)
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            mean = float(np.mean(values))
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            sd = math.sqrt(variance)
            row[f"{col}_mean"] = mean
            row[f"{col}_variance"] = variance
            row[f"{col}_sd"] = sd
            row[f"{col}_lo"] = mean - sd
            row[f"{col}_hi"] = mean + sd
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    secret = os.environ.get("CS1_LOCAL_API_KEY")
    if not args.prepare_only and not secret:
        raise RuntimeError(
            "CS1_LOCAL_API_KEY is required and is passed only to child-process environments"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    brainpilot_runtime = verify_brainpilot_runtime(args)

    scored = load_results(args.all_tests, args.gt_top_frac)
    public_registry = build_public_registry(scored)
    registry = build_registry(scored)
    registry_path = args.out_dir / "cs1_public_registry_summary.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    public_registry.to_json(
        args.out_dir / "cs1_public_registry.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    brainpilot_locks = {url: threading.Lock() for url in args.brainpilot_urls}

    def run_seed(method: str, seed: int) -> pd.DataFrame | None:
        seed_dir = args.out_dir / method / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        previous: list[dict[str, Any]] = []
        batch_payloads: list[tuple[int, int, dict[str, Any] | None, str | None]] = []
        brainpilot_url = (
            args.brainpilot_urls[args.seeds.index(seed) % len(args.brainpilot_urls)]
            if method == "brainpilot_native"
            else None
        )
        for start_rank in range(1, args.n_hypotheses + 1, args.batch_size):
            end_rank = min(args.n_hypotheses, start_rank + args.batch_size - 1)
            batch_size = end_rank - start_rank + 1
            batch_dir = seed_dir / f"batch_{start_rank:03d}_{end_rank:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = batch_dir / "prompt.txt"
            prompt_path.write_text(
                registry_prompt(
                    method, registry, seed, start_rank, batch_size, previous
                ),
                encoding="utf-8",
            )
            payload = None
            error = None
            if not args.prepare_only:
                final_text = None
                infrastructure_errors: list[str] = []
                for attempt in range(1, args.max_retries + 1):
                    try:
                        if method == "brainpilot_native":
                            assert brainpilot_url is not None
                            with brainpilot_locks[brainpilot_url]:
                                final_text = run_native_batch(
                                    method,
                                    prompt_path,
                                    batch_dir,
                                    args,
                                    secret or "",
                                    brainpilot_url,
                                    force_run=attempt > 1,
                                )
                        else:
                            final_text = run_native_batch(
                                method,
                                prompt_path,
                                batch_dir,
                                args,
                                secret or "",
                                force_run=attempt > 1,
                            )
                        break
                    except InfrastructureFailure as exc:
                        infrastructure_errors.append(
                            f"attempt={attempt}:{type(exc).__name__}:{exc}"
                        )
                        if attempt < args.max_retries:
                            time.sleep((2, 8, 30)[min(attempt - 1, 2)])
                if final_text is None:
                    (batch_dir / "infrastructure_errors.json").write_text(
                        json.dumps(infrastructure_errors, indent=2),
                        encoding="utf-8",
                    )
                    raise InfrastructureFailure(
                        f"{method} seed={seed} batch={start_rank}-{end_rank} "
                        f"failed after {args.max_retries} attempts"
                    )
                try:
                    payload = extract_json(final_text)
                    (batch_dir / "parsed.json").write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    items = payload.get("hypotheses") or []
                    if isinstance(items, list):
                        previous.extend(
                            item for item in items if isinstance(item, dict)
                        )
                except Exception as exc:
                    # Structurally invalid agent output is a scientific-output failure,
                    # not an infrastructure failure, so it is not silently retried.
                    error = f"invalid_native_output:{type(exc).__name__}"
                    (batch_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            batch_payloads.append((start_rank, end_rank, payload, error))
            print(
                f"{method} seed={seed} batch={start_rank}-{end_rank} status={error or 'ok'}",
                flush=True,
            )
        if args.prepare_only:
            return None
        validated = validate_seed_outputs(method, seed, batch_payloads, public_registry)
        mapped = map_validated(validated, scored)
        mapped.to_csv(seed_dir / "mapped_hypotheses.csv", index=False)
        valid_hypotheses = []
        for row in validated[validated["native_schema_valid"]].to_dict(
            orient="records"
        ):
            valid_hypotheses.append(
                {
                    "rank": int(row["generated_rank"]),
                    "candidate_id": row["generated_candidate_id"],
                    "rationale": row["generated_rationale"],
                    "confidence": row["generated_confidence"],
                }
            )
        (seed_dir / "standardized.json").write_text(
            json.dumps(
                {"method": method, "hypotheses": valid_hypotheses},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return mapped

    mapped_parts: list[pd.DataFrame] = []
    infrastructure_failures: list[dict[str, Any]] = []
    jobs = [(method, seed) for seed in args.seeds for method in args.methods]
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.max_workers, len(jobs)))
    ) as pool:
        futures = {
            pool.submit(run_seed, method, seed): (method, seed) for method, seed in jobs
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                mapped = future.result()
            except InfrastructureFailure as exc:
                infrastructure_failures.append(
                    {"method": method, "seed": seed, "error": str(exc)}
                )
                continue
            if mapped is not None:
                mapped_parts.append(mapped)

    if args.prepare_only:
        print(args.out_dir)
        return
    exclusion_path = args.out_dir / "excluded_infrastructure_trials.json"
    if infrastructure_failures:
        exclusion_path.write_text(
            json.dumps(infrastructure_failures, indent=2),
            encoding="utf-8",
        )
        if not args.allow_incomplete:
            raise InfrastructureFailure(
                f"{len(infrastructure_failures)} native trials remained unavailable after retries"
            )
    else:
        exclusion_path.unlink(missing_ok=True)
    mapped_all = (
        pd.concat(mapped_parts, ignore_index=True) if mapped_parts else pd.DataFrame()
    )
    mapped_all.to_csv(
        args.out_dir / "native_baselines_mapped_hypotheses.csv", index=False
    )
    mapped_all.to_csv(
        args.out_dir / "generation_first_mapped_hypotheses.csv", index=False
    )
    policy_payloads: list[dict[str, Any]] = []
    for (method, seed), sub in mapped_all.groupby(["method", "seed"], sort=False):
        anchors = []
        for row in sub.sort_values("generated_rank", kind="mergesort").to_dict(
            orient="records"
        ):
            if row.get("mapping_status") != "mapped":
                continue
            confidence = pd.to_numeric(row.get("generated_confidence"), errors="coerce")
            anchors.append(
                PolicyAnchor(
                    candidate_id=str(row["mapped_candidate_id"]),
                    score=float(confidence) if pd.notna(confidence) else 1.0,
                    rationale=str(row.get("generated_rationale") or ""),
                )
            )
        if anchors:
            policy_payloads.append(
                policy_to_payload(
                    SearchPolicy(
                        method=str(method),
                        trial=int(seed),
                        anchors=tuple(anchors),
                        metadata={"adapter": "native", "proposal_count": len(sub)},
                    )
                )
            )
    with (args.out_dir / "case1_search_policies.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for payload in policy_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    n_gt = int(scored["is_gt_top"].sum())
    direct_budgets = sorted(
        {
            args.n_hypotheses,
            *[budget for budget in [20, 40, 60, 80] if budget <= args.n_hypotheses],
        }
    )
    direct = direct_seed_summary(mapped_all, n_gt, direct_budgets)
    direct.to_csv(
        args.out_dir / "native_baselines_direct_seed_summary.csv", index=False
    )
    aggregate_numeric(direct, ["method", "budget"]).to_csv(
        args.out_dir / "native_baselines_direct_summary.csv", index=False
    )
    curves, targets = full_curve(mapped_all, scored, n_gt)
    curves.to_csv(args.out_dir / "native_baselines_full_curve_seed.csv", index=False)
    targets.to_csv(args.out_dir / "native_baselines_recall_cost_seed.csv", index=False)
    aggregate_numeric(curves, ["method", "budget"]).to_csv(
        args.out_dir / "native_baselines_full_curve_summary.csv", index=False
    )
    aggregate_numeric(targets, ["method"]).to_csv(
        args.out_dir / "native_baselines_recall_cost_summary.csv", index=False
    )
    manifest = {
        "all_tests": str(args.all_tests),
        "out_dir": str(args.out_dir),
        "methods": args.methods,
        "seeds": args.seeds,
        "n_hypotheses_per_seed": args.n_hypotheses,
        "batch_size": args.batch_size,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "brainpilot_runtime": brainpilot_runtime,
        "gt_top_frac": args.gt_top_frac,
        "gt_total": n_gt,
        "registry_counts": {
            "diseases": len(registry["diseases"]),
            "features": len(registry["features"]),
            "anatomy": len(registry["anatomy"]),
        },
        "dispersion": "sample variance across seeds; plotting bounds are mean +/- one standard deviation",
        "failure_policy": (
            "Invalid or duplicate scientific outputs count as failed native proposals. "
            "Service and transport failures are retried and then excluded, never converted "
            "to zero-hit scientific results. Full-space curves compile valid anchors into "
            "a deterministic SearchPolicy with no random tail."
        ),
        "blinding": "no GT, effect size, p/FDR, experiment labels, NeuroDiscovery score, or closed-loop feedback supplied",
        "credential_policy": "API key supplied only through process environment and never serialized",
    }
    (args.out_dir / "native_baselines_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()


# Last Updated At: 2026-08-01 10:20 HKT
