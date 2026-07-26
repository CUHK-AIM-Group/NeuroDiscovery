"""Run and score native BrainPilot and Biomni baselines for Case Study 1.

The native agents see only the registered disease, feature, and anatomy spaces.
They never receive exhaustive outcomes, effect sizes, FDR values, GT labels,
NeuroDiscovery scores, or closed-loop feedback.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from case1_generation_baseline_experiment import (
    DISEASE_DESCRIPTIONS,
    FEATURE_DESCRIPTIONS,
    build_candidate_index,
    map_one_hypothesis,
)
from case1_method_comparison import DEFAULT_ALL_TESTS, load_results


ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = ROOT.parent / "autoresearch_baselines"
DEFAULT_OUT_DIR = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260707_fullv2_kg_rerun\native_baselines_gpt56sol_high"
)
METHODS = ("brainpilot_native", "biomni_native")
METHOD_LABELS = {"brainpilot_native": "BrainPilot", "biomni_native": "Biomni"}
ALLOWED_HEMISPHERES = {"left", "right", "bilateral", "unspecified"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-hypotheses", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--brainpilot-urls",
        nargs="+",
        default=["http://127.0.0.1:9460/api"],
        help="Independent BrainPilot backends; seeds are assigned round-robin.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel independent method/seed runs.")
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
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
    diseases = sorted(str(x) for x in scored["disease"].dropna().unique())
    features = sorted(str(x) for x in scored["feature"].dropna().unique())
    anatomy = scored[["source", "anatomy_full", "hemisphere"]].drop_duplicates().copy()
    anatomy = anatomy[anatomy["source"].notna() & anatomy["anatomy_full"].notna()]
    anatomy["source"] = anatomy["source"].astype(str).str.strip()
    anatomy["anatomy_full"] = anatomy["anatomy_full"].astype(str).str.strip()
    anatomy["hemisphere"] = anatomy["hemisphere"].map(normalize_hemisphere)
    anatomy["anatomy_query"] = anatomy["source"] + " :: " + anatomy["anatomy_full"]
    anatomy = anatomy.drop_duplicates("anatomy_query").sort_values("anatomy_query", kind="mergesort")
    return {
        "diseases": diseases,
        "features": features,
        "anatomy": anatomy[["anatomy_query", "hemisphere"]].to_dict(orient="records"),
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
        f"- {code}: {DISEASE_DESCRIPTIONS.get(code, code)}" for code in registry["diseases"]
    )
    feature_lines = "\n".join(
        f"- {code}: {FEATURE_DESCRIPTIONS.get(code, code)}" for code in registry["features"]
    )
    anatomy_lines = "\n".join(
        f"- {row['anatomy_query']} | hemisphere={row['hemisphere']}" for row in registry["anatomy"]
    )
    previous_lines = "\n".join(
        f"- {item.get('disease')} | {item.get('feature')} | {item.get('anatomy_query')}"
        for item in previous
        if isinstance(item, dict)
    ) or "- none"
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
disease x feature x anatomy combination. Use your native research-agent reasoning,
but do not retrieve external information and do not inspect local experiment files.
You are not given GT, effect sizes, p-values, FDR values, experiment labels,
NeuroDiscovery scores, or closed-loop feedback.

Allowed disease codes (use the code exactly):
{disease_lines}

Allowed feature codes (use the code exactly):
{feature_lines}

Allowed anatomy registry entries. anatomy_query must exactly equal the text before
"| hemisphere=" on one line below. Do not invent, shorten, or paraphrase it:
{anatomy_lines}

Already generated combinations for this seed; do not repeat them:
{previous_lines}

Constraints:
- Return exactly ranks {start_rank} through {end_rank}, each once.
- disease, feature, and anatomy_query must be copied exactly from the registries.
- hemisphere must be left, right, bilateral, or unspecified.
- Keep the set diverse across diseases, features, atlases, and anatomy.
- rationale is one concise sentence and confidence is a number from 0 to 1.
- Invalid, duplicate, missing, or unmappable outputs are failures and will not be repaired.

Return this JSON object and no explanatory prose:
{{
  "method": "{method}",
  "hypotheses": [
    {{
      "rank": {start_rank},
      "disease": "one allowed disease code",
      "feature": "one allowed feature code",
      "anatomy_query": "one exact allowed anatomy_query",
      "hemisphere": "left/right/bilateral/unspecified",
      "rationale": "one concise sentence",
      "confidence": 0.75
    }}
  ]
}}

{solution_rule}
"""


def extract_json(text: str) -> dict[str, Any]:
    tagged = re.search(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.S | re.I)
    candidate = tagged.group(1) if tagged else text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            array_start = candidate.find("[")
            array_end = candidate.rfind("]")
            if array_start < 0 or array_end <= array_start:
                raise
            payload = json.loads(candidate[array_start : array_end + 1])
        else:
            payload = json.loads(candidate[start : end + 1])
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


def run_command(command: list[str], env: dict[str, str], out_dir: Path, timeout: int, secret: str) -> None:
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
            raise RuntimeError(f"Native client exited {result.returncode}; see {out_dir}")
    except subprocess.TimeoutExpired as exc:
        (out_dir / "launcher.stderr.log").write_text(
            f"Timed out after {timeout} seconds\n{sanitize_log(str(exc), secret)}", encoding="utf-8"
        )
        raise
    finally:
        (out_dir / "launcher_duration_seconds.txt").write_text(f"{time.time() - started:.3f}\n", encoding="utf-8")


def run_native_batch(
    method: str,
    prompt_path: Path,
    batch_dir: Path,
    args: argparse.Namespace,
    secret: str,
    brainpilot_url: str | None = None,
) -> str:
    final_path = batch_dir / "final.txt"
    if final_path.exists() and not args.force:
        return final_path.read_text(encoding="utf-8")
    batch_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if method == "brainpilot_native":
        env["ANTHROPIC_API_KEY"] = secret
        env["ANTHROPIC_MODEL"] = args.model
        env["BP_THINKING_LEVEL"] = args.reasoning_effort
        command = [
            "node",
            str(ROOT / "core/scripts/brainpilot_cs1_batch_client.mjs"),
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
            str(ROOT / "core/scripts/biomni_cs1_batch_client.py"),
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
    run_command(command, env, batch_dir, args.timeout_seconds, secret)
    if not final_path.exists():
        raise RuntimeError(f"Native client produced no final.txt in {batch_dir}")
    return final_path.read_text(encoding="utf-8")


def validate_seed_outputs(
    method: str,
    seed: int,
    batch_payloads: list[tuple[int, int, dict[str, Any] | None, str | None]],
    registry: dict[str, Any],
) -> pd.DataFrame:
    disease_set = set(registry["diseases"])
    feature_set = set(registry["features"])
    anatomy_set = {row["anatomy_query"] for row in registry["anatomy"]}
    rows: list[dict[str, Any]] = []
    seen_combinations: set[tuple[str, str, str, str]] = set()
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
            item = candidates[0] if len(candidates) == 1 and isinstance(candidates[0], dict) else None
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
            disease = item.get("disease")
            feature = item.get("feature")
            anatomy_query = item.get("anatomy_query")
            hemisphere = str(item.get("hemisphere") or "").strip().lower()
            rationale = item.get("rationale")
            confidence = item.get("confidence")
            if disease not in disease_set:
                errors.append("invalid_disease")
            if feature not in feature_set:
                errors.append("invalid_feature")
            if anatomy_query not in anatomy_set:
                errors.append("invalid_anatomy")
            if hemisphere not in ALLOWED_HEMISPHERES:
                errors.append("invalid_hemisphere")
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append("missing_rationale")
            try:
                confidence_value = float(confidence)
                if not 0.0 <= confidence_value <= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                confidence_value = np.nan
                errors.append("invalid_confidence")
            combo = (str(disease), str(feature), str(anatomy_query), hemisphere)
            if not errors and combo in seen_combinations:
                errors.append("duplicate_hypothesis")
            if not errors:
                seen_combinations.add(combo)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "generated_rank": rank,
                    "generated_disease": disease,
                    "generated_feature": feature,
                    "generated_anatomy_query": anatomy_query,
                    "generated_hemisphere": hemisphere,
                    "generated_rationale": rationale,
                    "generated_confidence": confidence_value,
                    "native_validation_status": "valid" if not errors else ";".join(dict.fromkeys(errors)),
                    "native_schema_valid": not errors,
                }
            )
    return pd.DataFrame(rows).sort_values("generated_rank", kind="mergesort").reset_index(drop=True)


def map_validated(validated: pd.DataFrame, index: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for record in validated.to_dict(orient="records"):
        row = dict(record)
        if not bool(row["native_schema_valid"]):
            row["mapping_status"] = "native_validation_failed"
            row["mapping_score"] = 0.0
            rows.append(row)
            continue
        hyp = {
            "disease": row["generated_disease"],
            "feature": row["generated_feature"],
            "anatomy_query": row["generated_anatomy_query"],
            "hemisphere": row["generated_hemisphere"],
        }
        mapped, score, status = map_one_hypothesis(hyp, index, used)
        row["mapping_status"] = status
        row["mapping_score"] = score
        if mapped is not None:
            candidate_id = str(mapped["candidate_id"])
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


def direct_seed_summary(mapped: pd.DataFrame, n_gt: int, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for (method, seed), sub in mapped.groupby(["method", "seed"], sort=False):
        sub = sub.sort_values("generated_rank", kind="mergesort")
        for budget in budgets:
            head = sub.head(budget)
            is_mapped = head["mapping_status"].eq("mapped")
            gt = head.get("mapped_is_gt_top", pd.Series(False, index=head.index)).fillna(False).astype(bool)
            strict = head.get("mapped_is_strict_fdr", pd.Series(False, index=head.index)).fillna(False).astype(bool)
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


def full_curve(mapped: pd.DataFrame, scored: pd.DataFrame, n_gt: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_to_idx = {str(cid): i for i, cid in enumerate(scored["candidate_id"].astype(str))}
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    gt_all = scored["is_gt_top"].to_numpy(bool)
    strict_all = scored["is_strict_fdr"].to_numpy(bool)
    budgets = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    targets = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    curve_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for (method, seed), sub in mapped.groupby(["method", "seed"], sort=False):
        sub = sub.sort_values("generated_rank", kind="mergesort")
        used_ids: set[str] = set()
        prefix_gt: list[bool] = []
        prefix_strict: list[bool] = []
        for row in sub.to_dict(orient="records"):
            cid = row.get("mapped_candidate_id") if row.get("mapping_status") == "mapped" else None
            idx = id_to_idx.get(str(cid)) if cid is not None else None
            if idx is None or str(cid) in used_ids:
                prefix_gt.append(False)
                prefix_strict.append(False)
                continue
            used_ids.add(str(cid))
            prefix_gt.append(bool(gt_all[idx]))
            prefix_strict.append(bool(strict_all[idx]))
        used_mask = np.array([cid in used_ids for cid in candidate_ids], dtype=bool)
        tail_idx = np.flatnonzero(~used_mask)
        method_seed = int.from_bytes(hashlib.blake2b(str(method).encode(), digest_size=4).digest(), "little")
        rng = np.random.default_rng(int(seed) + method_seed)
        tail_idx = tail_idx[np.argsort(rng.random(len(tail_idx)), kind="mergesort")]
        ordered_gt = np.concatenate([np.asarray(prefix_gt, dtype=bool), gt_all[tail_idx]])
        ordered_strict = np.concatenate([np.asarray(prefix_strict, dtype=bool), strict_all[tail_idx]])
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
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        col for col in frame.select_dtypes(include=[np.number]).columns if col not in {*group_cols, "seed"}
    ]
    for keys, sub in frame.groupby(group_cols, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub else len(sub)
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_lo"] = float(np.quantile(values, 0.025))
            row[f"{col}_hi"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.n_hypotheses % args.batch_size:
        raise ValueError("n_hypotheses must be divisible by batch_size")
    secret = os.environ.get("CS1_LOCAL_API_KEY")
    if not args.prepare_only and not secret:
        raise RuntimeError("CS1_LOCAL_API_KEY is required and is passed only to child-process environments")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scored = load_results(args.all_tests, args.gt_top_frac)
    index = build_candidate_index(scored)
    registry = build_registry(scored)
    registry_path = args.out_dir / "cs1_public_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
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
            end_rank = start_rank + args.batch_size - 1
            batch_dir = seed_dir / f"batch_{start_rank:03d}_{end_rank:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = batch_dir / "prompt.txt"
            prompt_path.write_text(
                registry_prompt(method, registry, seed, start_rank, args.batch_size, previous),
                encoding="utf-8",
            )
            payload = None
            error = None
            if not args.prepare_only:
                try:
                    if method == "brainpilot_native":
                        assert brainpilot_url is not None
                        with brainpilot_locks[brainpilot_url]:
                            final_text = run_native_batch(
                                method, prompt_path, batch_dir, args, secret or "", brainpilot_url
                            )
                    else:
                        final_text = run_native_batch(method, prompt_path, batch_dir, args, secret or "")
                    payload = extract_json(final_text)
                    (batch_dir / "parsed.json").write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    items = payload.get("hypotheses") or []
                    if isinstance(items, list):
                        previous.extend(item for item in items if isinstance(item, dict))
                except Exception as exc:
                    error = f"batch_failed:{type(exc).__name__}"
                    (batch_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            batch_payloads.append((start_rank, end_rank, payload, error))
            print(f"{method} seed={seed} batch={start_rank}-{end_rank} status={error or 'ok'}", flush=True)
        if args.prepare_only:
            return None
        validated = validate_seed_outputs(method, seed, batch_payloads, registry)
        mapped = map_validated(validated, index)
        mapped.to_csv(seed_dir / "mapped_hypotheses.csv", index=False)
        valid_hypotheses = []
        for row in validated[validated["native_schema_valid"]].to_dict(orient="records"):
            valid_hypotheses.append(
                {
                    "rank": int(row["generated_rank"]),
                    "disease": row["generated_disease"],
                    "feature": row["generated_feature"],
                    "anatomy_query": row["generated_anatomy_query"],
                    "hemisphere": row["generated_hemisphere"],
                    "rationale": row["generated_rationale"],
                    "confidence": row["generated_confidence"],
                }
            )
        (seed_dir / "standardized.json").write_text(
            json.dumps({"method": method, "hypotheses": valid_hypotheses}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return mapped

    mapped_parts: list[pd.DataFrame] = []
    jobs = [(method, seed) for seed in args.seeds for method in args.methods]
    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(jobs)))) as pool:
        futures = {pool.submit(run_seed, method, seed): (method, seed) for method, seed in jobs}
        for future in as_completed(futures):
            mapped = future.result()
            if mapped is not None:
                mapped_parts.append(mapped)

    if args.prepare_only:
        print(args.out_dir)
        return
    mapped_all = pd.concat(mapped_parts, ignore_index=True) if mapped_parts else pd.DataFrame()
    mapped_all.to_csv(args.out_dir / "native_baselines_mapped_hypotheses.csv", index=False)
    n_gt = int(scored["is_gt_top"].sum())
    direct_budgets = [budget for budget in [20, 40, 60, 80] if budget <= args.n_hypotheses]
    direct = direct_seed_summary(mapped_all, n_gt, direct_budgets)
    direct.to_csv(args.out_dir / "native_baselines_direct_seed_summary.csv", index=False)
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
        "gt_top_frac": args.gt_top_frac,
        "gt_total": n_gt,
        "registry_counts": {
            "diseases": len(registry["diseases"]),
            "features": len(registry["features"]),
            "anatomy": len(registry["anatomy"]),
        },
        "interval": "2.5th to 97.5th percentile across 10 seeds",
        "failure_policy": "invalid, duplicate, missing, and unmappable outputs consume an experiment slot and receive zero hit; no manual repair",
        "blinding": "no GT, effect size, p/FDR, experiment labels, NeuroDiscovery score, or closed-loop feedback supplied",
        "credential_policy": "API key supplied only through process environment and never serialized",
    }
    (args.out_dir / "native_baselines_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
