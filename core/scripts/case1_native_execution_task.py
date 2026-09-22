"""Prepare and score a blinded native-execution task for Case Study 1.

The public bundle contains real TCP subject-level feature values and the exact
analysis contract. Hidden aggregate results are stored separately and are used
only by the deterministic scorer after a framework finishes its own code run.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from core.scripts.case1_exhaustive_full import build_atlas_feature_matrices
    from core.scripts.case1_exhaustive_v1 import disease_masks, load_metadata
    from core.scripts.case1_exhaustive_v2 import build_covariates, ols_case_effect
except ModuleNotFoundError:
    from case1_exhaustive_full import build_atlas_feature_matrices
    from case1_exhaustive_v1 import disease_masks, load_metadata
    from case1_exhaustive_v2 import build_covariates, ols_case_effect


DEFAULT_SOURCES = (
    "aal_116_multiatlas",
    "glasser_360_multiatlas",
    "schaefer_400_7net_multiatlas",
    "harvard_oxford_cort_multiatlas",
)
RESULT_FIELDS = (
    "n_case",
    "n_control",
    "adjusted_beta_case_minus_control",
    "adjusted_beta_se",
    "adjusted_t",
    "p_value",
    "adjusted_residual_d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--all-tests", type=Path, required=True)
    prepare.add_argument("--transdiag-root", type=Path, required=True)
    prepare.add_argument("--diagnosis", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--n-candidates", type=int, default=12)
    prepare.add_argument("--seed", type=int, default=20260807)
    prepare.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    prepare.add_argument(
        "--candidate-file",
        type=Path,
        help="Optional CSV/JSON/JSONL file of exact candidate IDs to preserve.",
    )

    execute = subparsers.add_parser("execute-neuroruntime")
    execute.add_argument("--bundle-public", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--bundle-dir", type=Path, required=True)
    reference.add_argument("--out", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--bundle-dir", type=Path, required=True)
    score.add_argument("--result-file", type=Path, required=True)
    score.add_argument("--method", required=True)
    score.add_argument("--trial", type=int, required=True)
    score.add_argument("--out-dir", type=Path, required=True)
    score.add_argument("--rtol", type=float, default=1e-4)
    score.add_argument("--atol", type=float, default=1e-7)
    return parser.parse_args()


def candidate_id(row: pd.Series | dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["modality"]),
            str(row["source"]),
            str(row["disease"]),
            str(row["feature"]),
            str(int(row["roi_index"])),
        ]
    )


def candidate_id_series(frame: pd.DataFrame) -> pd.Series:
    roi_index = pd.to_numeric(frame["roi_index"], errors="raise").astype("Int64")
    return (
        frame["modality"].astype(str)
        + "|"
        + frame["source"].astype(str)
        + "|"
        + frame["disease"].astype(str)
        + "|"
        + frame["feature"].astype(str)
        + "|"
        + roi_index.astype(str)
    )


@lru_cache(maxsize=8)
def _cached_atlas_features(
    transdiag_root: str,
    atlas: str,
) -> tuple[list[str], Any, dict[str, np.ndarray]]:
    return build_atlas_feature_matrices(
        Path(transdiag_root),
        atlas,
        requested_subjects=None,
    )


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def select_candidates(
    all_tests: pd.DataFrame,
    *,
    sources: list[str],
    n_candidates: int,
    seed: int,
) -> pd.DataFrame:
    frame = all_tests[
        all_tests["modality"].eq("fmri")
        & all_tests["source"].isin(sources)
        & np.isfinite(
            pd.to_numeric(all_tests["adjusted_residual_d"], errors="coerce")
        )
    ].copy()
    frame["candidate_id"] = candidate_id_series(frame)
    frame["selection_key"] = frame["candidate_id"].map(
        lambda value: _stable_key(seed, value)
    )
    frame = frame.sort_values("selection_key", kind="stable")

    source_quota = math.ceil(n_candidates / max(1, len(sources)))
    selected: list[pd.Series] = []
    source_counts: dict[str, int] = {}
    disease_counts: dict[str, int] = {}
    feature_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        source = str(row["source"])
        disease = str(row["disease"])
        feature = str(row["feature"])
        if source_counts.get(source, 0) >= source_quota:
            continue
        if disease_counts.get(disease, 0) >= 2:
            continue
        if feature_counts.get(feature, 0) >= 2:
            continue
        selected.append(row)
        source_counts[source] = source_counts.get(source, 0) + 1
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
        if len(selected) >= n_candidates:
            break
    if len(selected) != n_candidates:
        raise RuntimeError(
            f"Could select only {len(selected)} of {n_candidates} balanced candidates"
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def load_candidate_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        for column in ("candidate_id", "mapped_candidate_id"):
            if column in frame:
                values = frame[column].dropna().astype(str).tolist()
                break
        else:
            raise ValueError(f"{path} has no candidate ID column")
    elif path.suffix.lower() == ".jsonl":
        values = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                values.append(
                    str(row.get("candidate_id") or row.get("mapped_candidate_id") or "")
                )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = (
                payload.get("candidate_ids")
                or payload.get("hypotheses")
                or payload.get("candidates")
                or []
            )
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a list of candidate IDs")
        values = [
            str(
                item.get("candidate_id")
                or item.get("mapped_candidate_id")
                or ""
            )
            if isinstance(item, dict)
            else str(item)
            for item in payload
        ]

    values = [value.strip() for value in values if value.strip()]
    if not values:
        raise ValueError(f"{path} contains no candidate IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate candidate IDs")
    return values


def select_candidate_ids(
    all_tests: pd.DataFrame,
    candidate_ids: list[str],
) -> pd.DataFrame:
    frame = all_tests.copy()
    frame["candidate_id"] = candidate_id_series(frame)
    duplicate_ids = frame.loc[frame["candidate_id"].duplicated(), "candidate_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"all-tests contains duplicate IDs: {duplicate_ids[:3].tolist()}")
    indexed = frame.set_index("candidate_id", drop=False)
    missing = [value for value in candidate_ids if value not in indexed.index]
    if missing:
        raise KeyError(f"Candidate IDs absent from all-tests: {missing[:5]}")
    selected = indexed.loc[candidate_ids].copy().reset_index(drop=True)
    finite = np.isfinite(
        pd.to_numeric(selected["adjusted_residual_d"], errors="coerce")
    )
    if not bool(finite.all()):
        invalid = selected.loc[~finite, "candidate_id"].tolist()
        raise ValueError(f"Candidates lack finite reference results: {invalid[:5]}")
    return selected


def _scalar(stats_row: dict[str, Any], key: str) -> float:
    value = np.asarray(stats_row[key]).reshape(-1)
    return float(value[0])


def prepare_bundle(args: argparse.Namespace) -> int:
    if not args.candidate_file and args.n_candidates < 1:
        raise ValueError("--n-candidates must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    public_dir = args.out_dir / "public"
    hidden_dir = args.out_dir / "hidden"
    public_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    all_tests = pd.read_csv(args.all_tests, low_memory=False)
    if args.candidate_file:
        selected = select_candidate_ids(
            all_tests,
            load_candidate_ids(args.candidate_file),
        )
        selection_mode = "exact_candidate_ids"
    else:
        selected = select_candidates(
            all_tests,
            sources=list(args.sources),
            n_candidates=args.n_candidates,
            seed=args.seed,
        )
        selection_mode = "balanced_hash"

    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    candidate_manifest: list[dict[str, Any]] = []
    source_consistency_rows: list[dict[str, Any]] = []
    for source, group in selected.groupby("source", sort=False):
        atlas = str(source).removesuffix("_multiatlas")
        subjects, _roi_meta, feature_matrices = _cached_atlas_features(
            str(args.transdiag_root.resolve()),
            atlas,
        )
        meta, _diseases = load_metadata(args.diagnosis, subjects, min_cases=2)
        covariates = build_covariates(meta)
        aligned_subjects = meta["subjectkey"].astype(str).tolist()
        if aligned_subjects != subjects:
            index = {subject: idx for idx, subject in enumerate(subjects)}
            order = [index[subject] for subject in aligned_subjects]
        else:
            order = list(range(len(subjects)))

        for _, selected_row in group.iterrows():
            cid = str(selected_row["candidate_id"])
            disease = str(selected_row["disease"])
            feature = str(selected_row["feature"])
            roi_index = int(selected_row["roi_index"])
            matrix = feature_matrices[feature][order]
            if roi_index < 0 or roi_index >= matrix.shape[1]:
                raise IndexError(f"ROI index out of range for {cid}")
            values = matrix[:, roi_index]
            case_mask, control_mask = disease_masks(meta, disease)
            stats_row = ols_case_effect(
                values[:, None], case_mask, control_mask, covariates
            )
            gold = {
                "candidate_id": cid,
                "n_case": int(stats_row["n_case"]),
                "n_control": int(stats_row["n_control"]),
                "adjusted_beta_case_minus_control": _scalar(stats_row, "beta"),
                "adjusted_beta_se": _scalar(stats_row, "se"),
                "adjusted_t": _scalar(stats_row, "t"),
                "p_value": _scalar(stats_row, "p"),
                "adjusted_residual_d": _scalar(stats_row, "residual_d"),
            }
            expected_d = float(selected_row["adjusted_residual_d"])
            expected_p = float(selected_row["p_value"])
            d_matches = bool(np.isclose(
                gold["adjusted_residual_d"], expected_d, rtol=1e-8, atol=1e-10
            ))
            p_matches = bool(
                np.isclose(gold["p_value"], expected_p, rtol=1e-8, atol=1e-12)
            )
            source_consistency_rows.append(
                {
                    "candidate_id": cid,
                    "historical_adjusted_residual_d": expected_d,
                    "recomputed_adjusted_residual_d": gold[
                        "adjusted_residual_d"
                    ],
                    "historical_p_value": expected_p,
                    "recomputed_p_value": gold["p_value"],
                    "d_matches": d_matches,
                    "p_matches": p_matches,
                }
            )
            gold_rows.append(gold)
            candidate_manifest.append(
                {
                    "candidate_id": cid,
                    "disease": disease,
                    "feature": feature,
                    "source": source,
                    "roi_index": roi_index,
                    "anatomy_full": str(selected_row.get("anatomy_full") or ""),
                }
            )
            for idx, subject in enumerate(aligned_subjects):
                row: dict[str, Any] = {
                    "candidate_id": cid,
                    "subject_id": subject,
                    "case": int(case_mask[idx]),
                    "control": int(control_mask[idx] and not case_mask[idx]),
                    "value": float(values[idx]),
                }
                for column in covariates.columns:
                    row[column] = float(covariates.iloc[idx][column])
                public_rows.append(row)

    public = pd.DataFrame(public_rows)
    covariate_columns = [
        column
        for column in public.columns
        if column
        not in {"candidate_id", "subject_id", "case", "control", "value"}
    ]
    public.to_csv(public_dir / "tcp_subject_level_features.csv", index=False)
    (public_dir / "task_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "case1-native-execution-v1",
                "candidate_count": len(candidate_manifest),
                "candidates": candidate_manifest,
                "covariate_columns": covariate_columns,
                "required_result_fields": ["candidate_id", *RESULT_FIELDS],
                "blinding": {
                    "aggregate_results_exposed": False,
                    "gt_labels_exposed": False,
                    "neurodiscovery_scores_exposed": False,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (public_dir / "TASK.md").write_text(
        task_markdown(covariate_columns), encoding="utf-8"
    )
    (hidden_dir / "gold_results.json").write_text(
        json.dumps({"results": gold_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    consistency = pd.DataFrame(source_consistency_rows)
    consistency.to_csv(hidden_dir / "source_consistency_audit.csv", index=False)
    (hidden_dir / "provenance.json").write_text(
        json.dumps(
            {
                "all_tests": str(args.all_tests),
                "transdiag_root": str(args.transdiag_root),
                "diagnosis": str(args.diagnosis),
                "selection_seed": args.seed,
                "selection_mode": selection_mode,
                "candidate_file": str(args.candidate_file or ""),
                "selected_candidate_ids": selected["candidate_id"].tolist(),
                "selection_uses_effect_magnitude_or_significance": False,
                "eligibility_requires_finite_reference_result": True,
                "hidden_gold_source": "recomputed_from_public_subject_level_data",
                "historical_source_matches": int(
                    (consistency["d_matches"] & consistency["p_matches"]).sum()
                ),
                "historical_source_mismatches": int(
                    (~(consistency["d_matches"] & consistency["p_matches"])).sum()
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(public_dir)
    return 0


def execute_neuroruntime(args: argparse.Namespace) -> int:
    manifest = json.loads(
        (args.bundle_public / "task_manifest.json").read_text(encoding="utf-8")
    )
    public = pd.read_csv(args.bundle_public / "tcp_subject_level_features.csv")
    covariates = [str(value) for value in manifest.get("covariate_columns", [])]
    results: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates", []):
        cid = str(candidate["candidate_id"])
        group = public.loc[public["candidate_id"].astype(str).eq(cid)].copy()
        group = group.reset_index(drop=True)
        case_mask = group["case"].astype(int).eq(1).to_numpy()
        control_mask = group["control"].astype(int).eq(1).to_numpy()
        stats_row = ols_case_effect(
            group["value"].to_numpy(float)[:, None],
            case_mask,
            control_mask,
            group[covariates].astype(float),
        )
        results.append(
            {
                "candidate_id": cid,
                "n_case": int(stats_row["n_case"]),
                "n_control": int(stats_row["n_control"]),
                "adjusted_beta_case_minus_control": _scalar(stats_row, "beta"),
                "adjusted_beta_se": _scalar(stats_row, "se"),
                "adjusted_t": _scalar(stats_row, "t"),
                "p_value": _scalar(stats_row, "p"),
                "adjusted_residual_d": _scalar(stats_row, "residual_d"),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.out)
    return 0


def task_markdown(covariates: list[str]) -> str:
    cov_text = ", ".join(covariates)
    return f"""# Blinded TCP native-execution task

Analyze every candidate in `tcp_subject_level_features.csv` and return one
result object per candidate. Do not use external result tables or hidden files.

For each candidate:

1. Keep rows where `case == 1` or `control == 1`.
   These indicators are mutually exclusive; after filtering, `case == 0`
   denotes the control group.
2. Drop rows with non-finite `value` or covariates.
3. Fit OLS with columns `[intercept, case, {cov_text}]`.
4. Report the case coefficient, its standard error, t statistic, and two-sided
   Student-t p-value using residual degrees of freedom.
5. Separately residualize `value` on `[intercept, {cov_text}]` and calculate
   pooled-SD Cohen's d for case residuals minus control residuals.

Use `numpy.linalg.pinv`. Write valid JSON with this exact outer schema:

```json
{{
  "results": [
    {{
      "candidate_id": "...",
      "n_case": 10,
      "n_control": 20,
      "adjusted_beta_case_minus_control": 0.0,
      "adjusted_beta_se": 0.0,
      "adjusted_t": 0.0,
      "p_value": 1.0,
      "adjusted_residual_d": 0.0
    }}
  ]
}}
```

The task is complete only when every registered candidate has exactly one
finite result. Do not omit null or negative findings.
"""


def reference_result(bundle_dir: Path) -> dict[str, Any]:
    return json.loads(
        (bundle_dir / "hidden" / "gold_results.json").read_text(encoding="utf-8")
    )


def write_reference(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(reference_result(args.bundle_dir), indent=2), encoding="utf-8"
    )
    print(args.out)
    return 0


def extract_json(text: str) -> dict[str, Any]:
    tagged = re.findall(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.S | re.I)
    streamed_chunks = re.findall(
        r"^\[TEXT_MESSAGE_CONTENT\]\s?(.*)$", text, flags=re.M
    )
    streamed = "".join(streamed_chunks)
    candidates = [*reversed(tagged)]
    if streamed:
        candidates.append(streamed)
    candidates.append(text.strip())
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        decoded: list[tuple[int, int, dict[str, Any]]] = []
        for start, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, length = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                decoded.append((start + length, length, value))
        if decoded:
            return max(decoded, key=lambda item: (item[0], item[1]))[2]
    raise ValueError("No JSON object found in native result")


def score_results(args: argparse.Namespace) -> int:
    gold_payload = reference_result(args.bundle_dir)
    gold = {
        str(row["candidate_id"]): row for row in gold_payload.get("results", [])
    }
    parse_error = ""
    try:
        payload = extract_json(args.result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {}
        parse_error = f"{type(exc).__name__}: {exc}"
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raw_results = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in raw_results:
        if isinstance(row, dict):
            by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(row)

    unknown_result_count = sum(
        len(rows) for cid, rows in by_candidate.items() if cid not in gold
    )

    audit_rows = []
    for cid, expected in gold.items():
        candidates = by_candidate.get(cid, [])
        errors: list[str] = []
        if len(candidates) != 1:
            errors.append("missing_or_duplicate_result")
            observed: dict[str, Any] = {}
        else:
            observed = candidates[0]
        for field in RESULT_FIELDS:
            if field in {"n_case", "n_control"}:
                try:
                    if int(observed.get(field)) != int(expected[field]):
                        errors.append(f"mismatch:{field}")
                except (TypeError, ValueError):
                    errors.append(f"invalid:{field}")
                continue
            try:
                value = float(observed.get(field))
            except (TypeError, ValueError):
                errors.append(f"invalid:{field}")
                continue
            if not np.isfinite(value):
                errors.append(f"nonfinite:{field}")
            elif not np.isclose(
                value, float(expected[field]), rtol=args.rtol, atol=args.atol
            ):
                errors.append(f"mismatch:{field}")
        audit_rows.append(
            {
                "method": args.method,
                "trial": args.trial,
                "candidate_id": cid,
                "execution_success": not errors,
                "errors": ";".join(errors),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out_dir / "candidate_execution_audit.csv", index=False)
    successes = int(audit["execution_success"].sum())
    summary = {
        "method": args.method,
        "trial": args.trial,
        "attempted_candidates": len(audit),
        "successful_candidates": successes,
        "execution_success_rate": successes / len(audit) if len(audit) else 0.0,
        "unknown_result_count": unknown_result_count,
        "complete_valid_submission": bool(
            len(audit) and successes == len(audit) and unknown_result_count == 0
        ),
        "parse_error": parse_error,
        "human_repairs": 0,
        "result_file": str(args.result_file),
    }
    (args.out_dir / "execution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare_bundle(args)
    if args.command == "execute-neuroruntime":
        return execute_neuroruntime(args)
    if args.command == "reference":
        return write_reference(args)
    if args.command == "score":
        return score_results(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
