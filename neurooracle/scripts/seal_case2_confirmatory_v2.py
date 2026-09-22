"""Verify and seal the completed Case Study 2 v2 confirmation run.

The seal is additive: it never changes the frozen protocol, rankings, hidden
confirmation results, or generator evaluation.  It binds those artifacts into
one final audit record and records the strict negative endpoint without
reinterpreting missing positives as zero-valued ranking performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from neurooracle.scripts.case2_confirmatory_statistics import (
    global_chain_labels,
    pathway_outcome_family_labels,
)


CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_PROTOCOL_ROOT = CASE2_ROOT / "protocols" / "case2_adni_endpoint_holdout_v2"
EXPECTED_METHODS = {
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
    "neurodiscovery",
}


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


def _verify_confirmation_results(
    result_root: Path,
    *,
    freeze_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_lock_path = result_root / "CONFIRMATION_RESULTS.lock.json"
    result_manifest_path = result_root / "manifest.json"
    result_lock = _load_json(result_lock_path)
    result_manifest = _load_json(result_manifest_path)
    if result_lock["protocol_freeze_id"] != freeze_id:
        raise ValueError("confirmation result lock belongs to another protocol freeze")
    if result_lock.get("do_not_retune_after_this_lock") is not True:
        raise ValueError("confirmation result lock does not prohibit post-outcome tuning")

    result_parquet = result_root / "case2_confirmation_results.parquet"
    result_csv = result_root / "case2_confirmation_results.csv"
    analysis_rows = result_root / "case2_confirmation_analysis_rows.parquet"
    for path, key, label in (
        (result_parquet, "results_sha256", "confirmation parquet"),
        (result_csv, "results_csv_sha256", "confirmation CSV"),
        (analysis_rows, "analysis_rows_sha256", "confirmation analysis rows"),
        (result_manifest_path, "manifest_sha256", "confirmation manifest"),
    ):
        _require_sha(path, str(result_lock[key]), label)

    results = pd.read_csv(result_csv)
    alpha = 0.05
    family_labels, family_q = pathway_outcome_family_labels(results, alpha)
    global_labels, global_q = global_chain_labels(results, alpha)
    nominal_labels = (
        pd.to_numeric(results["sobel_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
    )
    if not np.array_equal(
        family_labels, results["family_fdr_chain_hit"].astype(bool).to_numpy()
    ):
        raise ValueError("stored family-FDR labels do not reproduce")
    if not np.array_equal(
        global_labels, results["global_fdr_chain_hit"].astype(bool).to_numpy()
    ):
        raise ValueError("stored global-FDR labels do not reproduce")
    if not np.array_equal(
        nominal_labels, results["nominal_chain_hit"].astype(bool).to_numpy()
    ):
        raise ValueError("stored nominal labels do not reproduce")
    if not np.allclose(
        family_q,
        pd.to_numeric(results["sobel_q_family"], errors="coerce").to_numpy(),
    ):
        raise ValueError("stored family-FDR q values do not reproduce")
    if not np.allclose(
        global_q,
        pd.to_numeric(results["sobel_q_global"], errors="coerce").to_numpy(),
    ):
        raise ValueError("stored global-FDR q values do not reproduce")

    counts = {
        "candidate_count": int(len(results)),
        "estimated_candidates": int(results["executable"].astype(bool).sum()),
        "nominal_chain_hits": int(nominal_labels.sum()),
        "family_fdr_chain_hits": int(family_labels.sum()),
        "global_fdr_chain_hits": int(global_labels.sum()),
        "minimum_family_q": float(np.min(family_q)),
        "minimum_global_q": float(np.min(global_q)),
    }
    expected = {
        "candidate_count": int(result_manifest["candidate_count"]),
        "estimated_candidates": int(result_manifest["estimated_candidates"]),
        "nominal_chain_hits": int(result_manifest["nominal_chain_hits"]),
        "family_fdr_chain_hits": int(
            result_manifest["pathway_outcome_family_fdr_chain_hits"]
        ),
        "global_fdr_chain_hits": int(result_manifest["global_fdr_chain_hits"]),
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(
                f"confirmation manifest mismatch for {key}: {value} != {counts[key]}"
            )
    return counts, {
        "result_lock": _pin(result_lock_path),
        "result_manifest": _pin(result_manifest_path),
        "results_parquet": _pin(result_parquet),
        "results_csv": _pin(result_csv),
        "analysis_rows": _pin(analysis_rows),
    }


def _verify_generator_evaluation(
    evaluation_root: Path,
    *,
    freeze_id: str,
    confirmation_result_lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = evaluation_root / "manifest.json"
    verdict_path = evaluation_root / "case2_sota_verdict.json"
    manifest = _load_json(manifest_path)
    verdict = _load_json(verdict_path)
    if manifest["protocol_freeze_id"] != freeze_id:
        raise ValueError("generator evaluation belongs to another protocol freeze")
    if manifest["confirmation_result_lock_sha256"] != confirmation_result_lock_sha256:
        raise ValueError("generator evaluation is not bound to the current result lock")
    if set(manifest["methods"]) != EXPECTED_METHODS:
        raise ValueError("generator evaluation does not contain the seven expected methods")
    if any(int(value) != 10 for value in manifest["trials_per_method"].values()):
        raise ValueError("generator evaluation does not contain ten trials per method")
    if verdict != manifest["verdict"]:
        raise ValueError("standalone SOTA verdict differs from the evaluation manifest")
    if verdict.get("verdict_status") != (
        "not_identifiable_no_family_fdr_positive_labels"
    ):
        raise ValueError("unexpected formal SOTA verdict")
    if verdict.get("sota_evaluable") is not False or verdict.get("clear_sota") is not False:
        raise ValueError("zero-positive endpoint was incorrectly treated as evaluable SOTA")

    output_pins: dict[str, Any] = {}
    for name, raw_path in manifest["outputs"].items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing generator evaluation output: {path}")
        output_pins[name] = _pin(path)
    for record in manifest["neurodiscovery_batch_commit_manifests"]:
        _require_sha(Path(record["path"]), record["sha256"], "ND batch commits")
        _require_sha(Path(record["trace_path"]), record["trace_sha256"], "ND trace")
    for record in manifest["neurodiscovery_overlay_manifests"]:
        _require_sha(Path(record["path"]), record["sha256"], "ND overlay")
    return verdict, {
        "evaluation_manifest": _pin(manifest_path),
        "sota_verdict": _pin(verdict_path),
        "evaluation_outputs": output_pins,
    }


def build_seal(protocol_root: Path, run_root: Path) -> dict[str, Any]:
    protocol_manifest_path = protocol_root / "protocol_freeze_manifest.json"
    protocol_lock_path = protocol_root / "FREEZE.lock.json"
    protocol_manifest = _load_json(protocol_manifest_path)
    protocol_lock = _load_json(protocol_lock_path)
    freeze_id = str(protocol_manifest["freeze_id"])
    if protocol_lock["freeze_id"] != freeze_id:
        raise ValueError("protocol lock freeze ID mismatch")
    _require_sha(
        protocol_manifest_path,
        str(protocol_lock["manifest_sha256"]),
        "protocol freeze manifest",
    )
    if run_root.name != freeze_id[:12]:
        raise ValueError("run directory does not match the protocol freeze ID")

    result_root = run_root / "hidden_confirmation_results"
    counts, result_pins = _verify_confirmation_results(
        result_root,
        freeze_id=freeze_id,
    )
    result_lock_sha = result_pins["result_lock"]["sha256"]
    verdict, evaluation_pins = _verify_generator_evaluation(
        run_root / "generator_evaluation",
        freeze_id=freeze_id,
        confirmation_result_lock_sha256=result_lock_sha,
    )

    result_manifest = _load_json(result_root / "manifest.json")
    ranking_locks = {
        "baseline_rankings": run_root
        / "official_baselines_gpt55_high"
        / "BASELINE_RANKINGS.lock.json",
        "neurodiscovery_initial_ranking": run_root
        / "neurodiscovery_policy"
        / "INITIAL_RANKING.lock.json",
        "evaluation_plan": run_root / "generator_evaluation" / "EVALUATION_PLAN.lock.json",
    }
    expected_ranking_hashes = {
        "baseline_rankings": result_manifest["lock_hashes"]["baseline_ranking_lock"],
        "neurodiscovery_initial_ranking": result_manifest["lock_hashes"][
            "neurodiscovery_initial_ranking_lock"
        ],
        "evaluation_plan": result_manifest["lock_hashes"][
            "generator_evaluation_plan_lock"
        ],
    }
    for name, path in ranking_locks.items():
        _require_sha(path, expected_ranking_hashes[name], name)

    seal = {
        "schema_version": "neurooracle.case2_final_result_seal.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "sealed_strict_negative_primary_endpoint",
        "protocol_id": protocol_manifest["protocol_id"],
        "protocol_freeze_id": freeze_id,
        "run_root": str(run_root),
        "result_summary": counts,
        "formal_verdict": verdict,
        "interpretation": {
            "scientific_result": (
                "No registered chain passed pathway-outcome family FDR or global FDR."
            ),
            "generator_result": (
                "Formal SOTA is not identifiable because the strict endpoint has no "
                "positive labels; exploratory nominal metrics must remain secondary."
            ),
            "not_equivalent_to_falsification": True,
        },
        "immutability_policy": {
            "modify_v2_protocol": False,
            "retune_v2_after_outcome_access": False,
            "relax_v2_thresholds": False,
            "future_changes_require_new_protocol_id": True,
        },
        "pins": {
            "protocol_lock": _pin(protocol_lock_path),
            "protocol_manifest": _pin(protocol_manifest_path),
            "ranking_locks": {
                name: _pin(path) for name, path in ranking_locks.items()
            },
            "confirmation": result_pins,
            "generator_evaluation": evaluation_pins,
        },
    }
    seal["content_sha256"] = _canonical_sha256(seal)
    return seal


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_manifest = _load_json(args.protocol_root / "protocol_freeze_manifest.json")
    run_root = args.run_root or (
        CASE2_ROOT
        / "experiments"
        / "case2_adni_confirmatory_v1"
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
        stable_fields = (
            "protocol_id",
            "protocol_freeze_id",
            "run_root",
            "result_summary",
            "formal_verdict",
            "interpretation",
            "immutability_policy",
            "pins",
        )
        changed = [field for field in stable_fields if existing.get(field) != seal.get(field)]
        if changed:
            raise ValueError(
                "current artifacts differ from the existing final seal: "
                + ", ".join(changed)
            )
        print(
            json.dumps(
                {"execution_status": "verified_existing", "output": str(output_path)},
                indent=2,
            )
        )
        return existing
    output_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"execution_status": "created", "output": str(output_path), **seal},
            indent=2,
        )
    )
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


# Last Updated At: 2026-08-16 12:18 HKT
