from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_temporal_kg_snapshot import build_snapshot


@dataclass(frozen=True)
class Window:
    freeze_year: int
    future_start_year: int
    future_end_year: int

    @property
    def label(self) -> str:
        return f"kg{self.freeze_year}_to_{self.future_start_year}_{self.future_end_year}"


DEFAULT_WINDOWS = (
    Window(2016, 2017, 2018),
    Window(2018, 2019, 2020),
    Window(2020, 2021, 2022),
    Window(2022, 2023, 2024),
)


def parse_window(raw: str) -> Window:
    parts = raw.replace(",", ":").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"window must be freeze:start:end, got {raw!r}"
        )
    try:
        freeze, start, end = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"window years must be integers, got {raw!r}"
        ) from exc
    if not (freeze < start <= end):
        raise argparse.ArgumentTypeError(
            f"window must satisfy freeze < start <= end, got {raw!r}"
        )
    return Window(freeze, start, end)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(
    *,
    kg_path: Path,
    future_claims: Path,
    output_dir: Path,
    window: Window,
    max_candidates: int,
    classification_max_examples: int,
    seed: int,
    candidate_claim_edge_policy: str,
    candidate_terminal_support_weight: float,
    candidate_kge_path_weight: float,
    classification_shared_weight: float,
    classification_path_weight: float,
    classification_endpoint_weight: float,
    classification_kge_weight: float,
    kge_checkpoint: Path | None,
    strict_candidate_anchors: bool,
    force: bool,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        return load_json(metrics_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("case3_hindcasting_eval.py")),
        "--kg",
        str(kg_path),
        "--future-claims",
        str(future_claims),
        "--output-dir",
        str(output_dir),
        "--freeze-year",
        str(window.freeze_year),
        "--future-start-year",
        str(window.future_start_year),
        "--future-end-year",
        str(window.future_end_year),
        "--max-candidates",
        str(max_candidates),
        "--classification-max-examples",
        str(classification_max_examples),
        "--seed",
        str(seed),
        "--candidate-claim-edge-policy",
        candidate_claim_edge_policy,
        "--candidate-terminal-support-weight",
        str(candidate_terminal_support_weight),
        "--candidate-kge-path-weight",
        str(candidate_kge_path_weight),
        "--classification-shared-weight",
        str(classification_shared_weight),
        "--classification-path-weight",
        str(classification_path_weight),
        "--classification-endpoint-weight",
        str(classification_endpoint_weight),
        "--classification-kge-weight",
        str(classification_kge_weight),
    ]
    if kge_checkpoint is not None:
        cmd.extend(["--kge-checkpoint", str(kge_checkpoint)])
    if strict_candidate_anchors:
        cmd.append("--strict-candidate-anchors")
    subprocess.run(cmd, check=True)
    return load_json(metrics_path)


def train_kge(
    *,
    kg_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    negatives_per_pos: int,
    eval_every: int,
    early_stop_patience: int,
    min_confidence: float,
    seed: int,
    device: str | None,
    force: bool,
) -> dict[str, Any]:
    if checkpoint_path.is_file() and report_path.is_file() and not force:
        return load_json(report_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "neurooracle.phase4",
        "kge-train",
        "--kg",
        str(kg_path),
        "--output",
        str(checkpoint_path),
        "--report",
        str(report_path),
        "--dim",
        str(dim),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--negatives-per-pos",
        str(negatives_per_pos),
        "--eval-every",
        str(eval_every),
        "--early-stop-patience",
        str(early_stop_patience),
        "--min-confidence",
        str(min_confidence),
        "--seed",
        str(seed),
    ]
    if device:
        cmd.extend(["--device", device])
    subprocess.run(cmd, check=True)
    return load_json(report_path)


def flatten_row(
    window: Window,
    snapshot_manifest: dict[str, Any],
    metrics: dict[str, Any],
    kge_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    future_stats = metrics.get("future_pool_stats") or {}
    recall = metrics.get("forward_recall") or {}
    cls = metrics.get("classification_proxy") or {}
    return {
        "freeze_year": window.freeze_year,
        "future_window": f"{window.future_start_year}-{window.future_end_year}",
        "snapshot_concepts": (snapshot_manifest.get("output_stats") or {}).get("n_concepts"),
        "snapshot_edges": (snapshot_manifest.get("output_stats") or {}).get("n_edges"),
        "historical_claims": snapshot_manifest.get("claims_kept_year_le_cutoff"),
        "future_claims_total": future_stats.get("future_claims_total"),
        "future_evaluable_novel": future_stats.get("future_evaluable_novel"),
        "evaluable_gene_disease": future_stats.get("evaluable_gene_disease"),
        "evaluable_gene_imaging": future_stats.get("evaluable_gene_imaging"),
        "evaluable_imaging_disease": future_stats.get("evaluable_imaging_disease"),
        "already_direct_in_frozen_kg": future_stats.get("already_direct_in_frozen_kg")
        or future_stats.get("already_direct_in_kg2020"),
        "candidate_count": metrics.get("candidate_count"),
        "recall@100": recall.get("recall@100"),
        "recall@1000": recall.get("recall@1000"),
        "terminal_gene_disease_recall@100": recall.get("terminal_gene_disease_recall@100"),
        "terminal_gene_disease_recall@1000": recall.get("terminal_gene_disease_recall@1000"),
        "terminal_gene_disease_hits@1000": recall.get("terminal_gene_disease_hits@1000"),
        "terminal_gene_disease_total@1000": recall.get("terminal_gene_disease_total@1000"),
        "auc_proxy_decoy": cls.get("auc"),
        "auprc_proxy_decoy": cls.get("auprc"),
        "positive_mean_score": cls.get("positive_mean_score"),
        "negative_mean_score": cls.get("negative_mean_score"),
        "positive_mean_kge_score": cls.get("positive_mean_kge_score"),
        "negative_mean_kge_score": cls.get("negative_mean_kge_score"),
        "kge_checkpoint": metrics.get("kge_checkpoint"),
        "kge_train_triples": (kge_report or {}).get("n_triples"),
        "kge_test_auroc": (kge_report or {}).get("test_auroc"),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]], windows: list[Window]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"windows": [w.__dict__ for w in windows], "rows": rows}, f, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0]) if rows else []
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Case Study 3 Rolling Hindcasting",
        "",
        "Positive pools are future claims in each horizon. Classification negatives are type-preserving decoys, not hard negative scientific claims.",
        "",
        "| Freeze | Future | Evaluable | Recall@1000 | Terminal GD Recall@1000 | AUC proxy | AUPRC proxy | KGE test AUROC | KGE pos mean | KGE decoy mean |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {freeze_year} | {future_window} | {future_evaluable_novel} | {recall1000} | {gd1000} | {auc} | {auprc} | {kge_auc} | {kge_pos} | {kge_neg} |".format(
                freeze_year=row["freeze_year"],
                future_window=row["future_window"],
                future_evaluable_novel=row["future_evaluable_novel"],
                recall1000=_fmt(row["recall@1000"]),
                gd1000=_fmt(row["terminal_gene_disease_recall@1000"]),
                auc=_fmt(row["auc_proxy_decoy"]),
                auprc=_fmt(row["auprc_proxy_decoy"]),
                kge_auc=_fmt(row.get("kge_test_auroc")),
                kge_pos=_fmt(row.get("positive_mean_kge_score")),
                kge_neg=_fmt(row.get("negative_mean_kge_score")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- The short-horizon windows are the fairest tests of next-step field development.",
            "- Later windows should be read as rolling forecasts from their own frozen KG, not as KG_2020 predicting every downstream discovery.",
            "- Decoy classification is a sanity check. It should be replaced or supplemented with retraction/null-result/low-citation labels when available.",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-window Case Study 3 hindcasting evaluations.")
    parser.add_argument("--input-dir", type=Path, default=Path("neurooracle/data/full_snapshot_v1"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("neurooracle/data/snapshots"))
    parser.add_argument("--output-root", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/rolling_windows_full_snapshot_v1"))
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS), help="Windows as freeze:start:end, e.g. 2020:2021:2022.")
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--classification-max-examples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--candidate-claim-edge-policy", choices=("prefer", "require-any", "require-all", "off"), default="require-any")
    parser.add_argument("--candidate-terminal-support-weight", type=float, default=0.20)
    parser.add_argument("--candidate-kge-path-weight", type=float, default=0.0)
    parser.add_argument("--classification-shared-weight", type=float, default=1.0)
    parser.add_argument("--classification-path-weight", type=float, default=0.0)
    parser.add_argument("--classification-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--classification-kge-weight", type=float, default=0.0)
    parser.add_argument("--train-kge", action="store_true")
    parser.add_argument("--kge-root", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/kge_checkpoints_full_snapshot_v1"))
    parser.add_argument("--kge-dim", type=int, default=64)
    parser.add_argument("--kge-epochs", type=int, default=20)
    parser.add_argument("--kge-batch-size", type=int, default=4096)
    parser.add_argument("--kge-lr", type=float, default=1e-3)
    parser.add_argument("--kge-negatives-per-pos", type=int, default=5)
    parser.add_argument("--kge-eval-every", type=int, default=5)
    parser.add_argument("--kge-early-stop-patience", type=int, default=2)
    parser.add_argument("--kge-min-confidence", type=float, default=0.2)
    parser.add_argument("--kge-device", default=None)
    parser.add_argument("--strict-candidate-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Rebuild snapshots and rerun window metrics even when outputs exist.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    future_claims = args.input_dir / "extracted_claims.jsonl"
    for window in args.windows:
        snapshot_dir = args.snapshot_root / f"kg_{window.freeze_year}_from_full_snapshot_v1"
        snapshot_manifest_path = snapshot_dir / "manifest.json"
        if args.force or not snapshot_manifest_path.is_file():
            print(f"[snapshot] building KG_{window.freeze_year} -> {snapshot_dir}", flush=True)
            snapshot_manifest = build_snapshot(args.input_dir, snapshot_dir, window.freeze_year)
        else:
            print(f"[snapshot] using existing KG_{window.freeze_year} -> {snapshot_dir}", flush=True)
            snapshot_manifest = load_json(snapshot_manifest_path)

        run_dir = args.output_root / window.label
        kge_report = None
        kge_checkpoint = None
        if args.train_kge:
            kge_checkpoint = args.kge_root / f"kg_{window.freeze_year}_complex.pt"
            kge_report_path = args.kge_root / f"kg_{window.freeze_year}_complex_report.json"
            print(f"[kge] training/loading KG_{window.freeze_year} ComplEx -> {kge_checkpoint}", flush=True)
            kge_report = train_kge(
                kg_path=snapshot_dir / "knowledge_graph.json",
                checkpoint_path=kge_checkpoint,
                report_path=kge_report_path,
                dim=args.kge_dim,
                epochs=args.kge_epochs,
                batch_size=args.kge_batch_size,
                lr=args.kge_lr,
                negatives_per_pos=args.kge_negatives_per_pos,
                eval_every=args.kge_eval_every,
                early_stop_patience=args.kge_early_stop_patience,
                min_confidence=args.kge_min_confidence,
                seed=args.seed,
                device=args.kge_device,
                force=args.force,
            )
        print(
            f"[eval] KG_{window.freeze_year} -> claims {window.future_start_year}-{window.future_end_year}",
            flush=True,
        )
        metrics = run_eval(
            kg_path=snapshot_dir / "knowledge_graph.json",
            future_claims=future_claims,
            output_dir=run_dir,
            window=window,
            max_candidates=args.max_candidates,
            classification_max_examples=args.classification_max_examples,
            seed=args.seed,
            candidate_claim_edge_policy=args.candidate_claim_edge_policy,
            candidate_terminal_support_weight=args.candidate_terminal_support_weight,
            candidate_kge_path_weight=args.candidate_kge_path_weight,
            classification_shared_weight=args.classification_shared_weight,
            classification_path_weight=args.classification_path_weight,
            classification_endpoint_weight=args.classification_endpoint_weight,
            classification_kge_weight=args.classification_kge_weight,
            kge_checkpoint=kge_checkpoint,
            strict_candidate_anchors=args.strict_candidate_anchors,
            force=args.force,
        )
        rows.append(flatten_row(window, snapshot_manifest, metrics, kge_report))

    write_summary(args.output_root, rows, list(args.windows))
    print(f"[done] wrote rolling summary to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
