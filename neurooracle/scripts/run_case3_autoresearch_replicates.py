from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "neurooracle/data/experiments/case3/autoresearch_replicates"
DEFAULT_KG = ROOT / "neurooracle/data/experiments/case3/snapshots_full_v2_5year_2016_2020/kg_2020/knowledge_graph.json"
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
TOP_KS = (1, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000)
PUBLICATION_FIELDS = (
    "paper_title",
    "paper_year",
    "venue",
    "doi",
    "source_url",
    "baseline_family",
    "adaptation_note",
)


METHODS: dict[str, dict[str, Any]] = {
    "ai_scientist": {
        "script": "generate_ai_scientist_gid_hypotheses.py",
        "batch_size": 20,
        "label": "AI Scientist",
        "paper_title": "Towards end-to-end automation of AI research",
        "paper_year": 2026,
        "venue": "Nature",
        "doi": "10.1038/s41586-026-10265-5",
        "source_url": "https://www.nature.com/articles/s41586-026-10265-5",
        "baseline_family": "single-system autoresearch agent",
        "adaptation_note": (
            "Adapts the AI Scientist ideation stage to fixed-budget "
            "GENE_TARGET -> IMAGING_MARKER -> DISEASE hypothesis generation; "
            "no NeuroOracle graph traversal or future labels are exposed."
        ),
    },
    "co_scientist": {
        "script": "generate_coscientist_style_gid_hypotheses.py",
        "batch_size": 50,
        "label": "Co-Scientist",
        "paper_title": "Accelerating scientific discovery with Co-Scientist",
        "paper_year": 2026,
        "venue": "Nature",
        "doi": "10.1038/s41586-026-10644-y",
        "source_url": "https://www.nature.com/articles/s41586-026-10644-y",
        "baseline_family": "multi-agent hypothesis generation",
        "adaptation_note": (
            "Adapts the generate-critique-refine/rank hypothesis workflow "
            "to the Case Study 3 schema; no NeuroOracle graph traversal or "
            "future labels are exposed."
        ),
    },
    "data_to_paper": {
        "script": "generate_data_to_paper_style_gid_hypotheses.py",
        "batch_size": 25,
        "label": "data-to-paper",
        "paper_title": "Autonomous LLM-Driven Research - from Data to Human-Verifiable Research Papers",
        "paper_year": 2025,
        "venue": "NEJM AI",
        "doi": "10.1056/AIoa2400555",
        "source_url": "https://ai.nejm.org/doi/abs/10.1056/AIoa2400555",
        "baseline_family": "traceable data-to-paper autonomous research workflow",
        "adaptation_note": (
            "Adapts the research-question/data-schema/planned-claim workflow "
            "to produce a ranked hypothesis table instead of a manuscript; "
            "no NeuroOracle graph traversal or future labels are exposed."
        ),
    },
    "sciagents": {
        "script": "generate_sciagents_style_gid_hypotheses.py",
        "batch_size": 20,
        "label": "SciAgents",
        "paper_title": "SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning",
        "paper_year": 2025,
        "venue": "Advanced Materials",
        "doi": "10.1002/adma.202413523",
        "source_url": "https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adma.202413523",
        "baseline_family": "knowledge-graph-guided multi-agent reasoning",
        "adaptation_note": (
            "Adapts graph-ontologist/scientist/critic/ranker reasoning over "
            "the frozen historical KG context; NeuroDiscovery scores, future "
            "labels, and graph-degree-only ranking are not exposed."
        ),
        "extra_args": ["--cards-per-batch", "30"],
    },
    "virtual_lab_style": {
        "script": "generate_virtual_lab_style_gid_hypotheses.py",
        "batch_size": 25,
        "label": "Virtual Lab-style",
        "paper_title": "The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies",
        "paper_year": 2025,
        "venue": "Nature",
        "doi": "10.1038/s41586-025-09442-9",
        "source_url": "https://www.nature.com/articles/s41586-025-09442-9",
        "baseline_family": "multi-agent virtual scientific team",
        "adaptation_note": (
            "Adapts only the compact PI/scientist-agent meeting workflow to "
            "fixed-budget GENE_TARGET -> IMAGING_MARKER -> DISEASE hypothesis "
            "generation; wet-lab validation, nanobody-specific tools, "
            "structural-biology tools, NeuroOracle graph traversal, and future "
            "labels are not exposed."
        ),
        "extra_args": ["--retries", "8", "--retry-sleep", "8", "--max-tokens", "12000"],
    },
    "openscholar_rag": {
        "script": "generate_openscholar_rag_gid_hypotheses.py",
        "batch_size": 25,
        "label": "OpenScholar-RAG",
        "paper_title": "Synthesizing scientific literature with retrieval-augmented language models",
        "paper_year": 2026,
        "venue": "Nature",
        "doi": "10.1038/s41586-025-10072-4",
        "source_url": "https://www.nature.com/articles/s41586-025-10072-4",
        "baseline_family": "retrieval-augmented scientific literature synthesis",
        "adaptation_note": (
            "Adapts the retrieval-augmented literature-synthesis workflow to "
            "generate citation-backed ranked GID hypotheses from pre-freeze "
            "literature snippets only; post-freeze literature, NeuroOracle graph "
            "traversal, and future labels are not exposed."
        ),
        "extra_args": [
            "--claims",
            str(DEFAULT_CLAIMS),
            "--freeze-year",
            "2020",
            "--retries",
            "12",
            "--retry-sleep",
            "8",
            "--max-tokens",
            "6000",
            "--passages-per-batch",
            "18",
        ],
    },
    # Backward-compatible aliases for runs created before the baseline
    # registry was switched from in-house style names to paper-published names.
    "ai_scientist_v2": {
        "alias_for": "ai_scientist",
        "label": "AI Scientist",
    },
    "co_scientist_style": {
        "alias_for": "co_scientist",
        "label": "Co-Scientist",
    },
    "data_to_paper_style": {
        "alias_for": "data_to_paper",
        "label": "data-to-paper",
    },
    "sciagents_style": {
        "alias_for": "sciagents",
        "label": "SciAgents",
    },
    "virtual_lab": {
        "alias_for": "virtual_lab_style",
        "label": "Virtual Lab-style",
    },
    "openscholar": {
        "alias_for": "openscholar_rag",
        "label": "OpenScholar-RAG",
    },
}

DEFAULT_METHODS = [
    "ai_scientist_v2",
    "co_scientist_style",
    "data_to_paper_style",
    "sciagents_style",
    "virtual_lab_style",
    "openscholar_rag",
]


def method_config(method: str) -> dict[str, Any]:
    cfg = METHODS[method]
    alias_for = cfg.get("alias_for")
    if not alias_for:
        return cfg
    merged = METHODS[str(alias_for)].copy()
    merged.update(cfg)
    return merged


def publication_metadata(method: str) -> dict[str, Any]:
    cfg = method_config(method)
    return {field: cfg.get(field, "") for field in PUBLICATION_FIELDS}


def manifest_method_config(method: str) -> dict[str, Any]:
    cfg = method_config(method)
    return {
        "method": method,
        "canonical_method": str(cfg.get("alias_for") or method),
        "label": str(cfg["label"]),
        **publication_metadata(method),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def count_hypotheses(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("hypotheses"), list):
        return len(payload["hypotheses"])
    return 0


def run_command(cmd: list[str], log_path: Path, err_path: Path, env: dict[str, str], dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    with log_path.open("ab") as stdout, err_path.open("ab") as stderr:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)


def generate_one(
    *,
    python: str,
    method: str,
    seed: int,
    n: int,
    output: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    cfg = method_config(method)
    script = ROOT / "neurooracle/scripts" / str(cfg["script"])
    cmd = [
        python,
        str(script),
        "--n",
        str(n),
        "--batch-size",
        str(cfg["batch_size"]),
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--resume",
    ]
    cmd.extend(str(x) for x in cfg.get("extra_args", []))
    run_command(cmd, output.parent / "generation.log", output.parent / "generation.err.log", env, dry_run)


def evaluate_one(
    *,
    python: str,
    ideas: Path,
    output_dir: Path,
    kg: Path,
    claims: Path,
    seed: int,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    script = ROOT / "neurooracle/scripts/evaluate_external_gid_hypotheses.py"
    cmd = [
        python,
        str(script),
        "--ideas",
        str(ideas),
        "--kg",
        str(kg),
        "--future-claims",
        str(claims),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
    ]
    run_command(cmd, output_dir / "evaluation.log", output_dir / "evaluation.err.log", env, dry_run)


def load_metrics(method: str, seed_index: int, run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return [], None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    topk = metrics.get("topk") or []
    for row in topk:
        rows.append(
            {
                "method": method,
                "method_label": method_config(method)["label"],
                **publication_metadata(method),
                "seed_index": seed_index,
                "k": int(row["k"]),
                "hits": float(row["hits"]),
                "recall": float(row["recall"]),
                "future_evaluable_novel": int(row["future_evaluable_novel"]),
            }
        )
    metric_row = {
        "method": method,
        "method_label": method_config(method)["label"],
        **publication_metadata(method),
        "seed_index": seed_index,
        "n_ideas": int(metrics.get("n_ideas") or 0),
        "valid_gid": int(metrics.get("valid_gid") or 0),
        "valid_gid_pct": 100.0 * int(metrics.get("valid_gid") or 0) / max(1, int(metrics.get("n_ideas") or 0)),
        "auc": float(metrics.get("auc") or 0.0),
        "auprc": float(metrics.get("auprc") or 0.0),
    }
    for k in TOP_KS:
        found = next((float(row["hits"]) for row in topk if int(row["k"]) == k), 0.0)
        metric_row[f"top{k}_hits"] = found
    return rows, metric_row


def summarize(root: Path, methods: list[str], seeds: list[int]) -> None:
    topk_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for method in methods:
        for seed_index, _seed in enumerate(seeds):
            run_dir = root / method / f"seed_{seed_index:02d}"
            rows, metric = load_metrics(method, seed_index, run_dir)
            topk_rows.extend(rows)
            if metric:
                metric_rows.append(metric)
    write_csv(root / "replicate_topk.csv", topk_rows)
    write_csv(root / "replicate_metrics.csv", metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS3 autoresearch baseline replicates.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--seed-base", type=int, default=260620)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--future-claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--openai-timeout", type=float, default=180.0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("OPENAI_TIMEOUT", str(args.openai_timeout))
    manifest = {
        "methods": args.methods,
        "method_configs": [manifest_method_config(method) for method in args.methods],
        "seed_indices": args.seeds,
        "seed_base": args.seed_base,
        "n": args.n,
        "kg": str(args.kg),
        "future_claims": str(args.future_claims),
    }
    if args.dry_run:
        print("DRY-RUN manifest:", json.dumps(manifest, indent=2))
    else:
        args.root.mkdir(parents=True, exist_ok=True)
        (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for method in args.methods:
        for seed_index in args.seeds:
            seed = args.seed_base + seed_index
            run_dir = args.root / method / f"seed_{seed_index:02d}"
            ideas = run_dir / "hypotheses.json"
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
            if not args.eval_only and count_hypotheses(ideas) < args.n:
                print(f"[generate] {method} seed_{seed_index:02d} -> {ideas}")
                generate_one(
                    python=args.python,
                    method=method,
                    seed=seed,
                    n=args.n,
                    output=ideas,
                    env=env,
                    dry_run=args.dry_run,
                )
            if not args.generate_only and count_hypotheses(ideas) >= args.n and not (run_dir / "metrics.json").exists():
                print(f"[evaluate] {method} seed_{seed_index:02d}")
                evaluate_one(
                    python=args.python,
                    ideas=ideas,
                    output_dir=run_dir,
                    kg=args.kg,
                    claims=args.future_claims,
                    seed=seed,
                    env=env,
                    dry_run=args.dry_run,
                )
            if not args.dry_run:
                summarize(args.root, args.methods, args.seeds)


if __name__ == "__main__":
    main()
