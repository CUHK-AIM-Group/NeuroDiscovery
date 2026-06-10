from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, deque
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from case3_hindcasting_eval import (
    build_neighborhood_graph,
    build_pair_evidence,
    load_future_claim_pairs,
    load_kg_index,
)


RUN_RE = re.compile(r"kg(?P<freeze>\d{4})_to_(?P<start>\d{4})_(?P<end>\d{4})$")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _window_from_dir(path: Path) -> tuple[int, int, int] | None:
    m = RUN_RE.match(path.name)
    if not m:
        return None
    return int(m["freeze"]), int(m["start"]), int(m["end"])


def _candidate_pair(candidate: dict[str, Any]) -> tuple[str, str]:
    return str(candidate["gene_id"]), str(candidate["disease_id"])


def _terminal_pairs(future: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {tuple(row["pair"]) for row in future if row["key"] == "gene_disease"}


def _first_terminal_year(future: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    first: dict[tuple[str, str], int] = {}
    for row in future:
        if row["key"] != "gene_disease":
            continue
        pair = tuple(row["pair"])
        year = int(row["year"])
        first[pair] = min(first.get(pair, year), year)
    return first


def _exact_metrics(
    selected: list[tuple[str, str]],
    future_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    selected_set = set(selected)
    hits = selected_set & future_pairs
    return {
        "k": len(selected),
        "hits": len(hits),
        "precision": len(hits) / len(selected) if selected else 0.0,
        "recall": len(hits) / len(future_pairs) if future_pairs else 0.0,
    }


def _random_pool_stats(
    *,
    pool: list[tuple[str, str]],
    k: int,
    future_pairs: set[tuple[str, str]],
    rng: random.Random,
    trials: int,
) -> dict[str, Any]:
    if not pool or k <= 0:
        return {"applicable": False, "mean_hits": None, "mean_precision": None, "p95_hits": None}
    if k > len(pool):
        return {"applicable": False, "mean_hits": None, "mean_precision": None, "p95_hits": None}
    hits: list[int] = []
    for _ in range(trials):
        sample = rng.sample(pool, k)
        hits.append(len(set(sample) & future_pairs))
    hits_sorted = sorted(hits)
    p95 = hits_sorted[int(0.95 * (len(hits_sorted) - 1))] if hits_sorted else 0
    return {
        "applicable": True,
        "mean_hits": mean(hits) if hits else 0.0,
        "mean_precision": (mean(hits) / k) if hits else 0.0,
        "p95_hits": p95,
    }


def _role_random_pool(
    candidates: list[dict[str, Any]],
    historical_direct_pairs: set[tuple[str, str]],
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    genes = sorted({str(c["gene_id"]) for c in candidates})
    diseases = sorted({str(c["disease_id"]) for c in candidates})
    out: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max_pairs * 50
    while len(out) < max_pairs and attempts < max_attempts and genes and diseases:
        attempts += 1
        pair = (rng.choice(genes), rng.choice(diseases))
        if pair in historical_direct_pairs:
            continue
        out.add(pair)
    return sorted(out)


def _adjacency_sets(weighted_adj: dict[str, dict[str, float]]) -> dict[str, set[str]]:
    return {node: set(nbrs) for node, nbrs in weighted_adj.items()}


def _distance_limited(
    adj: dict[str, set[str]],
    source: str,
    target: str,
    max_depth: int,
    extra_edges: dict[str, set[str]] | None = None,
) -> int | None:
    if source == target:
        return 0
    if source not in adj and not (extra_edges and source in extra_edges):
        return None
    seen = {source}
    q: deque[tuple[str, int]] = deque([(source, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= max_depth:
            continue
        nbrs = set(adj.get(node, set()))
        if extra_edges:
            nbrs |= extra_edges.get(node, set())
        for nbr in nbrs:
            if nbr == target:
                return depth + 1
            if nbr in seen:
                continue
            seen.add(nbr)
            q.append((nbr, depth + 1))
    return None


def _edge_map(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    extra: dict[str, set[str]] = {}
    for left, right in edges:
        extra.setdefault(left, set()).add(right)
        extra.setdefault(right, set()).add(left)
    return extra


def _frontier_metrics(
    *,
    adj: dict[str, set[str]],
    selected_edges: list[tuple[str, str]],
    future_pairs: set[tuple[str, str]],
    max_depth: int,
) -> dict[str, Any]:
    extra = _edge_map(selected_edges)
    baseline_reachable = 0
    augmented_reachable = 0
    newly_reachable = 0
    reduced = 0
    deltas: list[int] = []
    for gene, disease in sorted(future_pairs):
        base = _distance_limited(adj, gene, disease, max_depth=max_depth)
        aug = _distance_limited(adj, gene, disease, max_depth=max_depth, extra_edges=extra)
        if base is not None:
            baseline_reachable += 1
        if aug is not None:
            augmented_reachable += 1
        if base is None and aug is not None:
            newly_reachable += 1
        if base is not None and aug is not None and aug < base:
            reduced += 1
            deltas.append(base - aug)
    denom = len(future_pairs)
    return {
        "future_terminal_pairs": denom,
        "baseline_reachable": baseline_reachable,
        "augmented_reachable": augmented_reachable,
        "newly_reachable": newly_reachable,
        "distance_reduced": reduced,
        "mean_distance_reduction": mean(deltas) if deltas else 0.0,
        "baseline_reachable_rate": baseline_reachable / denom if denom else 0.0,
        "augmented_reachable_rate": augmented_reachable / denom if denom else 0.0,
    }


def _lead_time_metrics(
    selected: list[tuple[str, str]],
    first_year: dict[tuple[str, str], int],
    freeze_year: int,
) -> dict[str, Any]:
    lead_times = [first_year[pair] - freeze_year for pair in selected if pair in first_year]
    return {
        "validated": len(lead_times),
        "median_lead_time_years": median(lead_times) if lead_times else None,
        "mean_lead_time_years": mean(lead_times) if lead_times else None,
        "max_lead_time_years": max(lead_times) if lead_times else None,
    }


def evaluate_run(
    *,
    run_dir: Path,
    all_run_dirs: list[Path],
    top_ks: tuple[int, ...],
    random_trials: int,
    role_random_pool_size: int,
    seed: int,
    max_depth: int,
) -> dict[str, Any]:
    freeze, start, end = _window_from_dir(run_dir) or (None, None, None)
    if freeze is None:
        raise ValueError(f"cannot parse window from {run_dir}")

    metrics = _load_json(run_dir / "metrics.json")
    candidates = _load_json(run_dir / "candidates_top.json")
    concepts, edges, _names = load_kg_index(Path(metrics["kg_path"]))
    gene_imaging, imaging_disease, gene_disease, historical_direct_pairs = build_pair_evidence(
        concepts,
        edges,
        strict_candidate_anchors=bool(metrics.get("strict_candidate_anchors", True)),
    )
    del gene_imaging, imaging_disease, gene_disease

    future_claims_path = Path(metrics["future_claims_path"])
    all_future, _stats = load_future_claim_pairs(
        claims_path=future_claims_path,
        concepts=concepts,
        start_year=freeze + 1,
        end_year=max(
            int((_window_from_dir(d) or (freeze, start, end))[2])
            for d in all_run_dirs
        ),
        historical_direct_pairs=historical_direct_pairs,
    )
    first_year = _first_terminal_year(all_future)

    immediate_future, _ = load_future_claim_pairs(
        claims_path=future_claims_path,
        concepts=concepts,
        start_year=start,
        end_year=end,
        historical_direct_pairs=historical_direct_pairs,
    )
    immediate_pairs = _terminal_pairs(immediate_future)

    downstream_pairs: set[tuple[str, str]] = set()
    downstream_start = end + 1
    downstream_end = end + 2
    if downstream_start <= max(row["year"] for row in all_future):
        downstream_future, _ = load_future_claim_pairs(
            claims_path=future_claims_path,
            concepts=concepts,
            start_year=downstream_start,
            end_year=downstream_end,
            historical_direct_pairs=historical_direct_pairs,
        )
        downstream_pairs = _terminal_pairs(downstream_future)

    candidate_pairs = [_candidate_pair(c) for c in candidates]
    rng = random.Random(seed + freeze)
    role_pool = _role_random_pool(
        candidates,
        historical_direct_pairs,
        max_pairs=role_random_pool_size,
        rng=rng,
    )

    neighbor = build_neighborhood_graph(
        concepts,
        edges,
        strict_candidate_anchors=bool(metrics.get("strict_candidate_anchors", True)),
    )
    adj = _adjacency_sets(neighbor.adj)

    topk_results: dict[str, Any] = {}
    for k in top_ks:
        selected = candidate_pairs[: min(k, len(candidate_pairs))]
        remaining_pool = candidate_pairs[min(k, len(candidate_pairs)) :]
        exact_immediate = _exact_metrics(selected, immediate_pairs)
        exact_downstream = _exact_metrics(selected, downstream_pairs)
        topk_results[str(k)] = {
            "immediate_exact": exact_immediate,
            "downstream_exact": exact_downstream,
            "lead_time_all_future": _lead_time_metrics(selected, first_year, freeze),
            "same_candidate_pool_random_immediate": _random_pool_stats(
                pool=remaining_pool,
                k=len(selected),
                future_pairs=immediate_pairs,
                rng=rng,
                trials=random_trials,
            ),
            "role_random_immediate": _random_pool_stats(
                pool=role_pool,
                k=len(selected),
                future_pairs=immediate_pairs,
                rng=rng,
                trials=random_trials,
            ),
            "frontier_downstream": _frontier_metrics(
                adj=adj,
                selected_edges=selected,
                future_pairs=downstream_pairs,
                max_depth=max_depth,
            ) if downstream_pairs else None,
        }

    return {
        "run_dir": str(run_dir),
        "freeze_year": freeze,
        "immediate_window": f"{start}-{end}",
        "downstream_window": f"{downstream_start}-{downstream_end}" if downstream_pairs else None,
        "n_candidates_loaded": len(candidates),
        "n_terminal_immediate": len(immediate_pairs),
        "n_terminal_all_future": len(first_year),
        "topk": topk_results,
    }


def write_report(output_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Case Study 3 Growth Evaluation",
        "",
        "This experiment asks whether generated hypotheses behave like useful research actions, not only one-step predictions.",
        "",
        "## Exact Future Validation",
        "| Freeze | Immediate | K | Hits | Precision | Recall | Random hits | Role-random hits | Validated later | Median lead |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for k, row in result["topk"].items():
            exact = row["immediate_exact"]
            pool_rand = row["same_candidate_pool_random_immediate"]
            role_rand = row["role_random_immediate"]
            lead = row["lead_time_all_future"]
            lines.append(
                "| {freeze} | {window} | {k} | {hits} | {precision:.4f} | {recall:.4f} | {pool} | {role} | {validated} | {median_lead} |".format(
                    freeze=result["freeze_year"],
                    window=result["immediate_window"],
                    k=k,
                    hits=exact["hits"],
                    precision=exact["precision"],
                    recall=exact["recall"],
                    pool=_fmt(pool_rand["mean_hits"]),
                    role=_fmt(role_rand["mean_hits"]),
                    validated=lead["validated"],
                    median_lead="NA" if lead["median_lead_time_years"] is None else f"{lead['median_lead_time_years']:.1f}",
                )
            )

    lines.extend([
        "",
        "## Frontier Expansion",
        "| Freeze | Downstream | K | Baseline reachable | Augmented reachable | Newly reachable | Distance reduced |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        for k, row in result["topk"].items():
            frontier = row["frontier_downstream"]
            if not frontier:
                continue
            lines.append(
                "| {freeze} | {window} | {k} | {base:.4f} | {aug:.4f} | {new} | {reduced} |".format(
                    freeze=result["freeze_year"],
                    window=result["downstream_window"],
                    k=k,
                    base=frontier["baseline_reachable_rate"],
                    aug=frontier["augmented_reachable_rate"],
                    new=frontier["newly_reachable"],
                    reduced=frontier["distance_reduced"],
                )
            )

    lines.extend([
        "",
        "## Notes",
        "- Random hits are averaged over repeated samples from the remaining candidate pool; role-random hits sample gene-disease pairs with matched endpoint roles.",
        "- Frontier expansion adds Top-K generated gene-disease hypotheses as shortcut edges and tests whether downstream future terminal claims become reachable within a limited graph depth.",
        "- This is still a proxy experiment: validation uses future literature claims, and negatives are not hard scientific failures.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Case Study 3 as iterative scientific-growth support.")
    parser.add_argument("--rolling-root", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/rolling_windows_kge_complex_full_snapshot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/growth_eval_kge_complex_full_snapshot_v1"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--random-trials", type=int, default=500)
    parser.add_argument("--role-random-pool-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        d for d in args.rolling_root.iterdir()
        if d.is_dir() and (d / "metrics.json").is_file() and (d / "candidates_top.json").is_file()
    )
    results = [
        evaluate_run(
            run_dir=run_dir,
            all_run_dirs=run_dirs,
            top_ks=tuple(args.top_k),
            random_trials=args.random_trials,
            role_random_pool_size=args.role_random_pool_size,
            seed=args.seed,
            max_depth=args.max_depth,
        )
        for run_dir in run_dirs
    ]
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"rolling_root": str(args.rolling_root), "results": results}, f, indent=2, ensure_ascii=False)
    write_report(args.output_dir, results)
    print(json.dumps({"output_dir": str(args.output_dir), "n_runs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
