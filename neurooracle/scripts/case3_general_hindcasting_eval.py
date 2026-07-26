from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    from .case3_hindcasting_eval import _claim_year, _role_for_node, load_kg_index
except ImportError:  # Direct script execution.
    from case3_hindcasting_eval import _claim_year, _role_for_node, load_kg_index


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_paper(claim: dict[str, Any]) -> dict[str, Any]:
    paper = claim.get("source_paper")
    return paper if isinstance(paper, dict) else {}


def _node_roles(concepts: dict[str, dict[str, Any]], node_id: str) -> tuple[str, ...]:
    node = concepts.get(node_id)
    if not node:
        return ()
    role = _role_for_node(node)
    if role:
        return (role,)
    tags = node.get("domain_tags") or []
    return tuple(str(tag) for tag in tags[:2])


def _edge_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _hyp_path_edges(hyp: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for link in hyp.get("path") or []:
        src = str(link.get("from_id") or "")
        dst = str(link.get("to_id") or "")
        if src and dst and src != dst:
            out.append((src, dst))
    return out


def _hyp_endpoint_pair(hyp: dict[str, Any]) -> tuple[str, str] | None:
    src = str(hyp.get("source_id") or "")
    dst = str(hyp.get("target_id") or "")
    if src and dst and src != dst:
        return src, dst
    edges = _hyp_path_edges(hyp)
    if edges:
        return edges[0][0], edges[-1][1]
    return None


def _hyp_signature(concepts: dict[str, dict[str, Any]], hyp: dict[str, Any]) -> str:
    roles: list[str] = []
    endpoint = _hyp_endpoint_pair(hyp)
    if endpoint:
        roles.extend("/".join(_node_roles(concepts, endpoint[0]) or ("unknown",)).split("|"))
    for src, dst in _hyp_path_edges(hyp):
        if not roles:
            roles.extend(_node_roles(concepts, src) or ("unknown",))
        roles.extend(_node_roles(concepts, dst) or ("unknown",))
    clean: list[str] = []
    for role in roles:
        if role and (not clean or clean[-1] != role):
            clean.append(role)
    return " -> ".join(clean[:8]) if clean else "unknown"


def _historical_pairs(edges: list[dict[str, Any]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for edge in edges:
        src = str(edge.get("source_id") or "")
        dst = str(edge.get("target_id") or "")
        rel = str(edge.get("relation_type") or "")
        if not src or not dst or src == dst or rel in {"is_a", "part_of", "about"}:
            continue
        out.add(_edge_pair(src, dst))
    return out


def _future_indexes(
    claims_path: Path,
    concepts: dict[str, dict[str, Any]],
    historical_pairs: set[tuple[str, str]],
    start_year: int,
    end_year: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    novel_pair_year: dict[tuple[str, str], int] = {}
    all_pair_year: dict[tuple[str, str], int] = {}
    pair_claims: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()
    with claims_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            claim = json.loads(line)
            year = _claim_year(claim)
            if year is None or year < start_year or year > end_year:
                continue
            stats["future_claims_total"] += 1
            sid = str(claim.get("subject_id") or "")
            oid = str(claim.get("object_id") or "")
            if not sid or not oid or sid == oid:
                stats["missing_endpoint"] += 1
                continue
            if sid not in concepts or oid not in concepts:
                stats["endpoint_not_in_frozen_kg"] += 1
                continue
            pair = _edge_pair(sid, oid)
            all_pair_year[pair] = min(all_pair_year.get(pair, year), year)
            paper = _safe_paper(claim)
            pair_claims[pair].append(
                {
                    "claim_id": claim.get("id"),
                    "pmid": paper.get("pmid"),
                    "doi": paper.get("doi"),
                    "title": paper.get("title"),
                    "journal": paper.get("journal"),
                    "year": year,
                    "predicate": claim.get("predicate"),
                    "subject_id": sid,
                    "subject_name": claim.get("subject_name"),
                    "object_id": oid,
                    "object_name": claim.get("object_name"),
                    "raw_text": claim.get("raw_text"),
                }
            )
            if pair in historical_pairs:
                stats["already_direct_in_frozen_kg"] += 1
                continue
            novel_pair_year[pair] = min(novel_pair_year.get(pair, year), year)
            stats["future_evaluable_claims"] += 1
    stats["future_unique_pairs"] = len(novel_pair_year)
    stats["future_all_unique_pairs"] = len(all_pair_year)
    return {
        "novel_pair_year": novel_pair_year,
        "all_pair_year": all_pair_year,
        "pair_claims": pair_claims,
    }, dict(stats)


def _score_hypothesis(hyp: dict[str, Any], future: dict[str, Any], freeze_year: int) -> dict[str, Any]:
    novel_pair_year: dict[tuple[str, str], int] = future["novel_pair_year"]
    all_pair_year: dict[tuple[str, str], int] = future["all_pair_year"]
    endpoint = _hyp_endpoint_pair(hyp)
    endpoint_pair = _edge_pair(*endpoint) if endpoint else None
    endpoint_year = novel_pair_year.get(endpoint_pair) if endpoint_pair else None
    path_edges = [_edge_pair(src, dst) for src, dst in _hyp_path_edges(hyp)]
    hit_edges = [pair for pair in path_edges if pair in all_pair_year]
    hit_years = [all_pair_year[pair] for pair in hit_edges]
    any_years = ([endpoint_year] if endpoint_year is not None else []) + hit_years
    support_candidates: list[tuple[str, tuple[str, str], int]] = []
    if endpoint_pair is not None and endpoint_year is not None:
        support_candidates.append(("endpoint", endpoint_pair, endpoint_year))
    support_candidates.extend(("path_edge", pair, all_pair_year[pair]) for pair in hit_edges)
    support_kind = None
    support_pair = None
    if support_candidates:
        support_kind, support_pair, _ = min(support_candidates, key=lambda item: item[2])
    return {
        "endpoint_hit": endpoint_year is not None,
        "endpoint_year": endpoint_year,
        "endpoint_lead_time": (endpoint_year - freeze_year) if endpoint_year is not None else None,
        "endpoint_pair": "|".join(endpoint_pair) if endpoint_pair else "",
        "path_edges": len(path_edges),
        "path_edge_hits": len(hit_edges),
        "path_edge_hit_pairs": ";".join("|".join(pair) for pair in hit_edges),
        "path_edge_hit_rate": len(hit_edges) / len(path_edges) if path_edges else 0.0,
        "any_path_edge_hit": bool(hit_edges),
        "all_path_edges_hit": bool(path_edges) and len(hit_edges) == len(path_edges),
        "any_future_hit": bool(any_years),
        "first_future_year": min(any_years) if any_years else None,
        "lead_time": (min(any_years) - freeze_year) if any_years else None,
        "support_kind": support_kind,
        "support_pair": "|".join(support_pair) if support_pair else "",
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "endpoint_hits": 0,
            "endpoint_hit_rate": 0.0,
            "any_path_edge_hits": 0,
            "any_path_edge_hit_rate": 0.0,
            "any_future_hits": 0,
            "any_future_hit_rate": 0.0,
            "mean_path_edge_hit_rate": 0.0,
        }
    lead_times = [r["lead_time"] for r in rows if r["lead_time"] is not None]
    return {
        "n": n,
        "endpoint_hits": sum(1 for r in rows if r["endpoint_hit"]),
        "endpoint_hit_rate": sum(1 for r in rows if r["endpoint_hit"]) / n,
        "any_path_edge_hits": sum(1 for r in rows if r["any_path_edge_hit"]),
        "any_path_edge_hit_rate": sum(1 for r in rows if r["any_path_edge_hit"]) / n,
        "all_path_edges_hits": sum(1 for r in rows if r["all_path_edges_hit"]),
        "any_future_hits": sum(1 for r in rows if r["any_future_hit"]),
        "any_future_hit_rate": sum(1 for r in rows if r["any_future_hit"]) / n,
        "mean_path_edge_hit_rate": mean(r["path_edge_hit_rate"] for r in rows),
        "mean_lead_time": mean(lead_times) if lead_times else None,
    }


def _random_baseline(
    scored: list[dict[str, Any]],
    k: int,
    trials: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if k > len(scored):
        return {"applicable": False}, []
    hits: list[int] = []
    endpoint_hits: list[int] = []
    trial_rows: list[dict[str, Any]] = []
    for _ in range(trials):
        sample = rng.sample(scored, k)
        hits.append(sum(1 for row in sample if row["any_future_hit"]))
        endpoint_hits.append(sum(1 for row in sample if row["endpoint_hit"]))
        trial_rows.append({
            "trial": len(trial_rows) + 1,
            "k": k,
            "any_future_hits": hits[-1],
            "endpoint_hits": endpoint_hits[-1],
        })
    return {
        "applicable": True,
        "trials": trials,
        "mean_any_future_hits": mean(hits),
        "mean_endpoint_hits": mean(endpoint_hits),
        "sd_any_future_hits": stdev(hits) if len(hits) > 1 else 0.0,
        "sd_endpoint_hits": stdev(endpoint_hits) if len(endpoint_hits) > 1 else 0.0,
        "ci95_any_future_hits": _percentile_interval(hits),
        "ci95_endpoint_hits": _percentile_interval(endpoint_hits),
    }, trial_rows


def _percentile_interval(values: list[int]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    xs = sorted(values)
    return [_percentile(xs, 0.025), _percentile(xs, 0.975)]


def _percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _empirical_p_ge(values: list[int], observed: int) -> float | None:
    if not values:
        return None
    return (1 + sum(1 for value in values if value >= observed)) / (len(values) + 1)


def _support_claim(future: dict[str, Any], support_pair: str, first_year: int | None) -> dict[str, Any]:
    if not support_pair or first_year is None:
        return {}
    parts = support_pair.split("|")
    if len(parts) != 2:
        return {}
    pair = tuple(parts)
    claims = future["pair_claims"].get(pair) or []
    if not claims:
        return {}
    return sorted(claims, key=lambda claim: (claim.get("year") or 9999, str(claim.get("claim_id") or "")))[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    *,
    kg_path: Path,
    hypotheses_path: Path,
    future_claims_path: Path,
    output_dir: Path,
    freeze_year: int,
    future_start_year: int,
    future_end_year: int,
    top_ks: list[int],
    random_trials: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    concepts, edges, names = load_kg_index(kg_path)
    del names
    historical_pairs = _historical_pairs(edges)
    future, future_stats = _future_indexes(
        future_claims_path,
        concepts,
        historical_pairs,
        future_start_year,
        future_end_year,
    )
    payload = _load_json(hypotheses_path)
    hypotheses = payload.get("hypotheses") or []
    scored: list[dict[str, Any]] = []
    for idx, hyp in enumerate(hypotheses):
        score = _score_hypothesis(hyp, future, freeze_year)
        row = {
            "rank": idx + 1,
            "id": hyp.get("id"),
            "hypothesis_type": hyp.get("hypothesis_type") or "unknown",
            "task_name": (hyp.get("metadata") or {}).get("task_name") or (hyp.get("metadata") or {}).get("chain_name") or "unknown",
            "task_kind": (hyp.get("metadata") or {}).get("task_kind") or "unknown",
            "signature": _hyp_signature(concepts, hyp),
            "source_id": hyp.get("source_id"),
            "source_name": hyp.get("source_name"),
            "target_id": hyp.get("target_id"),
            "target_name": hyp.get("target_name"),
            "composite_score": hyp.get("composite_score"),
            **score,
        }
        scored.append(row)

    rng = random.Random(seed + freeze_year)
    topk: dict[str, Any] = {}
    random_trial_rows: list[dict[str, Any]] = []
    for k in top_ks:
        subset = scored[: min(k, len(scored))]
        random_summary, trial_rows = _random_baseline(scored, len(subset), random_trials, rng)
        random_values = [int(row["any_future_hits"]) for row in trial_rows]
        random_endpoint_values = [int(row["endpoint_hits"]) for row in trial_rows]
        observed = _aggregate(subset)
        random_summary["p_any_future_hits_ge_observed"] = _empirical_p_ge(
            random_values,
            int(observed["any_future_hits"]),
        )
        random_summary["p_endpoint_hits_ge_observed"] = _empirical_p_ge(
            random_endpoint_values,
            int(observed["endpoint_hits"]),
        )
        random_trial_rows.extend(trial_rows)
        topk[str(k)] = {
            "observed": observed,
            "random_same_hypothesis_pool": random_summary,
        }

    by_type = {
        key: _aggregate(rows)
        for key, rows in _group(scored, "hypothesis_type").items()
    }
    by_task = {
        key: _aggregate(rows)
        for key, rows in _group(scored, "task_name").items()
    }
    by_signature = {
        key: _aggregate(rows)
        for key, rows in sorted(
            _group(scored, "signature").items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:50]
    }

    recovered_examples: list[dict[str, Any]] = []
    for row in scored:
        if not row["any_future_hit"]:
            continue
        claim = _support_claim(future, str(row.get("support_pair") or ""), row.get("first_future_year"))
        recovered_examples.append({
            **row,
            "support_claim_id": claim.get("claim_id"),
            "support_pmid": claim.get("pmid"),
            "support_doi": claim.get("doi"),
            "support_title": claim.get("title"),
            "support_journal": claim.get("journal"),
            "support_year": claim.get("year"),
            "support_predicate": claim.get("predicate"),
            "support_subject_name": claim.get("subject_name"),
            "support_object_name": claim.get("object_name"),
            "support_raw_text": claim.get("raw_text"),
        })

    manifest = {
        "kg_path": str(kg_path),
        "hypotheses_path": str(hypotheses_path),
        "future_claims_path": str(future_claims_path),
        "output_dir": str(output_dir),
        "freeze_year": freeze_year,
        "future_start_year": future_start_year,
        "future_end_year": future_end_year,
        "n_hypotheses": len(scored),
        "future_stats": future_stats,
        "topk": topk,
        "by_type": by_type,
        "by_task": by_task,
        "by_signature_top50": by_signature,
        "validated_sample": recovered_examples[:100],
        "note": (
            "General Case Study 3 hindcasting: hypotheses are not restricted to gene-imaging-disease. "
            "A hit means a future claim supports the hypothesis endpoint pair or at least one path edge."
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _write_window_tables(output_dir, scored, random_trial_rows, recovered_examples, topk)
    write_report(output_dir, manifest)
    return manifest


def _write_window_tables(
    output_dir: Path,
    scored: list[dict[str, Any]],
    random_trial_rows: list[dict[str, Any]],
    recovered_examples: list[dict[str, Any]],
    topk: dict[str, Any],
) -> None:
    scored_fields = [
        "rank", "id", "hypothesis_type", "task_name", "task_kind", "signature",
        "source_id", "source_name", "target_id", "target_name", "composite_score",
        "endpoint_hit", "endpoint_year", "endpoint_lead_time", "endpoint_pair",
        "path_edges", "path_edge_hits", "path_edge_hit_pairs", "path_edge_hit_rate",
        "any_path_edge_hit", "all_path_edges_hit", "any_future_hit",
        "first_future_year", "lead_time", "support_kind", "support_pair",
    ]
    _write_csv(output_dir / "scored_hypotheses.csv", scored, scored_fields)
    _write_csv(
        output_dir / "random_trials.csv",
        random_trial_rows,
        ["trial", "k", "any_future_hits", "endpoint_hits"],
    )
    example_fields = scored_fields + [
        "support_claim_id", "support_pmid", "support_doi", "support_title",
        "support_journal", "support_year", "support_predicate",
        "support_subject_name", "support_object_name", "support_raw_text",
    ]
    _write_csv(output_dir / "recovered_examples.csv", recovered_examples[:200], example_fields)
    rows: list[dict[str, Any]] = []
    for k, row in topk.items():
        obs = row["observed"]
        rand = row["random_same_hypothesis_pool"]
        rows.append({
            "k": k,
            "n": obs["n"],
            "endpoint_hits": obs["endpoint_hits"],
            "any_future_hits": obs["any_future_hits"],
            "any_future_hit_rate": obs["any_future_hit_rate"],
            "mean_lead_time": obs.get("mean_lead_time"),
            "random_any_future_hits_mean": rand.get("mean_any_future_hits"),
            "random_any_future_hits_sd": rand.get("sd_any_future_hits"),
            "random_any_future_hits_ci95_low": (rand.get("ci95_any_future_hits") or [None, None])[0],
            "random_any_future_hits_ci95_high": (rand.get("ci95_any_future_hits") or [None, None])[1],
            "p_any_future_hits_ge_observed": rand.get("p_any_future_hits_ge_observed"),
            "random_endpoint_hits_mean": rand.get("mean_endpoint_hits"),
            "p_endpoint_hits_ge_observed": rand.get("p_endpoint_hits_ge_observed"),
        })
    _write_csv(
        output_dir / "metrics_by_k.csv",
        rows,
        [
            "k", "n", "endpoint_hits", "any_future_hits", "any_future_hit_rate",
            "mean_lead_time", "random_any_future_hits_mean",
            "random_any_future_hits_sd", "random_any_future_hits_ci95_low",
            "random_any_future_hits_ci95_high", "p_any_future_hits_ge_observed",
            "random_endpoint_hits_mean", "p_endpoint_hits_ge_observed",
        ],
    )


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key) or "unknown")].append(row)
    return dict(out)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Case Study 3 General Hindcasting",
        "",
        "This evaluates arbitrary generated hypotheses against future claims without constraining the hypothesis space to gene-imaging-disease.",
        "",
        "## Setup",
        f"- Freeze year: {manifest['freeze_year']}",
        f"- Future window: {manifest['future_start_year']}-{manifest['future_end_year']}",
        f"- Hypotheses: {manifest['n_hypotheses']}",
        f"- Future unique evaluable pairs: {manifest['future_stats'].get('future_unique_pairs', 0)}",
        "",
        "## Top-K",
        "| K | Endpoint hits | Endpoint hit rate | Any future hits | Any future hit rate | Path-edge hit rate | Random any hits | Random endpoint hits |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, row in manifest["topk"].items():
        obs = row["observed"]
        rand = row["random_same_hypothesis_pool"]
        lines.append(
            "| {k} | {endpoint_hits} | {endpoint_rate} | {any_hits} | {any_rate} | {path_rate} | {rand_any} | {rand_endpoint} |".format(
                k=k,
                endpoint_hits=obs["endpoint_hits"],
                endpoint_rate=_fmt(obs["endpoint_hit_rate"]),
                any_hits=obs["any_future_hits"],
                any_rate=_fmt(obs["any_future_hit_rate"]),
                path_rate=_fmt(obs["mean_path_edge_hit_rate"]),
                rand_any=_fmt(rand.get("mean_any_future_hits")),
                rand_endpoint=_fmt(rand.get("mean_endpoint_hits")),
            )
        )
    lines.extend([
        "",
        "## By Hypothesis Type",
        "| Type | N | Endpoint hit rate | Any future hit rate | Path-edge hit rate |",
        "|:---|---:|---:|---:|---:|",
    ])
    for key, row in sorted(manifest["by_type"].items(), key=lambda item: item[1]["n"], reverse=True):
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['endpoint_hit_rate'])} | {_fmt(row['any_future_hit_rate'])} | {_fmt(row['mean_path_edge_hit_rate'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "- Endpoint hits are stricter: future literature supports the source-target pair.",
        "- Any future hits are broader: future literature supports either the endpoint pair or at least one path edge.",
        "- Random baseline samples from the same hypothesis file, so it tests ranking quality rather than generator-vs-random-space quality.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="General Case Study 3 hindcasting evaluator.")
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--future-claims", type=Path, default=Path("neurooracle/data/full_v2/extracted_claims.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-year", type=int, required=True)
    parser.add_argument("--future-start-year", type=int, required=True)
    parser.add_argument("--future-end-year", type=int, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--random-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    manifest = evaluate(
        kg_path=args.kg,
        hypotheses_path=args.hypotheses,
        future_claims_path=args.future_claims,
        output_dir=args.output_dir,
        freeze_year=args.freeze_year,
        future_start_year=args.future_start_year,
        future_end_year=args.future_end_year,
        top_ks=args.top_k,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n_hypotheses": manifest["n_hypotheses"]}, indent=2))


if __name__ == "__main__":
    main()
