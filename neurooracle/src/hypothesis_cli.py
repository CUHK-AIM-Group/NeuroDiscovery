"""Hypothesis engine CLI for the NeuroClaw knowledge graph.

Usage (run from project root):
    python -m neurooracle.phase3 batch --output neurooracle/data/hypotheses.json
    python -m neurooracle.phase3 imaging-batch --dataset UKB --output neurooracle/data/hypotheses_imaging_ukb.json
    python -m neurooracle.phase3 rank --input neurooracle/data/hypotheses.json --top 20
    python -m neurooracle.phase3 paths "hippocampus" "Alzheimer Disease"
    python -m neurooracle.phase3 bridge "hippocampus" --target-domain disease
    python -m neurooracle.phase3 contradictions --domain disease
    python -m neurooracle.phase3 gaps --domain-a neuroanatomy --domain-b disease
    python -m neurooracle.phase3 explore "hippocampus"
    python -m neurooracle.phase3 stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .storage import load_graph
from .hypothesis_engine import (
    HypothesisEngine, Hypothesis, Contradiction, Gap,
)
from .critic_agent import CriticAgent
from .evolution_engine import EvolutionEngine
from .feedback_state import FeedbackState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ── formatters ─────────────────────────────────────────────────────────

def format_hypothesis(h: Hypothesis, index: int) -> str:
    lines = [
        f"--- #{index} [{h.id}] ({h.hypothesis_type}) ---",
        f"  {h.source_name} --> {h.target_name}",
        f"  Path:",
    ]
    for link in h.path:
        lines.append(f"    {link.from_name} --[{link.relation_type}]--> {link.to_name}")
        if link.raw_text:
            lines.append(f"      Evidence: '{link.raw_text[:120]}...'")
        if link.source_paper.get("pmid"):
            lines.append(f"      Source: PMID {link.source_paper['pmid']} ({link.source_paper.get('year', '?')})")

    lines.append(
        f"  Scores: confidence={h.confidence_score:.2f} "
        f"novelty={h.novelty_score:.2f} "
        f"evidence={h.evidence_score:.2f} "
        f"testability={h.testability_score:.2f} "
        f"composite={h.composite_score:.4f}"
    )
    if h.testability_reason:
        lines.append(f"  Testability: {h.testability_reason}")
    lines.append(f"  Claims: {len(h.supporting_claims)}")
    return "\n".join(lines)


def format_contradiction(c: Contradiction, index: int) -> str:
    return (
        f"--- Contradiction #{index} (severity={c.severity:.1f}) ---\n"
        f"  Concepts: {c.concept_a_name} <-> {c.concept_b_name}\n"
        f"  Claim FOR: [{c.claim_for_predicate}] {c.claim_for_text[:100]}\n"
        f"  Claim AGAINST: [{c.claim_against_predicate}] {c.claim_against_text[:100]}"
    )


def format_gap(g: Gap, index: int) -> str:
    return (
        f"--- Gap #{index} ---\n"
        f"  {g.concept_a_name} ({g.domain_a}) <--?--> {g.concept_b_name} ({g.domain_b})\n"
        f"  Distance: {g.distance} hops via {g.connecting_concepts}\n"
        f"  Potential relation: {g.potential_relation}"
    )


# ── commands ───────────────────────────────────────────────────────────

def cmd_batch(engine, output, domain_pairs=None, max_hops=4, max_paths=5,
              max_seeds=50, as_json=False, legacy_domain_pairs=False,
              task_filter=None, chain_filter=None,
              target_per_task=None, max_retries=3, retry_scale=2.0,
              min_hops=2, metapath_min_domains=2, prefer_longer_paths=True):
    """Batch-generate hypotheses across the entire graph.

    Default (atom-task driven): traverses every Task in CANONICAL_TASKS via
    ``batch_generate_for_task`` and every TaskChain in CANONICAL_CHAINS via
    ``batch_generate_for_chain``. Generated hypotheses are tagged with
    ``task_name`` / ``chain_name`` / ``task_kind`` so downstream consumers
    can filter by task.

    Legacy mode (``--legacy-domain-pairs``): traverses raw KG domain pairs
    with no task tagging. ``domain_pairs`` may be:
        None / "default" — DEFAULT_DOMAIN_PAIRS (clinical outcomes)
        "imaging"        — IMAGING_DOMAIN_PAIRS (UKB/ADNI/HCP)
        "decoding"       — DECODING_DOMAIN_PAIRS (NSD/BOLD5000/SEED brain decoding)
        "all"            — union of all three
        list[tuple]      — explicit (src_domain, tgt_domain) tuples

    ``task_filter`` / ``chain_filter`` (task-driven only): comma-separated
    name allow-lists, e.g. "biomarker_discovery,brain_age". None = run all.
    """
    if legacy_domain_pairs:
        # Resolve string presets to actual pair lists
        if isinstance(domain_pairs, str):
            from .hypothesis_engine import (
                DEFAULT_DOMAIN_PAIRS, IMAGING_DOMAIN_PAIRS, DECODING_DOMAIN_PAIRS,
            )
            preset_map = {
                "default":  DEFAULT_DOMAIN_PAIRS,
                "imaging":  IMAGING_DOMAIN_PAIRS,
                "decoding": DECODING_DOMAIN_PAIRS,
                "all":      DEFAULT_DOMAIN_PAIRS + IMAGING_DOMAIN_PAIRS + DECODING_DOMAIN_PAIRS,
            }
            if domain_pairs not in preset_map:
                raise ValueError(f"Unknown domain-pairs preset: {domain_pairs}")
            print(f"  legacy domain-pairs preset: {domain_pairs} ({len(preset_map[domain_pairs])} pairs)")
            domain_pairs = preset_map[domain_pairs]

        print(f"Legacy batch (max_hops={max_hops}, min_hops={min_hops}, "
              f"metapath_min_domains={metapath_min_domains}, "
              f"max_paths_per_pair={max_paths}, max_seeds={max_seeds})...")
        hypotheses = engine.batch_generate(
            domain_pairs=domain_pairs,
            max_hops=max_hops,
            max_paths_per_pair=max_paths,
            max_seeds_per_domain=max_seeds,
            min_hops=min_hops,
            metapath_min_domains=metapath_min_domains,
            prefer_longer_paths=prefer_longer_paths,
        )
    else:
        from .atoms import CANONICAL_TASKS, CANONICAL_CHAINS

        task_names = None
        if task_filter is not None:
            task_names = {n.strip() for n in task_filter.split(",") if n.strip()}
        chain_names = None
        if chain_filter is not None:
            chain_names = {n.strip() for n in chain_filter.split(",") if n.strip()}

        tasks_to_run = [t for t in CANONICAL_TASKS
                        if task_names is None or t.name in task_names]
        chains_to_run = [c for c in CANONICAL_CHAINS
                         if chain_names is None or c.name in chain_names]

        print(f"Task-driven batch: {len(tasks_to_run)} tasks + {len(chains_to_run)} chains "
              f"(max_hops={max_hops}, min_hops={min_hops}, "
              f"metapath_min_domains={metapath_min_domains}, "
              f"max_paths={max_paths}, max_seeds={max_seeds})")

        hypotheses: list = []
        for task in tasks_to_run:
            print(f"  task: {task.name} [{task.signature}]")
            seen_ids: set = set()
            kept: list = []
            cur_paths, cur_seeds = max_paths, max_seeds
            for attempt in range(1, max_retries + 1):
                hs = engine.batch_generate_for_task(
                    task,
                    max_hops=max_hops,
                    max_paths_per_pair=cur_paths,
                    max_seeds_per_domain=cur_seeds,
                    min_hops=min_hops,
                    metapath_min_domains=metapath_min_domains,
                    prefer_longer_paths=prefer_longer_paths,
                )
                added = 0
                for h in hs:
                    if h.id not in seen_ids:
                        seen_ids.add(h.id)
                        kept.append(h)
                        added += 1
                print(f"    attempt {attempt}: +{added} (total {len(kept)}, paths={cur_paths}, seeds={cur_seeds})")
                if target_per_task is None or len(kept) >= target_per_task:
                    break
                cur_paths = max(int(cur_paths * retry_scale), cur_paths + 1)
                cur_seeds = max(int(cur_seeds * retry_scale), cur_seeds + 1)
            print(f"    -> {len(kept)} hypotheses")
            hypotheses.extend(kept)

        for chain in chains_to_run:
            print(f"  chain: {chain.name} [{chain.signature}]")
            # Dedup by anchor prefix (source + intermediate mediators), not
            # by id — chain ids reset per engine call, but the same anchor
            # prefix produced across retries is the same mechanism.
            best_by_prefix: dict[tuple, "Hypothesis"] = {}
            cur_paths, cur_seeds = max_paths, max_seeds
            for attempt in range(1, max_retries + 1):
                hs = engine.batch_generate_for_chain(
                    chain,
                    max_hops_per_segment=max(max_hops // 2, 2),
                    max_paths_per_segment=cur_paths,
                    max_seeds=cur_seeds,
                )
                added = 0
                for h in hs:
                    prefix = (h.source_id, *((h.metadata or {}).get("mediator_ids") or []))
                    cur = best_by_prefix.get(prefix)
                    if cur is None:
                        best_by_prefix[prefix] = h
                        added += 1
                    elif h.composite_score > cur.composite_score:
                        best_by_prefix[prefix] = h
                kept = list(best_by_prefix.values())
                print(f"    attempt {attempt}: +{added} (total {len(kept)}, paths={cur_paths}, seeds={cur_seeds})")
                if target_per_task is None or len(kept) >= target_per_task:
                    break
                cur_paths = max(int(cur_paths * retry_scale), cur_paths + 1)
                cur_seeds = max(int(cur_seeds * retry_scale), cur_seeds + 1)
            kept = list(best_by_prefix.values())
            print(f"    -> {len(kept)} hypotheses")
            hypotheses.extend(kept)

    print(f"Generated {len(hypotheses)} raw hypotheses")

    # Cross-task / cross-chain deduplication: same path can be reachable from
    # multiple task templates. Keep first occurrence (task_name tag wins) and
    # drop any hypothesis lacking a stable id.
    seen_ids: set = set()
    deduped = []
    n_dups = 0
    n_empty = 0
    for h in hypotheses:
        if not h.id:
            n_empty += 1
            continue
        if h.id in seen_ids:
            n_dups += 1
            continue
        seen_ids.add(h.id)
        deduped.append(h)
    if n_dups or n_empty:
        print(f"  deduplicated: -{n_dups} duplicate id(s), -{n_empty} empty id(s)")
    hypotheses = deduped
    print(f"After dedup: {len(hypotheses)} unique hypotheses")

    # auto-rank
    ranked = engine.rank_hypotheses(hypotheses)
    print(f"Top {len(ranked)} hypotheses ranked")

    # save
    engine.save_hypotheses(hypotheses, output)
    print(f"Saved to {output}")

    # print top 10
    print(f"\nTop 10:")
    for i, h in enumerate(ranked[:10], 1):
        print(format_hypothesis(h, i))
        print()


def cmd_rank(engine, input_path, top_n=20, as_json=False):
    """Load and re-rank saved hypotheses."""
    hypotheses = engine.load_hypotheses(input_path)
    ranked = engine.rank_hypotheses(hypotheses, top_n=top_n)

    if as_json:
        print(json.dumps([h.to_dict() for h in ranked], indent=2, ensure_ascii=False))
    else:
        print(f"Top {len(ranked)} hypotheses (of {len(hypotheses)} total):\n")
        for i, h in enumerate(ranked, 1):
            print(format_hypothesis(h, i))
            print()


def _base_score_for_feedback_audit(h: Hypothesis) -> float:
    return (
        (max(h.confidence_score, 0.01) ** 0.20)
        * (max(h.evidence_score, 0.01) ** 0.20)
        * (max(h.novelty_score, 0.01) ** 0.25)
        * (max(h.testability_score, 0.01) ** 0.35)
    )


def cmd_feedback_audit(input_path, feedback_state_path, output_path, top_n=0, as_json=False):
    """Quantify how closed-loop feedback changes saved hypothesis ranking."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    raw_hypotheses = raw.get("hypotheses", raw if isinstance(raw, list) else [])
    hypotheses = [Hypothesis.from_dict(h) for h in raw_hypotheses]
    if top_n and top_n > 0:
        hypotheses = sorted(hypotheses, key=lambda h: h.composite_score, reverse=True)[:top_n]

    feedback = FeedbackState.load(feedback_state_path)
    before = sorted(hypotheses, key=lambda h: h.composite_score, reverse=True)
    before_rank = {id(h): i + 1 for i, h in enumerate(before)}

    rows = []
    for h in hypotheses:
        base_score = _base_score_for_feedback_audit(h)
        adjustment = feedback.score(h)
        adjusted_score = feedback.apply(base_score, adjustment)
        rows.append(
            {
                "hypothesis_id": h.id,
                "source_id": h.source_id,
                "source_name": h.source_name,
                "target_id": h.target_id,
                "target_name": h.target_name,
                "base_composite_score": float(base_score),
                "previous_composite_score": float(h.composite_score),
                "adjusted_composite_score": float(adjusted_score),
                "previous_rank": before_rank[id(h)],
                **adjustment.as_dict(),
            }
        )

    rows.sort(key=lambda r: r["adjusted_composite_score"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["adjusted_rank"] = i
        row["rank_shift"] = int(row["previous_rank"] - i)

    summary = {
        "input": str(input_path),
        "feedback_state": str(feedback_state_path),
        "n_feedback_records": len(feedback.records),
        "n_hypotheses_audited": len(rows),
        "n_matched": sum(1 for r in rows if r["matched_records"] > 0),
        "n_exact_supported": sum(1 for r in rows if r["exact_supported"]),
        "n_exact_contradicted": sum(1 for r in rows if r["exact_contradicted"]),
        "n_exact_execution_failed": sum(1 for r in rows if r["exact_execution_failed"]),
        "n_downweighted": sum(1 for r in rows if r["adjusted_composite_score"] < r["base_composite_score"]),
        "n_upweighted": sum(1 for r in rows if r["adjusted_composite_score"] > r["base_composite_score"]),
    }
    payload = {"summary": summary, "hypotheses": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Feedback records: {summary['n_feedback_records']}")
        print(f"Hypotheses audited: {summary['n_hypotheses_audited']}")
        print(f"Matched by feedback: {summary['n_matched']}")
        print(f"Downweighted / upweighted: {summary['n_downweighted']} / {summary['n_upweighted']}")
        print(f"Saved audit: {output_path}")


def cmd_paths(engine, source_query, target_query, max_hops=3, max_paths=20, as_json=False):
    source_id = engine.resolve_name(source_query)
    target_id = engine.resolve_name(target_query)

    if not source_id:
        print(f"Concept not found: '{source_query}'")
        suggestions = engine.kg.search_by_name(source_query, limit=5)
        if suggestions:
            print("  Did you mean:")
            for s in suggestions:
                print(f"    {s.id} - {s.preferred_name}")
        return

    if not target_id:
        print(f"Concept not found: '{target_query}'")
        suggestions = engine.kg.search_by_name(target_query, limit=5)
        if suggestions:
            print("  Did you mean:")
            for s in suggestions:
                print(f"    {s.id} - {s.preferred_name}")
        return

    hypotheses = engine.find_paths(source_id, target_id, max_hops, max_paths)

    if not hypotheses:
        print(f"No paths found between '{source_query}' and '{target_query}'")
        print("  They may be in different connected components. Try 'bridge' instead.")
        return

    if as_json:
        print(json.dumps([h.to_dict() for h in hypotheses], indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(hypotheses)} hypothesis path(s):\n")
        for i, h in enumerate(hypotheses, 1):
            print(format_hypothesis(h, i))
            print()


def cmd_bridge(engine, concept_query, target_domain, max_hops=3, max_results=20, as_json=False):
    concept_id = engine.resolve_name(concept_query)
    if not concept_id:
        print(f"Concept not found: '{concept_query}'")
        suggestions = engine.kg.search_by_name(concept_query, limit=5)
        if suggestions:
            print("  Did you mean:")
            for s in suggestions:
                print(f"    {s.id} - {s.preferred_name}")
        return

    hypotheses = engine.bridge_discovery(concept_id, target_domain, max_hops, max_results)

    if not hypotheses:
        print(f"No bridge connections found from '{concept_query}' to domain '{target_domain}'")
        return

    if as_json:
        print(json.dumps([h.to_dict() for h in hypotheses], indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(hypotheses)} bridge connection(s):\n")
        for i, h in enumerate(hypotheses, 1):
            print(format_hypothesis(h, i))
            print()


def cmd_contradictions(engine, domain=None, max_results=50, as_json=False):
    contradictions = engine.contradiction_detection(domain_filter=domain, max_results=max_results)

    if not contradictions:
        print("No contradictions found.")
        return

    if as_json:
        print(json.dumps([asdict(c) for c in contradictions], indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(contradictions)} contradiction(s):\n")
        for i, c in enumerate(contradictions, 1):
            print(format_contradiction(c, i))
            print()


def cmd_gaps(engine, domain_a, domain_b=None, max_results=50, as_json=False):
    gaps = engine.gap_detection(domain_a, domain_b, max_results)

    if not gaps:
        print(f"No gaps found between '{domain_a}' and '{domain_b or domain_a}'")
        return

    if as_json:
        print(json.dumps([asdict(g) for g in gaps], indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(gaps)} gap(s):\n")
        for i, g in enumerate(gaps, 1):
            print(format_gap(g, i))
            print()


def cmd_explore(engine, concept_query, max_hops=2, as_json=False):
    concept_id = engine.resolve_name(concept_query)
    if not concept_id:
        print(f"Concept not found: '{concept_query}'")
        suggestions = engine.kg.search_by_name(concept_query, limit=5)
        if suggestions:
            print("  Did you mean:")
            for s in suggestions:
                print(f"    {s.id} - {s.preferred_name}")
        return

    node = engine._index[concept_id]
    print(f"Exploring: {node.preferred_name} ({concept_id})")
    print(f"  Domains: {', '.join(node.domain_tags)}")
    print()

    all_results = {}

    for domain in ["disease", "neuroanatomy", "gene", "drug", "cognitive_function"]:
        bridges = engine.bridge_discovery(concept_id, domain, max_hops, max_results=5)
        if bridges:
            all_results[f"bridge_to_{domain}"] = bridges

    if as_json:
        output = {}
        for key, hyps in all_results.items():
            output[key] = [h.to_dict() for h in hyps]
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        for key, hyps in all_results.items():
            print(f"=== {key} ({len(hyps)} results) ===")
            for i, h in enumerate(hyps, 1):
                print(format_hypothesis(h, i))
                print()


def cmd_stats(engine):
    stats = engine.kg.stats()
    claim_count = sum(1 for n in engine._index.values() if "claim" in n.domain_tags)
    print(f"Graph: {stats['n_concepts']} concepts, {stats['n_edges']} edges")
    print(f"  Claims: {claim_count}")
    print(f"  Connected components: {stats['connected_components']}")
    print(f"  Domains: {json.dumps(stats['domains'], indent=4)}")
    print(f"  Relations: {json.dumps(stats['relations'], indent=4)}")


def cmd_discover(engine, concept_query, max_hops=3, max_results=20, as_json=False):
    """Find hypotheses radiating from a single concept."""
    concept_id = engine.resolve_name(concept_query)
    if not concept_id:
        print(f"Concept not found: '{concept_query}'")
        suggestions = engine.kg.search_by_name(concept_query, limit=5)
        if suggestions:
            print("  Did you mean:")
            for s in suggestions:
                print(f"    {s.id} - {s.preferred_name}")
        return

    hypotheses = engine.discover_hypotheses(concept_id, max_hops=max_hops, max_results=max_results)

    if not hypotheses:
        print(f"No hypotheses found for '{concept_query}'")
        return

    if as_json:
        print(json.dumps([h.to_dict() for h in hypotheses], indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(hypotheses)} hypothesis(es) from '{concept_query}':\n")
        for i, h in enumerate(hypotheses, 1):
            print(format_hypothesis(h, i))
            print()


def cmd_trending(engine, since_year=2020, direction="strengthening", min_claims=3, max_results=20, as_json=False):
    """Find concept pairs with strengthening/weakening evidence."""
    results = engine.find_trending(
        since_year=since_year,
        min_claims=min_claims,
        direction=direction,
        max_results=max_results,
    )

    if not results:
        print(f"No {direction} trends found since {since_year}")
        return

    if as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(results)} {direction} trend(s) since {since_year}:\n")
        for i, r in enumerate(results, 1):
            print(f"--- Trend #{i} ---")
            print(f"  {r['concept_a']} <-> {r['concept_b']}")
            print(f"  Claims: {r['n_claims']}, Slope: {r['slope']}")
            print(f"  Year distribution: {r['year_counts']}")
            print()


def cmd_critic(engine, input_path, top_k, output_path, max_rounds, threshold, max_workers, as_json):
    """Run Critic Agent on top-K hypotheses."""
    hypotheses = engine.load_hypotheses(input_path)
    if not hypotheses:
        print("No hypotheses found.")
        return

    # Rank to get top-K
    ranked = engine.rank_hypotheses(hypotheses, top_n=top_k, skip_post_process=True)
    to_review = ranked[:top_k]
    rest = ranked[top_k:]

    print(f"Reviewing top {len(to_review)} hypotheses (max_rounds={max_rounds}, threshold={threshold}, workers={max_workers})...")

    critic = CriticAgent(max_rounds=max_rounds, pass_threshold=threshold)
    results = critic.refine_batch(to_review, max_workers=max_workers)

    # Separate passed/failed/revised
    refined = []
    passed = 0
    failed = 0
    revised_count = 0

    for i, (final_h, rounds) in enumerate(results):
        original = to_review[i]
        if final_h.critic_score >= threshold:
            refined.append(final_h)
            passed += 1
        else:
            failed += 1
        if final_h.critic_rounds > 1:
            revised_count += 1

    # Merge: refined top-K + rest
    all_hypotheses = refined + list(rest)

    # Re-rank by composite * critic_score
    for h in all_hypotheses:
        if h.critic_score > 0:
            h.composite_score = h.composite_score * 0.7 + h.critic_score * 0.3

    all_hypotheses.sort(key=lambda h: h.composite_score, reverse=True)

    # Deduplicate by id (keep highest composite_score, which is first after sort)
    seen_ids: set = set()
    deduped = []
    n_dups = 0
    for h in all_hypotheses:
        if h.id in seen_ids:
            n_dups += 1
            continue
        seen_ids.add(h.id)
        deduped.append(h)
    if n_dups:
        print(f"  deduplicated {n_dups} duplicate id(s)")
    all_hypotheses = deduped

    out_path = output_path or input_path
    engine.save_hypotheses(all_hypotheses, out_path)

    if as_json:
        print(json.dumps({
            "reviewed": len(to_review),
            "passed": passed,
            "failed": failed,
            "revised": revised_count,
            "total_output": len(all_hypotheses),
            "output_path": out_path,
        }, indent=2))
    else:
        print(f"\nCritic Agent Results:")
        print(f"  Reviewed: {len(to_review)}")
        print(f"  Passed:   {passed}")
        print(f"  Failed:   {failed}")
        print(f"  Revised:  {revised_count}")
        print(f"  Output:   {len(all_hypotheses)} hypotheses → {out_path}")

        # Show top 5 refined
        print(f"\nTop 5 after critic:")
        for i, h in enumerate(all_hypotheses[:5]):
            print(f"  #{i+1} [{h.id}] {h.source_name} → {h.target_name}")
            print(f"      composite={h.composite_score:.3f} critic={h.critic_score:.2f} rounds={h.critic_rounds}")


def cmd_novelty(engine, input_path, top_k, output_path, alpha, skip_pubmed, skip_semantic, as_json):
    """Check hypothesis novelty against published literature."""
    from .novelty_checker import NoveltyChecker

    hypotheses = engine.load_hypotheses(input_path)
    if not hypotheses:
        print("No hypotheses found.")
        return

    if top_k > 0:
        to_check = hypotheses[:top_k]
    else:
        to_check = hypotheses

    cache_path = str(Path(input_path).parent / "novelty_cache.json")
    checker = NoveltyChecker(
        alpha=alpha,
        use_pubmed=not skip_pubmed,
        use_semantic=not skip_semantic,
        cache_path=cache_path,
    )

    print(f"Checking novelty for {len(to_check)} hypotheses (alpha={alpha})...")
    results = checker.check_batch(to_check)

    # Update hypothesis scores
    result_map = {r.hypothesis_id: r for r in results}
    for h in hypotheses:
        if h.id in result_map:
            r = result_map[h.id]
            # Preserve original structural novelty before overwriting
            h.metadata["structural_novelty"] = h.novelty_score
            h.novelty_score = r.final_novelty
            h.metadata["lit_novelty"] = r.lit_novelty
            h.metadata["pubmed_hits"] = r.pubmed_hits
            h.metadata["semantic_hits"] = r.semantic_hits
            h.metadata["final_novelty"] = r.final_novelty

    # Re-sort by composite score (novelty changed)
    for h in hypotheses:
        h.composite_score = (
            h.confidence_score ** 0.20
            * h.evidence_score ** 0.20
            * h.novelty_score ** 0.25
            * h.testability_score ** 0.35
        )
    hypotheses.sort(key=lambda h: h.composite_score, reverse=True)

    out_path = output_path or input_path
    engine.save_hypotheses(hypotheses, out_path)

    if as_json:
        print(json.dumps({
            "checked": len(to_check),
            "output_path": out_path,
            "results": [r.to_dict() for r in results],
        }, indent=2))
    else:
        print(f"\nNovelty Check Results:")
        print(f"  Checked: {len(to_check)}")
        print(f"  Output:  {len(hypotheses)} hypotheses → {out_path}")

        # Show results sorted by final_novelty
        results.sort(key=lambda r: r.final_novelty, reverse=True)
        print(f"\nTop 10 by novelty:")
        for i, r in enumerate(results[:10]):
            h = next((h for h in hypotheses if h.id == r.hypothesis_id), None)
            name = f"{h.source_name} → {h.target_name}" if h else r.hypothesis_id
            print(f"  #{i+1} [{r.hypothesis_id}] {name}")
            print(f"      pubmed={r.pubmed_hits} semantic={r.semantic_hits} lit={r.lit_novelty:.2f} final={r.final_novelty:.2f}")


def cmd_imaging_batch(engine, dataset, output, max_paths, max_seeds, max_hops, include_connectivity, as_json):
    """Generate imaging-feature-driven hypotheses for a specific dataset."""
    print(f"Generating imaging-driven hypotheses for {dataset}...")
    print(f"  max_paths={max_paths}, max_seeds={max_seeds}, max_hops={max_hops}, connectivity={include_connectivity}")

    hypotheses = engine.batch_generate_imaging(
        dataset=dataset,
        max_paths_per_pair=max_paths,
        max_seeds=max_seeds,
        max_hops=max_hops,
        include_connectivity=include_connectivity,
    )
    print(f"Generated {len(hypotheses)} imaging hypotheses")

    # auto-rank
    ranked = engine.rank_hypotheses(hypotheses)
    print(f"Top {len(ranked)} hypotheses ranked")

    # save
    engine.save_hypotheses(hypotheses, output)
    print(f"Saved to {output}")

    if as_json:
        print(json.dumps({
            "dataset": dataset,
            "total": len(hypotheses),
            "ranked": len(ranked),
            "output_path": output,
            "top_10": [h.to_dict() for h in ranked[:10]],
        }, indent=2, ensure_ascii=False))
    else:
        # modality breakdown
        modality_counts = {}
        for h in hypotheses:
            mod = h.metadata.get("input_modality", "unknown")
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        print(f"\nModality breakdown:")
        for mod, cnt in sorted(modality_counts.items(), key=lambda x: -x[1]):
            print(f"  {mod}: {cnt}")

        print(f"\nTop 10:")
        for i, h in enumerate(ranked[:10], 1):
            feat = h.metadata.get("input_feature", h.source_name)
            print(f"  #{i} [{h.id}] {feat} → {h.target_name}")
            print(f"      modality={h.metadata.get('input_modality', '?')} "
                  f"tool={h.metadata.get('input_tool', '?')} "
                  f"composite={h.composite_score:.4f}")
            if h.testability_reason:
                print(f"      testability: {h.testability_reason}")
            print()


def cmd_evolve(engine, input_path, output_path, population, generations,
               mutation_rate, crossover_rate, tournament_size, elitism_n, as_json):
    """Evolve hypotheses via mutation, crossover, and selection."""
    hypotheses = engine.load_hypotheses(input_path)
    if not hypotheses:
        print("No hypotheses found.")
        return

    seeds = hypotheses[:min(len(hypotheses), population)]
    print(f"Evolving {len(seeds)} seed hypotheses "
          f"(pop={population}, gen={generations}, mut={mutation_rate}, cross={crossover_rate})...")

    evo = EvolutionEngine(
        engine=engine,
        population_size=population,
        n_generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        elitism_n=elitism_n,
    )

    evolved = evo.evolve(seeds)

    # merge evolved with original (deduplicate by id)
    seen = {h.id for h in evolved}
    for h in hypotheses:
        if h.id not in seen:
            evolved.append(h)
            seen.add(h.id)

    # re-rank
    evolved.sort(key=lambda h: h.composite_score, reverse=True)

    out_path = output_path or input_path
    engine.save_hypotheses(evolved, out_path)

    # Dump dropped EVO variants so we can diagnose post-filter wipeouts
    # (cycle_005 saw raw_evo=0 even though K-Paths reported 15 kept).
    dropped = getattr(evo, "_dropped_evos", None)
    if dropped:
        from pathlib import Path
        diag_path = str(Path(out_path).parent / "_diag_evos.json")
        with open(diag_path, "w", encoding="utf-8") as fh:
            json.dump({"n_dropped": len(dropped), "dropped": dropped},
                      fh, ensure_ascii=False, indent=2)
        print(f"  diag: dumped {len(dropped)} dropped EVO variants -> {diag_path}")

    # count operators used
    op_counts = {}
    for h in evolved:
        op = h.metadata.get("operator", "")
        if op:
            op_counts[op] = op_counts.get(op, 0) + 1

    if as_json:
        print(json.dumps({
            "seed_count": len(seeds),
            "evolved_count": len(evolved),
            "output_path": out_path,
            "operator_counts": op_counts,
        }, indent=2))
    else:
        print(f"\nEvolution Results:")
        print(f"  Seeds:   {len(seeds)}")
        print(f"  Evolved: {len(evolved)}")
        print(f"  Output:  {out_path}")
        if op_counts:
            print(f"  Operators used:")
            for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
                print(f"    {op}: {cnt}")

        # show top 10
        print(f"\nTop 10 after evolution:")
        for i, h in enumerate(evolved[:10]):
            op = h.metadata.get("operator", "original")
            print(f"  #{i+1} [{h.id}] {h.source_name} → {h.target_name}")
            print(f"      composite={h.composite_score:.4f} fitness={h.metadata.get('fitness', 0):.4f} op={op}")


# ── main ───────────────────────────────────────────────────────────────


def cmd_im_brainstorm(graph_path: str, output: str, n: int = 50,
                      model: Optional[str] = None, batch_size: int = 30,
                      seed: int = 0) -> None:
    """Brainstorm imaging markers (IMs) from KG primitives via LLM.

    The LLM sees a structured palette pulled from the KG: imaging modalities
    (sMRI/dMRI/fMRI/PET/EEG/MEG), the 15 IF:* operations (thickness, FA, FC,
    SUVR, ...), top-degree NN/VROI regions, and Cognitive Atlas tasks /
    concepts. It composes IMs by picking modality + operation + regions
    (and optional task conditioning) from this palette — never inventing
    new ones. Outputs are validated against a static modality<->operation
    compatibility table; rejects are logged with reasons.

    Output JSON: ``{"metadata": {...}, "palette": {...},
    "imaging_markers": [...]}``. IMs are NOT injected into the KG —
    disease/gene -> IM edges come from Phase 2 paper extraction.
    """
    import os
    import httpx
    from openai import OpenAI

    from .recipe import (build_im_palette, brainstorm_ims,
                          brainstorm_ims_batched, validate_ims,
                          link_ims_to_kg, tag_atoms)

    with open(graph_path, encoding="utf-8") as f:
        kg = json.load(f)
    concepts = kg.get("concepts") or {}
    edges = kg.get("edges") or []
    palette = build_im_palette(concepts, edges=edges)
    print(f"palette: {len(palette.modalities)} modalities, "
          f"{len(palette.operations)} ops, "
          f"{len(palette.core_regions)} core + "
          f"{len(palette.enigma_regions)} ENIGMA + "
          f"{len(palette.visual_rois)} VROI regions, "
          f"{len(palette.tasks)} tasks, {len(palette.concepts)} concepts")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    keys_raw = os.environ.get("OPENAI_API_KEYS", "") or os.environ.get("OPENAI_API_KEY", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS / OPENAI_API_KEY not set")
    client = OpenAI(base_url=base_url, api_key=keys[0],
                     http_client=httpx.Client(verify=False))

    def _llm_call(prompt: str, system_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return resp.choices[0].message.content.strip()

    if n <= batch_size:
        raw = brainstorm_ims(palette, n=n, llm_call=_llm_call,
                              model_name=model_name)
    else:
        raw = brainstorm_ims_batched(
            palette, n_total=n, llm_call=_llm_call,
            model_name=model_name, batch_size=batch_size, seed=seed)
    print(f"LLM returned {len(raw)} raw IMs (after batch dedup)")

    report = validate_ims(raw, palette)
    print(f"validation: {report.n_accepted} accepted, "
          f"{report.n_rejected} rejected -> {report.reject_reasons()}")

    accepted = report.accepted
    link_ims_to_kg(accepted, palette)
    tag_atoms(accepted)

    from collections import Counter
    family_counts = Counter(im.family for im in accepted)
    modality_counts = Counter(im.modality for im in accepted)
    print(f"family coverage:   {dict(family_counts.most_common())}")
    print(f"modality coverage: {dict(modality_counts.most_common())}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "n_imaging_markers": len(accepted),
            "n_rejected":        report.n_rejected,
            "reject_reasons":    report.reject_reasons(),
            "llm_model":         model_name,
        },
        "palette": palette.to_dict(),
        "imaging_markers": [im.to_dict() for im in accepted],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote -> {out_path}")


# == main =================================================================


def cmd_gm_brainstorm(graph_path: str, output: str, n: int = 50,
                       model: Optional[str] = None, batch_size: int = 30,
                       seed: int = 0) -> None:
    """Brainstorm genetic markers (GMs) from KG primitives + curated palette.

    Mirrors `cmd_im_brainstorm`: the LLM sees data types
    (genotype_array/wgs/wes/rnaseq/methylation/mtdna_seq), operations
    (PRS, rare-variant burden, expression aggregate, methylation clock,
    TWAS, ...), curated gene sets (AD/PD/Synaptic/Dopaminergic/...),
    top-degree GENE:* symbols, GTEx brain v9 tissues, methylation clock
    names, and disease GWAS sources. It composes GMs by picking from this
    palette; output is validated against family/operation/data-type
    compatibility tables and family-specific slot requirements.

    Output JSON: ``{"metadata": {...}, "palette": {...},
    "genetic_markers": [...]}``. GMs are NOT injected into the KG;
    disease/anatomy -> GM edges come from Phase 2 paper extraction.
    """
    import os
    import httpx
    from openai import OpenAI

    from .recipe import (build_gm_palette, brainstorm_gms,
                          brainstorm_gms_batched, validate_gms,
                          link_gms_to_kg, tag_gm_atoms)

    with open(graph_path, encoding="utf-8") as f:
        kg = json.load(f)
    concepts = kg.get("concepts") or {}
    edges = kg.get("edges") or []
    palette = build_gm_palette(concepts, edges=edges)
    print(f"palette: {len(palette.data_types)} data types, "
          f"{len(palette.operations)} ops, "
          f"{len(palette.gene_sets)} gene sets, "
          f"{len(palette.top_genes)} top genes, "
          f"{len(palette.tissues)} tissues, "
          f"{len(palette.clocks)} clocks, "
          f"{len(palette.diseases)} diseases")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    keys_raw = os.environ.get("OPENAI_API_KEYS", "") or os.environ.get("OPENAI_API_KEY", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("OPENAI_API_KEYS / OPENAI_API_KEY not set")
    client = OpenAI(base_url=base_url, api_key=keys[0],
                     http_client=httpx.Client(verify=False))

    def _llm_call(prompt: str, system_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return resp.choices[0].message.content.strip()

    if n <= batch_size:
        raw = brainstorm_gms(palette, n=n, llm_call=_llm_call,
                              model_name=model_name)
    else:
        raw = brainstorm_gms_batched(
            palette, n_total=n, llm_call=_llm_call,
            model_name=model_name, batch_size=batch_size, seed=seed)
    print(f"LLM returned {len(raw)} raw GMs (after batch dedup)")

    report = validate_gms(raw, palette)
    print(f"validation: {report.n_accepted} accepted, "
          f"{report.n_rejected} rejected -> {report.reject_reasons()}")

    accepted = report.accepted
    link_gms_to_kg(accepted, palette, concepts)
    tag_gm_atoms(accepted)

    from collections import Counter
    family_counts = Counter(gm.family for gm in accepted)
    op_counts = Counter(gm.operation for gm in accepted)
    print(f"family coverage:    {dict(family_counts.most_common())}")
    print(f"operation coverage: {dict(op_counts.most_common())}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "n_genetic_markers": len(accepted),
            "n_rejected":        report.n_rejected,
            "reject_reasons":    report.reject_reasons(),
            "llm_model":         model_name,
        },
        "palette": palette.to_dict(),
        "genetic_markers": [gm.to_dict() for gm in accepted],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote -> {out_path}")


# == main =================================================================

def cmd_case_study(case_study_name, output_dir, stages, kge_path, kg_path,
                   graph_path, snapshot_2022_kg=None, snapshot_2022_kge=None,
                   feedback_state=None, as_json=False):
    """Run the four-stage autoresearch cycle for a registered case study.

    Reads a :class:`CaseStudy` from :mod:`case_studies`, dispatches stage
    [1/4] (raw generation) by ``case.generator``, then forwards the rest
    of the pipeline to the existing ``cmd_novelty / cmd_critic /
    cmd_plausibility`` with parameters from ``case.stage_params``.

    Output layout mirrors run_cycle.sh:
        <output_dir>/hypotheses_raw.json    [1/4]
        <output_dir>/hypotheses_novel.json  [2/4]
        <output_dir>/hypotheses_critic.json [3/4]
        <output_dir>/hypotheses_final.json  [4/4]
    """
    from .case_studies import (
        case_study_by_name,
        GENERATOR_TASK, GENERATOR_CHAIN,
        GENERATOR_CASE1_CANDIDATE, GENERATOR_ATOM_SUBSTITUTION,
    )

    case = case_study_by_name(case_study_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_out    = out_dir / "hypotheses_raw.json"
    novel_out  = out_dir / "hypotheses_novel.json"
    critic_out = out_dir / "hypotheses_critic.json"
    final_out  = out_dir / "hypotheses_final.json"

    print("=" * 72)
    print(f"Case study: {case.name}")
    print(f"  CN: {case.chinese_name}")
    print(f"  EN: {case.english_name}")
    print(f"  generator: {case.generator}")
    if case.task is not None:
        print(f"  task:  {case.task.name} [{case.task.signature}]")
    if case.chain is not None:
        print(f"  chain: {case.chain.name} [{case.chain.signature}]")
    print(f"  output_dir: {out_dir}")
    print(f"  stages: {stages}")
    print("=" * 72)

    stages_set = {s.strip() for s in stages.split(",") if s.strip()}

    # ── Stage [1/4]: generation ────────────────────────────────────────
    if "batch" in stages_set:
        bp = case.stage_params.batch
        if case.generator == GENERATOR_TASK:
            kg = load_graph(Path(graph_path))
            engine = HypothesisEngine(kg)
            if feedback_state:
                engine.load_feedback_state(feedback_state)
            for hook in case.pre_hooks:
                hook(engine, case)
            cmd_batch(
                engine, str(raw_out),
                max_hops=bp.max_hops, min_hops=bp.min_hops,
                metapath_min_domains=bp.metapath_min_domains,
                max_paths=bp.max_paths, max_seeds=bp.max_seeds,
                target_per_task=bp.target_per_task,
                max_retries=bp.max_retries, retry_scale=bp.retry_scale,
                prefer_longer_paths=bp.prefer_longer_paths,
                task_filter=case.task.name, chain_filter="",
                as_json=as_json,
            )
            for hook in case.post_hooks:
                hook(engine, case, raw_out)

        elif case.generator == GENERATOR_CHAIN:
            kg = load_graph(Path(graph_path))
            engine = HypothesisEngine(kg)
            if feedback_state:
                engine.load_feedback_state(feedback_state)
            for hook in case.pre_hooks:
                hook(engine, case)
            cmd_batch(
                engine, str(raw_out),
                max_hops=bp.max_hops, min_hops=bp.min_hops,
                metapath_min_domains=bp.metapath_min_domains,
                max_paths=bp.max_paths, max_seeds=bp.max_seeds,
                target_per_task=bp.target_per_task,
                max_retries=bp.max_retries, retry_scale=bp.retry_scale,
                prefer_longer_paths=bp.prefer_longer_paths,
                task_filter="", chain_filter=case.chain.name,
                as_json=as_json,
            )
            for hook in case.post_hooks:
                hook(engine, case, raw_out)

        elif case.generator == GENERATOR_CASE1_CANDIDATE:
            kg = load_graph(Path(graph_path))
            engine = HypothesisEngine(kg)
            if feedback_state:
                engine.load_feedback_state(feedback_state)
            for hook in case.pre_hooks:
                hook(engine, case)
            extras = case.extras or {}
            print(f"  case1 candidate-space generation: task={case.task.name if case.task else '?'}")
            hypotheses = engine.generate_case1_hypotheses(
                methods=tuple(extras.get("generation_methods", ())),
                disease_names=tuple(extras.get("disease_include_names", ())),
                atlas_rois=tuple(extras.get("atlas_rois", ())),
                atlas_label_names=tuple(extras.get("atlas_label_names", ())),
                atlas_label_sources=dict(extras.get("atlas_label_sources", {})),
                feature_space=tuple(extras.get("feature_space", ())),
                max_per_method=extras.get("max_hypotheses_per_method", bp.target_per_task),
                random_seed=int(extras.get("random_seed", 0)) or None,
            )
            print(f"  -> {len(hypotheses)} candidate hypothesis(es)")
            for h in hypotheses:
                meta = h.metadata or {}
                if case.task is not None:
                    meta["task_name"] = case.task.name
                    meta["task_signature"] = case.task.signature
                    meta["task_modifier"] = (
                        case.task.modifier.value
                        if case.task.modifier is not None
                        else None
                    )
                meta["case_study"] = case.name
                h.metadata = meta
            engine.save_hypotheses(hypotheses, str(raw_out))
            print(f"Saved to {raw_out}")
            for hook in case.post_hooks:
                hook(engine, case, raw_out)

        elif case.generator == GENERATOR_ATOM_SUBSTITUTION:
            raise NotImplementedError(
                f"case-study '{case.name}' uses generator='atom_substitution'; "
                "implementation lands in Case Study 3 hindcasting."
            )

        else:
            raise ValueError(f"unknown generator: {case.generator}")

    # ── Stage [2/4]: novelty ───────────────────────────────────────────
    if "novelty" in stages_set:
        if not raw_out.exists():
            raise FileNotFoundError(f"stage 'novelty' needs {raw_out} from stage 'batch'")
        kg = load_graph(Path(graph_path))
        engine = HypothesisEngine(kg)
        np_ = case.stage_params.novelty
        cmd_novelty(
            engine, str(raw_out), np_.top, str(novel_out), np_.alpha,
            skip_pubmed=np_.skip_pubmed, skip_semantic=np_.skip_semantic,
            as_json=as_json,
        )

    # ── Stage [3/4]: critic ────────────────────────────────────────────
    if "critic" in stages_set:
        if not novel_out.exists():
            raise FileNotFoundError(f"stage 'critic' needs {novel_out} from stage 'novelty'")
        kg = load_graph(Path(graph_path))
        engine = HypothesisEngine(kg)
        cp = case.stage_params.critic
        cmd_critic(
            engine, str(novel_out), cp.top, str(critic_out),
            cp.max_rounds, cp.threshold, cp.max_workers,
            as_json=as_json,
        )

    # ── Stage [4/4]: plausibility ──────────────────────────────────────
    if "plausibility" in stages_set:
        if not critic_out.exists():
            raise FileNotFoundError(f"stage 'plausibility' needs {critic_out} from stage 'critic'")
        from .kge.cli import cmd_plausibility
        pp = case.stage_params.plausibility
        cmd_plausibility(
            input_path=str(critic_out),
            kge_checkpoint=kge_path,
            output=str(final_out),
            novelty_cache=str(out_dir / "novelty_cache.json"),
            skip_existing=pp.skip_existing,
            no_pubmed=pp.no_pubmed,
            top=pp.top,
            device=pp.device,
            kg_path=kg_path,
            enable_surprise=pp.enable_surprise,
            surprise_alpha=pp.surprise_alpha,
            evo_surprise_min=pp.evo_surprise_min,
        )

    print(f"=== case-study '{case.name}' done -> {out_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Hypothesis engine for NeuroClaw knowledge graph")
    parser.add_argument("--graph", default=None, help="Path to graph JSON")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--feedback-state",
        default=None,
        help=(
            "Optional JSON/JSONL closed-loop feedback records. Supported statuses: "
            "supported, contradicted, execution_failed."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # batch
    p_batch = sub.add_parser("batch", help="Batch-generate hypotheses across the entire graph")
    p_batch.add_argument("--output", default="neurooracle/data/hypotheses_baseline.json", help="Output JSON path")
    p_batch.add_argument("--max-hops", type=int, default=4)
    p_batch.add_argument("--min-hops", type=int, default=2,
                          help="Minimum edges per kept path. 2 = the historical "
                               "no-direct-edge guarantee; 3 = force >=3-hop chains.")
    p_batch.add_argument("--metapath-min-domains", type=int, default=2,
                          help="Minimum number of distinct atom domains a path "
                               "must touch. 2 = source/target in different "
                               "domains; 3 = require an explicit cross-domain "
                               "mediator between them.")
    p_batch.add_argument("--no-prefer-longer-paths", action="store_true",
                          help="Disable the longer-paths-first preference when "
                               "the candidate set exceeds the per-pair quota.")
    p_batch.add_argument("--max-paths", type=int, default=5, help="Max paths per domain pair seed")
    p_batch.add_argument("--max-seeds", type=int, default=50, help="Max seed concepts per domain")
    p_batch.add_argument("--legacy-domain-pairs", action="store_true",
                          help="Use raw KG domain-pair traversal (pre-atom-algebra). "
                               "Default is task-driven generation over CANONICAL_TASKS + CANONICAL_CHAINS.")
    p_batch.add_argument("--domain-pairs", choices=["default", "imaging", "decoding", "all"],
                          default="default",
                          help="(legacy mode only) Which domain-pair set to traverse: default (clinical "
                               "outcomes), imaging (UKB/ADNI/HCP imaging), decoding (NSD/BOLD5000/SEED "
                               "brain decoding), or all (union of the three)")
    p_batch.add_argument("--tasks", default=None,
                          help="Comma-separated task names to run (default: all CANONICAL_TASKS). "
                               "Only used in task-driven mode.")
    p_batch.add_argument("--chains", default=None,
                          help="Comma-separated chain names to run (default: all CANONICAL_CHAINS). "
                               "Only used in task-driven mode. Pass '' to skip chains entirely.")
    p_batch.add_argument("--target-per-task", type=int, default=None,
                          help="Retry generation per task/chain until at least N unique hypotheses "
                               "are kept after post_process (default: no retry).")
    p_batch.add_argument("--max-retries", type=int, default=3,
                          help="Max retry attempts per task/chain when --target-per-task is set.")
    p_batch.add_argument("--retry-scale", type=float, default=2.0,
                          help="Scale factor applied to max-paths and max-seeds on each retry.")

    # rank
    p_rank = sub.add_parser("rank", help="Load and re-rank saved hypotheses")
    p_rank.add_argument("--input", default="neurooracle/data/hypotheses_baseline.json", help="Input JSON path")
    p_rank.add_argument("--top", type=int, default=20)

    # feedback-audit
    p_feedback = sub.add_parser(
        "feedback-audit",
        help="Audit how supported/contradicted/execution_failed feedback changes ranking",
    )
    p_feedback.add_argument("--input", required=True, help="Input hypotheses JSON path")
    p_feedback.add_argument("--feedback-state", required=True, help="Feedback JSON/JSONL path")
    p_feedback.add_argument("--output", required=True, help="Output audit JSON path")
    p_feedback.add_argument("--top", type=int, default=0, help="Audit only top K by previous composite score (0=all)")

    # paths
    p_paths = sub.add_parser("paths", help="Find hypothesis paths between two concepts")
    p_paths.add_argument("source")
    p_paths.add_argument("target")
    p_paths.add_argument("--max-hops", type=int, default=3)
    p_paths.add_argument("--max-paths", type=int, default=20)

    # bridge
    p_bridge = sub.add_parser("bridge", help="Cross-domain bridge discovery")
    p_bridge.add_argument("concept")
    p_bridge.add_argument("--target-domain", required=True)
    p_bridge.add_argument("--max-hops", type=int, default=3)
    p_bridge.add_argument("--max-results", type=int, default=20)

    # contradictions
    p_contra = sub.add_parser("contradictions", help="Find contradictory claims")
    p_contra.add_argument("--domain", default=None)
    p_contra.add_argument("--max-results", type=int, default=50)

    # gaps
    p_gaps = sub.add_parser("gaps", help="Detect unexplored relationships")
    p_gaps.add_argument("--domain-a", required=True)
    p_gaps.add_argument("--domain-b", default=None)
    p_gaps.add_argument("--max-results", type=int, default=50)

    # explore
    p_explore = sub.add_parser("explore", help="General exploration of a concept")
    p_explore.add_argument("concept")
    p_explore.add_argument("--max-hops", type=int, default=2)

    # stats
    sub.add_parser("stats", help="Show graph statistics")

    # discover
    p_discover = sub.add_parser("discover", help="Find hypotheses radiating from a concept")
    p_discover.add_argument("concept")
    p_discover.add_argument("--max-hops", type=int, default=3)
    p_discover.add_argument("--max-results", type=int, default=20)

    # trending
    p_trending = sub.add_parser("trending", help="Find strengthening/weakening evidence trends")
    p_trending.add_argument("--since", type=int, default=2020, help="Start year")
    p_trending.add_argument("--direction", choices=["strengthening", "weakening", "emerging"], default="strengthening")
    p_trending.add_argument("--min-claims", type=int, default=3)
    p_trending.add_argument("--max-results", type=int, default=20)

    # critic
    p_critic = sub.add_parser("critic", help="Review top-K hypotheses with Critic Agent")
    p_critic.add_argument("--input", required=True, help="Input hypotheses JSON file")
    p_critic.add_argument("--top", type=int, default=20, help="Review top K hypotheses")
    p_critic.add_argument("--output", default=None, help="Output file (default: overwrite input)")
    p_critic.add_argument("--max-rounds", type=int, default=3, help="Max refinement rounds")
    p_critic.add_argument("--threshold", type=float, default=0.6, help="Pass threshold (0-1)")
    p_critic.add_argument("--max-workers", type=int, default=12, help="Parallel workers for batch review (12 recommended with 4 API keys)")

    # novelty
    p_novelty = sub.add_parser("novelty", help="Check hypothesis novelty against literature")
    p_novelty.add_argument("--input", required=True, help="Input hypotheses JSON file")
    p_novelty.add_argument("--top", type=int, default=0, help="Check top K (0=all)")
    p_novelty.add_argument("--output", default=None, help="Output file (default: overwrite input)")
    p_novelty.add_argument("--alpha", type=float, default=0.5, help="Weight for graph novelty (0-1)")
    p_novelty.add_argument("--no-pubmed", action="store_true", help="Skip PubMed check")
    p_novelty.add_argument("--no-semantic", action="store_true", help="Skip Semantic Scholar check")

    # evolve
    p_evolve = sub.add_parser("evolve", help="Evolve hypotheses via mutation/crossover/selection")

    # imaging-batch
    p_imaging = sub.add_parser("imaging-batch", help="Generate imaging-feature-driven hypotheses for a dataset")
    p_imaging.add_argument("--dataset", default="UKB",
                           choices=["UKB", "ADNI", "HCP_YA",
                                    "ABIDE", "ADHD200", "COBRE",
                                    "UCLA", "HCP_EP", "HCP_AGING"],
                           help="Target dataset (default: UKB)")
    p_imaging.add_argument("--output", default="neurooracle/data/hypotheses_imaging.json",
                           help="Output JSON path")
    p_imaging.add_argument("--max-paths", type=int, default=5, help="Max paths per ROI-outcome pair")
    p_imaging.add_argument("--max-seeds", type=int, default=50, help="Max AAL region seeds")
    p_imaging.add_argument("--max-hops", type=int, default=3, help="Max hops in graph traversal")
    p_imaging.add_argument("--no-connectivity", action="store_true",
                           help="Skip connectivity features (FC/EC/SC)")

    p_evolve.add_argument("--input", required=True, help="Input hypotheses JSON file")
    p_evolve.add_argument("--output", default=None, help="Output file (default: overwrite input)")
    p_evolve.add_argument("--population", type=int, default=50, help="Population size")
    p_evolve.add_argument("--generations", type=int, default=10, help="Number of generations")
    p_evolve.add_argument("--mutation-rate", type=float, default=0.5, help="Mutation probability (0-1)")
    p_evolve.add_argument("--crossover-rate", type=float, default=0.3, help="Crossover probability (0-1)")
    p_evolve.add_argument("--tournament-size", type=int, default=3, help="Tournament selection size")
    p_evolve.add_argument("--elitism", type=int, default=5, help="Elite individuals preserved per generation")

    # kge-train (Phase 4.3 — plausibility scorer)
    p_kge_train = sub.add_parser("kge-train", help="Train ComplEx KG embedding for plausibility scoring")
    p_kge_train.add_argument("--kg", required=True, help="Path to knowledge_graph.json")
    p_kge_train.add_argument("--output", required=True, help="Output checkpoint path (.pt)")
    p_kge_train.add_argument("--report", default=None, help="Optional JSON report with AUROC + loss curve")
    p_kge_train.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    p_kge_train.add_argument("--epochs", type=int, default=50)
    p_kge_train.add_argument("--batch-size", type=int, default=1024)
    p_kge_train.add_argument("--lr", type=float, default=1e-3)
    p_kge_train.add_argument("--negatives-per-pos", type=int, default=10)
    p_kge_train.add_argument("--weight-decay", type=float, default=1e-6,
                              help="L2 regularisation strength (default 1e-6)")
    p_kge_train.add_argument("--eval-every", type=int, default=5,
                              help="Run val AUROC every N epochs (default 5)")
    p_kge_train.add_argument("--early-stop-patience", type=int, default=0,
                              help="Stop if val AUROC hasn't improved for N evals (0 = off)")
    p_kge_train.add_argument("--min-confidence", type=float, default=0.2,
                              help="Drop edges with confidence below this (default 0.2)")
    p_kge_train.add_argument("--seed", type=int, default=42)
    p_kge_train.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    # plausibility (Phase 4.3 — score hypotheses with trained ComplEx)
    p_plaus = sub.add_parser("plausibility", help="Score hypotheses with KG plausibility + PubMed attestation")
    p_plaus.add_argument("--input", required=True, help="Hypotheses JSON to score")
    p_plaus.add_argument("--kge", required=True, help="Trained KGE checkpoint (.pt)")
    p_plaus.add_argument("--output", default=None, help="Output file (default: overwrite input)")
    p_plaus.add_argument("--novelty-cache", default=None,
                          help="Reuse PubMed novelty cache file to avoid re-querying")
    p_plaus.add_argument("--no-pubmed", action="store_true",
                          help="Skip PubMed attestation; only compute kge_score")
    p_plaus.add_argument("--top", type=int, default=0,
                          help="Score only top-K by composite_score (0 = all)")
    p_plaus.add_argument("--no-skip-existing", action="store_true",
                          help="Re-score even hypotheses already having kge_score (default: skip)")
    p_plaus.add_argument("--kg", default=None,
                          help="Optional KG path; enables hub/CLM specificity filter")
    p_plaus.add_argument("--enable-surprise", action="store_true",
                          help="Add alpha * max(0, surprise_gap) into composite_score (default OFF)")
    p_plaus.add_argument("--surprise-alpha", type=float, default=0.1,
                          help="Weight for surprise_gap when --enable-surprise is set (default 0.1)")
    p_plaus.add_argument("--evo-surprise-min", type=float, default=None,
                          help="Drop EVO:* hypotheses with surprise_gap below this threshold "
                               "(e.g. 0.0 = require positive surprise; default: no filter)")
    p_plaus.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    # im-brainstorm (Phase 1 IM catalogue: LLM-brainstormed imaging markers)
    p_im = sub.add_parser("im-brainstorm",
                            help="Brainstorm imaging markers (IMs) from KG primitives")
    p_im.add_argument("--graph", dest="im_graph", default=None,
                        help="KG path (default: --graph or neurooracle/data/full_snapshot_v2/knowledge_graph.json)")
    p_im.add_argument("--output", default="neurooracle/data/full_snapshot_v2/imaging_markers.json")
    p_im.add_argument("--n", type=int, default=50, help="Number of IMs to brainstorm")
    p_im.add_argument("--model", default=None, help="LLM model override")
    p_im.add_argument("--batch-size", type=int, default=30,
                        help="IMs per LLM call when n > batch_size (default 30)")
    p_im.add_argument("--seed", type=int, default=0,
                        help="RNG seed for batched family rotation")

    # gm-brainstorm (Phase 1 GM catalogue: LLM-brainstormed genetic markers)
    p_gm = sub.add_parser("gm-brainstorm",
                            help="Brainstorm genetic markers (GMs) from KG primitives")
    p_gm.add_argument("--graph", dest="gm_graph", default=None,
                        help="KG path (default: --graph or neurooracle/data/full_snapshot_v2/knowledge_graph.json)")
    p_gm.add_argument("--output", default="neurooracle/data/full_snapshot_v2/genetic_markers.json")
    p_gm.add_argument("--n", type=int, default=50, help="Number of GMs to brainstorm")
    p_gm.add_argument("--model", default=None, help="LLM model override")
    p_gm.add_argument("--batch-size", type=int, default=30,
                        help="GMs per LLM call when n > batch_size (default 30)")
    p_gm.add_argument("--seed", type=int, default=0,
                        help="RNG seed for batched family rotation")

    # case-study (Nature paper rollout — orchestrates the 4-stage cycle for
    # one of Case Study 1 / 2 / 3; reads stage params from case_studies.py)
    from .case_studies import list_case_study_names
    p_cs = sub.add_parser("case-study",
                          help="Run autoresearch cycle for a registered Nature-paper case study")
    p_cs.add_argument("name", choices=list(list_case_study_names()),
                      help="Which case study to run")
    p_cs.add_argument("--output-dir", required=True,
                      help="Directory for stage outputs (raw / novel / critic / final JSONs)")
    p_cs.add_argument("--stages", default="batch,novelty,critic,plausibility",
                      help="Comma-separated subset of {batch,novelty,critic,plausibility} to run")
    p_cs.add_argument("--kge",
                      default="neurooracle/data/full_snapshot_v2/kge_complex.pt",
                      help="Trained KGE checkpoint for stage [4/4]")
    p_cs.add_argument("--kg-for-plausibility",
                      default=None,
                      help="KG path passed to plausibility stage (default: --graph)")
    p_cs.add_argument("--snapshot-2022-kg",
                      default=None,
                      help="(Case Study 3 only) Path to 2022 KG snapshot; overrides extras default")
    p_cs.add_argument("--snapshot-2022-kge",
                      default=None,
                      help="(Case Study 3 only) Path to 2022 KGE checkpoint; overrides extras default")

    # host-agent autoresearch: file-based protocol for Codex / Claude Code /
    # Cursor and similar host agents that provide their own model.
    p_ha_init = sub.add_parser(
        "host-agent-init",
        help="Create a host-agent-driven autoresearch run and first task",
    )
    p_ha_init.add_argument("case_study", choices=list(list_case_study_names()),
                           help="Which case study to run")
    p_ha_init.add_argument("--output-dir", required=True,
                           help="Directory for host-agent run_state, tasks, and outputs")
    p_ha_init.add_argument("--graph", dest="ha_graph", default=None,
                           help="KG path for deterministic NeuroOracle support stages")
    p_ha_init.add_argument("--kge", default="neurooracle/data/full_snapshot_v2/kge_complex.pt",
                           help="KGE checkpoint for deterministic plausibility support")
    p_ha_init.add_argument("--max-rounds", type=int, default=5,
                           help="Maximum host-agent autoresearch rounds")
    p_ha_init.add_argument("--deterministic-stages", default="batch,novelty",
                           help="Case-study stages the host agent may run for support artifacts")

    p_ha_next = sub.add_parser(
        "host-agent-next",
        help="Validate current host-agent output and create the next task",
    )
    p_ha_next.add_argument("--run-dir", required=True,
                           help="Host-agent autoresearch run directory")

    p_ha_status = sub.add_parser(
        "host-agent-status",
        help="Print host-agent autoresearch run_state.json",
    )
    p_ha_status.add_argument("--run-dir", required=True,
                             help="Host-agent autoresearch run directory")

    args = parser.parse_args()

    if args.command == "host-agent-init":
        from .host_agent_autoresearch import init_run
        graph_p = args.ha_graph or args.graph or "neurooracle/data/full_snapshot_v2/knowledge_graph.json"
        result = init_run(
            case_study=args.case_study,
            output_dir=args.output_dir,
            graph_path=graph_p,
            kge_path=args.kge,
            max_rounds=args.max_rounds,
            deterministic_stages=args.deterministic_stages,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "host-agent-next":
        from .host_agent_autoresearch import advance_run
        result = advance_run(args.run_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "host-agent-status":
        from .host_agent_autoresearch import load_status
        print(json.dumps(load_status(args.run_dir), indent=2, ensure_ascii=False))
        return

    if args.command == "feedback-audit":
        cmd_feedback_audit(
            input_path=args.input,
            feedback_state_path=args.feedback_state,
            output_path=args.output,
            top_n=args.top,
            as_json=args.json,
        )
        return

    # Commands that don't need the full HypothesisEngine — short-circuit so we
    # don't spend ~1 min loading the KG into a NetworkX graph for nothing.
    if args.command in ("kge-train", "plausibility"):
        from .kge.cli import cmd_kge_train, cmd_plausibility
        if args.command == "kge-train":
            cmd_kge_train(
                kg_path=args.kg, output=args.output, report=args.report,
                dim=args.dim, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, negatives_per_pos=args.negatives_per_pos,
                weight_decay=args.weight_decay, eval_every=args.eval_every,
                early_stop_patience=args.early_stop_patience,
                min_confidence=args.min_confidence, seed=args.seed, device=args.device,
            )
        else:
            cmd_plausibility(
                input_path=args.input, kge_checkpoint=args.kge,
                output=args.output, novelty_cache=args.novelty_cache,
                skip_existing=not args.no_skip_existing,
                no_pubmed=args.no_pubmed, top=args.top, device=args.device,
                kg_path=args.kg,
                enable_surprise=args.enable_surprise,
                surprise_alpha=args.surprise_alpha,
                evo_surprise_min=args.evo_surprise_min,
            )
        return

    if args.command == "im-brainstorm":
        graph_p = args.im_graph or args.graph or "neurooracle/data/full_snapshot_v2/knowledge_graph.json"
        cmd_im_brainstorm(graph_path=graph_p, output=args.output, n=args.n,
                          model=args.model, batch_size=args.batch_size, seed=args.seed)
        return

    if args.command == "gm-brainstorm":
        graph_p = args.gm_graph or args.graph or "neurooracle/data/full_snapshot_v2/knowledge_graph.json"
        cmd_gm_brainstorm(graph_path=graph_p, output=args.output, n=args.n,
                          model=args.model, batch_size=args.batch_size, seed=args.seed)
        return

    if args.command == "case-study":
        graph_p = args.graph or "neurooracle/data/full_snapshot_v2/knowledge_graph.json"
        kg_for_plaus = args.kg_for_plausibility or graph_p
        cmd_case_study(
            case_study_name=args.name,
            output_dir=args.output_dir,
            stages=args.stages,
            kge_path=args.kge,
            kg_path=kg_for_plaus,
            graph_path=graph_p,
            snapshot_2022_kg=args.snapshot_2022_kg,
            snapshot_2022_kge=args.snapshot_2022_kge,
            feedback_state=args.feedback_state,
            as_json=args.json,
        )
        return

    graph_path = Path(args.graph) if args.graph else Path("neurooracle/data/full_snapshot_v2/knowledge_graph.json")
    kg = load_graph(graph_path)
    engine = HypothesisEngine(kg)
    if args.feedback_state:
        engine.load_feedback_state(args.feedback_state)

    as_json = args.json

    if args.command == "batch":
        cmd_batch(engine, args.output, domain_pairs=args.domain_pairs,
                  max_hops=args.max_hops, max_paths=args.max_paths,
                  max_seeds=args.max_seeds, as_json=as_json,
                  legacy_domain_pairs=args.legacy_domain_pairs,
                  task_filter=args.tasks, chain_filter=args.chains,
                  target_per_task=args.target_per_task,
                  max_retries=args.max_retries,
                  retry_scale=args.retry_scale,
                  min_hops=args.min_hops,
                  metapath_min_domains=args.metapath_min_domains,
                  prefer_longer_paths=not args.no_prefer_longer_paths)
    elif args.command == "rank":
        cmd_rank(engine, args.input, top_n=args.top, as_json=as_json)
    elif args.command == "paths":
        cmd_paths(engine, args.source, args.target, args.max_hops, args.max_paths, as_json)
    elif args.command == "bridge":
        cmd_bridge(engine, args.concept, args.target_domain, args.max_hops, args.max_results, as_json)
    elif args.command == "contradictions":
        cmd_contradictions(engine, args.domain, args.max_results, as_json)
    elif args.command == "gaps":
        cmd_gaps(engine, args.domain_a, args.domain_b, args.max_results, as_json)
    elif args.command == "explore":
        cmd_explore(engine, args.concept, args.max_hops, as_json)
    elif args.command == "stats":
        cmd_stats(engine)
    elif args.command == "discover":
        cmd_discover(engine, args.concept, args.max_hops, args.max_results, as_json)
    elif args.command == "trending":
        cmd_trending(engine, args.since, args.direction, args.min_claims, args.max_results, as_json)
    elif args.command == "critic":
        cmd_critic(engine, args.input, args.top, args.output, args.max_rounds, args.threshold, args.max_workers, as_json)
    elif args.command == "novelty":
        cmd_novelty(engine, args.input, args.top, args.output, args.alpha, args.no_pubmed, args.no_semantic, as_json)
    elif args.command == "evolve":
        cmd_evolve(engine, args.input, args.output, args.population, args.generations,
                   args.mutation_rate, args.crossover_rate, args.tournament_size, args.elitism, as_json)
    elif args.command == "imaging-batch":
        cmd_imaging_batch(engine, args.dataset, args.output, args.max_paths, args.max_seeds,
                          args.max_hops, not args.no_connectivity, as_json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
