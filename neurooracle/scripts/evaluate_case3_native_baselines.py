"""Ground and hindcast BrainPilot/Biomni task-wise hypotheses.

Concept files are streamed with ijson because each frozen graph is close to a
gigabyte. Only candidate endpoint pairs are retained while scanning future
claims, so all methods and tasks share one pass per freeze window.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

import ijson
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio

from neurooracle.src.atoms import atoms_for_domain


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATION_ROOT = (
    ROOT / "neurooracle/data/experiments/case3/native_baselines_gpt56sol_high_20260722"
)
DEFAULT_SNAPSHOT_ROOT = (
    ROOT / "neurooracle/data/experiments/case3/snapshots_full_v2_5year_2016_2020_20260722"
)
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
WINDOWS = (
    (2016, 2017, 2021),
    (2017, 2018, 2022),
    (2018, 2019, 2023),
    (2019, 2020, 2024),
    (2020, 2021, 2025),
)
TOP_KS = (10, 20, 50, 100, 200, 500, 1000)
STOPWORDS = {
    "and", "with", "from", "into", "over", "under", "after", "before", "between",
    "brain", "disease", "disorder", "response", "outcome", "score", "level", "change",
    "reduced", "increased", "decreased", "higher", "lower", "regional", "functional",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, default=DEFAULT_GENERATION_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--future-claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(TOP_KS))
    parser.add_argument("--freeze-years", type=int, nargs="*", default=None)
    parser.add_argument("--min-fuzzy-score", type=float, default=0.72)
    return parser.parse_args()


def normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def informative_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token for token in normalise(value).split()
        if len(token) >= 4 and token not in STOPWORDS
    )


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def node_atoms(node: dict[str, Any]) -> set[str]:
    metadata = node.get("metadata") or {}
    explicit = metadata.get("atom_types") or node.get("atom_types") or []
    atoms = {str(value) for value in explicit if str(value)}
    for domain in node.get("domain_tags") or []:
        atoms.update(atom.value for atom in atoms_for_domain(str(domain)))
    return atoms


class ConceptMatcher:
    def __init__(self, graph_path: Path, queries: dict[str, set[str]]) -> None:
        self.graph_path = graph_path
        self.query_norms = {
            atom: {normalise(query) for query in values if normalise(query)}
            for atom, values in queries.items()
        }
        self.query_tokens = {
            atom: set().union(*(informative_tokens(query) for query in values)) if values else set()
            for atom, values in queries.items()
        }
        self.concept_ids: set[str] = set()
        self.names: dict[str, str] = {}
        self.exact: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.token_candidates: dict[tuple[str, str], list[tuple[str, str, frozenset[str]]]] = defaultdict(list)
        self._stream_concepts()

    def _stream_concepts(self) -> None:
        with self.graph_path.open("rb") as handle:
            for concept_id, node in ijson.kvitems(handle, "concepts"):
                concept_id = str(concept_id)
                self.concept_ids.add(concept_id)
                preferred = str(node.get("preferred_name") or concept_id)
                self.names[concept_id] = preferred
                preferred_norm = normalise(preferred)
                preferred_tokens = informative_tokens(preferred)
                for atom in node_atoms(node):
                    if atom not in self.query_norms:
                        continue
                    if preferred_norm in self.query_norms[atom]:
                        self.exact[(atom, preferred_norm)].append(concept_id)
                    for token in preferred_tokens & self.query_tokens[atom]:
                        self.token_candidates[(atom, token)].append(
                            (concept_id, preferred_norm, preferred_tokens)
                        )
                    for alias in (node.get("aliases") or [])[:40]:
                        alias_norm = normalise(alias)
                        if alias_norm in self.query_norms[atom]:
                            self.exact[(atom, alias_norm)].append(concept_id)

    def match(self, atom: str, query: str, threshold: float) -> tuple[str, str, float, str]:
        query_norm = normalise(query)
        exact_ids = sorted(set(self.exact.get((atom, query_norm), [])))
        if exact_ids:
            concept_id = exact_ids[0]
            return concept_id, self.names[concept_id], 1.0, "exact"

        query_tokens = informative_tokens(query_norm)
        candidates: dict[str, tuple[str, frozenset[str]]] = {}
        for token in query_tokens:
            for concept_id, candidate_norm, candidate_tokens in self.token_candidates.get((atom, token), []):
                candidates.setdefault(concept_id, (candidate_norm, candidate_tokens))
        best: tuple[float, str] | None = None
        for concept_id, (candidate_norm, candidate_tokens) in candidates.items():
            union = query_tokens | candidate_tokens
            token_score = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            sequence_score = ratio(query_norm, candidate_norm) / 100.0
            containment = 1.0 if query_norm in candidate_norm or candidate_norm in query_norm else 0.0
            score = max(sequence_score, 0.7 * token_score + 0.3 * containment)
            if best is None or score > best[0] or (score == best[0] and concept_id < best[1]):
                best = (score, concept_id)
        if best is None or best[0] < threshold:
            return "", "", best[0] if best else 0.0, "unmapped"
        return best[1], self.names[best[1]], best[0], "fuzzy"


def historical_pairs(graph_path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with graph_path.open("rb") as handle:
        for edge in ijson.items(handle, "edges.item"):
            source = str(edge.get("source_id") or edge.get("source") or "")
            target = str(edge.get("target_id") or edge.get("target") or "")
            relation = str(edge.get("relation_type") or "")
            if not source or not target or source == target or relation in {"is_a", "part_of", "about"}:
                continue
            pairs.add(tuple(sorted((source, target))))
    return pairs


def claim_year(claim: dict[str, Any]) -> int | None:
    value = (claim.get("source_paper") or {}).get("year") or claim.get("year")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def scan_candidate_support(
    claims_path: Path,
    candidate_pairs: set[tuple[str, str]],
    historical: set[tuple[str, str]],
    concept_ids: set[str],
    start_year: int,
    end_year: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    support: dict[tuple[str, str], dict[str, Any]] = {}
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            claim = json.loads(line)
            year = claim_year(claim)
            if year is None or year < start_year or year > end_year:
                continue
            source = str(claim.get("subject_id") or "")
            target = str(claim.get("object_id") or "")
            if source not in concept_ids or target not in concept_ids or source == target:
                continue
            pair = tuple(sorted((source, target)))
            if pair not in candidate_pairs or pair in historical:
                continue
            previous = support.get(pair)
            if previous is not None and int(previous["year"]) <= year:
                continue
            paper = claim.get("source_paper") or {}
            support[pair] = {
                "year": year,
                "claim_id": claim.get("id"),
                "pmid": paper.get("pmid"),
                "doi": paper.get("doi"),
                "title": paper.get("title"),
                "journal": paper.get("journal"),
                "predicate": claim.get("predicate"),
                "raw_text": claim.get("raw_text"),
            }
    return support


def map_window(
    hypotheses: pd.DataFrame,
    graph_path: Path,
    claims_path: Path,
    freeze_year: int,
    future_start: int,
    future_end: int,
    threshold: float,
) -> pd.DataFrame:
    queries: dict[str, set[str]] = defaultdict(set)
    for row in hypotheses.to_dict(orient="records"):
        if not bool_value(row.get("native_schema_valid")):
            continue
        queries[str(row["source_atom"])].add(str(row["source_entity"]))
        queries[str(row["target_atom"])].add(str(row["target_entity"]))
    matcher = ConceptMatcher(graph_path, queries)
    history = historical_pairs(graph_path)
    mapped_rows: list[dict[str, Any]] = []
    candidate_pairs: set[tuple[str, str]] = set()
    for row in hypotheses.to_dict(orient="records"):
        out = dict(row)
        out.update({"freeze_year": freeze_year, "future_start": future_start, "future_end": future_end})
        if not bool_value(row.get("native_schema_valid")):
            out.update(
                {
                    "mapping_status": "native_validation_failed",
                    "source_id": "",
                    "source_name": "",
                    "source_mapping_score": 0.0,
                    "target_id": "",
                    "target_name": "",
                    "target_mapping_score": 0.0,
                    "candidate_pair": "",
                }
            )
            mapped_rows.append(out)
            continue
        source_id, source_name, source_score, source_status = matcher.match(
            str(row["source_atom"]), str(row["source_entity"]), threshold
        )
        target_id, target_name, target_score, target_status = matcher.match(
            str(row["target_atom"]), str(row["target_entity"]), threshold
        )
        pair = tuple(sorted((source_id, target_id))) if source_id and target_id and source_id != target_id else None
        status = (
            f"mapped:{source_status}+{target_status}"
            if pair is not None
            else "unmapped_endpoint"
        )
        if pair is not None:
            candidate_pairs.add(pair)
        out.update(
            {
                "mapping_status": status,
                "source_id": source_id,
                "source_name": source_name,
                "source_mapping_score": source_score,
                "target_id": target_id,
                "target_name": target_name,
                "target_mapping_score": target_score,
                "candidate_pair": "|".join(pair) if pair else "",
                "already_historical": bool(pair in history) if pair else False,
            }
        )
        mapped_rows.append(out)

    support = scan_candidate_support(
        claims_path,
        candidate_pairs,
        history,
        matcher.concept_ids,
        future_start,
        future_end,
    )
    for row in mapped_rows:
        raw_pair = str(row.get("candidate_pair") or "")
        pair = tuple(raw_pair.split("|", 1)) if "|" in raw_pair else None
        evidence = support.get(pair) if pair else None
        row["future_supported"] = evidence is not None
        row["first_future_year"] = evidence.get("year") if evidence else None
        row["lead_time"] = int(evidence["year"]) - freeze_year if evidence else None
        for key in ("claim_id", "pmid", "doi", "title", "journal", "predicate", "raw_text"):
            row[f"support_{key}"] = evidence.get(key) if evidence else None
    return pd.DataFrame(mapped_rows)


def metrics_by_k(scored: pd.DataFrame, top_ks: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = scored.groupby(["method", "task", "seed", "freeze_year", "future_start", "future_end"], sort=False)
    for keys, sub in groups:
        method, task, seed, freeze_year, future_start, future_end = keys
        sub = sub.sort_values("generated_rank", kind="mergesort")
        for k in top_ks:
            head = sub.head(k)
            hits = int(head["future_supported"].fillna(False).astype(bool).sum())
            mapped = int(head["mapping_status"].astype(str).str.startswith("mapped:").sum())
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "seed": int(seed),
                    "freeze_year": int(freeze_year),
                    "future_start": int(future_start),
                    "future_end": int(future_end),
                    "k": int(k),
                    "n": len(head),
                    "schema_valid": int(head["native_schema_valid"].map(bool_value).sum()),
                    "mapped": mapped,
                    "mapping_rate": mapped / len(head) if len(head) else 0.0,
                    "future_supported_hits": hits,
                    "future_supported_hit_rate": hits / len(head) if len(head) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, task, k), sub in metrics.groupby(["method", "task", "k"], sort=False):
        for column in ("mapping_rate", "future_supported_hits", "future_supported_hit_rate"):
            values = pd.to_numeric(sub[column], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            if not any(row.get("method") == method and row.get("task") == task and row.get("k") == k for row in rows):
                rows.append({"method": method, "task": task, "k": int(k), "n_runs": len(sub)})
            row = rows[-1]
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_lo"] = float(np.quantile(values, 0.025))
            row[f"{column}_hi"] = float(np.quantile(values, 0.975))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source = args.generation_root / "native_hypotheses_all.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir = args.output_dir or (args.generation_root / "hindcasting")
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = pd.read_csv(source, keep_default_na=False)
    scored_parts: list[pd.DataFrame] = []
    windows = tuple(
        window for window in WINDOWS
        if args.freeze_years is None or window[0] in set(args.freeze_years)
    )
    if not windows:
        raise ValueError("no requested freeze years match the registered windows")
    for freeze_year, future_start, future_end in windows:
        graph_path = args.snapshot_root / f"kg_{freeze_year}" / "knowledge_graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        print(f"mapping/scoring freeze={freeze_year} future={future_start}-{future_end}", flush=True)
        scored_parts.append(
            map_window(
                hypotheses,
                graph_path,
                args.future_claims,
                freeze_year,
                future_start,
                future_end,
                args.min_fuzzy_score,
            )
        )
    scored = pd.concat(scored_parts, ignore_index=True)
    scored.to_csv(output_dir / "native_baseline_scored_hypotheses.csv", index=False)
    metrics = metrics_by_k(scored, args.top_k)
    metrics.to_csv(output_dir / "native_baseline_metrics_by_k.csv", index=False)
    aggregate_metrics(metrics).to_csv(output_dir / "native_baseline_metrics_summary.csv", index=False)
    recovered = scored[scored["future_supported"].fillna(False).astype(bool)].copy()
    recovered.to_csv(output_dir / "native_baseline_recovered_examples.csv", index=False)
    manifest = {
        "generation_root": str(args.generation_root),
        "snapshot_root": str(args.snapshot_root),
        "future_claims": str(args.future_claims),
        "windows": [list(window) for window in windows],
        "top_k": args.top_k,
        "min_fuzzy_score": args.min_fuzzy_score,
        "n_generated_rows": len(hypotheses),
        "n_scored_rows": len(scored),
        "n_future_supported_rows": len(recovered),
        "failure_policy": "Invalid and unmapped ranks remain in the denominator with zero credit.",
    }
    (output_dir / "native_baseline_hindcasting_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
