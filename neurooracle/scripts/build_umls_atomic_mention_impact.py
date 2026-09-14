#!/usr/bin/env python3
"""Build a read-only UMLS atomic-mention alignment impact report.

The script streams only the ``concepts`` object of the formal KG, preserves
every ``CLM_CONCEPT`` identifier, derives conservative atomic mention
projections, and aligns those projections to UMLS by normalized-exact terms
or exact source identifiers.  It never rewrites the formal graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping

import ijson

from neurooracle.src.umls_mention_mapping import (
    UMLS_MENTION_PREFIX,
    atomize_mention_candidates,
    biomedical_context,
    is_lexically_eligible,
    normalize_term,
    preferred_domain,
    select_atomic_units,
    semantic_compatibility,
)


SCHEMA_VERSION = "neurooracle.umls_atomic_mention_impact.v1"
MAPPING_SOURCE = "UMLS_2026AA_normalized_exact_atomic_mention_alignment"
MAX_CUIS_PER_TERM = 64
MAX_EMITTED_MAPPINGS_PER_ATOM = 8
CUI_RE = re.compile(r"^C\d{7}$")

COL_CUI = 0
COL_LAT = 1
COL_TS = 2
COL_ISPREF = 6
COL_SAB = 11
COL_TTY = 12
COL_CODE = 13
COL_STR = 14
COL_SUPPRESS = 16

SAB_PRIORITY = {
    "MSH": 0,
    "SNOMEDCT_US": 1,
    "NCI": 2,
    "FMA": 3,
    "HPO": 4,
    "HGNC": 5,
    "RXNORM": 6,
    "ATC": 7,
    "MED-RT": 8,
    "LNC": 9,
}

EXTERNAL_ID_TO_SAB = {
    "MeSH_UI": "MSH",
    "MSH": "MSH",
    "SNOMEDCT_US": "SNOMEDCT_US",
    "NCI": "NCI",
    "FMA": "FMA",
    "HPO": "HPO",
    "HGNC": "HGNC",
    "RXNORM": "RXNORM",
    "ATC": "ATC",
    "OMIM": "OMIM",
}

METHOD_CONFIDENCE = {
    "existing_umls_cui": 1.0,
    "external_source_code_exact": 0.995,
    "preferred_name_normalized_exact": 0.98,
    "atomic_surface_normalized_exact": 0.98,
    "alias_normalized_exact": 0.96,
    "parenthetical_base_exact": 0.92,
    "parenthetical_inner_exact": 0.90,
    "leading_modifier_stripped_exact": 0.90,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "last_write_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "mtime_ns": stat.st_mtime_ns,
    }


def metadata_unchanged(before: Mapping, after: Mapping) -> bool:
    return (
        before["path"] == after["path"]
        and before["bytes"] == after["bytes"]
        and before["mtime_ns"] == after["mtime_ns"]
    )


def validate_cached_file_metadata(path: Path, expected: Mapping, label: str) -> dict:
    observed = path_metadata(path)
    if observed["bytes"] != int(expected.get("bytes", -1)):
        raise RuntimeError(f"cached byte-size mismatch for {label}")
    expected_mtime = datetime.fromisoformat(
        str(expected["last_write_utc"]).replace("Z", "+00:00")
    ).timestamp()
    if abs(path.stat().st_mtime - expected_mtime) > 0.001:
        raise RuntimeError(f"cached mtime mismatch for {label}")
    return observed


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AtomicJsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.temp = path.with_name(path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: BinaryIO = self.temp.open("wb")
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def write(self, payload: Mapping) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.handle.write(encoded)
        self.digest.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)

    def close(self) -> dict:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            os.replace(self.temp, self.path)
        return {
            "path": str(self.path.resolve()),
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def update_state(state_path: Path, phase: str, **details: object) -> None:
    atomic_write_json(
        state_path,
        {
            "schema_version": "neurooracle.umls_atomic_mention_run_state.v1",
            "status": "RUNNING",
            "phase": phase,
            "updated_at_utc": utc_now(),
            **details,
        },
    )


def validate_release_manifest(manifest_path: Path) -> tuple[dict, dict[str, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release") != "2026AA":
        raise RuntimeError("release manifest is not UMLS 2026AA")
    archive = manifest.get("archive") or {}
    if not archive.get("md5_matches"):
        raise RuntimeError("release manifest does not prove the official MD5 match")

    paths: dict[str, Path] = {}
    for name in ("MRCONSO.RRF", "MRSTY.RRF", "MRSAB.RRF", "MRFILES.RRF"):
        expected = (manifest.get("files") or {}).get(name) or {}
        path = Path(expected.get("path") or "").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_cached_file_metadata(path, expected, name)
        paths[name] = path
    return manifest, paths


def validate_formal_graph(graph_path: Path, current_state_path: Path) -> tuple[dict, dict]:
    current = json.loads(current_state_path.read_text(encoding="utf-8"))
    canonical = current["canonical_files"]["knowledge_graph"]
    resolved = graph_path.resolve()
    if resolved != Path(canonical["path"]).resolve():
        raise RuntimeError("graph path is not the CURRENT_STATE canonical graph")
    observed = validate_cached_file_metadata(resolved, canonical, "formal graph")
    return current, observed


def values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if item is not None:
                yield str(item)
    elif value is not None:
        yield str(value)


def existing_direct_cuis(node: Mapping) -> list[str]:
    found: set[str] = set()
    external_ids = node.get("external_ids") or {}
    for key in ("UMLS_CUI", "umls_cui"):
        for value in values(external_ids.get(key)):
            candidate = value.removeprefix("CUI:").strip()
            if CUI_RE.fullmatch(candidate):
                found.add(candidate)
    metadata = node.get("metadata") or {}
    for key in ("UMLS_CUI", "umls_cui"):
        for value in values(metadata.get(key)):
            candidate = value.removeprefix("CUI:").strip()
            if CUI_RE.fullmatch(candidate):
                found.add(candidate)
    return sorted(found)


def external_code_keys(node: Mapping) -> list[dict]:
    result: list[dict] = []
    external_ids = node.get("external_ids") or {}
    for label, raw_value in external_ids.items():
        sab = EXTERNAL_ID_TO_SAB.get(str(label))
        if not sab or str(label).casefold() in {"umls_cui"}:
            continue
        for item in values(raw_value):
            code = item.strip()
            if not code:
                continue
            if sab == "HGNC" and not code.startswith("HGNC:"):
                code = f"HGNC:{code}"
            result.append({"sab": sab, "code": code, "source_label": str(label)})
    deduplicated = {(item["sab"], item["code"]): item for item in result}
    return [deduplicated[key] for key in sorted(deduplicated)]


def collect_source_mentions(
    graph_path: Path,
    source_path: Path,
    state_path: Path,
) -> tuple[dict, set[str], set[tuple[str, str]], set[str]]:
    writer = AtomicJsonlWriter(source_path)
    target_terms: set[str] = set()
    target_codes: set[tuple[str, str]] = set()
    existing_cui_nodes: set[str] = set()
    counts = Counter()
    source_id_digest = hashlib.sha256()

    try:
        with graph_path.open("rb") as handle:
            for node_id, node in ijson.kvitems(handle, "concepts", use_float=True):
                counts["concept_nodes_scanned"] += 1
                if node_id.startswith("CUI:"):
                    suffix = node_id.split(":", 1)[1]
                    if CUI_RE.fullmatch(suffix):
                        existing_cui_nodes.add(suffix)
                if not node_id.startswith(UMLS_MENTION_PREFIX):
                    continue

                counts["source_mentions"] += 1
                source_id_digest.update(node_id.encode("utf-8"))
                source_id_digest.update(b"\n")

                metadata = node.get("metadata") or {}
                domain_tags = [str(value) for value in node.get("domain_tags") or []]
                atom_types = [str(value) for value in metadata.get("atom_types") or []]
                context_eligible, context_domains, context_reason = biomedical_context(
                    domain_tags, atom_types
                )
                preferred_name = str(node.get("preferred_name") or "")
                aliases = [str(value) for value in node.get("aliases") or [] if value]
                candidates = atomize_mention_candidates(
                    node_id,
                    preferred_name,
                    aliases,
                )
                direct_cuis = existing_direct_cuis(node)
                code_keys = external_code_keys(node)

                if context_eligible:
                    counts["biomedical_source_mentions"] += 1
                    units = [candidates["full"], *(candidates.get("split") or [])]
                    for unit in units:
                        for variant in unit.get("variants") or []:
                            normalized = str(variant.get("normalized") or "")
                            if normalized:
                                target_terms.add(normalized)
                    for item in code_keys:
                        target_codes.add((item["sab"], item["code"]))
                else:
                    counts[f"excluded_context:{context_reason}"] += 1

                if candidates.get("has_syntactic_composite"):
                    counts["syntactic_composite_source_mentions"] += 1
                if direct_cuis:
                    counts["source_mentions_with_existing_direct_cui"] += 1
                if node.get("semantic_types"):
                    counts["source_mentions_with_semantic_types"] += 1

                writer.write(
                    {
                        "source_mention_id": node_id,
                        "preferred_name": preferred_name,
                        "domain_tags": domain_tags,
                        "atom_types": atom_types,
                        "source_semantic_types": list(node.get("semantic_types") or []),
                        "context_eligible": context_eligible,
                        "context_domains": list(context_domains),
                        "context_reason": context_reason,
                        "existing_direct_cuis": direct_cuis,
                        "external_code_keys": code_keys,
                        **candidates,
                    }
                )

                if counts["source_mentions"] % 100_000 == 0:
                    print(
                        f"collect: {counts['source_mentions']:,} CLM_CONCEPT, "
                        f"{len(target_terms):,} target terms",
                        flush=True,
                    )
                    update_state(
                        state_path,
                        "COLLECT_SOURCE_MENTIONS",
                        source_mentions=counts["source_mentions"],
                        target_terms=len(target_terms),
                    )
    except BaseException:
        writer.abort()
        raise

    source_artifact = writer.close()
    summary = dict(counts)
    summary["source_mention_id_ordered_sha256"] = source_id_digest.hexdigest()
    summary["target_terms"] = len(target_terms)
    summary["target_external_codes"] = len(target_codes)
    summary["existing_cui_nodes"] = len(existing_cui_nodes)
    summary["artifact"] = source_artifact
    return summary, target_terms, target_codes, existing_cui_nodes


def sab_rank(sab: str) -> int:
    return SAB_PRIORITY.get(sab, 100)


def candidate_row_rank(row: Mapping) -> tuple:
    return (
        0 if row.get("is_preferred") else 1,
        sab_rank(str(row.get("sab") or "")),
        str(row.get("tty") or ""),
        str(row.get("code") or ""),
        str(row.get("cui") or ""),
    )


def retain_candidate(
    bucket: dict[str, dict],
    candidate: dict,
    overflow_keys: set[str],
    overflow_key: str,
) -> None:
    cui = candidate["cui"]
    current = bucket.get(cui)
    if current is not None:
        if candidate_row_rank(candidate) < candidate_row_rank(current):
            bucket[cui] = candidate
        return
    if len(bucket) < MAX_CUIS_PER_TERM:
        bucket[cui] = candidate
        return
    overflow_keys.add(overflow_key)
    worst_cui = max(bucket, key=lambda item: candidate_row_rank(bucket[item]))
    if candidate_row_rank(candidate) < candidate_row_rank(bucket[worst_cui]):
        del bucket[worst_cui]
        bucket[cui] = candidate


def scan_mrconso(
    path: Path,
    target_terms: set[str],
    target_codes: set[tuple[str, str]],
    state_path: Path,
) -> tuple[
    dict[str, dict[str, dict]],
    dict[tuple[str, str], dict[str, dict]],
    set[str],
    dict,
]:
    term_matches: dict[str, dict[str, dict]] = {}
    code_matches: dict[tuple[str, str], dict[str, dict]] = {}
    overflow_terms: set[str] = set()
    overflow_codes: set[str] = set()
    stats = Counter()

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            stats["rows_scanned"] += 1
            parts = line.rstrip("\r\n").split("|")
            if stats["rows_scanned"] % 2_000_000 == 0:
                print(
                    f"MRCONSO: {stats['rows_scanned']:,} rows, "
                    f"{len(term_matches):,} matched target terms",
                    flush=True,
                )
                update_state(
                    state_path,
                    "SCAN_MRCONSO",
                    rows_scanned=stats["rows_scanned"],
                    matched_target_terms=len(term_matches),
                )
            if len(parts) <= COL_SUPPRESS:
                stats["malformed_rows"] += 1
                continue
            if parts[COL_LAT] != "ENG":
                continue
            if parts[COL_SUPPRESS] in {"O", "Y"}:
                continue
            sab = parts[COL_SAB]
            code = parts[COL_CODE]
            term = normalize_term(parts[COL_STR])
            term_hit = term in target_terms
            code_key = (sab, code)
            code_hit = code_key in target_codes
            if not term_hit and not code_hit:
                continue

            row = {
                "cui": parts[COL_CUI],
                "sab": sab,
                "tty": parts[COL_TTY],
                "code": code,
                "matched_term": parts[COL_STR].strip(),
                "is_preferred": parts[COL_TS] == "P" or parts[COL_ISPREF] == "Y",
            }
            if term_hit:
                bucket = term_matches.setdefault(term, {})
                retain_candidate(bucket, row, overflow_terms, term)
                stats["term_match_rows"] += 1
            if code_hit:
                bucket = code_matches.setdefault(code_key, {})
                retain_candidate(bucket, row, overflow_codes, f"{sab}|{code}")
                stats["code_match_rows"] += 1

    stats["matched_target_terms"] = len(term_matches)
    stats["matched_external_codes"] = len(code_matches)
    stats["overflow_target_terms"] = len(overflow_terms)
    stats["overflow_external_codes"] = len(overflow_codes)
    stats["retained_term_cui_candidates"] = sum(len(value) for value in term_matches.values())
    stats["retained_code_cui_candidates"] = sum(len(value) for value in code_matches.values())
    stats["overflow_term_values"] = sorted(overflow_terms)[:100]
    return term_matches, code_matches, overflow_terms, dict(stats)


def scan_mrsty(path: Path, matched_cuis: set[str], state_path: Path) -> tuple[dict, dict]:
    semantics: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    stats = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            stats["rows_scanned"] += 1
            parts = line.rstrip("\r\n").split("|")
            if stats["rows_scanned"] % 1_000_000 == 0:
                print(
                    f"MRSTY: {stats['rows_scanned']:,} rows, "
                    f"{len(semantics):,} matched CUIs",
                    flush=True,
                )
                update_state(
                    state_path,
                    "SCAN_MRSTY",
                    rows_scanned=stats["rows_scanned"],
                    cuis_with_semantics=len(semantics),
                )
            if len(parts) < 4:
                stats["malformed_rows"] += 1
                continue
            cui = parts[0]
            if cui not in matched_cuis:
                continue
            key = (parts[1], parts[3])
            if key not in seen[cui]:
                seen[cui].add(key)
                semantics[cui].append({"tui": parts[1], "sty": parts[3], "stn": parts[2]})
    stats["requested_cuis"] = len(matched_cuis)
    stats["cuis_with_semantics"] = len(semantics)
    stats["cuis_without_semantics"] = len(matched_cuis - set(semantics))
    return dict(semantics), dict(stats)


def method_for_variant(unit: Mapping, variant: Mapping) -> str:
    rule = str(variant.get("rule") or "")
    if rule == "normalized_exact":
        if str(variant.get("source_field")) == "preferred_name":
            if unit.get("atomization_rule") == "top_level_composite_split":
                return "atomic_surface_normalized_exact"
            return "preferred_name_normalized_exact"
    if rule == "alias_normalized_exact":
        return "alias_normalized_exact"
    return rule


def combined_candidate_rank(candidate: Mapping) -> tuple:
    compatibility_rank = {"compatible": 0, "unknown": 1, "incompatible": 2}
    return (
        compatibility_rank.get(str(candidate.get("semantic_compatibility")), 3),
        int(candidate.get("method_priority", 999)),
        sab_rank(str(candidate.get("sab") or "")),
        0 if candidate.get("is_preferred") else 1,
        str(candidate.get("cui") or ""),
    )


def resolve_unit(
    record: Mapping,
    unit: Mapping,
    term_matches: Mapping[str, Mapping[str, Mapping]],
    code_matches: Mapping[tuple[str, str], Mapping[str, Mapping]],
    semantics: Mapping[str, list[dict]],
    overflow_terms: set[str],
    *,
    include_source_identifiers: bool,
) -> dict:
    candidates_by_cui: dict[str, dict] = {}

    def consider(raw: Mapping, method: str, method_priority: int, variant: Mapping | None) -> None:
        cui = str(raw.get("cui") or "")
        if not CUI_RE.fullmatch(cui):
            return
        semantic_rows = semantics.get(cui, [])
        compatibility, compatible_stys, basis = semantic_compatibility(
            record.get("domain_tags") or [],
            record.get("atom_types") or [],
            semantic_rows,
        )
        candidate = {
            "cui": cui,
            "sab": str(raw.get("sab") or ""),
            "tty": str(raw.get("tty") or ""),
            "code": str(raw.get("code") or ""),
            "matched_term": str(raw.get("matched_term") or unit.get("text") or ""),
            "is_preferred": bool(raw.get("is_preferred")),
            "method": method,
            "method_priority": method_priority,
            "lookup_variant": dict(variant) if variant else None,
            "semantic_compatibility": compatibility,
            "semantic_compatibility_basis": basis,
            "compatible_semantic_type_names": list(compatible_stys),
            "semantic_types": semantic_rows,
        }
        current = candidates_by_cui.get(cui)
        if current is None or combined_candidate_rank(candidate) < combined_candidate_rank(current):
            candidates_by_cui[cui] = candidate

    for variant in unit.get("variants") or []:
        normalized = str(variant.get("normalized") or "")
        for row in (term_matches.get(normalized) or {}).values():
            consider(
                row,
                method_for_variant(unit, variant),
                int(variant.get("priority", 999)),
                variant,
            )

    if include_source_identifiers:
        for cui in record.get("existing_direct_cuis") or []:
            consider(
                {
                    "cui": cui,
                    "sab": "UMLS",
                    "tty": "",
                    "code": cui,
                    "matched_term": unit.get("text") or "",
                    "is_preferred": True,
                },
                "existing_umls_cui",
                0,
                None,
            )
        for key in record.get("external_code_keys") or []:
            code_key = (str(key.get("sab") or ""), str(key.get("code") or ""))
            for row in (code_matches.get(code_key) or {}).values():
                consider(row, "external_source_code_exact", 1, key)

    all_candidates = sorted(candidates_by_cui.values(), key=combined_candidate_rank)
    compatible = [
        candidate
        for candidate in all_candidates
        if candidate["semantic_compatibility"] == "compatible"
    ]
    if not compatible:
        return {
            "mappings": [],
            "lexical_candidate_count": len(all_candidates),
            "semantic_rejected_count": sum(
                candidate["semantic_compatibility"] == "incompatible"
                for candidate in all_candidates
            ),
            "semantic_unknown_count": sum(
                candidate["semantic_compatibility"] == "unknown"
                for candidate in all_candidates
            ),
        }

    best_method_priority = min(int(candidate["method_priority"]) for candidate in compatible)
    selected = [
        candidate
        for candidate in compatible
        if int(candidate["method_priority"]) == best_method_priority
    ][:MAX_EMITTED_MAPPINGS_PER_ATOM]
    ambiguous = len(selected) > 1
    overflow = any(
        str((candidate.get("lookup_variant") or {}).get("normalized") or "") in overflow_terms
        for candidate in selected
    )

    for candidate in selected:
        generic_basis = candidate["semantic_compatibility_basis"] == "generic_claim_concept"
        transformed = candidate["method"] in {
            "parenthetical_base_exact",
            "parenthetical_inner_exact",
            "leading_modifier_stripped_exact",
        }
        needs_review = ambiguous or overflow or generic_basis or transformed
        confidence = METHOD_CONFIDENCE.get(candidate["method"], 0.90)
        if ambiguous or overflow:
            confidence -= 0.08
        if generic_basis:
            confidence -= 0.03
        candidate["confidence"] = round(max(0.0, confidence), 3)
        candidate["review_status"] = "needs_review" if needs_review else "auto_accepted_exact"
        candidate["ambiguous_best_cui_count"] = len(selected)
        candidate["candidate_overflow"] = overflow

    return {
        "mappings": selected,
        "lexical_candidate_count": len(all_candidates),
        "semantic_rejected_count": sum(
            candidate["semantic_compatibility"] == "incompatible"
            for candidate in all_candidates
        ),
        "semantic_unknown_count": sum(
            candidate["semantic_compatibility"] == "unknown"
            for candidate in all_candidates
        ),
    }


def empty_bucket() -> Counter:
    return Counter(
        {
            "atomic_mentions": 0,
            "eligible": 0,
            "mapped": 0,
            "auto_accepted": 0,
            "needs_review": 0,
            "unmapped": 0,
            "maps_to_edges": 0,
        }
    )


def finalize_projection(
    source_path: Path,
    output_dir: Path,
    term_matches: Mapping[str, Mapping[str, Mapping]],
    code_matches: Mapping[tuple[str, str], Mapping[str, Mapping]],
    semantics: Mapping[str, list[dict]],
    existing_cui_nodes: set[str],
    overflow_terms: set[str],
    state_path: Path,
) -> tuple[dict, dict]:
    atom_writer = AtomicJsonlWriter(output_dir / "ATOMIC_MENTIONS.jsonl")
    edge_writer = AtomicJsonlWriter(output_dir / "UMLS_MAPS_TO.jsonl")
    target_writer = AtomicJsonlWriter(output_dir / "UMLS_TARGETS.jsonl")
    totals = empty_bucket()
    stats = Counter()
    by_domain: dict[str, Counter] = defaultdict(empty_bucket)
    by_rule: dict[str, Counter] = defaultdict(empty_bucket)
    excluded_reasons = Counter()
    target_rows: dict[str, dict] = {}

    try:
        with source_path.open("r", encoding="utf-8") as source_handle:
            for line in source_handle:
                record = json.loads(line)
                stats["source_mentions_processed"] += 1
                full_resolution = resolve_unit(
                    record,
                    record["full"],
                    term_matches,
                    code_matches,
                    semantics,
                    overflow_terms,
                    include_source_identifiers=True,
                )
                full_has_compatible = bool(full_resolution["mappings"])
                selected_units, selection_reason = select_atomic_units(
                    record,
                    full_has_compatible,
                )
                if len(selected_units) > 1:
                    stats["source_mentions_split"] += 1
                else:
                    stats["source_mentions_kept_full"] += 1

                report_domain = preferred_domain(
                    record.get("domain_tags") or [],
                    record.get("atom_types") or [],
                )

                for ordinal, unit in enumerate(selected_units):
                    totals["atomic_mentions"] += 1
                    by_domain[report_domain]["atomic_mentions"] += 1
                    by_rule[unit["atomization_rule"]]["atomic_mentions"] += 1
                    lexical_eligible, lexical_reason = is_lexically_eligible(unit["text"])
                    eligible = bool(record.get("context_eligible")) and lexical_eligible
                    if not eligible:
                        exclusion_reason = (
                            f"context:{record.get('context_reason')}"
                            if not record.get("context_eligible")
                            else f"lexical:{lexical_reason}"
                        )
                        excluded_reasons[exclusion_reason] += 1
                    else:
                        exclusion_reason = None

                    if unit["id"] == record["full"]["id"]:
                        resolution = full_resolution
                    else:
                        resolution = resolve_unit(
                            record,
                            unit,
                            term_matches,
                            code_matches,
                            semantics,
                            overflow_terms,
                            include_source_identifiers=False,
                        )

                    mappings = resolution["mappings"] if eligible else []
                    mapping_status = "excluded"
                    if eligible:
                        totals["eligible"] += 1
                        by_domain[report_domain]["eligible"] += 1
                        by_rule[unit["atomization_rule"]]["eligible"] += 1
                        if mappings:
                            mapping_status = (
                                "needs_review"
                                if any(item["review_status"] == "needs_review" for item in mappings)
                                else "auto_accepted_exact"
                            )
                            totals["mapped"] += 1
                            by_domain[report_domain]["mapped"] += 1
                            by_rule[unit["atomization_rule"]]["mapped"] += 1
                            if mapping_status == "needs_review":
                                totals["needs_review"] += 1
                                by_domain[report_domain]["needs_review"] += 1
                                by_rule[unit["atomization_rule"]]["needs_review"] += 1
                            else:
                                totals["auto_accepted"] += 1
                                by_domain[report_domain]["auto_accepted"] += 1
                                by_rule[unit["atomization_rule"]]["auto_accepted"] += 1
                        else:
                            mapping_status = "unmapped"
                            totals["unmapped"] += 1
                            by_domain[report_domain]["unmapped"] += 1
                            by_rule[unit["atomization_rule"]]["unmapped"] += 1

                    atom_semantic_types = sorted(
                        {
                            semantic["tui"]
                            for mapping in mappings
                            for semantic in mapping.get("semantic_types") or []
                            if semantic.get("tui")
                        }
                    )
                    atom_writer.write(
                        {
                            "id": unit["id"],
                            "preferred_name": unit["text"],
                            "semantic_types": atom_semantic_types,
                            "domain_tags": record.get("domain_tags") or [],
                            "source_vocab": "CLM_CONCEPT_atomic_projection",
                            "aliases": [],
                            "external_ids": {},
                            "spatial_mapping": None,
                            "metadata": {
                                "source_mention_id": record["source_mention_id"],
                                "source_mention_preserved": True,
                                "source_text": record["source_text"],
                                "evidence_span": {
                                    "field": "preferred_name",
                                    "start": unit["start"],
                                    "end": unit["end"],
                                    "text": unit["text"],
                                },
                                "atom_ordinal": ordinal,
                                "atomization_rule": unit["atomization_rule"],
                                "atomization_selection_reason": selection_reason,
                                "biomedical_eligible": eligible,
                                "eligibility_exclusion_reason": exclusion_reason,
                                "mapping_status": mapping_status,
                                "mapping_count": len(mappings),
                                "lexical_candidate_count": resolution["lexical_candidate_count"],
                                "semantic_rejected_count": resolution["semantic_rejected_count"],
                                "semantic_unknown_count": resolution["semantic_unknown_count"],
                                "umls_release": "2026AA",
                            },
                        }
                    )

                    for mapping in mappings:
                        totals["maps_to_edges"] += 1
                        by_domain[report_domain]["maps_to_edges"] += 1
                        by_rule[unit["atomization_rule"]]["maps_to_edges"] += 1
                        target_id = f"CUI:{mapping['cui']}"
                        edge_writer.write(
                            {
                                "source_id": unit["id"],
                                "target_id": target_id,
                                "relation_type": "maps_to",
                                "source": MAPPING_SOURCE,
                                "confidence": mapping["confidence"],
                                "metadata": {
                                    "umls_release": "2026AA",
                                    "method": mapping["method"],
                                    "matched_term": mapping["matched_term"],
                                    "matched_source": mapping["sab"],
                                    "matched_code": mapping["code"],
                                    "matched_tty": mapping["tty"],
                                    "matched_source_preferred": mapping["is_preferred"],
                                    "review_status": mapping["review_status"],
                                    "semantic_compatibility": mapping["semantic_compatibility"],
                                    "semantic_compatibility_basis": mapping[
                                        "semantic_compatibility_basis"
                                    ],
                                    "compatible_semantic_type_names": mapping[
                                        "compatible_semantic_type_names"
                                    ],
                                    "semantic_types": mapping["semantic_types"],
                                    "lookup_variant": mapping["lookup_variant"],
                                    "ambiguous_best_cui_count": mapping[
                                        "ambiguous_best_cui_count"
                                    ],
                                    "candidate_overflow": mapping["candidate_overflow"],
                                    "evidence_span": {
                                        "source_mention_id": record["source_mention_id"],
                                        "field": "preferred_name",
                                        "start": unit["start"],
                                        "end": unit["end"],
                                        "text": unit["text"],
                                    },
                                    "original_clm_concept_preserved": True,
                                },
                            }
                        )

                        target = target_rows.setdefault(
                            mapping["cui"],
                            {
                                "id": target_id,
                                "existing_in_formal_graph": mapping["cui"] in existing_cui_nodes,
                                "semantic_types": mapping["semantic_types"],
                                "external_ids": {"UMLS_CUI": mapping["cui"]},
                                "umls_release": "2026AA",
                                "example_matched_term": mapping["matched_term"],
                                "example_matched_source": mapping["sab"],
                                "incoming_maps_to": 0,
                            },
                        )
                        target["incoming_maps_to"] += 1

                if stats["source_mentions_processed"] % 100_000 == 0:
                    print(
                        f"finalize: {stats['source_mentions_processed']:,} source mentions, "
                        f"{totals['eligible']:,} eligible atoms, {totals['mapped']:,} mapped",
                        flush=True,
                    )
                    update_state(
                        state_path,
                        "FINALIZE_PROJECTION",
                        source_mentions=stats["source_mentions_processed"],
                        eligible_atomic_mentions=totals["eligible"],
                        mapped_atomic_mentions=totals["mapped"],
                    )

        for cui in sorted(target_rows):
            target_writer.write(target_rows[cui])
    except BaseException:
        atom_writer.abort()
        edge_writer.abort()
        target_writer.abort()
        raise

    artifacts = {
        "atomic_mentions": atom_writer.close(),
        "maps_to_edges": edge_writer.close(),
        "umls_targets": target_writer.close(),
    }
    stats.update(
        {
            "unique_cui_targets": len(target_rows),
            "existing_cui_targets": sum(
                bool(row["existing_in_formal_graph"]) for row in target_rows.values()
            ),
            "new_cui_targets_required": sum(
                not bool(row["existing_in_formal_graph"]) for row in target_rows.values()
            ),
        }
    )
    return {
        "totals": dict(totals),
        "by_primary_domain": {key: dict(value) for key, value in sorted(by_domain.items())},
        "by_atomization_rule": {key: dict(value) for key, value in sorted(by_rule.items())},
        "excluded_atomic_mentions": dict(excluded_reasons),
        "projection": dict(stats),
    }, artifacts


def add_coverage(rows: Mapping[str, Mapping]) -> dict:
    result: dict[str, dict] = {}
    for key, raw in rows.items():
        item = dict(raw)
        denominator = int(item.get("eligible", 0))
        item["coverage"] = item.get("mapped", 0) / denominator if denominator else None
        result[key] = item
    return result


def render_markdown(report: Mapping) -> str:
    coverage = report["coverage"]
    impact = report["read_only_impact"]
    lines = [
        "# UMLS 2026AA atomic biomedical mention impact",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Result",
        "",
        f"- Eligible atomic biomedical mentions: **{coverage['eligible_atomic_biomedical_mentions']:,}**",
        f"- Exact semantically compatible mapped mentions: **{coverage['mapped_atomic_biomedical_mentions']:,}**",
        f"- Coverage: **{coverage['coverage']:.4%}**",
        f"- Auto-accepted exact mentions: **{coverage['auto_accepted_exact_mentions']:,}**",
        f"- Mentions requiring review: **{coverage['needs_review_mentions']:,}**",
        f"- Unmapped eligible mentions: **{coverage['unmapped_atomic_biomedical_mentions']:,}**",
        "",
        "## Read-only graph impact",
        "",
        f"- Preserved `CLM_CONCEPT` nodes: **{impact['preserved_clm_concept_nodes']:,}**",
        "- Renamed or removed `CLM_CONCEPT` nodes: **0**",
        f"- Proposed eligible `CLM_ATOM` nodes: **{impact['proposed_atomic_mention_nodes']:,}**",
        f"- Proposed `maps_to` edges: **{impact['proposed_maps_to_edges']:,}**",
        f"- Unique UMLS CUI targets: **{impact['unique_cui_targets']:,}**",
        f"- CUI targets already in formal KG: **{impact['existing_cui_targets']:,}**",
        f"- New canonical CUI targets needed by a future apply: **{impact['new_cui_targets_required']:,}**",
        "",
        "No formal KG, extracted-claims store, or `CURRENT_STATE.json` file was modified.",
        "",
        "## Coverage by primary domain",
        "",
        "| domain | eligible | mapped | coverage | auto | review | unmapped |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, row in report["coverage_by_primary_domain"].items():
        value = "n/a" if row["coverage"] is None else f"{row['coverage']:.2%}"
        lines.append(
            f"| {domain} | {row['eligible']:,} | {row['mapped']:,} | {value} | "
            f"{row['auto_accepted']:,} | {row['needs_review']:,} | {row['unmapped']:,} |"
        )
    lines.extend(
        [
            "",
            "## Method boundary",
            "",
            "Matching is limited to Unicode/underscore/dash/whitespace normalized exact terms "
            "and exact source identifiers, followed by MRSTY semantic compatibility. Full "
            "phrases win when they have a compatible exact match; otherwise conservative "
            "top-level composites are split into span-traceable atoms. No fuzzy or embedding "
            "candidate is promoted. Ambiguous, transformed, or generic-domain matches are "
            "marked `needs_review`.",
            "",
            "This is an impact artifact only. Applying it to the formal KG requires separate, "
            "explicit user authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("neurooracle/data/full_v2/knowledge_graph.json"),
    )
    parser.add_argument(
        "--current-state",
        type=Path,
        default=Path("neurooracle/data/full_v2/CURRENT_STATE.json"),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("neurooracle/data/raw/umls/2026AA/UMLS_RELEASE_MANIFEST.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "neurooracle/data/umls_mapping/umls_2026AA_atomic_mentions_v1_20260906"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_path = args.graph.resolve()
    current_state_path = args.current_state.resolve()
    manifest_path = args.release_manifest.resolve()
    output_dir = args.output_dir.resolve()

    protected_dir = graph_path.parent.resolve()
    legacy_raw = (Path("neurooracle/data/raw") / "MRCONSO.RRF").resolve()
    if output_dir == protected_dir or protected_dir in output_dir.parents:
        raise RuntimeError("output directory must not be inside full_v2")
    if output_dir == legacy_raw or legacy_raw in output_dir.parents:
        raise RuntimeError("output directory must not overwrite legacy raw MRCONSO")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "RUN_STATE.json"
    update_state(state_path, "PREFLIGHT")
    release_manifest, rrf_paths = validate_release_manifest(manifest_path)
    rrf_before = {name: path_metadata(path) for name, path in rrf_paths.items()}
    current_state, graph_before = validate_formal_graph(graph_path, current_state_path)
    current_state_before = path_metadata(current_state_path)
    claims_canonical = current_state["canonical_files"]["extracted_claims"]
    claims_path = Path(claims_canonical["path"]).resolve()
    claims_before = validate_cached_file_metadata(
        claims_path,
        claims_canonical,
        "extracted claims",
    )

    source_path = output_dir / "SOURCE_MENTIONS_INTERMEDIATE.jsonl"
    source_summary, target_terms, target_codes, existing_cui_nodes = collect_source_mentions(
        graph_path,
        source_path,
        state_path,
    )

    update_state(
        state_path,
        "SCAN_MRCONSO",
        target_terms=len(target_terms),
        target_external_codes=len(target_codes),
    )
    term_matches, code_matches, overflow_terms, mrconso_summary = scan_mrconso(
        rrf_paths["MRCONSO.RRF"],
        target_terms,
        target_codes,
        state_path,
    )
    if mrconso_summary["rows_scanned"] != int(
        release_manifest["files"]["MRCONSO.RRF"]["rows"]
    ):
        raise RuntimeError("MRCONSO row count differs from the release manifest")
    matched_cuis = set()
    for rows in term_matches.values():
        matched_cuis.update(rows)
    for rows in code_matches.values():
        matched_cuis.update(rows)
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            matched_cuis.update(json.loads(line).get("existing_direct_cuis") or [])

    update_state(state_path, "SCAN_MRSTY", matched_cuis=len(matched_cuis))
    semantics, mrsty_summary = scan_mrsty(
        rrf_paths["MRSTY.RRF"],
        matched_cuis,
        state_path,
    )
    if mrsty_summary["rows_scanned"] != int(
        release_manifest["files"]["MRSTY.RRF"]["rows"]
    ):
        raise RuntimeError("MRSTY row count differs from the release manifest")

    update_state(state_path, "FINALIZE_PROJECTION")
    projection, artifacts = finalize_projection(
        source_path,
        output_dir,
        term_matches,
        code_matches,
        semantics,
        existing_cui_nodes,
        overflow_terms,
        state_path,
    )

    graph_after = path_metadata(graph_path)
    current_state_after = path_metadata(current_state_path)
    claims_after = path_metadata(claims_path)
    rrf_after = {name: path_metadata(path) for name, path in rrf_paths.items()}
    graph_unchanged = metadata_unchanged(graph_before, graph_after)
    if not graph_unchanged:
        raise RuntimeError("formal graph path/size/mtime changed during read-only analysis")
    current_state_unchanged = metadata_unchanged(
        current_state_before,
        current_state_after,
    )
    if not current_state_unchanged:
        raise RuntimeError("CURRENT_STATE.json changed during read-only analysis")
    claims_unchanged = metadata_unchanged(claims_before, claims_after)
    if not claims_unchanged:
        raise RuntimeError("extracted claims changed during read-only analysis")
    rrf_unchanged = all(
        metadata_unchanged(rrf_before[name], rrf_after[name]) for name in rrf_paths
    )
    if not rrf_unchanged:
        raise RuntimeError("one or more staged UMLS inputs changed during analysis")

    totals = projection["totals"]
    eligible = int(totals["eligible"])
    mapped = int(totals["mapped"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "READ_ONLY_IMPACT_COMPLETE",
        "generated_at_utc": utc_now(),
        "umls_release": "2026AA",
        "release_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "archive_official_md5": release_manifest["archive"]["official_md5"],
            "archive_observed_md5": release_manifest["archive"]["observed_md5"],
            "archive_md5_matches": release_manifest["archive"]["md5_matches"],
            "staged_inputs_before": rrf_before,
            "staged_inputs_after": rrf_after,
            "staged_inputs_path_size_mtime_unchanged": rrf_unchanged,
        },
        "formal_source": {
            "before": graph_before,
            "after": graph_after,
            "path_size_mtime_unchanged": graph_unchanged,
            "trusted_sha256_from_current_state": current_state["canonical_files"][
                "knowledge_graph"
            ]["sha256"],
            "full_sha256_recomputed": False,
            "current_state_before": current_state_before,
            "current_state_after": current_state_after,
            "current_state_path_size_mtime_unchanged": current_state_unchanged,
            "extracted_claims_before": claims_before,
            "extracted_claims_after": claims_after,
            "extracted_claims_path_size_mtime_unchanged": claims_unchanged,
            "extracted_claims_trusted_sha256_from_current_state": claims_canonical[
                "sha256"
            ],
            "extracted_claims_full_sha256_recomputed": False,
        },
        "coverage_denominator": {
            "name": "eligible_atomic_biomedical_mentions",
            "definition": (
                "Selected full-or-conservatively-split atomic units derived from preserved "
                "CLM_CONCEPT preferred-name spans, restricted to biomedical KG domain or "
                "atom-type context and entity-like lexical forms. Infrastructure-only, "
                "missing-domain, generic non-entity, URL, numeric-only, overlong, and "
                "fragmentary forms are excluded."
            ),
        },
        "coverage_numerator": {
            "name": "exact_semantically_compatible_mapped_mentions",
            "definition": (
                "Eligible atomic mentions with at least one normalized-exact term or exact "
                "source-identifier UMLS CUI candidate compatible with MRSTY."
            ),
        },
        "coverage": {
            "eligible_atomic_biomedical_mentions": eligible,
            "mapped_atomic_biomedical_mentions": mapped,
            "coverage": mapped / eligible if eligible else 0.0,
            "auto_accepted_exact_mentions": int(totals["auto_accepted"]),
            "needs_review_mentions": int(totals["needs_review"]),
            "unmapped_atomic_biomedical_mentions": int(totals["unmapped"]),
        },
        "coverage_by_primary_domain": add_coverage(projection["by_primary_domain"]),
        "coverage_by_atomization_rule": add_coverage(projection["by_atomization_rule"]),
        "source_scan": source_summary,
        "mrconso_scan": mrconso_summary,
        "mrsty_scan": mrsty_summary,
        "excluded_atomic_mentions": projection["excluded_atomic_mentions"],
        "read_only_impact": {
            "preserved_clm_concept_nodes": int(source_summary["source_mentions"]),
            "renamed_clm_concept_nodes": 0,
            "removed_clm_concept_nodes": 0,
            "proposed_atomic_mention_nodes": eligible,
            "proposed_maps_to_edges": int(totals["maps_to_edges"]),
            "unique_cui_targets": projection["projection"]["unique_cui_targets"],
            "existing_cui_targets": projection["projection"]["existing_cui_targets"],
            "new_cui_targets_required": projection["projection"][
                "new_cui_targets_required"
            ],
            "formal_kg_modified": False,
            "extracted_claims_modified": False,
            "current_state_modified": False,
        },
        "mapping_policy": {
            "original_clm_concept_preserved": True,
            "mapping_relation": "CLM_ATOM -> maps_to -> CUI",
            "normalization": "NFKC + underscore/dash harmonization + casefold + whitespace collapse",
            "full_phrase_precedence": True,
            "top_level_split_only_after_full_exact_miss": True,
            "semantic_compatibility_source": "MRSTY.RRF from UMLS 2026AA",
            "ambiguous_or_transformed_matches_require_review": True,
            "fuzzy_or_embedding_promoted": False,
            "external_ids_policy": (
                "UMLS_CUI is emitted only on canonical UMLS target records, never on "
                "CLM_ATOM mention records."
            ),
        },
        "artifacts": artifacts,
        "limitations": [
            "Normalized-exact matching favors precision and leaves abbreviation, spelling, and paraphrase misses unmapped.",
            "Top-level syntax rules do not infer shared heads or resolve coordinated ellipsis.",
            "Mappings marked needs_review are candidates, not authorized formal KG facts.",
        ],
    }

    report_json = output_dir / "UMLS_2026AA_IMPACT_REPORT.json"
    atomic_write_json(report_json, report)
    report_md = output_dir / "UMLS_2026AA_IMPACT_REPORT.md"
    report_md.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    report_artifacts = {
        "json_report": {
            **path_metadata(report_json),
            "sha256": sha256_file(report_json),
        },
        "markdown_report": {
            **path_metadata(report_md),
            "sha256": sha256_file(report_md),
        },
    }
    freeze = {
        "schema_version": "neurooracle.umls_atomic_mention_impact_freeze.v1",
        "status": "READ_ONLY_IMPACT_COMPLETE_AND_FROZEN",
        "created_at_utc": utc_now(),
        "umls_release": "2026AA",
        "release_manifest_sha256": report["release_manifest"]["sha256"],
        "formal_graph_path_size_mtime_unchanged": graph_unchanged,
        "extracted_claims_path_size_mtime_unchanged": claims_unchanged,
        "staged_inputs_path_size_mtime_unchanged": rrf_unchanged,
        "formal_graph_trusted_sha256": report["formal_source"][
            "trusted_sha256_from_current_state"
        ],
        "formal_graph_sha256_recomputed": False,
        "artifacts": {**artifacts, **report_artifacts},
        "formal_apply_authorized": False,
    }
    freeze_path = output_dir / "UMLS_2026AA_IMPACT_FREEZE.json"
    atomic_write_json(freeze_path, freeze)
    final_state = {
        "schema_version": "neurooracle.umls_atomic_mention_run_state.v1",
        "status": "COMPLETE",
        "phase": "READ_ONLY_IMPACT_FROZEN",
        "updated_at_utc": utc_now(),
        "report": str(report_json),
        "freeze": str(freeze_path),
        "coverage": report["coverage"],
        "formal_graph_unchanged": graph_unchanged,
    }
    atomic_write_json(state_path, final_state)
    print(json.dumps(final_state, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
