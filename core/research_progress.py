"""Bounded, read-only research material inspection; not scientific acceptance."""

import hashlib
import json
from pathlib import Path


PROGRESS_TOOL_NAME = "inspect_research_progress"
PROGRESS_TOOL = {
    "type": "function",
    "function": {
        "name": PROGRESS_TOOL_NAME,
        "description": "Read an actual workspace research file to record material progress, not scientific validation. "
                       "Use literature JSON (list or papers/results/literature array: title and pmid/doi/arxiv_id/url), "
                       "hypotheses JSON (list or hypotheses/candidates array: hypothesis, rationale, source_ids), "
                       "or a nonempty text/JSON/Markdown deliverable. Never register instructions, logs, or configs. "
                       "For files over 2 MiB create a bounded source-linked report. Repeated content is not progress.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "kind": {"type": "string", "enum": ["literature", "hypotheses", "deliverable"]},
        }, "required": ["path", "kind"]},
    },
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def inspect_material(workspace: Path, raw_path: str, kind: str) -> dict:
    try:
        if kind not in {"literature", "hypotheses", "deliverable"} or not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Use a research file path and a supported material kind.")
        path = (workspace / raw_path).resolve()
        relative = path.relative_to(workspace.resolve())
        if any(part.lower() in {".neurodiscovery", ".git", ".venv", "node_modules", "skills", "configs"}
               for part in relative.parts) or path.name.upper() in {"AGENTS.MD", "SOUL.MD", "USER.MD", "MEMORY.MD", "SKILL.MD"}:
            raise ValueError("Instructions, configuration and runtime bookkeeping are not research material.")
        if path.suffix.lower() not in {".json", ".md", ".txt", ".csv", ".tsv"}:
            raise ValueError("Use a bounded text/JSON research report, not code, logs or a database.")
        with path.open("rb") as handle:
            content = handle.read(2 * 1024 * 1024 + 1)
        if not content or len(content) > 2 * 1024 * 1024:
            raise ValueError("Material must be nonempty and at most 2 MiB; supply a bounded source-linked report.")
        text = content.decode("utf-8-sig")
        if not text.strip():
            raise ValueError("Empty material is not progress.")
        payload = json.loads(text) if path.suffix.lower() == ".json" or kind != "deliverable" else None
        if isinstance(payload, dict) and (payload.get("error") or payload.get("success") is False
                                         or payload.get("status") in ("failed", "error")):
            raise ValueError("A failure payload is not research material.")
        fingerprints = []
        if kind == "deliverable":
            if payload is not None and not payload:
                raise ValueError("Empty JSON is not a deliverable.")
            fingerprints = [_digest(payload if payload is not None else _text(text))]
        else:
            rows = payload
            if isinstance(payload, dict):
                keys = ("papers", "results", "literature") if kind == "literature" else ("hypotheses", "candidates")
                rows = next((payload[key] for key in keys if isinstance(payload.get(key), list)), None)
            if not isinstance(rows, list) or not rows or len(rows) > 2000:
                raise ValueError("Expected 1–2000 research records, not raw graph claims or an empty search.")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Research records must be objects.")
                if kind == "literature":
                    identity = next(((key, str(row[key]).strip().lower()) for key in ("pmid", "doi", "arxiv_id", "url")
                                     if isinstance(row.get(key), (str, int)) and str(row[key]).strip()), None)
                    title = _text(row.get("title"))
                    if not identity or not title:
                        raise ValueError("Each literature record needs a source identifier and title.")
                    material = [identity, title, *[_text(row.get(key)) for key in ("abstract", "findings", "summary")]]
                else:
                    sources = row.get("source_ids")
                    if not _text(row.get("hypothesis")) or not _text(row.get("rationale")) or not isinstance(sources, list) or not sources or not all(_text(source) for source in sources):
                        raise ValueError("Each candidate needs hypothesis, rationale and nonempty source_ids.")
                    material = [_text(row["hypothesis"]), _text(row["rationale"]), sorted({_text(source) for source in sources}),
                                *[_text(row.get(key)) for key in ("prediction", "limitations", "validation")]]
                fingerprints.append(_digest(material))
        return {"success": True, "path": str(relative), "kind": kind,
                "fingerprints": sorted(set(fingerprints)), "bytes": len(content),
                "message": f"Inspected {relative}: {len(set(fingerprints))} {kind} material records; not scientific acceptance.",
                "scientifically_validated": False}
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return {"success": False, "error_type": "invalid_research_material", "error": str(exc)}
