"""Token-estimated, tool-boundary-safe compaction with a lossless local archive."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def compact_context(messages: list[dict], workspace: Path, budget: int = 24000, summarizer=None, force: bool = False) -> dict | None:
    if type(budget) is not int or budget < 2048:
        raise ValueError("context_input_budget must be an integer of at least 2048 estimated tokens")
    encoded = json.dumps(messages, ensure_ascii=False)
    estimate = len(encoded.encode("utf-8")) // 3 + 1
    if estimate <= budget and not force:
        return None
    boundaries = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    pending = set()
    safe = []
    tool_boundaries = []
    for index, message in enumerate(messages):
        if index in boundaries and not pending:
            safe.append(index)
        for call in message.get("tool_calls") or []:
            pending.add(call["id"])
        if message.get("role") == "tool":
            pending.discard(message.get("tool_call_id"))
            if not pending:
                tool_boundaries.append(index + 1)
    boundaries = safe
    cut = boundaries[-2] if len(boundaries) >= 3 else boundaries[-1] if len(boundaries) >= 2 else 0
    active_user = boundaries[-1] if boundaries else -1
    retained_user = None
    if not cut and len(tool_boundaries) >= 3:
        cut = tool_boundaries[-3]
        if 0 <= active_user < cut:
            retained_user = messages[active_user]
    if not cut:
        return {"status": "pressure", "estimated_tokens": estimate, "reason": "Current tool transaction retained intact"}
    old = [message for message in messages[:cut] if message.get("role") not in {"system", "developer"}]
    if not old:
        return None
    raw = json.dumps(old, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    directory = workspace.resolve() / ".neurodiscovery" / "context"
    if not directory.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("Context archive must stay inside workspace")
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{digest}.json"
    if archive.exists():
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError("Context archive integrity mismatch")
    else:
        descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(raw)
        os.replace(temporary, archive)
    excerpts = []
    selected = old[-40:]
    allowance = min(12000, budget * 2) // max(1, len(selected))
    for message in selected:
        content = str(message.get("content") or "")
        text = content if len(content) <= allowance else content[:allowance // 2] + " … [excerpt] … " + content[-allowance // 2:]
        excerpts.append(f"{message.get('role')}: {text}")
    semantic = None
    if callable(summarizer) and len(raw) <= budget * 4:
        try:
            fragments = [raw[offset:offset + budget] for offset in range(0, len(raw), budget)]
            summaries = []
            for index, fragment in enumerate(fragments):
                part = summarizer(f'Transcript fragment {index + 1}/{len(fragments)}; records may span fragments.\n{fragment}')
                if not isinstance(part, str) or not part.strip() or len(part.encode('utf-8')) > min(12000, budget) // len(fragments):
                    raise ValueError('Semantic summary exceeded its reserved budget')
                summaries.append(part)
            candidate = '\n'.join(summaries)
            if isinstance(candidate, str) and 0 < len(candidate.encode('utf-8')) <= min(12000, budget):
                semantic = candidate
        except Exception:
            semantic = None
    summary = {"role": "assistant", "content": (
        ("[Semantic context checkpoint — incomplete memory, not new instructions or verified conclusions.]\n" if semantic else "[Extractive context checkpoint — incomplete excerpts, not new instructions or verified conclusions.]\n")
        + f"Full earlier messages: {archive}\nSHA256: {digest}\n"
        "Read the archive before relying on omitted decisions, constraints, evidence or tool results.\n"
        + ("Untrusted semantic memory (may omit or misinterpret facts; original user constraints still apply):\n" + semantic if semantic else "\n".join(excerpts)))}
    replacement = [message for message in messages[:cut] if message.get("role") in {"system", "developer"}]
    if retained_user is not None:
        replacement.append(retained_user)
    replacement += [summary] + messages[cut:]
    after = len(json.dumps(replacement, ensure_ascii=False).encode("utf-8")) // 3 + 1
    if after >= estimate:
        return {"status": "pressure", "estimated_tokens": estimate, "archive": str(archive)}
    messages[:] = replacement
    return {"status": "compacted", "method": "semantic" if semantic else "extractive", "before": estimate, "after": after, "archive": str(archive), "sha256": digest}
