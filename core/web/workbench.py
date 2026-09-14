"""Local workbench state, resumable events and request-level usage (no credentials)."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid

from core.llm.adapters import IncompleteModelResponse, ProviderClient, objects, plain


def default_database() -> Path:
    override = os.environ.get("NEURODISCOVERY_WORKBENCH_DB")
    if override:
        return Path(override).expanduser()
    # This is also the desktop launcher's explicitly confirmed application-reset
    # boundary. Keep transcripts/usage outside both browser storage and projects.
    return Path.home() / ".neurodiscovery" / "workbench.sqlite3"


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def normalize_usage(raw: Any, protocol: str = "chat_completions") -> dict:
    """Input includes cache; details are subsets, not additional billable totals."""
    data = plain(raw) if raw is not None else {}
    data = data if isinstance(data, dict) else {}
    # ProviderClient marks synthesized compatibility totals explicitly.
    if data.get("_nd_reported") is False:
        data = {}
    input_tokens = _count(data.get("prompt_tokens", data.get("input_tokens")))
    output_tokens = _count(data.get("completion_tokens", data.get("output_tokens")))
    if data.get("_nd_input_reported") is False:
        input_tokens = None
    if data.get("_nd_output_reported") is False:
        output_tokens = None
    input_details = data.get("prompt_tokens_details") or data.get("input_tokens_details") or {}
    output_details = data.get("completion_tokens_details") or data.get("output_tokens_details") or {}
    cache_read = _count(data.get("cache_read_input_tokens", input_details.get("cached_tokens")))
    cache_write = _count(data.get("cache_creation_input_tokens"))
    if protocol == "anthropic" and "prompt_tokens" not in data and input_tokens is not None:
        input_tokens += (cache_read or 0) + (cache_write or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
            "reasoning_tokens": _count(output_details.get("reasoning_tokens")),
            "reported": input_tokens is not None and output_tokens is not None}


class StateConflict(ValueError):
    pass


class WorkbenchStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_database()
        self._lock = threading.RLock()

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, revision INTEGER, body TEXT);
            CREATE TABLE IF NOT EXISTS recovery (id TEXT PRIMARY KEY, created REAL, body TEXT);
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY, request_id TEXT, chat_id TEXT, project_id TEXT,
                provider TEXT, model TEXT, source TEXT, started REAL, ended REAL,
                status TEXT, input_tokens INTEGER, output_tokens INTEGER,
                cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER,
                reported INTEGER DEFAULT 0);
            CREATE INDEX IF NOT EXISTS calls_started ON calls(started);
            CREATE TABLE IF NOT EXISTS runs (request_id TEXT PRIMARY KEY, chat_id TEXT, body TEXT);
        """)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def state(self) -> dict:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT revision, body FROM state WHERE id=1").fetchone()
        return {"revision": row["revision"], "state": json.loads(row["body"])} if row else {"revision": 0, "state": None}

    def save_state(self, revision: int, state: dict) -> dict:
        if type(revision) is not int or not isinstance(state, dict):
            raise ValueError("Invalid state revision")
        if not isinstance(state.get("sessions"), list) or not isinstance(state.get("projects"), list):
            raise ValueError("Expected sessions and projects")
        body = json.dumps(state, ensure_ascii=False)
        if len(body.encode("utf-8")) > 32 * 1024 * 1024:
            raise ValueError("Workbench history exceeds 32 MB; export large conversations first")
        conflict_id = None
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT revision FROM state WHERE id=1").fetchone()
            current = row[0] if row else 0
            if revision != current:
                conflict_id = uuid.uuid4().hex
                conn.execute("INSERT INTO recovery VALUES (?,?,?)", (conflict_id, time.time(), body))
            else:
                conn.execute("INSERT OR REPLACE INTO state VALUES (1, ?, ?)", (current + 1, body))
        if conflict_id:
            error = StateConflict("History changed in another window. An independent recovery copy was saved; export it before reloading.")
            error.recovery_id = conflict_id
            raise error
        return {"revision": current + 1}

    def recovery(self, recovery_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT body FROM recovery WHERE id=?", (recovery_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def start_call(self, call_id: str, meta: dict) -> None:
        fields = ("request_id", "chat_id", "project_id", "provider", "model", "source")
        with self._lock, self._connect() as conn:
            conn.execute("INSERT INTO calls(call_id,request_id,chat_id,project_id,provider,model,source,started,status) VALUES(?,?,?,?,?,?,?,?,?)",
                         (call_id, *(str(meta.get(key) or "")[:256] for key in fields), time.time(), "running"))

    def finish_call(self, call_id: str, status: str, usage: dict) -> None:
        fields = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "reported")
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE calls SET ended=?,status=?,input_tokens=?,output_tokens=?,cache_read_tokens=?,cache_write_tokens=?,reasoning_tokens=?,reported=? WHERE call_id=?",
                         (time.time(), status, *(usage.get(key) for key in fields), call_id))

    def usage(self, **filters) -> dict:
        conditions, args = [], []
        for key in ("request_id", "chat_id", "project_id", "provider", "model"):
            if filters.get(key):
                conditions.append(f"{key}=?")
                args.append(str(filters[key]))
        if filters.get("days"):
            conditions.append("started>=?")
            args.append(time.time() - min(max(int(filters["days"]), 1), 36500) * 86400)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._lock, self._connect() as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM calls" + where + " ORDER BY started DESC", args)]
        keys = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
        totals = {key: sum(row[key] for row in rows if row[key] is not None) for key in keys}
        totals.update(requests=len(rows), unreported=sum(not row["reported"] for row in rows))
        for status in ("completed", "failed", "cancelled", "running", "interrupted"):
            totals[status] = sum(row["status"] == status for row in rows)
        totals["known_fields"] = {key: sum(row[key] is not None for row in rows) for key in keys}
        return {"totals": totals, "calls": rows, "scope": "local_client", "account_balance": None,
                "cost": None, "cost_status": "not_reported", "updated_at": time.time()}

    def save_run(self, run: "WorkbenchRun") -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?)", (run.id, run.chat_id, json.dumps(run.snapshot(), ensure_ascii=False)))

    def saved_run(self, request_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT body FROM runs WHERE request_id=?", (request_id,)).fetchone()
        if not row:
            return None
        result = json.loads(row[0])
        if result["status"] in {"running", "stopping"}:
            result["status"] = "interrupted"
            with self._lock, self._connect() as conn:
                conn.execute("UPDATE calls SET status='interrupted' WHERE request_id=? AND status='running'", (request_id,))
            for item in result.get("blocks", []) + result.get("tools", []):
                if item.get("status") == "running":
                    item["status"] = "interrupted"
            result["result"] = {"type": "error", "message": "Runtime restarted; this request was interrupted. No request was replayed.",
                                "usage": self.usage(request_id=request_id)}
            with self._lock, self._connect() as conn:
                conn.execute("UPDATE runs SET body=? WHERE request_id=?", (json.dumps(result, ensure_ascii=False), request_id))
        return result


class WorkbenchRun:
    def __init__(self, request_id: str, chat_id: str, store: WorkbenchStore):
        self.id, self.chat_id, self.store = request_id, chat_id, store
        self.events = deque(maxlen=1024)
        self.seq = 0
        self.status = "running"
        self.result = None
        self.blocks: dict[str, dict] = {}
        self.tools: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._last_save = time.monotonic()

    def emit(self, event: dict) -> None:
        with self._lock:
            self.seq += 1
            event = {**event, "seq": self.seq, "request_id": self.id, "chat_id": self.chat_id}
            call_id = str(event.get("call_id") or "")
            if event["type"] in {"model_start", "text", "reasoning", "model_end", "usage"}:
                block = self.blocks.setdefault(call_id, {"call_id": call_id, "text": "", "reasoning": "", "status": "running"})
                if event["type"] in {"text", "reasoning"}:
                    key = "text" if event["type"] == "text" else "reasoning"
                    combined = block[key] + str(event.get("text") or "")
                    block[key] = combined[:1_000_000]
                    block["truncated"] = len(combined) > 1_000_000 or block.get("truncated", False)
                else:
                    block.update({key: event[key] for key in ("status", "model", "source") if key in event})
                    if event["type"] == "usage":
                        block["usage"] = event.get("usage")
            if event["type"] in {"tool_start", "tool_end"}:
                tool_id = str(event.get("tool_id") or "")
                self.tools.setdefault(tool_id, {}).update(event)
            self.events.append(event)
            if time.monotonic() - self._last_save >= 1:
                self._last_save = time.monotonic()
                self.store.save_run(self)

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy({"request_id": self.id, "chat_id": self.chat_id, "seq": self.seq,
                             "status": self.status, "blocks": list(self.blocks.values()),
                             "tools": list(self.tools.values()), "result": self.result})

    def poll(self, after: int) -> dict:
        with self._lock:
            reset = after == 0 or (bool(self.events) and after < self.events[0]["seq"] - 1)
            return {"snapshot": self.snapshot() if reset else None,
                    "events": [] if reset else [deepcopy(e) for e in self.events if e["seq"] > after],
                    "seq": self.seq, "status": self.status, "result": self.result}

    def finish(self, result: dict, status: str) -> None:
        with self._lock:
            self.result, self.status = result, status
            self.emit({"type": "finish", "status": status})
            self.store.save_run(self)


class ObservedClient:
    """Opt-in web facade: preserve the serial executor and its native tool protocol.

    One ledger row per actual create attempt. Reconnection only reads events; it
    never replays a model request. Auxiliary/delegated calls retain their identity.
    """
    def __init__(self, client: Any, store: WorkbenchStore, meta: dict, emit, cancel_event=None):
        from types import SimpleNamespace
        if isinstance(client, ProviderClient) and callable(getattr(client.raw, "with_options", None)):
            # SDK-internal retries would be invisible to this ledger. The executor's
            # existing retry policy remains authoritative and each attempt is counted.
            client = ProviderClient(client.raw.with_options(max_retries=0), client.cfg)
        self.client, self.store, self.meta, self.emit = client, store, meta, emit
        self.cancel_event = cancel_event
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __getattr__(self, name):
        return getattr(self.client, name)

    def create(self, *, model: str, messages: list, **kwargs):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("Request cancelled before dispatch")
        call_id = uuid.uuid4().hex
        meta = {**self.meta, "model": model}
        if model != self.meta.get("model"):
            meta["source"] = "auxiliary"
        self.store.start_call(call_id, meta)
        self.emit({"type": "model_start", "call_id": call_id, "model": model, "source": meta.get("source", "main")})
        usage, status, response = normalize_usage(None), "failed", None
        stream, seen = None, False
        buffers = {"text": "", "reasoning": ""}
        last_flush = time.monotonic()

        def flush():
            nonlocal last_flush
            for kind in buffers:
                if buffers[kind]:
                    self.emit({"type": kind, "call_id": call_id, "text": buffers[kind]})
                    buffers[kind] = ""
            last_flush = time.monotonic()

        try:
            # Legacy json_object requires adapter-side non-stream validation.
            can_stream = isinstance(self.client, ProviderClient) and not kwargs.get("response_format")
            if can_stream:
                options = {**kwargs, "stream": True}
                stream = self.client.chat.completions.create(model=model, messages=messages, **options)
                for chunk in stream:
                    seen = True
                    data = plain(chunk)
                    if data.get("usage"):
                        usage = normalize_usage(data["usage"])
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise IncompleteModelResponse("Request cancelled; partial output preserved")
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta") or {}
                        for key, kind in (("content", "text"), ("reasoning_content", "reasoning")):
                            if isinstance(delta.get(key), str):
                                buffers[kind] += delta[key]
                    if time.monotonic() - last_flush >= .05:
                        flush()
                response = self.client.last_response
                if response is None:
                    raise IncompleteModelResponse("No completed model response")
            else:
                response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
                message = plain(response).get("choices", [{}])[0].get("message", {})
                for key, kind in (("content", "text"), ("reasoning_content", "reasoning")):
                    if isinstance(message.get(key), str):
                        buffers[kind] += message[key]
            final_usage = normalize_usage(getattr(response, "usage", None))
            if final_usage["reported"] or not usage["reported"]:
                usage = final_usage
            status = "completed"
            return response
        except Exception as exc:
            if isinstance(exc, IncompleteModelResponse):
                partial = normalize_usage(getattr(exc, "usage", None))
                if partial["reported"]:
                    usage = partial
            if self.cancel_event is not None and self.cancel_event.is_set():
                status = "cancelled"
            # Never retry a transport failure after any upstream stream event.
            if seen and not isinstance(exc, IncompleteModelResponse):
                raise IncompleteModelResponse("Stream interrupted; no automatic replay") from exc
            raise
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                flush()
                self.store.finish_call(call_id, status, usage)
                self.emit({"type": "model_end", "call_id": call_id, "status": status})
                self.emit({"type": "usage", "call_id": call_id, "usage": usage, "status": status})


def observe_local_call(request, store, meta, emit, cancel_event=None):
    """Observe the legacy non-streaming Ollama path without changing its request."""
    from types import SimpleNamespace
    captured = {}

    def create(**_kwargs):
        data = request()
        captured["data"] = data
        message = data.get("message") or {}
        return objects({"choices": [{"message": {"content": message.get("content"),
                         "reasoning_content": message.get("thinking")}}],
                        "usage": {"prompt_tokens": data.get("prompt_eval_count"),
                                  "completion_tokens": data.get("eval_count")}})

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ObservedClient(client, store, meta, emit, cancel_event).create(model=meta["model"], messages=[])
    return captured["data"]
